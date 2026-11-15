"""Conditional, state-initialised forecasts; separate from v25 steady-state APIs."""

import calendar
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.agent import outlook
from fieldshift.agent.runtime import InputError, clean
from fieldshift.sim import state
from fieldshift.sim.calendar import VARS
from fieldshift.sim.engine import ledger_pass, prepare, water_pass

FIELDS = {"relative_yield": "yield_relative", "operating_return_usd_ha": "income",
          "applied_irrigation_mm": "irrigation_mm", "nitrate_leached_kg_n_ha": "n_leached_kg_ha",
          "soil_water_at_planting_mm": "water_at_planting_mm", "soil_c_input_t_c_ha": "soil_carbon_input_t_ha"}

class ForecastUnavailable(RuntimeError):
    """Provider failure, distinct from an invalid farm request."""


def validate(payload):
    if not isinstance(payload, dict) or payload.keys() - {"rotations", "years", "weather"}:
        raise InputError("forecast accepts rotations, years (1–4), weather (outlook or climatology)")
    years = payload.get("years", 1)
    weather = payload.get("weather", "outlook")
    if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 4:
        raise InputError("years must be an integer 1–4")
    rotations = payload.get("rotations")
    if not isinstance(rotations, list) or not 1 <= len(rotations) <= 12:
        raise InputError("rotations must contain 1–12 cycles")
    if weather not in ("outlook", "climatology"):
        raise InputError("weather must be outlook or climatology")
    return rotations, years, weather


def calendar_weather(farm, dates, donors):
    """Historical blocks, preserving month/day; absent Feb29 repeats Feb28.

    All donors have equal weight. Successive target years use successive donor
    years with wraparound, rather than independent random weather days.
    """
    device = farm.base["device"]
    daily = farm.weather().copy()
    if "rsm_anomaly" not in daily:
        daily["rsm_anomaly"] = daily.rsm - daily.groupby(daily.date.dt.strftime("%m-%d")).rsm.transform("mean")
    years = farm.base["years"]
    cube = torch.zeros((len(years), 366, len(VARS)), device=device)
    for i, year in enumerate(years):
        row = daily.loc[daily.date.dt.year.eq(year), list(VARS)].to_numpy("float32")
        cube[i, :len(row)] = torch.tensor(row, device=device)
    # These are calendar indices, not host-side simulation.
    offset = torch.tensor(dates.year.to_numpy()-dates.year.min(), device=device)
    y = (donors[:, None]+offset[None]) % len(years)
    leap = torch.tensor([calendar.isleap(year) for year in years], device=device)
    nonleap_day = np.array([pd.Timestamp(2001, d.month, min(d.day, 28) if d.month == 2 else d.day).dayofyear-1
                           for d in dates])
    # Nonleap Feb28/29 share day58; leap-source Feb29 is explicitly selected.
    base = torch.tensor(nonleap_day, device=device)[None].expand_as(y)
    after_feb = torch.tensor((dates.month > 2).astype(np.int64), device=device)[None]
    feb29 = torch.tensor(((dates.month == 2) & (dates.day == 29)).astype(np.int64), device=device)[None]
    day = base + leap[y] * (after_feb+feb29)
    return cube[y, day]


def outlook_weather(farm, dates, baseline, donors, cached, now):
    """Same monthly mean-bias transfer as outlook.py, applied to daily tensors."""
    source_dates, arrays = outlook.member_arrays(cached["response"])
    device = baseline.device
    hist = outlook.record(farm)
    h = torch.tensor(np.ascontiguousarray(hist[["prec", "tavg"]].to_numpy("float32")), device=device)
    bucket = torch.tensor((hist.date.dt.year.to_numpy()-1991)*12+hist.date.dt.month.to_numpy()-1, device=device)
    totals = torch.zeros((360, 2), device=device).index_add_(0, bucket, h)
    count = torch.zeros(360, device=device).index_add_(0, bucket, torch.ones(len(hist), device=device))
    if bool((count == 0).any()):
        raise ForecastUnavailable("Need all 30 county baseline years, 1991–2020")
    normals = totals.reshape(30, 12, 2)
    rain_normal = normals[..., 0].mean(0)
    temp_normal = (normals[..., 1]/count.reshape(30, 12)).mean(0)
    raw = {k: torch.tensor(np.ascontiguousarray(v, dtype=np.float32), device=device) for k, v in arrays.items()}
    values = baseline.repeat(51, 1, 1)
    members = torch.arange(51, device=device).repeat_interleave(len(donors))
    factor = 1+farm.record["overrides"].get("rain_change_pct", 0)/100
    delta = farm.record["overrides"].get("temp_change_c", 0)
    available = []
    monthly = cached["response"]["monthly"]
    for mi, stamp in enumerate(pd.DatetimeIndex(monthly["time"])):
        if stamp.to_period("M") < pd.Timestamp(now.date()).to_period("M"):
            continue
        month = stamp.strftime("%Y-%m")
        full_month = source_dates.strftime("%Y-%m") == month
        full_i = np.flatnonzero(full_month)
        if len(full_i) != calendar.monthrange(stamp.year, stamp.month)[1]:
            continue
        index = torch.tensor(full_i, device=device)
        if not all(bool(torch.isfinite(v[:, index]).all()) for v in raw.values()):
            raise ForecastUnavailable("A SEAS5 member has missing days; no silent member/day dropping")
        model_rain = float(monthly["precipitation_mean"][mi])-float(monthly["precipitation_anomaly"][mi])
        model_temp = float(monthly["temperature_2m_mean"][mi])-float(monthly["temperature_2m_anomaly"][mi])
        if not np.isfinite([model_rain, model_temp]).all() or model_rain <= 0:
            raise ForecastUnavailable("Provider monthly model-climatology anomaly unavailable")
        scale = rain_normal[stamp.month-1]/model_rain
        shift = temp_normal[stamp.month-1]-model_temp
        mask = (source_dates.strftime("%Y-%m") == month) & (source_dates.date >= now.date())
        source_i = np.flatnonzero(mask)
        target_i = dates.get_indexer(source_dates[source_i])
        valid = target_i >= 0
        source_i, target_i = source_i[valid], target_i[valid]
        if not len(source_i):
            continue
        ix = torch.tensor(target_i, device=device)
        for key, variable in (("precipitation_sum", "prec"), ("temperature_2m_mean", "tavg"),
                               ("temperature_2m_min", "tmin"), ("temperature_2m_max", "tmax"),
                               ("et0_fao_evapotranspiration_sum", "et0")):
            source_index = torch.tensor(source_i, device=device)
            a = raw[key][:, source_index][members].clone()
            if variable == "prec":
                a *= scale*factor
            elif variable.startswith("t"):
                a += shift+delta
            elif delta:
                temp = raw["temperature_2m_mean"][:, source_index][members]+shift
                a *= ((temp+delta+17.8)/(temp+17.8).clamp_min(1)).clamp_min(0)
            if not bool(torch.isfinite(a).all()):
                raise ValueError("Daily SEAS5 member is missing; no silent member/day imputation")
            values[:, ix, VARS.index(variable)] = a
        available.extend(source_dates[source_i].strftime("%Y-%m-%d").tolist())
    if not available:
        raise ValueError("No current finite SEAS5 daily values in requested forecast window")
    return values, {"issue_date": cached["issue_date"], "first_day": min(available), "last_day": max(available),
                    "daily_members": 51, "bias_method": outlook.CAVEAT,
                    "licence": "Open-Meteo / ECMWF SEAS5, CC BY 4.0; free API non-commercial only"}


def trajectory(farm, cycles, dates, first_season, source_count, replicas, device):
    """Joint weather/state samples × rotation, no cycle-phase averaging."""
    years = sorted(set(dates.year))
    horizon = len(years)-1  # final calendar-only padding year
    n = len(cycles)*source_count*replicas
    rotation = np.repeat(np.arange(len(cycles)), source_count*replicas)
    site = np.tile(np.repeat(np.arange(source_count), replicas), len(cycles))
    replica = torch.arange(replicas, device=device).repeat(len(cycles)*source_count)
    crop = np.zeros((n, horizon+2), dtype=np.int64)
    cover = np.zeros_like(crop)
    double = np.zeros_like(crop, dtype=bool)
    position = np.zeros_like(crop)
    uniform = np.zeros(n, dtype=bool)
    for i, cycle in enumerate(cycles):
        mask = rotation == i
        uniform[mask] = len({r["crop"] for r in cycle}) == 1
        for t, year in enumerate(range(years[0]-1, years[-1]+1)):
            if year < first_season:
                step = {"crop": farm.site["existing"][(year-2019) % len(farm.site["existing"])]}
            else:
                step = cycle[(year-first_season) % len(cycle)]
            crop[mask, t] = farm.keys.index(step["crop"])
            if "winter" in step:
                cover[mask, t] = farm.cover_keys.index(step["winter"])+1
            double[mask, t] = bool(step.get("double_crop"))
            position[mask, t] = (year-first_season) % len(cycle)
    traj = {"rotation": rotation, "site": site, "year0": np.zeros(n, dtype=np.int64),
            "crop": crop, "cover": cover, "double": double, "position": position,
            "uniform": uniform, "horizon": horizon, "spin": 0}
    return traj, replica


def _fixed_calendar(site):
    """Known historical median dates, not unobserved future state reports."""
    planting = {}
    for crop, record in site.get("planting", {}).items():
        planting[crop] = {k: v for k, v in record.items() if k != "by_year"}
        planting[crop]["by_year"] = {}
        for key in ("maturity", "harvested"):
            if key in planting[crop]:
                planting[crop][key] = {k: v for k, v in planting[crop][key].items() if k != "by_year"}
                planting[crop][key]["by_year"] = {}
    return planting


@torch.inference_mode()
def run(farm, cycles, years, mode, nowcast, initial, now=None, cached_outlook=None):
    started = time.perf_counter()
    now = now or datetime.now(UTC)
    device = farm.base["device"]
    last_record = int(nowcast["as_of"][:4])
    first_season = max(last_record+1, now.year+int(now.month >= 8))
    # The unobserved gap is explicitly climate-bridged, never called observed.
    dates = pd.date_range(f"{last_record+1}-01-01", f"{first_season+years}-12-31")
    donors = torch.arange(len(farm.base["years"]), device=device)
    baseline = calendar_weather(farm, dates, donors)
    metadata = {"daily_members": 0}
    if mode == "outlook":
        if cached_outlook is None:
            static = state._static().loc[farm.site["adm_id"]]
            cached_outlook = outlook.fetch(farm.site["adm_id"], static.centroid_lat, static.centroid_lon,
                                          root=state.ROOT/"outlook", now=now)
        weather, metadata = outlook_weather(farm, dates, baseline, donors, cached_outlook, now)
        base = baseline.repeat(51, 1, 1)
    else:
        weather, base = baseline, baseline
    source_count = len(weather)
    # Metadata frame shared by all weather sites; numerical weather stays on GPU.
    template = pd.DataFrame({"date": dates})
    site = dict(farm.site, daily=template, planting=_fixed_calendar(farm.site))
    historical_scales = torch.tensor(farm.base["timeline"]["gdd_scale"], device=device)
    known_scales = {crop.key: historical_scales[0, ci, -10:].median() for ci, crop in enumerate(farm.crops)}
    prepared = prepare([site]*source_count, farm.crops, farm.covers, farm.double, irrigated=farm.irrigated,
                       learned=False, daily_values=weather, base_daily_values=base, calendar_scales=known_scales)
    particles = len(initial["initial"]["water"])
    traj, replica = trajectory(farm, cycles, dates, first_season, source_count, 1 if mode == "outlook" else 51, device)
    weather_index = torch.as_tensor(traj["site"], device=device)
    member = weather_index//len(donors) if mode == "outlook" else replica
    donor = weather_index % len(donors)
    particle = (member+7*donor) % particles
    lookup = {c.key: i for i, c in enumerate(initial["crops"])}
    inverse = torch.tensor([farm.keys.index(c.key) if c.key in farm.keys else -1 for c in initial["crops"]],
                           device=device, dtype=torch.long)
    state_initial = {k: v[particle] for k, v in initial["initial"].items()}
    state_initial["last"] = inverse[state_initial["last"].clamp_min(0)]
    context = {"initial": state_initial, "soil_index": particle,
               "kc_scale": initial["parameters"]["kc_scale"][particle],
               "root_scale": initial["parameters"]["root_scale"][particle], "economics": farm.economics_context()}
    water = water_pass(prepared, initial["soils"], farm.crops, farm.covers, farm.double, traj,
                       farm.irrigated, continuation=context)
    water["temperature_month"] *= initial["parameters"]["mineralisation_rate_20c"][particle, None, None]/.02
    reference = torch.ones((source_count, len(farm.crops)+1), device=device)
    for i, crop in enumerate(farm.crops):
        reference[:, i] = initial["reference"][0, lookup[crop.key]]
    normals = np.tile(farm.normals, (source_count, 1))
    ledger = ledger_pass(prepared, water, initial["soils"], normals, farm.crops, farm.covers, farm.double, traj,
                         farm.irrigated, fao_reference=reference, continuation=context)
    result = []
    site_index = torch.as_tensor(traj["site"], device=device)
    soil_cap = torch.tensor([s["awc_mm_per_m"]*s["profile_depth_m"] for s in initial["soils"]], device=device)[particle]
    source_years = sorted(set(dates.year))
    for ri, cycle in enumerate(cycles):
        mask = torch.as_tensor(traj["rotation"] == ri, device=device)
        seasons = []
        for year in range(first_season, first_season+years):
            j = source_years.index(year)
            crop_i = farm.keys.index(cycle[(year-first_season) % len(cycle)]["crop"])
            crop = farm.crops[crop_i]
            if crop.v("autumn_sown") and j:
                planting_water = water["water_at_next_sowing_mm"][:, j-1]
            else:
                planting_water = water["water_at_planting_mm"][:, j]
            quantities = {}
            for name, column in FIELDS.items():
                values = planting_water if name == "soil_water_at_planting_mm" else ledger[column][:, j]
                quantities[name] = quantiles(values[mask])
            # Calendar/weather risk calculations in chunks, no full daily-state archive.
            frost = []
            for begin in range(0, len(site_index), 24000):
                ix = site_index[begin:begin+24000]
                active = prepared["main"][ix, crop_i, j]
                # Sowing to flowering: emergence date is not modelled, autumn frost after flowering excluded.
                frost.append(((prepared["weather"][ix, j, :, 2] < 0) &
                               (active[..., 0] < 2) & (active[..., 2] > 0)).any(-1))
            frost = torch.cat(frost)
            step = cycle[(year-first_season) % len(cycle)]
            cover_failure = None
            if step.get("winter"):
                ci = farm.cover_keys.index(step["winter"])
                cover_failure = float((~prepared["covers"]["alive"][site_index[mask], ci, crop_i, j]).float().mean())
            seasons.append({"year": year, "crop": crop.key, "distribution": quantities,
                            "risk_probability": {
                                "dry_planting": float((planting_water[mask]/soil_cap[mask] <
                                                        1-crop.v("depletion_fraction")).float().mean()),
                                "cover_failing_to_establish": cover_failure,
                                "late_frost_after_emergence": float(frost[mask].float().mean())}})
        result.append({"rotation": cycle, "seasons": seasons})
    return {"results": result, "weather": mode, "seas5": metadata,
            "method": {"initial_state_as_of": nowcast["as_of"], "first_crop_year": first_season,
                       "state_particles": particles, "historical_donors": len(donors), "donor_years": farm.base["years"],
                       "trajectories": len(particle), "equal_weight": True,
                       "joint_sampling": "All 51×45 weather pairs, each paired with state particle (member+7*donor)%32; all 32 states represented. Not a full weather×state Cartesian product. Climatology uses the identical pairings.",
                       "bridge": f"Unobserved gap from {last_record+1}-01-01 through {now.date()} uses historical members; NOT observed weather.",
                       "splice": "Every SEAS5 member × every historical donor; historical blocks outside current daily horizon. Equal weights, no donor or weather-member subsampling.",
                       "yield": "FAO-33 conditional response / historical uncapped reference; historical yield MLP is NOT supplied fabricated future covariates.",
                       "calendar": "Known state median planting dates; fixed median thermal scale from last 10 historical years, then weather-dependent maturity. No future observed dates or weather-conditioned cultivar selection.",
                       "risk_definitions": {"dry_planting": "Available profile fraction below FAO-56 1-p threshold at sowing",
                                            "late_frost_after_emergence": "Daily air Tmin<0 C between sowing and flowering of the harvest-year main crop (emergence date not modelled, so pre-emergence frost counts); exposure, not damage or crop-specific hardiness",
                                            "cover_failing_to_establish": "Thermal-time/winter survival screen only, NOT seedbed-water or measured stand success"},
                       "limitations": ["County surrogate, uncalibrated probabilities; no field winterkill/disease/management measurements.",
                                       "Intervening weather is unknown, so this is not a live-observed field forecast.",
                                       "Late autumn sowing decisions may precede issue date; verify actual standing crop before acting.",
                                       "Feb29 repeats Feb28 if donor year is nonleap."]},
            "latency_seconds": time.perf_counter()-started}


def quantiles(values):
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite forecast output; cannot silently drop members")
    q = torch.quantile(values.float(), torch.tensor([.1, .5, .9], device=values.device)).tolist()
    return {"p10": q[0], "median": q[1], "p90": q[2]}


def build(farm, payload, now=None):
    rotations, years, mode = validate(payload)
    cycles = [farm.parse(c) for c in rotations]
    cycles = [[c[0]]*4 if all(s["crop"] == "alfalfa" for s in c) else c for c in cycles]
    now = now or datetime.now(UTC)
    start = time.perf_counter()
    record, initial = state.load(farm)
    cached = None
    if mode == "outlook":
        static = state._static().loc[farm.site["adm_id"]]
        try:
            cached = outlook.fetch(farm.site["adm_id"], static.centroid_lat, static.centroid_lon,
                                   root=state.ROOT/"outlook", now=now)
        except ValueError as exc:
            raise ForecastUnavailable(f"No current SEAS5 forecast: {exc}. Climatology remains available.") from exc
    signature = hashlib.sha256(json.dumps([state.key(farm, 32), payload, farm.record.get("economics"),
                                           now.date().isoformat(), cached and cached["issue_date"],
                                           Path(__file__).read_text()], sort_keys=True).encode()).hexdigest()[:24]
    path = state.ROOT/"forecast"/(signature+".json")
    if path.exists():
        result = json.loads(path.read_text())
        result["cache_hit"] = True
        result["latency_seconds"] = time.perf_counter()-start
        return result
    climate = run(farm, cycles, years, "climatology", record, initial, now)
    result = run(farm, cycles, years, mode, record, initial, now, cached) if mode == "outlook" else climate
    for row, comparison in zip(result["results"], climate["results"], strict=True):
        for season, baseline in zip(row["seasons"], comparison["seasons"], strict=True):
            season["climatology"] = baseline["distribution"]
            season["difference_from_climatology"] = {
                k: season["distribution"][k]["median"]-baseline["distribution"][k]["median"] for k in FIELDS}
            season["risk_difference_from_climatology"] = {
                k: (season["risk_probability"][k]-v if v is not None else None)
                for k, v in baseline["risk_probability"].items()}
    result.update(simulator_version="fieldshift-v26", cache_hit=False, latency_seconds=time.perf_counter()-start)
    result = clean(result)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)
    return result
