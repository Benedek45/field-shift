"""v26 historical state ensemble. No changes to the default v25 scoring path.

SMAP L4 is itself a land-model analysis, not an independent field measurement.
The soil profile is the simulator's single, well-mixed plant-available bucket.
Its 0--1 m projection is compared with SMAP; two independent layers are NOT
invented. Parameter intervals below are sensitivity assumptions, not fitted
posteriors or calibrated confidence intervals.
"""

import calendar
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.agent.farms import physical_record
from fieldshift.planner.data import CDL, _static
from fieldshift.sim.engine import ledger_pass, prepare, water_pass
from fieldshift.sim.traits import CATALOG

ROOT = Path("data/processed/engine_v26")
PROTECTED_ASSETS = (
    "models/yield_response_v2_power.pt", "static/county_static.parquet",
    "planner/nass_reference_prices.json", "planner/nass_index_v2.parquet", "nass_crop_progress_dates.parquet",
    "planner/nass_practice_yields_v1.parquet", "planner/nass_state_yields_v3.parquet",
    "weather/power_county_daily.f32", "weather/power_county_daily.json",
    "weather/prism_county_8day.f32", "weather/prism_county_8day.json",
)
SMAP = "https://smap.jpl.nasa.gov/system/internal_resources/details/original/272_L4_SM_RevA_web.pdf"
SMAP_ERROR = "https://nsidc.org/sites/default/files/documents/technical-reference/igarss2026_reichle_et_al_v3_for_nsidc.pdf"
PARAMETERS = {
    "awc_scale": {"range": [.8, 1.2], "basis": "assumption: ±20% coarse county soil-map sensitivity"},
    "initial_om_scale": {"range": [.7, 1.3], "basis": "assumption: 1981 OM unknown, ±30% sensitivity"},
    "root_scale": {"range": [.8, 1.2], "basis": "assumption: ±20% around FAO-56 crop rooting-depth prior"},
    "kc_scale": {"range": [.85, 1.15], "basis": "assumption: ±15% county/cultivar Kc sensitivity, not a FAO error bar"},
    "mineralisation_rate_20c": {"range": [.01, .03], "basis": "assumption: 1–3%/yr around existing uncalibrated 2% prior"},
    "initial_available_water_fraction": {"range": [.1, 1.], "basis": "assumption: unknown 1981 depletion"},
    "initial_mineral_n": {"range": [5., 100.], "unit": "kg N/ha", "basis": "assumption: unknown 1981 fertiliser/residual N"},
    "initial_residue_n": {"range": [0., 80.], "unit": "kg N/ha", "basis": "assumption: unknown pre-1981 residue"},
    "initial_family_pressure": {"range": [0., 2.], "basis": "assumption: uncalibrated legacy family-pressure index"},
}
OBSERVATION_ERROR = {
    "sd_m3_m3": .06, "root_zone_depth_m": 1.,
    "source": SMAP_ERROR,
    "basis": "ASSUMPTION: 0.06 includes representativeness/soil-map error. SMAP V8 root-zone ubRMSE "
             "is 0.026 and product target 0.04; neither is a field observation-error SD. No tuning to innovations.",
    "operator": "8-day mean of well-mixed profile water projected to fixed SMAP 0–1m, plus soil-map wilting point",
    "frozen_mask": "8-day mean air temperature <=0 C excluded as a coarse frozen-ground proxy (assumption)",
}
STATE_NAMES = ("carbon", "mineral", "organic", "fast", "fast_rate_mass", "live_cover", "last",
               "streak", "erosion", "residue_carbon_proxy")


def kalman_update(state, predicted, observation, error_sd):
    """Deterministic scalar-observation EnSRF, ensemble axis 0.

    ``state`` is [members, variables]; ``predicted`` is [members]. No
    perturbed-observation sampling; mean/covariance equal the linear Kalman
    answer before physical clipping. Whitaker & Hamill (2002),
    doi:10.1175/1520-0493(2002)130<1913:EDAWPO>2.0.CO;2.
    """
    mean = state.mean(0)
    anomaly = state - mean
    hmean = predicted.mean()
    ha = predicted - hmean
    variance = ha.square().sum() / (len(predicted) - 1)
    covariance = (anomaly * ha[:, None]).sum(0) / (len(predicted) - 1)
    total = variance + error_sd ** 2
    gain = covariance / total.clamp_min(1e-12)
    alpha = 1 / (1 + torch.sqrt(torch.as_tensor(error_sd ** 2, device=state.device) / total))
    return mean + gain * (observation - hmean) + anomaly - alpha * ha[:, None] * gain


def history(site, years):
    """County representative CDL sequence, NOT observed history of this field."""
    existing = site["existing"]
    result = {y: existing[(y - 2019) % len(existing)] for y in range(min(years) - 1, max(years) + 2)}
    source = "Inferred cycle, phase anchored to 2019 (assumption); no field-level crop-history observation."
    if site.get("existing_origin") == "farmer":
        return result, "Current cycle stated by the farmer; extrapolating it back to 1981 with phase anchored to 2019 is an ASSUMPTION, not a measured historical sequence."
    path = Path("data/processed/qdann_points.csv.gz")
    if path.exists():
        names = [f"cdl_{y}" for y in range(2006, 2023)]
        frame = pd.read_csv(path, usecols=["county_fips", *names], dtype={"county_fips": "string"})
        frame = frame.loc[frame.county_fips.str.zfill(5).eq(site["adm_id"][3:5] + site["adm_id"][6:])]
        counts = Counter(tuple(CDL.get(int(v), "") if pd.notna(v) else "" for v in row)
                         for row in frame[names].itertuples(index=False, name=None))
        complete = {seq: count for seq, count in counts.items() if all(k in CATALOG for k in seq)}
        if complete:
            sequence = max(complete, key=lambda x: (complete[x], x))
            result.update(dict(zip(range(2006, 2023), sequence, strict=True)))
            source = (f"Modal complete 2006–2022 CDL sequence among county QDANN points ({complete[sequence]}/{len(frame)}); "
                      "county proxy, NOT this field's measured history. Inferred cycle outside observed years, anchored to 2019.")
    return result, source


def historical_trajectory(site, crops, years, members, covers=None):
    sequence, source = history(site, years)
    keys = [c.key for c in crops]
    array = np.array([keys.index(sequence[y]) for y in range(min(years)-1, max(years)+2)])
    n = len(years)
    cover = np.zeros(n + 2, dtype=np.int64)
    double = np.zeros(n + 2, dtype=bool)
    if site.get("existing_steps"):
        ck = [c.key for c in (covers or [])]
        for j, year in enumerate(range(min(years) - 1, max(years) + 2)):
            step = site["existing_steps"][(year - 2019) % len(site["existing_steps"])]
            if step.get("winter"):
                cover[j] = ck.index(step["winter"]) + 1
            double[j] = bool(step.get("double_crop"))
    return {
        "crop": np.tile(array, (members, 1)), "cover": np.tile(cover, (members, 1)),
        "double": np.tile(double, (members, 1)), "site": np.zeros(members, dtype=np.int64),
        "year0": np.zeros(members, dtype=np.int64), "rotation": np.zeros(members, dtype=np.int64),
        "position": np.tile(np.arange(n+2) % (4 if site.get("existing_origin") == "farmer"
                                             and set(site["existing"]) == {"alfalfa"} else len(site["existing"])), (members, 1)),
        "uniform": np.full(members, len(set(site["existing"])) == 1), "horizon": n, "spin": 0,
    }, source


def design(members, device):
    """Deterministic stratified parameter design (one stratum per member)."""
    index = torch.arange(members, device=device, dtype=torch.float32)
    primes = (1, 7, 11, 13, 17, 19, 23, 29, 31)
    values = {}
    for i, (name, spec) in enumerate(PARAMETERS.items()):
        u = ((index * primes[i] + i * 3) % members + .5) / members
        lo, hi = spec["range"]
        values[name] = lo + (hi - lo) * u
    return values


def initial_ensemble(farm, members):
    device = farm.base["device"]
    index = torch.arange(members, device=device, dtype=torch.float32)
    values = design(members, device)
    for key, name in (("awc_mm_per_m", "awc_scale"), ("organic_matter_pct", "initial_om_scale")):
        if key in farm.record["overrides"]:
            values[name] = torch.ones_like(index)
    soils = [dict(farm.site["soil"]) for _ in range(members)]
    # Metadata transfer once, outside the daily loop.
    scales = torch.stack((values["awc_scale"], values["initial_om_scale"]), -1).tolist()
    for soil, (a, o) in zip(soils, scales, strict=True):
        soil["awc_mm_per_m"] *= a
        soil["organic_matter_pct"] *= o
    capacity = torch.tensor([s["awc_mm_per_m"] * s["profile_depth_m"] for s in soils], device=device)
    mass = farm.site["soil"]["bulk_density_kg_m3"] * .3 * 10
    carbon = farm.site["soil"]["organic_matter_pct"] * values["initial_om_scale"] * mass / 100 / 1.724
    zeros = torch.zeros_like(index)
    initial = {"water": capacity * values["initial_available_water_fraction"], "carbon": carbon,
               "mineral": values["initial_mineral_n"], "organic": carbon * 1000 / 12,
               "fast": values["initial_residue_n"], "fast_rate_mass": values["initial_residue_n"] * .3,
               "live_cover": zeros.clone(), "pressure": values["initial_family_pressure"][:, None].expand(-1, 8).clone(),
               "last": torch.full_like(index, -1, dtype=torch.long), "streak": zeros.clone(), "erosion": zeros.clone()}
    context = {"initial": initial, "soil_index": torch.arange(members, device=device),
               "root_scale": values["root_scale"], "kc_scale": values["kc_scale"]}
    return soils, values, context


class Assimilator:
    """Water-only EnSRF: no unsupported assimilation of N, C or disease."""

    def __init__(self, site, years, depth, wilting, enabled=True):
        self.depth, self.wilting = depth, wilting
        self.year_lengths = [366 if calendar.isleap(y) else 365 for y in years]
        self.steps, self.log = {}, []
        self.total, self.count = None, 0
        self.last_observation = None
        if site.get("eo") is not None:
            for row in site["eo"].itertuples(index=False):
                date = pd.Timestamp(row.date)
                if date.year not in years:
                    continue
                end = min(date + pd.Timedelta(days=7), pd.Timestamp(date.year, 12, 31))
                theta = getattr(row, "smap_rootzone", float("nan"))
                weather = site["daily"].loc[site["daily"].date.between(date, end)]
                good = (enabled and np.isfinite(wilting) and pd.notna(theta) and 0 < theta < 1
                        and len(weather) and weather.tavg.mean() > 0)
                self.steps[(years.index(date.year), end.day_of_year-1)] = (float(theta) if good else None, end)

    def __call__(self, year, day, water, capacity, part):
        if part.start:
            raise ValueError("Historical EnSRF members must be kept in one chunk")
        if day >= self.year_lengths[year]:
            return water
        # Padded day 366 in a non-leap year has no observation or elapsed time.
        self.total = water.clone() if self.total is None else self.total + water
        self.count += 1
        if (year, day) not in self.steps:
            return water
        theta, date = self.steps[year, day]
        if theta is not None:
            forecast = self.total / self.count / self.depth + self.wilting * 1000
            updated = kalman_update(water[:, None], forecast, theta * 1000,
                                    OBSERVATION_ERROR["sd_m3_m3"] * 1000).squeeze(-1)
            updated = torch.minimum(updated.clamp_min(0), capacity)
            self.log.append(torch.stack((theta * 1000 - forecast.mean(), forecast.std(),
                                         water.std()/self.depth, updated.std()/self.depth,
                                         (updated-water).mean())))
            water = updated
            self.last_observation = date.date().isoformat()
        self.total, self.count = None, 0
        return water

    def diagnostics(self):
        if not self.log:
            return {"updates": 0, "verdict": "No valid unfrozen SMAP observations/soil wilting point"}
        log = torch.stack(self.log)
        sd = OBSERVATION_ERROR["sd_m3_m3"] * 1000
        return {"updates": len(log), "last_observation": self.last_observation,
                "innovation_mean_mm_0_1m": float(log[:, 0].mean()),
                "innovation_sd_mm_0_1m": float(log[:, 0].std()),
                "assumed_observation_sd_mm_0_1m": sd,
                "expected_innovation_sd_mm": float((log[:, 1].square()+sd**2).mean().sqrt()),
                "forecast_spread_mm": float(log[:, 2].mean()), "analysis_spread_mm": float(log[:, 3].mean()),
                "spread_reduction_fraction": float(1-log[:, 3].mean()/log[:, 2].mean().clamp_min(1e-8)),
                "net_analysis_increment_profile_mm": float(log[:, 4].sum()),
                "verdict": "Diagnostics, NOT calibration or independent validation; SMAP L4 is a model analysis."}


def key(farm, members):
    code = Path(__file__).parents[1]
    names = ("state", "engine", "nitrogen", "calendar", "traits", "components", "economics", "alfalfa", "kansas")
    sources = [code/"sim"/(name+".py") for name in names] + [code/"planner"/"data.py"]
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    assets = []
    for name in ("eo/eo_county_8day.f32", "eo/eo_county_8day.json", "qdann_points.csv.gz"):
        path = Path("data/processed")/name
        assets.append((name, path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else (name, None))
    for name in PROTECTED_ASSETS:
        path = Path("data/processed")/name
        assets.append((name, path.stat().st_size, path.stat().st_mtime_ns))
    body = json.dumps([physical_record(farm.record), members, fingerprint, PARAMETERS, OBSERVATION_ERROR, assets], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:24]


def variable(values, unit, source, basis="model_only", last_observation=None):
    q = torch.quantile(values.float(), torch.tensor([.1, .5, .9], device=values.device)).tolist()
    return {"value": q[1], "unit": unit, "p10": q[0], "p90": q[2], "basis": basis,
            "last_observation": last_observation, "source": source}


@torch.inference_mode()
def replay(farm, members=32, assimilate=True):
    """Return JSON record plus accelerator arrays used by conditional forecasts."""
    started = time.perf_counter()
    if members not in (32, 64):
        raise ValueError("State ensemble must have 32 or 64 members")
    years = farm.base["years"]
    seq, crop_source = history(farm.site, years)
    keys = list(dict.fromkeys(farm.keys + list(seq.values())))
    crops = [next((c for c in farm.crops if c.key == k), CATALOG[k]) for k in keys]
    site = dict(farm.site, daily=farm.weather())
    # One calendar-only padding year permits the *observed final autumn* to
    # carry a winter crop into the next harvest year. It is never replayed.
    padding = site["daily"].loc[site["daily"].date.dt.year.eq(max(years))].copy()
    target = pd.date_range(f"{max(years)+1}-01-01", f"{max(years)+1}-12-31")
    padding = padding.iloc[np.minimum(np.arange(len(target)), len(padding)-1)].copy()
    padding["date"] = target
    calendar_site = dict(site, daily=pd.concat([site["daily"], padding], ignore_index=True))
    prepared = prepare([calendar_site], crops, farm.covers, farm.double, irrigated=farm.irrigated, learned=False)
    traj, _ = historical_trajectory(site, crops, years, members, farm.covers)
    soils, parameters, context = initial_ensemble(farm, members)
    static = _static().loc[site["adm_id"]]
    wilting = float(static.wilting_point_0_100)
    depth = site["soil"]["profile_depth_m"]
    assimilator = Assimilator(site, years, depth, wilting, assimilate)
    context["observe"] = assimilator
    water = water_pass(prepared, soils, crops, farm.covers, farm.double, traj, farm.irrigated,
                       continuation=context, daily_log=True)
    water["temperature_month"] *= parameters["mineralisation_rate_20c"][:, None, None] / .02
    reference = torch.ones((1, len(crops)+1), device=prepared["device"])
    reference[:, :len(farm.crops)] = farm.reference[:, :len(farm.crops)]
    normals = [[site["normals"].get(c.key, c.v("normal_t_ha")) for c in crops]]
    ledger = ledger_pass(prepared, water, soils, normals, crops, farm.covers, farm.double, traj,
                         farm.irrigated, fao_reference=reference, continuation=context, nitrogen_log=True)
    terminal = {k: ledger["ledger_state"][:, -1, i] for i, k in enumerate(STATE_NAMES)}
    terminal["last"] = terminal["last"].long()
    terminal["pressure"] = ledger["ledger_state"][:, -1, 10:]
    ws = water["water_state"][:, -1]
    terminal.update(water=ws[:, 0], carry_a=ws[:, 1:5], carry_p=ws[:, 5:9],
                    carry_bio=ws[:, 9], carry_gdd=ws[:, 10])
    if "organic_matter_pct" in farm.record["overrides"]:
        terminal["carbon"] = torch.full_like(terminal["carbon"], site["soil"]["bulk_density_kg_m3"]*.3*10
                                              *site["soil"]["organic_matter_pct"]/100/1.724)
    diagnostics = assimilator.diagnostics()
    model_source = "v26 historical FAO/bucket + conservative monthly N/carbon ledger; county proxy; uncalibrated"
    basis = "observed+assimilated" if diagnostics["updates"] else "model_only"
    last = assimilator.last_observation
    variables = {
        "profile_available_water": variable(terminal["water"], "mm above wilting point", model_source, basis, last),
        "root_zone_total_water": variable(terminal["water"]/depth + wilting*1000, "mm in fixed SMAP 0–1m",
                                         SMAP+"; homogeneous profile projection; soil-map wilting point", basis, last),
        "profile_total_water": variable(terminal["water"] + wilting*1000*depth, "mm", model_source, basis, last),
        "mineral_n": variable(terminal["mineral"], "kg N/ha", model_source),
        "organic_c_0_30cm": variable(terminal["carbon"], "t C/ha", model_source),
        "organic_n": variable(terminal["organic"], "kg N/ha", model_source),
        "residue_n": variable(terminal["fast"], "kg N/ha", model_source),
        "residue_c_proxy": variable(terminal["residue_carbon_proxy"], "t C/ha",
                                    "Fast residue N × current residue C:N; derived proxy, NOT separately conserved C store"),
        "living_cover_n": variable(terminal["live_cover"], "kg N/ha", model_source),
        "cumulative_erosion_since_1981": variable(terminal["erosion"], "t soil/ha", model_source),
    }
    for name, column, unit in (("available_water_capacity", "awc_mm_per_m", "mm/m"),
                               ("organic_matter", "organic_matter_pct", "% by mass"), ("slope", "slope_pct", "%")):
        v = torch.tensor([s[column] for s in soils], device=prepared["device"])
        if column == "organic_matter_pct":
            v = terminal["carbon"] * 1.724 / (site["soil"]["bulk_density_kg_m3"]*.3*10) * 100
        if column in farm.record["overrides"]:
            v = torch.full_like(v, site["soil"][column])
        variables[name] = variable(v, unit, "farmer registry (date/measurement error not supplied)" if
                                   column in farm.record["overrides"] else "OpenLandMap county soil / SRTM; parameter sensitivity",
                                   "farmer" if column in farm.record["overrides"] else "model_only")
    if "organic_matter_pct" in farm.record["overrides"]:
        variables["organic_c_0_30cm"] = variable(terminal["carbon"], "t C/ha",
                                               "farmer OM converted with bulk density and C=OM/1.724", "farmer")
    for family in range(8):
        variables[f"family_pressure_{family}"] = variable(terminal["pressure"][:, family], "uncalibrated index", model_source)
        encountered = []
        for j, year in enumerate(years):
            ci, cn = int(traj["crop"][0, j+1]), int(traj["crop"][0, j+2])
            for c, table in ((ci, prepared["main"]), (cn, prepared["lead"])):
                if int(crops[c].v("family")) == family:
                    active = table[0, c, j, :, 2] > 0
                    indices = torch.where(active)[0]
                    if len(indices):
                        encountered.append((pd.Timestamp(year, 1, 1)+pd.Timedelta(days=int(indices[-1]))).date())
        end = pd.Timestamp(max(years), 12, 31).date()
        age = (end-max(encountered)).days if encountered else None
        variables[f"days_since_family_{family}"] = {
            "value": age, "p10": age, "p90": age, "unit": "days", "basis": "model_only", "last_observation": None,
            "source": crop_source + ("; not encountered since 1981; earlier history unknown" if age is None else "")}
    current = []
    for ci, crop in enumerate(crops):
        if bool(prepared["main"][0, ci, len(years)-1, 364, 2] > 0) and crop.key == seq[years[-1]]:
            current.append(crop.key)
        if bool(prepared["lead"][0, ci, len(years)-1, 364, 2] > 0) and crop.key == seq[years[-1]+1]:
            current.append(crop.key+" (next harvest year; calendar proxy)")
    ndvi = ndvi_agreement(site, prepared, traj)
    record = {"simulator_version": "fieldshift-v26", "as_of": f"{max(years)}-12-31", "members": members,
              "scope": "County surrogate, NOT a measured field or live-date nowcast; all state quantiles are uncalibrated",
              "variables": variables, "current_crop_or_cover": {
                  "value": current or ["bare/retained residues"], "p10": None, "p90": None, "unit": "crop identity",
                  "basis": "model_only", "last_observation": None, "source": crop_source},
              "history_source": crop_source, "assimilation": diagnostics, "observation_error": OBSERVATION_ERROR,
              "ndvi_calendar_check": ndvi, "parameter_ranges": PARAMETERS,
              "limitations": ["SMAP L4 shares reanalysis/model information; observations are county averages.",
                             "Single well-mixed profile: root and profile water are NOT independent layers.",
                             "N, C, residue, erosion and disease indices are unvalidated model-only estimates.",
                             "No independent residue-C pool: reported residue C is a C:N proxy.",
                             "Weather ends 2025; no claim to have observed the intervening year.",
                             "A repeated final weather year supplies only next-year winter-crop calendar padding; not replayed.",
                             "Organic C and N pools are separate simplifications; no validated coupled soil C:N dynamics."],
              "latency_seconds": time.perf_counter()-started}
    arrays = {"initial": terminal, "parameters": parameters, "soils": soils, "crops": crops,
              "prepared": prepared, "trajectory": traj, "water": water, "ledger": ledger,
              "reference": reference, "years": years}
    return record, arrays


def load(farm, members=32):
    """Private versioned cache; only final serialization crosses to host memory."""
    started = time.perf_counter()
    directory = ROOT / "state"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / key(farm, members)
    meta, data = path.with_suffix(".json"), path.with_suffix(".npz")
    if meta.exists() and data.exists():
        saved = json.loads(meta.read_text())
        device = farm.base["device"]
        with np.load(data) as archive:
            initial = {k.removeprefix("initial_"): torch.tensor(archive[k], device=device)
                       for k in archive.files if k.startswith("initial_")}
            parameters = {k.removeprefix("parameter_"): torch.tensor(archive[k], device=device)
                          for k in archive.files if k.startswith("parameter_")}
            reference = torch.tensor(archive["reference"], device=device)
        arrays = {"initial": initial, "parameters": parameters, "reference": reference,
                  "soils": saved["soils"], "crops": [CATALOG[k] for k in saved["crops"]],
                  "years": saved["years"]}
        record = saved["record"]
        record["cache_hit"] = True
        record["latency_seconds"] = time.perf_counter()-started
        return record, arrays
    record, arrays = replay(farm, members)
    saved = {"record": record, "soils": arrays["soils"], "crops": [c.key for c in arrays["crops"]],
             "years": [int(y) for y in arrays["years"]]}
    host = {"initial_"+k: v.cpu().numpy() for k, v in arrays["initial"].items()}
    host.update({"parameter_"+k: v.cpu().numpy() for k, v in arrays["parameters"].items()})
    host["reference"] = arrays["reference"].cpu().numpy()
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(temporary, **host)
    temporary.chmod(0o600)
    temporary.replace(data)
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(saved, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(meta)
    record["cache_hit"] = False
    return record, arrays


def ndvi_agreement(site, prepared, traj):
    eo = site.get("eo")
    if eo is None:
        return {"observations": 0}
    years = list(prepared["years"])
    rows = eo.loc[eo.date.ge("2000-02-18") & eo.ndvi.notna()]
    dates = pd.to_datetime(rows.date)
    keep = dates.dt.year.isin(years).to_numpy()
    if not keep.any():
        return {"observations": 0}
    dates, ndvi = dates[keep], rows.ndvi.to_numpy()[keep]
    device = prepared["device"]
    # Composite index metadata is built once on the host; the calendar lookup
    # and window means run as one gather on the accelerator.
    j = torch.tensor([years.index(y) for y in dates.dt.year], device=device)
    start = torch.tensor(dates.dt.day_of_year.to_numpy() - 1, device=device)
    length = torch.tensor(np.where(dates.dt.is_leap_year, 366, 365), device=device)
    crop = torch.as_tensor(np.asarray(traj["crop"][0]), device=device)
    c, cn = crop[j + 1], crop[j + 2]
    window = start[:, None] + torch.arange(8, device=device)[None]
    valid = window < torch.minimum(start + 8, length)[:, None]
    window = window.clamp(max=365)
    active = ((prepared["main"][0, c, j, :, 2] > 0) | (prepared["lead"][0, cn, j, :, 2] > 0)).float()
    share = (active.gather(1, window) * valid).sum(-1) / valid.sum(-1)
    flags = share >= .5
    y = torch.as_tensor(ndvi >= .3, device=device)
    return {"observations": len(ndvi), "agreement_share": float((flags == y).float().mean()),
            "threshold": .3, "basis": "ASSUMPTION NDVI>=0.3 indicates county greenness; NOT a field crop classifier.",
            "source": "MODIS MOD13A2 county composites; check only, NOT assimilated"}
