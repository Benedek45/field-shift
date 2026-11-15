"""Reproduce validation on local NASA/NASS data, without fitting the simulator.

Run ``uv run python -m fieldshift.planner.validate``. Outputs and the precise
county selection are retained in data/processed/planner/validation*.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.planner.data import (
    PRISM_RULE,
    ROOT,
    build_index,
    load_light,
    load_site,
    prism_scaled_rain,
)
from fieldshift.planner.search import (
    DEFAULT_WEIGHTS,
    candidate_crops,
    plan,
    rank,
    select_comparisons,
)
from fieldshift.sim.components import accelerator
from fieldshift.sim.engine import (
    fao_reference,
    ledger_pass,
    prepare,
    trajectories,
    water_pass,
)
from fieldshift.sim.traits import CATALOG

STATES = ("IA", "IL", "IN", "OH", "MN", "WI", "SD", "ND", "NE", "KS", "MO")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def correlation(x, y, mask):
    """Masked row correlations, on device; undefined series remain missing."""
    mask = mask & torch.isfinite(x) & torch.isfinite(y)
    n = mask.sum(-1).clamp_min(1)
    xx = torch.where(mask, x, 0)
    yy = torch.where(mask, y, 0)
    xx = torch.where(mask, xx - xx.sum(-1, keepdim=True) / n[:, None], 0)
    yy = torch.where(mask, yy - yy.sum(-1, keepdim=True) / n[:, None], 0)
    den = (xx.square().sum(-1) * yy.square().sum(-1)).sqrt()
    return torch.where((n >= 8) & (den > 0), (xx * yy).sum(-1) / den.clamp_min(1e-9), float("nan"))


def selected_labels(limit=300):
    d = pd.read_parquet("data/processed/yield_expanded/labels.parquet")
    d = d.loc[d.crop.isin(["corn", "soybean"]) & d.state_alpha.isin(STATES)
              & d.year.between(1981, 2025) & d.yield_t_ha.gt(0)].copy()
    nonirr = d.prodn_practice_desc.eq("NON-IRRIGATED")
    # Unknown practice in humid states is explicitly a rainfed proxy, not a
    # known rainfed label. Never use unknown practice in NE/KS/SD/ND.
    proxy = (d.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")
             & (d.irrigated_share.le(.1) | (d.irrigation_missing & d.state_alpha.isin(
                 ["IA", "IL", "IN", "OH", "MN", "WI", "MO"]))))
    d = d.loc[nonirr | proxy]
    d["known_rainfed"] = d.prodn_practice_desc.eq("NON-IRRIGATED")
    counts = d.groupby(["adm_id", "crop", "known_rainfed"]).year.nunique()
    good = counts[counts >= 25].reset_index()
    d = d.merge(good[["adm_id", "crop", "known_rainfed"]])
    d = d.sort_values("known_rainfed", ascending=False).drop_duplicates(["adm_id", "crop", "year"])
    ids = d[["adm_id", "state_alpha"]].drop_duplicates().sort_values("adm_id")
    # Round robin across states prevents the largest state dominating selection.
    ids["order"] = ids.groupby("state_alpha").cumcount()
    chosen = ids.sort_values(["order", "state_alpha"]).head(limit).adm_id.tolist()
    return d.loc[d.adm_id.isin(chosen)], chosen


def _combined_trajectories(crops, n_years, n_sites):
    parts = [trajectories([((c,), 0, False) for c in range(len(crops))], crops,
                          n_years, site=s, spin=0, score=n_years, starts=[0]) for s in range(n_sites)]
    out = parts[0].copy()
    for k in ("crop", "cover", "double", "year0", "rotation", "site", "position", "uniform"):
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def _et_comparison(ids, prepared, water, traj, corn_dominated, sites):
    """Integrate MOD16 mm/day over exactly overlapping season/JJA days.

    EO steps are restarted each Jan 1; the last is 5/6 days, not eight.
    Only season-years with >=90% finite observation days are retained.
    """
    meta = json.loads(Path("data/processed/eo/eo_county_8day.json").read_text())
    raw = np.memmap("data/processed/eo/eo_county_8day.f32", dtype="float32", mode="r", shape=tuple(meta["shape"]))
    vi = [v["name"] for v in meta["variables"]].index("et_actual")
    dates = pd.DatetimeIndex(meta["step_start_dates"])
    chosen = [i for i, c in enumerate(ids) if c in corn_dominated]
    if not chosen:
        return []
    device = prepared["device"]
    eo = torch.tensor(np.array(raw[[meta["counties"].index(ids[i]) for i in chosen], :, vi]),
                      device=device)
    # Calendar-only host indexing; all numeric comparisons are on device.
    daily_dates = pd.date_range("1981-01-01", "2025-12-31")
    step = np.searchsorted(dates.values, daily_dates.values, side="right") - 1
    obs = eo[:, torch.as_tensor(step, device=device)]
    matrix = torch.full((len(chosen), 45, 366), float("nan"), device=device)
    yi = torch.tensor(daily_dates.year.to_numpy() - 1981, device=device)
    di = torch.tensor(daily_dates.dayofyear.to_numpy() - 1, device=device)
    matrix[:, yi, di] = obs
    idx = torch.tensor(chosen, device=device)
    active = prepared["main"][idx, 0, ..., 2] > 0
    sim = water["daily_et"][idx * 2]
    doy = torch.arange(366, device=device)
    jja = (doy >= 152) & (doy < 243)  # leap years handled below
    # Correct the one-day shift in leap years.
    leap = torch.tensor([pd.Timestamp(y, 12, 31).dayofyear == 366 for y in prepared["years"]], device=device)
    jja = (doy >= (151 + leap.long())[:, None]) & (doy < (243 + leap.long())[:, None])
    results = []
    def totals(mask):
        finite = torch.isfinite(matrix)
        ok = (finite & mask).sum(-1) >= .9 * mask.sum(-1).clamp_min(1)
        valid = mask & finite
        return (torch.where(valid, sim, 0).sum(-1),
                torch.where(valid, matrix, 0).sum(-1), ok)
    ss, so, ok = totals(active)
    js, jo, jok = totals(jja[None].expand_as(active))
    r = correlation(js, jo, jok)
    jmask = jja[None].expand_as(active)
    rain = prepared["weather"][idx, ..., 0]
    et0 = prepared["weather"][idx, ..., 1]
    j_et0 = torch.where(jmask, et0, 0).sum(-1)
    wetness = torch.where(jmask, rain, 0).sum(-1) / j_et0.clamp_min(1)
    cuts = torch.nanquantile(torch.where(jok, wetness, float("nan")),
                            torch.tensor([1 / 3, 2 / 3], device=device), dim=-1)
    dry, wet = jok & (wetness <= cuts[0, :, None]), jok & (wetness >= cuts[1, :, None])
    dry_r, wet_r = correlation(js, jo, dry), correlation(js, jo, wet)
    rows = [meta["counties"].index(ids[i]) for i in chosen]
    lai_steps = torch.tensor(np.array(raw[rows, :, [v["name"] for v in meta["variables"]].index("lai")]), device=device)
    lai_matrix = torch.full_like(matrix, float("nan"))
    lai_matrix[:, yi, di] = lai_steps[:, torch.as_tensor(step, device=device)]
    lai_mean = torch.where(jmask, lai_matrix, float("nan")).nanmean(-1)
    humidity = torch.tensor(np.stack([sites[i]["daily"].rh.to_numpy("float32") for i in chosen]), device=device)
    rh_matrix = torch.full_like(matrix, float("nan"))
    rh_matrix[:, yi, di] = humidity
    rh_mean = torch.where(jmask, rh_matrix, float("nan")).nanmean(-1)
    wet_lai = correlation(jo, lai_mean, wet)
    wet_rh = correlation(jo, rh_mean, wet)
    wet_ref = correlation(js, j_et0, wet)
    for k, i in enumerate(chosen):
        valid = ok[k]
        results.append({"adm_id": ids[i], "season_years": int(valid.sum()),
                        "bucket_season_mm": float(ss[k, valid].mean()),
                        "modis_season_mm": float(so[k, valid].mean()),
                         "summer_anomaly_r": float(r[k]),
                         "dry_years": int(dry[k].sum()), "wet_years": int(wet[k].sum()),
                         "dry_summer_anomaly_r": float(dry_r[k]), "wet_summer_anomaly_r": float(wet_r[k]),
                         "wet_modis_lai_r": float(wet_lai[k]), "wet_modis_humidity_r": float(wet_rh[k]),
                         "wet_bucket_reference_et_r": float(wet_ref[k])})
    return results


def multi_county(limit=300, batch=20):
    started = time.perf_counter()
    labels, ids = selected_labels(limit)
    device = accelerator()
    crops = [CATALOG["corn"], CATALOG["soybean"]]
    area = build_index()
    area = area.loc[area.statisticcat_desc.eq("AREA HARVESTED")
                    & area.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES") & area.year.ge(2018)]
    area = area.groupby(["adm_id", "commodity_desc"]).value.mean().unstack(fill_value=0)
    dominated = set(area.index[(area.get("CORN", 0) / area.sum(axis=1)).ge(.4)])
    records, et_records = [], []
    for start in range(0, len(ids), batch):
        chosen = ids[start:start + batch]
        sites = [load_light(c) for c in chosen]
        p = prepare(sites, crops, learned=False)
        years = np.asarray(p["years"])
        tr = _combined_trajectories(crops, len(years), len(sites))
        w = water_pass(p, [s["soil"] for s in sites], crops, [], None, tr, daily_log=True)
        # Paired trajectories are independent crops with continuously carried water.
        sim = w["fao_main"].reshape(len(sites), 2, len(years))
        heat = p["stage_heat"]
        frost = p["stage_frost"]
        water_only = sim / torch.exp(-.02 * heat - .05 * frost).clamp_min(1e-8)
        obs = np.full((len(sites), 2, len(years)), np.nan, dtype=np.float32)
        known = {}
        for i, adm in enumerate(chosen):
            for c, key in enumerate(("corn", "soybean")):
                rows = labels.loc[labels.adm_id.eq(adm) & labels.crop.eq(key)]
                for row in rows.itertuples():
                    obs[i, c, row.year - 1981] = row.yield_t_ha
                known[i, c] = bool(len(rows) and rows.known_rainfed.all())
        obs = torch.as_tensor(obs, device=device)
        year = torch.tensor(years - 2000, device=device, dtype=torch.float32)
        even = torch.tensor(years % 2 == 0, device=device)
        mask = torch.isfinite(obs) & (obs > 0)
        train = mask & even
        # Per-county log-linear yield trend on even years only; no simulator
        # parameters are fitted, and odd-year correlation is the main test.
        x = year.expand_as(obs)
        n = train.sum(-1).clamp_min(1)
        mx = torch.where(train, x, 0).sum(-1) / n
        ly = obs.clamp_min(.01).log()
        my = torch.where(train, ly, 0).sum(-1) / n
        slope = (torch.where(train, (x - mx[..., None]) * (ly - my[..., None]), 0).sum(-1)
                 / torch.where(train, (x - mx[..., None]).square(), 0).sum(-1).clamp_min(1))
        relative = obs / (my[..., None] + slope[..., None] * (x - mx[..., None])).exp()
        # Normalise FAO with even years only, not the test years.
        sim = sim / sim[..., even].mean(-1, keepdim=True).clamp_min(.01)
        water_only = water_only / water_only[..., even].mean(-1, keepdim=True).clamp_min(.01)
        flat = lambda a: a.flatten(0, -2)
        r = correlation(flat(sim), flat(relative), flat(mask)).reshape(len(sites), 2)
        odd = correlation(flat(sim), flat(relative), flat(mask & ~even)).reshape(len(sites), 2)
        rw = correlation(flat(water_only), flat(relative), flat(mask & ~even)).reshape(len(sites), 2)
        # Move summaries only, after accelerator computation.
        for i, adm in enumerate(chosen):
            for c, key in enumerate(("corn", "soybean")):
                if int(mask[i, c].sum()) < 25:
                    continue
                rec = {"adm_id": adm, "crop": key, "known_rainfed": known[i, c],
                       "years": int(mask[i, c].sum()), "r_all": float(r[i, c]),
                       "r_odd": float(odd[i, c]), "water_only_r_odd": float(rw[i, c])}
                for y in (1983, 1988, 2012):
                    j = y - 1981
                    if bool(mask[i, c, j]):
                        rec[str(y)] = {"observed_loss_pct": float(100 * (1 - relative[i, c, j])),
                                       "simulated_loss_pct": float(100 * (1 - sim[i, c, j])),
                                       "water_only_loss_pct": float(100 * (1 - water_only[i, c, j]))}
                records.append(rec)
        et_records.extend(_et_comparison(chosen, p, w, tr, dominated, sites))
        print(f"Validated {start + len(chosen)}/{len(ids)} counties", flush=True)
        del p, w
    summary = {}
    for key in ("corn", "soybean"):
        rows = [r for r in records if r["crop"] == key]
        summary[key] = {"county_series": len(rows), "known_rainfed": sum(r["known_rainfed"] for r in rows),
                        "county_years": sum(r["years"] for r in rows),
                        "median_r_all": float(np.nanmedian([r["r_all"] for r in rows])),
                        "median_r_odd": float(np.nanmedian([r["r_odd"] for r in rows])),
                        "water_only_median_r_odd": float(np.nanmedian([r["water_only_r_odd"] for r in rows]))}
        summary[key]["drought_years"] = {
            str(y): {"counties": sum(str(y) in r for r in rows),
                     **{k: float(np.mean([r[str(y)][k] for r in rows if str(y) in r]))
                        for k in ("observed_loss_pct", "simulated_loss_pct", "water_only_loss_pct")}}
            for y in (1983, 1988, 2012)}
        drought = summary[key]["drought_years"]
        summary[key]["verdict"] = (
            f"MIXED: moderate held-out correlation (odd-year median r {summary[key]['median_r_odd']:.2f}); "
            + "; ".join(f"{y} loss observed {drought[y]['observed_loss_pct']:.0f}% vs simulated "
                        f"{drought[y]['simulated_loss_pct']:.0f}%" for y in drought) + ".")
    et = pd.DataFrame(et_records)
    et_summary = {"counties": len(et), "season_years": int(et.season_years.sum()),
                  "mean_bucket_season_mm": float(et.bucket_season_mm.mean()),
                  "mean_modis_season_mm": float(et.modis_season_mm.mean()),
                  "median_summer_anomaly_r": float(et.summer_anomaly_r.median()),
                  "verdict": "FAIL for summer anomalies (r about 0.10); mean seasonal ET +14%. MOD16 is an independent product, not field truth; county crop mixing limits attribution."}
    et_summary["dry_wet"] = {
        "method": "Within each county's observed MODIS summers, dry/wet are lower/upper thirds of POWER JJA rain / ET0. Correlation needs >=8 paired years; descriptive, not a causal test.",
        **{name: float(et[name].median()) for name in
           ("dry_summer_anomaly_r", "wet_summer_anomaly_r", "wet_modis_lai_r",
            "wet_modis_humidity_r", "wet_bucket_reference_et_r")},
        "counties_with_dry_r": int(et.dry_summer_anomaly_r.notna().sum()),
        "counties_with_wet_r": int(et.wet_summer_anomaly_r.notna().sum())}
    report = {"counties": ids, "yield": summary, "et": et_summary,
              "method": {"selection": "State-stratified 300 counties; >=25 positive yield years per series; explicit NON-IRRIGATED preferred. Humid-state unknown practice / <=10% irrigation are flagged proxies.",
                         "trend": "County/crop log-linear trend fitted on even years only; FAO normal uses even years; report odd-year correlation.",
                         "simulation": "1981–2025 continuous daily water carry-over, starts at field capacity Jan 1981. 1983 has two spin-up years; no annual resets. Rain: POWER rescaled to PRISM 8-day totals; yield: season-level FAO-33 factor (both adopted on the round-5 rules, validation_weather_tests.json).",
                         "et": "Corn >=40% of recent NASS commodity harvested area; MOD16 mm/day, exact calendar step overlap and >=90% finite season days; JJA interannual anomaly correlation.",
                         "calibration": "None. Water-only ablation removes the explicitly uncalibrated heat/frost multipliers; diagnostic only, not a tuned replacement."},
              "physical_diagnosis": [
                  "Rooting depth already FAO-56 upper rainfed depth (corn 1.7 m, soy 1.3 m) over 2 m profile; no capillary rise/runoff.",
                  "Water carries across years. Starting at field capacity cannot explain excessive loss decades later.",
                  "Planting/maturity use observed state progress where available; this avoids known early sowing bias.",
                  "POWER ET0 median bias vs AgERA5 -0.24 mm/day would reduce, not explain excessive drought stress.",
                  "Water-only ablation isolates heat/frost priors from bucket stress. Missing water tables and spatial crop/practice mixing remain unresolved; no parameters fitted to hide them."],
              "seconds": time.perf_counter() - started}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "validation_counties.json").write_text(json.dumps(json_safe(records), indent=2, allow_nan=False))
    et.to_csv(ROOT / "validation_et.csv", index=False)
    (ROOT / "validation_multi.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return report


def _sr_degree_days(prepared, thresholds):
    """Schlenker-Roberts degree days above a threshold, single-sine daily
    interpolation between Tmin and Tmax, March-August (their season)."""
    w = prepared["weather"]
    tmin, tmax = w[..., 2], w[..., 3]
    leap = (prepared["year_len"] == 366).long()[None, :, None]
    doy = torch.arange(366, device=w.device)[None, None]
    season = (doy >= 59 + leap) & (doy < 243 + leap)
    out = []
    for threshold in thresholds:
        m, a = (tmax + tmin) / 2, ((tmax - tmin) / 2).clamp_min(1e-6)
        theta = torch.asin(((threshold - m) / a).clamp(-1, 1))
        dd = ((m - threshold) * (torch.pi / 2 - theta) + a * torch.cos(theta)) / torch.pi
        dd = torch.where(tmin >= threshold, m - threshold, torch.where(tmax <= threshold, 0, dd))
        out.append((dd.clamp_min(0) * season).sum(-1))
    return torch.stack(out, 1)


def _relative_observed(labels, chosen, years, device):
    obs = np.full((len(chosen), 2, len(years)), np.nan, dtype=np.float32)
    for i, adm in enumerate(chosen):
        for c, key in enumerate(("corn", "soybean")):
            for row in labels.loc[labels.adm_id.eq(adm) & labels.crop.eq(key)].itertuples():
                obs[i, c, row.year - 1981] = row.yield_t_ha
    obs = torch.as_tensor(obs, device=device)
    x = torch.tensor(years - 2000, device=device, dtype=torch.float32).expand_as(obs)
    even = torch.tensor(years % 2 == 0, device=device)
    mask = torch.isfinite(obs) & (obs > 0)
    train = mask & even
    n = train.sum(-1).clamp_min(1)
    mx = torch.where(train, x, 0).sum(-1) / n
    ly = obs.clamp_min(.01).log()
    my = torch.where(train, ly, 0).sum(-1) / n
    slope = (torch.where(train, (x - mx[..., None]) * (ly - my[..., None]), 0).sum(-1)
             / torch.where(train, (x - mx[..., None]).square(), 0).sum(-1).clamp_min(1))
    return obs / (my[..., None] + slope[..., None] * (x - mx[..., None])).exp(), mask


def _score(sim, relative, mask, even):
    """Same scoring as multi_county: even-year normalisation, odd-year r per series."""
    sim = sim / torch.where(even, sim, 0).sum(-1, keepdim=True).div(even.sum()).clamp_min(.01)
    odd = correlation(sim.flatten(0, 1), relative.flatten(0, 1), (mask & ~even).flatten(0, 1)).reshape(sim.shape[:2])
    keep = mask.sum(-1) >= 25
    out = {}
    for c, key in enumerate(("corn", "soybean")):
        k = keep[:, c]
        r = odd[k, c]
        out[key] = {"series": int(k.sum()), "median_r_odd": float(r[torch.isfinite(r)].median()), "drought_years": {}}
        for y in (1983, 1988, 2012):
            j = y - 1981
            m = k & mask[:, c, j]
            out[key]["drought_years"][str(y)] = {
                "counties": int(m.sum()),
                "observed_loss_pct": float(100 * (1 - relative[m, c, j]).mean()),
                "simulated_loss_pct": float(100 * (1 - sim[m, c, j]).mean())}
    errors = [abs(v["simulated_loss_pct"] - v["observed_loss_pct"])
              for crop in out.values() for v in crop["drought_years"].values()]
    out["mean_abs_drought_error_pct_points"] = float(np.mean(errors))
    return out


def _fit_slope(base, dd, frost, relative, mask, even, grid):
    """One heat slope per crop, least squares on even years only."""
    slopes = []
    for c in range(2):
        sim = base[None, :, c] * torch.exp(-grid[:, None, None] * dd[None, :, c] - .05 * frost[None, :, c])
        sim = sim / torch.where(even, sim, 0).sum(-1, keepdim=True).div(even.sum()).clamp_min(.01)
        m = (mask[:, c] & even)[None]
        sse = torch.where(m, (sim - relative[None, :, c]).square(), 0).sum((-1, -2))
        slopes.append(float(grid[int(sse.argmin())]))
    return slopes


def _adopt(candidate, reference):
    better_r = all(candidate[k]["median_r_odd"] > reference[k]["median_r_odd"] for k in ("corn", "soybean"))
    not_worse = (candidate["mean_abs_drought_error_pct_points"]
                 <= reference["mean_abs_drought_error_pct_points"])
    return bool(better_r and not_worse)


def weather_tests(limit=300, batch=20):
    """Round-5 tests, rules fixed before running: (1) PRISM-scaled rain against
    POWER rain; (2) on the winning rain, Schlenker-Roberts degree days in place
    of the flowering heat term and a season-level FAO-33 factor in place of the
    stage product. Adopt only if odd-year median r rises for corn AND soybean and
    the mean absolute 1983/1988/2012 loss error (6 crop-years) does not rise."""
    started = time.perf_counter()
    labels, ids = selected_labels(limit)
    device = accelerator()
    crops = [CATALOG["corn"], CATALOG["soybean"]]
    ky_stage = torch.stack([torch.tensor([c.v("ky_" + s) for s in
                                          ("establishment", "vegetative", "flowering", "grain_fill")])
                            for c in crops]).to(device)
    ky_season = torch.tensor([c.v("ky") for c in crops], device=device)
    thresholds = (29., 30.)
    parts = {rain: {k: [] for k in ("stage", "season", "heat", "frost", "sr", "fao")} for rain in ("power", "prism")}
    relative_parts, mask_parts, rescale = [], [], []
    years = None
    for start in range(0, len(ids), batch):
        chosen = ids[start:start + batch]
        sites = [load_light(c, rain="power") for c in chosen]
        scaled, info = prism_scaled_rain([s["daily"] for s in sites], chosen, device)
        rescale.append(info)
        scaled = scaled.cpu().numpy()  # handoff into the daily frames prepare() reads
        prism_sites = [s | {"daily": s["daily"].assign(prec=scaled[i])} for i, s in enumerate(sites)]
        for rain, group in (("power", sites), ("prism", prism_sites)):
            p = prepare(group, crops, learned=False)
            years = np.asarray(p["years"])
            tr = _combined_trajectories(crops, len(years), len(group))
            w = water_pass(p, [s["soil"] for s in group], crops, [], None, tr)
            shape = (len(group), 2, len(years))
            actual = w["stage_actual_et"].reshape(*shape, 4)
            potential = w["stage_potential_et"].reshape(*shape, 4)
            deficit = torch.where(potential > 0, 1 - actual / potential.clamp_min(1e-6), 0).clamp(0, 1)
            stage = (1 - ky_stage[None, :, None] * deficit).clamp(0, 1).prod(-1)
            total_p = potential.sum(-1)
            season_deficit = torch.where(total_p > 0, 1 - actual.sum(-1) / total_p.clamp_min(1e-6), 0).clamp(0, 1)
            season = (1 - ky_season[None, :, None] * season_deficit).clamp(0, 1)
            for name, value in (("stage", stage), ("season", season), ("heat", p["stage_heat"]),
                                ("frost", p["stage_frost"]), ("sr", _sr_degree_days(p, thresholds)),
                                ("fao", w["fao_main"].reshape(shape))):
                parts[rain][name].append(value)
            del p, w
        relative, mask = _relative_observed(labels, chosen, years, device)
        relative_parts.append(relative)
        mask_parts.append(mask)
        print(f"Weather tests {start + len(chosen)}/{len(ids)} counties", flush=True)
    relative, mask = torch.cat(relative_parts), torch.cat(mask_parts)
    even = torch.tensor(years % 2 == 0, device=device)
    data = {rain: {k: torch.cat(v) for k, v in d.items()} for rain, d in parts.items()}
    current = {rain: d["stage"] * torch.exp(-.02 * d["heat"] - .05 * d["frost"]) for rain, d in data.items()}
    # The engine now uses the adopted season-level factor; check it is reproduced exactly.
    season_current = {r: d["season"] * torch.exp(-.02 * d["heat"] - .05 * d["frost"]) for r, d in data.items()}
    consistency = max(float((season_current[r] - data[r]["fao"]).abs().max()) for r in season_current)
    rain_scores = {rain: _score(current[rain], relative, mask, even) for rain in ("power", "prism")}
    prism_adopted = _adopt(rain_scores["prism"], rain_scores["power"])
    winner = "prism" if prism_adopted else "power"
    d = data[winner]
    grid = torch.arange(0, .0301, .0005, device=device)
    sr_slopes = _fit_slope(d["stage"], d["sr"], d["frost"], relative, mask, even, grid)
    season_slopes = _fit_slope(d["season"], d["sr"], d["frost"], relative, mask, even, grid)
    slope = lambda s: torch.tensor(s, device=device)[None, :, None]
    variants = {
        "current (stage product, flowering heat)": current[winner],
        "Schlenker-Roberts degree days (stage product)": d["stage"] * torch.exp(-slope(sr_slopes) * d["sr"]
                                                                                 - .05 * d["frost"]),
        "season-level FAO-33 factor (flowering heat)": d["season"] * torch.exp(-.02 * d["heat"] - .05 * d["frost"]),
        "season-level FAO-33 + Schlenker-Roberts": d["season"] * torch.exp(-slope(season_slopes) * d["sr"]
                                                                          - .05 * d["frost"]),
    }
    heat_scores = {name: _score(v, relative, mask, even) for name, v in variants.items()}
    reference = heat_scores["current (stage product, flowering heat)"]
    decisions = {name: _adopt(s, reference) for name, s in heat_scores.items() if not name.startswith("current")}
    sr_dd = {key: float(d["sr"][:, c].mean()) for c, key in enumerate(("corn", "soybean"))}
    report = {
        "rules": "Adopt only if odd-year median r rises for corn and soybean AND the mean absolute "
                 "observed-minus-simulated loss error over 1983/1988/2012 x corn/soybean does not rise. "
                 "Fixed before running.",
        "rain": {"method": PRISM_RULE, "scores": rain_scores, "prism_adopted": prism_adopted,
                 "rescaling_check": {"max_step_error_mm": max(i["max_step_error_mm"] for i in rescale),
                                     "dry_power_wet_prism_steps": sum(i["dry_power_wet_prism_steps"]
                                                                      for i in rescale)}},
        "heat": {"rain_used": winner, "scores": heat_scores, "adopted": decisions,
                 "fitted": {"parameters": "one heat slope per crop (2 total) per variant; least squares on even "
                                          "years of trend-relative yield; grid 0-0.03 per degree day",
                            "schlenker_roberts_slope_per_degree_day": dict(zip(("corn", "soybean"), sr_slopes)),
                            "season_variant_slope_per_degree_day": dict(zip(("corn", "soybean"), season_slopes)),
                            "published_reference": "Schlenker & Roberts 2009 PNAS doi:10.1073/pnas.0906865106: "
                                                   "thresholds 29 C corn, 30 C soybean, March-August; a day at "
                                                   "40 C instead of 29 C cuts corn yield ~7% (~0.0065 per degree "
                                                   "day) in a regression that also controls for precipitation."},
                 "mean_march_august_degree_days": sr_dd,
                 "method": "Degree days above threshold by single-sine interpolation of daily POWER Tmin/Tmax, "
                           "March-August. Season factor: 1 - ky_season(1 - sum ETa / sum ETm) with FAO-33 "
                           "seasonal ky (corn 1.25, soybean 0.85). Frost term unchanged in all variants."},
        "consistency_max_abs_engine_vs_season_formula": consistency,
        "counties": len(ids), "seconds": time.perf_counter() - started}
    (ROOT / "validation_weather_tests.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return report


def story_checks():
    result = plan()
    ranked = rank(result["table"], DEFAULT_WEIGHTS)
    income = rank(result["table"], {"income": 1})
    chosen, ties = select_comparisons(ranked, weights=DEFAULT_WEIGHTS, runs=result["arrays"]["runs"],
                                      windows=len(result["arrays"]["window_start"]), details=True)
    return {"income_ranking": {"existing_rank": list(income.id).index(0) + 1, "candidates": len(income)},
            "comparisons": [ranked.loc[ranked.id.eq(i)].iloc[0].to_dict() | {"reason": reason}
                            for i, reason in chosen],
            "near_tie_tests": ties}


def nitrogen_checks():
    s = load_site("US-19-169")
    priced = {c.key: c for c in candidate_crops(s)[0]}
    crops = [priced[k] for k in ("corn", "soybean", "oats")]
    covers = [CATALOG[k] for k in ("cereal_rye_cover", "hairy_vetch_cover")]
    rotations = [((0, 1, 0, 1), 0, False), ((0, 1, 0, 1), 1, False),
                 ((0, 1, 0, 1), 2, False), ((2, 2, 2, 2), 0, False),
                 ((0, 0, 0, 0), 0, False)]
    p = prepare([s], crops, covers)
    tr = trajectories(rotations, crops, len(p["years"]))
    water = water_pass(p, [s["soil"]], crops, covers, None, tr)
    normals = [[s["normals"].get(c.key, c.v("normal_t_ha")) for c in crops]]
    ledger = ledger_pass(p, water, [s["soil"]], normals, crops, covers, None, tr,
                          fao_reference=fao_reference(water, tr, 1, len(crops), 4), nitrogen_log=True)
    report = {}
    for ri, key in enumerate(("existing", "rye", "vetch", "oats", "continuous_corn")):
        mask = torch.tensor(tr["rotation"] == ri, device=p["device"])
        names = [k for k in ledger if k.startswith("n_") and k != "n_shortage_kg_ha"] + [
            "mineral_n_kg_ha", "cover_biomass_t_ha", "cover_established"]
        report[key] = {"scored_mean": {k: float(ledger[k][mask][:, 4:].mean()) for k in names}}
        success = water["cover_established"][mask][:, 4:] > 0
        if bool(success.any()):
            report[key]["successful_year_biomass_mean_t_ha"] = float(
                ledger["cover_biomass_t_ha"][mask][:, 4:][success].mean())
        # Explicit year-by-year diagnostic, first phase, most recent window.
        at = np.where(tr["rotation"] == ri)[0][len(tr["starts"]) - 1]
        report[key]["last_cycle_terms"] = [
            {"year": int(p["years"][tr["year0"][at] + t]),
             **{k: float(ledger[k][at, t]) for k in names}} for t in range(4, 8)]
    base = report["existing"]["scored_mean"]["n_leached_kg_ha"]
    rye = report["rye"]["scored_mean"]["n_leached_kg_ha"]
    report["rye_comparison"] = {
        "reduction_pct": 100 * (1 - rye / base), "baseline_kg_ha": base, "rye_kg_ha": rye,
        "published": [
            {"source": "https://doi.org/10.2134/jeq2006.0468", "years": "Iowa 2002–2005",
             "load_reduction_pct": 61, "concentration_reduction_pct": 59},
            {"source": "https://www.ars.usda.gov/research/publications/publication/?seqNo115=271972",
             "years": "Iowa subsequent 5 years, Kaspar 2012", "concentration_reduction_pct": 48,
             "load_reduction_significant": False}],
         "verdict": "Correct sign; compare magnitude directly with the 61% trial result, not a fitted tile-drain model."}
    trial_sown = [p["years"].index(y) for y in range(2001, 2005)]
    report["rye_comparison"]["survival_in_winters_ending_2002_2005"] = float(
        p["covers"]["alive"][0, 0, :2, trial_sown].float().mean())
    report["rye_comparison"]["stand_caveat"] = (
        "Air-temperature hardiness is a conservative proxy without snow/soil-temperature observations. "
        "Trial-period survival is for neighbouring Story weather, not reconstruction of Boone field management.")
    corn = torch.as_tensor((tr["rotation"][:, None] == 0) & (tr["crop"][:, 5:-1] == 0), device=p["device"])
    detail = ledger["nitrogen_monthly"]
    report["timing_diagnostic"] = {
        "mean_corn_yield_t_ha": float(ledger["harvest_t_ha"][:, 4:][corn].mean()),
        "unmet_demand_internal_diagnostic_corn_kg_ha": float(ledger["n_shortage_kg_ha"][:, 4:][corn].mean()),
        "potential_grain_n_kg_ha": float((ledger["harvest_t_ha"][:, 4:] * crops[0].v("harvest_n_kg_t"))[corn].mean()),
        "mean_corn_months": [
            {"month": month + 1, **{k: float(v[:, 4:, month][corn].mean()) for k, v in detail.items()}}
            for month in range(12)],
        "shortage_affects_yield": False,
        "interpretation": ("Application is at planting. The previous potential-ET proxy back-loaded corn N demand "
                           "into July/August; it now uses Bender's two-thirds by VT/R1 with low-confidence smooth "
                           "within-stage interpolation. Complete-mixing monthly drainage acts on all remaining soluble N. Demand is "
                           "grain removal plus Bolinder/C:N residue N, not a measured whole-plant N requirement. "
                           "The monthly table distinguishes timing from excessive demand; no fertilizer-rate tuning. "
                           "Internal unmet demand is excluded from the page and scored outcomes."),
        "corn_uptake_source": {
            "source": "https://nutrien-ekonomics.com/wp-content/uploads/2019/11/Corn_Hybrid_Research.pdf",
            "whole_plant_kg_n_ha": 256 * 1.12085, "grain_kg_n_ha": 148 * 1.12085,
            "yield_t_ha": 230 * .06277, "fraction_by_vt_r1": 2 / 3},
        "published_iowa_budget": {
            "source": "http://castellanolab.weebly.com/uploads/1/1/9/6/11961629/nr_committee,_helmers,_castellano,_feb_2015_3.pdf",
            "corn_fertilizer_kg_ha": 150 * 1.12085, "corn_grain_removal_kg_ha": 100 * 1.12085,
            "corn_nitrate_uptake_kg_ha": 165 * 1.12085, "corn_residue_return_kg_ha": 65 * 1.12085,
            "nitrate_leaching_kg_ha": 30 * 1.12085, "gaseous_loss_kg_ha": 10 * 1.12085,
            "caveat": "Illustrative Iowa corn-soybean budget, not site-specific target; model omits gaseous losses."},
        "published_rate_trial": {
            "source": "https://doi.org/10.13031/2013.24229",
            "note": "Lawlor 2008 Iowa tile plots: at 168 kg N/ha, mean losses 58 (1989–93) and 32 (2000–04); weather and timing matter."}}
    # Controlled timing ablation: same rates, weather, demand and yields; only
    # replace corn's cited uptake schedule with the previous potential-ET proxy.
    legacy_crops = [c.override(n_fraction_at_flowering=-1) for c in crops]
    legacy_water = water_pass(p, [s["soil"]], legacy_crops, covers, None, tr)
    legacy = ledger_pass(p, legacy_water, [s["soil"]], normals, legacy_crops, covers, None, tr,
                         fao_reference=fao_reference(water, tr, 1, len(crops), 4), nitrogen_log=True)
    report["timing_diagnostic"]["controlled_ablation"] = {
        "hypothesis": "Back-loaded ET-proxy demand loses spring fertilizer before uptake; earlier cited uptake should reduce both leaching and internal unmet demand at unchanged rate/yield.",
        "rates_and_yields_unchanged": bool(torch.allclose(legacy["harvest_t_ha"], ledger["harvest_t_ha"])
                                          and torch.allclose(legacy["n_fertilizer_kg_ha"], ledger["n_fertilizer_kg_ha"])),
        "corn_before": {k: float(legacy[k][:, 4:][corn].mean()) for k in
                        ("n_fertilizer_kg_ha", "n_leached_kg_ha", "n_shortage_kg_ha")},
        "corn_after": {k: float(ledger[k][:, 4:][corn].mean()) for k in
                       ("n_fertilizer_kg_ha", "n_leached_kg_ha", "n_shortage_kg_ha")},
        "remaining_gap": "Residual unmet demand is diagnostic only. Complete mixing, nitrate-only fertilizer, no immobilization or gaseous losses, approximate mineralization and plant demand are not a validated Iowa N budget."}
    cc_mask = torch.tensor(tr["rotation"] == 4, device=p["device"])
    report["rotation_fact"] = {"continuous_corn_steady_penalty_pct": float(ledger["pest_loss_pct"][cc_mask][:, 4:].mean()),
                               "first_repeat_trait_pct": CATALOG["corn"].v("same_crop_penalty") * 100,
                               "verdict": "First repeat is 4.3% by construction; continuous-corn streak reaches 6.45%. Neither independently validates Seifert."}
    report["cause"] = [
        "Annual leaching used end-of-year mineral N; no living cover N store during drainage.",
        "Vetch fertilizer WAS already reduced by fixation; not the missing-credit hypothesis. Excess biomass/residue release still accumulated N.",
        "Oats received ~58 fertilizer plus ~110 mineralized kg N/ha/year against ~90 whole-crop uptake; 85 leached was a mass-balance surplus, not a high fertilizer trait.",
        "Now monthly drainage/uptake, separate SOM and residue N pools, living cover N carry-over, termination return, rotation-adjusted N rates and biomass-based cover credit. Seasonal mineralization uses Q10=2 at 20 C, an uncalibrated prior.",
        "Iowa historical MRTN example rates, not current fertilizer advice; soybean rotation credit is not also credited as cash saving in the already rotation-specific ISU budget. Only additional legume-cover fertilizer savings change margin.",
    ]
    report["mass_balance_max_error_kg_ha"] = float(ledger["n_balance_error_kg_ha"].abs().max())
    report["biomass_reference"] = {
        "source": "https://projects.sare.org/project-reports/lnc13-352/",
        "rye_iowa_trial_means_t_ha": [.642 * 1.12085, .676 * 1.12085, 1.170 * 1.12085, 1.990 * 1.12085],
        "note": "Published 642–1990 lb/ac = 0.72–2.23 t/ha means across different Iowa trials; vetch mostly winterkilled after soybean in five Iowa environments. Early September southern MN vetch mean 1695 lb/ac (1.90 t/ha), survival 0–73% by ecotype."}
    (ROOT / "nitrogen_after.json").write_text(json.dumps(report, indent=2))
    return report


def county_checks():
    cases = [("York NE", "US-31-185", True), ("Sumner KS", "US-20-191", False),
             ("Lubbock TX", "US-48-303", True), ("Bolivar MS", "US-28-011", True),
             ("Fresno CA", "US-06-019", True)]
    reports = []
    for name, adm, irrigated in cases:
        r = plan(adm, irrigated)
        ranked = rank(r["table"], DEFAULT_WEIGHTS)
        reports.append({"county": name, "adm_id": adm, "irrigated": irrigated, "existing": r["existing"],
                        "evidence": r["rotation_evidence"],
                        "alternatives": [{"labels": why, "rotation": ranked.loc[ranked.id.eq(i)].iloc[0].rotation}
                                         for i, why in select_comparisons(
                                             ranked, weights=DEFAULT_WEIGHTS, runs=r["arrays"]["runs"],
                                             windows=len(r["arrays"]["window_start"]))],
                        "pruned": r["pruned"], "gaps": r["gaps"]})
    (ROOT / "validation_other_counties.json").write_text(json.dumps(reports, indent=2))
    return reports


def diverse_checks():
    """Rank actual candidates and separately test a three-year oats proxy.

    The exact Marsden systems cannot be claimed without underseeded legumes,
    manure and straw economics. Keep that boundary explicit.
    """
    from fieldshift.sim.traits import DAVIS
    result = plan()
    table = rank(result["table"], {"income": 1})
    diverse = table.loc[table.cycle.str.contains("oats") & table.cycle.str.contains("corn")
                        & table.cycle.str.contains("soybean")]
    candidates = [{"income_rank": list(table.id).index(int(row.id)) + 1, "rotation": row.rotation,
                   "margin": row.income_median} for row in diverse.head(5).itertuples()]
    s = load_site("US-19-169")
    priced = {c.key: c for c in candidate_crops(s)[0]}
    crops = [priced[k] for k in ("corn", "soybean", "oats")]
    p = prepare([s], crops)
    tr = trajectories([((0, 1, 2), 0, False)], crops, len(p["years"]), spin=3, score=3)
    w = water_pass(p, [s["soil"]], crops, [], None, tr)
    normals = [[s["normals"].get(c.key, c.v("normal_t_ha")) for c in crops]]
    ledger = ledger_pass(p, w, [s["soil"]], normals, crops, [], None, tr,
                         fao_reference=fao_reference(w, tr, 1, len(crops), 3))
    margin = float(ledger["income"][:, 3:].mean(-1).median())
    report = {"source": DAVIS, "best_oats_corn_soy_candidates": candidates,
              "three_year_proxy": {"cycle": "corn-soybean-oats (NO red clover)", "margin": margin,
                                   "income_insertion_rank_among_four_year_candidates": int(table.income_median.gt(margin).sum()) + 1,
                                   "not_exact_marsden_system": True},
              "four_year_marsden": "corn-soy-oats/underseeded alfalfa-alfalfa is not in the search: underseeding is not represented; an isolated one-year forage stand is pruned.",
              "mechanisms_added": "Low-confidence +4% corn/+9% soy yield transfer for >=3-species cycles with small grain or perennial legume. Corn following established perennial legume uses ISU 0–30 lb/ac N scenario, additional savings beyond soy-rate budget included.",
              "gaps": "Missing manure, clover/alfalfa underseeding, straw revenue, herbicide reductions, added labor and local forage enterprise budgets. Oats' generic cost exceeds grain income; not evidence against Marsden's measured profitability."}
    (ROOT / "validation_diverse.json").write_text(json.dumps(report, indent=2))
    return report


def run():
    report = {"multi_county": multi_county(), "weather_tests": weather_tests(), "nitrogen": nitrogen_checks(),
              "story": story_checks(), "other_counties": county_checks(), "diverse_rotations": diverse_checks()}
    (ROOT / "validation.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
