"""Benchmark D: physical diagnosis of the 1983 corn miss (diagnosis only, no fit).

The v22 300-county check (planner.validate.multi_county) simulates a 4.0% mean
corn loss in 1983 against 23.8% observed, while 1988 and 2012 are close. This
module re-runs exactly that set and records, per county-year, the quantities
that the candidate explanations depend on:

- flowering timing: simulated flowering window vs USDA NASS state 50% silking;
- heat at flowering: simulated heat term (threshold = corn heat_critical_c) vs
  daily Tmax degree days above 30/35 C in the observed silking window;
- water: stage ETa/ETm, root-zone water at simulated flowering, June-August
  rain and ET0;
- rooting depth: one variant with the FAO-56 Table 22 lower maize rooting depth.

No parameter is fitted here. Variants are labelled sensitivity scenarios.
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from fieldshift.planner.data import _progress, load_light
from fieldshift.planner.validate import (
    _combined_trajectories,
    _relative_observed,
    correlation,
    json_safe,
    selected_labels,
)
from fieldshift.sim.components import accelerator
from fieldshift.sim.engine import MONTH_OF_DAY, prepare, water_pass
from fieldshift.sim.state import ROOT
from fieldshift.sim.traits import CATALOG

YEARS = (1983, 1988, 2012)
SILK_WINDOW = 7  # days either side of NASS 50% silking (state level)
FAO56_MAIZE_LOWER_ZR = 1.0  # FAO-56 Table 22, maize (grain) Zr 1.0-1.7 m


def _silking():
    p = _progress()
    p = p.loc[p.commodity.eq("CORN") & p.stage.eq("silking") & p.class_desc.eq("ALL CLASSES") & p.date_50.notna()]
    p = p.drop_duplicates(["state", "year"])
    return {(r.state, int(r.year)): pd.Timestamp(r.date_50).day_of_year - 1 for r in p.itertuples()}


def _normalise(sim, even):
    return sim / torch.where(even, sim, 0).sum(-1, keepdim=True).div(even.sum()).clamp_min(.01)


def _window_sum(x, start, end):
    """x [S,Y,366]; start/end [S,Y] inclusive day indices (-1 = missing)."""
    d = torch.arange(366, device=x.device)
    m = (d >= start[..., None]) & (d <= end[..., None]) & (start[..., None] >= 0)
    return torch.where(m, x, 0).sum(-1), m.sum(-1)


def batch(chosen, labels, silk, device):
    sites = [load_light(c) for c in chosen]
    crops = [CATALOG["corn"], CATALOG["soybean"]]
    p = prepare(sites, crops, learned=False)
    years = np.asarray(p["years"])
    tr = _combined_trajectories(crops, len(years), len(sites))
    soils = [s["soil"] for s in sites]
    w = water_pass(p, soils, crops, [], None, tr, daily_log=True)
    shallow = [crops[0].override(bucket_root_depth_m=FAO56_MAIZE_LOWER_ZR), crops[1]]
    ws = water_pass(p, soils, shallow, [], None, tr)
    n_s = len(sites)
    corn = torch.arange(n_s, device=device) * 2
    relative, mask = _relative_observed(labels, chosen, years, device)
    relative, mask = relative[:, 0], mask[:, 0]
    prev = torch.as_tensor(np.asarray(p["prev_len"]), device=device)[None]
    tl = p["timeline"]
    fs = torch.as_tensor(tl["flower_start"][:, 0], device=device) - prev
    fe = torch.as_tensor(tl["flower_end"][:, 0], device=device) - prev
    fs = torch.where(torch.as_tensor(tl["flower_start"][:, 0] >= 0, device=device), fs, -1)
    fe = torch.where(fs >= 0, fe, -1)
    silk_doy = torch.tensor([[silk.get((s["state"], int(y)), -1) for y in years] for s in sites], device=device)
    weather = p["weather"]  # [S, Y, 366, (prec, et0, tmin, tmax, tavg)], harvest-year calendar
    tmax = weather[..., 3]
    out = {"years": years}
    out["sim"] = w["fao_main"][corn]
    out["shallow"] = ws["fao_main"][corn]
    heat_sim = p["stage_heat"][:, 0]
    out["water_only"] = out["sim"] / torch.exp(-.02 * heat_sim - .05 * p["stage_frost"][:, 0]).clamp_min(1e-8)
    obs_start, obs_end = (silk_doy - SILK_WINDOW).clamp_min(-1), torch.where(silk_doy >= 0, silk_doy + SILK_WINDOW, -1)
    for thr in (30, 35):
        dd = (tmax - thr).clamp_min(0)
        out[f"dd{thr}_observed_silking"], _ = _window_sum(dd, obs_start, obs_end)
        out[f"dd{thr}_simulated_flowering"], _ = _window_sum(dd, fs, fe)
    out["tmax_observed_silking"] = _window_sum(tmax, obs_start, obs_end)[0] / (2 * SILK_WINDOW + 1)
    days = _window_sum(torch.ones_like(tmax), fs, fe)[1].clamp_min(1)
    out["tmax_simulated_flowering"] = _window_sum(tmax, fs, fe)[0] / days
    out["heat_term_simulated"] = heat_sim
    out["flower_start_doy"] = fs.float() + 1
    out["silking_doy"] = torch.where(silk_doy >= 0, silk_doy.float() + 1, float("nan"))
    ratio = w["stage_actual_et"][corn] / w["stage_potential_et"][corn].clamp_min(1e-6)
    for k, name in enumerate(("establishment", "vegetative", "flowering", "grain_fill")):
        out[f"eta_etm_{name}"] = ratio[..., k]
    capacity = torch.tensor([s["awc_mm_per_m"] * s["profile_depth_m"] for s in soils], device=device)
    dw = w["daily_water"][corn]
    at = fs.clamp(0, 365)
    out["water_at_flowering_fraction"] = torch.where(
        fs >= 0, dw.gather(-1, at[..., None])[..., 0] / capacity[:, None], float("nan"))
    month = torch.as_tensor(MONTH_OF_DAY, device=device)
    for m, name in ((4, "may"), (5, "jun"), (6, "jul"), (7, "aug")):
        sel = (month == m).float()
        out[f"rain_{name}_mm"] = (weather[..., 0] * sel).sum(-1)
        out[f"et0_{name}_mm"] = (weather[..., 1] * sel).sum(-1)
    return out, relative, mask


def run(limit=300, batch_size=20):
    started = time.perf_counter()
    device = accelerator()
    labels, ids = selected_labels(limit)
    silk = _silking()
    parts = []
    for i in range(0, len(ids), batch_size):
        parts.append(batch(ids[i:i + batch_size], labels, silk, device))
        print(f"D batch {i + len(ids[i:i + batch_size])}/{len(ids)}", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
    years = parts[0][0]["years"]
    cat = lambda k: torch.cat([q[0][k] for q in parts])
    relative = torch.cat([q[1] for q in parts])
    mask = torch.cat([q[2] for q in parts])
    keep = mask.sum(-1) >= 25
    even = torch.tensor(years % 2 == 0, device=device)
    report = {"counties": int(keep.sum()), "variants": {}, "diagnostics": {}}
    for name in ("sim", "water_only", "shallow"):
        sim = _normalise(cat(name), even)
        odd = correlation(sim[keep], relative[keep], (mask & ~even)[keep])
        v = {"median_r_odd": float(odd[torch.isfinite(odd)].median()), "loss_pct": {}}
        for y in YEARS:
            j = int(np.where(years == y)[0][0])
            m = keep & mask[:, j]
            v["loss_pct"][str(y)] = {"counties": int(m.sum()),
                                     "observed": float(100 * (1 - relative[m, j]).mean()),
                                     "simulated": float(100 * (1 - sim[m, j]).mean())}
        report["variants"][name] = v
    names = [k for k in parts[0][0] if k not in ("years", "sim", "shallow", "water_only")]
    for y in YEARS + ("median_all_years",):
        rows = {}
        if y == "median_all_years":
            m = keep[:, None] & mask
            for k in names:
                x = cat(k)
                vals = x[m & torch.isfinite(x)]
                rows[k] = float(vals.median()) if len(vals) else None
        else:
            j = int(np.where(years == y)[0][0])
            m = keep & mask[:, j]
            for k in names:
                x = cat(k)[:, j]
                vals = x[m & torch.isfinite(x)]
                rows[k] = float(vals.mean()) if len(vals) else None
            rows["counties"] = int(m.sum())
        report["diagnostics"][str(y)] = rows
    report["method"] = {
        "selection": "Same 300-county set and scoring as planner.validate.multi_county (state round-robin, rainfed or proxy, >=25 years, even-year trend and FAO normalisation, odd-year r).",
        "diagnostics": "Year rows are means over counties with an observation that year; median_all_years is the median over all county-years. DOY are 1-based. Silking = USDA NASS state 50% silking (crop progress), window +-7 days.",
        "variants": {"sim": "v22/v25 default (season-level FAO-33, flowering heat term above corn heat_critical_c 40 C, slope .02)",
                     "water_only": "default without the heat/frost multipliers",
                     "shallow": "corn bucket_root_depth_m 1.0 m (FAO-56 Table 22 lower maize Zr), sensitivity only"},
        "tuning": "None.",
    }
    report["seconds"] = time.perf_counter() - started
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "d_1983_diagnosis.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    out = run()
    print(json.dumps(json_safe(out), indent=1)[:6000])
