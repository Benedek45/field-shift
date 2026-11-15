"""Bounded search over rotation cycles, never hidden.

A candidate is a cycle of four main crops (one representative per set of cyclic
shifts), a per-cycle winter option (none or a cover crop wherever one fits) and,
where an autumn-sown crop is in the cycle and the season allows, double-crop
soybean after it. Every candidate is scored in steady state over all historical
weather windows and cycle phases.
"""

import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.features.traits import Trait
from fieldshift.planner.data import ROOT, load_site
from fieldshift.sim.engine import (
    LEDGER_COLUMNS,
    cover_eligible,
    fao_reference,
    ledger_pass,
    prepare,
    trajectories,
    water_pass,
)
from fieldshift.sim.traits import CATALOG, COVER_RECORDS, Crop

VERSION = "fieldshift-v26"
CYCLE = 4
MAX_CROPS = 7
DEFAULT_WEIGHTS = {"income": 5, "water": 2, "soil": 3, "nitrogen": 1, "nitrate": 1, "erosion": 2,
                   "cover": 1, "forage": 0, "resilience": 3, "market": 2}
METRICS = tuple(DEFAULT_WEIGHTS)
GOALS = (
    ("Best match to your priorities", None),
    ("Existing crops plus a winter cover", "cover_variant"),
    ("Highest margin", {"income": 1}),
    ("Most soil carbon input", {"soil": 1}),
    ("Lowest water use", {"water": 1}),
    ("Highest bad-year margin", {"resilience": 1}),
)
COVER_KEYS = tuple(COVER_RECORDS)
SUMMARY = ("income", "return_over_listed_costs_usd_ha", "yield_relative", "et_mm", "et_total_mm", "irrigation_mm", "water_at_planting_mm",
           "mineral_n_kg_ha", "organic_matter_pct", "soil_carbon_input_t_ha", "erosion_t_ha", "bare_days",
           "forage", "n_fixed_kg_ha", "n_leached_kg_ha", "drainage_mm", "pest_loss_pct",
           "predictive_sd", "n_credit_kg_ha", "cover_biomass_t_ha", "cover_established", "cover_cost_usd_ha",
           "double_crop_t_ha", "n_fertilizer_kg_ha", "n_fertilizer_saving_usd_ha", "n_cover_uptake_kg_ha")
# Per-run scored-cycle means kept for the near-tie test (columns behind METRICS).
RUN_COLUMNS = ("income", "et_total_mm", "soil_carbon_input_t_ha", "mineral_n_kg_ha", "n_credit_kg_ha",
               "n_leached_kg_ha", "erosion_t_ha", "bare_days", "forage")
TIE_DRAWS = 2000


def _priced(crop, site):
    if crop.key in site.get("prices", {}):
        price = site["prices"][crop.key]
        return Crop(crop.key, crop.traits | {"price_usd_t": Trait(price["value"], price["source"], price["confidence"])})
    return crop


def candidate_crops(site, no_equipment=False):
    """Existing and locally grown crops first, then a few agronomic options."""
    existing = list(dict.fromkeys(site["existing"]))
    locally = sorted(site["areas"], key=site["areas"].get, reverse=True)
    keys = existing if no_equipment else list(dict.fromkeys(existing + locally + ["oats", "alfalfa",
                                                                                  "winter_wheat", "sorghum"]))
    chosen, pruned = [], []
    for key in keys:
        crop = CATALOG[key]
        if crop.v("harvest_storage_root") or crop.v("harvest_fiber") or crop.v("requires_flooding"):
            if key in existing:
                chosen.append(key)
            else:
                pruned.append(f"{key}: specialist equipment or market not confirmed")
        elif len(chosen) < MAX_CROPS or key in existing:
            chosen.append(key)
        else:
            pruned.append(f"{key}: the search keeps at most {MAX_CROPS} main crops")
    if len(chosen) < 2:
        chosen += [k for k in (("fallow",) if no_equipment else ("hay", "fallow")) if k not in chosen][:2 - len(chosen)]
    stated_covers = {s.get("winter") for s in site.get("existing_steps", [])}
    covers = [CATALOG[k] for k in COVER_KEYS if not no_equipment or k in stated_covers]
    if no_equipment:
        pruned.append("winter covers: excluded by 'no new crop equipment' (seeding and termination passes)")
    has_double = any(s.get("double_crop") for s in site.get("existing_steps", []))
    double = None if no_equipment and not has_double else _priced(Crop("double_crop_soybean", CATALOG["double_crop_soybean"].traits), site)
    if double is not None and "soybean" in site.get("prices", {}):
        double = Crop(double.key, double.traits | {"price_usd_t": _priced(CATALOG["soybean"], site).traits["price_usd_t"]})
    return [_priced(CATALOG[k], site) for k in chosen], covers, double, pruned


def canonical(cycle):
    return min(tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle)))


def enumerate_rotations(crops, covers, prepared, retained, existing, pruned):
    """One representative per set of cyclic shifts, then winter options."""
    autumn = np.array([bool(c.v("autumn_sown")) for c in crops])
    perennial = np.array([bool(c.v("perennial")) for c in crops])
    eligible = cover_eligible(crops)
    sow = prepared["timeline"]["sow"][0]
    operation = prepared["operation"][0].cpu().numpy()
    prev_len = prepared["prev_len"]
    # An autumn-sown crop needs the previous crop harvested before its sowing date.
    follows = np.ones((len(crops), len(crops)), dtype=bool)
    for b in np.where(autumn)[0]:
        sow_doy = sow[b, 1:]  # offsets inside the previous calendar year
        for a in range(len(crops)):
            harvest_prev = operation[a, :-1]
            ok = (harvest_prev >= 0) & (harvest_prev < sow_doy) & (sow_doy < prev_len[1:])
            follows[a, b] = ok.mean() >= .6 if not crops[a].v("fallow") else True
            if not follows[a, b]:
                pruned.append(f"{crops[a].key} → {crops[b].key}: {crops[a].key} is harvested after the "
                              f"{crops[b].key} sowing date in {1 - ok.mean():.0%} of years")
    cycles = sorted({canonical(list(c)) for c in product(retained, repeat=CYCLE)})
    cycles = [existing] + [c for c in cycles if c != existing]
    double_ok = prepared["double"]["maturity"][0].cpu().numpy() >= .6 if prepared["double"] is not None else None
    rotations, short_perennial, bad_transition = [], 0, 0
    for cycle in cycles:
        pairs = list(zip(cycle, cycle[1:] + cycle[:1], strict=True))
        if cycle != existing and not all(follows[a, b] for a, b in pairs):
            bad_transition += 1
            continue
        runs = [i for i, c in enumerate(cycle) if perennial[c] and cycle[i - 1] != c
                and cycle[(i + 1) % len(cycle)] != c]
        if cycle != existing and runs and len(set(cycle)) > 1:
            short_perennial += 1
            continue
        rotations.append((cycle, 0, False))
        if covers and any(eligible[a, b] for a, b in pairs):
            for k in range(len(covers)):
                alive = prepared["covers"]["alive"][0, k].float().mean(-1).cpu().numpy()
                shares = [alive[a] for a, b in pairs if eligible[a, b]]
                # Keep marginal options visible with a quantified failure rate;
                # near-total failures are pruned, never recommended as reliable.
                if shares and np.mean(shares) >= .2:
                    rotations.append((cycle, k + 1, False))
                    if cycle == existing and np.mean(shares) < .6:
                        pruned.append(f"{covers[k].key}: retained as a HIGH-RISK option, "
                                      f"{1 - np.mean(shares):.0%} establishment/winter failure; "
                                      "cost is paid even in failed years")
                elif cycle == existing:
                    pruned.append(f"{covers[k].key}: establishment/winter survival fails in "
                                  f"{1 - np.mean(shares):.0%} of eligible existing-rotation years; "
                                  "requires earlier seeding or locally proven cultivar/snow protection")
        if double_ok is not None and any(autumn[c] and double_ok[c] for c in cycle):
            rotations.append((cycle, 0, True))
    if bad_transition:
        pruned.append(f"{bad_transition} cycles: an autumn-sown crop would follow a crop still in the field")
    if short_perennial:
        pruned.append(f"{short_perennial} cycles: a perennial forage kept for only one year (stands need 2+ years)")
    if double_ok is not None and not any(autumn[c] and double_ok[c] for c in range(len(crops))):
        pruned.append("double-crop soybean after wheat: matures before the first autumn frost in fewer than 60% "
                      "of years here")
    return rotations


def label(rotation, crops, covers):
    cycle, option, double = rotation
    text = " → ".join(crops[c].key.replace("_", " ") for c in cycle)
    if option:
        text += " + winter " + covers[option - 1].key.replace("_cover", "").replace("_", " ") + " cover"
    if double:
        text += " + double-crop soybean after wheat"
    return text


def _cache_paths(adm_id, irrigated, rain, temp, no_equipment, must_forage):
    sources = [Path("data/processed/models/yield_response_v2_power.pt"),
               Path("data/processed/static/county_static.parquet"),
               Path("data/processed/eo/eo_county_8day.json"), ROOT / "nass_reference_prices.json",
               Path("data/processed/rma_cause_of_loss.parquet"),
               Path("data/processed/nass_crop_progress_dates.parquet")]
    # Simulator source text is part of the key, so code changes never serve stale plans.
    code = Path(__file__).parents[1]
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(
        list((code / "sim").glob("*.py")) + [code / "planner" / "search.py", code / "planner" / "data.py"]))).hexdigest()
    token = json.dumps([VERSION, adm_id, irrigated, rain, temp, no_equipment, must_forage, fingerprint,
                        [(str(p), p.stat().st_mtime_ns if p.exists() else None) for p in sources]])
    stem = ROOT / (adm_id + "_" + hashlib.sha256(token.encode()).hexdigest()[:14])
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def plan(adm_id="US-19-169", irrigated=False, rain_change=0., temp_change=0.,
         no_equipment=False, must_forage=False, progress=None, force=False):
    """Simulate (or load) every candidate rotation for a county and scenario."""
    meta_path, array_path = _cache_paths(adm_id, irrigated, rain_change, temp_change, no_equipment, must_forage)
    if meta_path.exists() and array_path.exists() and not force:
        result = json.loads(meta_path.read_text())
        with np.load(array_path) as arrays:
            result["arrays"] = {k: arrays[k] for k in arrays.files}
        return result
    started = time.perf_counter()
    edited = bool(rain_change or temp_change)
    site = load_site(adm_id, irrigated)
    baseline = None
    if edited:
        # Same candidates and trajectories as the historical-weather plan.
        baseline = plan(adm_id, irrigated, 0., 0., no_equipment, must_forage, progress)
    crops, covers, double, pruned = candidate_crops(site, no_equipment)
    if progress:
        progress(.02, "Preparing phenology, cover windows and the saved yield model on the accelerator")
    prepared = prepare([site], crops, covers, double, rain_change, temp_change, irrigated)
    maturity = prepared["maturity"][0].tolist()
    existing_keys = list(dict.fromkeys(site["existing"]))
    for key in existing_keys:
        m = maturity[[c.key for c in crops].index(key)]
        if m < .6:
            site["gaps"].append(f"{key} (existing crop) reaches its trait maturity GDD in only {m:.0%} of years: "
                                "the generic cultivar heat requirement is likely too long for local varieties.")
    if baseline is None:
        retained = [i for i, c in enumerate(crops) if maturity[i] >= .6 or c.key in existing_keys]
        for i, crop in enumerate(crops):
            if i not in retained:
                pruned.append(f"{crop.key}: reaches maturity in only {maturity[i]:.0%} of weather years (<60%)")
        keys = [c.key for c in crops]
        existing = canonical([keys.index(k) for k in site["existing"]])
        rotations = enumerate_rotations(crops, covers, prepared, retained, existing, pruned)
    else:
        rotations = [(tuple(r[0]), int(r[1]), bool(r[2])) for r in baseline["rotations"]]
        pruned = baseline["pruned"]
    baseline_warnings = []
    existing_cycle = rotations[0][0]
    for a, b in zip(existing_cycle, existing_cycle[1:] + existing_cycle[:1], strict=True):
        if crops[b].v("autumn_sown"):
            sow = prepared["timeline"]["sow"][0, b, 1:]
            done = prepared["operation"][0, a, :-1].cpu().numpy()
            overlap = float(np.mean(done >= sow))
            if overlap > .4:
                baseline_warnings.append(f"Baseline calendar conflict: {crops[a].key} → {crops[b].key} overlaps "
                                         f"in {overlap:.0%} of years. The NASS-area inferred order is not a proven "
                                         "feasible field history; confirm harvest/planting dates before using its margin.")
    site["gaps"].extend(baseline_warnings)
    site["gaps"].append("Monthly nitrogen uses separate living-cover, residue and soil stores; Q10=2 and 2%/year "
                        "mineralisation at 20 C are uncalibrated process priors. No field soil-N observations or "
                        "tile-drain calibration; shortages do not reduce learned yields.")
    traj = trajectories(rotations, crops, len(prepared["years"]))
    soils = [site["soil"]]
    normals = np.array([[site["normals"].get(c.key, c.v("normal_t_ha")) for c in crops]], dtype=np.float32)
    if progress:
        progress(.1, f"Daily water balance: {len(rotations)} rotations × {len(traj['rotation']) // len(rotations)} "
                 "weather windows and phases")
    water = water_pass(prepared, soils, crops, covers, double, traj, irrigated)
    unlimited = water
    if baseline is None and "irrigation_capacity_mm_yr" in site["soil"]:
        unlimited_soils = [dict(site["soil"])]
        unlimited_soils[0].pop("irrigation_capacity_mm_yr")
        unlimited = water_pass(prepared, unlimited_soils, crops, covers, double, traj, irrigated)
    if baseline is None:
        reference = fao_reference(unlimited, traj, 1, len(crops), traj["spin"])
        base_fao = {"fao_main": unlimited["fao_main"]} if unlimited is not water else None
    else:
        reference = torch.as_tensor(baseline["arrays"]["fao_reference"], device=prepared["device"])
        base_fao = {"fao_main": torch.as_tensor(baseline["arrays"].get("fao_unlimited", baseline["arrays"]["fao_main"]), device=prepared["device"])}
        prepared["learned"] = (torch.as_tensor(baseline["arrays"]["learned"], device=prepared["device"])
                               if baseline["arrays"]["learned"].size else None)
    ledger = ledger_pass(prepared, water, soils, normals, crops, covers, double, traj, irrigated,
                         fao_reference=reference, base_fao=base_fao)
    table = summarize(ledger, traj, rotations, crops, covers, site, must_forage)
    sensitivity = []
    if baseline is None:
        if progress:
            progress(.8, "Checking which low-confidence traits change the top three")
        sensitivity = trait_sensitivity(prepared, water, soils, normals, crops, covers, double, traj,
                                        rotations, site, table, irrigated, reference, must_forage)
    keys = [c.key for c in crops]
    tl = prepared["timeline"]
    scored = slice(traj["spin"], None)
    first_phase = _first_phase(traj, len(rotations))
    per = len(traj["rotation"]) // len(rotations)
    arrays = {
        "runs": _to_numpy(torch.stack([ledger[k][:, traj["spin"]:].nanmean(-1) for k in RUN_COLUMNS], -1)
                          .reshape(len(rotations), per, len(RUN_COLUMNS))).astype(np.float32),
        "ledger": np.stack([_to_numpy(ledger[k][first_phase][:, scored]) for k in LEDGER_COLUMNS], -1).astype(np.float32),
        "water_month": _to_numpy(water["water_month"][first_phase][:, scored]).astype(np.float32),
        "et_month": _to_numpy(water["et_month"][first_phase][:, scored]).astype(np.float32),
        "fao_main": _to_numpy(water["fao_main"]).astype(np.float32),
        "fao_unlimited": _to_numpy(unlimited["fao_main"] if baseline is None else baseline["arrays"]["fao_unlimited"]).astype(np.float32),
        "fao_reference": _to_numpy(reference).astype(np.float32),
        "learned": (_to_numpy(prepared["learned"]).astype(np.float32) if prepared["learned"] is not None
                    else np.zeros(0, np.float32)),
        "window_start": np.asarray(traj["starts"], dtype=np.int32),
        "sow": tl["sow"][0], "end": tl["end"][0], "flower_start": tl["flower_start"][0],
        "flower_end": tl["flower_end"][0], "gdd_scale": tl["gdd_scale"][0],
        "operation": _to_numpy(prepared["operation"][0]).astype(np.int32),
        "plant": _to_numpy(prepared["plant"][0]).astype(np.int32),
        "harvest": _to_numpy(prepared["harvest"][0]).astype(np.int32),
        "prev_len": np.asarray(prepared["prev_len"], dtype=np.int32),
        "cover_alive": (_to_numpy(prepared["covers"]["alive"][0]) if prepared["covers"] is not None
                        else np.zeros((0,), bool)),
    }
    result = {
        "version": VERSION, "adm_id": adm_id, "label": site["label"], "soil": site["soil"],
        "existing": list(site["existing"]), "rotation_evidence": site["rotation_evidence"],
        "baseline_warnings": baseline_warnings,
        "loss_evidence": site["loss_evidence"], "loss_months": site["loss_months"].to_dict("records"),
        "planting_sources": {k: v["source"] for k, v in site["planting"].items()},
        "harvest_sources": {k: v["harvested"]["source"] for k, v in site["planting"].items() if "harvested" in v},
        "yield_path": prepared["backend"].path, "yield_note": prepared["backend"].reason,
        "rain_change": rain_change, "temp_change": temp_change, "irrigated": irrigated,
        "crops": keys, "covers": [c.key for c in covers], "double_crop": double.key if double else None,
        "rotations": [[list(map(int, r[0])), int(r[1]), bool(r[2])] for r in rotations],
        "crop_traits": {c.key: {k: {"value": t.value, "source": t.source, "confidence": t.confidence}
                                for k, t in c.traits.items()} for c in crops + covers},
        "table": table, "sensitivity": sensitivity, "sensitivity_gaps": _sensitivity_text(sensitivity),
        "years": [int(y) for y in prepared["years"]], "ledger_columns": list(LEDGER_COLUMNS),
        "pruned": pruned, "maturity": dict(zip(keys, maturity, strict=True)), "gaps": site["gaps"],
        "normal_yields_t_ha": {c.key: float(normals[0, i]) for i, c in enumerate(crops)},
        "normal_sources": {c.key: site["normal_sources"].get(c.key, "trait reference yield (no NASS series)")
                           for c in crops},
        "normal_yield_gaps": [c.key for c in crops if c.key not in site["normals"]],
        "trajectories_per_rotation": int(len(traj["rotation"]) // len(rotations)),
        "spin_years": traj["spin"], "water_seconds": water["seconds"], "device": str(prepared["device"]),
        "total_seconds": time.perf_counter() - started,
        "search_note": (f"Exhaustive over {len(rotations)} rotation cycles built from {len(crops)} main crops "
                        f"and {len(covers)} winter covers; not a global optimum."),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path.with_suffix(".part.npz"), **arrays)
    array_path.with_suffix(".part.npz").replace(array_path)
    meta_path.with_suffix(".part").write_text(json.dumps(result, allow_nan=True, separators=(",", ":")))
    meta_path.with_suffix(".part").replace(meta_path)
    result["arrays"] = arrays
    return result


def _first_phase(traj, n_rotations):
    """Trajectory indices of phase 0 (the cycle in its listed order), all windows."""
    per = len(traj["rotation"]) // n_rotations
    windows = len(traj["starts"])
    return np.concatenate([np.arange(r * per, r * per + windows) for r in range(n_rotations)])


LEDGER_TRAITS = (
    "price_usd_t", "cost_usd_ha", "same_crop_extra_cost_usd_ha", "harvest_n_kg_t", "harvest_index",
    "normal_t_ha", "fixed_n_kg_ha", "residue_cn", "shoot_root_ratio", "extra_root_fraction", "same_crop_penalty",
    "penalty_years_to_plateau", "family_penalty", "fertilizer_replacement", "fixed_n_fraction", "cover_yield_effect",
    "diverse_rotation_yield_gain",
)


def _sensitivity_text(sensitivity):
    if not sensitivity:
        return ["Trait sensitivity is screened on historical weather only; see the plan without weather edits."]
    sensitive = sorted({s["trait"] for s in sensitivity if s["changes_top3"]})
    screened = len({s["trait"] for s in sensitivity})
    if not sensitive:
        return [(f"Screened {screened} low-confidence ledger traits at ×0.5 and ×1.5: none changes the top 3 "
                "under default priorities.")]
    return [(f"Screened {screened} low-confidence ledger traits at ×0.5 and ×1.5. These change the top 3 "
            f"under default priorities and need local values: {', '.join(sensitive)}.")]


def _top3(table):
    return list(rank(table, DEFAULT_WEIGHTS).rotation.head(3))


def trait_sensitivity(prepared, water, soils, normals, crops, covers, double, traj, rotations, site, table,
                      irrigated, reference, must_forage=False, factors=(.5, 1.5)):
    """Re-run the ledger pass with each low-confidence ledger trait at ×0.5 and ×1.5.

    A uniform screening range, not per-trait literature ranges. Water-side
    traits (Kc, rooting, phenology) would need the daily pass and are not screened.
    """
    baseline = _top3(table)
    everything = crops + covers
    names = [n for n in LEDGER_TRAITS if any(n in c.traits and c.traits[n].confidence == "low"
                                             and c.traits[n].value for c in everything)]
    if "normal_t_ha" in names and all(c.key in site["normals"] for c in crops):
        names.remove("normal_t_ha")
    results = []
    for name in names:
        for factor in factors:
            def scaled(c, name=name, factor=factor):
                trait = c.traits.get(name)
                if trait is None or trait.confidence != "low":
                    return c
                return c.override(**{name: trait.value * factor})
            changed, changed_covers = [scaled(c) for c in crops], [scaled(c) for c in covers]
            n = np.array(normals)
            if name == "normal_t_ha":
                n = np.array([[v * factor if crops[i].key not in site["normals"] else v
                               for i, v in enumerate(normals[0])]], dtype=np.float32)
            ledger = ledger_pass(prepared, water, soils, n, changed, changed_covers, double, traj, irrigated,
                                 fao_reference=reference)
            after = _top3(summarize(ledger, traj, rotations, changed, changed_covers, site, must_forage))
            results.append({"trait": name, "factor": factor, "changes_top3": set(after) != set(baseline),
                            "top3_after": " | ".join(after)})
    return results


def market_support(key, site):
    """NASS county harvested-area SHARE, not a buyer or a hay-grade guarantee."""
    areas = site.get("areas", {})
    total = sum(max(float(v), 0.) for v in areas.values())
    return max(float(areas.get(key, 0.)), 0.) / total if total else 0.


def summarize(ledger, traj, rotations, crops, covers, site, must_forage=False):
    """Per rotation: median and P10-P90 over windows × phases of the scored-cycle mean."""
    spin = traj["spin"]
    per = len(traj["rotation"]) // len(rotations)
    stacked = torch.stack([ledger[k][:, spin:].nanmean(-1) for k in SUMMARY], -1)
    stacked = stacked.reshape(len(rotations), per, len(SUMMARY))
    quantiles = torch.nanquantile(stacked, torch.tensor([.1, .5, .9], device=stacked.device), dim=1)
    q = _to_numpy(quantiles)
    # Organic matter change over the scored cycle, %-points per decade.
    om = ledger["organic_matter_pct"]
    trend = ((om[:, -1] - om[:, spin - 1]) / (om.shape[1] - spin) * 10).reshape(len(rotations), per)
    trend = _to_numpy(trend.median(1).values)
    table = []
    for i, rotation in enumerate(rotations):
        cycle, option, double = rotation
        row = {"id": i, "rotation": label(rotation, crops, covers), "cycle": " → ".join(crops[c].key for c in cycle),
               "cover_option": covers[option - 1].key if option else "", "double_crop": bool(double),
               "is_existing": i == 0, "feasible": True, "constraint_note": ""}
        low = any(crops[c].key not in site["normals"] or crops[c].key in site.get("mixed_practice_normals", [])
                  or any(crops[c].traits[k].confidence == "low" for k in ("price_usd_t", "cost_usd_ha"))
                  for c in cycle) or (option and covers[option - 1].traits["cost_usd_ha"].confidence == "low")
        row["income_confidence"] = "low" if low else "medium"
        confidence_reasons = []
        for c in set(cycle):
            crop = crops[c]
            for name in ("price_usd_t", "cost_usd_ha"):
                if crop.traits[name].confidence == "low":
                    confidence_reasons.append(f"{crop.key} {name}: {crop.traits[name].source}")
            if crop.key not in site["normals"]:
                confidence_reasons.append(f"{crop.key}: generic yield normal")
            if crop.key in site.get("mixed_practice_normals", []):
                confidence_reasons.append(f"{crop.key}: county yield mixes irrigation practices")
        if option and covers[option - 1].traits["cost_usd_ha"].confidence == "low":
            confidence_reasons.append(f"{covers[option - 1].key}: {covers[option - 1].traits['cost_usd_ha'].source}")
        row["income_confidence_reason"] = "; ".join(confidence_reasons) or "Published reference costs/prices; not a local enterprise budget."
        if must_forage and not any(crops[c].v("forage_value") > .5 for c in cycle):
            row["feasible"], row["constraint_note"] = False, "Must include forage"
        for j, metric in enumerate(SUMMARY):
            row[metric + "_p10"], row[metric + "_median"], row[metric + "_p90"] = (float(q[0, i, j]), float(q[1, i, j]),
                                                                                 float(q[2, i, j]))
        row["om_trend_pct_per_decade"] = float(trend[i])
        row["income"] = row["income_median"]
        row["water"] = -row["et_total_mm_median"]
        row["soil"] = row["soil_carbon_input_t_ha_median"]
        row["nitrogen"] = row["mineral_n_kg_ha_median"] + row["n_credit_kg_ha_median"]
        row["nitrate"] = -row["n_leached_kg_ha_median"]
        row["erosion"] = -row["erosion_t_ha_median"]
        row["cover"] = -row["bare_days_median"]
        row["forage"] = row["forage_median"]
        row["resilience"] = row["income_p10"]
        row["market"] = sum(market_support(crops[c].key, site) for c in cycle) / len(cycle)
        table.append(row)
    return table


def rank(table, weights, max_erosion=None, min_income=None):
    """Pure reranking of stored results. No weather or model access."""
    d = pd.DataFrame(table).copy()
    allowed = d.feasible.copy()
    if max_erosion is not None:
        allowed &= d.erosion_t_ha_p90.le(max_erosion)
    if min_income is not None:
        allowed &= d.income_p10.ge(min_income)
    d["allowed"] = allowed
    score = pd.Series(0., index=d.index)
    weight_sum = sum(max(0, weights.get(k, 0)) for k in METRICS)
    for metric in METRICS:
        low, high = d[metric].min(), d[metric].max()
        scaled = (d[metric] - low) / (high - low) if high > low else .5
        score += scaled * max(0, weights.get(metric, 0)) / max(weight_sum, 1e-9)
    d["score"] = score * 100
    return d.sort_values(["allowed", "score"], ascending=[False, False], kind="stable")


def comparison_pool(ranked, margin_share=.7, weights=None):
    """Pareto-optimal on every actively scored outcome; no hidden rerun.

    All metrics are oriented higher-is-better by summarize(). Strict improvement
    on at least one outcome is needed to dominate; ties remain on the front.
    A negative baseline cannot justify accepting an even larger loss.
    """
    if not 0 <= margin_share <= 1:
        raise ValueError("Margin share must be between zero and one")
    from fieldshift.sim.components import accelerator
    d = ranked.copy()
    baseline = float(d.loc[d.is_existing, "income_median"].iloc[0])
    floor = baseline * margin_share if baseline > 0 else baseline
    metrics = [k for k in METRICS if (DEFAULT_WEIGHTS if weights is None else weights).get(k, 0) > 0]
    if not metrics:
        metrics = ["income"]
    valid = d.allowed & np.isfinite(d[metrics]).all(axis=1)
    values = torch.tensor(d.loc[valid, metrics].to_numpy(dtype=np.float32, copy=True), device=accelerator())
    flags = []
    for i in range(0, len(values), 128):
        block = values[i:i + 128]
        dominated = ((values[None] >= block[:, None]).all(-1)
                     & (values[None] > block[:, None]).any(-1)).any(-1)
        flags.extend((~dominated).tolist())
    d["pareto"] = False
    d.loc[valid, "pareto"] = flags
    d["margin_floor"] = floor
    d["display_allowed"] = d.allowed & d.pareto & d.income_median.ge(floor)
    return d


def _bounds(frame):
    return {m: (float(frame[m].min()), float(frame[m].max())) for m in METRICS}


def _metric_values(runs, market):
    """METRICS from resampled runs [..., n, RUN_COLUMNS], same definitions as summarize()."""
    q = lambda j, level=.5: torch.nanquantile(runs[..., j], level, dim=-1)
    c = {k: i for i, k in enumerate(RUN_COLUMNS)}
    return {"income": q(c["income"]), "water": -q(c["et_total_mm"]), "soil": q(c["soil_carbon_input_t_ha"]),
            "nitrogen": q(c["mineral_n_kg_ha"]) + q(c["n_credit_kg_ha"]), "nitrate": -q(c["n_leached_kg_ha"]),
            "erosion": -q(c["erosion_t_ha"]), "cover": -q(c["bare_days"]), "forage": q(c["forage"]),
            "resilience": q(c["income"], .1), "market": torch.full_like(q(c["income"]), float(market))}


def _weighted(values, weights, bounds):
    total = sum(max(0, weights.get(k, 0)) for k in METRICS)
    score = 0.
    for m in METRICS:
        low, high = bounds[m]
        scaled = (values[m] - low) / (high - low) if high > low else values[m] * 0 + .5
        score = score + scaled * max(0, weights.get(m, 0)) / max(total, 1e-9)
    return score * 100


def near_tie(frame, runs, candidate, existing, weights, windows, draws=TIE_DRAWS, seed=0):
    """Is ``candidate``'s score advantage over ``existing`` inside the weather noise?

    Runs are cycle phases x consecutive weather windows (phase-major). Weather
    windows are resampled with replacement (phases stay paired with their window,
    and both rotations share the same draw), the medians/P10 behind each metric
    are recomputed, and the paired score difference is rescored on the min-max
    bounds of ``frame`` (all candidates, as for the displayed score). Tolerance = 1.96 x the standard
    deviation of that difference (a normal 95% half-width). A gap at or below
    the tolerance is a tie. No tolerance value is chosen by hand.
    """
    from fieldshift.sim.components import accelerator
    device = accelerator()
    bounds = _bounds(frame)
    rows = frame.set_index("id")
    data = torch.as_tensor(runs[[candidate, existing]], device=device)
    phases = data.shape[1] // windows
    if phases * windows != data.shape[1]:
        raise ValueError("Runs are not phases x weather windows")
    data = data.reshape(2, phases, windows, -1)
    generator = torch.Generator(device=device).manual_seed(seed)
    draw = torch.randint(windows, (draws, windows), device=device, generator=generator)
    sample = data[:, :, draw].permute(0, 2, 1, 3, 4).reshape(2, draws, phases * windows, -1)
    scores = [_weighted(_metric_values(sample[i], rows.loc[r, "market"]), weights, bounds)
              for i, r in enumerate((candidate, existing))]
    tolerance = float(1.96 * (scores[0] - scores[1]).std())
    point = {r: _weighted({m: torch.tensor(float(rows.loc[r, m])) for m in METRICS}, weights, bounds)
             for r in (candidate, existing)}
    gap = float(point[candidate] - point[existing])
    return {"gap": gap, "tolerance": tolerance, "tie": gap <= tolerance, "draws": draws,
            "windows": windows, "phases": phases}


def select_comparisons(ranked, slots=4, margin_share=.7, weights=None, runs=None, windows=None, details=False):
    """Existing rotation plus up to ``slots`` substantively different alternatives.

    Each strategy is named for the goals it wins among all allowed candidates;
    goals the existing rotation already wins are added to its labels. When two
    goals pick the same rotation it is shown once with both labels and the next
    goal fills the slot. With per-run results (``runs``), a winner whose score
    advantage over an allowed existing rotation is inside the weather noise
    (``near_tie``) is not shown as a separate winner: the existing rotation
    carries the label instead. Returns [(id, [labels])] (and tie details).
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    ranked = comparison_pool(ranked, margin_share, weights)
    existing = ranked.loc[ranked.is_existing].iloc[0]
    existing_id = int(existing.id)
    allowed = ranked.loc[ranked.display_allowed]
    labels = {existing_id: ["Existing rotation"]}
    order, ties = [], {}

    def tied(winner, goal_weights, frame):
        if runs is None or winner == existing_id or not bool(existing.allowed):
            return False
        return near_tie(frame, runs, winner, existing_id, goal_weights, windows)

    for name, goal_weights in GOALS:
        if not len(allowed):
            break
        if goal_weights == "cover_variant":
            pool = allowed.loc[allowed.cycle.eq(existing.cycle) & allowed.cover_option.ne("")
                                & ~allowed.double_crop & allowed.display_allowed]
            if not len(pool):
                continue
            winner = int(pool.iloc[0].id)
        elif goal_weights is None:
            winner = int(allowed.iloc[0].id)
            test = tied(winner, weights, ranked)
            if test:
                ties[name] = test | {"winner": winner}
                if test["tie"]:
                    labels[existing_id].append("As good a match as any candidate")
                    continue
        else:
            winner = int(rank(allowed.to_dict("records"), goal_weights).iloc[0].id)
            # Scores for the tie test use min-max bounds over all candidates, as displayed.
            test = tied(winner, goal_weights, ranked)
            if test:
                ties[name] = test | {"winner": winner}
                if test["tie"]:
                    labels[existing_id].append(name + " (tied within weather noise)")
                    continue
        if winner in labels:
            labels[winner].append(name)
        elif len(order) < slots:
            labels[winner] = [name]
            order.append(winner)
    # Fill empty slots only with candidates that beat the existing rotation's
    # score beyond the weather noise; a tied candidate is not shown as a match.
    for i in allowed.loc[allowed.display_allowed].id:
        if len(order) >= slots:
            break
        if int(i) not in labels:
            if float(ranked.loc[ranked.id.eq(i), "score"].iloc[0]) <= float(existing.score):
                break
            test = tied(int(i), weights, ranked)
            if test and test["tie"]:
                continue
            labels[int(i)] = ["Next best match to your priorities"]
            order.append(int(i))
    out = [(existing_id, labels[existing_id])] + [(i, labels[i]) for i in order]
    return (out, ties) if details else out


def explain(result, row, reasons, baseline=None):
    """Five plain-language sections for one strategy."""
    from fieldshift.sim.traits import (
        CTIC,
        FAO22,
        FAO56,
        MARCILLO,
        N_RATES,
        NICHOLS,
        SEIFERT,
        VETCH_CREDIT,
    )
    cycle, option, _ = result["rotations"][int(row["id"])]
    keys = [result["crops"][i] for i in cycle]
    traits = result["crop_traits"]
    cover_key = result["covers"][option - 1] if option else None
    low = sorted({name for key in keys + ([cover_key] if cover_key else []) for name, trait in traits[key].items()
                  if trait["confidence"] == "low"})
    gaps = list(result["gaps"])
    from fieldshift.sim.traits import DAVIS
    gaps.append("Diverse-cycle yield gains are a LOW-confidence transfer of corn +4% / soy +9% from " + DAVIS
                + ". Manure, red-clover/alfalfa underseeding, straw sales, reduced herbicides, extra labor and "
                "field-specific forage budgets are not represented. This is not a reproduction of Marsden profitability.")
    comparison = []
    if baseline is not None and int(baseline["id"]) != int(row["id"]):
        def delta(metric, unit, digits=0):
            return f"{row[metric] - baseline[metric]:+,.{digits}f} {unit}"
        comparison = [
            "Against the existing rotation: margin " + delta("income_median", "$/ha/yr")
            + ", bad-year (10th percentile) margin " + delta("income_p10", "$/ha/yr")
            + ", field water use " + delta("et_total_mm_median", "mm/yr")
            + ", humified soil carbon input " + delta("soil_carbon_input_t_ha_median", "t C/ha/yr", 2)
            + ", erosion " + delta("erosion_t_ha_median", "t/ha/yr", 1)
            + ", nitrate leached " + delta("n_leached_kg_ha_median", "kg N/ha/yr")
            + ", days without living cover " + delta("bare_days_median", "days/yr") + ".",
        ]
    families = [traits[k]["family"]["value"] for k in keys]
    repeats = sum(a == b for a, b in zip(families, families[1:] + families[:1], strict=True))
    opportunities = []
    if repeats == 0:
        opportunities.append("No plant family follows itself, so soil-borne pest and disease cycles are broken every year.")
    else:
        opportunities.append(f"{repeats} same-family back-to-back year(s) per cycle (wrap-around included): pest "
                             f"carry-over penalty median {row['pest_loss_pct_median']:.1f}% of yield.")
    if any(traits[k]["legume"]["value"] >= .5 for k in keys):
        opportunities.append(f"Includes a legume: about {row['n_fixed_kg_ha_median']:.0f} kg N/ha/yr fixed; "
                             f"{row['n_credit_kg_ha_median']:.0f} kg N/ha potential release from the residue store "
                             "(not an additional fertilizer recommendation).")
    if any(traits[k]["perennial"]["value"] >= .5 for k in keys):
        opportunities.append("Includes a perennial forage: longer living cover, deep roots, and a hay income stream.")
    if cover_key:
        name = cover_key.replace("_cover", "").replace("_", " ")
        opportunities.append(
            f"Winter {name} after each harvest where it fits: established in {row['cover_established_median']:.0%} "
            f"of years, {row['cover_biomass_t_ha_median']:.1f} t/ha dry matter at termination, "
            f"costing ${row['cover_cost_usd_ha_median']:.0f}/ha/yr (seed, seeding and termination).")
        opportunities.append(f"Soil water at the next planting: {row['water_at_planting_mm_median']:.0f} mm in the "
                              "2 m profile (the cover's spring water use is already subtracted).")
        opportunities.append(f"Cover mineral-N uptake {row.get('n_cover_uptake_kg_ha_median', 0):.0f} kg/ha/yr; "
                             f"legume-cover fertilizer saving ${row.get('n_fertilizer_saving_usd_ha_median', 0):.0f}/ha/yr "
                             "(historical fertilizer-price scenario, included in margin).")
        if row["cover_established_median"] < .6:
            opportunities.append("HIGH FAILURE RISK: this late-sown cover fails in most years. Earlier seeding "
                                 "or a locally tested cultivar is needed; it is an opportunity scenario, not a recommendation.")
    elif row["bare_days_median"] > 120:
        opportunities.append(f"About {row['bare_days_median']:.0f} days a year without living cover between harvest "
                             "and the next crop: a window for a winter cover crop.")
    vulnerable = []
    loss_months = pd.DataFrame(result.get("loss_months", []))
    arrays = result.get("arrays", {})
    for key in dict.fromkeys(keys):
        i = result["crops"].index(key)
        if "flower_start" in arrays:
            start = arrays["flower_start"][i]
            valid = start >= 0
            if valid.any():
                prev = arrays["prev_len"]
                doy = np.where(start >= prev, start - prev, start)[valid]
                month = int(pd.Series(pd.to_datetime(doy, unit="D", origin="2001-01-01").month).mode().iloc[0])
                note = f"{key.replace('_', ' ')}: most sensitive to drought and heat around month {month} (flowering)"
                if len(loss_months):
                    share = loss_months.loc[loss_months.month.eq(month), "indemnity"].sum() / loss_months.indemnity.sum()
                    note += f"; {share:.0%} of this county's insured losses fall in that month"
                vulnerable.append(note + ".")
    if result["normal_yield_gaps"]:
        gaps.append("No local yield normal for: " + ", ".join(result["normal_yield_gaps"])
                    + "; the trait reference yield is used.")
    borrowed = [f"{k}: {s}" for k, s in result.get("normal_sources", {}).items() if "no county series" in s and k in keys]
    if borrowed:
        gaps.append("Yield normals borrowed from nearby counties or the state: " + "; ".join(borrowed) + ".")
    gaps.append("Low-confidence traits in this strategy: " + ", ".join(low) + ".")
    gaps.append("P10–P90 spans historical weather windows and cycle phases; it is not a confidence interval.")
    if cover_key:
        gaps.append("Cover-crop effect on the next cash crop's yield is set to zero: " + MARCILLO
                    + ". Farmer surveys report small gains (" + CTIC + "); not used because they are self-reported.")
    return {
        "Why suggested": " ".join([(f"{' and '.join(reasons)}. Priority score {row['score']:.1f}/100, relative "
                                   "to this candidate set.")] + comparison),
        "Supporting evidence": result.get("loss_evidence", []) + [
            result["rotation_evidence"],
            (f"Daily NASA POWER 1981–2025, scored in steady state (one cycle of spin-up, every phase); yield path: "
            f"{result['yield_path']}."),
            f"FAO-56 daily Kc/Ks water bucket: {FAO56}; {FAO22}.",
            f"Rainfed corn-after-corn −4.3%, soybean-after-soybean −10.3%: {SEIFERT}.",
            f"Reference N rates (historical, not current advice): {N_RATES}. Legume-cover N credit: {VETCH_CREDIT}.",
            ("Independent QDANN satellite sample: 34,866 points, interior penalties corn 0.96%, soy 4.03%, soy "
             "penalty rising from 2.99% to 6.47% with longer repetition."),
        ] + ([(f"Cover establishment: {NICHOLS}. Species minimum time before frost: "
              f"{traits[cover_key]['minimum_frost_free_days']['source']}. Winter survival: "
              f"{traits[cover_key]['winter_min_c']['source']}. Cost: "
              f"{traits[cover_key]['cost_usd_ha']['source']}.")] if cover_key else []),
        "Benefits and opportunities": [
            (f"Operating margin median ${row['income_median']:,.0f}/ha/yr; bad years (P10) ${row['income_p10']:,.0f}/ha "
             f"({row['income_confidence'].upper()} confidence; excludes land)."),
            "Margin-confidence reason: " + row.get("income_confidence_reason", "Reference scenarios, not field quotes."),
            (f"Field water use (crops, covers and bare soil) median {row['et_total_mm_median']:.0f} mm/yr; ideal "
            f"net irrigation {row['irrigation_mm_median']:.0f} mm/yr."),
            (f"Nitrate leached {row['n_leached_kg_ha_median']:.0f} kg N/ha/yr; humified soil carbon input "
            f"{row['soil_carbon_input_t_ha_median']:.2f} t C/ha/yr (organic matter "
            f"{row['om_trend_pct_per_decade']:+.2f} %-points per decade)."),
        ] + opportunities + vulnerable,
        "Information gaps": gaps + result.get("sensitivity_gaps", []),
        "Risks needing local review": [
            ("This supports and does not replace farmer judgment. Confirm soil tests, drainage, varieties, planting "
            "dates and pest history."),
            "Confirm buyers, machinery, herbicide carry-over, forage use and crop insurance rules before changing.",
            ("Nitrogen flows are low-confidence accounting, not fertilizer advice. N shortage does not affect yield; "
             "the internal unmet-demand diagnostic is not shown as an agronomic outcome."),
            "Erosion uses an annual-rain erosivity proxy; tonnes per hectare are screening values only.",
        ] + ([("Cover crops need timely seeding after harvest and termination about two weeks before corn "
              "planting; a wet spring can delay both.")] if cover_key else []),
    }
