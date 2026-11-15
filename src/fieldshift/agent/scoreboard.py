"""Deterministic, disk-cached ranking over the existing v22 planner search."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.agent import farms
from fieldshift.agent.runtime import (
    METRIC_DEFINITIONS,
    METRICS,
    WINTER_ALIASES,
    InputError,
    clean,
    private_text,
)
from fieldshift.planner import search
from fieldshift.sim.components import accelerator
from fieldshift.sim.engine import cover_eligible, ledger_pass, trajectories, water_pass

VERSION = "scoreboard-v1"
KEYS = ("income", "water", "soil", "nitrogen", "erosion", "cover", "forage", "resilience", "market")
DEFAULT_WEIGHTS = {k: search.DEFAULT_WEIGHTS[k] for k in KEYS}
WEIGHTS_HELP = {
    "income": {"metrics": ["margin_usd_ha_yr.median"], "direction": "higher",
               "description": "Prefer higher median return over operating costs; ownership/overhead and land are separate."},
    "water": {"metrics": ["field_water_use_mm_yr.median"], "direction": "lower",
              "description": "Prefer less total annual field evapotranspiration, not irrigation withdrawals."},
    "soil": {"metrics": ["soil_carbon_input_t_c_ha_yr.median"], "direction": "higher",
             "description": "Prefer higher annual soil carbon input, not a measured soil-carbon stock."},
    "nitrogen": {"metrics": ["nitrate_leached_kg_n_ha_yr.median"], "direction": "lower",
                 "description": "Prefer lower nitrate leaching; unlike the legacy planner weight, this does not reward retained mineral N."},
    "erosion": {"metrics": ["erosion_t_ha_yr.median"], "direction": "lower",
                "description": "Prefer lower estimated soil erosion."},
    "cover": {"metrics": ["days_without_living_cover_yr.median"], "direction": "lower",
              "description": "Prefer fewer days without living crop or cover."},
    "forage": {"metrics": ["forage_fraction.median"], "direction": "higher",
               "description": "Prefer a larger fraction of crop-years in forage."},
    "resilience": {"metrics": ["margin_usd_ha_yr.p10"], "direction": "higher",
                   "description": "Prefer a higher bad-weather-cycle margin (10th percentile), not a confidence bound."},
    "market": {"metrics": ["local_market_fraction.median"], "direction": "higher",
                "description": "Prefer larger county harvested-area shares; production-depth proxy, not verified hay buyers, grades or price liquidity."},
}
NUMERIC_LIMITS = {
    "max_nitrate_kg_n_ha_yr": "n_leached_kg_ha_median",
    "max_field_water_use_mm_yr": "et_total_mm_median",
    "max_erosion_t_ha_yr": "erosion_t_ha_median",
    "max_days_without_living_cover_yr": "bare_days_median",
}
BOOL_LIMITS = ("require_winter_cover", "no_new_equipment", "keep_forage_crop")
LIST_LIMITS = ("require_crops", "exclude_crops")
FIELDS = METRICS


def validate(payload):
    if not isinstance(payload, dict) or payload.keys() - {"weights", "limits", "rain_change_pct", "temp_change_c", "top"}:
        raise InputError("Body has weights, optional limits, rain_change_pct, temp_change_c and top only")
    weights = payload.get("weights")
    if not isinstance(weights, dict) or weights.keys() - set(KEYS):
        raise InputError(f"weights must be an object over exactly these valid keys (missing = 0): {list(KEYS)}")
    weights = {k: farms.number(weights.get(k, 0), f"weight {k}", 0, math.inf) for k in KEYS}
    maximum = max(weights.values())
    if not maximum:
        raise InputError("At least one weight must be positive; all-zero weights cannot rank rotations")
    # Avoid overflow for otherwise valid, very large finite weights.
    total = math.fsum(v / maximum for v in weights.values())
    weights = {k: (v / maximum) / total for k, v in weights.items()}
    limits = payload.get("limits", {})
    valid = set(NUMERIC_LIMITS) | set(BOOL_LIMITS) | set(LIST_LIMITS) | {"min_margin_share_of_existing"}
    if not isinstance(limits, dict) or limits.keys() - valid:
        raise InputError(f"Valid limits: {sorted(valid)}")
    limits = dict(limits)
    for key in set(NUMERIC_LIMITS) | {"min_margin_share_of_existing"}:
        if key in limits:
            limits[key] = farms.number(limits[key], key, 0, 1 if key == "min_margin_share_of_existing" else math.inf)
    for key in BOOL_LIMITS:
        if key in limits and not isinstance(limits[key], bool):
            raise InputError(f"{key} must be true or false")
    for key in LIST_LIMITS:
        if key in limits:
            if not isinstance(limits[key], list) or any(not isinstance(k, str) for k in limits[key]):
                raise InputError(f"{key} must be a list of crop keys from /options")
            limits[key] = sorted({WINTER_ALIASES.get(k, k) for k in limits[key]})
    top = payload.get("top", 15)
    if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 40:
        raise InputError("top must be an integer from 1 to 40")
    rain = farms.number(payload.get("rain_change_pct", 0), "rain_change_pct", -100, 200)
    temp = farms.number(payload.get("temp_change_c", 0), "temp_change_c", -15, 15)
    return weights, limits, rain, temp, top


def _fingerprint():
    code = Path(__file__).parents[1]
    files = list((code / "sim").glob("*.py")) + [
        code / "planner" / "data.py", code / "planner" / "search.py",
        Path(__file__), code / "agent" / "runtime.py",
        code / "agent" / "farms.py",
    ]
    h = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(files)))
    assets = [
        "models/yield_response_v2_power.pt", "static/county_static.parquet", "planner/nass_reference_prices.json",
        "planner/nass_index_v2.parquet", "nass_crop_progress_dates.parquet",
        "planner/nass_practice_yields_v1.parquet", "planner/nass_state_yields_v3.parquet",
        "weather/power_county_daily.f32", "weather/power_county_daily.json",
        "weather/prism_county_8day.f32", "weather/prism_county_8day.json",
    ]
    for name in assets:
        p = Path("data/processed") / name
        h.update(str((name, p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (name, None)).encode())
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.touch(mode=0o600)
    part.chmod(0o600)
    part.write_text(json.dumps(clean(value), separators=(",", ":"), allow_nan=False))
    part.replace(path)


def export(rotation, farm):
    cycle, option, double = rotation
    eligible = cover_eligible(farm.crops)
    out = []
    for i, c in enumerate(cycle):
        step = {"crop": farm.keys[c]}
        if double and farm.crops[c].v("autumn_sown"):
            step["double_crop"] = "soybean"
        elif option and eligible[c, cycle[(i + 1) % len(cycle)]]:
            step["winter"] = farm.cover_keys[option - 1]
        out.append(step)
    return out


def simulate_pool(farm, rain, temp):
    """One batched daily/ledger pass, never one simulation call per cycle."""
    r, t = farm.changes(rain, temp)
    p = farm.preparation(r, t)
    maturity = p["maturity"][0].tolist()
    retained = [i for i, c in enumerate(farm.crops) if maturity[i] >= .6 or c.key in farm.site["existing"]]
    pruned = list(farm.excluded)
    for i, crop in enumerate(farm.crops):
        if i not in retained:
            pruned.append(f"{crop.key}: reaches maturity in {maturity[i]:.0%} of weather years (<60%)")
    stated = "existing_rotation" in farm.record
    existing = tuple(farm.keys.index(s["crop"]) for s in farm.existing) if stated else search.canonical([farm.keys.index(k) for k in farm.site["existing"]])
    if stated and all(s["crop"] == "alfalfa" for s in farm.existing):
        existing = (existing[0],) * 4  # same four-year stand policy as Farm.evaluate
    enumerated = search.enumerate_rotations(farm.crops, farm.covers, p, retained, existing, pruned)
    checks = {"operation": p["operation"][0].tolist(), "maturity": maturity,
              "cover_survival": p["covers"]["alive"][0].float().mean(-1).tolist() if farm.covers else [],
              "double_maturity": p["double"]["maturity"][0].tolist() if p["double"] is not None else []}
    rotations, cycles, notes, errors = [], [], [], []
    for i, rotation in enumerate(enumerated):
        cycle = farm.existing if stated and i == 0 else export(rotation, farm)
        warnings, invalid = farm.warnings(cycle, p, checks)
        if i and invalid:
            pruned.append(search.label(rotation, farm.crops, farm.covers) + ": " + "; ".join(invalid))
            continue
        rotations.append(rotation)
        cycles.append(cycle)
        notes.append(warnings)
        errors.append(invalid)
    score = farm.score_years if stated else 4
    tr = trajectories(rotations, farm.crops, len(p["years"]), score=score, phases=score if stated else None)
    if stated:
        # The first reference can have a different cover on each season.
        for a in np.where(tr["rotation"] == 0)[0]:
            tr["cover"][a] = [farm.cover_keys.index(farm.existing[j % len(farm.existing)]["winter"]) + 1
                             if farm.existing[j % len(farm.existing)].get("winter") else 0 for j in tr["position"][a]]
            tr["double"][a] = [bool(farm.existing[j % len(farm.existing)].get("double_crop")) for j in tr["position"][a]]
    water = water_pass(p, [farm.site["soil"]], farm.crops, farm.covers, farm.double, tr, farm.irrigated)
    base_fao = None
    if r or t or "irrigation_capacity_mm_yr" in farm.site["soil"]:
        base = water_pass(farm.base, farm.reference_soils, farm.crops, farm.covers, farm.double, tr, farm.irrigated)
        base_fao = {"fao_main": base["fao_main"]}
    ledger = ledger_pass(p, water, [farm.site["soil"]], farm.normals, farm.crops, farm.covers,
                         farm.double, tr, farm.irrigated, fao_reference=farm.reference, base_fao=base_fao)
    points = search.summarize(ledger, tr, rotations, farm.crops, farm.covers, farm.site)
    count, per, spin = len(rotations), len(tr["rotation"]) // len(rotations), tr["spin"]
    stack = torch.stack([ledger[k][:, spin:].nanmean(-1) for k in FIELDS.values()], -1).reshape(count, per, -1)
    q = torch.nanquantile(stack, torch.tensor([.1, .5, .9], device=p["device"]), dim=1).tolist()
    om = ((ledger["organic_matter_pct"][:, -1] - ledger["organic_matter_pct"][:, spin - 1]) / score * 10).reshape(count, per)
    oq = torch.nanquantile(om, torch.tensor([.1, .5, .9], device=p["device"]), dim=1).tolist()
    runs = torch.stack([ledger[k][:, spin:].nanmean(-1) for k in search.RUN_COLUMNS], -1).reshape(count, per, -1)
    from fieldshift.sim.economics import ECON_FIELDS
    physical = torch.stack([ledger[k][:, spin:] for k in ECON_FIELDS], -1).reshape(count, per, -1)
    runs = torch.cat([runs, physical], -1)
    rows = []
    for i, rotation in enumerate(rotations):
        cycle, warnings, infeasible = cycles[i], list(notes[i]), errors[i]
        if rotation[2]:
            warnings.append("Double-crop soybean uses generic maturity and costs; review the following autumn planting window locally")
        if i == 0:
            warnings.extend("Existing reference: " + s for s in infeasible)
        metrics = {name: {label: q[j][i][k] for j, label in enumerate(("p10", "median", "p90"))}
                   for k, name in enumerate(FIELDS)}
        metrics["organic_matter_trend_pct_points_decade"] = {
            label: oq[j][i] for j, label in enumerate(("p10", "median", "p90"))}
        metrics["local_market_fraction"] = dict.fromkeys(("p10", "median", "p90"), points[i]["market"])
        points[i]["nitrogen"] = -points[i]["n_leached_kg_ha_median"]
        if points[i]["income_confidence"] == "low":
            warnings.append("Income confidence: " + points[i]["income_confidence_reason"])
        if rotation[2]:
            points[i]["income_confidence"] = "low"
            warnings.append("Double-crop yield fraction and costs are low-confidence transfers, not a local enterprise budget")
        label = (" → ".join(s["crop"] + (" + " + s["winter"] if s.get("winter") else "") for s in cycle)
                 if stated and i == 0 else search.label(rotation, farm.crops, farm.covers))
        rows.append({"rotation": cycle, "label": label,
                     "metrics": metrics, "income_confidence": points[i]["income_confidence"],
                     "warnings": list(dict.fromkeys(warnings)), "infeasible": infeasible,
                     "forage": any(farm.crops[c].v("forage_value") > .5 for c in rotation[0]),
                      "equipment": (any(s["crop"] not in farm.site["existing"]
                                        or s.get("winter", "") not in {e.get("winter", "") for e in farm.existing}
                                        or (s.get("double_crop") and not any(e.get("double_crop") for e in farm.existing))
                                        for s in cycle) if stated else
                                    bool(rotation[1] or rotation[2] or any(farm.keys[c] not in farm.site["existing"] for c in rotation[0])))})
    valid_keys = {s["crop"] for row in rows for s in row["rotation"]}
    valid_keys |= {s["winter"] for row in rows for s in row["rotation"] if s.get("winter")}
    if any(r[2] for r in rotations):
        valid_keys.add("double_crop_soybean")
    # A key published by /options remains valid when requested weather makes
    # that crop infeasible: requiring it gives no survivors; excluding it is a
    # legitimate no-op, not an "unknown crop" error.
    options = farm.options()
    valid_keys |= {c["crop"] for c in options["main_crops"] + options["winter_options"]}
    if options["double_crop_option"]:
        valid_keys.add(options["double_crop_option"]["crop"])
    data = {"points": points, "rows": rows, "pruned": pruned, "windows": len(tr["starts"]),
             "economics": p["economics"][0], "score_years": score,
            "county_label": farm.site["label"],
            "valid_keys": sorted(valid_keys), "existing_id": 0, "n_cycles": count}
    return clean(data), runs.detach().cpu().numpy()


def violations(data, limits):
    baseline = data["points"][0]["income_median"]
    floor = baseline * limits["min_margin_share_of_existing"] if baseline > 0 and "min_margin_share_of_existing" in limits else baseline
    out = []
    for p, row in zip(data["points"], data["rows"], strict=True):
        bad = ["infeasible_sequence: " + s for s in row["infeasible"]]
        for key, metric in NUMERIC_LIMITS.items():
            if key in limits and (not math.isfinite(p[metric]) or p[metric] > limits[key]):
                bad.append(key)
        if "min_margin_share_of_existing" in limits and p["income_median"] < floor:
            bad.append("min_margin_share_of_existing")
        present = {s["crop"] for s in row["rotation"]} | {s["winter"] for s in row["rotation"] if s.get("winter")}
        if any(s.get("double_crop") for s in row["rotation"]):
            present.add("double_crop_soybean")
        for key in limits.get("require_crops", []):
            if key not in present:
                bad.append("require_crops:" + key)
        for key in limits.get("exclude_crops", []):
            if key in present:
                bad.append("exclude_crops:" + key)
        if limits.get("require_winter_cover") and not any(s.get("winter") for s in row["rotation"]):
            bad.append("require_winter_cover")
        if limits.get("no_new_equipment") and row["equipment"]:
            bad.append("no_new_equipment")
        if limits.get("keep_forage_crop") and not row["forage"]:
            bad.append("keep_forage_crop")
        if any(not math.isfinite(p[k]) for k in KEYS):
            bad.append("nonfinite_score_metric")
        out.append(bad)
    return out


def ranking(data, runs, weights, limits, top, economic_overrides=None):
    """Pure reranking except accelerator bootstrap; no weather/model calls."""
    economic_drivers = {}
    if "economics" in data:
        from fieldshift.sim.economics import ECON_FIELDS, income_evidence, reprice
        data = copy.deepcopy(data)
        tensor = torch.as_tensor(runs, device=accelerator())
        actual = tensor[..., len(search.RUN_COLUMNS):].reshape(*tensor.shape[:2], data.get("score_years", 4), len(ECON_FIELDS))
        fields = {name: actual[..., i] for i, name in enumerate(ECON_FIELDS)}
        income, listed, irrigation, ctx = reprice(fields, data["economics"], economic_overrides)
        runs = tensor[..., :len(search.RUN_COLUMNS)].clone()
        runs[..., search.RUN_COLUMNS.index("income")] = income.mean(-1)
        q = torch.quantile(income.mean(-1), torch.tensor([.1, .5, .9], device=tensor.device), dim=1).tolist()
        iq = torch.quantile(irrigation.mean(-1), torch.tensor([.1, .5, .9], device=tensor.device), dim=1).tolist()
        lq = torch.quantile(listed.mean(-1), torch.tensor([.1, .5, .9], device=tensor.device), dim=1).tolist()
        for i, (p, row) in enumerate(zip(data["points"], data["rows"], strict=True)):
            p.update(income=q[1][i], resilience=q[0][i], income_median=q[1][i], income_p10=q[0][i], income_p90=q[2][i])
            row["metrics"]["margin_usd_ha_yr"] = dict(zip(("p10", "median", "p90"), (v[i] for v in q), strict=True))
            row["metrics"]["irrigation_cost_usd_ha_yr"] = dict(zip(("p10", "median", "p90"), (v[i] for v in iq), strict=True))
            row["metrics"]["return_over_listed_costs_usd_ha_yr"] = dict(zip(("p10", "median", "p90"), (v[i] for v in lq), strict=True))
            confidence, drivers = income_evidence(ctx, [s["crop"] for s in row["rotation"]]+[s.get("winter", "") for s in row["rotation"]])
            if confidence:
                row["income_confidence"] = confidence
            economic_drivers[i] = drivers
    bad = violations(data, limits)
    allowed = [i for i, v in enumerate(bad) if not v]
    points = pd.DataFrame(data["points"])
    frame = points.iloc[allowed].copy()
    scores, order, rows = {}, [], []
    if len(frame):
        ranked = search.rank(frame.to_dict("records"), weights)
        scores = dict(zip(ranked.id.astype(int), ranked.score.astype(float), strict=True))
        order = ranked.id.astype(int).tolist()
        bounds = search._bounds(frame)
        # An excluded reference is projected onto eligible bounds WITHOUT changing them.
        if 0 not in scores:
            scores[0] = float(search._weighted({k: points.loc[0, k] for k in search.METRICS}, weights, bounds))
        best = order[0]
        device = accelerator()
        tie_runs = torch.as_tensor(runs, device=device).clone()
        col = {name: i for i, name in enumerate(search.RUN_COLUMNS)}
        tie_runs[..., col["mineral_n_kg_ha"]] = -tie_runs[..., col["n_leached_kg_ha"]]
        tie_runs[..., col["n_credit_kg_ha"]] = 0
        for position, i in enumerate(order[:top], 1):
            test = ({"tolerance": 0., "tie": True} if i == best else
                    search.near_tie(frame, tie_runs, best, i, weights, data["windows"]))
            original = data["rows"][i]
            rows.append({k: original[k] for k in ("rotation", "label", "metrics", "income_confidence", "warnings")} |
                         {"rank": position, "score": scores[i], "tie_tolerance": test["tolerance"],
                          "income_confidence_drivers": economic_drivers.get(i, ""),
                         "tied_with_best": scores[best] - scores[i] <= test["tolerance"]})
    existing_tie = {"tolerance": None, "tie": False}
    if order and 0 in allowed:
        existing_tie = ({"tolerance": 0., "tie": True} if best == 0 else
                        search.near_tie(frame, tie_runs, best, 0, weights, data["windows"]))
    existing_rank = 1 + sum(scores[i] > scores[0] for i in allowed) if scores else None
    existing = {"rank": existing_rank, "score": scores.get(0), "metrics": data["rows"][0]["metrics"],
                 "rotation": data["rows"][0]["rotation"], "violates_limits": bad[0],
                "income_confidence_drivers": economic_drivers.get(0, ""),
                 "rank_is_counterfactual": bool(bad[0]), "tie_tolerance": existing_tie["tolerance"],
                 "tied_with_best": bool(order and 0 in allowed and scores[order[0]]-scores[0] <= existing_tie["tolerance"])}
    pruned = list(data["pruned"])
    reasons = sorted({reason for items in bad for reason in items})
    pruned.extend(f"{reason}: excludes {sum(reason in v for v in bad)} cycles" for reason in reasons)
    return {"existing": existing, "rows": rows, "n_cycles_scored": data["n_cycles"],
            "n_after_limits": len(allowed), "pruned": pruned}


def build(runtime, farm_id, payload):
    started = time.perf_counter()
    weights, limits, rain, temp, top = validate(payload)
    with runtime.lock:
        record = farms.get(farm_id, runtime.root)
        from fieldshift.agent.runtime import constraints
        equipment, forage, _ = constraints(record)
        if equipment:
            limits["no_new_equipment"] = True
        if forage:
            limits["keep_forage_crop"] = True
        physical = farms.physical_record(record)
        key = _hash([VERSION, search.VERSION, _fingerprint(), physical, rain, temp])
        root = Path(runtime.root or "data/processed/engine_v26") / "scoreboard"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        response_path = root / ("rank_" + _hash([key, farm_id, record["anonymous"], record.get("economics"), weights, limits, top]) + ".json")
        if response_path.exists():
            result = json.loads(response_path.read_text())
            result["latency_seconds"] = time.perf_counter() - started
            return result
        meta_path, arrays_path = root / (key + ".json"), root / (key + ".npz")
        if meta_path.exists() and arrays_path.exists():
            data = json.loads(meta_path.read_text())
            with np.load(arrays_path) as arrays:
                runs = arrays["runs"]
        else:
            farm = runtime.farm(farm_id)
            known = set(farm.keys + farm.cover_keys) | ({"double_crop_soybean"} if farm.double is not None else set())
            for key_name in LIST_LIMITS:
                unknown = set(limits.get(key_name, [])) - known
                if unknown:
                    raise InputError(f"Unknown/excluded crop keys {sorted(unknown)} in {key_name}; valid keys: {sorted(known)}")
            data, runs = simulate_pool(farm, rain, temp)
            # Lazy source extraction may have created input caches during preparation.
            # Publish under the final fingerprint so a price fork cannot rerun physics.
            key = _hash([VERSION, search.VERSION, _fingerprint(), physical, rain, temp])
            meta_path, arrays_path = root / (key + ".json"), root / (key + ".npz")
            root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(arrays_path.with_suffix(".part.npz"), runs=runs)
            arrays_path.with_suffix(".part.npz").chmod(0o600)
            arrays_path.with_suffix(".part.npz").replace(arrays_path)
            _write(meta_path, data)
        for key_name in LIST_LIMITS:
            unknown = set(limits.get(key_name, [])) - set(data["valid_keys"])
            if unknown:
                raise InputError(f"Unknown/excluded crop keys {sorted(unknown)} in {key_name}; valid keys: {data['valid_keys']}")
        economic_overrides = dict(record.get("economics", {}))
        if "irrigation_capacity_mm_yr" in record["overrides"]:
            economic_overrides["irrigation_capacity_mm_yr"] = record["overrides"]["irrigation_capacity_mm_yr"]
        result = ranking(data, runs, weights, limits, top, economic_overrides)
        if record["anonymous"]:
            site = {"adm_id": record["adm_id"], "label": data["county_label"]}
            result["pruned"] = [private_text(s, site) for s in result["pruned"]]
            for row in result["rows"]:
                row["warnings"] = [private_text(s, site) for s in row["warnings"]]
                row["income_confidence_drivers"] = private_text(row["income_confidence_drivers"], site)
            result["existing"]["income_confidence_drivers"] = private_text(result["existing"]["income_confidence_drivers"], site)
        overrides = record["overrides"]
        result["applied_weather_changes"] = {
            "rain_change_pct_total": 100*((1+overrides.get("rain_change_pct", 0)/100)*(1+rain/100)-1),
            "temp_change_c_total": overrides.get("temp_change_c", 0)+temp,
            "rain_rule": "farm multiplier × request multiplier",
            "request_rain_change_pct": rain, "request_temp_change_c": temp}
        result.update({"farm_id": farm_id, "simulator_version": search.VERSION, "metric_definitions":METRIC_DEFINITIONS, "weights_normalised": weights,
                       "limits_applied": limits,
                        "method": "Exhaustive v23 four-year canonical cycles, all phases and historical windows after four-year spin-up. "
                                 "Hard limits apply to cycle medians (margin share uses existing under the same requested weather). "
                                 "Min-max weighted planner.rank scores are scaled only over survivors, 0–100. nitrogen means LOWER nitrate leaching. "
                                 "Ties: best lead <= 1.96 SD of paired score differences, seed 0, 2000 resampled weather windows, phases paired; "
                                 "fixed survivor min-max bounds. Existing rank is its placement against survivors, counterfactual if excluded; "
                                  "rank/score are null when none survive. P10/P90 describe weather variation, not confidence intervals. "
                                   "v22 MLP x FAO response retained; annual irrigation cap where known, applied water priced; no N-yield feedback. "
                                   "Price/cost edits reprice stored yields, fertilizer and irrigation without field resimulation. "
                                  "ET anomalies fail validation and the N budget is unvalidated; scenario comparisons, not agronomic prescriptions. "
                                  "Forecasts are not used to replace historical simulation weather.",
                        "latency_seconds": time.perf_counter() - started})
        if "existing_rotation" in record:
            result["comparison_years_per_run"] = data.get("score_years", 4)
            result["method"] += (" Farmer-stated reference retained, including per-season covers. "
                                 "All candidates use a common LCM(4, stated cycle length) scoring horizon and equally repeated cycle phases.")
        response_path = root / ("rank_" + _hash([key, farm_id, record["anonymous"], record.get("economics"), weights, limits, top]) + ".json")
        result = clean(result)
        _write(response_path, result)
        return result
