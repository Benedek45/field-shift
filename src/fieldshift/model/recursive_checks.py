"""Free-running counterfactual and carry-over diagnostics, never test-score tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.model.expanded import target_for_fold
from fieldshift.model.neural import accelerator_device
from fieldshift.model.recursive import load_recursive
from fieldshift.model.recursive_data import FieldData
from fieldshift.model.recursive_train import FEATURES, FieldRows, run_batch


@torch.inference_mode()
def scenario_summary(model, batch):
    rollout, (mean, sd, _, _) = run_batch(model, batch)
    states = rollout["states"][:, :, 1:].mean(0)
    season = batch["season"] / batch["season"].sum(1, keepdim=True).clamp_min(1)
    wetness = (states[..., 0] * season).sum(1)
    canopy = (states[..., 3] * season).sum(1)
    median = mean.exp()
    arithmetic = (mean + sd.square() / 2).exp()
    return torch.stack([wetness, canopy, median, arithmetic], -1)


def _edited(rows, batch, start, stop, kind):
    field = rows.field
    def transform(raw, day):
        active = day.ge(start[:, None]) & day.lt(stop[:, None])
        if kind == "rain":
            raw[..., field.variables["prec"]] *= torch.where(active, .7, 1.)
        elif kind == "heat":
            raw[..., field.variables["tmax"]] += active * 6
            raw[..., field.variables["tmin"]] += active * 3
            raw[..., field.variables["tavg"]] += active * 4.5
        else:
            raise ValueError(kind)
        return raw
    driver, season, _ = field.drivers(
        batch["county"], batch["crop"], batch["year"], years=2, daily_transform=transform)
    return batch | {"drivers": driver, "season": season}


def _period_boundary(field, batch, period):
    return field.first[batch["positions"].gather(1, period[:, None]).squeeze(1)]


@torch.inference_mode()
def what_if(model_path, features=FEATURES, max_cases=256):
    model, metadata = load_recursive(model_path)
    field = FieldData()
    if field.static_names != metadata["static_names"]:
        raise ValueError("Counterfactual inputs differ from the frozen model snapshot")
    data = pd.read_parquet(features)
    fit = data.year.isin(metadata["fit_years"]).to_numpy()
    target, _ = target_for_fold(data, fit)
    rows = FieldRows(data, field, target)
    # A prespecified sample of primary corn counties in held years; irrigation is
    # actively set to the two scenario values, not silently treating unknown as 0.
    subset = data.loc[data.primary & data.crop.eq("corn")
                      & data.year.isin(metadata["test_years"]) & data.year.gt(1981)]
    subset = subset.sample(min(max_cases, len(subset)), random_state=20261114)
    indices = rows.indices(data.index.isin(subset.index))
    if not len(indices):
        raise ValueError("No held-year corn rows for the counterfactual diagnostic")
    climo = field.climatology(metadata["fit_years"])
    batch = rows.batch(indices, climo, years=2)
    active = batch["season"] > 0
    clock = torch.arange(active.shape[1], device=field.device)[None].expand_as(active)
    first = torch.where(active, clock, active.shape[1]).amin(1)
    last = torch.where(active, clock, 0).amax(1)
    start = _period_boundary(field, batch, first)
    stop = _period_boundary(field, batch, last) + field.counts[
        batch["positions"].gather(1, last[:, None]).squeeze(1)].long()
    rain, heat = {}, {}
    summaries = []
    for share in (0., .8):
        scenario = dict(batch)
        scenario["static"] = batch["static"].clone()
        scenario["static"][:, -1] = share
        scenario["share"] = torch.full_like(batch["share"], share)
        baseline = scenario_summary(model, scenario)
        edited = scenario_summary(model, _edited(rows, scenario, start, stop, "rain"))
        summaries.append(torch.cat([baseline.mean(0), edited.mean(0), (edited - baseline).mean(0)]))
    # Same +6/+3 C (maximum/minimum) eight-day anomaly at a baseline vegetative
    # versus flowering location; GDD/stages recomputed, demand/humidity fixed.
    stage = batch["drivers"]
    vegetative = (stage[..., 17] > 0) & active
    flowering = (stage[..., 18] > 0) & active
    available = vegetative.any(1) & flowering.any(1)
    heat_batch = {name: value[available] for name, value in batch.items()}
    heat_batch["static"] = heat_batch["static"].clone()
    heat_batch["static"][:, -1] = 0.
    heat_batch["share"] = torch.zeros_like(heat_batch["share"])
    heat_values = []
    if len(heat_batch["share"]):
        heat_values.append(scenario_summary(model, heat_batch).mean(0))
        for phase in (vegetative[available], flowering[available]):
            period = phase.float().argmax(1)
            a = _period_boundary(field, heat_batch, period)
            heat_values.append(scenario_summary(model, _edited(rows, heat_batch, a, a + 8, "heat")).mean(0))
    # Previous September through the target-year March 1, fixed target weather.
    rainfed = dict(batch)
    rainfed["static"] = batch["static"].clone()
    rainfed["static"][:, -1] = 0.
    rainfed["share"] = torch.zeros_like(batch["share"])
    autumn = field.date_tables["sep"][batch["year"]]
    march = field.date_tables["mar"][batch["year"] + 1]
    baseline_carry = scenario_summary(model, rainfed).mean(0)
    changed_carry = scenario_summary(model, _edited(rows, rainfed, autumn, march, "rain")).mean(0)
    # Cross the device boundary only after all scenario inference.
    names = ("mean_root_wetness", "mean_ndvi", "relative_yield_median", "relative_yield_mean")
    for share, values in zip(("rainfed", "irrigated_share_0.8"), torch.stack(summaries).tolist()):
        rain[share] = {"baseline": dict(zip(names, values[:4])),
                       "rain_minus_30pct": dict(zip(names, values[4:8])),
                       "change": dict(zip(names, values[8:]))}
    if heat_values:
        heat = {name: dict(zip(names, values)) for name, values in
                zip(("baseline", "vegetative_heatwave", "flowering_heatwave"),
                    torch.stack(heat_values).tolist())}
    carry_values = torch.stack([baseline_carry, changed_carry, changed_carry - baseline_carry]).tolist()
    carry = {name: dict(zip(names, values)) for name, values in
             zip(("baseline", "dry_autumn_winter", "change"), carry_values)}
    return {"rows": len(indices), "model": str(model_path), "rainfall": rain, "heat": heat,
            "carry_over": carry,
            "protocol": (
                "Same initial state; no subsequent observed state. P -30% in target active 8-day "
                "periods or preceding Sep-Feb; atmospheric demand held fixed. Heat +6/+3 C for "
                "8 days with GDD/stages recomputed, holding VPD/ET0 fixed: controlled anomalies, "
                "not a physically complete weather front. Tests model consistency, not causality.")}


def empirical_carry_over(predictions, weather_path, meta, output, year_effects=False):
    """Offline conditional association, not an intervention or a neural fit."""
    data = pd.read_parquet(predictions)
    # Census share <= 5% on the all-practice series: predominantly rainfed counties
    # (exact zeros are rare and the explicit NON-IRRIGATED strata are not primary).
    data = data.loc[data.primary & data.crop.eq("corn") & data.irrigated_share.le(.05)
                    & data.year.gt(1981)].copy()
    if len(data) < 10:
        raise ValueError("Insufficient predominantly rainfed corn observations for carry-over regression")
    device = accelerator_device()
    weather = torch.tensor(np.memmap(weather_path, mode="r", dtype="float32", shape=tuple(meta["shape"])),
                           dtype=torch.float16, device=device)
    variables = meta["variables"]
    if variables and isinstance(variables[0], dict):
        variables = [v["name"] for v in variables]
    dates = pd.date_range(meta["start_date"], periods=weather.shape[1])
    first_year, counties = dates[0].year, len(meta["counties"])
    n_years = dates[-1].year - first_year + 1
    winter_mask = torch.tensor((dates.month >= 9) | (dates.month <= 2), device=device)
    summer_mask = torch.tensor((dates.month >= 5) & (dates.month <= 9), device=device)
    annual = torch.tensor(dates.year.to_numpy() - first_year, dtype=torch.long, device=device)
    winter_year = annual + torch.tensor((dates.month >= 9).astype(int), device=device)
    winter = torch.zeros(counties, n_years + 1, device=device).index_add_(
        1, winter_year[winter_mask], weather[:, winter_mask, variables.index("prec")].float())
    summer = torch.zeros(counties, n_years, 3, device=device).index_add_(
        1, annual[summer_mask],
        weather[:, summer_mask][:, :, [variables.index(v) for v in ("prec", "tmax", "et0")]].float())
    days = torch.bincount(annual[summer_mask], minlength=n_years)
    summer[..., 1] /= days[None]
    county = torch.tensor(pd.Index(meta["counties"]).get_indexer(data.adm_id), device=device)
    year = torch.tensor(data.year.to_numpy() - first_year, device=device)
    x = torch.cat([winter[county, year, None], summer[county, year]], -1)
    y = torch.tensor(data.target.to_numpy(dtype=np.float32), device=device)
    count = torch.bincount(county, minlength=counties).float().clamp_min(1)
    cx = torch.zeros(counties, 4, device=device).index_add_(0, county, x) / count[:, None]
    cy = torch.zeros(counties, device=device).index_add_(0, county, y) / count
    x, y = x - cx[county], y - cy[county]
    if year_effects:
        # Alternating projections for an unbalanced county/year panel. This
        # sensitivity analysis, including every group reduction, stays on-device.
        values = torch.cat([x, y[:, None]], -1)
        for _ in range(40):
            for index, groups in ((county, counties), (year, n_years)):
                size = torch.bincount(index, minlength=groups).float().clamp_min(1)
                grouped = torch.zeros(groups, 5, device=device).index_add_(0, index, values)
                values -= (grouped / size[:, None])[index]
        x, y = values[:, :4], values[:, 4]
    scale = x.square().mean(0).sqrt().clamp_min(1e-5)
    matrix = torch.cat([torch.ones(len(x), 1, device=device), x / scale], -1)
    gram = matrix.T @ matrix / len(x) + torch.eye(5, device=device) * 1e-6
    # Tiny positive-definite system: accelerator-only Gauss-Jordan avoids
    # unsupported platform-specific LAPACK paths, not a CPU fallback.
    augmented = torch.cat([gram, torch.eye(5, device=device)], -1)
    for pivot in range(5):
        row = augmented[pivot] / augmented[pivot, pivot]
        factor = augmented[:, pivot].clone()
        augmented = augmented - factor[:, None] * row[None]
        augmented[pivot] = row
    inverse = augmented[:, 5:]
    beta = inverse @ (matrix.T @ y / len(x))
    residual = y - matrix @ beta
    clustered = torch.zeros(counties, 5, device=device).index_add_(0, county, matrix * residual[:, None])
    bread = inverse / len(x)
    covariance = bread @ (clustered.T @ clustered) @ bread
    coefficient = beta[1] / scale[0] * 100
    se = covariance[1, 1].clamp_min(0).sqrt() / scale[0] * 100
    estimate, low, high = torch.stack([coefficient, coefficient - 1.96 * se, coefficient + 1.96 * se]).tolist()
    result = {
        "rows": len(data), "counties": data.adm_id.nunique(),
        "maximum_irrigated_share": .05, "year_fixed_effects": year_effects,
        "log_yield_change_per_100mm_preceding_winter_rain": estimate,
        "clustered_95_ci": [low, high],
        "interpretation": ("Within-county observational association controlling May-Sep P/Tmax/ET0"
                           + (" and year fixed effects" if year_effects else "")
                           + "; not causal."),
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=256)
    args = parser.parse_args()
    result = what_if(args.model, args.features, args.cases)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
