"""Independent drought-monitor and insurance checks; neither is a training input."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.model.expanded import target_for_fold
from fieldshift.model.recursive import load_recursive
from fieldshift.model.recursive_data import FieldData
from fieldshift.model.recursive_train import FEATURES, FieldRows, run_batch


def correlation(x, y, mask):
    x, y = x[mask].float(), y[mask].float()
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / (x.square().sum() * y.square().sum()).sqrt().clamp_min(1e-8)


@torch.inference_mode()
def external_checks(model_path, output, features=FEATURES, year=2012):
    model, metadata = load_recursive(model_path)
    if year not in metadata["test_years"]:
        raise ValueError(f"{year} must be an outer test year, not a fitted/deployment year")
    field = FieldData()
    if field.static_names != metadata["static_names"]:
        raise ValueError("Use the model's frozen statics")
    frame = pd.read_parquet(features)
    target, _ = target_for_fold(frame, frame.year.isin(metadata["fit_years"]).to_numpy())
    rows = FieldRows(frame, field, target)
    choose = frame.primary & frame.crop.eq("corn") & frame.year.eq(year) & frame.adm_id.str.startswith("US-19-")
    chosen = frame.loc[choose].copy()
    if chosen.empty:
        raise ValueError("No held-year Iowa corn rows")
    climatology = field.climatology(metadata["fit_years"])
    batch = rows.batch(rows.indices(choose.to_numpy()), climatology, years=2)
    batch["static"] = batch["static"].clone()
    batch["static"][:, -1] = 0.
    batch["share"] = torch.zeros_like(batch["share"])
    rollout, prediction = run_batch(model, batch)
    baseline = (prediction[0] + prediction[1].square() / 2).exp()
    before, after = rollout["states"][:, :, :-1, 0].mean(0), rollout["states"][:, :, 1:, 0].mean(0)
    wetness = (before + after) / 2
    positions = batch["positions"][:, :-1]
    normal = climatology[batch["county"][:, None], positions % 46, 0]
    anomaly = wetness - normal
    usdm_path = Path("data/processed/usdm/usdm_8day.f32")
    usdm_meta = json.loads(usdm_path.with_suffix(".json").read_text())
    if usdm_meta["counties"] != field.meta["counties"] or usdm_meta["shape"][:2] != [field.n_counties, field.n_steps]:
        raise ValueError("USDM county/step alignment differs from POWER")
    usdm = torch.tensor(np.memmap(usdm_path, mode="r", dtype=np.float32,
                                  shape=tuple(usdm_meta["shape"])), device=field.device)
    severity = usdm[batch["county"][:, None], positions, usdm_meta["names"].index("d2")]
    active = (batch["season"] > 0) & torch.isfinite(severity)
    relationship = correlation(anomaly, severity, active)
    # Restoration to monthly fit-year daily means, not +30% of a near-zero July
    # rainfall total. GDD and stages are recalculated after every edit.
    dates = pd.date_range(field.meta["start_date"], periods=field.daily.shape[1])
    fitted = torch.tensor(dates.year.isin(metadata["fit_years"]), device=field.device)
    month = torch.tensor(dates.month.to_numpy(), device=field.device)
    counties, crop, harvest = batch["county"], batch["crop"], batch["year"]
    changes = []
    start_ordinal = dates[0].toordinal()
    for m in range(5, 10):
        historical = fitted & (month == m)
        reference = field.daily[counties][:, historical][:, :, [
            field.variables[k] for k in ("prec", "tmin", "tmax")]].float().mean(1)
        start = pd.Timestamp(year, m, 1).toordinal() - start_ordinal
        stop = (pd.Timestamp(year, m, 1) + pd.offsets.MonthBegin()).toordinal() - start_ordinal

        def restore(raw, day, reference=reference, start=start, stop=stop):
            selected = (day >= start) & (day < stop)
            for j, name in enumerate(("prec", "tmin", "tmax")):
                original = raw[..., field.variables[name]]
                changed = (torch.maximum(original, reference[:, j, None]) if name == "prec"
                           else torch.minimum(original, reference[:, j, None]))
                raw[..., field.variables[name]] = torch.where(selected, changed, original)
            original = raw[..., field.variables["tavg"]]
            changed = (raw[..., field.variables["tmin"]] + raw[..., field.variables["tmax"]]) / 2
            raw[..., field.variables["tavg"]] = torch.where(selected, changed, original)
            return raw

        drivers, season, _ = field.drivers(counties, crop, harvest, years=2, daily_transform=restore)
        _, distribution = run_batch(model, batch | {"drivers": drivers, "season": season})
        edited = (distribution[0] + distribution[1].square() / 2).exp()
        changes.append((edited - baseline).mean())
    # Transfer only final diagnostic numbers, not per-batch training/inference data.
    usdm_r, *restored = torch.stack([relationship, *changes]).tolist()
    insurance = pd.read_parquet("data/processed/rma_cause_share.parquet")
    insurance = insurance.loc[insurance.year.eq(year) & insurance.commodity_key.eq("CORN")
                               & insurance.adm_id.isin(chosen.adm_id)]
    planted = insurance.groupby("adm_id").planted_acres.max().dropna()
    drought = insurance.loc[insurance.cause_group.eq("drought")]
    damage = drought.groupby(["adm_id", "month"]).damage_share_of_planted.sum()
    damage = damage.reindex(pd.MultiIndex.from_product([planted.index, range(5, 10)]), fill_value=0)
    table = damage.unstack()
    weighted = table.mul(planted, axis=0).sum() / planted.sum()
    annual_indemnity = insurance.indemnity.sum()
    monthly_indemnity = insurance.groupby("month").indemnity.sum()
    result = {
        "model": str(model_path), "year": year, "counties": len(chosen),
        "usdm_root_anomaly_vs_d2_area_r": usdm_r,
        "months": {str(m): {"restored_weather_relative_yield_mean_change": change,
                             "rma_drought_damage_share_of_planted": float(weighted[m]),
                             "rma_all_cause_indemnity_fraction": float(monthly_indemnity.get(m, 0)
                                                                       / annual_indemnity)}
                   for m, change in zip(range(5, 10), restored)},
        "limitations": (
            "USDM county area is not crop-specific. Monthly controlled restoration keeps VPD/ET0 "
            "fixed; it is not a full weather-front intervention. RMA damage/indemnity timing is "
            "an external observational check, not causal attribution. Both diagnostics use held-year "
            "Iowa corn with common initial states and rainfed management."),
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result
