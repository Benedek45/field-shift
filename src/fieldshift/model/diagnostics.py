"""Source-audited phenology checks and out-of-year corn tree permutation ranks."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from fieldshift.features.traits import load_traits
from fieldshift.model.expanded import fold_masks, target_for_fold
from fieldshift.model.expanded_data import apply_climatology, feature_columns
from fieldshift.model.recursive_data import ROOT
from fieldshift.model.recursive_train import FEATURES


def crossing_date(frame, percentage=50):
    """Monotone weekly cumulative progress, interpolating only a bracketed crossing."""
    weekly = frame.groupby("week_ending").value.mean().sort_index()
    if len(weekly) < 2:
        return pd.NaT
    values = np.maximum.accumulate(weekly.to_numpy(dtype=float))
    if values.min() > percentage or values.max() < percentage:
        return pd.NaT
    high = int(np.searchsorted(values, percentage))
    if values[high] == percentage:
        return pd.Timestamp(weekly.index[high])
    low = high - 1
    fraction = (percentage - values[low]) / (values[high] - values[low])
    return pd.Timestamp(weekly.index[low]) + fraction * (weekly.index[high] - weekly.index[low])


def progress_dates(path="data/processed/nass_crop_progress.parquet"):
    source = pd.read_parquet(path)
    source["class_desc"] = source.class_desc.str.strip()
    source["week_ending"] = pd.to_datetime(source.week_ending)
    source = source.loc[source.value.between(0, 100)].copy()
    mappings = {
        "corn": ("CORN", "ALL CLASSES", "silking"),
        "soybean": ("SOYBEANS", "ALL CLASSES", "blooming"),
        "winter_wheat": ("WHEAT", "WINTER", "headed"),
        "spring_wheat": ("WHEAT", "SPRING, (EXCL DURUM)", "headed"),
        "durum": ("WHEAT", "SPRING, DURUM", "headed"),
        "barley": ("BARLEY", "ALL CLASSES", "headed"),
        "oats": ("OATS", "ALL CLASSES", "headed"),
        "sorghum": ("SORGHUM", "ALL CLASSES", "headed"),
        "rice": ("RICE", "ALL CLASSES", "headed"),
        "rye": ("RYE", "ALL CLASSES", "headed"),
        "flaxseed": ("FLAXSEED", "ALL CLASSES", "blooming"),
        "peanuts": ("PEANUTS", "ALL CLASSES", "blooming"),
        "potatoes": ("POTATOES", "ALL CLASSES", None),
        "sugarbeets": ("SUGARBEETS", "ALL CLASSES", None),
        "cotton": ("COTTON", "UPLAND", None),
        "sunflower": ("SUNFLOWER", "ALL CLASSES", "blooming"),
        "dry_beans": ("BEANS", "DRY EDIBLE, INCL CHICKPEAS", "blooming"),
    }
    records = []
    for crop, (commodity, klass, flowering) in mappings.items():
        selected = source.loc[source.commodity.eq(commodity) & source.class_desc.eq(klass)]
        for kind, stage in (("sowing", "planted"), ("flowering", flowering), ("maturity", "mature")):
            if stage is None:
                continue
            matched = selected.loc[selected.stage.eq(stage)].copy()
            # Planting in autumn is for the following harvest year. Use actual
            # weekly dates, not the bulk's sometimes forward-labelled crop year.
            matched["harvest_year"] = matched.week_ending.dt.year + int(
                crop in {"winter_wheat", "rye"} and kind == "sowing")
            for (state, year), group in matched.groupby(["state", "harvest_year"]):
                date = crossing_date(group)
                if pd.notna(date):
                    records.append({"crop": crop, "state_alpha": state, "year": year,
                                    "kind": kind, "date": date, "source_stage": stage,
                                    "proxy": stage == "headed"})
    return pd.DataFrame(records)


def phenology_check(features=FEATURES, output=None):
    data = pd.read_parquet(features)
    data = data.loc[data.primary].copy()
    origin = pd.Timestamp("1970-01-01").toordinal()
    estimates = []
    for kind in ("sowing", "flowering", "maturity"):
        ordinal = data.season_start_ordinal.copy()
        if kind == "flowering":
            ordinal += data.establishment__length_days + data.vegetative__length_days
        if kind == "maturity":
            ordinal = data.season_end_ordinal.where(data.maturity_reached.eq(1))
        prediction = data[["crop", "state_alpha", "year", "harvested_acres"]].copy()
        prediction["ordinal"] = ordinal
        prediction = prediction.loc[prediction.ordinal.notna() & prediction.harvested_acres.gt(0)]
        prediction["weighted"] = prediction.ordinal * prediction.harvested_acres
        summed = prediction.groupby(["crop", "state_alpha", "year"])[["weighted", "harvested_acres"]].sum()
        summed["ordinal"] = summed.weighted / summed.harvested_acres
        summed["kind"] = kind
        estimates.append(summed.reset_index())
    predicted = pd.concat(estimates, ignore_index=True)
    observed = progress_dates()
    paired = predicted.merge(observed, on=["crop", "state_alpha", "year", "kind"], validate="one_to_one")
    paired["predicted_date"] = pd.to_datetime(paired.ordinal - origin, unit="D")
    paired["error_days"] = (paired.predicted_date - paired.date).dt.total_seconds() / 86400
    result = []
    for (crop, kind), group in paired.groupby(["crop", "kind"]):
        error = group.error_days.to_numpy()
        result.append({"crop": crop, "stage": kind, "state_years": len(group),
                       "bias_days": float(error.mean()), "mae_days": float(np.abs(error).mean()),
                       "median_absolute_days": float(np.median(np.abs(error))),
                       "proxy": bool(group.proxy.any())})
    output = output or FEATURES.parents[1] / "phenology_check.json"
    output.write_text(json.dumps({
        "comparison": "Area-weighted county thermal estimate vs state 50%-progress date; heading is only a flowering proxy.",
        "scope": "Diagnostic only: no thresholds or planting drivers were tuned using these records.",
        "results": result,
    }, indent=2))
    paired.to_parquet(output.with_suffix(".parquet"), index=False)
    management = observed.loc[observed.kind.eq("sowing"), ["crop", "state_alpha", "year", "date"]]
    # Corn silage shares the parent's planting operation, not its harvest date.
    management = pd.concat([management, management.loc[management.crop.eq("corn")].assign(crop="corn_silage")])
    directory = ROOT
    directory.mkdir(exist_ok=True)
    management.to_parquet(directory / "planting_dates.parquet", index=False)
    data[["adm_id", "state_alpha"]].drop_duplicates().to_parquet(directory / "county_states.parquet", index=False)
    return result


def recursive_phenology_check(field, features=FEATURES, output=None, batch_size=512):
    """Daily thermal boundaries from the actual R3 driver engine, not 8-day rounding."""
    import torch

    data = pd.read_parquet(features)
    data = data.loc[data.primary & data.year.gt(1981)].reset_index(drop=True)
    county = torch.tensor(pd.Index(field.meta["counties"]).get_indexer(data.adm_id), device=field.device)
    crop = torch.tensor(pd.Index(field.crops).get_indexer(data.crop), device=field.device)
    year = torch.tensor(data.year.to_numpy() - 1981, device=field.device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            selected = slice(start, start + batch_size)
            *_, dates = field.drivers(county[selected], crop[selected], year[selected], years=2,
                                      return_dates=True)
            predictions.append(dates)
        predicted = torch.cat(predictions).tolist()
    predicted = np.asarray(predicted, dtype=float)
    origin = pd.Timestamp(field.meta["start_date"]).toordinal()
    tables = []
    for index, kind in enumerate(("sowing", "flowering", "maturity")):
        table = data[["crop", "state_alpha", "year", "harvested_acres"]].copy()
        table["ordinal"] = predicted[:, index] + origin
        table = table.loc[table.ordinal.notna() & table.harvested_acres.gt(0)]
        table["weighted"] = table.ordinal * table.harvested_acres
        grouped = table.groupby(["crop", "state_alpha", "year"])[["weighted", "harvested_acres"]].sum()
        grouped["ordinal"] = grouped.weighted / grouped.harvested_acres
        grouped["kind"] = kind
        tables.append(grouped.reset_index())
    paired = pd.concat(tables).merge(progress_dates(), on=["crop", "state_alpha", "year", "kind"],
                                      validate="one_to_one")
    observed = ((pd.to_datetime(paired.date) - pd.Timestamp("1970-01-01")).dt.total_seconds()
                / 86400 + pd.Timestamp("1970-01-01").toordinal())
    paired["error_days"] = paired.ordinal - observed
    result = []
    for (crop_name, kind), group in paired.groupby(["crop", "kind"]):
        result.append({"crop": crop_name, "stage": kind, "state_years": len(group),
                       "bias_days": float(group.error_days.mean()),
                       "mae_days": float(group.error_days.abs().mean()),
                       "proxy": bool(group.proxy.any())})
    destination = Path(output or "data/processed/yield_expanded/recursive_phenology_check.json")
    payload = {"scope": "R3 management-informed sowing and daily thermal boundaries; given observed "
                        "planting dates are not an independently predicted sowing-date validation.",
               "maturity_reached_fraction": {
                    str(name): (None if load_traits(str(name))["perennial"].value else
                                float(np.isfinite(predicted[:, 2])[indices].mean()))
                   for name, indices in data.groupby("crop").indices.items()},
               "results": result}
    destination.write_text(json.dumps(payload, indent=2))
    paired.to_parquet(destination.with_suffix(".parquet"), index=False)
    return payload


def corn_permutation(features=FEATURES, repeats=3, output=None):
    data = pd.read_parquet(features)
    columns = feature_columns(data)
    individual = {name: [] for name in columns}
    joint = {
        "flowering_all_heat_aliases": [c for c in columns if c.startswith("flowering__heat_dd")],
        "flowering_water_supply_and_demand": [c for c in columns if c.startswith("flowering__")
                                             and any(s in c for s in ("rsm", "prec", "et0", "cwb", "water_deficit", "vpd"))],
        "all_stage_heat_aliases": [c for c in columns if "__heat_dd" in c],
    }
    grouped = {name: [] for name in joint}
    rng = np.random.default_rng(20261114)
    for fold, (_, test_index) in enumerate(GroupKFold(5).split(data, groups=data.year)):
        years = sorted(data.iloc[test_index].year.unique().tolist())
        fit, _, test, fit_years, _ = fold_masks(data, years)
        target, _ = target_for_fold(data, fit)
        prepared = apply_climatology(data, fit_years, features.parent)
        test = test & data.crop.eq("corn").to_numpy() & data.primary.to_numpy()
        x, z = prepared.loc[fit, columns].to_numpy(float), prepared.loc[test, columns].to_numpy(float)
        count = np.isfinite(x).sum(0).clip(1)
        center = np.nansum(x, 0) / count
        scale = np.sqrt(np.nansum((x - center) ** 2, 0) / count)
        scale[scale < 1e-5] = 1.
        x, z = (x - center) / scale, (z - center) / scale
        weights = 1 / data.loc[fit].groupby(["crop", "adm_id", "year"]).year.transform("size")
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=23, learning_rate=.05,
                                             min_samples_leaf=50, l2_regularization=15,
                                             early_stopping=False, random_state=2026)
        y = target[test]
        with threadpool_limits(limits=6):
            model.fit(x, target[fit], sample_weight=weights)
            baseline = np.mean((model.predict(z) - y) ** 2)
            variance = np.var(y)
            for name, indices, collector in (
                [(name, [j], individual) for j, name in enumerate(columns)]
                + [(name, [columns.index(c) for c in values], grouped) for name, values in joint.items()]
            ):
                for _ in range(repeats):
                    edited = z.copy()
                    edited[:, indices] = z[rng.permutation(len(z))][:, indices]
                    increase = np.mean((model.predict(edited) - y) ** 2) - baseline
                    collector[name].append({"fold": fold, "n": len(z), "mse_increase": float(increase),
                                            "r2_drop": float(increase / variance)})
        print(f"Corn permutation completed fold {fold}", flush=True)
    def ranking(collection):
        return sorted([{"feature": name, "mean_r2_drop": float(np.mean([v["r2_drop"] for v in values])),
                        "mean_mse_increase": float(np.mean([v["mse_increase"] for v in values])),
                        "fold_repeat_values": values} for name, values in collection.items()],
                      key=lambda record: -record["mean_mse_increase"])
    output = output or FEATURES.parents[1] / "corn_permutation.json"
    result = {"method": "R2-D stage tree; every outer fold, corn ALL-practice test rows, 3 permutations/feature.",
              "caution": "Associational marginal permutation; correlated aliases can dilute importance. Groups jointly shuffled.",
              "individual": ranking(individual), "groups": ranking(grouped)}
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corn", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.progress:
        print(json.dumps(phenology_check(), indent=2))
    if args.corn:
        print(json.dumps(corn_permutation()["individual"][:15], indent=2))
