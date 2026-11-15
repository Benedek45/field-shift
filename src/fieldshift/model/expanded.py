"""Multi-crop year/crop held-out evaluation with calibrated neural ensembles."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from fieldshift.features.traits import numeric_traits
from fieldshift.model.expanded_data import (
    apply_climatology,
    build_expanded,
    feature_columns,
)
from fieldshift.model.probabilistic import (
    ResidentRows,
    ResidentWeather,
    distribution,
    load_response,
    recalibrate,
    save_response,
    train_response,
)
from fieldshift.model.trend import YieldTrend

MODELS = ("trend", "season_tree", "stage_tree", "mlp", "sequence", "recursive", "teacher_forced")
DROUGHT_STATES = ("IA", "IL", "IN", "KS", "MO", "NE", "SD")


def score(g, model, area=False):
    actual, predicted = g.target.to_numpy(), g[model].to_numpy()
    w = g.harvested_acres.fillna(0).to_numpy() if area else np.ones(len(g))
    keep = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(w) & (w > 0)
    actual, predicted, w = actual[keep], predicted[keep], w[keep]
    if not len(w):
        return {"n": 0}
    w = w / w.sum()
    error = predicted - actual
    variance = (w * (actual - (w * actual).sum()) ** 2).sum()
    result = {
        "n": len(actual),
        "r2": float(1 - (w * error ** 2).sum() / variance) if variance > 0 else None,
        "rmse_log": float(np.sqrt((w * error ** 2).sum())),
        "mape": float(100 * (w * np.abs(np.expm1(error))).sum()),
        "observed_loss_pct": float(100 * (w * (1 - np.exp(actual))).sum()),
        "predicted_loss_pct": float(100 * (w * (1 - np.exp(predicted))).sum()),
    }
    if model + "_std" in g:
        std = g[model + "_std"].to_numpy()[keep]
        result["coverage_1_std"] = float((w * (np.abs(error) <= std)).sum())
        result["coverage_2_std"] = float((w * (np.abs(error) <= 2 * std)).sum())
        result["mean_std_log"] = float((w * std).sum())
    return result


def summary(predictions):
    results = {}
    for crop, g in predictions.groupby("crop"):
        results[crop] = {
            model: {"equal_county": score(g, model), "area_weighted": score(g, model, True)}
            for model in MODELS if model in g
        }
    macro = {}
    for model in MODELS:
        values = [r[model]["equal_county"]["r2"] for r in results.values()
                  if model in r and r[model]["equal_county"].get("r2") is not None]
        if values:
            macro[model] = float(np.mean(values))
    overall = {
        model: {"equal_county": score(predictions, model),
                "area_weighted": score(predictions, model, True)}
        for model in MODELS if model in predictions
    }
    return {"per_crop": results, "macro_r2": macro, "overall": overall}


def report_tables(predictions, held):
    primary = predictions.loc[predictions.primary]
    rainfed = predictions.loc[predictions.irrigated_share.eq(0)].sort_values("primary")
    rainfed = rainfed.drop_duplicates(["crop", "adm_id", "year"])
    report = {
        "within_crop": summary(primary),
        "all_strata": summary(predictions),
        "rainfed_share_zero": summary(rainfed),
        "held_crop": summary(held.loc[held.primary]) if len(held) else {},
        "drought": {},
    }
    for year in (1983, 1988, 2012):
        g = primary.loc[primary.year.eq(year)]
        report["drought"][str(year)] = {
            "all": summary(g),
            "drought_states": summary(g.loc[g.state_alpha.isin(DROUGHT_STATES)]),
        }
    return report


def crop_groups(crops):
    groups = []
    used = set()
    for family in (("corn", "corn_silage"), ("winter_wheat", "spring_wheat", "durum")):
        present = sorted(set(crops) & set(family))
        if present:
            groups.append(present)
            used.update(present)
    groups.extend([[c] for c in sorted(set(crops) - used)])
    return groups


def fold_masks(data, test_years):
    train_years = sorted(set(data.year) - set(test_years))
    calibration_years = train_years[::5]
    fit_years = sorted(set(train_years) - set(calibration_years))
    return (data.year.isin(fit_years).to_numpy(),
            data.year.isin(calibration_years).to_numpy(),
            data.year.isin(test_years).to_numpy(), fit_years, calibration_years)


def target_for_fold(data, fit_mask):
    # Stratified baselines do not conflate an irrigated yield trend with rainfed.
    trend_data = data.assign(crop=data.series_id)
    trend = YieldTrend().fit(trend_data.loc[fit_mask])
    return trend.transform(trend_data), trend


def tree_predict(data, train_mask, test_mask, target, columns):
    train, test = data.loc[train_mask, columns], data.loc[test_mask, columns]
    x, z = train.to_numpy(dtype=float), test.to_numpy(dtype=float)
    count = np.isfinite(x).sum(axis=0).clip(1)
    center = np.nansum(x, axis=0) / count
    scale = np.sqrt(np.nansum((x - center) ** 2, axis=0) / count)
    scale[scale < 1e-5] = 1.
    weights = 1 / data.loc[train_mask].groupby(["crop", "adm_id", "year"]).year.transform("size")
    model = HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=23, learning_rate=.05, min_samples_leaf=50,
        l2_regularization=15, early_stopping=False, random_state=2026,
    )
    with threadpool_limits(limits=6):
        model.fit((x - center) / scale, target[train_mask], sample_weight=weights)
        return model.predict((z - center) / scale)


def fit_predict(data, rows, masks, columns, epochs, weather=None, sequence=False, trees=True):
    train_mask, calibration_mask, test_mask = masks
    index = rows.indices(test_mask)
    result = data.loc[test_mask, [
        "crop", "adm_id", "year", "series_id", "state_alpha", "harvested_acres",
        "irrigated_share", "irrigation_source", "primary", "prodn_practice_desc",
    ]].copy()
    result["target"] = np.asarray(rows.y[rows.indices(test_mask)].tolist())
    result["trend"] = 0.
    timings = {}
    if trees:
        target = np.asarray(rows.y.tolist())
        season = [c for c in columns if c.startswith("season__") or c in {"irrigated_share", "irrigation_missing"}]
        result["season_tree"] = tree_predict(data, train_mask, test_mask, target, season)
        result["stage_tree"] = tree_predict(data, train_mask, test_mask, target, columns)
    trait_indices = [i for i, c in enumerate(columns) if c.startswith("trait__")]
    for name in (["mlp", "sequence"] if sequence else ["mlp"]):
        net, timing = train_response(
            rows, rows.indices(train_mask), trait_indices, epochs=epochs,
            weather=weather, sequence=name == "sequence"
        )
        calibration_index = rows.indices(calibration_mask & data.primary.to_numpy())
        calibration = recalibrate(net, rows, calibration_index, weather)
        predicted = np.asarray(distribution(net, rows, index, weather).tolist())
        result[name], result[name + "_std"] = predicted
        timings[name] = timing | {"calibration": calibration}
        del net
    return result, timings


def benchmark(data, path, meta, root, epochs=2):
    columns = feature_columns(data)
    years = sorted(data.year.unique())[-4:]
    fit, _calibration, test, fit_years, _ = fold_masks(data, years)
    prepared = apply_climatology(data, fit_years, root)
    target, _ = target_for_fold(data, fit)
    rows = ResidentRows(prepared, columns, target)
    weather = ResidentWeather(path, meta)
    traits = [i for i, c in enumerate(columns) if c.startswith("trait__")]
    result = {}
    for sequence in (False, True):
        net, timing = train_response(rows, rows.indices(fit), traits, epochs=epochs,
                                     weather=weather, sequence=sequence)
        output = distribution(net, rows, rows.indices(test), weather)
        result["sequence" if sequence else "mlp"] = timing | {"test_rows": output.shape[1]}
        del net, output
    (root / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def previous_comparison(predictions):
    path = Path("data/processed/yield_stress/predictions.parquet")
    if not path.exists():
        return {}
    old = pd.read_parquet(path)
    keys = ["crop", "adm_id", "year", "prodn_practice_desc"]
    names = ["target", "trend", "season_tree", "stage_tree", "mlp"]
    old = old[keys + names].rename(columns={c: c + "_old" for c in names})
    paired = predictions.merge(old, on=keys, validate="one_to_one")
    report = {}
    for crop, g in paired.groupby("crop"):
        mapped = g.copy()
        offset = g.target_old - g.target
        mapped["target"] = g.target_old
        report[crop] = {}
        for model in MODELS:
            if model in g:
                mapped[model] = g[model] + offset
                report[crop][model] = score(mapped, model)
        drought = mapped.loc[mapped.year.eq(2012) & mapped.state_alpha.isin(DROUGHT_STATES)]
        report[crop]["drought_2012_old_normal"] = {
            model: score(drought, model) for model in MODELS if model in drought
        } if len(drought) else {}
    return {"note": "Matched county/crop/practice/year. New predictions recentered onto the old county normal using target_old - target_new; spread unchanged.",
            "per_crop": report}


def evaluate(source="cybench", folds=5, epochs=40, sequence=False, held_crops=True,
             min_year=None, max_year=None, cohort=None):
    started = time.monotonic()
    data, path, meta, root = build_expanded(source)
    if min_year is not None:
        data = data.loc[data.year.ge(min_year)]
    if max_year is not None:
        data = data.loc[data.year.le(max_year)]
    if cohort:
        if source == cohort:
            raise ValueError("Specify a different weather archive for a common-cohort comparison")
        reference, _, _, _ = build_expanded(cohort)
        keys = ["crop", "adm_id", "year", "prodn_practice_desc"]
        data = data.merge(reference[keys], on=keys, how="inner", validate="one_to_one")
    data = data.reset_index(drop=True)
    columns = feature_columns(data)
    weather = ResidentWeather(path, meta) if sequence else None
    groups = crop_groups(data.crop.unique())
    ordinary, transferred, timings, design = [], [], [], []
    out = root / f"evaluation_{folds}fold_{epochs}epoch_seq{int(sequence)}_{data.year.min()}_{data.year.max()}{'_cohort_' + cohort if cohort else ''}"
    out.mkdir(exist_ok=True)
    for fold, (_, test_index) in enumerate(GroupKFold(n_splits=folds).split(data, groups=data.year)):
        test_years = sorted(data.iloc[test_index].year.unique().tolist())
        fit, calibration, test, fit_years, calibration_years = fold_masks(data, test_years)
        prepared = apply_climatology(data, fit_years, root)
        target, _ = target_for_fold(data, fit)
        rows = ResidentRows(prepared, columns, target)
        design.append({"test_years": test_years, "fit_years": fit_years, "calibration_years": calibration_years})
        cached = out / f"fold_{fold}.parquet"
        if cached.exists():
            ordinary.append(pd.read_parquet(cached))
        else:
            result, timing = fit_predict(prepared, rows, (fit, calibration, test), columns,
                                         epochs, weather, sequence)
            result.to_parquet(cached, index=False)
            (out / f"fold_{fold}_timing.json").write_text(json.dumps(timing, indent=2))
            ordinary.append(result)
            timings.append(timing)
            print(f"{source} fold {fold + 1}: held years {test_years}", flush=True)
        if held_crops:
            for number, group in enumerate(groups):
                if number % folds != fold:
                    continue
                held_path = out / f"held_{'_'.join(group)}.parquet"
                if held_path.exists():
                    transferred.append(pd.read_parquet(held_path))
                    continue
                keep = ~data.crop.isin(group).to_numpy()
                masks = (fit & keep, calibration & keep, test & ~keep)
                result, timing = fit_predict(prepared, rows, masks, columns, epochs, weather, sequence)
                result.to_parquet(held_path, index=False)
                (out / f"held_{'_'.join(group)}_timing.json").write_text(json.dumps(timing, indent=2))
                transferred.append(result)
                print(f"  Held family/crop {group}: {len(result):,} observations", flush=True)
        del rows, prepared
    predictions = pd.concat(ordinary, ignore_index=True)
    held = pd.concat(transferred, ignore_index=True) if transferred else pd.DataFrame()
    predictions.to_parquet(out / "predictions.parquet", index=False)
    held.to_parquet(out / "held_crop_predictions.parquet", index=False)
    report = report_tables(predictions, held)
    report["paired_previous_cohort"] = previous_comparison(predictions)
    report["design"] = {
        "source": source, "cohort": cohort, "rows": len(data), "crops": sorted(data.crop.unique()),
        "folds": design, "held_crop_groups": groups, "epochs": epochs, "seeds": 5,
        "runtime_seconds": time.monotonic() - started, "feature_columns": columns,
        "point_estimate": "Mean log yield ratio; spread is calibrated predictive log SD.",
        "irrigation": "Census interpolated plus explicit strata; missing is not zero.",
        "climatology": "Fit-calendar-year-only county/month climatology, weighted by each stage's actual calendar-month days.",
        "primary_metrics": "ALL-practice county yields only; explicit strata additionally reported in all_strata and rainfed subset.",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    # Deployment keeps independent calibration years; do not refit on them.
    fit, calibration, _, fit_years, calibration_years = fold_masks(data, [])
    prepared = apply_climatology(data, fit_years, root)
    target, _ = target_for_fold(data, fit)
    rows = ResidentRows(prepared, columns, target)
    traits = [i for i, c in enumerate(columns) if c.startswith("trait__")]
    net, timing = train_response(rows, rows.indices(fit), traits, epochs=epochs)
    calibration_record = recalibrate(net, rows, rows.indices(calibration & data.primary.to_numpy()))
    suffix = f"_cohort_{cohort}" if cohort else ""
    model_path = Path(f"data/processed/models/yield_response_v2_{source}{suffix}.pt")
    save_response(net, columns, model_path, {
        "source": source, "calibration_years": calibration_years,
        "fit_years": fit_years, "calibration": calibration_record,
        "timing": timing, "evaluation": str(out / "metrics.json"),
    })
    if sequence:
        net, timing = train_response(rows, rows.indices(fit), traits, epochs=epochs,
                                     weather=weather, sequence=True)
        calibration_record = recalibrate(net, rows, rows.indices(calibration & data.primary.to_numpy()), weather)
        save_response(net, columns, model_path.with_stem(model_path.stem + "_sequence"), {
            "source": source, "calibration_years": calibration_years,
            "fit_years": fit_years, "calibration": calibration_record, "timing": timing,
            "weather_path": str(path), "weather_meta": meta,
            "evaluation": str(out / "metrics.json"),
        })
    print(f"Saved {out / 'metrics.json'} and {model_path}", flush=True)
    return report


def predict_calibrated_relative_yield(stage_features, traits, soil, irrigated_share,
                                      model_path="data/processed/models/yield_response_v2_power.pt"):
    """Return arithmetic mean and predictive SD of the calibrated lognormal ratio.

    Caller supplies physical stage summaries and irrigation (None if unknown).
    This deployment interface uses the stage MLP, not a sequence model requiring
    archive-specific county IDs. Calibrated log-space moments are also available
    through the saved ResponseEnsemble.distribution method.
    """
    import torch

    from fieldshift.model.neural import mps_device

    if irrigated_share is not None and (
        not np.isfinite(irrigated_share) or not 0 <= irrigated_share <= 1
    ):
        raise ValueError("irrigated_share must be None or a finite fraction in [0, 1]")
    net, columns, _ = load_response(model_path)
    values = dict(stage_features) | dict(soil)
    values.update({f"trait__{k}": v for k, v in numeric_traits(traits).items()})
    values["irrigated_share"] = np.nan if irrigated_share is None else irrigated_share
    values["irrigation_missing"] = float(irrigated_share is None)
    missing = set(columns) - values.keys()
    if missing:
        raise ValueError(f"Missing stage/soil features: {sorted(missing)}")
    x = torch.tensor([[values[c] for c in columns]], dtype=torch.float32, device=mps_device())
    with torch.inference_mode():
        mean, std = net.distribution(x)
        variance = std.square()
        ratio_mean = (mean + variance / 2).exp()
        ratio_std = (mean + variance / 2 + variance.expm1().log() / 2).exp()
    return float(ratio_mean.item()), float(ratio_std.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", choices=["cybench", "power", "power_matched"], default="cybench")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--no-held-crops", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--cohort", choices=["cybench", "power", "power_matched"])
    args = parser.parse_args()
    if args.benchmark or args.build_only:
        data, path, meta, root = build_expanded(args.weather)
        if args.benchmark:
            benchmark(data, path, meta, root)
        return
    report = evaluate(args.weather, args.folds, args.epochs, args.sequence,
                      not args.no_held_crops, args.min_year, args.max_year, args.cohort)
    print(json.dumps({"within_macro_r2": report["within_crop"]["macro_r2"],
                      "held_macro_r2": report["held_crop"].get("macro_r2")}, indent=2))


if __name__ == "__main__":
    main()
