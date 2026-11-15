"""Build, evaluate and save interpretable yield stress models.

uv run python -m fieldshift.model.yield_stress
"""

import argparse
import json
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from fieldshift.features.traits import CROPS, numeric_traits
from fieldshift.model.dataset import ROOT, build_features
from fieldshift.model.neural import (
    load_ensemble,
    predict_members,
    save_ensemble,
    train_ensemble,
)
from fieldshift.model.trend import YieldTrend

MODEL_PATH = Path("data/processed/models/yield_stress_mlp.pt")
DROUGHT_STATES = ("IA", "IL", "IN", "KS", "MO", "NE", "SD")
EXCLUDE = {
    "crop", "adm_id", "year", "yield_t_ha", "state_alpha", "prodn_practice_desc",
    "irrigated_share", "irrigation_unknown",
    "season_start_ordinal", "season_end_ordinal",
}


def feature_columns(data):
    return sorted(set(data.columns) - EXCLUDE)


def _tree_predictions(train, test, target, columns):
    x = train[columns].to_numpy(dtype=np.float64)
    z = test[columns].to_numpy(dtype=np.float64)
    count = np.isfinite(x).sum(axis=0).clip(1)
    center = np.nansum(x, axis=0) / count
    scale = np.sqrt(np.nansum((x - center) ** 2, axis=0) / count)
    scale[scale < 1e-5] = 1
    # HGB handles remaining NaNs natively.
    x, z = (x - center) / scale, (z - center) / scale
    model = HistGradientBoostingRegressor(
        max_iter=150, max_leaf_nodes=15, learning_rate=.06,
        min_samples_leaf=40, l2_regularization=10, early_stopping=False, random_state=2026,
    )
    with threadpool_limits(limits=6):
        model.fit(x, target)
        return model.predict(z)


def _fit_predict(train, test, target, epochs):
    full = feature_columns(train)
    season = [c for c in full if c.startswith("season__")]
    out = {"trend": np.zeros(len(test))}
    out["season_tree"] = _tree_predictions(train, test, target, season)
    out["stage_tree"] = _tree_predictions(train, test, target, full)
    net = train_ensemble(train[full].to_numpy(dtype=np.float32), target, epochs=epochs)
    members = predict_members(net, test[full].to_numpy(dtype=np.float32))
    out["mlp"], out["mlp_std"] = members.mean(axis=0), members.std(axis=0, ddof=1)
    return out


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    residual = prediction - truth
    sst = np.sum((truth - truth.mean()) ** 2)
    return {
        "n": len(truth),
        "r2_log_ratio": float(1 - np.sum(residual ** 2) / sst) if sst > 0 else None,
        "rmse_log_ratio": float(np.sqrt(np.mean(residual ** 2))),
        "mean_absolute_percent_yield_error": float(100 * np.mean(np.abs(np.expm1(residual)))),
        "observed_mean_yield_loss_pct": float(100 * np.mean(1 - np.exp(truth))),
        "predicted_mean_yield_loss_pct": float(100 * np.mean(1 - np.exp(prediction))),
    }


def summarise(predictions):
    report = {}
    for crop, g in predictions.groupby("crop"):
        models = {}
        for model in ("trend", "season_tree", "stage_tree", "mlp"):
            models[model] = metrics(g.target, g[model])
        error = np.abs(g.mlp - g.target)
        models["ensemble_calibration_log_space"] = {
            "coverage_1_std": float((error <= g.mlp_std).mean()),
            "coverage_2_std": float((error <= 2 * g.mlp_std).mean()),
            "mean_std": float(g.mlp_std.mean()),
            "note": "Uncalibrated seed spread, epistemic only; not predictive intervals.",
        }
        report[crop] = models
    return report


def evaluate(data, folds=5, epochs=50, root=ROOT):
    """Year-grouped CV; crop-heldout predictors never see that crop's labels.

    Heldout-crop target normals necessarily use that crop's *other* years:
    this tests response transfer conditional on a supplied county crop normal,
    not zero-shot prediction of absolute yield.
    """
    columns = feature_columns(data)
    if any(c in columns for c in EXCLUDE):
        raise AssertionError("Identifier or label in feature matrix")
    grouped, held_crop, fold_years = [], [], []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=folds).split(data, groups=data.year)):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        years = sorted(test.year.unique().tolist())
        fold_years.append(years)
        trend = YieldTrend().fit(train)
        target_train, target_test = trend.transform(train), trend.transform(test)
        base = test[["crop", "adm_id", "year", "state_alpha", "irrigation_unknown", "prodn_practice_desc"]].copy()
        base["target"] = target_test
        start = time.monotonic()
        predictions = _fit_predict(train, test, target_train, epochs)
        for name, values in predictions.items():
            base[name] = values
        grouped.append(base)
        print(f"Year fold {fold + 1}/{folds}, held years {years}: {time.monotonic() - start:.1f}s", flush=True)
        for crop in sorted(data.crop.unique()):
            crop_train = train.crop.ne(crop).to_numpy()
            crop_test = test.crop.eq(crop).to_numpy()
            result = base.loc[crop_test, ["crop", "adm_id", "year", "state_alpha", "irrigation_unknown", "target"]].copy()
            predictions = _fit_predict(train.loc[crop_train], test.loc[crop_test], target_train[crop_train], epochs)
            for name, values in predictions.items():
                result[name] = values
            held_crop.append(result)
            print(f"  Held crop {crop}, fold {fold + 1}: {len(result):,} tests", flush=True)
        # Durable checkpoint: no need to rerun completed fits after later failures.
        pd.concat(grouped).to_parquet(root / "predictions_partial.parquet", index=False)
        pd.concat(held_crop).to_parquet(root / "held_crop_partial.parquet", index=False)
    predictions = pd.concat(grouped, ignore_index=True)
    held = pd.concat(held_crop, ignore_index=True)
    predictions.to_parquet(root / "predictions.parquet", index=False)
    held.to_parquet(root / "held_crop_predictions.parquet", index=False)
    drought = predictions.loc[predictions.year.eq(2012)]
    drought_states = drought.loc[drought.state_alpha.isin(DROUGHT_STATES)]
    return {
        "design": {
            "evaluation": f"{folds}-fold grouped by harvest year, not rolling-origin forecasting",
            "held_years": fold_years, "epochs": epochs, "seeds": [11, 23, 37, 51, 79],
            "features": columns, "drought_states": DROUGHT_STATES,
            "target": "log(yield / training-fold county-crop trend)",
            "dependency_versions": {name: version(name) for name in ("torch", "scikit-learn", "pandas", "pyarrow")},
            "crop_records": {crop: {k: asdict(v) for k, v in traits.items()} for crop, traits in CROPS.items()},
            "point_prediction": "Mean ensemble log-ratio; simulator returns arithmetic mean of exponentiated seed predictions.",
            "aggregation": "County-season equal weights, not harvested-area weights.",
            "mape": "100 * mean(abs(exp(predicted_log_ratio - actual_log_ratio) - 1))",
            "held_crop_baseline": "Other years of held crop used only for county normal, never response model/scaler.",
            "climatology": "Leave-calendar-year-out county month/day RSM, label-free; includes other folds' weather.",
            "limitations": [
                "Unknown irrigation ALL-practice yields retained; not a pure rainfed dataset.",
                "County crop-mask averages are not field weather or independent spatial holdouts.",
                "Generic thermal traits omit soybean photoperiod and wheat cultivar differences.",
                "Seed spread omits observation/structural noise and may be badly underdispersed.",
            ],
        },
        "grouped_year": summarise(predictions),
        "drought_2012_all": summarise(drought),
        "drought_2012_states": summarise(drought_states),
        "held_crop_and_year": summarise(held),
        "irrigation_screened_subset": summarise(predictions.loc[~predictions.irrigation_unknown]),
        "explicit_nonirrigated_only": summarise(
            predictions.loc[predictions.prodn_practice_desc.eq("NON-IRRIGATED")]
        ),
    }


def predict_relative_yield(stage_features, traits, soil, model_path=None, *, irrigated_share=None):
    """Return mean ratio and predictive SD from a calibrated v2 model if present.

    stage_features is one compute_stress row (including season features).
    traits contains numeric values or Trait records. No crop identifier is used.
    Explicitly selecting the archived MODEL_PATH retains its seed-only spread.
    """
    if model_path is None:
        for source in ("power", "cybench"):
            path = Path(f"data/processed/models/yield_response_v2_{source}.pt")
            if path.exists():
                from fieldshift.model.expanded import predict_calibrated_relative_yield

                return predict_calibrated_relative_yield(
                    stage_features, traits, soil, irrigated_share, str(path)
                )
        model_path = MODEL_PATH
    warnings.warn(
        "Archived v1 model: spread is seed disagreement, not calibrated predictive uncertainty. "
        "Train fieldshift.model.expanded for calibrated output.",
        UserWarning, stacklevel=2,
    )
    net, columns = load_ensemble(str(Path(model_path).resolve()))
    values = dict(stage_features)
    values.update(soil)
    values.update({f"trait__{k}": v for k, v in numeric_traits(traits).items()})
    required = [c for c in columns if "__rsm" not in c]
    missing = set(required) - values.keys()
    if missing:
        raise ValueError(f"Missing physical features: {sorted(missing)}")
    x = np.array([[values.get(c, np.nan) for c in columns]], dtype=np.float32)
    ratio = np.exp(predict_members(net, x)[:, 0])
    return float(ratio.mean()), float(ratio.std(ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    data = build_features()
    if args.build_only:
        return
    report = evaluate(data, args.folds, args.epochs)
    (ROOT / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    columns = feature_columns(data)
    trend = YieldTrend().fit(data)
    net = train_ensemble(data[columns].to_numpy(dtype=np.float32), trend.transform(data), epochs=args.epochs)
    save_ensemble(net, columns, MODEL_PATH)
    (ROOT / "training_baseline.json").write_text(json.dumps({
        "training_years": trend.training_years,
        "global_log_trend": trend.global_,
        "description": "Simulator supplies county crop normal separately; model returns relative yield.",
    }, indent=2))
    print(json.dumps(report["grouped_year"], indent=2), flush=True)
    print(f"Saved {MODEL_PATH} and {ROOT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
