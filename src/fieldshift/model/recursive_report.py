"""Merge R3 fold outputs and compare them with R2-D on identical rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.model.expanded import report_tables
from fieldshift.model.recursive_data import SLOTS
from fieldshift.model.recursive_train import BASELINE, OUTPUT

KEYS = ["crop", "adm_id", "year", "series_id"]
BASELINE_MODELS = ["trend", "season_tree", "stage_tree", "mlp", "mlp_std"]


def _merge(recursive, baseline):
    merged = recursive.merge(baseline[KEYS + ["target"] + BASELINE_MODELS], on=KEYS, how="left",
                             suffixes=("", "_r2d"), validate="one_to_one")
    missing = merged.mlp.isna() & merged.target.notna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} R3 rows have no R2-D counterpart")
    gap = (merged.target - merged.target_r2d).abs().max()
    if not gap < 1e-4:
        raise ValueError(f"R3 and R2-D targets differ (max {gap}); normals/folds are not identical")
    return merged.drop(columns="target_r2d")


def state_leads(reports):
    errors = np.zeros((3, 92, len(SLOTS)))
    counts = np.zeros((92, len(SLOTS)))
    for report in reports:
        rmse = np.asarray(report["state_rmse"], dtype=float)
        count = np.asarray(report["state_counts"], dtype=float)
        errors += np.nan_to_num(rmse ** 2) * count[None]
        counts += count
    rmse = np.sqrt(errors / np.maximum(counts, 1)[None])
    result = {}
    for slot, name in enumerate(SLOTS):
        if not counts[:, slot].any():
            result[name] = {"observations": 0}
            continue
        valid = counts[:, slot] > 0
        better = (rmse[0, :, slot] < rmse[2, :, slot]) | ~valid
        first_loss = int(np.argmin(better)) if not better.all() else 92
        result[name] = {
            "observations": int(counts[:, slot].sum()),
            "leads_beating_climatology_from_start": first_loss,
            "leads_beating_climatology_total": int((rmse[0, :, slot] < rmse[2, :, slot])[valid].sum()),
            "leads_beating_persistence_total": int((rmse[0, :, slot] < rmse[1, :, slot])[valid].sum()),
            "valid_leads": int(valid.sum()),
            "rmse": {model: [None if not v else float(r) for r, v in zip(rmse[k, :, slot], valid)]
                     for k, model in enumerate(("recursive", "persistence", "climatology"))},
            "count": counts[:, slot].astype(int).tolist(),
        }
    return result


def merge(output=OUTPUT, baseline=BASELINE, destination=None, contaminated=False):
    output, baseline = Path(output), Path(baseline)
    folds = sorted(p for p in output.glob("fold_*.parquet")
                   if p.stem.removeprefix("fold_").isdigit())
    held = sorted(p for p in output.glob("held_*.parquet")
                  if p.stem.removeprefix("held_").isdigit())
    if not folds:
        raise FileNotFoundError(f"No R3 fold outputs in {output}")
    ordinary = _merge(pd.concat(map(pd.read_parquet, folds), ignore_index=True),
                      pd.read_parquet(baseline / "predictions.parquet"))
    held_frame = pd.DataFrame()
    if held:
        held_frame = _merge(pd.concat(map(pd.read_parquet, held), ignore_index=True),
                            pd.read_parquet(baseline / "held_crop_predictions.parquet"))
    reports = [json.loads(p.with_suffix(".json").read_text()) for p in folds]
    held_reports = [json.loads(p.with_suffix(".json").read_text()) for p in held]
    metrics = report_tables(ordinary, held_frame)
    metrics["state_by_lead"] = state_leads(reports)
    for name, values in metrics["state_by_lead"].items():
        values["valid"] = not (contaminated and name in {"lai", "fpar"})
        if not values["valid"]:
            values["invalid_reason"] = "Scored against snapshot targets corrupted by MODIS fill codes"
    metrics["contaminated_eo_targets"] = contaminated
    metrics["folds"] = [r["fold"] for r in reports]
    metrics["held_folds"] = {str(r["fold"]): r["held_crops"] for r in held_reports}
    metrics["timing_seconds"] = {
        **{f"fold_{r['fold']}": r["seconds"] for r in reports},
        **{f"held_{r['fold']}": r["seconds"] for r in held_reports},
    }
    metrics["epoch_seconds_median"] = float(np.median(
        [s for r in reports + held_reports for s in r["timing"]["epoch_seconds"]]))
    metrics["calibration_multipliers"] = {
        f"{'held' if r['held'] else 'fold'}_{r['fold']}": r["calibration"]["inner_year_sd_multiplier"]
        for r in reports + held_reports}
    metrics["parameters"] = reports[0]["timing"]["parameters"]
    metrics["complete"] = len(folds) == 5 and len(held) == 5
    metrics["design"] = (
        "R2-D year folds and per-fold county normals; R3 held fits remove every crop group "
        "assigned to that outer fold at once (harder than R2-D's one-group-at-a-time). "
        "Baseline columns are the R2-D predictions on identical rows.")
    destination = Path(destination or output / "metrics.json")
    destination.write_text(json.dumps(metrics, indent=2))
    ordinary.to_parquet(output / "predictions.parquet", index=False)
    if len(held_frame):
        held_frame.to_parquet(output / "held_crop_predictions.parquet", index=False)
    return metrics


def compare_versions(contaminated, clean, destination):
    """Paired comparison only; never compare partial v2 folds to all clean folds."""
    result = {
        "design": "Identical row intersection and county normals; v2 LAI/FPAR state scores invalid. "
                  "R3b changes target quality and adds ET/LST/SMAP supervision, not architecture.",
    }
    for filename, label in (("predictions.parquet", "within_crop"),
                            ("held_crop_predictions.parquet", "held_crop")):
        old_path, new_path = Path(contaminated) / filename, Path(clean) / filename
        if not old_path.exists() or not new_path.exists():
            continue
        old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
        common = old[KEYS].merge(new[KEYS], validate="one_to_one")
        old = common.merge(old, on=KEYS, validate="one_to_one")
        new = common.merge(new, on=KEYS, validate="one_to_one")
        if not len(common) or (old.target - new.target).abs().max() >= 1e-4:
            raise ValueError("No valid paired v2/R3b rows or county normals changed")
        result[label] = {"rows": len(common),
                         "v2": report_tables(old, pd.DataFrame())["within_crop"],
                         "r3b": report_tables(new, pd.DataFrame())["within_crop"]}
    Path(destination).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--contaminated", action="store_true", help="invalidate v2 LAI/FPAR state scores")
    args = parser.parse_args()
    result = merge(args.output, args.baseline, contaminated=args.contaminated)
    print(json.dumps({"complete": result["complete"], "within_macro": result["within_crop"]["macro_r2"],
                      "held_macro": result["held_crop"].get("macro_r2")}, indent=2))
