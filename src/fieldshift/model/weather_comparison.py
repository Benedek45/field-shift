"""Paired accuracy comparison of completed, identical-cohort weather evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.model.expanded import MODELS, report_tables


def compare_weather(reference_path, candidate_path, output=None):
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if reference["design"]["folds"] != candidate["design"]["folds"]:
        raise ValueError("Weather comparison requires identical fit/calibration/test years")
    if reference["design"]["epochs"] != candidate["design"]["epochs"]:
        raise ValueError("Weather comparison requires the same training budget")
    first = pd.read_parquet(reference_path.with_name("predictions.parquet"))
    second = pd.read_parquet(candidate_path.with_name("predictions.parquet"))
    keys = ["crop", "adm_id", "year", "prodn_practice_desc"]
    first, second = [x.sort_values(keys).reset_index(drop=True) for x in (first, second)]
    if not first[keys].equals(second[keys]):
        raise ValueError("Weather comparisons must use identical county/crop/practice/year rows")
    if not np.allclose(first.target, second.target, atol=1e-7, rtol=1e-7):
        raise ValueError("Weather comparison has different county normals or yield labels")
    if not np.allclose(first.harvested_acres, second.harvested_acres, equal_nan=True):
        raise ValueError("Weather comparison has different area weights")
    reference = report_tables(first, pd.DataFrame())["within_crop"]
    candidate = report_tables(second, pd.DataFrame())["within_crop"]
    result = {"reference": str(reference_path), "candidate": str(candidate_path),
              "rows": len(first), "per_crop": {}, "macro": {}}
    for crop, metrics in reference["per_crop"].items():
        result["per_crop"][crop] = {}
        for model in MODELS:
            if model not in metrics or model not in candidate["per_crop"][crop]:
                continue
            a, b = metrics[model]["equal_county"], candidate["per_crop"][crop][model]["equal_county"]
            result["per_crop"][crop][model] = {
                "reference_r2": a["r2"], "candidate_r2": b["r2"],
                "delta_r2": b["r2"] - a["r2"],
                "reference_mape": a["mape"], "candidate_mape": b["mape"],
                "delta_mape_points": b["mape"] - a["mape"],
            }
    for model in MODELS:
        values = [x[model] for x in result["per_crop"].values() if model in x]
        if not values:
            continue
        delta_r2 = float(np.mean([v["delta_r2"] for v in values]))
        delta_mape = float(np.mean([v["delta_mape_points"] for v in values]))
        result["macro"][model] = {
            "delta_r2": delta_r2, "delta_mape_points": delta_mape,
            "not_materially_worse": delta_r2 >= -.01 and delta_mape <= 1.,
            "area_weighted_mape_delta_points": (
                candidate["overall"][model]["area_weighted"]["mape"]
                - reference["overall"][model]["area_weighted"]["mape"]
            ),
        }
    result["note"] = "Fixed OOF predictions, identical labels/normals/years/weights. Point comparison, not proof of statistical noninferiority."
    if output:
        Path(output).write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_weather(args.reference, args.candidate, args.output)["macro"], indent=2))


if __name__ == "__main__":
    main()
