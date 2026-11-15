"""Continuous-cropping yield penalty from QDANN field points, compared within county-year."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data.qdann import CROPS, YIELD_YEARS

CODE = {name: code for code, name in CROPS.items()}
OTHER = {"corn": "soy", "soy": "corn"}


def same_crop_run(points, crop):
    """Years in a row the point grew `crop` right before each yield year (0 = rotated)."""
    runs = {}
    for year in YIELD_YEARS:
        run = pd.Series(0, index=points.index)
        open_ = pd.Series(True, index=points.index)
        for back in range(1, year - 2006 + 1):
            open_ &= points[f"cdl_{year - back}"].eq(CODE[crop])
            run += open_.astype(int)
        runs[year] = run
    return runs


def table(points, interior_only=True):
    rows = []
    for crop in CROPS.values():
        runs = same_crop_run(points, crop)
        for year in YIELD_YEARS:
            value = points[f"{crop}_{year}"]
            keep = points[f"cdl_{year}"].eq(CODE[crop]) & value.gt(0)
            # Rotated means last year was the partner crop, so both arms are corn/soy systems.
            rotated = points[f"cdl_{year - 1}"].eq(CODE[OTHER[crop]])
            keep &= rotated | runs[year].gt(0)
            if interior_only:
                keep &= points[f"interior_{year}"].eq(1) & points[f"interior_{year - 1}"].eq(1)
            rows.append(pd.DataFrame({
                "crop": crop, "year": year, "state": points.loc[keep, "state"],
                "county": points.loc[keep, "county_fips"],
                "log_yield": np.log(value[keep]), "run": runs[year][keep],
            }))
    return pd.concat(rows, ignore_index=True)


def penalty(frame, rng, draws=500):
    """Within county-year difference of continuous vs rotated, with a county bootstrap CI."""
    frame = frame.assign(cont=frame["run"].gt(0).astype(float))
    group = frame.groupby(["county", "year"])
    mixed = group["cont"].transform("mean").between(0, 1, inclusive="neither")
    frame = frame[mixed]
    if frame.empty:
        return None
    group = frame.groupby(["county", "year"])
    x = frame["cont"] - group["cont"].transform("mean")
    y = frame["log_yield"] - group["log_yield"].transform("mean")
    by_county = pd.DataFrame({"county": frame["county"], "xy": x * y, "xx": x * x})
    sums = by_county.groupby("county")[["xy", "xx"]].sum()
    beta = sums["xy"].sum() / sums["xx"].sum()
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    boot = sums["xy"].to_numpy()[picks].sum(1) / sums["xx"].to_numpy()[picks].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    to_pct = lambda b: round(float(100 * (1 - np.exp(b))), 2)
    return {
        "penalty_pct": to_pct(beta), "ci95_pct": [to_pct(high), to_pct(low)],
        "continuous_rows": int(frame["cont"].sum()), "rotated_rows": int((1 - frame["cont"]).sum()),
        "counties": len(sums),
    }


def analyse(points, seed=0):
    rng = np.random.default_rng(seed)
    result = {}
    for interior_only in (True, False):
        data = table(points, interior_only)
        key = "interior_pixels" if interior_only else "all_pixels"
        result[key] = {}
        for crop in CROPS.values():
            subset = data[data["crop"] == crop]
            entry = {"overall": penalty(subset, rng)}
            entry["by_run_years"] = {
                str(run): penalty(subset[subset["run"].isin([0, run])], rng)
                for run in (1, 2, 3)
            }
            entry["by_run_years"]["4+"] = penalty(
                subset[subset["run"].eq(0) | subset["run"].ge(4)], rng
            )
            entry["by_state"] = {
                state: penalty(part, rng) for state, part in subset.groupby("state")
            }
            result[key][crop] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    points = pd.read_csv(args.data_dir / "processed" / "qdann_points.csv.gz")
    result = analyse(points)
    output = args.data_dir / "processed" / "rotation_penalty.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
