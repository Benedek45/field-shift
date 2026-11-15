"""Weekly state crop progress and condition from the NASS Quick Stats bulk file.

The bulk file holds one row per state x crop x week for every reported progress
stage (percent planted, emerged, silking, mature, harvested, ...) and condition
rating (percent excellent .. very poor).  A single streaming pass keeps those rows
and writes three tables: the weekly percentages, the interpolated dates at which a
stage crossed 10/50/90 percent, and the weekly condition ratings.

Progress is a cumulative percentage, so the crossing date is linearly interpolated
between the two weekly reports that bracket the threshold.  A stage already past
the threshold in its first report of the season has an unknown earlier date; that
row is flagged ``left_censored`` instead of guessing.  Missing values stay NaN.
"""

import argparse
import gzip
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STATISTICS = ("PROGRESS", "CONDITION")
THRESHOLDS = (10, 50, 90)  # percent of the crop, cumulative
KEEP = [
    "SOURCE_DESC", "SECTOR_DESC", "GROUP_DESC", "COMMODITY_DESC", "CLASS_DESC",
    "PRODN_PRACTICE_DESC", "UTIL_PRACTICE_DESC", "STATISTICCAT_DESC", "UNIT_DESC",
    "AGG_LEVEL_DESC", "STATE_FIPS_CODE", "STATE_ALPHA", "YEAR", "FREQ_DESC",
    "WEEK_ENDING", "VALUE", "CV_%",
]
PROGRESS_NAME = "nass_crop_progress.parquet"
DATES_NAME = "nass_crop_progress_dates.parquet"
CONDITION_NAME = "nass_crop_condition.parquet"
SUMMARY_NAME = "nass_crop_progress.summary.json"
QC_NAME = "nass_crop_progress_qc.json"
RENAMES = {
    "SOURCE_DESC": "source", "SECTOR_DESC": "sector", "GROUP_DESC": "group",
    "COMMODITY_DESC": "commodity", "CLASS_DESC": "class_desc",
    "PRODN_PRACTICE_DESC": "prodn_practice", "UTIL_PRACTICE_DESC": "util_practice",
    "UNIT_DESC": "unit", "AGG_LEVEL_DESC": "agg_level", "STATE_FIPS_CODE": "state_fips",
    "STATE_ALPHA": "state", "YEAR": "year", "FREQ_DESC": "freq",
    "WEEK_ENDING": "week_ending", "VALUE": "value", "CV_%": "cv",
    "STATISTICCAT_DESC": "statistic",
}
ROW_KEY = ["state", "commodity", "class_desc", "prodn_practice", "util_practice"]
_LOG = logging.getLogger(__name__)


def stage_name(unit):
    """``'PCT PLANTED'`` -> ``'planted'``; condition units keep their rating."""
    return unit.strip().removeprefix("PCT ").strip().lower()


def read_rows(path, chunksize=250_000):
    """Stream the bulk file once and return the weekly state progress/condition rows."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    missing = set(KEEP) - set(header)
    if missing:
        raise RuntimeError(f"Missing bulk columns: {sorted(missing)}")
    pieces = []
    with opener(path, "rt") as handle:
        for chunk in pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=KEEP, chunksize=chunksize):
            selected = chunk.loc[
                chunk.STATISTICCAT_DESC.isin(STATISTICS)
                & chunk.AGG_LEVEL_DESC.eq("STATE")
                & chunk.FREQ_DESC.eq("WEEKLY")
            ]
            if not selected.empty:
                pieces.append(selected)
    if not pieces:
        raise RuntimeError(f"No state weekly progress rows in {path}")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.rename(columns=RENAMES)
    frame["stage"] = frame.unit.map(stage_name)
    frame["value_raw"] = frame.value.str.strip()
    frame["value"] = pd.to_numeric(frame.value_raw.str.replace(",", "", regex=False),
                                   errors="coerce")
    # A handful of rows report impossible percentages (e.g. 799); the raw text stays.
    impossible = frame.value.notna() & ((frame.value < 0) | (frame.value > 100))
    frame.loc[impossible, "value"] = np.nan
    frame["cv"] = pd.to_numeric(frame.cv.str.replace("\x00", "", regex=False).str.strip(),
                                errors="coerce")
    frame["year"] = pd.to_numeric(frame.year, errors="raise").astype("int16")
    frame["week_ending"] = pd.to_datetime(frame.week_ending, errors="raise")
    return frame


def weekly(frame):
    """Progress table: one row per state, crop, year, week and stage."""
    columns = ROW_KEY + ["group", "sector", "statistic", "stage", "unit", "source",
                         "year", "week_ending", "value", "value_raw", "cv"]
    table = frame.loc[:, columns].sort_values(
        ROW_KEY + ["year", "stage", "week_ending"]).reset_index(drop=True)
    # A few stage/date pairs are repeated by a later load; keep the latest report.
    return table.drop_duplicates(ROW_KEY + ["year", "stage", "week_ending"], keep="last")


def crossing(dates, values, threshold):
    """Linear-interpolated date at which cumulative progress first reaches a threshold.

    Returns ``(date, left_censored)``; the date is NaT when the series never reaches
    the threshold and only the first report is at or above it.
    """
    order = np.argsort(dates)
    dates = np.asarray(dates, dtype="datetime64[D]")[order]
    values = np.asarray(values, float)[order]
    valid = np.isfinite(values)
    dates, values = dates[valid], values[valid]
    reached = np.flatnonzero(values >= threshold)
    if not reached.size:
        return np.datetime64("NaT", "D"), False
    first = int(reached[0])
    if first == 0:
        return dates[0], True
    low, high = values[first - 1], values[first]
    share = (threshold - low) / (high - low)
    span = float((dates[first] - dates[first - 1]).astype("timedelta64[D]").astype(int))
    offset = round(share * span)
    return dates[first - 1] + np.timedelta64(offset, "D"), False


def crossings(table, thresholds=THRESHOLDS):
    """Per state, crop, year and stage: the dates at which progress crossed thresholds."""
    keys = ROW_KEY + ["year", "stage"]
    records = []
    for (state, commodity, klass, prodn, util, year, stage), group in table.groupby(
            keys, observed=True, sort=True):
        dates = group.week_ending.to_numpy()
        values = group.value.to_numpy(float)
        record = {"state": state, "commodity": commodity, "class_desc": klass,
                  "prodn_practice": prodn, "util_practice": util, "year": int(year),
                  "stage": stage}
        censored = False
        for threshold in thresholds:
            date, clipped = crossing(dates, values, threshold)
            name = f"date_{threshold}"
            record[name] = date
            record[f"{name}_censored"] = clipped
            censored |= clipped
        record["first_report"] = pd.Timestamp(dates.min())
        record["last_report"] = pd.Timestamp(dates.max())
        record["max_progress"] = float(np.nanmax(values)) if np.isfinite(values).any() else np.nan
        record["n_reports"] = len(values)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    for threshold in thresholds:
        result[f"date_{threshold}"] = pd.to_datetime(result[f"date_{threshold}"])
    return result.loc[:, [*ROW_KEY, "year", "stage", "date_10", "date_50", "date_90",
                          "date_10_censored", "date_50_censored", "date_90_censored",
                          "first_report", "last_report", "max_progress", "n_reports"]]


def condition(frame):
    """Condition table: one row per state, crop, year, week and rating."""
    table = frame.loc[frame.statistic.eq("CONDITION"),
                      ROW_KEY + ["group", "stage", "unit", "year", "week_ending",
                                 "value", "value_raw"]]
    table = table.rename(columns={"stage": "rating"})
    return table.sort_values(ROW_KEY + ["year", "rating", "week_ending"]).reset_index(
        drop=True).drop_duplicates(ROW_KEY + ["year", "rating", "week_ending"], keep="last")


def quality(progress, dates, ratings):
    """Coverage, range checks and gap statistics for the three tables."""
    def span(frame):
        return [str(frame.week_ending.min().date()), str(frame.week_ending.max().date())]

    def gaps(frame, keys):
        """Longest run of missing weeks inside each key's reported span."""
        if frame.empty:
            return {"counties": 0, "median_max_gap_weeks": 0.0, "max_gap_weeks": 0}
        longest = []
        for _, group in frame.groupby(keys, observed=True):
            weeks = group.week_ending.sort_values().drop_duplicates()
            if len(weeks) < 2:
                continue
            step = weeks.diff().dt.days.fillna(7)
            longest.append(float((step - 7).clip(lower=0).max() / 7))
        if not longest:
            return {"groups": 0}
        return {"groups": len(longest),
                "median_max_gap_weeks": float(np.median(longest)),
                "max_gap_weeks": float(np.max(longest))}

    percent = progress.value.dropna()
    numeric = pd.to_numeric(progress.value_raw.str.replace(",", "", regex=False),
                            errors="coerce")
    bad = progress.loc[numeric.notna() & progress.value.isna()]
    return {
        "units": {"value": "percent of the crop, cumulative for progress stages; "
                           "percent of acres for condition ratings"},
        "source": "NASS Quick Stats bulk crops file (qs.crops_*.txt.gz), "
                  "source_desc SURVEY, agg_level STATE, freq WEEKLY",
        "time_step": "weekly (week ending Saturday)",
        "date_range": {"progress": span(progress), "condition": span(ratings)},
        "rows": {"progress": len(progress), "dates": len(dates),
                 "condition": len(ratings)},
        "states": int(progress.state.nunique()),
        "commodities": {"progress": int(progress.commodity.nunique()),
                        "condition": int(ratings.commodity.nunique())},
        "stages": sorted(progress.stage.unique()),
        "missing_percent": float(progress.value.isna().mean()),
        "range_checks": {"min": float(percent.min()), "max": float(percent.max()),
                         "p01": float(percent.quantile(0.01)),
                         "p99": float(percent.quantile(0.99))},
        "out_of_range": len(bad),
        "out_of_range_rows": [
            {"state": row.state, "commodity": row.commodity, "stage": row.stage,
             "year": int(row.year), "week_ending": str(row.week_ending),
             "value_raw": row.value_raw}
            for row in bad.itertuples()],
        "duplicate_keys": int(progress.duplicated(
            ["state", "commodity", "class_desc", "prodn_practice", "util_practice",
             "year", "stage", "week_ending"]).sum()),
        "gaps": gaps(progress, ["state", "commodity", "stage", "year"]),
        "crossing_dates": {"rows": len(dates),
                           "date_50_available": int(dates.date_50.notna().sum()),
                           "left_censored_50": int(dates.date_50_censored.sum())},
        "masking": "none: every reported row is kept, including non-numeric codes "
                   "which are kept in value_raw",
        "note": "Progress percentages are cumulative within a season and are not "
                "interpolated; crossing dates are the only derived quantity.",
    }


def _write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def build(data_dir=Path("data"), source=None, chunksize=250_000):
    """Read the bulk file once and write the progress, crossing-date and condition tables."""
    data_dir = Path(data_dir)
    source = Path(source) if source else max((data_dir / "raw" / "nass").glob("qs.crops_*.txt.gz"))
    raw = read_rows(source, chunksize)
    table = weekly(raw)
    progress = table.loc[table.statistic.eq("PROGRESS")].reset_index(drop=True)
    dates = crossings(progress)
    ratings = condition(raw)
    out = data_dir / "processed"
    _write(progress, out / PROGRESS_NAME)
    _write(dates, out / DATES_NAME)
    _write(ratings, out / CONDITION_NAME)
    progress_stages = progress.groupby("commodity", observed=True).stage.unique()
    summary = {
        "source": str(source),
        "rows": {"progress": len(progress), "dates": len(dates), "condition": len(ratings)},
        "years": [int(raw.year.min()), int(raw.year.max())],
        "states": int(raw.state.nunique()),
        "state_codes": sorted(raw.state.unique()),
        "progress_commodities": int(progress.commodity.nunique()),
        "condition_commodities": int(ratings.commodity.nunique()),
        "progress_stages": sorted(progress.stage.unique()),
        "condition_ratings": sorted(ratings.rating.unique()),
        "stages_by_commodity": {name: sorted(value) for name, value in
                                progress_stages.items()},
        "missing_values": int(table.value.isna().sum()),
        "out_of_range_values": len(raw.loc[raw.value_raw.str.fullmatch(r"-?\d+(\.\d+)?")
                                                 & raw.value.isna()]),
        "non_numeric_codes": dict(Counter(table.loc[table.value.isna(), "value_raw"])),
        "sources": sorted(raw.source.unique()),
        "left_censored_50pct": int(dates.date_50_censored.sum()),
        "note": "Progress is a cumulative percentage; crossing dates are linearly "
                "interpolated between the bracketing weekly reports. Rows flagged "
                "censored were already past the threshold in the first report.",
    }
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / QC_NAME).write_text(
        json.dumps(quality(progress, dates, ratings), indent=2, default=str) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = build(args.data_dir, args.source)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "stages_by_commodity"}, indent=2, default=str))


if __name__ == "__main__":
    main()
