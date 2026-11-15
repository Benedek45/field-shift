"""PRISM daily weather reduced to county means on the shared 8-day steps.

PRISM (Parameter-elevation Regressions on Independent Slopes Model) is the
station-based 4 km climate record for the conterminous US, published daily by the
PRISM Group at Oregon State University.  It replaces the coarse MERRA-2 weather in
``fieldshift.data.power`` for the variables it provides, because reanalysis rain is
weak on summer convective storms, which is the drought signal the yield model needs.

Earth Engine asset: ``OREGONSTATE/PRISM/ANd`` (the collection that superseded
``OREGONSTATE/PRISM/AN81d`` in 2025).  Verified from the catalog page:
1981-01-01..2026-11-08, 4638.3 m, bands ``ppt`` (mm), ``tmean``, ``tmin``,
``tmax``, ``tdmean`` (degC), ``vpdmin``, ``vpdmax`` (hPa).  An image carries the
date it is stamped with but starts at 12:00 UTC on that day, so an 8-day step
``[begin, end)`` holds exactly the images stamped ``begin .. end-1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data import eo

_LOG = logging.getLogger(__name__)

ASSET = "OREGONSTATE/PRISM/ANd"
CUBE_NAME = "prism_county_8day.f32"
DAILY_NAME = "prism_county_daily_ppt.f32"
CACHE = Path("raw") / "prism"

# Source bands, as they are named on Earth Engine.
BANDS = ["ppt", "tmean", "tmin", "tmax", "tdmean", "vpdmin", "vpdmax"]

VARIABLES = {
    "ppt_sum": ("mm", f"{ASSET} ppt, sum over the step"),
    "tmean": ("degC", f"{ASSET} tmean, mean over the step"),
    "tmin": ("degC", f"{ASSET} tmin, mean over the step"),
    "tmax": ("degC", f"{ASSET} tmax, mean over the step"),
    "tdmean": ("degC", f"{ASSET} tdmean, mean over the step"),
    "vpdmin": ("hPa", f"{ASSET} vpdmin, mean over the step"),
    "vpdmax": ("hPa", f"{ASSET} vpdmax, mean over the step"),
    "wet_days": ("days", f"{ASSET} ppt >= 1 mm, count over the step"),
    "ppt_max": ("mm", f"{ASSET} ppt, largest single day of the step"),
}

# Physical bounds of a county mean; a published value outside these is a bug.
BOUNDS = {
    "ppt_sum": (0.0, 1500.0),
    "tmean": (-60.0, 55.0),
    "tmin": (-60.0, 55.0),
    "tmax": (-60.0, 60.0),
    "tdmean": (-60.0, 45.0),
    "vpdmin": (0.0, 100.0),
    "vpdmax": (0.0, 200.0),
    "wet_days": (0.0, 8.0),
    "ppt_max": (0.0, 800.0),
}

# Per-cell bounds of a step composite, applied on the raw grid before any reduction.
CELL_BOUNDS = {"ppt_sum": (0.0, 2000.0), "tmean": (-60.0, 60.0), "tmin": (-60.0, 60.0),
               "tmax": (-60.0, 60.0), "tdmean": (-60.0, 50.0),
               "vpdmin": (0.0, 100.0), "vpdmax": (0.0, 200.0),
               "wet_days": (0.0, 8.0), "ppt_max": (0.0, 800.0)}


def day_expression(begin, end):
    """Nine-band step composite of PRISM over ``[begin, end)``.

    Every band is masked where the step is incomplete, because Earth Engine's
    ``sum``/``mean`` silently skip masked days and would otherwise return a
    partial total for a coast or edge cell.
    """
    import ee

    days = int((np.datetime64(end) - np.datetime64(begin)) / np.timedelta64(1, "D"))
    collection = ee.ImageCollection(ASSET).filterDate(str(begin), str(end))
    weather = collection.select(BANDS)
    precipitation = collection.select("ppt")
    complete = precipitation.count().eq(days)
    return ee.Image.cat([
        precipitation.sum().updateMask(complete).rename("ppt_sum"),
        weather.select(["tmean", "tmin", "tmax", "tdmean", "vpdmin", "vpdmax"])
        .mean().updateMask(complete),
        precipitation.map(lambda image: image.gte(1)).sum().updateMask(complete)
        .rename("wet_days"),
        precipitation.max().updateMask(complete).rename("ppt_max"),
    ])


def _clean(data):
    """Drop cells outside a band's physical range, so no fill code averages in."""
    out = {}
    for name, array in data.items():
        low, high = CELL_BOUNDS[name]
        value = np.asarray(array, np.float32)
        out[name] = np.where((value < low) | (value > high), np.nan, value)
    return out


def _step_series(step, cache, index, value, project=None):
    """County means of one step, from cache when it exists."""
    path = Path(cache) / f"{step}.npy"
    if path.exists():
        array = np.load(path)
        data = {name: array[position] for position, name in enumerate(VARIABLES)}
    else:
        eo.initialize(project)
        begin, end = eo.step_bounds(step)
        data = eo.fetch(day_expression(begin, end), list(VARIABLES))
        array = np.stack([np.asarray(data[name], np.float32) for name in VARIABLES])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        with temporary.open("wb") as handle:
            np.save(handle, array)
        temporary.replace(path)
    clean = _clean(data)
    return {name: eo.county_mean(clean[name].reshape(1, -1), index, value)[0]
            for name in VARIABLES}


def build_steps(steps, data_dir=Path("data"), workers=3, project=None):
    """County series ``(counties, N_STEPS)`` for the requested steps."""
    data_dir = Path(data_dir)
    table, order, _ = eo.load_weights(data_dir)
    index, value = eo.weight_matrix(table, order)
    cache = data_dir / "raw" / "prism"
    series = {name: np.full((len(order), eo.N_STEPS), np.nan, np.float32)
              for name in VARIABLES}
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_step_series, step, cache, index, value, project): step
                   for step in steps}
        for finished, future in enumerate(futures, start=1):
            step = futures[future]
            try:
                counties = future.result()
            except Exception as error:  # noqa: BLE001 - one step must not kill the run
                _LOG.warning("step %d (%s) failed: %s", step, eo.step_bounds(step)[0],
                             error)
                continue
            for name, values in counties.items():
                series[name][:, step] = values
            if finished % 100 == 0:
                _LOG.info("%d of %d steps done", finished, len(futures))
    return series, order


def check_bounds(arrays, variables=None):
    """Raise when a published value is outside its physical bounds."""
    problems = []
    for name, array in arrays.items():
        if name not in BOUNDS:
            continue
        low, high = BOUNDS[name]
        valid = np.isfinite(array)
        outside = valid & ((array < low) | (array > high))
        if outside.any():
            problems.append({"variable": name, "count": int(outside.sum()),
                             "min": float(np.nanmin(array)),
                             "max": float(np.nanmax(array)),
                             "bounds": [low, high]})
    if problems:
        raise ValueError(f"values outside physical bounds: {problems}")
    return True


def build(data_dir=Path("data"), workers=3, project=None, steps=None):
    """Fetch every step, write the cube and the metadata."""
    data_dir = Path(data_dir)
    steps = list(range(eo.N_STEPS)) if steps is None else list(steps)
    began = time.monotonic()
    series, order = build_steps(steps, data_dir, workers, project)
    check_bounds(series)
    variables = {}
    for name, (units, source) in VARIABLES.items():
        array = series[name]
        valid = np.isfinite(array)
        indices = np.flatnonzero(valid.any(axis=0))
        variables[name] = {
            "name": name, "units": units, "source": source,
            "nan_fraction": round(float(1 - valid.mean()), 6),
            "first_valid": str(eo.step_bounds(indices[0])[0]) if indices.size else None,
            "last_valid": str(eo.step_bounds(indices[-1])[0]) if indices.size else None,
        }
    out = data_dir / "processed" / "weather"
    out.mkdir(parents=True, exist_ok=True)
    _, _, area_only = eo.load_weights(data_dir)
    metadata = eo.write_cube(out / CUBE_NAME, series, order, variables, {
        "grid": {"crs": "EPSG:4326", "resolution_deg": eo.GRID_CELL,
                 "lat_range": [eo.GRID_BOTTOM, eo.GRID_TOP],
                 "lon_range": [eo.GRID_LEFT, eo.GRID_RIGHT],
                 "shape": [eo.HEIGHT, eo.WIDTH],
                 "resampling": "bilinear sampling of the 4 km PRISM grid onto 0.05 "
                               "degree cells, then a cropland-weighted county mean"},
        "aggregation": {"expression": "ppt.sum, wet_days = count(ppt >= 1 mm), "
                                      "ppt_max = ppt.max, other bands .mean, all "
                                      "computed on the server per 8-day step",
                        "day_definition": "a PRISM image is stamped with its date "
                                          "and starts at 12:00 UTC, so a step "
                                          "[begin, end) holds the images stamped "
                                          "begin..end-1",
                        "asset_supersedes": "OREGONSTATE/PRISM/AN81d (deprecated)"},
        "weights": {"file": "processed/eo/eo_county_weights.parquet",
                    "area_only_counties": area_only},
        "bounds": {name: list(value) for name, value in BOUNDS.items()},
        "requests": len(steps),
        "seconds": round(time.monotonic() - began, 1),
    })
    return metadata


def load_prism(path=None):
    """Open the PRISM cube and its JSON metadata as ``(memmap, dict)``."""
    root = Path(path) if path else Path("data") / "processed" / "weather"
    path = root / CUBE_NAME if root.is_dir() else root
    metadata = json.loads(path.with_suffix(".json").read_text())
    cube = np.memmap(path, dtype="<f4", mode="r", shape=tuple(metadata["shape"]))
    return cube, metadata


def county_frame(cube, metadata, adm_id):
    """One county's ``(steps, variables)`` block with step start dates."""
    row = metadata["counties"].index(adm_id)
    names = [variable["name"] for variable in metadata["variables"]]
    frame = pd.DataFrame(np.asarray(cube[row]), columns=names)
    frame.insert(0, "start_date", pd.to_datetime(metadata["step_start_dates"]))
    return frame


def power_steps(data_dir=Path("data")):
    """POWER county-daily weather plus its variables, for the same grid."""
    from fieldshift.data import power

    return power.load_county_daily(Path(data_dir) / "processed" / "weather" /
                                   power.CUBE_NAME)


def _step_windows(metadata):
    dates = pd.to_datetime(metadata["step_start_dates"])
    return dates


def validate(data_dir=Path("data")):
    """PRISM against POWER on the same steps, and the drought years."""
    data_dir = Path(data_dir)
    cube, metadata = load_prism(data_dir / "processed" / "weather")
    names = [variable["name"] for variable in metadata["variables"]]
    order = metadata["counties"]
    weather, weather_meta = power_steps(data_dir)
    origin = pd.Timestamp(weather_meta["start_date"])
    power_names = [item["name"] for item in weather_meta["variables"]]
    dates = _step_windows(metadata)

    def power_step(column, reducer="mean"):
        index = power_names.index(column)
        out = np.full((len(order), eo.N_STEPS), np.nan, np.float32)
        for step in range(eo.N_STEPS):
            offset = (dates[step] - origin).days
            window = weather[:, offset:offset + eo.steps_covered(step), index]
            out[:, step] = (np.nansum(window, axis=1) if reducer == "sum"
                            else np.nanmean(window, axis=1))
        return out

    ours = {name: np.asarray(cube[:, :, names.index(name)], float) for name in names}
    theirs = {"ppt_sum": power_step("prec", "sum"), "tmin": power_step("tmin"),
              "tmax": power_step("tmax")}
    report = {"window": "1981-2025", "counties": len(order)}
    for name, series in theirs.items():
        correlation, _ = eo._correlation(ours[name], series)
        bias = np.nanmean(ours[name] - series, axis=1)
        report[name] = {
            "counties": int(np.isfinite(correlation).sum()),
            "median_r": float(np.nanmedian(correlation)),
            "quartiles": [float(np.nanquantile(correlation, 0.25)),
                          float(np.nanquantile(correlation, 0.75))],
            "median_bias": float(np.nanmedian(bias)),
            "bias_units": "mm per 8 days" if name == "ppt_sum" else "degC",
        }
    jja = np.asarray(dates.month.isin([6, 7, 8]))
    correlation, _ = eo._correlation(ours["ppt_sum"][:, jja], theirs["ppt_sum"][:, jja])
    report["ppt_sum_jja"] = {
        "counties": int(np.isfinite(correlation).sum()),
        "median_r": float(np.nanmedian(correlation)),
        "median_bias": float(np.nanmedian(
            np.nanmean(ours["ppt_sum"][:, jja] - theirs["ppt_sum"][:, jja], axis=1))),
    }
    story = order.index("US-19-169")
    report["story_county_jja_ppt_mm"] = {
        str(year): {"prism": float(np.nansum(ours["ppt_sum"][story][
            jja & (np.asarray(dates.year) == year)])),
            "power": float(np.nansum(theirs["ppt_sum"][story][
                jja & (np.asarray(dates.year) == year)]))}
        for year in (1988, 1993, 2012)}
    states = {"19": "Iowa", "17": "Illinois", "18": "Indiana", "29": "Missouri",
              "31": "Nebraska", "20": "Kansas"}
    mask = np.array([adm_id[3:5] in states for adm_id in order])
    # A seasonal total per year, so the 1991-2020 reference is a mean seasonal
    # total and not a per-step mean (they differ by the number of JJA steps).
    def seasonal_total(series, year):
        inside = jja & (np.asarray(dates.year) == year)
        return np.nansum(series[mask][:, inside], axis=1)

    anomalies = {}
    for year in (1983, 1988, 2012):
        entry = {}
        for label, series in (("prism", ours["ppt_sum"]), ("power", theirs["ppt_sum"])):
            reference = np.mean([seasonal_total(series, value)
                                 for value in range(1991, 2021)], axis=0)
            annual = seasonal_total(series, year)
            with np.errstate(invalid="ignore", divide="ignore"):
                entry[label] = float(np.nanmean(100 * (annual / reference - 1)))
        anomalies[str(year)] = entry
    report["region_jja_ppt_anomaly_percent_of_1991_2020"] = {
        "states": list(states.values()), "values": anomalies}
    path = data_dir / "processed" / "weather" / "prism_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def quality(data_dir=Path("data")):
    """Missingness, gaps, bounds and weights for the published PRISM cube."""
    data_dir = Path(data_dir)
    out = data_dir / "processed" / "weather"
    metadata = json.loads((out / CUBE_NAME).with_suffix(".json").read_text())
    cube, _ = load_prism(out)
    names = [variable["name"] for variable in metadata["variables"]]
    years = np.arange(eo.FIRST_YEAR, eo.LAST_YEAR + 1)
    dates = np.array(metadata["step_start_dates"], dtype="datetime64[D]")
    report = {"shape": metadata["shape"], "counties": len(metadata["counties"]),
              "steps": eo.N_STEPS, "start_date": metadata["start_date"],
              "end_date": metadata["end_date"],
              "time_step": f"8 days ({eo.STEPS_PER_YEAR} per year, last step short)",
              "source": ASSET, "grid": metadata["grid"],
              "aggregation": metadata["aggregation"],
              "variables": {},
              "area_only_counties": metadata["weights"]["area_only_counties"],
              "duplicate_county_keys": int(len(metadata["counties"]) -
                                           len(set(metadata["counties"]))),
              "accessed": pd.Timestamp.now().date().isoformat()}
    for position, name in enumerate(names):
        array = np.asarray(cube[:, :, position], float)
        missing = np.isnan(array)
        per_year = missing.reshape(len(array), len(years), eo.STEPS_PER_YEAR)
        gaps = np.array([eo._longest_run(missing[row]) for row in range(len(array))])
        low, high = BOUNDS[name]
        valid = np.isfinite(array)
        report["variables"][name] = {
            "units": VARIABLES[name][0],
            "bounds": [low, high],
            "out_of_bounds": int((valid & ((array < low) | (array > high))).sum()),
            "min": float(np.nanmin(array)) if valid.any() else None,
            "max": float(np.nanmax(array)) if valid.any() else None,
            "nan_fraction": round(float(missing.mean()), 6),
            "nan_fraction_by_year": {int(year): round(float(value), 5) for year, value
                                     in zip(years, per_year.mean(axis=(0, 2)),
                                            strict=True)},
            "longest_gap_steps": {"median": float(np.median(gaps)),
                                  "max": float(gaps.max()),
                                  "max_days": float(gaps.max() * eo.STEP_DAYS)},
        }
    report["all_missing_ranges"] = [
        [str(dates[first]), str(dates[last])]
        for first, last in eo._runs(np.isnan(np.asarray(cube[:, :, 0], float)).all(axis=0))]
    (out / "prism_county_8day_qc.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--steps", default=None,
                        help="inclusive step range a:b, for a timed test")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--qc", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.steps:
        first, _, last = args.steps.partition(":")
        steps = list(range(int(first), int(last or first) + 1))
    else:
        steps = None
    if args.build:
        metadata = build(args.data_dir, args.workers, args.project, steps)
        print(json.dumps({"shape": metadata["shape"], "steps": metadata["requests"],
                          "seconds": metadata["seconds"]}))
    if args.validate:
        print(json.dumps(validate(args.data_dir), indent=2))
    if args.qc:
        print(json.dumps(quality(args.data_dir)["variables"], indent=2)[:2000])


if __name__ == "__main__":
    main()
