"""8-day county satellite series for the contiguous US from Earth Engine.

Every composite is fetched once as a small regular grid over CONUS (0.05 degree,
520 x 1180 cells) with :meth:`ee.data.computePixels`, cached as a raw float32
array, and reduced to counties locally with cropland-fraction x cell-area weights.
That is one request per composite instead of one per county.

The grid request reprojects the source product to EPSG:4326 with bilinear
resampling, so a cell value is a local 4-pixel interpolation of the native grid
(MODIS 500 m - 1 km), not a box average of the whole 5.5 km cell.  Box averaging
through ``reduceResolution`` would be 30-100x more server work per composite.

Cropland is the complement of the CDL classes ``power.NON_CULTIVATED``, kept where
at least half of the 2008-2022 years classify as cropland; counties with almost no
cropland fall back to pure area weights and are listed in the metadata.

Missing values are NaN everywhere; nothing is interpolated across a gap.
"""

import argparse
import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data import power

FIRST_YEAR, LAST_YEAR = 1981, 2025
STEP_DAYS, STEPS_PER_YEAR = 8, 46
N_STEPS = STEPS_PER_YEAR * (LAST_YEAR - FIRST_YEAR + 1)
GRID_LEFT, GRID_TOP, GRID_CELL = -125.0, 50.0, 0.05
GRID_BOTTOM, GRID_RIGHT = 24.0, -66.0
HEIGHT = round((GRID_TOP - GRID_BOTTOM) / GRID_CELL)
WIDTH = round((GRID_RIGHT - GRID_LEFT) / GRID_CELL)
AREA_CRS = "EPSG:5070"
MIN_CULTIVATED = 0.01

MODIS_NDVI = "MODIS/061/MOD13A2"
AVHRR_NDVI = "NOAA/CDR/AVHRR/NDVI/V5"
MODIS_LAI = "MODIS/061/MOD15A2H"
MODIS_ET = "MODIS/061/MOD16A2GF"
MODIS_LST = "MODIS/061/MOD11A2"
SMAP = "NASA/SMAP/SPL4SMGP/008"
GLDAS = "NASA/GLDAS/V021/NOAH/G025/T3H"
GRACE = "NASA/GRACE/MASS_GRIDS_V04/MASCON"

CUBE_NAME = "eo_county_8day.f32"
META_NAME = "eo_county_8day.json"
QC_NAME = "eo_county_8day_qc.json"
WEIGHTS_NAME = "eo_county_weights.parquet"
PARTS = "parts"
GLDAS_LAYERS = (("SoilMoi0_10cm_inst", 0.10), ("SoilMoi10_40cm_inst", 0.30),
                ("SoilMoi40_100cm_inst", 0.60))
_LOG = logging.getLogger(__name__)
_EE_READY = False


def initialize(project=None):
    global _EE_READY
    if not _EE_READY:
        import ee

        ee.Initialize(project=project or os.environ.get("EE_PROJECT"))
        _EE_READY = True


# --- 8-day time axis -------------------------------------------------------

def day(value):
    """``datetime64[D]`` from a date-like value."""
    return np.datetime64(pd.Timestamp(value).date(), "D")


def step_starts():
    """First day of every 8-day step, 1981-01-01 .. 2025-12-28."""
    return np.concatenate([
        np.datetime64(f"{year}-01-01", "D")
        + np.arange(STEPS_PER_YEAR) * np.timedelta64(STEP_DAYS, "D")
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
    ])


def step_bounds(step):
    """Half-open ``(first_day, last_day_exclusive)`` of one step.

    The 46th step of a year stops at the next 1 January, so it is 5 days long in a
    common year and 6 in a leap year.
    """
    step = int(step)
    if not 0 <= step < N_STEPS:
        raise IndexError(f"Step {step} outside 0..{N_STEPS - 1}")
    year = FIRST_YEAR + step // STEPS_PER_YEAR
    index = step % STEPS_PER_YEAR
    first = np.datetime64(f"{year}-01-01", "D") + np.timedelta64(
        STEP_DAYS * index, "D")
    if index == STEPS_PER_YEAR - 1:
        return first, np.datetime64(f"{year + 1}-01-01", "D")
    return first, first + np.timedelta64(STEP_DAYS, "D")


def step_of(value):
    """Global step index holding a date; raises outside the 1981-2025 window."""
    date = day(value)
    first = step_starts()
    step = int(np.searchsorted(first, date, side="right")) - 1
    if step < 0 or date >= step_bounds(N_STEPS - 1)[1]:
        raise ValueError(f"{date} outside {first[0]}..{step_bounds(N_STEPS - 1)[1]}")
    return step


def steps_covered(step):
    """Days of a step."""
    first, last = step_bounds(step)
    return int((last - first).astype(int))


def assign_steps(starts, length):
    """Steps covered by each composite of a fixed revisit period.

    ``starts`` are composite start days (any order).  A step belongs to the
    composite whose half-open ``[start, start + length)`` window contains the
    step's first day, so a tiling of composites fills every step exactly once.
    Returns ``(steps, composites)`` arrays of equal length.
    """
    starts = np.asarray(starts, dtype="datetime64[D]")
    steps = step_starts()
    order = np.argsort(starts)
    sorted_starts = starts[order]
    position = np.searchsorted(sorted_starts, steps, side="right") - 1
    inside = (position >= 0) & (
        steps - sorted_starts[np.clip(position, 0, len(starts) - 1)]
        < np.timedelta64(int(length), "D")
    )
    chosen = np.flatnonzero(inside)
    return chosen, order[position[chosen]]


# --- grid and counties -----------------------------------------------------

def cell_centers(rows=HEIGHT, cols=WIDTH, cell=GRID_CELL):
    """Cell centre latitudes (north first) and longitudes (west first) of the grid."""
    lat = GRID_TOP - (np.arange(rows) + 0.5) * cell
    lon = GRID_LEFT + (np.arange(cols) + 0.5) * cell
    return lat.astype(np.float32), lon.astype(np.float32)


def ee_grid(rows=HEIGHT, cols=WIDTH, cell=GRID_CELL):
    """Earth Engine ``PixelGrid`` over the same cells as :func:`cell_centers`."""
    return {
        "crsCode": "EPSG:4326",
        "dimensions": {"width": int(cols), "height": int(rows)},
        "affineTransform": {"scaleX": cell, "scaleY": -cell,
                            "translateX": GRID_LEFT, "translateY": GRID_TOP},
    }


def county_order(data_dir=Path("data")):
    """County adm_ids in the shared order taken from the weather archive."""
    path = Path(data_dir) / "processed" / "weather" / "power_county_daily.json"
    return json.loads(path.read_text())["counties"]


def county_frame(data_dir=Path("data")):
    """CONUS county polygons ordered like the weather archive."""
    frame = power.counties(Path(data_dir) / "raw" / "cybench" / "polygons.zip")
    order = county_order(data_dir)
    frame = frame.set_index("adm_id").loc[order].reset_index()
    return frame


def grid_cells(rows=HEIGHT, cols=WIDTH):
    """GeoDataFrame of the rectangular grid cells with their row and column."""
    import shapely

    lat, lon = cell_centers(rows=rows, cols=cols)
    half = GRID_CELL / 2
    records = [
        {"cell_row": i, "cell_col": j,
         "geometry": shapely.box(x - half, y - half, x + half, y + half)}
        for i, y in enumerate(lat) for j, x in enumerate(np.asarray(lon, float))
    ]
    return pd.DataFrame(records)


def area_weights(counties, cells, area_crs=AREA_CRS):
    """Intersection area in km2 of every county x cell pair."""
    import geopandas as gpd
    import shapely

    left = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326").to_crs(area_crs)
    right = counties[["adm_id", "geometry"]].to_crs(area_crs)
    pairs = gpd.sjoin(left, right, predicate="intersects", how="inner")
    left_geometry = left.geometry.to_numpy()
    right_geometry = right.geometry.to_numpy()
    pieces = shapely.intersection(
        left_geometry[pairs.index.to_numpy()],
        right_geometry[pairs.index_right.to_numpy()],
    )
    table = pd.DataFrame({
        "adm_id": right.adm_id.to_numpy()[pairs.index_right.to_numpy()],
        "cell_row": pairs.cell_row.to_numpy().astype(np.int16),
        "cell_col": pairs.cell_col.to_numpy().astype(np.int16),
        "area_km2": shapely.area(pieces) / 1e6,
    })
    table = table.loc[np.isfinite(table.area_km2) & table.area_km2.gt(1e-6)]
    return table.reset_index(drop=True)


def combine_weights(areas, cultivated, counties_order):
    """Area x cultivated-fraction weights per county, with an area-only fallback.

    ``cultivated`` is a DataFrame of ``adm_id``, ``cell_row``, ``cell_col`` and the
    cultivated fraction of the cell.  Returns the weight table and the counties
    that fell back to area weights.
    """
    table = areas.merge(cultivated, on=["cell_row", "cell_col"], how="left")
    table["cultivated"] = table.cultivated.fillna(0.0).clip(0.0, 1.0)
    table["weighted_km2"] = table.area_km2 * table.cultivated
    area = table.groupby("adm_id").area_km2.transform("sum")
    share = table.groupby("adm_id").weighted_km2.transform("sum") / area
    fallback_ids = sorted(table.loc[share.lt(MIN_CULTIVATED), "adm_id"].unique())
    fallback = table.adm_id.isin(fallback_ids).to_numpy()
    table["weight"] = np.where(fallback, table.area_km2, table.weighted_km2)
    table["area_only"] = fallback
    table = table.loc[table.weight.gt(0)].copy()
    table["weight"] /= table.groupby("adm_id").weight.transform("sum")
    table["adm_id"] = pd.Categorical(table.adm_id, categories=list(counties_order),
                                     ordered=True)
    missing = sorted(set(counties_order) - set(table.adm_id.astype(str)))
    if missing:
        raise ValueError(f"Counties without weights: {missing[:5]} (+{len(missing)} more)")
    return table.sort_values(["adm_id", "cell_row", "cell_col"]).reset_index(drop=True), \
        fallback_ids


def weight_matrix(weights, counties_order):
    """Padded (county, degree) cell linear index and weight arrays."""
    degree = int(weights.groupby("adm_id", observed=True).size().max())
    index = np.zeros((len(counties_order), degree), np.int32)
    value = np.zeros((len(counties_order), degree), np.float32)
    lookup = {adm_id: row for row, adm_id in enumerate(counties_order)}
    for adm_id, group in weights.groupby("adm_id", observed=True, sort=False):
        row = lookup[str(adm_id)]
        count = len(group)
        index[row, :count] = (group.cell_row.to_numpy(np.int32) * WIDTH
                              + group.cell_col.to_numpy(np.int32))
        value[row, :count] = group.weight.to_numpy(np.float32)
    return index, value


def county_mean(grid, index, value):
    """Weighted county mean of one or more gridded steps, dropping missing cells.

    ``grid`` is ``(steps, HEIGHT * WIDTH)``; the result is ``(steps, counties)``.
    A county whose cells are all missing on a step stays NaN.
    """
    grid = np.atleast_2d(np.asarray(grid, np.float32))
    total = np.zeros((grid.shape[0], index.shape[0]), np.float32)
    weight = np.zeros_like(total)
    for column in range(index.shape[1]):
        sample = grid[:, index[:, column]]
        finite = np.isfinite(sample)
        total += np.where(finite, value[:, column] * np.nan_to_num(sample), 0).astype(
            np.float32)
        weight += np.where(finite, value[:, column], 0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / weight, np.nan).astype(np.float32)


def flatten(raster):
    """``(rows, cols)`` (or ``(bands, rows, cols)``) raster as grid cells per step."""
    raster = np.asarray(raster, np.float32)
    return raster.reshape(-1, HEIGHT * WIDTH) if raster.ndim == 3 else \
        raster.reshape(1, -1)


# --- Earth Engine fetching -------------------------------------------------

def cultivated_mask():
    """CDL cultivation over 2008-2022, one where at least half the years were cropland."""
    import ee

    values = [value for value in range(1, 255) if value not in power.NON_CULTIVATED]

    def cultivate(image):
        crop = image.select("cropland")
        return crop.remap(values, [1] * len(values), 0).toFloat()

    collection = ee.ImageCollection("USDA/NASS/CDL").filterDate("2008-01-01", "2023-01-01")
    return collection.map(cultivate).mean().gte(0.5).rename("cultivated")


def fetch(expression, bands, grid=None, attempts=4):
    """Compute pixels of one expression over the CONUS grid; retries transient errors."""
    import ee

    grid = grid or ee_grid()
    for attempt in range(1, attempts + 1):
        try:
            data = ee.data.computePixels({
                "expression": expression, "fileFormat": "NUMPY_NDARRAY",
                "bandIds": list(bands), "grid": grid,
            })
            if isinstance(data, dict):
                return {name: np.asarray(data[name]) for name in bands}
            array = np.asarray(data)
            if array.dtype.names:
                return {name: array[name] for name in array.dtype.names}
            return {bands[0]: array}
        except Exception as error:  # transient EE/HTTP failures
            if attempt == attempts:
                raise
            _LOG.warning("Request failed (%s), retry %d/%d in %ds", error, attempt,
                         attempts, 4 * attempt)
            time.sleep(4 * attempt)
    raise RuntimeError("unreachable")


def _cached(cache, key, bands, build, scale=None, fill=None):
    """One composite's bands from the raw cache, computing them on first use."""
    path = Path(cache) / f"{key}.npy"
    if path.exists():
        array = np.load(path)
        return {name: array[i] for i, name in enumerate(bands)}
    data = build()
    array = np.stack([np.asarray(data[name], np.float32) for name in bands])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)
    return {name: array[i] for i, name in enumerate(bands)}


def _dn_mask(array, low=None, high=None, fill=None):
    """Drop cells outside the declared valid digital-number window."""
    out = np.asarray(array, np.float32)
    missing = np.zeros(out.shape, bool)
    if fill is not None:
        missing |= out == fill
    if low is not None:
        missing |= out < low
    if high is not None:
        missing |= out > high
    return np.where(missing, np.nan, out).astype(np.float32)


def _masked(value, scale=1.0, fill=None, low=None, high=None):
    """Scale a band to float and set the fill value (and optional bounds) to NaN."""
    value = np.asarray(value, float)
    out = value * scale
    missing = np.zeros(value.shape, bool)
    if fill is not None:
        missing |= value == fill
    if low is not None:
        missing |= out < low
    if high is not None:
        missing |= out > high
    out[missing] = np.nan
    return out.astype(np.float32)


def _collection_index(asset, first, last):
    """``(keys, start_days)`` of a composite collection, ordered by time."""
    import ee

    collection = ee.ImageCollection(asset).filterDate(str(first), str(last))
    keys = collection.aggregate_array("system:index").getInfo()
    starts = np.array([day(key.replace("_", "-")) for key in keys])
    order = np.argsort(starts)
    return [keys[i] for i in order], starts[order]


def _run(keys, build, names, cache, workers, index, value, mask=None):
    """Fetch (or reuse) every composite and reduce it to counties in the same pass.

    ``mask`` runs on the raw grid before the reduction, so quality flags become
    NaN per cell instead of being averaged into the county mean.  A window with no
    source images leaves an image without bands; that is a real gap in the
    product, so the composite stays NaN and is counted in the log.
    """
    empty = []

    def one(item):
        position, key = item
        try:
            data = _cached(cache, key, names, lambda: fetch(build(key), names))
        except Exception as error:
            if "has no band named" not in str(error):
                raise
            empty.append(key)
            return position, None
        if mask is not None:
            data = mask(data)
        counties = {name: county_mean(
            np.asarray(data[name], np.float32).reshape(1, -1), index, value)[0]
            for name in names}
        return position, counties

    values = {name: np.full((len(keys), index.shape[0]), np.nan, np.float32)
              for name in names}
    with ThreadPoolExecutor(workers) as pool:
        for position, counties in pool.map(one, enumerate(keys)):
            if counties is None:
                continue
            for name in names:
                values[name][position] = counties[name]
    if empty:
        _LOG.warning("%d of %d windows had no source image: %s%s", len(empty), len(keys),
                     empty[:5], " ..." if len(empty) > 5 else "")
    return values


def _fill_steps(series, steps, composites):
    """Scatter composite-row series ``(composites, counties)`` onto the 8-day axis."""
    out = np.full((series.shape[1], N_STEPS), np.nan, np.float32)
    out[:, steps] = series.T[:, composites]
    return out


# --- variables -------------------------------------------------------------
#
# Every variable function takes the padded county weights and returns
# ``(counties, N_STEPS)`` float32 arrays already placed on the 8-day axis.

def ndvi_modis(data_dir, workers, index, value):
    """MODIS MOD13A2 NDVI with cloudy/snow pixels dropped, on 8-day steps."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "ndvi_modis"
    keys, starts = _collection_index(MODIS_NDVI, "2000-01-01", "2026-01-01")

    def build(key):
        image = ee.Image(f"{MODIS_NDVI}/{key}")
        return image.select("NDVI").updateMask(image.select("SummaryQA").lte(1)) \
            .rename("ndvi")

    values = _run(keys, build, ["ndvi"], cache, workers, index, value)
    ndvi = _masked(values["ndvi"], 1e-4, fill=-3000, high=10000)
    steps, composites = assign_steps(starts, 16)
    return _fill_steps(ndvi, steps, composites), {
        "asset": MODIS_NDVI, "dates": [str(starts[0]), str(starts[-1])],
        "composites": len(keys), "revisit_days": 16,
    }


def ndvi_avhrr(data_dir, workers, index, value):
    """NOAA AVHRR NDVI as an 8-day maximum composite (0.0001 units).

    Only steps that start before the record ends are fetched; the rest stay NaN.
    """
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "ndvi_avhrr"
    _, starts = _collection_index(AVHRR_NDVI, "1981-01-01", "2023-01-01")
    steps = [step for step in range(N_STEPS)
             if np.searchsorted(starts, step_bounds(step)[0]) < np.searchsorted(
                 starts, step_bounds(step)[1])]

    def build(step):
        begin, end = step_bounds(step)
        return ee.ImageCollection(AVHRR_NDVI).filterDate(str(begin), str(end)) \
            .select("NDVI").max().rename("ndvi")

    values = _run(steps, build, ["ndvi"], cache, workers, index, value)
    ndvi = _masked(values["ndvi"], 1e-4, fill=-3000, high=10000)
    out = np.full((ndvi.shape[1], N_STEPS), np.nan, np.float32)
    out[:, steps] = ndvi.T
    return out, {"asset": AVHRR_NDVI, "steps": len(steps),
                 "dates": [str(step_bounds(steps[0])[0]),
                           str(step_bounds(steps[-1])[0])]}


def ndvi_group(data_dir, workers, index, value):
    """MODIS NDVI with adjusted AVHRR before 2000, plus the source flag."""
    modis, modis_meta = ndvi_modis(data_dir, workers, index, value)
    avhrr, avhrr_meta = ndvi_avhrr(data_dir, workers, index, value)
    _LOG.info("Fitting AVHRR to MODIS over their overlap")
    adjusted, slope, intercept, r2, pairs = adjust_avhrr(avhrr, modis)
    valid = np.isfinite(modis)
    combined = np.where(valid, modis, adjusted).astype(np.float32)
    source = np.where(valid, 0.0,
                      np.where(np.isfinite(adjusted), 1.0, np.nan)).astype(np.float32)
    _write_fit(data_dir, slope, intercept, r2, pairs)
    return ({"ndvi": combined, "ndvi_source": source},
            {"modis": modis_meta, "avhrr": avhrr_meta,
             "adjustment": {
                 "method": "per-county OLS of MODIS on AVHRR over their overlap",
                 "counties": int((pairs >= 40).sum()),
                 "median_slope": float(np.nanmedian(slope)),
                 "median_intercept": float(np.nanmedian(intercept)),
                 "median_r2": float(np.nanmedian(r2)),
                 "median_pairs": float(np.median(pairs)),
                 "counties_below_r2_0.3": int((r2 < 0.3).sum()),
                 "counties_below_r2_0.5": int((r2 < 0.5).sum()),
                 "counties_using_median_fit": int((pairs < 40).sum()),
                 "per_county_file": "processed/eo/ndvi_adjustment.parquet",
             }})


FIT_NAME = "ndvi_adjustment.parquet"


def _write_fit(data_dir, slope, intercept, r2, pairs):
    """Per-county AVHRR-to-MODIS fit, so users can drop weakly adjusted counties."""
    fit = pd.DataFrame({"adm_id": county_order(data_dir), "slope": slope,
                        "intercept": intercept, "r2": r2, "pairs": pairs})
    fit["uses_median_fit"] = fit.pairs < 40
    path = Path(data_dir) / "processed" / "eo" / FIT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(path.with_name(path.name + ".part"), index=False)
    path.with_name(path.name + ".part").replace(path)
    return path


def _cached_series(cache, index, value, workers, name="ndvi"):
    """County series rebuilt from the raw cache alone, keyed by the file stems."""
    cache = Path(cache)
    keys = sorted(int(path.stem) if path.stem.isdigit() else path.stem
                  for path in cache.glob("*.npy"))

    def build(_key):
        raise RuntimeError(f"{cache} is incomplete: {_key} has no cached composite")

    values = _run(keys, build, [name], cache, workers, index, value)
    return keys, _masked(values[name], 1e-4, fill=-3000, high=10000)


def ndvi_fit(data_dir=Path("data"), workers=4):
    """Rebuild the per-county AVHRR-to-MODIS fit from cached composites only.

    ``ndvi_group`` does the same after fetching, but this needs no Earth Engine
    calls at all, which matters while the project quota is exhausted.
    """
    data_dir = Path(data_dir)
    table, order, _ = load_weights(data_dir)
    index, value = weight_matrix(table, order)
    modis_starts, modis = _cached_series(
        data_dir / "raw" / "eo" / "ndvi_modis", index, value, workers)
    avhrr_steps, avhrr = _cached_series(
        data_dir / "raw" / "eo" / "ndvi_avhrr", index, value, workers)
    starts = pd.to_datetime([str(key).replace("_", "-") for key in modis_starts])
    steps, composites = assign_steps(starts, 16)
    modis_series = _fill_steps(modis, steps, composites)
    avhrr_series = np.full((avhrr.shape[1], N_STEPS), np.nan, np.float32)
    avhrr_series[:, avhrr_steps] = avhrr.T
    _LOG.info("Fitting AVHRR to MODIS from %d and %d cached composites",
              modis.shape[0], avhrr.shape[0])
    _, slope, intercept, r2, pairs = adjust_avhrr(avhrr_series, modis_series)
    path = _write_fit(data_dir, slope, intercept, r2, pairs)
    return {"path": str(path), "counties": int(np.isfinite(r2).sum()),
            "median_r2": float(np.nanmedian(r2)),
            "median_slope": float(np.nanmedian(slope)),
            "median_intercept": float(np.nanmedian(intercept)),
            "median_pairs": float(np.median(pairs)),
            "below_r2_0.3": int((r2 < 0.3).sum()),
            "below_r2_0.5": int((r2 < 0.5).sum()),
            "uses_median_fit": int((pairs < 40).sum())}


def adjust_avhrr(avhrr, modis, min_pairs=40):
    """Per-county linear fit of MODIS NDVI on AVHRR NDVI over their overlap.

    Returns the adjusted AVHRR series plus per-county slope, intercept, R^2 and the
    number of paired steps.  Counties with fewer than ``min_pairs`` pairs take the
    median fit, so the returned series has no NaN that the input did not have.
    """
    counties = avhrr.shape[0]
    slope = np.full(counties, np.nan)
    intercept = np.full(counties, np.nan)
    r2 = np.full(counties, np.nan)
    pairs = np.zeros(counties, np.int32)
    block = 256
    for begin in range(0, counties, block):
        end = min(begin + block, counties)
        x = avhrr[begin:end].astype(float)
        y = modis[begin:end].astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        count = valid.sum(axis=1)
        xv = np.where(valid, x, 0.0)
        yv = np.where(valid, y, 0.0)
        sx, sy = xv.sum(axis=1), yv.sum(axis=1)
        sxx, sxy, syy = (xv * xv).sum(axis=1), (xv * yv).sum(axis=1), (yv * yv).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = count * sxy - sx * sy
            var_x = count * sxx - sx * sx
            var_y = count * syy - sy * sy
            b = np.where(var_x > 0, cov / var_x, np.nan)
            a = (sy - b * sx) / count
            r = np.where((var_x > 0) & (var_y > 0), cov / np.sqrt(var_x * var_y), np.nan)
        enough = (count >= min_pairs) & np.isfinite(b) & np.isfinite(a)
        rows = begin + np.flatnonzero(enough)
        slope[rows] = b[enough]
        intercept[rows] = a[enough]
        r2[rows] = r[enough] ** 2
        pairs[begin:end] = count
    healthy = np.isfinite(slope) & np.isfinite(intercept)
    if healthy.any():
        fallback = (float(np.median(slope[healthy])),
                    float(np.median(intercept[healthy])))
    else:
        fallback = (1.0, 0.0)
    slope = np.where(np.isfinite(slope), slope, fallback[0])
    intercept = np.where(np.isfinite(intercept), intercept, fallback[1])
    adjusted = intercept[:, None] + slope[:, None] * np.asarray(avhrr, np.float64)
    return (np.clip(adjusted, 0.0, 1.0).astype(np.float32), slope, intercept, r2, pairs)


def lai_fpar(data_dir, workers, index, value):
    """MOD15A2H LAI (0.1) and FPAR (0.01) on 8-day steps."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "lai_fpar"
    keys, starts = _collection_index(MODIS_LAI, "2000-01-01", "2026-01-01")

    def build(key):
        return ee.Image(f"{MODIS_LAI}/{key}").select(["Lai_500m", "Fpar_500m"])

    def mask(data):
        # 249 cloud, 250 other reasons, 255 fill; valid LAI and FPAR are 0..100.
        return {"Lai_500m": _dn_mask(data["Lai_500m"], low=0, high=100),
                "Fpar_500m": _dn_mask(data["Fpar_500m"], low=0, high=100)}

    values = _run(keys, build, ["Lai_500m", "Fpar_500m"], cache, workers, index, value,
                  mask=mask)
    lai = values["Lai_500m"] * 0.1
    fpar = values["Fpar_500m"] * 0.01
    steps, composites = assign_steps(starts, 8)
    return ({"lai": _fill_steps(lai, steps, composites),
             "fpar": _fill_steps(fpar, steps, composites)},
            {"asset": MODIS_LAI, "dates": [str(starts[0]), str(starts[-1])],
             "composites": len(keys), "revisit_days": 8})


def evapotranspiration(data_dir, workers, index, value):
    """MOD16A2GF ET (0.1 kg m-2 per 8 days) as mm/day."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "et"
    keys, starts = _collection_index(MODIS_ET, "2000-01-01", "2026-01-01")

    def build(key):
        return ee.Image(f"{MODIS_ET}/{key}").select("ET")

    values = _run(keys, build, ["ET"], cache, workers, index, value,
                  mask=lambda data: {"ET": _dn_mask(data["ET"], high=32760)})
    # One kg m-2 of water is 1 mm; the composite spans 8 days.
    et = values["ET"] * (0.1 / 8)
    steps, composites = assign_steps(starts, 8)
    return _fill_steps(et, steps, composites), {
        "asset": MODIS_ET, "dates": [str(starts[0]), str(starts[-1])],
        "composites": len(keys), "revisit_days": 8,
    }


def land_surface_temperature(data_dir, workers, index, value):
    """MOD11A2 daytime and nighttime LST in degrees C (0.02 K units)."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "lst"
    keys, starts = _collection_index(MODIS_LST, "2000-01-01", "2026-01-01")

    def build(key):
        return ee.Image(f"{MODIS_LST}/{key}").select(["LST_Day_1km", "LST_Night_1km"])

    def mask(data):
        return {name: _dn_mask(data[name], low=7500, high=65535)
                for name in ("LST_Day_1km", "LST_Night_1km")}

    values = _run(keys, build, ["LST_Day_1km", "LST_Night_1km"], cache, workers,
                  index, value, mask=mask)
    steps, composites = assign_steps(starts, 8)
    result = {}
    for name, band in (("lst_day", "LST_Day_1km"), ("lst_night", "LST_Night_1km")):
        kelvin = values[band] * 0.02
        result[name] = _fill_steps(kelvin - 273.15, steps, composites)
    return result, {"asset": MODIS_LST, "dates": [str(starts[0]), str(starts[-1])],
                    "composites": len(keys), "revisit_days": 8}


def soil_moisture_smap(data_dir, workers, index, value):
    """SPL4SMGP v008 root-zone and surface soil moisture, 8-day means in m3/m3."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "smap"
    parts = {"sm_rootzone": "smap_rootzone", "sm_surface": "smap_surface"}
    bands = list(parts)

    def build(step):
        begin, end = step_bounds(step)
        collection = ee.ImageCollection(SMAP).filterDate(str(begin), str(end)) \
            .select(bands)
        return collection.map(lambda image: image.updateMask(image.neq(-9999))).mean()

    steps = [step for step in range(N_STEPS) if step >= step_of("2015-03-31")]
    values = _run(steps, build, bands, cache, workers, index, value)
    out = {}
    for band, name in parts.items():
        series = _masked(values[band], 1.0, fill=-9999, low=0.0, high=1.0)
        out[name] = np.full((series.shape[1], N_STEPS), np.nan, np.float32)
        out[name][:, steps] = series.T
    return out, {"asset": SMAP, "steps": len(steps),
                 "dates": [str(step_bounds(steps[0])[0]),
                           str(step_bounds(steps[-1])[0])]}


def soil_moisture_gldas(data_dir, workers, index, value):
    """GLDAS-2.1 Noah 0-100 cm root-zone soil moisture (kg m-2 -> m3/m3).

    Each layer is divided by ``thickness * 1000 kg m-3`` to a volumetric fraction,
    then depth weighted (0.1, 0.3, 0.6) over 0-100 cm.
    """
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "gldas"
    names = [name for name, _ in GLDAS_LAYERS]

    def build(step):
        begin, end = step_bounds(step)
        collection = ee.ImageCollection(GLDAS).filterDate(str(begin), str(end)) \
            .select(names)
        return collection.map(lambda image: image.updateMask(image.neq(-9999))).mean()

    steps = [step for step in range(N_STEPS) if step >= step_of("2000-01-01")]
    values = _run(steps, build, names, cache, workers, index, value)
    layers = {name: _masked(values[name], 1.0 / (thickness * 1000.0), fill=-9999,
                            low=0.0, high=0.75)
              for name, thickness in GLDAS_LAYERS}
    total = np.zeros(layers[names[0]].shape, np.float32)
    weight = np.zeros_like(total)
    for name, thickness in GLDAS_LAYERS:
        finite = np.isfinite(layers[name])
        total += np.where(finite, np.nan_to_num(layers[name]) * thickness, 0).astype(
            np.float32)
        weight += np.where(finite, thickness, 0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.where(weight > 0, total / np.where(weight > 0, weight, 1), np.nan)
    out = {"gldas_rootzone": np.full((root.shape[1], N_STEPS), np.nan, np.float32)}
    out["gldas_rootzone"][:, steps] = root.T.astype(np.float32)
    return out, {"asset": GLDAS, "steps": len(steps),
                 "dates": [str(step_bounds(steps[0])[0]),
                           str(step_bounds(steps[-1])[0])]}


def terrestrial_water_storage(data_dir, workers, index, value):
    """GRACE mascon terrestrial water storage anomaly (cm), assigned by month."""
    import ee

    cache = Path(data_dir) / "raw" / "eo" / "grace"
    keys = ee.ImageCollection(GRACE).aggregate_array("system:index").getInfo()

    def build(key):
        return ee.Image(f"{GRACE}/{key}").select("lwe_thickness")

    values = _run(keys, build, ["lwe_thickness"], cache, workers, index, value)
    series = _masked(values["lwe_thickness"], 1.0, fill=-9999)
    months = {str(key)[:6]: position for position, key in enumerate(keys)}
    out = np.full((series.shape[1], N_STEPS), np.nan, np.float32)
    observed = np.zeros(N_STEPS, np.float32)
    for step in range(N_STEPS):
        month = str(step_bounds(step)[0])[:7].replace("-", "")
        position = months.get(month)
        if position is not None:
            out[:, step] = series[position]
            observed[step] = 1.0
    # The flag is step level; broadcasting keeps every part the same shape.
    flag = np.broadcast_to(observed, out.shape).astype(np.float32).copy()
    return {"grace_tws": out, "grace_observed": flag}, {
        "asset": GRACE, "composites": len(keys),
        "dates": [str(keys[0])[:8], str(keys[-1])[:8]],
    }

VARIABLES = {
    "ndvi": ("1", f"{MODIS_NDVI} NDVI, SummaryQA <= 1; {AVHRR_NDVI} adjusted"),
    "ndvi_source": ("code", "0 MODIS, 1 adjusted AVHRR"),
    "lai": ("m2/m2", f"{MODIS_LAI} Lai_500m x 0.1"),
    "fpar": ("1", f"{MODIS_LAI} Fpar_500m x 0.01"),
    "et_actual": ("mm/day", f"{MODIS_ET} ET x 0.1 kg m-2 per 8 days"),
    "lst_day": ("degC", f"{MODIS_LST} LST_Day_1km x 0.02 K"),
    "lst_night": ("degC", f"{MODIS_LST} LST_Night_1km x 0.02 K"),
    "smap_rootzone": ("m3/m3", f"{SMAP} sm_rootzone, 8-day mean"),
    "smap_surface": ("m3/m3", f"{SMAP} sm_surface, 8-day mean"),
    "gldas_rootzone": ("m3/m3", f"{GLDAS} 0-100 cm depth-weighted, 8-day mean"),
    "grace_tws": ("cm", f"{GRACE} lwe_thickness"),
    "grace_observed": ("flag", "1 where a GRACE month covers the step"),
}
ORDER = list(VARIABLES)

# Physical bounds of a county mean.  A published value outside them means a fill
# code or a scaling error leaked through, so publishing fails instead of shipping it.
PHYSICAL_BOUNDS = {
    "ndvi": (-1.0, 1.0), "ndvi_source": (0.0, 1.0), "lai": (0.0, 8.0),
    "fpar": (0.0, 1.0), "et_actual": (0.0, 15.0), "lst_day": (-70.0, 70.0),
    "lst_night": (-70.0, 70.0), "smap_rootzone": (0.0, 1.0),
    "smap_surface": (0.0, 1.0), "gldas_rootzone": (0.0, 0.75),
    "grace_tws": (-500.0, 500.0), "grace_observed": (0.0, 1.0),
}


def check_bounds(arrays):
    """Raise when a county value is outside its physical bounds."""
    problems = []
    for name, array in arrays.items():
        if name not in PHYSICAL_BOUNDS:
            continue
        low, high = PHYSICAL_BOUNDS[name]
        valid = np.isfinite(array)
        outside = valid & ((array < low) | (array > high))
        if outside.any():
            problems.append({"variable": name, "count": int(outside.sum()),
                             "min": float(np.nanmin(array)),
                             "max": float(np.nanmax(array)), "bounds": [low, high]})
    if problems:
        raise ValueError(f"values outside physical bounds: {problems}")
    return True


# --- weights ---------------------------------------------------------------

def cultivated_cells(data_dir=Path("data"), grid=None):
    """Cultivated fraction of every 0.05 degree cell, from the CDL mask.

    The binary mask is sampled at 0.025 degrees and averaged over each 2 x 2 block,
    which makes the cell value a local cultivated fraction instead of one 500 m
    pixel's flag.
    """

    fine = ee_grid(rows=HEIGHT * 2, cols=WIDTH * 2, cell=GRID_CELL / 2)
    _LOG.info("Fetching cultivated fraction raster")
    data = fetch(cultivated_mask().toFloat().rename("cultivated"), ["cultivated"],
                 grid=fine)
    fine_values = np.asarray(data["cultivated"], np.float32).reshape(HEIGHT, 2, WIDTH, 2)
    with np.errstate(invalid="ignore"):
        weights = np.nanmean(fine_values, axis=(1, 3))
    rows, cols = np.meshgrid(np.arange(HEIGHT), np.arange(WIDTH), indexing="ij")
    return pd.DataFrame({"cell_row": rows.ravel().astype(np.int16),
                         "cell_col": cols.ravel().astype(np.int16),
                         "cultivated": np.clip(weights.ravel(), 0.0, 1.0)})


def load_weights(data_dir=Path("data"), force=False, project=None):
    """Cached county x cell weight table, its counties list and area-only list."""
    path = Path(data_dir) / "processed" / "eo" / WEIGHTS_NAME
    order = county_order(data_dir)
    if path.exists() and not force:
        table = pd.read_parquet(path)
        area_only = sorted(table.loc[table.area_only, "adm_id"].astype(str).unique())
        return table, order, area_only
    initialize(project)
    counties = county_frame(data_dir)
    _LOG.info("County x cell intersections for %d counties", len(counties))
    areas = area_weights(counties, grid_cells())
    _LOG.info("Pairs: %d", len(areas))
    cultivated = cultivated_cells(data_dir)
    table, area_only = combine_weights(areas, cultivated, order)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)
    _LOG.info("Weights: %d pairs, %d area-only counties", len(table), len(area_only))
    return table, order, area_only


# --- parts, cube and loaders ----------------------------------------------

def _part_paths(data_dir):
    parts = Path(data_dir) / "processed" / "eo" / PARTS
    parts.mkdir(parents=True, exist_ok=True)
    return parts


def save_part(data_dir, name, array, meta):
    """Store one variable's county series plus its metadata, atomically."""
    if array.shape != (len(county_order(data_dir)), N_STEPS):
        raise ValueError(f"{name}: unexpected shape {array.shape}")
    parts = _part_paths(data_dir)
    path = parts / f"{name}.npy"
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array, np.float32))
    temporary.replace(path)
    valid = np.isfinite(array)
    steps = np.flatnonzero(valid.any(axis=0))
    record = dict(meta)
    record.update({
        "name": name, "units": VARIABLES[name][0], "nan_fraction": float(1 - valid.mean()),
        "first_valid": str(step_bounds(steps[0])[0]) if steps.size else None,
        "last_valid": str(step_bounds(steps[-1])[0]) if steps.size else None,
    })
    (parts / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def available_parts(data_dir):
    """Variable names with a stored series, in the canonical order."""
    parts = _part_paths(data_dir)
    return [name for name in ORDER if (parts / f"{name}.npy").exists()]


def write_cube(path, arrays, counties, variables, extra=None):
    """Write a float32 county cube and its JSON metadata, atomically.

    ``arrays`` maps variable name to a ``(counties, N_STEPS)`` array; the column
    order is the mapping's order.  The cube is written under a ``.part`` name and
    renamed, so a reader never sees a half-written file.
    """
    path = Path(path)
    names = list(arrays)
    if path.exists() and path.with_suffix(".json").exists():
        previous = json.loads(path.with_suffix(".json").read_text())
        if previous.get("counties") != list(counties):
            raise ValueError(f"{path} holds a different county order")
    metadata = {
        "shape": [len(counties), N_STEPS, len(names)],
        "dtype": "float32",
        "order": "C",
        "counties": list(counties),
        "steps_per_year": STEPS_PER_YEAR,
        "step_days": STEP_DAYS,
        "start_date": str(step_bounds(0)[0]),
        "end_date": str(step_bounds(N_STEPS - 1)[1]),
        "step_start_dates": [str(value) for value in step_starts()],
        "variables": [variables[name] for name in names],
    }
    metadata.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    cube = np.memmap(temporary, dtype="<f4", mode="w+",
                     shape=(len(counties), N_STEPS, len(names)))
    for position, name in enumerate(names):
        cube[:, :, position] = arrays[name]
    cube.flush()
    del cube
    temporary.replace(path)
    json_path = path.with_suffix(".json")
    json_temporary = json_path.with_name(json_path.name + ".part")
    json_temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    json_temporary.replace(json_path)
    return metadata


def publish(data_dir=Path("data"), notes=None):
    """Rewrite the cube and metadata from every stored part, atomically."""
    data_dir = Path(data_dir)
    out = data_dir / "processed" / "eo"
    names = available_parts(data_dir)
    if not names:
        raise FileNotFoundError("No variable parts to publish")
    order = county_order(data_dir)
    _, _, area_only = load_weights(data_dir)
    parts = _part_paths(data_dir)
    arrays = {name: np.load(parts / f"{name}.npy") for name in names}
    check_bounds(arrays)
    variables = {name: json.loads((parts / f"{name}.json").read_text()) for name in names}
    extra = {
        "grid": {"crs": "EPSG:4326", "resolution_deg": GRID_CELL,
                 "lat_range": [GRID_BOTTOM, GRID_TOP],
                 "lon_range": [GRID_LEFT, GRID_RIGHT],
                 "shape": [HEIGHT, WIDTH], "n_cells": HEIGHT * WIDTH,
                 "resampling": "bilinear from the native grid, not box averaged"},
        "weights": {"file": f"processed/eo/{WEIGHTS_NAME}",
                    "pairs": len(pd.read_parquet(out / WEIGHTS_NAME,
                                                 columns=["weight"])),
                    "min_cultivated_share": MIN_CULTIVATED,
                    "cropland": "USDA/NASS/CDL classes outside power.NON_CULTIVATED, "
                                "kept where at least half of 2008-2022 was cropland"},
        "area_only_counties": area_only,
        "notes": notes or [],
    }
    return write_cube(out / CUBE_NAME, arrays, order, variables, extra)


def load_eo(path=None):
    """Open the float32 cube and its JSON metadata as ``(memmap, dict)``."""
    root = Path(path) if path else Path("data") / "processed" / "eo"
    path = root / CUBE_NAME if root.is_dir() else root
    metadata = json.loads(path.with_suffix(".json").read_text())
    cube = np.memmap(path, dtype="<f4", mode="r", shape=tuple(metadata["shape"]))
    return cube, metadata


def county_series(cube, metadata, adm_id):
    """One county's ``(steps, variables)`` block with the step start dates."""
    row = metadata["counties"].index(adm_id)
    names = [variable["name"] for variable in metadata["variables"]]
    frame = pd.DataFrame(np.asarray(cube[row]), columns=names)
    frame.insert(0, "step", np.arange(len(frame)))
    frame.insert(1, "start_date", pd.to_datetime(metadata["step_start_dates"]))
    return frame


# --- build driver ----------------------------------------------------------

def build_variable(name, data_dir=Path("data"), workers=6, project=None):
    """Fetch one variable group, reduce it to counties and store its parts."""
    data_dir = Path(data_dir)
    initialize(project)
    table, order, _ = load_weights(data_dir)
    index, value = weight_matrix(table, order)
    groups = {
        "ndvi": ndvi_group, "lai": lai_fpar, "et_actual": evapotranspiration,
        "lst": land_surface_temperature, "smap": soil_moisture_smap,
        "gldas": soil_moisture_gldas, "grace": terrestrial_water_storage,
    }
    if name not in groups:
        raise ValueError(f"Unknown variable group {name}")
    began = time.monotonic()
    block, meta = groups[name](data_dir, workers, index, value)
    parts = block if isinstance(block, dict) else {name: block}
    records = []
    for key, array in parts.items():
        records.append(save_part(data_dir, key, array, meta))
        _LOG.info("%s: %.1f%% NaN, %s..%s, %.0f s", key,
                  100 * records[-1]["nan_fraction"], records[-1]["first_valid"],
                  records[-1]["last_valid"], time.monotonic() - began)
    return records




# --- validation ------------------------------------------------------------

SEASONS = {"jja": ("06-01", "09-01"), "jul_aug": ("07-01", "09-01")}


def summer_means(cube, metadata, adm_id, years=(2011, 2012, 2013),
                 variables=("ndvi", "lai", "et_actual", "lst_day", "smap_rootzone",
                            "gldas_rootzone")):
    """June-August and July-August county means of the requested variables by year.

    The drought signal of 2012 sits in late summer; June was warm and early that
    year, so the June-August mean alone understates it.
    """
    row = metadata["counties"].index(adm_id)
    names = [variable["name"] for variable in metadata["variables"]]
    dates = pd.to_datetime(metadata["step_start_dates"])
    month_day = dates.strftime("%m-%d").to_numpy()
    result = {}
    for name in variables:
        if name not in names:
            continue
        series = pd.Series(np.asarray(cube[row, :, names.index(name)], float), index=dates)
        result[name] = {}
        for label, (low, high) in SEASONS.items():
            inside = (month_day >= low) & (month_day < high)
            result[name][label] = {
                str(year): float(series[inside & (series.index.year == year)].mean())
                for year in years}
    return result


def _correlation(x, y):
    """Pearson correlation of paired rows, NaN where either is constant or short."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=1)
    xv = np.where(valid, x, 0.0)
    yv = np.where(valid, y, 0.0)
    sx, sy = xv.sum(axis=1), yv.sum(axis=1)
    sxx, syy, sxy = (xv * xv).sum(axis=1), (yv * yv).sum(axis=1), (xv * yv).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt((count * sxx - sx * sx) * (count * syy - sy * sy))
        value = (count * sxy - sx * sy) / spread
    return np.where((count >= 30) & (spread > 0), value, np.nan), count


def cybench_ndvi_series(data_dir=Path("data"), counties_order=None, paths=None):
    """CY-Bench daily county NDVI averaged onto the 8-day steps."""
    order = counties_order or county_order(data_dir)
    paths = paths or [Path(data_dir) / "raw" / "cybench" / "cybench-data" / crop / "US" /
                      f"ndvi_{crop}_US.csv" for crop in ("maize", "wheat")]
    pieces = [pd.read_csv(path, usecols=["adm_id", "date", "ndvi"])
              for path in paths if Path(path).exists()]
    if not pieces:
        raise FileNotFoundError("No CY-Bench NDVI files found")
    frame = pd.concat(pieces, ignore_index=True).drop_duplicates(["adm_id", "date"])
    dates = pd.to_datetime(frame.date.astype(str), format="%Y%m%d")
    frame["step"] = ((dates.dt.year.to_numpy() - FIRST_YEAR) * STEPS_PER_YEAR
                     + (dates.dt.dayofyear.to_numpy() - 1) // STEP_DAYS)
    frame["row"] = pd.Categorical(frame.adm_id, categories=order).codes
    frame = frame.loc[frame.row >= 0]
    grouped = frame.groupby(["row", "step"], observed=True).ndvi.mean()
    out = np.full((len(order), N_STEPS), np.nan, np.float32)
    out[grouped.index.get_level_values(0).to_numpy(),
        grouped.index.get_level_values(1).to_numpy()] = grouped.to_numpy()
    return out


def _window_mask(metadata, first, last):
    dates = pd.to_datetime(metadata["step_start_dates"])
    return (dates >= pd.Timestamp(first)) & (dates < pd.Timestamp(last))


def validate(data_dir=Path("data")):
    """Story County drought check plus the CY-Bench, POWER and NaN comparisons."""
    data_dir = Path(data_dir)
    cube, metadata = load_eo(data_dir / "processed" / "eo")
    names = [variable["name"] for variable in metadata["variables"]]
    report = {"story_county_iowa": summer_means(cube, metadata, "US-19-169")}
    report["nan_fraction"] = {variable["name"]: variable["nan_fraction"]
                              for variable in metadata["variables"]}
    report["first_last_valid"] = {
        variable["name"]: [variable["first_valid"], variable["last_valid"]]
        for variable in metadata["variables"]}

    if "ndvi" in names:
        theirs = cybench_ndvi_series(data_dir, metadata["counties"])
        window = _window_mask(metadata, "2003-01-01", "2024-01-01")
        ours = np.asarray(cube[:, :, names.index("ndvi")], float)
        correlation, _ = _correlation(ours[:, window], theirs[:, window])
        bias = np.nanmean(ours[:, window] - theirs[:, window], axis=1)
        report["cybench_ndvi"] = {
            "window": "2003-2023", "counties": int(np.isfinite(correlation).sum()),
            "median_r": float(np.nanmedian(correlation)),
            "quartiles": [float(np.nanquantile(correlation, 0.25)),
                          float(np.nanquantile(correlation, 0.75))],
            "median_bias": float(np.nanmedian(bias)),
        }
    if "smap_rootzone" in names:
        weather, weather_meta = power.load_county_daily(
            data_dir / "processed" / "weather" / power.CUBE_NAME)
        steps = np.flatnonzero(_window_mask(metadata, "2016-01-01", "2025-01-01"))
        dates = pd.to_datetime(metadata["step_start_dates"])
        origin = pd.Timestamp(weather_meta["start_date"])
        column = [item["name"] for item in weather_meta["variables"]].index("gwetroot")
        gwet = np.full((len(metadata["counties"]), len(steps)), np.nan, np.float32)
        for position, step in enumerate(steps):
            offset = (dates[step] - origin).days
            window = weather[:, offset:offset + steps_covered(int(step)), column]
            gwet[:, position] = np.nanmean(window, axis=1)
        ours = np.asarray(cube[:, steps, names.index("smap_rootzone")], float)
        correlation, _ = _correlation(ours, gwet)
        report["smap_vs_power_gwetroot"] = {
            "window": "2016-2024", "counties": int(np.isfinite(correlation).sum()),
            "median_r": float(np.nanmedian(correlation)),
            "quartiles": [float(np.nanquantile(correlation, 0.25)),
                          float(np.nanquantile(correlation, 0.75))],
        }
    path = data_dir / "processed" / "eo" / "eo_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def quality(data_dir=Path("data")):
    """Missingness, gap lengths and stitching fit for every published variable.

    County-level NaN fractions are the masked-out share after each product's own
    quality band, which is the form the model sees; the grid is sampled bilinearly
    so it never adds values, it only spreads the native grid.
    """
    data_dir = Path(data_dir)
    out = data_dir / "processed" / "eo"
    metadata = json.loads((out / META_NAME).read_text())
    order = metadata["counties"]
    parts = _part_paths(data_dir)
    years = np.arange(FIRST_YEAR, LAST_YEAR + 1)
    step_dates = np.array(metadata["step_start_dates"], dtype="datetime64[D]")
    report = {"variables": {}, "steps": int(N_STEPS), "counties": len(order),
              "start_date": metadata["start_date"], "end_date": metadata["end_date"],
              "time_step": f"8 days ({STEPS_PER_YEAR} per year, last step short)"}
    for item in metadata["variables"]:
        name = item["name"]
        array = np.load(parts / f"{name}.npy")
        missing = np.isnan(array)
        per_year = missing.reshape(len(order), len(years), STEPS_PER_YEAR).mean(axis=(0, 2))
        gaps = np.zeros(len(order), np.float32)
        for row in range(len(order)):
            gaps[row] = _longest_run(missing[row])
        present = np.isfinite(array).sum(axis=1)
        report["variables"][name] = {
            "units": item.get("units"),
            "source": {key: value for key, value in item.items()
                       if key in ("asset", "asset_filled", "modis", "avhrr",
                                  "adjustment", "bands", "revisit_days", "steps")},
            "masking": item.get("masking"),
            "nan_fraction": round(float(missing.mean()), 6),
            "nan_fraction_by_year": {int(year): round(float(value), 5)
                                     for year, value in zip(years, per_year, strict=True)},
            "counties_with_data": int((present > 0).sum()),
            "steps_with_data": np.flatnonzero(~missing.all(axis=0)).size,
            "longest_gap_steps": {"median": float(np.median(gaps)),
                                  "p90": float(np.quantile(gaps, 0.9)),
                                  "max": float(gaps.max()),
                                  "max_days": float(gaps.max() * STEP_DAYS)},
            "first_valid": item.get("first_valid"),
            "last_valid": item.get("last_valid"),
            "all_missing_ranges": [
                [str(step_dates[first]), str(step_dates[last])]
                for first, last in _runs(missing.all(axis=0))],
        }
        if name == "ndvi":
            source = np.load(parts / "ndvi_source.npy")
            avhrr = (source == 1.0).reshape(len(order), len(years), STEPS_PER_YEAR)
            report["variables"][name]["avhrr_share_by_year"] = {
                int(year): round(float(value), 3)
                for year, value in zip(years, avhrr.mean(axis=(0, 2)), strict=True)}
    report["grid"] = metadata["grid"]
    check = out / "eo_grid_check.json"
    if check.exists():
        report["grid_check"] = json.loads(check.read_text())
    report["weights"] = metadata["weights"]
    report["area_only_counties"] = metadata["area_only_counties"]
    report["step_start_dates"] = [str(value) for value in
                                  [step_dates[0], step_dates[-1]]]
    report["accessed"] = dt.date.today().isoformat()
    (out / QC_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _runs(mask):
    """``(first, last)`` index pairs of every run of True in a boolean array."""
    if not np.asarray(mask).any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], np.asarray(mask).view(np.int8),
                                                   [0]))))
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist(), strict=True))


def _longest_run(missing):
    """Longest consecutive run of True in a 1-D boolean array."""
    if not missing.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], missing.view(np.int8), [0]))))
    return float((edges[1::2] - edges[0::2]).max())


GRID_CHECK_CASES = {
    # one mid-summer composite per product, inside the Corn Belt
    "ndvi": {"asset": MODIS_NDVI, "band": "NDVI", "qa": ("SummaryQA", 1),
             "date": ("2012-07-01", "2012-07-31"), "scale": 1e-4, "high": 10000},
    "lai": {"asset": MODIS_LAI, "band": "Lai_500m", "qa": None,
            "date": ("2012-07-01", "2012-07-31"), "scale": 0.1, "high": 100},
    "et_actual": {"asset": MODIS_ET, "band": "ET", "qa": None,
                  "date": ("2012-07-01", "2012-07-31"), "scale": 0.1 / 8,
                  "high": 32760},
}


def _box_grid(west, north, rows, cols, cell=GRID_CELL):
    """``PixelGrid`` for a box whose upper-left corner is (west, north)."""
    return {"crsCode": "EPSG:4326",
            "dimensions": {"width": int(cols), "height": int(rows)},
            "affineTransform": {"scaleX": cell, "scaleY": -cell,
                                "translateX": west, "translateY": north}}


def _decode(image, case, grid):
    """One band of ``image`` on ``grid`` as float, outside the valid window NaN."""
    data = fetch(image, [case["band"]], grid=grid)
    value = np.asarray(data[case["band"]], np.float32) * case["scale"]
    return np.where(value > case["high"] * case["scale"], np.nan, value)


def grid_check(data_dir=Path("data"), names=("ndvi", "lai", "et_actual"),
               box=((-95.0, -88.0), (40.0, 44.0)), project=None):
    """Compare the delivered 0.05 degree sampling with a strict native reduction.

    The delivered grid asks ``computePixels`` for 0.05 degree cells, so Earth
    Engine reads the image pyramid and applies the quality mask on pyramided
    values.  The strict alternative masks at the native scale, averages with
    ``reduceResolution`` and reprojects onto the same 0.05 degree grid.
    """
    import ee

    initialize(project)
    (west, east), (south, north) = box
    rows = round((north - south) / GRID_CELL)
    cols = round((east - west) / GRID_CELL)
    grid = _box_grid(west, north, rows, cols)
    lat, lon = cell_centers()
    inside = ((lon[None, :] >= west) & (lon[None, :] < east)
              & (lat[:, None] >= south) & (lat[:, None] < north))
    cell_of_row = inside.ravel()
    cells = np.flatnonzero(cell_of_row)
    table, order, _ = load_weights(data_dir)
    index, value = weight_matrix(table, order)
    report = {"box": {"west": west, "east": east, "south": south, "north": north,
                      "cells": [rows, cols]},
              "resampling": {
                  "delivered": "computePixels on the 0.05 degree grid (image "
                               "pyramid scale, bilinear)",
                  "strict": "native quality mask, reduceResolution mean, reproject "
                            "to 0.05 degrees (EPSG:4326)"},
              "variables": {}}
    for name in names:
        case = GRID_CHECK_CASES[name]
        first, last = case["date"]
        chosen = ee.ImageCollection(case["asset"]).filterDate(first, last).first()
        key = chosen.get("system:index").getInfo()
        chosen = ee.Image(f"{case['asset']}/{key}").select(case["band"])
        plain = _decode(chosen, case, grid).ravel()
        native = chosen
        if case["qa"]:
            qa = ee.Image(f"{case['asset']}/{key}").select(case["qa"][0])
            native = native.updateMask(qa.lte(case["qa"][1]))
        strict = _decode(
            native.reduceResolution(ee.Reducer.mean(), maxPixels=65536)
            .reproject(crs="EPSG:4326",
                       crsTransform=[GRID_CELL, 0, west, 0, -GRID_CELL, north]),
            case, grid).ravel()
        differ = np.abs(plain - strict)
        kept = np.isfinite(plain) & np.isfinite(strict)
        entry = {
            "composite": str(key),
            "cells_compared": int(kept.sum()),
            "cells_only_delivered": int((np.isfinite(plain) & ~np.isfinite(strict)).sum()),
            "cells_only_strict": int((~np.isfinite(plain) & np.isfinite(strict)).sum()),
            "mean_abs_diff": float(np.nanmean(differ)),
            "p95_abs_diff": float(np.nanpercentile(differ[kept], 95)) if kept.any()
            else None,
            "max_abs_diff": float(np.nanmax(differ)),
            "box_mean_delivered": float(np.nanmean(plain)),
            "box_mean_strict": float(np.nanmean(strict)),
        }
        # Same weights, restricted to the box, for the county-level comparison.
        picked = np.isin(index, cells)
        if picked.any():
            lookup = {cell: position for position, cell in enumerate(cells.tolist())}
            local = np.full(index.shape, -1, int)
            weight = np.zeros(index.shape, np.float32)
            local[picked] = [lookup[cell] for cell in index[picked]]
            weight[picked] = value[picked]
            keep_row = local.max(axis=1) >= 0
            county_plain = county_mean(plain.reshape(1, -1), local[keep_row],
                                       weight[keep_row])[0]
            county_strict = county_mean(strict.reshape(1, -1), local[keep_row],
                                        weight[keep_row])[0]
            finite = np.isfinite(county_plain) & np.isfinite(county_strict)
            if finite.sum() > 2:
                entry["counties"] = int(finite.sum())
                entry["county_mean_r"] = float(np.corrcoef(
                    county_plain[finite], county_strict[finite])[0, 1])
                entry["county_mean_bias"] = float(
                    np.mean(county_plain[finite] - county_strict[finite]))
                entry["county_mean_max_abs_diff"] = float(
                    np.max(np.abs(county_plain[finite] - county_strict[finite])))
        report["variables"][name] = entry
    path = Path(data_dir) / "processed" / "eo" / "eo_grid_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", action="append", default=[],
                        choices=["ndvi", "lai", "et_actual", "lst", "smap", "gldas",
                                 "grace"],
                        help="variable group to fetch (repeatable)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--weights", action="store_true",
                        help="rebuild the county weights only")
    parser.add_argument("--qc", action="store_true",
                        help="write the quality report for the published cube")
    parser.add_argument("--fit", action="store_true",
                        help="rebuild the per-county NDVI fit from the raw cache "
                             "without Earth Engine")
    parser.add_argument("--grid-check", action="store_true",
                        help="compare the delivered grid with a strict native "
                             "reduction on three mid-summer composites")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.weights:
        table, _, area_only = load_weights(args.data_dir, force=True,
                                           project=args.project)
        print(f"{len(table)} pairs, {len(area_only)} area-only counties")
        return
    if (not args.variable and not args.publish and not args.validate
            and not args.qc and not args.grid_check and not args.fit):
        parser.error("nothing to do")
    for name in args.variable:
        build_variable(name, args.data_dir, args.workers, args.project)
        publish(args.data_dir)
    if args.publish:
        metadata = publish(args.data_dir)
        print(json.dumps({"shape": metadata["shape"],
                          "variables": [item["name"]
                                        for item in metadata["variables"]]}))
    if args.validate:
        print(json.dumps(validate(args.data_dir), indent=2))
    if args.fit:
        print(json.dumps(ndvi_fit(args.data_dir, args.workers), indent=2))
    if args.grid_check:
        print(json.dumps(grid_check(args.data_dir, project=args.project),
                         indent=2))
    if args.qc:
        report = quality(args.data_dir)
        print(json.dumps({name: value["nan_fraction"]
                          for name, value in report["variables"].items()}))


if __name__ == "__main__":
    main()
