"""County daily weather for the contiguous US from NASA POWER cloud Zarr stores.

Anonymous HTTPS S3 Zarr v2 archives supply MERRA-2 daily meteorology and one
continuous shortwave radiation series (syn1deg, then SRB, then flashflux).  A
county value is the intersection-area x cultivated-share weighted mean of the
MERRA-2 0.5 x 0.625 degree cells it overlaps, so it describes farmland rather
than whole-county area.  Days without any radiation product (1981-1983, before
SRB starts) are filled with FAO-56 Hargreaves radiation and flagged in
``rad_filled``.

Units: tmin/tmax/tavg degC, prec and et0 and cwb mm/day, rad MJ m-2 day-1,
wind m s-1, pressure kPa, GWET* and rad_filled 0-1 fractions, vpd kPa.
Missing values stay NaN; nothing is zero-filled.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import shapely
import zarr

BUCKET = "nasa-power"
MERRA2 = "/merra2/temporal/power_merra2_daily_temporal_lst.zarr"
RADIATION = (  # highest preference first
    ("syn1deg", "/syn1deg/temporal/power_syn1deg_daily_temporal_lst.zarr"),
    ("srb", "/srb/temporal/power_srb_daily_temporal_lst.zarr"),
    ("flashflux", "/flashflux/temporal/power_flashflux_daily_temporal_lst.zarr"),
)
RADIATION_VAR = "ALLSKY_SFC_SW_DWN"
START_DATE = "1981-01-01"
END_DATE = "2025-12-31"
LAT_RANGE = (24.0, 50.0)
LON_RANGE = (-125.0, -66.0)
MERRA_VARIABLES = (
    ("tmin", "T2M_MIN"),
    ("tmax", "T2M_MAX"),
    ("tavg", "T2M"),
    ("prec", "PRECTOTCORR"),
    ("tdew", "T2MDEW"),
    ("rh", "RH2M"),
    ("wind", "WS2M"),
    ("ps", "PS"),
    ("gwetroot", "GWETROOT"),
    ("gwettop", "GWETTOP"),
    ("gwetprof", "GWETPROF"),
)
VARIABLES = (
    ("tmin", "degC", "MERRA-2 T2M_MIN, county mean"),
    ("tmax", "degC", "MERRA-2 T2M_MAX, county mean"),
    ("tavg", "degC", "MERRA-2 T2M daily mean, county mean"),
    ("prec", "mm/day", "MERRA-2 PRECTOTCORR, county mean"),
    ("rad", "MJ/m2/day", "POWER shortwave down, county mean"),
    ("rad_filled", "1", "1 where rad is Hargreaves-filled, else 0"),
    ("vpd", "kPa", "FAO-56 es - ea from tmax, tmin and T2MDEW"),
    ("et0", "mm/day", "FAO-56 Penman-Monteith eq. 6"),
    ("cwb", "mm/day", "prec - et0"),
    ("rh", "%", "MERRA-2 RH2M, county mean"),
    ("wind", "m/s", "MERRA-2 WS2M, county mean"),
    ("gwetroot", "1", "MERRA-2 GWETROOT, county mean (0-1 wetness)"),
    ("gwettop", "1", "MERRA-2 GWETTOP, county mean (0-1 wetness)"),
    ("gwetprof", "1", "MERRA-2 GWETPROF, county mean (0-1 wetness)"),
)
# CDL classes that are not cultivated crops; the complement is cultivated.
NON_CULTIVATED = frozenset({
    0, 63, 64, 65, 81, 82, 83, 87, 88, 92, 111, 112,
    121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 195,
})
NON_CONUS_STATES = frozenset({"02", "15", "60", "66", "69", "72", "78"})
ALBEDO = 0.23
WATT_DAY_TO_MJ = 0.0864  # W m-2 daily mean to MJ m-2 day-1
HARGREAVES_KRS = 0.16
AREA_CRS = "EPSG:5070"
CUBE_NAME = "power_county_daily.f32"
WEIGHTS_NAME = "power_county_weights.parquet"
COMPARISON = ("tmax", "tmin", "prec", "rad", "et0", "vpd")
_LOG = logging.getLogger(__name__)
_EE_READY = False
_FILESYSTEM = None


def _filesystem():
    """One shared anonymous S3 session for every store."""
    global _FILESYSTEM
    if _FILESYSTEM is None:
        _FILESYSTEM = s3fs.S3FileSystem(anon=True, default_cache_type="none")
    return _FILESYSTEM


def _open(uri, fs=None):
    """Open one POWER Zarr v2 group over anonymous S3."""
    fs = fs or _filesystem()
    return zarr.open_group(s3fs.S3Map(root=BUCKET + uri, s3=fs, check=False),
                           mode="r", zarr_format=2)


def _origin(group):
    return pd.Timestamp(group["time"].attrs["units"].split("since")[1].strip())


def grid_window(lat, lon, lat_range=LAT_RANGE, lon_range=LON_RANGE):
    """Inclusive index window of the grid cells inside a bounding box."""
    rows = np.flatnonzero((lat >= lat_range[0]) & (lat <= lat_range[1]))
    cols = np.flatnonzero((lon >= lon_range[0]) & (lon <= lon_range[1]))
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def day_window(origin, first=START_DATE, last=END_DATE):
    """Half-open day indices of ``first``..``last`` counted from a store origin."""
    start = (pd.Timestamp(first) - pd.Timestamp(origin)).days
    stop = (pd.Timestamp(last) - pd.Timestamp(origin)).days + 1
    return start, stop


def _chunk_tiles(rows, cols, chunk_y, chunk_x):
    for y in range(rows.start // chunk_y * chunk_y, rows.stop, chunk_y):
        for x in range(cols.start // chunk_x * chunk_x, cols.stop, chunk_x):
            yield (
                max(y, rows.start), min(y + chunk_y, rows.stop),
                max(x, cols.start), min(x + chunk_x, cols.stop),
            )


def read_window(array, days, rows, cols, workers=16):
    """Read one window chunk by chunk in parallel; every task fetches whole chunks."""
    out = np.empty((days.stop - days.start, rows.stop - rows.start,
                    cols.stop - cols.start), "<f4")

    def read(tile):
        y0, y1, x0, x1 = tile
        return tile, array[days, y0:y1, x0:x1]

    tiles = list(_chunk_tiles(rows, cols, array.chunks[1], array.chunks[2]))
    with ThreadPoolExecutor(workers) as pool:
        for (y0, y1, x0, x1), values in pool.map(read, tiles):
            out[:, y0 - rows.start:y1 - rows.start, x0 - cols.start:x1 - cols.start] = values
    return out


def coverage(array, rows, cols):
    """First and last store day holding data in one CONUS chunk of cells.

    Availability is a property of the product, so one corner block of cells is
    enough and costs one chunk per time block instead of a full download.
    """
    block = array[:, rows.start:min(rows.start + 15, rows.stop),
                  cols.start:min(cols.start + 15, cols.stop)]
    valid = np.isfinite(block).any(axis=(1, 2))
    found = np.flatnonzero(valid)
    if not found.size:
        raise ValueError(f"No data in the CONUS window for {array.name}")
    return int(found[0]), int(found[-1]) + 1


def _blocks(chunk, start, stop):
    """Store-aligned day blocks covering ``start``..``stop``."""
    return [(max(begin, start), min(begin + chunk, stop))
            for begin in range(start // chunk * chunk, stop, chunk)]


def fetch_blocks(uri, var, directory, workers=16, force=False, limit=True, margin=0):
    """Cache one variable's CONUS window block by block.

    Blocks are the store's own time chunks, so no chunk is downloaded twice and an
    interrupted run resumes at the next missing block.  ``limit`` clips the window
    to the store's actual data coverage, which keeps trailing empty chunks off the
    wire.  Returns the source grid and the first cached day relative to
    ``START_DATE``.
    """
    group = _open(uri)
    lat = np.asarray(group["lat"][:], float)
    lon = np.asarray(group["lon"][:], float)
    rows, cols = grid_window(lat, lon)
    if margin:
        rows = slice(max(rows.start - margin, 0), min(rows.stop + margin, len(lat)))
        cols = slice(max(cols.start - margin, 0), min(cols.stop + margin, len(lon)))
    origin = _origin(group)
    array = group[var]
    start, stop = day_window(origin)
    if limit:
        first, last = coverage(array, rows, cols)
        start, stop = max(start, first), min(stop, last)
    start = max(start, 0)
    if stop <= start:
        raise ValueError(f"{uri} {var} does not overlap {START_DATE}..{END_DATE}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layout = {"rows": [rows.start, rows.stop], "cols": [cols.start, cols.stop]}
    layout_path = directory / "window.json"
    if layout_path.exists() and not force:
        if json.loads(layout_path.read_text()) != layout:
            raise ValueError(f"{directory} holds a different window; rerun with --force")
    else:
        layout_path.write_text(json.dumps(layout) + "\n")
    for store_start, store_stop in _blocks(array.chunks[0], start, stop):
        first = origin + pd.Timedelta(days=store_start)
        last = origin + pd.Timedelta(days=store_stop - 1)
        path = directory / f"{first:%Y%m%d}_{last:%Y%m%d}.npy"
        if path.exists() and not force:
            continue
        began = time.monotonic()
        values = read_window(array, slice(store_start, store_stop), rows, cols, workers)
        temporary = path.with_name(path.name + ".part")
        with temporary.open("wb") as handle:
            np.save(handle, values)
        temporary.replace(path)
        _LOG.info("%s %s %s..%s: %d days, %.0f MB in %.1f s", var, Path(uri).parts[1],
                  f"{first:%Y-%m-%d}", f"{last:%Y-%m-%d}", store_stop - store_start,
                  values.nbytes / 1e6, time.monotonic() - began)
    return {
        "lat": lat, "lon": lon, "rows": rows, "cols": cols, "origin": origin,
        "window_lat": lat[rows], "window_lon": lon[cols],
        "first": (origin + pd.Timedelta(days=start) - pd.Timestamp(START_DATE)).days,
        "days": stop - start,
    }


class GridBlocks:
    """Lazily read a variable's cached store-aligned day blocks as one array."""

    def __init__(self, directory):
        self.paths = sorted(Path(directory).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No cached blocks in {directory}")
        shapes = [np.load(path, mmap_mode="r").shape for path in self.paths]
        layout = Path(directory) / "window.json"
        if layout.exists():
            window = json.loads(layout.read_text())
            cells = (window["rows"][1] - window["rows"][0],
                     window["cols"][1] - window["cols"][0])
            if any(shape[1:] != cells for shape in shapes):
                raise ValueError(f"Stale blocks in {directory}; rerun with --force")
        self.offsets = np.cumsum([0] + [shape[0] for shape in shapes])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, days):
        pieces = []
        for index, path in enumerate(self.paths):
            low, high = int(self.offsets[index]), int(self.offsets[index + 1])
            first, last = max(days.start, low), min(days.stop, high)
            if first < last:
                values = np.load(path, mmap_mode="r")[first - low:last - low]
                pieces.append(np.asarray(values, np.float32))
        if not pieces:
            raise IndexError(f"Empty slice {days}")
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def regrid_bilinear(data, src_lat, src_lon, dst_lat, dst_lon, chunk=512):
    """Bilinear interpolation of a regular lat/lon grid onto another regular grid.

    The destination must sit inside the source window; both axes are clipped to
    the outermost cell centres, so an outside request would silently flatten.
    """
    data = np.asarray(data, np.float32)
    if (dst_lat.min() < src_lat.min() or dst_lat.max() > src_lat.max()
            or dst_lon.min() < src_lon.min() or dst_lon.max() > src_lon.max()):
        raise ValueError("Destination grid leaves the source window")

    def weights(source, target):
        upper = np.clip(np.searchsorted(source, target), 1, len(source) - 1)
        lower = upper - 1
        high = (target - source[lower]) / (source[upper] - source[lower])
        return lower, upper, high.astype(np.float32)

    i0, i1, fy = weights(src_lat, dst_lat)
    j0, j1, fx = weights(src_lon, dst_lon)
    out = np.empty((data.shape[0], len(dst_lat), len(dst_lon)), np.float32)
    for begin in range(0, data.shape[0], chunk):
        block = data[begin:begin + chunk]
        total = np.zeros((block.shape[0], len(dst_lat), len(dst_lon)), np.float32)
        weight = np.zeros_like(total)
        for rows, row_weight in ((i0, 1 - fy), (i1, fy)):
            for columns, column_weight in ((j0, 1 - fx), (j1, fx)):
                sample = block[:, rows, :][:, :, columns]
                share = (row_weight[:, None] * column_weight[None, :]).astype(np.float32)
                finite = np.isfinite(sample)
                total += np.where(finite, sample * share, 0)
                weight += np.where(finite, share, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[begin:begin + chunk] = np.where(weight > 0, total / weight, np.nan)
    return out


def saturation_vapour_pressure(temperature):
    """FAO-56 eq. 11, kPa from degC."""
    t = np.asarray(temperature, float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def mean_saturation_vapour_pressure(tmax, tmin):
    """FAO-56 eq. 12, kPa from daily extremes."""
    return (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin)) / 2


def vapour_pressure_deficit(tmax, tmin, tdew):
    """FAO-56 eqs. 11-14: es mean minus ea from the 2 m dewpoint, floored at 0.

    A dewpoint above tmax is an interpolation artefact; it cannot mean a negative
    deficit, so the result is clipped instead of carried as a negative number.
    """
    deficit = mean_saturation_vapour_pressure(tmax, tmin) - saturation_vapour_pressure(tdew)
    return np.maximum(deficit, 0.0)


def extraterrestrial_radiation(latitude, doy):
    """FAO-56 eq. 21, MJ m-2 day-1; latitude in degrees, day of year 1..366."""
    phi = np.radians(np.asarray(latitude, float))
    day = np.asarray(doy, float)
    inverse = 1 + 0.033 * np.cos(2 * np.pi * day / 365)
    declination = 0.409 * np.sin(2 * np.pi * day / 365 - 1.39)
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(declination), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * inverse * (
        sunset * np.sin(phi) * np.sin(declination)
        + np.cos(phi) * np.cos(declination) * np.sin(sunset)
    )


def hargreaves_radiation(latitude, doy, tmax, tmin, krs=HARGREAVES_KRS):
    """FAO-56 eq. 50 with kRs 0.16 (interior); MJ m-2 day-1."""
    return krs * extraterrestrial_radiation(latitude, doy) * np.sqrt(
        np.maximum(np.asarray(tmax, float) - np.asarray(tmin, float), 0.0)
    )


def elevation_from_pressure(pressure):
    """Inverse of the FAO-56 eq. 7 altitude term; hPa to m."""
    return (1 - (np.asarray(pressure, float) / 1013.25) ** (1 / 5.25588)) / 2.25577e-5


def penman_monteith(tmax, tmin, ea, rs, wind, pressure, elevation, latitude, doy):
    """FAO-56 eq. 6 daily reference evapotranspiration, mm/day.

    ``pressure`` in kPa, ``wind`` at 2 m in m/s, ``rs`` in MJ m-2 day-1, ``ea`` in
    kPa, ``elevation`` in m (used only by Rso, eq. 37).
    """
    tmax = np.asarray(tmax, float)
    tmin = np.asarray(tmin, float)
    ea = np.asarray(ea, float)
    tmean = (tmax + tmin) / 2
    slope = 4098 * saturation_vapour_pressure(tmean) / (tmean + 237.3) ** 2
    gamma = 0.000665 * np.asarray(pressure, float)
    rso = (0.75 + 2e-5 * np.asarray(elevation, float)) * extraterrestrial_radiation(latitude, doy)
    ratio = np.minimum(np.asarray(rs, float) / rso, 1.0)
    rnl = (
        4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
        * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
        * (1.35 * ratio - 0.35)
    )
    rn = (1 - ALBEDO) * np.asarray(rs, float) - rnl
    es = mean_saturation_vapour_pressure(tmax, tmin)
    numerator = (0.408 * slope * rn
                 + gamma * (900 / (tmean + 273)) * np.asarray(wind, float) * (es - ea))
    # Dark, cold days give a small negative ratio; reference ET cannot be negative.
    return np.maximum(numerator / (slope + gamma * (1 + 0.34 * np.asarray(wind, float))), 0.0)


def counties(path):
    """CONUS adm2 polygons from the CY-Bench archive, with repo-style adm_id."""
    import geopandas as gpd

    path = Path(path)
    if path.suffix == ".zip":
        path = f"/vsizip/{path.resolve()}/polygons/US/US.shp"
    frame = gpd.read_file(path)
    frame["adm_id"] = frame.adm_id.astype(str)
    frame = frame.loc[~frame.adm_id.str[3:5].isin(NON_CONUS_STATES), ["adm_id", "geometry"]]
    return frame.reset_index(drop=True).to_crs("EPSG:4326")


def grid_cells(lat, lon):
    """Rectangular polygons of the grid cells covering the CONUS window."""
    rows, cols = grid_window(lat, lon)
    import geopandas as gpd

    half = ((lat[1] - lat[0]) / 2, (lon[1] - lon[0]) / 2)
    records = [
        {
            "cell": (rows.start + i) * 1000 + cols.start + j,
            "cell_row": rows.start + i,
            "cell_col": cols.start + j,
            "geometry": shapely.box(x - half[1], y - half[0], x + half[1], y + half[0]),
        }
        for i, y in enumerate(lat[rows])
        for j, x in enumerate(lon[cols])
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def area_weights(county_frame, cells, area_crs=AREA_CRS):
    """Intersection area of every county x cell pair, in km2 of an equal-area CRS."""
    import geopandas as gpd

    left = cells.to_crs(area_crs).reset_index(names="cell_index")
    right = county_frame.to_crs(area_crs).reset_index(names="county_index")
    pairs = gpd.sjoin(left[["cell_index", "geometry"]], right[["county_index", "geometry"]],
                      predicate="intersects", how="inner")
    pieces = shapely.intersection(left.geometry.values[pairs.cell_index.values],
                                 right.geometry.values[pairs.county_index.values])
    table = pd.DataFrame({
        "cell": cells.cell.values[pairs.cell_index.values],
        "cell_row": cells.cell_row.values[pairs.cell_index.values],
        "cell_col": cells.cell_col.values[pairs.cell_index.values],
        "adm_id": county_frame.adm_id.values[pairs.county_index.values],
        "area_km2": shapely.area(pieces) / 1e6,
    })
    table = table.loc[np.isfinite(table.area_km2) & table.area_km2.gt(1e-6)]
    return table.sort_values(["adm_id", "cell"]).reset_index(drop=True)


def _initialize(project=None):
    global _EE_READY
    if not _EE_READY:
        import ee

        ee.Initialize(project=project)
        _EE_READY = True


def _cultivated_image():
    import ee

    values = [value for value in range(1, 255) if value not in NON_CULTIVATED]

    def cultivate(image):
        crop = image.select("cropland")
        return crop.remap(values, [1] * len(values), 0).toFloat().rename("cultivated")

    collection = ee.ImageCollection("USDA/NASS/CDL").filterDate("2008-01-01", "2023-01-01")
    return collection.map(cultivate).mean()


def cultivated_shares(state_fips, scale=500, tile_scale=4):
    """Mean cultivated fraction of each county x MERRA-2 cell piece, one state.

    Earth Engine groups a 500 m cultivation mask by MERRA-2 cell id inside each
    county polygon, which yields exactly the county x cell pieces of the weights.
    """
    import ee

    image = _cultivated_image()
    coordinates = ee.Image.pixelLonLat()
    cell = (
        coordinates.select("latitude").add(90).divide(0.5).floor().multiply(1000)
        .add(coordinates.select("longitude").add(180).divide(0.625).floor())
        .rename("cell")
    )
    collection = ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.eq("STATEFP", state_fips))
    reduced = image.addBands(cell).reduceRegions(
        collection, ee.Reducer.mean().group(1, "cell"), scale=scale, tileScale=tile_scale)
    frame = ee.data.computeFeatures(
        {"expression": reduced, "fileFormat": "PANDAS_DATAFRAME", "pageSize": 5000})
    rows = []
    for geoid, groups in zip(frame["GEOID"], frame["groups"], strict=True):
        adm_id = f"US-{geoid[:2]}-{geoid[2:]}"
        # GeoJSON drops a null mean, which means a piece with no usable CDL pixels.
        rows.extend({"adm_id": adm_id, "cell": int(group["cell"]),
                     "cultivated": float(group.get("mean") or 0.0)} for group in groups)
    return pd.DataFrame(rows)


def cultivated_table(county_frame, project=None, workers=4, path=None, force=False):
    """Cached per-state cultivated shares for every CONUS county."""
    if path and Path(path).exists() and not force:
        return pd.read_parquet(path)
    _initialize(project)
    states = sorted(county_frame.adm_id.str[3:5].unique())

    def shares(state):
        table = cultivated_shares(state)
        _LOG.info("Cultivated shares for state %s: %d pieces", state, len(table))
        return table

    with ThreadPoolExecutor(workers) as pool:
        table = pd.concat(list(pool.map(shares, states)), ignore_index=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    return table


def combine_weights(areas, cultivated, counties_order, min_cultivated=0.01):
    """Area x cultivated-share weights normalised per county, with area-only fallback.

    A county falls back to pure area weights when cultivated land is under
    ``min_cultivated`` of the area it overlaps; those adm_ids are returned too.
    """
    table = areas.merge(cultivated, on=["adm_id", "cell"], how="left")
    table["cultivated"] = table.cultivated.fillna(0.0).clip(0.0, 1.0)
    table["weighted_km2"] = table.area_km2 * table.cultivated
    share = (table.groupby("adm_id").weighted_km2.transform("sum")
             / table.groupby("adm_id").area_km2.transform("sum"))
    area_only = sorted(table.loc[share.lt(min_cultivated), "adm_id"].unique())
    fallback = table.adm_id.isin(area_only).to_numpy()
    table["weight"] = np.where(fallback, table.area_km2, table.weighted_km2)
    table["weight"] /= table.groupby("adm_id").weight.transform("sum")
    table["area_only"] = fallback
    table = table.loc[table.weight.gt(0)].reset_index(drop=True)
    table["adm_id"] = pd.Categorical(table.adm_id, categories=list(counties_order), ordered=True)
    missing = sorted(set(counties_order) - set(table.adm_id.astype(str)))
    if missing:
        raise ValueError(f"Counties without weights: {missing[:5]} (+{len(missing)} more)")
    return table.sort_values(["adm_id", "cell"]).reset_index(drop=True), area_only


def weight_matrix(weights, counties_order, cells):
    """Padded sparse weight table: cell index and weight for every county."""
    position = {cell: index for index, cell in enumerate(cells.cell.values)}
    degree = int(weights.groupby("adm_id", observed=True).size().max())
    index = np.zeros((len(counties_order), degree), np.int32)
    value = np.zeros((len(counties_order), degree), np.float32)
    lookup = {adm_id: row for row, adm_id in enumerate(counties_order)}
    for adm_id, group in weights.groupby("adm_id", observed=True, sort=False):
        row = lookup[str(adm_id)]
        count = len(group)
        index[row, :count] = [position[cell] for cell in group.cell]
        value[row, :count] = group.weight.to_numpy(np.float32)
    return index, value


def aggregate(grid, index, value):
    """Weighted county mean of gridded days, dropping missing cells per day.

    ``grid`` is (days, cells); the result is (counties, days).
    """
    total = np.zeros((index.shape[0], grid.shape[0]), np.float32)
    mask = np.zeros_like(total)
    for column in range(index.shape[1]):
        sample = grid[:, index[:, column]].T
        finite = np.isfinite(sample)
        total += np.where(finite, value[:, [column]] * np.nan_to_num(sample), 0).astype(np.float32)
        mask += np.where(finite, value[:, [column]], 0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mask > 0, total / mask, np.nan).astype(np.float32)


def calendar():
    return pd.date_range(START_DATE, END_DATE, freq="D")


def radiation_series(raw_dir, merra, workers=16, force=False):
    """Continuous daily shortwave radiation on the MERRA-2 grid, plus fill flags.

    Stores are merged per day in reverse preference order, so syn1deg wins where
    it exists, SRB covers 1984-2000, flashflux any later gap, and the remaining
    days are Hargreaves-filled from MERRA-2 temperature.  Returns read-only
    memmaps over the 1981-2025 window and the date ranges each source supplied.
    """
    raw_dir = Path(raw_dir)
    rad_path = raw_dir / "rad" / "conus_rad.npy"
    flag_path = raw_dir / "rad" / "conus_filled.npy"
    sources_path = raw_dir / "rad" / "sources.json"
    days = len(calendar())
    shape = (days, merra["rows"].stop - merra["rows"].start,
             merra["cols"].stop - merra["cols"].start)
    if rad_path.exists() and flag_path.exists() and sources_path.exists() and not force:
        return (np.load(rad_path, mmap_mode="r"), np.load(flag_path, mmap_mode="r"),
                json.loads(sources_path.read_text()))

    rad = np.full(shape, np.nan, np.float32)
    filled = np.zeros(shape, np.uint8)
    primary = np.full(days, "", dtype=object)
    destination_lat = merra["lat"][merra["rows"]]
    destination_lon = merra["lon"][merra["cols"]]
    for name, uri in reversed(RADIATION):
        source = fetch_blocks(uri, RADIATION_VAR, raw_dir / "rad" / name, workers, force,
                              margin=1)
        blocks = GridBlocks(raw_dir / "rad" / name)
        regridded = WATT_DAY_TO_MJ * regrid_bilinear(
            blocks[slice(0, len(blocks))], source["window_lat"],
            source["window_lon"], destination_lat, destination_lon)
        first = max(source["first"], 0)
        last = min(source["first"] + source["days"], days)
        if last > first:
            values = regridded[first - source["first"]:last - source["first"]]
            window = rad[first:last]
            np.copyto(window, values, where=np.isfinite(values))
            # Reverse preference order means a better source relabels the day.
            valid = np.isfinite(window).any(axis=(1, 2))
            primary[first:last] = np.where(valid, name, primary[first:last])
        del regridded

    doy = calendar().day_of_year.to_numpy()
    tmax = GridBlocks(raw_dir / "grid" / "tmax")
    tmin = GridBlocks(raw_dir / "grid" / "tmin")
    for begin in range(0, days, 730):
        end = min(begin + 730, days)
        gap = ~np.isfinite(rad[begin:end])
        if gap.any():
            estimate = hargreaves_radiation(destination_lat[None, :, None],
                                            doy[begin:end, None, None],
                                            tmax[slice(begin, end)], tmin[slice(begin, end)])
            rad[begin:end] = np.where(gap, estimate, rad[begin:end])
            filled[begin:end] = gap
    dates = calendar()
    ranges = {}
    for name, _ in RADIATION:
        chosen = np.flatnonzero(primary == name)
        if chosen.size:
            ranges[name] = {"first": f"{dates[chosen[0]]:%Y-%m-%d}",
                            "last": f"{dates[chosen[-1]]:%Y-%m-%d}",
                            "days": int(chosen.size)}
    gaps = np.flatnonzero((filled == 1).any(axis=(1, 2)))
    if gaps.size:
        ranges["hargreaves"] = {"first": f"{dates[gaps[0]]:%Y-%m-%d}",
                                "last": f"{dates[gaps[-1]]:%Y-%m-%d}", "days": int(gaps.size),
                                "cells": int(filled.sum())}
    rad_path.parent.mkdir(parents=True, exist_ok=True)
    for path, array in ((rad_path, rad), (flag_path, filled)):
        temporary = path.with_name(path.name + ".part")
        with temporary.open("wb") as handle:
            np.save(handle, array)
        temporary.replace(path)
    sources_path.write_text(json.dumps(ranges, indent=2) + "\n")
    return np.load(rad_path, mmap_mode="r"), np.load(flag_path, mmap_mode="r"), ranges


def _create_cube(path, n_counties, n_days, n_vars):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype="<f4", mode="w+", shape=(n_counties, n_days, n_vars))


def load_county_daily(path=None):
    """Open the float32 cube and its JSON metadata as (memmap, dict)."""
    path = Path(path or Path("data") / "processed" / "weather" / CUBE_NAME)
    metadata = json.loads(path.with_suffix(".json").read_text())
    cube = np.memmap(path, dtype="<f4", mode="r", shape=tuple(metadata["shape"]))
    return cube, metadata


def county_frame(cube, metadata, adm_id):
    """One county's daily series with the column names compute_stress expects.

    ``rsm`` is the MERRA-2 root-zone wetness GWETROOT, a 0-1 saturation fraction,
    not the GLDAS kg/m2 column the CY-Bench training data carries.
    """
    names = [variable["name"] for variable in metadata["variables"]]
    row = metadata["counties"].index(adm_id)
    frame = pd.DataFrame(cube[row], columns=names)
    frame.insert(0, "date", pd.date_range(metadata["start_date"], metadata["end_date"], freq="D"))
    frame["rsm"] = frame["gwetroot"]
    return frame


def scan_cube(cube):
    """NaN counts per variable and the counties with an all-NaN variable."""
    counts = {name: 0 for name, _, _ in VARIABLES}
    empty = {name: [] for name, _, _ in VARIABLES}
    block = 256
    for begin in range(0, cube.shape[0], block):
        slab = np.asarray(cube[begin:begin + block])
        missing = np.isnan(slab)
        for index, (name, _, _) in enumerate(VARIABLES):
            counts[name] += int(missing[:, :, index].sum())
            empty[name].extend(begin + int(row)
                               for row in np.flatnonzero(missing[:, :, index].all(axis=1)))
        del slab, missing
    return counts, empty


def _merge_temperature(tmin, tmax):
    """Enforce tmin <= tmax as the earlier AgERA5 build did; returns the count."""
    violations = np.flatnonzero(tmin > tmax)
    if violations.size:
        middle = (tmin[violations] + tmax[violations]) / 2
        tmin[violations] = middle
        tmax[violations] = middle
    return int(violations.size)


def build(data_dir=Path("data"), project=None, workers=16, force=False,
          polygons=None, cache_only=False):
    """Download, weight and aggregate the county daily cube."""
    data_dir = Path(data_dir)
    raw = data_dir / "raw" / "power"
    out = data_dir / "processed" / "weather"
    polygons = polygons or data_dir / "raw" / "cybench" / "polygons.zip"
    county_table = counties(polygons)
    order = sorted(county_table.adm_id)
    days = len(calendar())
    dates = calendar()
    _LOG.info("CONUS counties: %d, days: %d", len(order), days)

    merra = None
    for name, source in MERRA_VARIABLES:
        merra = fetch_blocks(MERRA2, source, raw / "grid" / name, workers, force, limit=False)
    grids = {name: GridBlocks(raw / "grid" / name) for name, _ in MERRA_VARIABLES}
    if len(grids["tmax"]) != days or merra["first"] != 0:
        raise ValueError("MERRA-2 blocks do not cover the requested window exactly")
    rad, filled, ranges = radiation_series(raw, merra, workers, force)
    if cache_only:
        return None

    cells = grid_cells(merra["lat"], merra["lon"])
    weights_path = out / WEIGHTS_NAME
    if weights_path.exists() and not force:
        weights = pd.read_parquet(weights_path)
        area_only = sorted(weights.loc[weights.area_only, "adm_id"].astype(str).unique())
    else:
        cultivated = cultivated_table(county_table, project, workers=4,
                                      path=raw / "cultivated.parquet", force=force)
        weights, area_only = combine_weights(area_weights(county_table, cells),
                                             cultivated, order)
        out.mkdir(parents=True, exist_ok=True)
        weights.to_parquet(weights_path, index=False)
    index, value = weight_matrix(weights, order, cells)
    _LOG.info("Weights: %d county-cell pairs, %d area-only counties, at most %d cells",
              len(weights), len(area_only), index.shape[1])

    ps_sum = np.zeros(len(cells), float)
    for begin in range(0, days, 730):
        end = min(begin + 730, days)
        ps_sum += np.nansum(grids["ps"][slice(begin, end)].reshape(end - begin, -1), axis=0)
    pressure_mean = aggregate((ps_sum / days)[None, :], index, value)[:, 0]
    elevation = elevation_from_pressure(pressure_mean)
    latitude = county_table.set_index("adm_id").loc[order].geometry.centroid.y.to_numpy()
    doy = dates.day_of_year.to_numpy()

    names = [name for name, _, _ in VARIABLES]
    cube_path = out / CUBE_NAME
    temporary = cube_path.with_name(cube_path.name + ".part")
    cube = _create_cube(temporary, len(order), days, len(names))
    inversions = 0
    block = 730
    for begin in range(0, days, block):
        end = min(begin + block, days)
        length = end - begin
        daily = {}
        for name, _ in MERRA_VARIABLES:
            daily[name] = aggregate(grids[name][slice(begin, end)].reshape(length, -1),
                                    index, value)
        daily["rad"] = aggregate(rad[begin:end].reshape(length, -1), index, value)
        daily["rad_filled"] = aggregate(
            np.asarray(filled[begin:end], np.float32).reshape(length, -1), index, value)
        inversions += _merge_temperature(daily["tmin"], daily["tmax"])
        dew = saturation_vapour_pressure(daily["tdew"])
        daily["vpd"] = np.maximum(
            mean_saturation_vapour_pressure(daily["tmax"], daily["tmin"]) - dew, 0.0
        ).astype(np.float32)
        daily["et0"] = penman_monteith(
            daily["tmax"], daily["tmin"], dew, daily["rad"], daily["wind"],
            daily["ps"] / 10.0, elevation[:, None], latitude[:, None], doy[None, begin:end],
        ).astype(np.float32)
        daily["cwb"] = (daily["prec"] - daily["et0"]).astype(np.float32)
        cube[:, begin:end, :] = np.stack([daily[name] for name in names], axis=2)
        _LOG.info("Aggregated %s..%s", f"{dates[begin]:%Y-%m-%d}", f"{dates[end - 1]:%Y-%m-%d}")
    cube.flush()
    del cube
    temporary.replace(cube_path)

    cube = np.memmap(cube_path, dtype="<f4", mode="r",
                     shape=(len(order), days, len(names)))
    nan_counts, empty = scan_cube(cube)
    metadata = {
        "shape": list(cube.shape),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "dtype": "float32",
        "order": "C",
        "counties": order,
        "variables": [{"name": name, "units": units, "source": source}
                      for name, units, source in VARIABLES],
        "sources": {
            "meteorology": {"store": BUCKET + MERRA2,
                            "variables": {name: source for name, source in MERRA_VARIABLES}},
            "radiation": {"variable": RADIATION_VAR, "date_ranges": ranges},
            "counties": "CY-Bench polygons.zip, US adm2 (Census 2018 500k)",
            "cultivated": "USDA/NASS/CDL cropland classes, 2008-2022 mean, 500 m",
        },
        "weights": {"file": WEIGHTS_NAME, "pairs": len(weights),
                    "max_cells_per_county": int(index.shape[1])},
        "grid": {"lat_range": list(LAT_RANGE), "lon_range": list(LON_RANGE),
                 "resolution": [float(merra["lat"][1] - merra["lat"][0]),
                                float(merra["lon"][1] - merra["lon"][0])],
                 "cells": len(cells)},
        "area_only_counties": area_only,
        "temperature_qc": {"inversions_replaced": inversions},
        "nan_counts": nan_counts,
        "all_nan_variables": {name: [order[row] for row in rows]
                              for name, rows in empty.items() if rows},
        "notes": [
            "County values are intersection-area x cultivated-share weighted grid means.",
            "tavg is the MERRA-2 daily mean T2M, not (tmin + tmax) / 2.",
            ("rad_filled is the cultivated-area-weighted share of cells whose rad is"
             " Hargreaves-filled; it is 1 for all 1981-1983 days and a small fraction where a"
             " radiation product is absent."),
            ("vpd is the es(tmax, tmin) mean minus es(T2MDEW), floored at 0. The AgERA5 VPD of the"
             " earlier model build behaves like es(Tmax) - ea and runs about 0.6 kPa higher."),
            "ET0 is floored at 0; negative FAO-56 daily values occur on dark cold days.",
            "ET0 uses the county mean surface pressure for gamma and PS-derived elevation for Rso.",
            "prec, rad and the GWET* fractions are county means of the raw grid, not rescaled.",
        ],
    }
    cube_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return cube, metadata


def compare_cybench(cube, metadata, daily_dir):
    """Median per-county daily correlation and mean bias against the AgERA5 caches."""
    names = [variable["name"] for variable in metadata["variables"]]
    columns = [names.index(name) for name in COMPARISON]
    position = {adm_id: row for row, adm_id in enumerate(metadata["counties"])}
    origin = pd.Timestamp(metadata["start_date"])
    size = cube.shape[0]
    sums = {name: {"n": np.zeros(size), "x": np.zeros(size), "y": np.zeros(size),
                   "xx": np.zeros(size), "yy": np.zeros(size), "xy": np.zeros(size)}
            for name in COMPARISON}
    for year in range(2003, 2024):
        path = Path(daily_dir) / f"{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["adm_id", "date", *COMPARISON])
        frame = frame.loc[frame.adm_id.isin(position)]
        rows = frame.adm_id.map(position).to_numpy()
        offsets = (frame.date - origin).dt.days.to_numpy()
        slab = np.asarray(cube[:, offsets.min():offsets.max() + 1, columns])
        local = offsets - offsets.min()
        for column, name in enumerate(COMPARISON):
            ours = slab[rows, local, column].astype(float)
            theirs = frame[name].to_numpy(float)
            valid = np.isfinite(ours) & np.isfinite(theirs)
            if not valid.any():
                continue
            ours, theirs, index = ours[valid], theirs[valid], rows[valid]
            sums[name]["n"] += np.bincount(index, minlength=size)
            sums[name]["x"] += np.bincount(index, weights=ours, minlength=size)
            sums[name]["y"] += np.bincount(index, weights=theirs, minlength=size)
            sums[name]["xx"] += np.bincount(index, weights=ours * ours, minlength=size)
            sums[name]["yy"] += np.bincount(index, weights=theirs * theirs, minlength=size)
            sums[name]["xy"] += np.bincount(index, weights=ours * theirs, minlength=size)
    result = {}
    for name, values in sums.items():
        n = values["n"]
        enough = n >= 365
        covariance = n * values["xy"] - values["x"] * values["y"]
        spread = np.sqrt(np.maximum(n * values["xx"] - values["x"] ** 2, 0)
                         * np.maximum(n * values["yy"] - values["y"] ** 2, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.where(enough & (spread > 0), covariance / spread, np.nan)
            bias = np.where(enough, (values["x"] - values["y"]) / n, np.nan)
        result[name] = {
            "counties": int(np.isfinite(correlation).sum()),
            "median_r": float(np.nanmedian(correlation)),
            "mean_bias": float(np.nanmean(bias)),
            "median_abs_bias": float(np.nanmedian(np.abs(bias))),
        }
    return result


def validate(data_dir=Path("data")):
    """Report the checks requested by the build spec."""
    data_dir = Path(data_dir)
    cube, metadata = load_county_daily(data_dir / "processed" / "weather" / CUBE_NAME)
    frame = county_frame(cube, metadata, "US-19-169")
    window = frame.loc[frame.date.between("2012-07-01", "2012-07-03")]
    total = cube.shape[0] * cube.shape[1]
    report = {
        "story_county_iowa": {
            "tmax_degc": [round(float(value), 2) for value in window.tmax],
            "prec_mm": [round(float(value), 2) for value in window.prec],
            "power_point_api": {"tmax_degc": [36.45, 37.45, 38.34],
                                "prec_mm": [0.04, 0.0, 0.0]},
        },
        "nan_fraction": {name: count / total for name, count in metadata["nan_counts"].items()},
        "all_nan_variables": metadata.get("all_nan_variables", {}),
        "cybench": compare_cybench(cube, metadata,
                                   data_dir / "processed" / "yield_stress" / "daily"),
        "radiation_sources": metadata["sources"]["radiation"]["date_ranges"],
        "area_only_counties": len(metadata["area_only_counties"]),
    }
    path = data_dir / "processed" / "weather" / "power_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="ignore cached blocks and weights")
    parser.add_argument("--cache-only", action="store_true", help="download raw subsets only")
    parser.add_argument("--validate", action="store_true", help="compare the finished cube")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.validate:
        validate(args.data_dir)
    else:
        build(args.data_dir, args.project, args.workers, args.force, cache_only=args.cache_only)


if __name__ == "__main__":
    main()
