"""Static county soil, terrain and groundwater trend for the contiguous US.

Soil comes from OpenLandMap v02 (SoilGrids-derived, 250 m, six standard depths) on
Earth Engine: USDA gSSURGO needs Soil Data Access spatial joins that do not
aggregate to counties in a few queries, and POLARIS is not in the catalog.  Each
layer's depth bands are sampled on the same 0.05 degree grid as the satellite
series and reduced with the cropland weights of :mod:`fieldshift.data.eo`.

Available water capacity and saturated conductivity are not published by
OpenLandMap, so they come from the Saxton & Rawls (2006) pedotransfer on the
sampled sand, clay and organic matter; that is a derived quantity, not an
observation, and the sources file says so.

Terrain is SRTM 30 m reduced per county on the server.  The cropland-weighted
centroid comes from the satellite weights, and the groundwater trend is the
per-county slope of the GRACE terrestrial water storage anomaly.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data import eo

DEPTHS = (("b0", 5.0), ("b10", 10.0), ("b30", 15.0), ("b60", 30.0), ("b100", 40.0))
TOP = tuple(item for item in DEPTHS[:3])
ROOT = DEPTHS
LAYERS = {
    "sand": ("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", 1.0, "percent"),
    "clay": ("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", 1.0, "percent"),
    "organic_carbon": ("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", 5.0, "g/kg"),
    "ph": ("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02", 0.1, "pH"),
    "bulk_density": (
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02", 0.01, "g/cm3"),
}
SRTM = "USGS/SRTMGL1_003"
TIGER = "TIGER/2018/Counties"
SLOPE_LIMIT = 5.0
STEEP_SCALE = 250
ORGANIC_MATTER = 1.724  # Walkley-Black conversion from organic carbon
STATIC_NAME = "county_static.parquet"
SOURCES_NAME = "county_static_sources.json"
_LOG = logging.getLogger(__name__)


def saxton_rawls(sand, clay, organic_matter):
    """Volumetric water contents, AWC and Ksat from the Saxton & Rawls (2006) model.

    ``sand`` and ``clay`` are mass fractions (0-1), ``organic_matter`` is percent.
    Returns field capacity, wilting point (m3/m3), available water capacity
    (mm/m) and saturated conductivity (mm/h), all as float arrays.
    """
    s = np.clip(np.asarray(sand, float), 0.0, 1.0)
    c = np.clip(np.asarray(clay, float), 0.0, 1.0)
    om = np.clip(np.asarray(organic_matter, float), 0.0, 20.0)
    t1500 = (-0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om
             - 0.013 * c * om + 0.068 * s * c + 0.031)
    wilting = t1500 + (0.14 * t1500 - 0.02)
    t33 = (-0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om
           - 0.027 * c * om + 0.452 * s * c + 0.299)
    field = t33 + (1.283 * t33 ** 2 - 0.374 * t33 - 0.015)
    ts33 = (0.278 * s + 0.034 * c + 0.022 * om - 0.018 * s * om
            - 0.027 * c * om - 0.584 * s * c + 0.078)
    # theta(S-33) is the pore space above field capacity, so saturation adds it back.
    saturated = field + (ts33 + (0.636 * ts33 - 0.107))
    field = np.clip(field, 0.02, 0.6)
    wilting = np.clip(wilting, 0.0, field - 1e-3)
    saturated = np.clip(saturated, field + 1e-3, 0.75)
    available = np.clip(field - wilting, 0.0, None)
    pore = np.clip(saturated - field, 1e-4, None)
    exponent = (np.log(field) - np.log(wilting)) / (np.log(1500) - np.log(33))
    conductivity = 1930.0 * np.power(pore, 3.0 - exponent)
    return field, wilting, available * 1000.0, conductivity


def depth_weight(values, depths):
    """Thickness-weighted mean over the given ``(band, thickness)`` layers."""
    total = np.zeros_like(values[depths[0][0]], dtype=float)
    weight = 0.0
    for band, thickness in depths:
        total += np.nan_to_num(values[band]) * thickness
        weight += thickness * np.isfinite(values[band])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / np.where(weight > 0, weight, 1), np.nan)


def sample_layer(asset, scale, bands):
    """Fetch the depth bands of one OpenLandMap layer on the county grid."""
    import ee

    data = eo.fetch(ee.Image(asset).select([band for band, _ in bands]),
                    [band for band, _ in bands])
    return {band: np.asarray(data[band], float) * scale
            for band, _ in bands}


def _cached_frame(path, build, force):
    """Return a cached parquet frame, computing it only when missing or forced."""
    path = Path(path)
    if path.exists() and not force:
        _LOG.info("Reusing %s", path)
        return pd.read_parquet(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return frame


def soil_table(data_dir=Path("data"), workers=3, force=False):
    """Cropland-weighted soil properties per county for the topsoil and root zone."""
    return _cached_frame(
        Path(data_dir) / "processed" / "static" / ".cache_soil.parquet",
        lambda: _soil_table(data_dir, workers), force)


def _soil_table(data_dir=Path("data"), workers=3):
    eo.initialize()
    table, order, _ = eo.load_weights(data_dir)
    index, weight = eo.weight_matrix(table, order)
    bands = ROOT

    def counties(raster):
        return eo.county_mean(eo.flatten(raster), index, weight)[0]

    layers = {}
    for name, (asset, scale, _) in LAYERS.items():
        _LOG.info("Fetching %s", asset)
        depth_values = sample_layer(asset, scale, bands)
        layers[name] = {band: counties(depth_values[band]) for band, _ in bands}

    sand = {band: np.clip(layers["sand"][band], 0.0, 100.0) for band, _ in bands}
    clay = {band: np.clip(layers["clay"][band], 0.0, 100.0) for band, _ in bands}
    # A few coastal pixels carry sand + clay above 100; rescale to a consistent texture.
    excess = {band: np.maximum(sand[band] + clay[band], 100.0) / 100.0 for band, _ in bands}
    sand = {band: sand[band] / excess[band] for band, _ in bands}
    clay = {band: clay[band] / excess[band] for band, _ in bands}
    silt = {band: np.clip(100.0 - sand[band] - clay[band], 0.0, 100.0)
            for band, _ in bands}
    organic_carbon = {band: layers["organic_carbon"][band] for band, _ in bands}
    organic_matter = {band: organic_carbon[band] / 10.0 * ORGANIC_MATTER
                      for band, _ in bands}

    result = {"adm_id": order}
    for label, depths in (("0_30", TOP), ("0_100", ROOT)):
        for name, values in (("sand", sand), ("silt", silt), ("clay", clay),
                             ("organic_matter", organic_matter),
                             ("ph", layers["ph"]), ("bulk_density", layers["bulk_density"])):
            result[f"{name}_{label}"] = depth_weight(values, depths)
    for label, depths in (("0_30", TOP), ("0_100", ROOT)):
        field, wilting, available, conductivity = {}, {}, {}, {}
        for band, _ in bands:
            field[band], wilting[band], available[band], conductivity[band] = saxton_rawls(
                sand[band] / 100.0, clay[band] / 100.0, organic_matter[band])
        for name, values in (("field_capacity", field), ("wilting_point", wilting),
                             ("awc", available), ("ksat", conductivity)):
            result[f"{name}_{label}"] = depth_weight(values, depths)
    frame = pd.DataFrame(result)
    counts = table.groupby("adm_id", observed=True).size()
    frame["n_cells"] = counts.loc[order].to_numpy()
    return frame


def cropland_centroid(data_dir=Path("data")):
    """Cropland-area weighted centroid of every county, degrees."""
    table, order, _ = eo.load_weights(data_dir)
    lat, lon = eo.cell_centers()
    frame = pd.DataFrame({
        "adm_id": table.adm_id.astype(str),
        "weight": table.weight.to_numpy(float),
        "lat": lat[table.cell_row.to_numpy()].astype(float),
        "lon": lon[table.cell_col.to_numpy()].astype(float),
    })
    frame["lat"] *= frame.weight
    frame["lon"] *= frame.weight
    grouped = frame.groupby("adm_id")[["weight", "lat", "lon"]].sum()
    centroids = pd.DataFrame({
        "adm_id": grouped.index,
        "centroid_lat": grouped.lat / grouped.weight,
        "centroid_lon": grouped.lon / grouped.weight,
    })
    return centroids.set_index("adm_id").loc[order].reset_index()


def terrain(data_dir=Path("data"), project=None, force=False):
    """SRTM elevation, slope percent and the cropland share steeper than 5 percent."""
    return _cached_frame(
        Path(data_dir) / "processed" / "static" / ".cache_terrain.parquet",
        lambda: _terrain(project), force)


def _terrain(project=None):
    import ee

    eo.initialize(project)
    srtm = ee.Image(SRTM).select("elevation")
    slope = ee.Terrain.slope(srtm)
    percent = slope.multiply(np.pi / 180.0).tan().multiply(100.0).rename("slope_percent")
    cultivated = eo.cultivated_mask()
    steep = percent.gt(SLOPE_LIMIT).multiply(cultivated).rename("steep_cropland")
    image = srtm.rename("elevation").addBands(percent).addBands(steep) \
        .addBands(cultivated.rename("cropland"))
    conus = ee.FeatureCollection(TIGER).filter(
        ee.Filter.inList("STATEFP", list(eo.power.NON_CONUS_STATES)).Not())
    reduced = image.reduceRegions(conus, ee.Reducer.mean(), STEEP_SCALE, tileScale=4)
    stripped = reduced.map(lambda feature: ee.Feature(None, {
        "GEOID": feature.get("GEOID"),
        "elevation": feature.get("elevation"),
        "slope_percent": feature.get("slope_percent"),
        "steep_cropland": feature.get("steep_cropland"),
        "cropland": feature.get("cropland"),
    }))
    frame = ee.data.computeFeatures({"expression": stripped,
                                    "fileFormat": "PANDAS_DATAFRAME",
                                    "pageSize": 5000})
    frame["adm_id"] = ("US-" + frame.GEOID.str[:2] + "-" + frame.GEOID.str[2:])
    for column in ("elevation", "slope_percent", "steep_cropland", "cropland"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["cropland_steep_share"] = np.where(
        frame.cropland.gt(0), frame.steep_cropland / frame.cropland, np.nan)
    return frame.loc[:, ["adm_id", "elevation", "slope_percent", "cropland_steep_share"]]


def groundwater_trend(data_dir=Path("data")):
    """Per-county GRACE terrestrial water storage slope, cm/year over 2002-2025.

    GRACE mascons are about 3 degrees wide, so every county inside one mascon gets
    the same slope; the trend is a regional depletion signal, not a field-scale one.
    The companion column counts the 8-day steps that carried a GRACE month.
    """
    parts = Path(data_dir) / "processed" / "eo" / eo.PARTS
    path = parts / "grace_tws.npy"
    order = eo.county_order(data_dir)
    columns = ["adm_id", "grace_tws_trend_cm_per_year", "grace_tws_trend_steps"]
    if not path.exists():
        _LOG.warning("No GRACE part yet; groundwater trend left out")
        return pd.DataFrame({"adm_id": order, columns[1]: np.nan, columns[2]: 0})
    series = np.load(path).astype(float)
    years = np.array([int(str(eo.step_bounds(step)[0])[:4]) for step in range(eo.N_STEPS)])
    use = (years >= 2002) & (years <= 2025)
    x = years[use].astype(float)
    valid = np.isfinite(series[:, use])
    block = 256
    slope = np.full(len(order), np.nan)
    span = valid.sum(axis=1)
    for begin in range(0, len(order), block):
        end = min(begin + block, len(order))
        y = np.where(valid[begin:end], series[begin:end][:, use], 0.0)
        count = valid[begin:end].sum(axis=1)
        sx = np.where(valid[begin:end], x, 0.0).sum(axis=1)
        sy = y.sum(axis=1)
        sxx = np.where(valid[begin:end], x * x, 0.0).sum(axis=1)
        sxy = (y * np.where(valid[begin:end], x, 0.0)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = (count * sxy - sx * sy) / (count * sxx - sx * sx)
        slope[begin:end] = np.where(count >= 4, value, np.nan)
    return pd.DataFrame({"adm_id": order, columns[1]: slope, columns[2]: span})


def build(data_dir=Path("data"), project=None, workers=3, force=False):
    """Write the static county table and its sources file."""
    data_dir = Path(data_dir)
    eo.initialize(project)
    soil = soil_table(data_dir, workers, force)
    _LOG.info("Soil done: %d counties", len(soil))
    heights = terrain(data_dir, project, force)
    _LOG.info("Terrain done: %d counties", len(heights))
    centroids = cropland_centroid(data_dir)
    trend = groundwater_trend(data_dir)
    frame = soil.merge(heights, on="adm_id", how="left") \
        .merge(centroids, on="adm_id", how="left") \
        .merge(trend, on="adm_id", how="left")
    order = eo.county_order(data_dir)
    frame["adm_id"] = pd.Categorical(frame.adm_id, categories=order, ordered=True)
    frame = frame.sort_values("adm_id").reset_index(drop=True)
    missing = [name for name in order if name not in set(frame.adm_id.astype(str))]
    if missing:
        raise ValueError(f"Counties without a static row: {missing[:5]}")
    out = data_dir / "processed" / "static"
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / (STATIC_NAME + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(out / STATIC_NAME)
    sources = {
        "soil": {
            "source": "OpenLandMap v02 on Earth Engine (SoilGrids-derived, 250 m)",
            "reason": "gSSURGO/gNATSGO need Spatial Data Access joins that do not "
                      "aggregate to 3,108 counties in a few queries; POLARIS is not "
                      "in the Earth Engine catalog.",
            "layers": {name: {"asset": asset, "scale": scale, "units": units}
                       for name, (asset, scale, units) in LAYERS.items()},
            "depths": {"bands": [[band, thickness] for band, thickness in DEPTHS],
                       "topsoil_0_30": [band for band, _ in TOP],
                       "root_zone_0_100": [band for band, _ in ROOT],
                       "weighting": "layer thickness in cm, applied to the sampled bands"},
            "derived": {
                "silt": "100 - sand - clay (percent)",
                "organic_matter": "organic carbon g/kg / 10 * 1.724 (percent)",
                "awc": "Saxton & Rawls 2006 field capacity - wilting point, mm per m",
                "ksat": "Saxton & Rawls 2006, mm/h",
                "field_capacity": "Saxton & Rawls 2006, m3/m3",
                "wilting_point": "Saxton & Rawls 2006, m3/m3",
            },
            "units": {"sand": "percent", "silt": "percent", "clay": "percent",
                      "organic_matter": "percent", "ph": "pH", "bulk_density": "g/cm3",
                      "awc": "mm/m", "ksat": "mm/h", "field_capacity": "m3/m3",
                      "wilting_point": "m3/m3"},
            "aggregation": "cropland-fraction x cell-area weighted 0.05 degree cells "
                           "(the weights of fieldshift.data.eo)",
        },
        "terrain": {
            "asset": SRTM, "scale_m": STEEP_SCALE,
            "units": {"elevation": "m", "slope_percent": "percent",
                      "cropland_steep_share": "fraction of cropland area"},
            "cropland_steep_share": f"mean(slope > {SLOPE_LIMIT}% x cultivated) / "
                                    "mean(cultivated) inside the county",
        },
        "centroid": "weight x cell-centre mean over the satellite weights (EPSG:4326)",
        "groundwater": {"asset": eo.GRACE,
                        "resolution": "about 3 degrees (mascon); all counties in one "
                                      "mascon share a trend",
                        "units": "cm/year of terrestrial water storage anomaly",
                        "window": "2002-2025, steps with a GRACE month only"},
    }
    (out / SOURCES_NAME).write_text(json.dumps(sources, indent=2) + "\n")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true",
                        help="ignore the cached soil and terrain frames")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    frame = build(args.data_dir, args.project, args.workers, args.force)
    print(frame.describe().T.to_string())


if __name__ == "__main__":
    main()
