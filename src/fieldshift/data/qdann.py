"""Sample QDANN 30 m corn/soybean yields and CDL crop history at random field points."""

import argparse
import logging
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ee
import pandas as pd

QDANN = "projects/sat-io/open-datasets/lobell-lab/VAE_QDANN_YIELD_MAP/CORN_SOY_MAP"
CDL = "USDA/NASS/CDL"
STATES = ("IA", "IL", "IN", "MN", "MO", "OH", "SD", "WI")
HISTORY_YEARS = range(2006, 2023)
YIELD_YEARS = range(2008, 2023)
CROPS = {1: "corn", 5: "soy"}
BATCH = 2500
_LOG = logging.getLogger(__name__)


def _stack(state):
    cdl = ee.ImageCollection(CDL).select("cropland")
    qdann = ee.ImageCollection(QDANN).filter(ee.Filter.eq("state_abbv", state))
    bands = []
    for year in HISTORY_YEARS:
        crop = (
            cdl.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic().unmask(0).rename(f"cdl_{year}")
        )
        # A pixel whose 3x3 neighbours share its class sits inside a field, not on an edge.
        around = crop.reduceNeighborhood(ee.Reducer.minMax(), ee.Kernel.square(1))
        interior = around.select(f"cdl_{year}_min").eq(around.select(f"cdl_{year}_max"))
        bands += [crop, interior.rename(f"interior_{year}")]
    observed = []
    for year in YIELD_YEARS:
        # Filter per state: neighbouring state rasters overlap and must not be mosaicked.
        image = ee.Image(
            qdann.filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
        ).unmask(0).rename([f"corn_{year}", f"soy_{year}"])
        bands.append(image)
        observed.append(image.reduce(ee.Reducer.max()).gt(0))
    counties = ee.FeatureCollection("TIGER/2018/Counties").map(
        lambda county: county.set("fips", ee.Number.parse(county.get("GEOID")))
    )
    bands.append(
        counties.reduceToImage(["fips"], ee.Reducer.first()).unmask(0).rename("county_fips")
    )
    bands.append(ee.ImageCollection(observed).sum().rename("n_yield"))
    return ee.Image.cat(bands)


def _seed(state, index):
    return zlib.crc32(f"{state}-{index}".encode())


def sample_batch(state, index, size=BATCH, min_yield_years=5):
    """Fetch one reproducible batch; only points with enough yield years are returned."""
    region = (
        ee.FeatureCollection("TIGER/2018/States")
        .filter(ee.Filter.eq("STUSPS", state)).geometry()
    )
    points = ee.FeatureCollection.randomPoints(region, size, _seed(state, index))
    grid = (
        ee.ImageCollection(CDL).filterDate("2020-01-01", "2021-01-01").first()
        .select("cropland").projection()
    )
    sampled = (
        _stack(state)
        .sampleRegions(collection=points, projection=grid, geometries=True)
        .filter(ee.Filter.gte("n_yield", min_yield_years))
    )
    frame = ee.data.computeFeatures(
        {"expression": sampled, "fileFormat": "PANDAS_DATAFRAME", "pageSize": 5000}
    )
    if frame.empty:
        return pd.DataFrame()
    coordinates = frame["geo"].map(lambda geometry: geometry["coordinates"])
    frame = frame.drop(columns=["geo"])
    frame.insert(0, "lat", coordinates.str[1])
    frame.insert(0, "lon", coordinates.str[0])
    frame.insert(0, "state", state)
    frame.insert(0, "point_id", [f"{state}-{index}-{row}" for row in range(len(frame))])
    return frame


def to_long(points):
    """One row per point and yield year, with the two preceding CDL classes."""
    rows = []
    for year in YIELD_YEARS:
        crop = points[f"cdl_{year}"]
        for code, name in CROPS.items():
            value = points[f"{name}_{year}"]
            keep = crop.eq(code) & value.gt(0)
            rows.append(pd.DataFrame({
                "point_id": points.loc[keep, "point_id"],
                "state": points.loc[keep, "state"],
                "county_fips": points.loc[keep, "county_fips"],
                "year": year,
                "crop": name,
                "yield_kg_ha": value[keep],
                "prev1": points.loc[keep, f"cdl_{year - 1}"],
                "prev2": points.loc[keep, f"cdl_{year - 2}"],
                "interior": points.loc[keep, f"interior_{year}"].eq(1)
                & points.loc[keep, f"interior_{year - 1}"].eq(1),
            }))
    return pd.concat(rows, ignore_index=True)


def _initialize(project):
    ee.Initialize(project=project)


def _fetch(task):
    state, index, size, part = task
    started = time.monotonic()
    frame = sample_batch(state, index, size)
    temporary = part.with_name(part.name + ".part")
    if frame.empty:
        # A zero-byte part marks a finished batch in which no point qualified.
        temporary.write_bytes(b"")
    else:
        frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(part)
    return state, index, len(frame), size, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("EE_PROJECT"))
    parser.add_argument("--density", type=float, default=0.08, help="random points per km2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not args.project:
        parser.error("pass --project or set EE_PROJECT (a registered Earth Engine project)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _initialize(args.project)
    land = dict(
        ee.FeatureCollection("TIGER/2018/States")
        .filter(ee.Filter.inList("STUSPS", list(STATES)))
        .reduceColumns(ee.Reducer.toList(2), ["STUSPS", "ALAND"]).get("list").getInfo()
    )
    directory = args.data_dir / "raw" / "qdann"
    directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    for state in STATES:
        total = round(float(land[state]) / 1e6 * args.density)
        for index in range(-(-total // BATCH)):
            size = min(BATCH, total - index * BATCH)
            tasks.append((state, index, size, directory / f"{state}_{index:03d}_{size}.csv.gz"))
    pending = [task for task in tasks if not task[3].exists()]
    _LOG.info("%d batches, %d still to fetch", len(tasks), len(pending))
    # Separate processes: each worker holds its own Earth Engine session.
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialize, initargs=(args.project,)
    ) as pool:
        for state, index, kept, size, seconds in pool.map(_fetch, pending):
            _LOG.info("%s batch %d: kept %d of %d points in %.0f s", state, index, kept, size, seconds)
    frames = [pd.read_csv(task[3]) for task in tasks if task[3].stat().st_size]
    if not frames:
        raise RuntimeError("No sampled point had enough QDANN yield years")
    points = pd.concat(frames, ignore_index=True)
    output = args.data_dir / "processed" / "qdann_points.csv.gz"
    points.to_csv(output, index=False)
    long = to_long(points)
    long.to_csv(output.with_name("qdann_point_years.csv.gz"), index=False)
    print(f"{len(points):,} points, {len(long):,} point-years -> {output}")
    print(points.groupby("state").size().to_string())


if __name__ == "__main__":
    main()
