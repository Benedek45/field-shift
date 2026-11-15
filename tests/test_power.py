import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from fieldshift.data import power


def toy_cells():
    """Four 0.5 degree cells over Kansas, in the layout grid_cells produces."""
    boxes = [
        (0, 0, -100.0, 38.0), (1, 0, -99.5, 38.0),
        (0, 1, -100.0, 38.5), (1, 1, -99.5, 38.5),
    ]
    return gpd.GeoDataFrame(
        [{"cell": row * 1000 + col, "cell_row": row, "cell_col": col,
          "geometry": shapely.box(lon, lat, lon + 0.5, lat + 0.5)}
         for row, col, lon, lat in boxes],
        geometry="geometry", crs="EPSG:4326",
    )


def toy_counties(geometries, names):
    return gpd.GeoDataFrame({"adm_id": names, "geometry": geometries},
                            geometry="geometry", crs="EPSG:4326")


def test_area_weights_single_cell_and_quarters():
    cells = toy_cells()
    frame = toy_counties(
        [shapely.box(-100.0, 38.0, -99.5, 38.5),
         shapely.box(-99.75, 38.25, -99.25, 38.75)],
        ["US-20-001", "US-20-002"],
    )
    table = power.area_weights(frame, cells)
    one = table.loc[table.adm_id.eq("US-20-001")]
    assert one.cell.tolist() == [0]
    assert one.area_km2.iloc[0] > 2000
    split = table.loc[table.adm_id.eq("US-20-002")]
    assert sorted(split.cell) == [0, 1, 1000, 1001]
    # Albers scale varies slightly with longitude, so quarters agree to a few parts per thousand.
    assert np.allclose(split.area_km2, split.area_km2.iloc[0], rtol=5e-3)
    assert np.isclose(split.area_km2.sum(), one.area_km2.iloc[0], rtol=1e-2)
    assert np.allclose(split.area_km2, one.area_km2.iloc[0] / 4, rtol=1e-2)


def test_combine_weights_normalises_and_falls_back_to_area():
    cells = toy_cells()
    frame = toy_counties(
        [shapely.box(-100.0, 38.0, -99.5, 38.5),
         shapely.box(-99.75, 38.25, -99.25, 38.75)],
        ["US-20-001", "US-20-002"],
    )
    areas = power.area_weights(frame, cells)
    cultivated = pd.DataFrame([
        {"adm_id": "US-20-001", "cell": 0, "cultivated": 0.0},
        {"adm_id": "US-20-002", "cell": 0, "cultivated": 0.5},
        {"adm_id": "US-20-002", "cell": 1, "cultivated": 1.0},
        {"adm_id": "US-20-002", "cell": 1000, "cultivated": 0.25},
        {"adm_id": "US-20-002", "cell": 1001, "cultivated": 0.75},
    ])
    table, area_only = power.combine_weights(areas, cultivated, ["US-20-001", "US-20-002"])
    assert area_only == ["US-20-001"]
    totals = table.groupby("adm_id", observed=True).weight.sum()
    assert np.allclose(totals.to_numpy(), 1.0)
    split = table.loc[table.adm_id.astype(str).eq("US-20-002")]
    expected = split.weighted_km2.to_numpy() / split.weighted_km2.sum()
    assert np.allclose(split.weight.to_numpy(), expected)


def test_aggregate_skips_missing_cells_per_day():
    grid = np.array([[10.0, np.nan, 30.0], [1.0, 2.0, 3.0]])
    index = np.array([[0, 1, 2]], np.int32)
    value = np.array([[0.5, 0.25, 0.25]], np.float32)
    result = power.aggregate(grid, index, value)
    assert np.allclose(result[0], [(5.0 + 7.5) / 0.75, 0.5 + 0.5 + 0.75])
    assert np.isnan(power.aggregate(np.full((1, 3), np.nan), index, value)[0, 0])


def test_extraterrestrial_radiation_matches_fao56_example():
    assert round(float(power.extraterrestrial_radiation(-20, 246)), 1) == 32.2


def test_saturation_vapour_pressure_matches_fao56_examples():
    assert round(float(power.saturation_vapour_pressure(24.5)), 3) == 3.075
    # FAO-56 Example 4 reports 2.617 kPa from rounded 3.168 and 2.065 kPa terms.
    assert abs(float(power.mean_saturation_vapour_pressure(25, 18)) - 2.617) < 2e-3
    assert abs(float(power.saturation_vapour_pressure(18)) - 2.065) < 2e-3


def test_vapour_pressure_deficit_uses_dewpoint_and_floors_at_zero():
    assert round(float(power.vapour_pressure_deficit(25, 18, 18)), 3) == 0.552
    assert float(power.vapour_pressure_deficit(10, 5, 20)) == 0.0


def test_penman_monteith_reproduces_fao56_example_18():
    # Brussels, 6 July: ET0 3.9 mm/day for Rs 22.07 MJ/m2, u2 2.078 m/s, ea 1.409 kPa.
    et0 = power.penman_monteith(21.5, 12.3, 1.409, 22.07, 2.078, 100.1, 100.0, 50.8, 187)
    assert abs(float(et0) - 3.9) < 0.05


def test_hargreaves_radiation_is_scaled_extraterrestrial_radiation():
    rs = power.hargreaves_radiation(40.0, 180, 30.0, 20.0)
    expected = 0.16 * power.extraterrestrial_radiation(40.0, 180) * np.sqrt(10.0)
    assert np.isclose(float(rs), float(expected))
    assert float(power.hargreaves_radiation(40.0, 180, 20.0, 25.0)) == 0.0


def test_memmap_round_trip(tmp_path):
    metadata = {
        "shape": [2, 3, 2], "start_date": "2012-07-01", "end_date": "2012-07-03",
        "counties": ["US-19-169", "US-20-001"],
        "variables": [{"name": "tmax", "units": "degC", "source": "test"},
                      {"name": "gwetroot", "units": "1", "source": "test"}],
    }
    cube = power._create_cube(tmp_path / "cube.f32", 2, 3, 2)
    cube[0, :, 0] = [1.0, np.nan, 3.0]
    cube[0, :, 1] = [0.5, 0.6, 0.7]
    cube[1, :, 1] = [0.1, 0.2, 0.3]
    cube.flush()
    del cube
    (tmp_path / "cube.json").write_text(json.dumps(metadata))
    loaded, restored = power.load_county_daily(tmp_path / "cube.f32")
    assert isinstance(loaded, np.memmap)
    frame = power.county_frame(loaded, restored, "US-19-169")
    assert list(frame.columns[:3]) == ["date", "tmax", "gwetroot"]
    assert frame.tmax.tolist()[0] == 1.0 and np.isnan(frame.tmax.iloc[1])
    assert frame.date.iloc[0] == pd.Timestamp("2012-07-01")
    assert np.allclose(frame.rsm.to_numpy(), [0.5, 0.6, 0.7])
    assert np.allclose(frame.rsm.to_numpy(), frame.gwetroot.to_numpy())
