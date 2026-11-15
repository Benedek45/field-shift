"""Offline tests for the 8-day step axis, county weighting and the cube writer."""

import json

import numpy as np
import pandas as pd
import pytest

from fieldshift.data import eo


def test_step_bounds_short_last_period():
    first, last = eo.step_bounds(45)
    assert str(first) == "1981-12-27"
    assert str(last) == "1982-01-01"
    assert eo.steps_covered(45) == 5


def test_step_bounds_leap_year_is_one_day_longer():
    common = eo.step_bounds(eo.STEPS_PER_YEAR - 1)
    leap_step = (1984 - eo.FIRST_YEAR) * eo.STEPS_PER_YEAR + eo.STEPS_PER_YEAR - 1
    leap = eo.step_bounds(leap_step)
    assert str(common[0]) == "1981-12-27"
    assert str(leap[0]) == "1984-12-26"
    assert (leap[1] - leap[0]).astype(int) == 6
    assert (common[1] - common[0]).astype(int) == 5


def test_step_bounds_every_step_is_contiguous():
    assert eo.N_STEPS == (eo.LAST_YEAR - eo.FIRST_YEAR + 1) * eo.STEPS_PER_YEAR
    for step in (0, 1, 44, 45, 46, 2000, eo.N_STEPS - 2, eo.N_STEPS - 1):
        first, last = eo.step_bounds(step)
        if step < eo.N_STEPS - 1:
            assert last == eo.step_bounds(step + 1)[0]
        assert 5 <= (last - first).astype(int) <= 8


def test_step_of_round_trips_and_bounds():
    assert eo.step_of("1981-01-01") == 0
    assert eo.step_of("1981-01-08") == 0
    assert eo.step_of("1981-01-09") == 1
    assert eo.step_of("2025-12-31") == eo.N_STEPS - 1
    for step in (0, 100, 1234, eo.N_STEPS - 1):
        assert eo.step_of(eo.step_bounds(step)[0]) == step
    with pytest.raises(ValueError):
        eo.step_of("1980-12-31")
    with pytest.raises(IndexError):
        eo.step_bounds(eo.N_STEPS)


def test_assign_steps_maps_each_step_to_the_composite_that_covers_it():
    starts = np.array([np.datetime64("1981-01-01", "D") + np.timedelta64(8 * k, "D")
                       for k in range(400)])
    steps, composites = eo.assign_steps(starts, 8)
    assert steps[0] == 0
    assert np.array_equal(np.diff(steps), np.ones(len(steps) - 1, int))
    # 400 eight-day composites span 3,200 days, which is 403 year-anchored steps.
    assert len(steps) == 403
    for step, composite in zip(steps[:50], composites[:50], strict=True):
        first = eo.step_bounds(int(step))[0]
        assert starts[composite] <= first < starts[composite] + np.timedelta64(8, "D")


def test_assign_steps_sixteen_day_composites_cover_two_steps_each():
    starts = np.array([np.datetime64("1981-01-01", "D") + np.timedelta64(16 * k, "D")
                       for k in range(50)])
    steps, composites = eo.assign_steps(starts, 16)
    # 50 sixteen-day windows span 800 days: 100 steps plus one short year-end step.
    assert len(steps) in (100, 101)
    assert len(set(composites.tolist())) == 50
    used = np.bincount(composites, minlength=50)
    # Two steps per composite, three when a composite straddles a short year-end step.
    assert used.min() == 2 and used.max() <= 3
    for step, composite in zip(steps, composites, strict=True):
        first = eo.step_bounds(int(step))[0]
        assert starts[composite] <= first < starts[composite] + np.timedelta64(16, "D")


def test_county_mean_applies_weights_and_drops_missing_cells():
    index = np.array([[0, 1], [2, 3]], np.int32)
    value = np.array([[0.75, 0.25], [0.5, 0.5]], np.float32)
    grid = np.array([[10.0, 20.0, 30.0, 40.0],
                     [np.nan, 20.0, 30.0, 40.0]], np.float32)
    result = eo.county_mean(grid, index, value)
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(12.5)
    # A missing high-weight cell leaves the mean over the remaining weight.
    assert result[1, 0] == pytest.approx(20.0)
    assert result[0, 1] == pytest.approx(35.0)


def test_county_mean_all_missing_is_nan():
    index = np.array([[0]], np.int32)
    value = np.array([[1.0]], np.float32)
    assert np.isnan(eo.county_mean(np.array([[np.nan]]), index, value)[0, 0])


def test_combine_weights_falls_back_to_area_for_uncultivated_counties():
    areas = pd.DataFrame({
        "adm_id": ["US-01-001", "US-01-001", "US-02-002", "US-02-002"],
        "cell_row": [0, 0, 1, 1],
        "cell_col": [0, 1, 0, 1],
        "area_km2": [10.0, 10.0, 5.0, 15.0],
    })
    cultivated = pd.DataFrame({
        "cell_row": [0, 0, 1, 1],
        "cell_col": [0, 1, 0, 1],
        "cultivated": [0.9, 0.1, 0.0, 0.0],
    })
    table, fallback = eo.combine_weights(areas, cultivated, ["US-01-001", "US-02-002"])
    assert fallback == ["US-02-002"]
    assert table.groupby("adm_id", observed=True).weight.sum().tolist() == pytest.approx(
        [1.0, 1.0])
    first = table.loc[table.adm_id.astype(str) == "US-01-001"].set_index("cell_col")
    assert first.weight[0] == pytest.approx(0.9)
    assert first.weight[1] == pytest.approx(0.1)
    second = table.loc[table.adm_id.astype(str) == "US-02-002"].set_index("cell_col")
    assert second.weight[0] == pytest.approx(0.25)


def test_adjust_avhrr_recovers_a_synthetic_linear_relation():
    rng = np.random.default_rng(4)
    counties, steps = 12, 400
    avhrr = np.clip(rng.normal(0.45, 0.15, (counties, steps)), 0, 1).astype(np.float32)
    slope = np.linspace(0.8, 1.4, counties)
    intercept = np.linspace(0.02, -0.04, counties)
    modis = (intercept[:, None] + slope[:, None] * avhrr).astype(np.float32)
    modis[rng.random(modis.shape) < 0.3] = np.nan
    adjusted, fitted_slope, _, r2, pairs = eo.adjust_avhrr(avhrr, modis)
    valid = np.isfinite(r2)
    assert valid.sum() >= counties - 1
    assert np.nanmedian(np.abs(fitted_slope[valid] - slope[valid])) < 1e-4
    assert np.nanmedian(r2[valid]) == pytest.approx(1.0, abs=1e-6)
    assert adjusted.shape == avhrr.shape
    assert np.nanmax(adjusted) <= 1.0
    assert pairs.max() >= 40


def test_adjust_avhrr_falls_back_to_the_median_fit():
    avhrr = np.full((2, 10), 0.4, np.float32)
    modis = np.full((2, 10), 0.5, np.float32)
    adjusted, slope, _, _, pairs = eo.adjust_avhrr(avhrr, modis)
    assert np.allclose(slope, slope[0])
    assert np.isfinite(adjusted).all()
    assert pairs.max() == 10


def test_write_cube_and_load_eo_round_trip(tmp_path):
    counties = ["US-01-001", "US-01-003"]
    values = {"ndvi": np.arange(2 * eo.N_STEPS, dtype="f4").reshape(
        2, eo.N_STEPS) / 100.0}
    variables = {"ndvi": {"name": "ndvi", "units": "1", "source": "test",
                          "nan_fraction": 0.0, "first_valid": "1981-01-01",
                          "last_valid": "2025-12-27"}}
    path = tmp_path / "eo_county_8day.f32"
    metadata = eo.write_cube(path, values, counties, variables)
    assert metadata["shape"] == [2, eo.N_STEPS, 1]
    cube, loaded = eo.load_eo(path)
    assert loaded["counties"] == counties
    assert np.allclose(cube[1, 10, 0], values["ndvi"][1, 10])
    assert not list(tmp_path.glob("*.part"))
    # Rewriting keeps the file readable and preserves the JSON contract.
    eo.write_cube(path, values, counties, variables)
    assert json.loads(path.with_suffix(".json").read_text())["shape"][0] == 2


def test_dn_mask_drops_quality_flags_before_any_averaging():
    dn = np.array([[249.0, 250.0, 255.0, 30.0, 0.0, 120.0, 100.0]], np.float32)
    out = eo._dn_mask(dn, low=0, high=100)
    assert np.isnan(out[0, :3]).all()
    assert np.isnan(out[0, 5])
    assert out[0, 3] == pytest.approx(30.0)
    assert out[0, 4] == pytest.approx(0.0)
    assert out[0, 6] == pytest.approx(100.0)


def test_dn_mask_keeps_the_lst_window():
    dn = np.array([[0.0, 7000.0, 7500.0, 30000.0, 65535.0, 65536.0]], np.float32)
    out = eo._dn_mask(dn, low=7500, high=65535)
    assert np.isnan(out[0, :2]).all()
    assert out[0, 2] == pytest.approx(7500.0)
    assert out[0, 3] * 0.02 - 273.15 == pytest.approx(326.85, abs=1e-6)
    assert np.isnan(out[0, 5])


def test_county_mean_ignores_cells_masked_to_nan():
    index = np.zeros((1, 2), int)
    value = np.ones((1, 2), np.float32)
    grid = np.array([[2.0, np.nan], [np.nan, np.nan]], np.float32)
    mean = eo.county_mean(grid, index, value)
    assert mean[0, 0] == pytest.approx(2.0)
    assert np.isnan(mean[1, 0])


def test_write_fit_round_trips_the_per_county_adjustment(tmp_path):
    weather = tmp_path / "processed" / "weather" / "power_county_daily.json"
    weather.parent.mkdir(parents=True)
    weather.write_text(json.dumps({"counties": ["US-19-169", "US-17-019"]}))
    slope = np.array([0.93, 0.40])
    intercept = np.array([0.08, 0.30])
    r2 = np.array([0.86, 0.12])
    pairs = np.array([1024, 12])
    path = eo._write_fit(tmp_path, slope, intercept, r2, pairs)
    table = pd.read_parquet(path)
    assert table.adm_id.tolist() == ["US-19-169", "US-17-019"]
    assert table.r2.tolist() == pytest.approx([0.86, 0.12])
    assert table.uses_median_fit.tolist() == [False, True]
    assert not list(path.parent.glob("*.part"))


def test_box_grid_puts_the_upper_left_corner_on_the_translation():
    grid = eo._box_grid(-95.0, 44.0, 80, 140)
    assert grid["dimensions"] == {"width": 140, "height": 80}
    assert grid["affineTransform"] == {"scaleX": eo.GRID_CELL,
                                       "scaleY": -eo.GRID_CELL,
                                       "translateX": -95.0, "translateY": 44.0}
