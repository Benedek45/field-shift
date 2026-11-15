"""Tests for the PRISM 8-day county weather cube."""

import json

import numpy as np
import pytest

from fieldshift.data import eo, prism


def write_weather_json(path, counties):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counties": counties}))


def test_variables_and_bounds_cover_every_published_name():
    assert list(prism.VARIABLES) == ["ppt_sum", "tmean", "tmin", "tmax", "tdmean",
                                     "vpdmin", "vpdmax", "wet_days", "ppt_max"]
    assert set(prism.BOUNDS) == set(prism.VARIABLES)
    assert set(prism.CELL_BOUNDS) == set(prism.VARIABLES)
    for name, (low, high) in prism.BOUNDS.items():
        assert low < high, name


def test_check_bounds_accepts_a_clean_cube():
    arrays = {"ppt_sum": np.array([[0.0, 120.0, np.nan]], np.float32),
              "wet_days": np.array([[0.0, 8.0, 3.0]], np.float32)}
    assert prism.check_bounds(arrays)


def test_check_bounds_rejects_a_fill_value():
    arrays = {"ppt_sum": np.array([[0.0, 9999.0]], np.float32)}
    with pytest.raises(ValueError, match="outside physical bounds"):
        prism.check_bounds(arrays)
    with pytest.raises(ValueError, match="ppt_sum"):
        prism.check_bounds({"ppt_sum": np.array([[-1.0]], np.float32)})


def test_check_bounds_ignores_nan_and_unknown_names():
    arrays = {"ppt_sum": np.array([[np.nan, 5.0]], np.float32),
              "not_a_variable": np.array([[1e9]], np.float32)}
    assert prism.check_bounds(arrays)


def test_clean_drops_cells_outside_the_cell_bounds():
    data = {"ppt_sum": np.array([[0.0, 5.0, 5000.0]], np.float32),
            "tmin": np.array([[-70.0, -5.0, 10.0]], np.float32),
            "wet_days": np.array([[0.0, 3.0, 9.0]], np.float32)}
    cleaned = prism._clean(data)
    assert np.isnan(cleaned["ppt_sum"][0, 2])
    assert np.isnan(cleaned["tmin"][0, 0])
    assert np.isnan(cleaned["wet_days"][0, 2])
    assert cleaned["ppt_sum"][0, 1] == pytest.approx(5.0)
    assert cleaned["tmin"][0, 1] == pytest.approx(-5.0)


def test_write_and_load_round_trip(tmp_path):
    counties = ["US-19-169", "US-17-019"]
    series = {name: np.full((2, eo.N_STEPS), np.nan, np.float32)
              for name in prism.VARIABLES}
    series["ppt_sum"][0, 10] = 25.0
    series["wet_days"][0, 10] = 2.0
    variables = {name: {"name": name, "units": prism.VARIABLES[name][0],
                        "source": prism.VARIABLES[name][1], "nan_fraction": 0.0,
                        "first_valid": "1981-01-09", "last_valid": "2025-12-27"}
                 for name in prism.VARIABLES}
    # build() needs weights; write_cube is the part that touches disk.
    metadata = eo.write_cube(tmp_path / prism.CUBE_NAME, series, counties, variables)
    assert metadata["shape"] == [2, eo.N_STEPS, 9]
    cube, loaded = prism.load_prism(tmp_path)
    assert loaded["counties"] == counties
    assert np.isclose(cube[0, 10, 0], 25.0)
    assert np.isclose(cube[0, 10, 7], 2.0)
    frame = prism.county_frame(cube, loaded, "US-19-169")
    assert list(frame.columns)[:2] == ["start_date", "ppt_sum"]
    assert frame["ppt_sum"].iloc[10] == pytest.approx(25.0)
    del cube


def test_step_series_reads_the_cache_without_earth_engine(tmp_path):
    cache = tmp_path / "prism"
    cache.mkdir()
    names = list(prism.VARIABLES)
    array = np.zeros((len(names), eo.HEIGHT, eo.WIDTH), np.float32)
    array[names.index("ppt_sum")] = 4.0
    array[names.index("tmin")] = -3.0
    np.save(cache / "7.npy", array)
    index = np.array([[0, 1]])
    value = np.array([[0.5, 0.5]], np.float32)
    counties = prism._step_series(7, cache, index, value)
    assert counties["ppt_sum"] == pytest.approx(4.0)
    assert counties["tmin"] == pytest.approx(-3.0)
    assert counties["ppt_max"] == pytest.approx(0.0)


def test_step_series_masks_a_cell_outside_the_bounds_on_reuse(tmp_path):
    cache = tmp_path / "prism"
    cache.mkdir()
    names = list(prism.VARIABLES)
    array = np.zeros((len(names), eo.HEIGHT, eo.WIDTH), np.float32)
    array[names.index("ppt_sum"), 0, 0] = 5000.0
    array[names.index("ppt_sum"), 0, 1] = 8.0
    np.save(cache / "9.npy", array)
    index = np.array([[0, 1]])
    value = np.array([[1.0, 1.0]], np.float32)
    counties = prism._step_series(9, cache, index, value)
    # One of the two cells is impossible, so the county mean uses the other one.
    assert counties["ppt_sum"] == pytest.approx(8.0)


def test_day_expression_is_not_called_for_a_cached_step(tmp_path, monkeypatch):
    cache = tmp_path / "prism"
    cache.mkdir()
    array = np.zeros((9, eo.HEIGHT, eo.WIDTH), np.float32)
    np.save(cache / "3.npy", array)

    def explode(*_args, **_kwargs):
        raise AssertionError("Earth Engine must not be called for a cached step")

    monkeypatch.setattr(prism, "day_expression", explode)
    index = np.array([[0]])
    value = np.array([[1.0]], np.float32)
    prism._step_series(3, cache, index, value)
