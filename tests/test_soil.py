"""Offline tests for the static soil helpers."""

import numpy as np
import pytest

from fieldshift.data import soil


def test_depth_weight_weights_by_thickness():
    values = {"b0": np.array([10.0]), "b10": np.array([20.0]), "b30": np.array([50.0])}
    depths = (("b0", 5.0), ("b10", 10.0), ("b30", 15.0))
    result = soil.depth_weight(values, depths)
    assert result[0] == pytest.approx((10 * 5 + 20 * 10 + 50 * 15) / 30)


def test_depth_weight_renormalises_over_available_layers():
    values = {"b0": np.array([10.0]), "b10": np.array([np.nan]), "b30": np.array([40.0])}
    depths = (("b0", 10.0), ("b10", 10.0), ("b30", 20.0))
    result = soil.depth_weight(values, depths)
    assert result[0] == pytest.approx((10 * 10 + 40 * 20) / 30)


def test_depth_weight_all_missing_is_nan():
    values = {"b0": np.array([np.nan]), "b10": np.array([np.nan])}
    assert np.isnan(soil.depth_weight(values, (("b0", 1.0), ("b10", 1.0)))[0])


def test_saxton_rawls_orders_the_water_contents():
    field, wilting, awc, ksat = soil.saxton_rawls(0.35, 0.25, 1.5)
    assert wilting < field
    assert 0.10 < float(awc) / 1000 < 0.30
    assert 0.05 < float(field) < 0.45
    assert 0.0 < float(ksat) < 500.0


def test_saxton_rawls_clay_and_organic_matter_raise_the_water_holding():
    sand = np.array([0.45, 0.35, 0.20])
    clay = np.array([0.10, 0.25, 0.40])
    organic = np.array([0.5, 1.5, 4.0])
    _, _, awc, _ = soil.saxton_rawls(sand, clay, organic)
    assert np.all(np.diff(awc) > 0)
    _, _, dry, _ = soil.saxton_rawls(0.35, 0.25, 0.5)
    _, _, rich, _ = soil.saxton_rawls(0.35, 0.25, 5.0)
    assert float(rich) >= float(dry)


def test_saxton_rawls_is_finite_at_the_clipped_extremes():
    field, wilting, awc, ksat = soil.saxton_rawls(
        np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 0.5]),
        np.array([0.0, 30.0, 2.0]))
    assert np.isfinite(field).all() and np.isfinite(wilting).all()
    assert np.isfinite(awc).all() and np.isfinite(ksat).all()
    assert np.all(awc >= 0) and np.all(ksat >= 0)
