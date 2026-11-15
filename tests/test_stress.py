import numpy as np
import pandas as pd
import pytest

from fieldshift.features.stress import compute_stress, daily_stages, sowing_dates
from fieldshift.features.traits import CROPS, load_traits, numeric_traits
from fieldshift.model.trend import YieldTrend
from fieldshift.model.yield_stress import feature_columns, metrics


def weather(start="2020-05-01", periods=20, temperature=20.0):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=periods), "adm_id": "field",
        "tmin": temperature - 5, "tmax": temperature + 5, "tavg": temperature,
        "prec": 2.0, "et0": 3.0, "rad": 15.0, "vpd": 1.0,
        "rsm": 100.0, "rsm_anomaly": -10.0,
    })


def short_traits():
    t = numeric_traits(load_traits("corn"))
    t.update(base_c=10, emergence_gdd=20, flowering_gdd=50,
             flowering_duration_gdd=20, maturity_gdd=100, heat_critical_c=23)
    return t


def test_gdd_boundaries_heat_and_sums():
    d = weather()
    t = short_traits()
    staged = daily_stages(d, t, "2020-05-01")
    assert staged.stage.tolist() == [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    assert staged.gdd.tolist() == [10] * 10
    f = compute_stress(d, t, "2020-05-01").iloc[0]
    assert f.flowering__heat_dd == 4
    assert f.establishment__prec_mm == 4
    assert f.grain_fill__cwb_mm == -3
    assert f.flowering__rsm_anomaly == -10
    assert f.season__length_days == 10
    assert f.maturity_reached == 1


def test_clipped_gdd_and_frost():
    d = weather(periods=1, temperature=40)
    t = short_traits()
    assert daily_stages(d, t, d.date.iloc[0]).gdd.iloc[0] == 20
    d = weather(periods=1, temperature=-5)
    f = compute_stress(d, t, d.date.iloc[0]).iloc[0]
    assert f.establishment__gdd == 0
    assert f.establishment__frost_days == 1
    assert np.isnan(f.flowering__heat_dd)


def test_winter_crossing_vernalization_and_dormancy():
    d = weather("2019-10-01", 300, temperature=10)
    cold = d.date.between("2019-11-01", "2020-02-29")
    d.loc[cold, ["tmin", "tmax", "tavg"]] = [-2, 4, 1]
    t = numeric_traits(load_traits("winter_wheat"))
    s = daily_stages(d, t, "2019-10-01", 2020)
    assert s.date.min().year == 2019
    assert s.date.max().year == 2020
    assert s.loc[s.date.dt.year.eq(2019)].gdd.sum() == 300
    assert s.loc[s.date.between("2019-11-01", "2020-02-29")].gdd.sum() == 0
    assert s.loc[s.date.ge("2020-03-01")].vernalized.eq(1).all()
    assert s.loc[s.date.ge("2020-03-01")].gdd.gt(0).all()


def test_no_chill_no_reproduction():
    d = weather("2019-09-01", 365, temperature=20)
    s = daily_stages(d, load_traits("winter_wheat"), "2019-09-01", 2020)
    assert s.gdd.sum() == 300
    assert s.stage.max() == 1


def test_sowing_dates_and_crop_records():
    d = weather("2020-02-01", 150)
    assert sowing_dates(d, load_traits("corn"), 2020).iloc[0] == pd.Timestamp("2020-03-01")
    d = weather("2019-08-01", 150, temperature=15)
    assert sowing_dates(d, load_traits("winter_wheat"), 2020).iloc[0] == pd.Timestamp("2019-09-01")
    for record in CROPS.values():
        assert all(v.source and v.confidence in {"low", "medium", "high"} for v in record.values())
        assert all(isinstance(v, float) for v in numeric_traits(record).values())


def test_missing_weather_and_duplicate_dates_rejected():
    d = weather()
    with pytest.raises(ValueError, match="Duplicate"):
        daily_stages(pd.concat([d, d]), short_traits(), "2020-05-01")
    d.loc[0, "tavg"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        daily_stages(d, short_traits(), "2020-05-01")
    with pytest.raises(ValueError, match="contiguous"):
        daily_stages(weather().drop(index=3), short_traits(), "2020-05-01")


def test_trend_excludes_test_year_and_short_record():
    train = pd.DataFrame({
        "crop": ["corn"] * 8, "adm_id": ["a"] * 6 + ["b"] * 2,
        "year": [2003, 2004, 2005, 2006, 2007, 2008, 2003, 2004],
        "yield_t_ha": [5, 5.2, 5.4, 5.6, 5.8, 6, 8, 8.2],
    })
    test = pd.DataFrame({"crop": ["corn"], "adm_id": ["a"], "year": [2012], "yield_t_ha": [1.]})
    trend = YieldTrend().fit(train)
    before = trend.predict_log(test)
    test.yield_t_ha = 10000
    np.testing.assert_array_equal(before, trend.predict_log(test))
    assert 2012 not in trend.training_years
    assert np.isfinite(trend.predict_log(test.assign(adm_id="unseen"))).all()


def test_no_crop_or_metadata_model_input_and_metric_units():
    d = pd.DataFrame({"crop": ["corn"], "adm_id": ["a"], "year": [2012],
                      "yield_t_ha": [1.], "trait__base_c": [8.], "season__prec_mm": [1.]})
    assert feature_columns(d) == ["season__prec_mm", "trait__base_c"]
    assert metrics(np.log([.5]), np.log([1.]))["mean_absolute_percent_yield_error"] == 100
