import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from fieldshift.data.crop_labels import (
    SPECS,
    census_shares,
    identify_crops,
    interpolate_shares,
)
from fieldshift.features.stress import compute_stress, explanation_stage_names
from fieldshift.features.traits import CROPS, load_traits, numeric_traits
from fieldshift.model.expanded import crop_groups, fold_masks, score
from fieldshift.model.expanded_data import (
    apply_climatology,
    feature_columns,
    weather_spec,
    yearly_frame,
)
from fieldshift.model.neural import accelerator_device
from fieldshift.model.probabilistic import (
    ResidentRows,
    ResidentWeather,
    distribution,
    load_response,
    recalibrate,
    save_response,
    train_response,
)


def test_crop_schema_and_parent_override():
    assert set(SPECS) == set(CROPS)
    keys = set(CROPS["corn"])
    assert all(set(record) == keys for record in CROPS.values())
    corn, silage = numeric_traits(load_traits("corn")), numeric_traits(load_traits("corn_silage"))
    assert silage["maturity_gdd"] < corn["maturity_gdd"]
    assert silage["base_c"] == corn["base_c"]
    assert silage["harvest_whole_plant"] == 1


def test_census_irrigation_and_interpolation():
    d = pd.DataFrame({
        "commodity_desc": ["CORN"] * 4, "class_desc": ["ALL CLASSES"] * 4,
        "util_practice_desc": ["GRAIN"] * 4, "year": ["2002", "2002", "2012", "2012"],
        "prodn_practice_desc": ["ALL PRODUCTION PRACTICES", "IRRIGATED"] * 2,
        "domain_desc": ["TOTAL"] * 4, "domaincat_desc": ["NOT SPECIFIED"] * 4,
        "state_ansi": ["01"] * 4, "state_fips_code": ["01"] * 4,
        "county_ansi": ["001"] * 4, "county_code": ["001"] * 4,
        "value": ["100", "20", "200", "120"],
    })
    shares = census_shares(d)
    np.testing.assert_allclose(shares.irrigated_share, [.2, .6])
    q = pd.DataFrame({"crop": ["corn"] * 5, "adm_id": ["US-01-001"] * 5,
                      "year": [1981, 2002, 2007, 2012, 2025]})
    np.testing.assert_allclose(interpolate_shares(shares, q).irrigated_share, [.2, .2, .4, .6, .6])
    unknown = interpolate_shares(shares, q.assign(adm_id="US-01-003"))
    assert unknown.irrigated_share.isna().all()
    # Processing potatoes are not a duplicate of the all-use harvested-area total.
    potatoes = d.iloc[:2].assign(commodity_desc="POTATOES", util_practice_desc=["FRESH MARKET", "ALL UTILIZATION PRACTICES"])
    assert identify_crops(potatoes).tolist() == ["", "potatoes"]


def test_perennial_multicut_has_no_fake_flowering():
    dates = pd.date_range("2020-01-01", "2020-12-31")
    d = pd.DataFrame({
        "date": dates, "adm_id": "field", "tmin": 10., "tmax": 30.,
        "tavg": 20., "prec": 1., "et0": 4., "vpd": 2., "rad": 20.,
    })
    d.loc[d.date.ge("2020-11-01"), "tmin"] = -5
    f = compute_stress(d, load_traits("alfalfa"), "2020-03-01", 2020).iloc[0]
    assert np.isnan(f.flowering__gdd)
    assert np.isnan(f.maturity_reached)
    assert list(explanation_stage_names(load_traits("alfalfa"))) == ["forage_regrowth"]
    assert explanation_stage_names(load_traits("potatoes"))["grain_fill"] == "bulking"
    assert f.forage__potential_cut_cycles >= 6
    assert f.forage_regrowth__gdd > 0
    assert pd.Timestamp.fromordinal(int(f.season_end_ordinal)) == pd.Timestamp("2020-10-31")
    assert f.season__heat_dd_30 == 0
    assert f.season__water_deficit_mm > 0


def test_split_and_feature_leakage_guards():
    d = pd.DataFrame({"year": range(1981, 2026)})
    fit, calibration, test, fit_years, cal_years = fold_masks(d, [1983, 1988, 2012])
    assert not (fit & calibration).any()
    assert not (fit & test).any()
    assert not (calibration & test).any()
    assert not set(fit_years) & set(cal_years)
    columns = feature_columns(pd.DataFrame(columns=[
        "crop", "year", "season_start_ordinal", "climo__flowering__1",
        "harvested_acres", "series_id", "trait__base_c", "irrigated_share",
    ]))
    assert columns == ["irrigated_share", "trait__base_c"]
    groups = crop_groups(CROPS)
    assert sorted(c for group in groups for c in group) == sorted(CROPS)
    assert ["corn", "corn_silage"] in groups


def test_area_weighted_metrics():
    d = pd.DataFrame({
        "target": [0., 0.], "mlp": [0., np.log(2)], "mlp_std": [.1, .1],
        "harvested_acres": [9., 1.],
    })
    assert score(d, "mlp")["mape"] == pytest.approx(50.)
    assert score(d, "mlp", True)["mape"] == pytest.approx(10.)


def test_moisture_climatology_excludes_test_year(tmp_path):
    weather = pd.DataFrame({
        "adm_id": ["county", "county"], "month": [1, 1], "year": [2000, 2012],
        "sum": [50., 5000.], "count": [10, 10],
    })
    weather.to_parquet(tmp_path / "monthly_rsm.parquet")
    columns = {"adm_id": ["county"], "season__rsm": [10.]}
    for name in ("establishment", "vegetative", "flowering", "grain_fill", "forage_regrowth"):
        columns[name + "__rsm"] = [10.]
        for month in range(1, 13):
            columns[f"climo__{name}__{month}"] = [10 if month == 1 else 0]
    result = apply_climatology(pd.DataFrame(columns), [2000], tmp_path)
    assert result["flowering__rsm_anomaly"].iloc[0] == pytest.approx(5.)
    assert result["season__rsm_anomaly"].iloc[0] == pytest.approx(5.)


def test_power_metadata_units_shape_and_complete_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data/processed/weather"
    root.mkdir(parents=True)
    names = ["tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "cwb", "gwetroot"]
    units = ["degC"] * 3 + ["mm/day", "MJ/m2/day", "kPa", "mm/day", "mm/day", "1"]
    metadata = {
        "shape": [1, 2, 9], "counties": ["US-01-001"], "start_date": "1981-01-01",
        "end_date": "1981-01-02",
        "variables": [{"name": name, "units": unit} for name, unit in zip(names, units)],
    }
    binary = root / "power_county_daily.f32"
    values = np.array([10, 30, 20, 1, 20, 1, 4, -3, .5], dtype="float32")
    np.tile(values, (1, 2, 1)).tofile(binary)
    sidecar = binary.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata))
    _, result = weather_spec("power")
    assert result["variables"] == names
    assert result["rsm_scale"] == 1.
    path, matched = weather_spec("power_matched")
    data = np.memmap(path, dtype="float32", mode="r", shape=(1, 2, 9))
    frame = yearly_frame(data, matched, 1981)
    es_max = .6108 * np.exp(17.27 * 30 / (30 + 237.3))
    es_min = .6108 * np.exp(17.27 * 10 / (10 + 237.3))
    assert frame.vpd.iloc[0] == pytest.approx(1 + .5 * (es_max - es_min), rel=1e-6)
    metadata["variables"][4]["units"] = "J/m2/day"
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="units"):
        weather_spec("power")
    metadata["variables"][4]["units"] = "MJ/m2/day"
    sidecar.write_text(json.dumps(metadata))
    binary.write_bytes(b"")
    with pytest.raises(ValueError, match="complete"):
        weather_spec("power")


@pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE") == "cpu" or
         torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Requires an accelerator or explicit CPU mode",
)
def test_heteroscedastic_training_and_calibration_device_residency(tmp_path):
    rng = np.random.default_rng(2026)
    data = pd.DataFrame({
        "x": rng.normal(size=200), "trait__base_c": 5., "county_index": 0,
        "start_index": 0, "season_length": 100, "crop": "corn",
        "adm_id": [str(i) for i in range(200)], "year": 2010,
    })
    rows = ResidentRows(data, ["x", "trait__base_c"], rng.normal(0, .2, 200))
    train = rows.indices(np.arange(200) < 50)
    calibration = rows.indices(np.arange(200) >= 50)
    net, _ = train_response(rows, train, [1], epochs=2)
    device = torch.empty(0, device=accelerator_device()).device
    assert all(p.device == device for p in net.parameters())
    before = distribution(net, rows, calibration)
    assert before.device == device
    record = recalibrate(net, rows, calibration)
    assert record["calibration_rows"] == 150
    assert .25 <= record["scale"] <= 4
    assert torch.isfinite(distribution(net, rows, calibration)).all()
    expected = distribution(net, rows, calibration)
    path = tmp_path / "response.pt"
    save_response(net, ["x", "trait__base_c"], path, {"calibration": record})
    loaded, columns, metadata = load_response(path)
    assert columns == ["x", "trait__base_c"]
    assert metadata["calibration"] == record
    torch.testing.assert_close(distribution(loaded, rows, calibration), expected)


@pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE") == "cpu" or
         torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Requires an accelerator or explicit CPU mode",
)
def test_sequence_gathers_and_pools_on_selected_device(tmp_path):
    variables = ["tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "cwb", "rsm"]
    path = tmp_path / "weather.f32"
    array = np.memmap(path, mode="w+", dtype="float32", shape=(2, 500, 9))
    array[:] = 1.
    array[0, 4, 8] = np.nan
    array.flush()
    weather = ResidentWeather(path, {"shape": [2, 500, 9], "variables": variables, "rsm_scale": 500.})
    assert weather.values.dtype == torch.float16
    d = pd.DataFrame({
        "x": np.linspace(-1, 1, 16), "trait__base_c": 5.,
        "county_index": np.arange(16) % 2, "start_index": 1, "season_length": 100,
        "crop": "corn", "adm_id": [str(i) for i in range(16)], "year": 2010,
    })
    rows = ResidentRows(d, ["x", "trait__base_c"], np.zeros(16))
    index = rows.indices(np.ones(16, dtype=bool))
    windows = weather.windows(rows.county[index], rows.start[index], rows.length[index])
    assert windows.device == torch.empty(0, device=accelerator_device()).device
    assert windows.shape == (16, 18, 366)
    assert windows[:, :, 100:].count_nonzero() == 0
    net, _ = train_response(rows, index, [1], epochs=1, sequence=True, weather=weather)
    predicted = distribution(net, rows, index, weather)
    assert predicted.device == windows.device
    assert torch.isfinite(predicted).all()
