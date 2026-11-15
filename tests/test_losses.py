import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from fieldshift.data import eo, losses


def rma_row(**overrides):
    """One cause-of-loss record with the documented 30 element order."""
    values = {
        "commodity_year": "2012", "state_fips": "19", "state": "IA",
        "county_fips": "169", "county_name": "Story", "commodity_code": "0041",
        "commodity": "CORN", "plan_code": "02", "plan": "RP",
        "coverage_category": "A", "stage_code": "H", "cause_code": "11",
        "cause": "Drought", "month": "07", "month_name": "JUL",
        "loss_year": "2012", "month_sequence": "1", "policies_indemnified": "3",
        "net_planted": "1000.0000000000", "net_endorsed_acres": ".0000000000",
        "liability": "500000.0000000000", "total_premium": "20000.0000000000",
        "producer_premium": "8000.0000000000", "subsidy": "12000.0000000000",
        "state_private_subsidy": ".0000000000", "additional_subsidy": ".0000000000",
        "efa_discount": ".0000000000", "net_determined": "400.0000000000",
        "indemnity": "120000.0000000000", "loss_ratio": "6.00",
    }
    values.update(overrides)
    return "|".join(values[k] for k in losses.RMA_COLUMNS)


def write_zip(path, rows):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("colsom12.txt", "\n".join(rows) + "\n")
    return path


def test_cause_groups_cover_the_reported_descriptions():
    assert losses.cause_group("Drought") == "drought"
    assert losses.cause_group("Heat") == "heat"
    assert losses.cause_group("Excess Moisture/Precipitation/Rain") == "excess_water"
    assert losses.cause_group("Cold Wet Weather") == "excess_water"
    assert losses.cause_group("Freeze") == "cold"
    assert losses.cause_group("Hurricane/Tropical Depression") == "wind"
    assert losses.cause_group("Plant Disease") == "disease_insects"
    assert losses.cause_group("Decline in Price") == "other"


def test_adm_id_pads_to_cybench_form():
    assert losses.adm_id("19", "169") == "US-19-169"
    assert losses.adm_id("1", "1") == "US-01-001"


def test_read_year_types_the_documented_elements(tmp_path):
    path = write_zip(tmp_path / "colsom_2012.zip", [rma_row()])
    frame = losses.read_year(path)
    assert frame.net_planted.iloc[0] == pytest.approx(1000.0)
    assert frame.indemnity.iloc[0] == pytest.approx(120000.0)
    assert frame["month"].iloc[0] == 7
    assert str(frame["month"].dtype) == "Int16"
    assert frame.commodity_year.iloc[0] == 2012
    assert frame.adm_id.iloc[0] == "US-19-169"
    assert frame.cause_group.iloc[0] == "drought"


def test_download_year_reuses_the_cached_file(tmp_path):
    path = write_zip(tmp_path / "colsom_2012.zip", [rma_row()])
    before = path.stat().st_mtime_ns
    assert losses.download_year(None, 2012, tmp_path) == path
    assert path.stat().st_mtime_ns == before


def write_nass(path, rows):
    frame = pd.DataFrame(rows, columns=[
        "commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc",
        "statisticcat_desc", "unit_desc", "state_ansi", "county_ansi", "year", "value"])
    frame.to_csv(path, index=False)
    return frame


def nass_row(commodity="CORN", statistic="AREA PLANTED", year=2012, value=5000.0,
             practice="ALL PRODUCTION PRACTICES"):
    return [commodity, "ALL CLASSES", practice, "ALL UTILIZATION PRACTICES",
            statistic, "ACRES", 19, 169.0, year, value]


def write_power_json(path, counties):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"counties": counties}))


def synthetic_table():
    return pd.DataFrame({
        "adm_id": ["US-19-169", "US-19-169", "US-19-169"],
        "state": ["IA"] * 3,
        "county_name": ["Story"] * 3,
        "commodity": ["CORN", "CORN", "SOYBEANS"],
        "commodity_code": ["0041"] * 3,
        "year": [2012, 2012, 2012],
        "month": pd.array([7, 8, 7], dtype="Int16"),
        "cause": ["Drought", "Heat", "Drought"],
        "cause_group": ["drought", "heat", "drought"],
        "plan": ["RP"] * 3,
        "coverage_category": ["A"] * 3,
        "net_planted": [1000.0, 1000.0, 500.0],
        "net_determined": [400.0, 100.0, 50.0],
        "indemnity": [120000.0, 30000.0, 10000.0],
        "total_premium": [20000.0, 20000.0, 5000.0],
        "liability": [500000.0, 500000.0, 200000.0],
        "loss_ratio": [6.0, 1.5, 2.0],
        "policies_indemnified": [3, 1, 2],
    })


def test_cause_share_counts_each_cause_once_and_joins_nass_planted(tmp_path):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    write_nass(data / "processed" / "nass_county_crops.csv.gz",
               [nass_row(), nass_row(commodity="SOYBEANS", value=1000.0),
                nass_row(statistic="YIELD", value=180.0),
                nass_row(practice="IRRIGATED", value=999.0)])
    share = losses.cause_share(synthetic_table(), data)
    corn = share.loc[share.commodity_key == "CORN"]
    assert corn.indemnity.sum() == pytest.approx(150000.0)
    assert corn.indemnity_share.sum() == pytest.approx(1.0)
    assert (corn.planted_acres == 5000.0).all()
    july = corn.loc[corn.month == 7].iloc[0]
    assert july.damage_share_of_planted == pytest.approx(400.0 / 5000.0)
    soy = share.loc[share.commodity_key == "SOYBEANS"].iloc[0]
    assert soy.indemnity_share == pytest.approx(1.0)
    assert soy.damage_share_of_planted == pytest.approx(50.0 / 1000.0)


def test_cause_share_is_blank_where_nass_has_no_planted_acres(tmp_path):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    write_nass(data / "processed" / "nass_county_crops.csv.gz",
               [nass_row(commodity="CORN")])
    share = losses.cause_share(synthetic_table(), data)
    assert share.loc[share.commodity_key == "SOYBEANS"].planted_acres.isna().all()


def test_usdm_html_parsing_reads_hidden_fields_and_county_boxes():
    html = (
        '<input type="hidden" name="__VIEWSTATE" value="/wEPaA==" id="x">'
        '<input type="checkbox" name="ctl00$ContentPlaceHolder1$chkStatisticsAOI_county$0"'
        ' value="19169" checked="checked">'
        '<input type="checkbox" name="ctl00$ContentPlaceHolder1$chkStatisticsAOI_county$1"'
        ' value="17019">')
    assert losses._hidden(html) == {"__VIEWSTATE": "/wEPaA=="}
    boxes = dict(losses._county_boxes(html))
    assert boxes == {"ctl00$ContentPlaceHolder1$chkStatisticsAOI_county$0": "19169",
                     "ctl00$ContentPlaceHolder1$chkStatisticsAOI_county$1": "17019"}


def test_usdm_steps_place_weeks_on_the_shared_grid(tmp_path):
    data = tmp_path / "data"
    write_power_json(data / "processed" / "weather" / "power_county_daily.json",
                     ["US-19-169", "US-17-019"])
    steps = eo.step_starts()
    first = pd.Timestamp(steps[0].astype("datetime64[D]"))
    weekly = pd.DataFrame({
        "adm_id": ["US-19-169", "US-19-169", "US-17-019"],
        "map_date": [first, first + pd.Timedelta(days=7), first],
        "d0": [10.0, 20.0, np.nan],
        "d1": [5.0, 5.0, 1.0],
        "d2": [np.nan, 0.0, 0.0],
        "d3": [0.0, 0.0, 0.0],
        "d4": [0.0, 0.0, 0.0],
        "validStart": [first - pd.Timedelta(days=3),
                       first + pd.Timedelta(days=4),
                       first - pd.Timedelta(days=3)],
        "validEnd": [first + pd.Timedelta(days=3),
                     first + pd.Timedelta(days=10),
                     first + pd.Timedelta(days=3)],
    })
    values = losses.usdm_steps(weekly, data)
    assert values["d0"].shape == (2, eo.N_STEPS)
    # Step 0 spans days 1-8 of the calendar; both weeks overlap it, so the first
    # step is a weighted mean of the two maps and the second is the second map.
    step0 = values["d0"][0, 0]
    assert np.isfinite(step0)
    assert values["d1"][0, 0] == pytest.approx(5.0)
    assert values["d1"][0, 1] == pytest.approx(5.0)
    assert values["d2"][1, 0] == pytest.approx(0.0)
    assert np.isnan(values["d0"][1, 0])
    assert np.isnan(values["d0"][0, 10])


def test_write_usdm_cube_round_trips_on_the_shared_grid(tmp_path):
    data = tmp_path / "data"
    write_power_json(data / "processed" / "weather" / "power_county_daily.json",
                     ["US-19-169", "US-17-019", "US-01-001"])
    values = {name: np.full((3, eo.N_STEPS), np.nan, np.float32)
              for name in losses.USDM_NAMES}
    values["d2"][1, 5] = 42.0
    metadata = losses.write_usdm_cube(values, data)
    assert metadata["shape"] == [3, eo.N_STEPS, 5]
    assert metadata["names"] == list(losses.USDM_NAMES)
    path = data / "processed" / "usdm" / "usdm_8day.f32"
    cube = np.memmap(path, dtype="<f4", mode="r", shape=tuple(metadata["shape"]))
    assert cube[1, 5, 2] == pytest.approx(42.0)
    assert np.isnan(cube[0, 5, 2])
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved["counties"] == ["US-19-169", "US-17-019", "US-01-001"]
    assert saved["steps_per_year"] == eo.STEPS_PER_YEAR
    del cube
    assert not list(path.parent.glob("*.part"))
