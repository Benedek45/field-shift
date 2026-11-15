"""Offline tests for the NASS progress crossing dates."""

import numpy as np
import pandas as pd

from fieldshift.data import progress


def test_stage_name_strips_the_percent_prefix():
    assert progress.stage_name("PCT PLANTED") == "planted"
    assert progress.stage_name("PCT DROPPING LEAVES") == "dropping leaves"
    assert progress.stage_name("PCT CLOSED MIDDLES (ROWS FILLED)") == (
        "closed middles (rows filled)")


def test_crossing_interpolates_between_weekly_reports():
    dates = pd.to_datetime(["2020-04-05", "2020-04-12", "2020-04-19", "2020-04-26"])
    values = np.array([5.0, 20.0, 60.0, 90.0])
    date, censored = progress.crossing(dates, values, 50)
    assert not censored
    # 50 percent sits 30 of the 40 points between 12 and 19 April, rounded to a day.
    assert date.astype("datetime64[D]") == np.datetime64("2020-04-17")


def test_crossing_exact_report_date_is_used():
    dates = pd.to_datetime(["2020-04-05", "2020-04-12"])
    date, censored = progress.crossing(dates, np.array([10.0, 50.0]), 50)
    assert pd.Timestamp(date) == pd.Timestamp("2020-04-12")
    assert not censored


def test_crossing_never_reached_is_nat():
    dates = pd.to_datetime(["2020-04-05", "2020-04-12"])
    date, censored = progress.crossing(dates, np.array([0.0, 12.0]), 50)
    assert np.isnat(np.datetime64(date))
    assert not censored


def test_crossing_already_past_in_the_first_report_is_censored():
    dates = pd.to_datetime(["2020-04-05", "2020-04-12"])
    date, censored = progress.crossing(dates, np.array([80.0, 95.0]), 50)
    assert censored
    assert pd.Timestamp(date) == pd.Timestamp("2020-04-05")


def test_crossing_ignores_missing_reports_and_unsorted_rows():
    dates = pd.to_datetime(["2020-04-19", "2020-04-05", "2020-04-12"])
    values = np.array([65.0, 10.0, np.nan])
    date, censored = progress.crossing(dates, values, 50)
    assert not censored
    # Between 5 April (10) and 19 April (65): 40 of 55 points is 10.2 days.
    assert date.astype("datetime64[D]") in (
        np.datetime64("2020-04-15"), np.datetime64("2020-04-16"))


def test_crossings_builds_one_row_per_state_crop_year_stage():
    frame = pd.DataFrame({
        "state": ["IA", "IA", "IA", "IA"],
        "commodity": ["CORN"] * 4,
        "class_desc": ["ALL CLASSES"] * 4,
        "prodn_practice": ["ALL PRODUCTION PRACTICES"] * 4,
        "util_practice": ["ALL UTILIZATION PRACTICES"] * 4,
        "year": [2012] * 4,
        "stage": ["planted"] * 4,
        "week_ending": pd.to_datetime(["2012-04-01", "2012-04-08", "2012-04-22",
                                       "2012-04-29"]),
        "value": [0.0, 1.0, 9.0, 50.0],
    })
    table = progress.crossings(frame)
    assert len(table) == 1
    row = table.iloc[0]
    assert row.date_50 == pd.Timestamp("2012-04-29")
    assert row.date_10 == pd.Timestamp("2012-04-22")
    assert not row.date_50_censored
    assert row.n_reports == 4
