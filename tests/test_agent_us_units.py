import copy

import pytest

from fieldshift.agent import checks, us_units
from fieldshift.sim.economics import ACRES_HA

METRICS = {
    "margin_usd_ha_yr": {"p10": 813.5786, "median": 1043.3174, "p90": 1126.6924},
    "field_water_use_mm_yr": {"p10": 451.82, "median": 477.01, "p90": 523.01},
    "soil_water_at_planting_mm": {"p10": 276.09, "median": 278.97, "p90": 279.85},
    "nitrate_leached_kg_n_ha_yr": {"p10": 3.9, "median": 51.23, "p90": 66.21},
    "erosion_t_ha_yr": {"p10": 1.56, "median": 2.3028, "p90": 2.8998},
    "cover_biomass_t_dm_ha_yr": {"p10": 0.0, "median": 1.2, "p90": None},
    "soil_carbon_input_t_c_ha_yr": {"p10": 1.0571, "median": 1.2058, "p90": 1.2664},
    "days_without_living_cover_yr": {"p10": 229.9, "median": 234.1, "p90": 239.5},
    "organic_matter_trend_pct_points_decade": {"p10": -0.1, "median": -0.05, "p90": -0.03},
    "forage_fraction": {"p10": 0.0, "median": 0.0, "p90": 0.0},
}


def response():
    return {"existing": {"rank": 1, "metrics": copy.deepcopy(METRICS)},
            "metric_definitions": {"margin_usd_ha_yr": "Return over operating costs (USD/ha/year)"},
            "economics": {"water_charge_usd_per_mm_ha": 0.5, "crop_price_usd_t": {"corn": 181.1}},
            "difference_from_climatology": {"operating_return_usd_ha": -24.7, "applied_irrigation_mm": 12.7}}


def test_key_names():
    assert us_units.us_key("margin_usd_ha_yr")[0] == "margin_usd_acre_yr"
    assert us_units.us_key("field_water_use_mm_yr")[0] == "field_water_use_in_yr"
    assert us_units.us_key("soil_water_at_planting_mm")[0] == "soil_water_at_planting_in"
    assert us_units.us_key("nitrate_leached_kg_n_ha_yr")[0] == "nitrate_leached_lb_n_acre_yr"
    assert us_units.us_key("soil_mineral_n_kg_n_ha")[0] == "soil_mineral_n_lb_n_acre"
    assert us_units.us_key("erosion_t_ha_yr")[0] == "erosion_us_tons_acre_yr"
    assert us_units.us_key("cover_biomass_t_dm_ha_yr")[0] == "cover_biomass_us_tons_dm_acre_yr"
    assert us_units.us_key("soil_carbon_input_t_c_ha_yr")[0] == "soil_carbon_input_us_tons_c_acre_yr"
    for key in ("days_without_living_cover_yr", "forage_fraction", "organic_matter_trend_pct_points_decade",
                "rank", "score", "crop_price_usd_t", "water_charge_usd_per_mm_ha", "income", "p10"):
        assert us_units.us_key(key) is None, key


def test_values_rounding_and_metric_untouched():
    data = response()
    before = copy.deepcopy(data)
    out = us_units.annotate(data)
    assert data == before  # input not mutated
    m = out["existing"]["metrics"]
    for key, value in METRICS.items():
        assert m[key] == value  # metric keys kept exactly
    assert m["margin_usd_acre_yr"] == {"p10": 329, "median": 422, "p90": 456}
    assert all(isinstance(v, int) for v in m["margin_usd_acre_yr"].values())
    assert m["field_water_use_in_yr"] == {"p10": 17.8, "median": 18.8, "p90": 20.6}
    assert m["nitrate_leached_lb_n_acre_yr"] == {"p10": 3.5, "median": 46, "p90": 59}
    assert m["erosion_us_tons_acre_yr"]["median"] == pytest.approx(round(2.3028 * 1.10231131 / ACRES_HA, 2))
    assert m["cover_biomass_us_tons_dm_acre_yr"] == {"p10": 0.0, "median": 0.54, "p90": None}
    assert m["soil_carbon_input_us_tons_c_acre_yr"]["median"] == 0.54
    # The US key sits right after its metric key.
    keys = list(m)
    assert keys[keys.index("margin_usd_ha_yr") + 1] == "margin_usd_acre_yr"
    # No US copies of percentages, days, ranks, prices, rates or text definitions.
    for key in ("days_without_living_cover_yr", "forage_fraction", "organic_matter_trend_pct_points_decade"):
        assert sum(k.startswith(key.rsplit("_", 1)[0]) for k in m) == 1
    assert out["economics"] == before["economics"] and out["existing"]["rank"] == 1
    assert out["metric_definitions"] == before["metric_definitions"]
    # Forecast differences are plain numbers.
    assert out["difference_from_climatology"]["operating_return_usd_acre"] == -10
    assert out["difference_from_climatology"]["applied_irrigation_in"] == 0.5
    assert round(-0.01, 1) + 0 == 0 and str(us_units._round(-0.01, "in")) == "0.0"


def test_note_only_when_annotated():
    assert "us_units_note" in us_units.with_note(response())
    assert "us_units_note" not in us_units.with_note({"rank": 1, "score": 2.0})


CALLS = {"1": us_units.annotate({"existing": {"metrics": copy.deepcopy(METRICS)}})}


def test_us_unit_check_pass_and_fail():
    ok = checks.numeric_claims({}, "Typical return about $422 per acre, water about 19 inches, "
                                   "nitrate about 46 lb N per acre (sim:1).", CALLS)
    assert ok["us_unit_errors"] == [] and ok["dollar_errors"] == []
    bad = checks.numeric_claims({}, "Field water use is about 193 inches and nitrate 51 lb N/acre (sim:1).", CALLS)
    assert len(bad["us_unit_errors"]) == 2
    # 51 is the kg figure, present in the call, so only the unit-aware check catches it.
    assert any("51 lb" in e for e in bad["us_unit_errors"])
    assert checks.numeric_claims({}, "Return about $423 per acre (sim:1).", CALLS)["dollar_errors"] != []


def test_card_cited_sentence_not_checked_against_simulator():
    line = ("The model estimates about 18.8 inches of water use (sim:1). A central Iowa study found rye cut "
            "nitrate load by 61% (card:kaspar2007-rye-nitrate). These are modelled numbers.")
    result = checks.numeric_claims({}, line, CALLS)
    assert result["passed"] and result["us_unit_errors"] == []
    # An uncited sentence on a sim-citing line is still checked against that line's call.
    mixed = checks.numeric_claims({}, "Water use is 18.8 inches (sim:1). It is 777 mm.", CALLS)
    assert mixed["numbers_unmatched"] == 1


def test_figure_checks_gate_on_us_units():
    found = checks.figure_checks("About 193 inches of water (sim:1).", CALLS)
    assert not found["us_unit_figures_match_cited_runs"]["passed"]
    assert found["dollar_figures_match_cited_runs"]["passed"]
    clean = checks.figure_checks("About 18.8 inches and $422 per acre [sim:1].", CALLS)
    assert all(v["passed"] for v in clean.values())
