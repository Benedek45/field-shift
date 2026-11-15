"""Farmer forms and the independent product checks; no provider calls."""

import copy
import json

import pytest

from fieldshift.agent import checks, farms, form
from fieldshift.sim.economics import ACRES_HA


def sample():
    return {"location": {"adm_id": "US-19-169", "irrigated": False},
            "priorities": {"preset": "steady"}, "weather_view": "record"}


def test_units_and_presets():
    data = sample()
    data["location"] |= {"irrigated": True, "well_capacity": {"value": 18, "unit": "in/yr"}}
    data["field_facts"] = {"water_holding_capacity": {"value": 2, "unit": "in/ft"}}
    data["priorities"] |= {"money": 2, "risk_attitude": "balanced"}
    data["economics"] = {"fertilizer_n_usd_lb": .5, "water_usd_acre_foot": 100,
                         "land_rent_usd_acre": 200, "energy_usd_acre_inch": 10,
                         "crop_prices": {"corn": {"value": 5.5, "unit": "usd/bu"},
                                         "cotton": {"value": .7, "unit": "usd/lb"},
                                         "alfalfa": {"value": 200, "unit": "usd/ton"}},
                         "crop_cost_parts_usd_acre": {"corn": {"seed": 100}}}
    data["limits"] = {"minimum_share_current_income": {"value": 80, "unit": "%"},
                      "max_nitrate_kg_n_ha_yr": {"value": 40, "unit": "lb_n/ac/yr"}}
    n = form.validate(data)
    assert n["farm"]["overrides"]["irrigation_capacity_mm_yr"] == pytest.approx(457.2)
    assert n["farm"]["overrides"]["awc_mm_per_m"] == pytest.approx(166.6666667)
    e = n["farm"]["economics"]
    assert e["fertilizer_n_usd_kg"] == pytest.approx(.5 / .45359237)
    assert e["water_charge_usd_per_mm_ha"] == pytest.approx(100 * ACRES_HA / 304.8)
    assert e["land_rent_usd_ha"] == pytest.approx(200 * ACRES_HA)
    assert e["irrigation_energy_usd_per_mm_ha"] == pytest.approx(10 * ACRES_HA / 25.4)
    assert e["crop_price_usd_t"]["corn"] == pytest.approx(5.5 * ACRES_HA / .06277)
    assert e["crop_price_usd_t"]["cotton"] == pytest.approx(.7 * 1000 / .45359237)
    assert e["crop_price_usd_t"]["alfalfa"] == pytest.approx(200 / .90718474)
    assert e["crop_cost_usd_ha"]["corn"]["seed"] == pytest.approx(100 * ACRES_HA)
    assert n["weights"]["income"] == 2 and n["weights"]["resilience"] == 3
    assert n["limits"]["min_margin_share_of_existing"] == .8
    assert n["limits"]["max_nitrate_kg_n_ha_yr"] == pytest.approx(40 * .45359237 * ACRES_HA)


def test_presets_match_a3():
    from pathlib import Path
    profiles = json.loads(Path("vm/a3_cases.json").read_text())
    for name, profile in zip(("profit", "steady", "soil_builder", "water_saver"),
                             ("p1", "p2", "p3", "p4"), strict=True):
        actual = next(p["weights"] for p in profiles if p["id"] == "story-" + profile)
        named = {k: v for k, v in actual.items() if v > 1 or k in ("income", "resilience")}
        assert form.PRESETS[name] == {k: named.get(k, 0) for k in actual}


def test_county_lookup(monkeypatch):
    monkeypatch.setattr(form, "county_options", lambda: {"US-19-169": "Story County, IA · US-19-169"})
    assert form.validate({"location": {"county": "Story, IA", "irrigated": False}})["farm"]["adm_id"] == "US-19-169"


@pytest.mark.parametrize("patch", [
    {"extra": 1}, {"priorities": {"money": 6}}, {"priorities": {"money": True}},
    {"priorities": {"preset": []}}, {"weather_view": "tomorrow"},
    {"current_rotation": [{"crop": "millet"}]}, {"current_rotation": [{"crop": "corn", "winter": False}]},
    {"limits": {"require_crops": ["corn"], "exclude_crops": ["corn"]}},
    {"economics": {"crop_prices": {"corn": {"value": 5, "unit": "usd/lb lint"}}}},
])
def test_invalid_forms(patch):
    with pytest.raises(ValueError):
        form.validate(sample() | patch)


def test_nonfinite_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"location":{"adm_id":"US-19-169","irrigated":false},"priorities":{"money":NaN}}')
    with pytest.raises(ValueError, match="Nonfinite"):
        form.load(path)


def fixture_decision():
    n = form.validate(sample())
    rotation = [{"crop": "corn"}, {"crop": "soybean"}]
    row = {"rotation": rotation, "rank": 1, "tied_with_best": True,
           "metrics": {"margin_usd_ha_yr": {"median": 100},
                       "nitrate_leached_kg_n_ha_yr": {"median": 20}}}
    board = {"existing": row, "rows": [row]}
    d = {"existing": {"rotation": row["rotation"]}, "weights": n["weights"],
         "weight_reasons": dict.fromkeys(n["weights"], "form"), "limits": n["limits"],
         "limit_reasons": dict.fromkeys(n["limits"], "form"),
         "recommended": {"rotation": row["rotation"], "override_reason": None},
         "alternatives": [], "why": "The checked choice follows the supplied priorities.",
         "evidence": [], "information_gaps": [], "risks_for_local_review": []}
    return n, board, d, {"metrics": row["metrics"]}


def test_checks_passing_failing_and_grounded_units():
    n, board, d, simulated = fixture_decision()
    assert checks.evaluate(d, n, board, simulated)["all_passed"]
    bad = copy.deepcopy(d)
    bad["weights"]["income"] = 0
    bad["limits"] = {"max_nitrate_kg_n_ha_yr": 10}
    bad["evidence"] = [{"source": "card:invented", "claim": "Better"}]
    result = checks.evaluate(bad, n, board, simulated)
    assert not result["form_weights"]["passed"]
    assert not result["limits_grounded_in_form_or_quote"]["passed"]
    assert not result["all_limits_and_feasibility"]["passed"]
    assert not result["cards_exist_and_opened"]["passed"]
    n["free_text"] = "No more than 40 lb N/acre nitrate loss."
    extra, reasons = checks.text_limits(n["free_text"])
    d["limits"] = extra
    d["limit_reasons"] = {k: {"quote": v} for k, v in reasons.items()}
    assert checks.ground_limits(d, n) == []
    d["limits"]["max_nitrate_kg_n_ha_yr"] = 40
    assert checks.ground_limits(d, n)


def test_numeric_rounding_and_missing_calls():
    calls = {"2": {"margin_usd_ha_yr": 1161.1213, "field_water_use_mm_yr": 500.01}}
    assert checks.numeric_claims({}, "$1161/ha, 500 mm (sim:2)", calls)["passed"]
    assert checks.numeric_claims({}, "$470/acre (sim:2)", calls)["passed"]
    assert not checks.numeric_claims({}, "$9999/ha (sim:2)", calls)["passed"]
    assert not checks.numeric_claims({}, "$1161/ha (sim:3)", calls)["passed"]
    assert not checks.numeric_claims({}, "1.1e-4 (sim:2)", calls)["passed"]
    assert checks.numeric_claims({}, "about $470 per acre (sim:2)", calls)["dollar_errors"] == []
    wrong = checks.numeric_claims({}, "about $450 per acre, 9999 mm (sim:2)", calls)
    assert len(wrong["dollar_errors"]) == 1 and wrong["numbers_unmatched"] == 2
    loss = {"3": {"margin_usd_acre_yr": {"median": -6, "p10": -34}}}
    assert checks.numeric_claims({}, "a typical return of **-$6/acre**, bad year −$34 [sim:3]", loss)["dollar_errors"] == []
    assert len(checks.numeric_claims({}, "a typical return of $6/acre [sim:3]", loss)["dollar_errors"]) == 1


def test_rotation_shape_and_physical_identity():
    base = farms.validate(form.validate(sample())["farm"])
    changed = copy.deepcopy(base)
    changed["existing_rotation"] = farms.validate_rotation(
        [{"crop": "corn"}, {"crop": "soybean", "winter": "cereal_rye"}, {"crop": "soybean"}])
    assert changed["existing_rotation"][1]["winter"] == "cereal_rye_cover"
    assert farms.physical_record(base) != farms.physical_record(changed)
