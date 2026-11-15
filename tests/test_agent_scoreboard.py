import http.client
import json
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from fieldshift.agent import farms, scoreboard
from fieldshift.agent.runtime import InputError, Runtime
from fieldshift.agent.service import make_server
from fieldshift.planner import search

ACCELERATOR = torch.backends.mps.is_available() or torch.cuda.is_available()
requires_accelerator = pytest.mark.skipif(not ACCELERATOR, reason="No CPU fallback for simulator or bootstrap")


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    root = tmp_path_factory.mktemp("scoreboard")
    farm_id = farms.add({"adm_id": "US-19-169"}, root)
    return root, farm_id, Runtime(root)


@pytest.mark.parametrize("payload", [
    {}, {"weights": {}}, {"weights": {"income": 0}}, {"weights": {"unknown": 1}},
    {"weights": {"nitrate": 1}}, {"weights": {"income": -1}}, {"weights": {"income": True}},
    {"weights": {"income": float("nan")}}, {"weights": {"income": 1}, "top": 41},
    {"weights": {"income": 1}, "limits": {"min_margin_share_of_existing": 1.1}},
    {"weights": {"income": 1}, "limits": {"require_winter_cover": 1}},
    {"weights": {"income": 1}, "limits": {"require_crops": "corn"}},
])
def test_reject_invalid_scoreboard_requests(payload):
    with pytest.raises((InputError, farms.FarmError)):
        scoreboard.validate(payload)


def test_large_finite_weights_normalise_without_overflow():
    weights, *_ = scoreboard.validate({"weights": {"income": 1e308, "nitrogen": 1e308}})
    assert sum(weights.values()) == pytest.approx(1)
    assert weights["income"] == .5


@requires_accelerator
def test_determinism_weights_and_limits_do_not_resimulate(context, monkeypatch):
    root, farm_id, runtime = context
    original = scoreboard.simulate_pool
    calls = []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(scoreboard, "simulate_pool", counted)
    request = {"weights": scoreboard.DEFAULT_WEIGHTS, "top": 5}
    first = runtime.scoreboard(farm_id, request)
    second = runtime.scoreboard(farm_id, request)
    assert json.dumps(first["rows"]) == json.dumps(second["rows"])
    assert first["n_cycles_scored"] == 434 and first["n_after_limits"] == 434
    # V24 operating return adds ISU2025 ownership102.05/69.92USD/ac and
    # substitutes actual124lb N/ac for the69USD/ac reference corn N bill.
    change=(102.05+69.92)/2*2.47105 + (69*2.47105-124*1.12085*.50/.45359237)/2
    assert first["existing"]["metrics"]["margin_usd_ha_yr"]["median"] == pytest.approx(940+change, abs=.51)
    changed = runtime.scoreboard(farm_id, {"weights": {"nitrogen": 5, "income": 1}, "top": 5})
    assert changed["rows"][0]["rotation"] != first["rows"][0]["rotation"]
    assert changed["existing"]["rank"] != first["existing"]["rank"]
    # A fresh Runtime/process can rank cached arrays without preparing any Farm.
    other = Runtime(root)
    monkeypatch.setattr(other, "farm", lambda *_: pytest.fail("Cached rerank prepared a farm"))
    constrained = other.scoreboard(farm_id, {"weights": {"income": 1},
                                           "limits": {"no_new_equipment": True, "exclude_crops": ["soybean"]}})
    assert constrained["n_after_limits"] == 1
    assert constrained["rows"][0]["score"] == pytest.approx(50)  # Constant metric AFTER limits.
    assert constrained["existing"]["rank"] is not None
    assert "exclude_crops:soybean" in constrained["existing"]["violates_limits"]
    assert len(calls) == 1


@requires_accelerator
def test_hard_limits_and_existing_are_explicit(context):
    _, farm_id, runtime = context
    request = {"weights": {"income": 1}, "top": 40, "limits": {
        "min_margin_share_of_existing": .7, "max_nitrate_kg_n_ha_yr": 50,
        "max_field_water_use_mm_yr": 650, "max_erosion_t_ha_yr": 20,
        "max_days_without_living_cover_yr": 100, "require_crops": ["corn"],
        "exclude_crops": ["sorghum"], "require_winter_cover": True}}
    response = runtime.scoreboard(farm_id, request)
    assert response["n_after_limits"] > 0
    assert "require_winter_cover" in response["existing"]["violates_limits"]
    assert response["existing"]["rank_is_counterfactual"]
    for row in response["rows"]:
        metrics = row["metrics"]
        assert metrics["margin_usd_ha_yr"]["median"] >= .7 * response["existing"]["metrics"]["margin_usd_ha_yr"]["median"]
        assert metrics["nitrate_leached_kg_n_ha_yr"]["median"] <= 50
        assert metrics["field_water_use_mm_yr"]["median"] <= 650
        assert metrics["erosion_t_ha_yr"]["median"] <= 20
        assert metrics["days_without_living_cover_yr"]["median"] <= 100
        assert "corn" in {s["crop"] for s in row["rotation"]}
        assert "sorghum" not in {s["crop"] for s in row["rotation"]}
        assert any(s.get("winter") for s in row["rotation"])
    forage = runtime.scoreboard(farm_id, {"weights": {"nitrogen": 1}, "limits": {"keep_forage_crop": True}})
    assert forage["rows"] and all(any(s["crop"] in ("hay", "alfalfa") for s in row["rotation"]) for row in forage["rows"])
    empty = runtime.scoreboard(farm_id, {"weights": {"income": 1}, "limits": {"max_field_water_use_mm_yr": 0}})
    assert empty["rows"] == [] and empty["n_after_limits"] == 0
    assert empty["existing"]["rank"] is None and empty["existing"]["violates_limits"]
    with pytest.raises(InputError, match="valid keys"):
        runtime.scoreboard(farm_id, {"weights": {"income": 1}, "limits": {"require_crops": ["millet"]}})


@requires_accelerator
def test_published_crop_can_be_excluded_after_weather_makes_it_infeasible(context):
    _, farm_id, runtime = context
    with runtime.lock:
        farm = runtime.farm(farm_id)
        options = farm.options()
        p = farm.preparation(*farm.changes(0, -15))
        unavailable = [c["crop"] for c in options["main_crops"] if c["crop"] not in farm.site["existing"]
                       and float(p["maturity"][0, farm.keys.index(c["crop"])]) < .6]
    assert unavailable
    result = runtime.scoreboard(farm_id, {"weights": {"income": 1}, "temp_change_c": -15,
                                        "limits": {"exclude_crops": [unavailable[0]]}, "top": 1})
    assert result["rows"]
    assert all(s["crop"] != unavailable[0] for row in result["rows"] for s in row["rotation"])


@requires_accelerator
def test_constructed_tie_matches_existing_bootstrap_rule():
    raw = np.tile(np.linspace(5, 15, 10), 4)
    runs = np.zeros((3, 40, len(search.RUN_COLUMNS)), dtype=np.float32)
    runs[0, :, 0] = raw
    runs[1, :, 0] = raw + np.tile([-.8, .8], 20) + .1
    runs[2, :, 0] = raw - 8
    points, rows = [], []
    for i in range(3):
        p = dict.fromkeys(search.METRICS, 0.)
        p.update(id=i, feasible=True, income=float(np.median(runs[i, :, 0])),
                 income_median=float(np.median(runs[i, :, 0])))
        points.append(p)
        rows.append({"infeasible": [], "rotation": [{"crop": "corn"}], "forage": False,
                     "equipment": False, "label": str(i), "metrics": {}, "income_confidence": "low", "warnings": []})
    data = {"points": points, "rows": rows, "pruned": [], "windows": 10, "n_cycles": 3}
    result = scoreboard.ranking(data, runs, {"income": 1}, {}, 3)
    best, middle, worst = result["rows"]
    assert best["label"] == "1" and best["tied_with_best"]
    assert middle["tied_with_best"] and not worst["tied_with_best"]
    expected = search.near_tie(pd.DataFrame(points), runs, 1, 0, {"income": 1}, 10)
    assert middle["tie_tolerance"] == pytest.approx(expected["tolerance"])
    for row in result["rows"]:
        assert row["tied_with_best"] == (best["score"] - row["score"] <= row["tie_tolerance"])


@requires_accelerator
def test_double_crop_scoreboard_row_is_executable(context):
    root, _, runtime = context
    farm_id = farms.add({"adm_id": "US-20-191"}, root)
    with runtime.lock:
        farm = runtime.farm(farm_id)
        # The saved v22 Sumner plan explicitly prunes double cropping (<60%
        # maturity). Exercise the executable transport with a SYNTHETIC
        # short-season cultivar; do not loosen the real feasibility gate.
        from fieldshift.sim.engine import prepare
        farm.double = farm.double.override(maturity_gdd=1000, flowering_gdd=500, flowering_duration_gdd=100)
        farm.base = prepare([farm.site], farm.crops, farm.covers, farm.double, irrigated=farm.irrigated)
        farm.prepared.clear()
        data, _ = scoreboard.simulate_pool(farm, 0, 0)
    row = next(r for r in data["rows"] if any(s.get("double_crop") for s in r["rotation"]) and not r["infeasible"])
    response = runtime.simulate(farm_id, {"rotations": [row["rotation"]]})["results"][0]
    for metric in ("margin_usd_ha_yr", "nitrate_leached_kg_n_ha_yr", "field_water_use_mm_yr"):
        assert response["metrics"][metric]["median"] == pytest.approx(row["metrics"][metric]["median"], abs=.01)
    assert response["metrics"]["double_crop_yield_t_ha_yr"]["median"] > 0


@requires_accelerator
def test_scoreboard_applies_all_farm_overrides_like_simulate(context):
    root, _, runtime = context
    farm_id = farms.add({"adm_id": "US-19-169", "overrides": {
        "rain_change_pct": -30, "temp_change_c": 1, "slope_pct": 8,
        "organic_matter_pct": 1, "awc_mm_per_m": 100}}, root)
    ranked = runtime.scoreboard(farm_id, {"weights": {"income": 1}, "top": 1, "rain_change_pct": -10})
    with runtime.lock:
        cycle = runtime.farm(farm_id).existing
    simulated = runtime.simulate(farm_id, {"rotations": [cycle], "rain_change_pct": -10})["results"][0]
    for metric in ("margin_usd_ha_yr", "nitrate_leached_kg_n_ha_yr", "erosion_t_ha_yr",
                   "field_water_use_mm_yr", "mineralized_n_kg_n_ha_yr", "organic_matter_trend_pct_points_decade"):
        assert ranked["existing"]["metrics"][metric]["median"] == pytest.approx(
            simulated["metrics"][metric]["median"], abs=.01)


@requires_accelerator
def test_http_scoreboard_auth_validation_and_weights_help(context, tmp_path):
    _, farm_id, runtime = context
    token = "local-scoreboard-test"
    log = tmp_path / "requests.jsonl"
    server = make_server(token, "127.0.0.1", 0, runtime, log)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, route, payload=None, auth=True):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=120)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + token
        connection.request(method, f"/farm/{farm_id}/{route}", json.dumps(payload) if payload is not None else None, headers)
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        return response.status, body
    try:
        assert request("POST", "scoreboard", {"weights": {"income": 1}}, auth=False)[0] == 401
        for payload in ({"weights": {}}, {"weights": {"unknown": 1}}):
            status, body = request("POST", "scoreboard", payload)
            assert status == 422 and body["error"] == "invalid_request"
        status, options = request("GET", "options")
        assert status == 200 and set(options["weights_help"]) == set(scoreboard.KEYS)
        assert options["weights_help"]["nitrogen"]["direction"] == "lower"
        status, result = request("POST", "scoreboard", {"weights": scoreboard.DEFAULT_WEIGHTS, "top": 5})
        assert status == 200 and len(result["rows"]) == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert any(json.loads(line)["scoreboard_request"] for line in log.read_text().splitlines())
