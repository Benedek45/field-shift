import http.client
import json
import os
import re
import threading

import pytest
import torch

from fieldshift.agent import farms
from fieldshift.agent.brief import build, climate
from fieldshift.agent.runtime import InputError, Runtime
from fieldshift.agent.service import make_server

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE", "").strip().lower() == "cpu"
         or torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Select FIELDSHIFT_DEVICE=cpu or provide MPS/CUDA")

STORY = [{"crop": "soybean"}, {"crop": "corn"}, {"crop": "soybean"}, {"crop": "corn"}]
RYE = [s | {"winter": "cereal_rye"} for s in STORY]


@pytest.fixture(autouse=True)
def offline_outlook(monkeypatch):
    from fieldshift.agent import outlook
    def offline(*_args, **_kwargs):
        raise ConnectionError("offline test: no seasonal forecast fetch")
    monkeypatch.setattr(outlook, "fetch", offline)


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent")
    ids = {
        "story": farms.add({"adm_id": "US-19-169"}, root),
        "anonymous": farms.add({"adm_id": "US-19-169", "anonymous": True}, root),
        "dry": farms.add({"adm_id": "US-19-169", "overrides": {"rain_change_pct": -30}}, root),
        "steep": farms.add({"adm_id": "US-19-169", "overrides": {"slope_pct": 8}}, root),
        "poor": farms.add({"adm_id": "US-19-169", "overrides": {"organic_matter_pct": 1}}, root),
    }
    return root, ids, Runtime(root, max_farms=3)


def med(result, metric):
    return result["metrics"][metric]["median"]


def test_story_physics_matches_v22_and_operating_returns_use_source_split(context):
    _, ids, runtime = context
    r = runtime.simulate(ids["story"], {"rotations": [STORY, RYE]})
    a, b = r["results"]
    # ISU2025 corn/soy fixed machinery102.05/69.92USD/ac. Remove corn's
    # embedded69USD/ac N and charge simulated124lb/ac at cited$.50/lb N.
    change = (102.05+69.92)/2*2.47105 + (69*2.47105-124*1.12085*.50/.45359237)/2
    assert med(a, "margin_usd_ha_yr") == pytest.approx(940+change, abs=.51)
    assert med(b, "margin_usd_ha_yr") == pytest.approx(782+change, abs=.51)
    assert med(a, "nitrate_leached_kg_n_ha_yr") == pytest.approx(51, abs=.51)
    assert med(b, "nitrate_leached_kg_n_ha_yr") == pytest.approx(42, abs=.51)
    assert a == r["existing"]
    partial = runtime.simulate(ids["story"], {"rotations": [[STORY[0], STORY[1] | {"winter": "cereal_rye"}]]})["results"][0]
    assert med(b, "days_without_living_cover_yr") < med(partial, "days_without_living_cover_yr") < med(a, "days_without_living_cover_yr")


def test_briefs_plain_anonymous_and_irrigated(context):
    root, ids, runtime = context
    plain = build(ids["story"], runtime)
    anonymous = build(ids["anonymous"], runtime)
    assert "Story County, IA" in plain
    for banned in ("Story", "Iowa", "IA", "19169", "2012", "1988"):
        assert banned not in anonymous
    assert not re.search(r"\b\d{4}\b", anonymous)
    assert len(anonymous.split()) < 2500
    assert "year -44 through year 0" in anonymous
    assert "No forecast available" in anonymous
    assert "model estimates, NOT measurements" in anonymous
    assert "1.87e-02" in anonymous  # Anonymization must not alter erodibility.
    lubbock = farms.add({"adm_id": "US-48-303", "irrigated": True}, root)
    brief = build(lubbock, runtime)
    assert "Lubbock" in brief and "annual cap" in brief and "applied water is priced" in brief
    assert "Rule 5.2(a)" in brief
    assert len(brief.split()) < 2500


def test_farm_overrides_and_request_edits_are_consistent(context):
    _, ids, runtime = context
    baseline = runtime.simulate(ids["story"], {"rotations": [STORY]})["results"][0]
    dry = runtime.simulate(ids["dry"], {"rotations": [STORY]})["results"][0]
    with runtime.lock:
        base_rain = climate(runtime.farm(ids["story"]))["seasonal_rain"]
        dry_rain = climate(runtime.farm(ids["dry"]))["seasonal_rain"]
    assert torch.allclose(dry_rain, base_rain * .7, atol=.001)
    for key in ("corn", "soybean"):
        assert dry["relative_yield_by_crop"][key]["median"] < baseline["relative_yield_by_crop"][key]["median"]
    what_if = runtime.simulate(ids["story"], {"rotations": [STORY], "rain_change_pct": -30})["results"][0]
    assert med(what_if, "margin_usd_ha_yr") == pytest.approx(med(dry, "margin_usd_ha_yr"), abs=.01)
    steep = runtime.simulate(ids["steep"], {"rotations": [STORY]})["results"][0]
    assert med(steep, "erosion_t_ha_yr") > med(baseline, "erosion_t_ha_yr")
    poor = runtime.simulate(ids["poor"], {"rotations": [STORY]})["results"][0]
    assert med(poor, "mineralized_n_kg_n_ha_yr") < med(baseline, "mineralized_n_kg_n_ha_yr")


@pytest.mark.parametrize("adm,irrigated", [
    ("US-19-169", False), ("US-31-185", True), ("US-20-191", False),
    ("US-48-303", True), ("US-28-011", True), ("US-06-019", True),
])
def test_six_farms_existing_references_respond(context, adm, irrigated):
    root, _, runtime = context
    farm_id = farms.add({"adm_id": adm, "irrigated": irrigated}, root)
    with runtime.lock:
        f = runtime.farm(farm_id)
        cycle = f.existing
        assert f.options()["main_crops"]
    response = runtime.simulate(farm_id, {"rotations": [cycle]})
    assert response["device"] in ("mps", "cuda:0", "cuda")
    assert response["existing"]["metrics"]["margin_usd_ha_yr"]["median"] is not None


def test_cycle_lengths_and_validation(context):
    _, ids, runtime = context
    for length in range(1, 7):
        cycle = [{"crop": "corn"}] * length
        result = runtime.simulate(ids["story"], {"rotations": [cycle]})["results"][0]
        assert len(result["rotation"]) == length
    for body, match in [
        ({"rotations": [[{"crop": "invented"}]]}, "valid main keys"),
        ({"rotations": [STORY] * 13}, "1 to 12"),
        ({"rotations": [[{"crop": "corn", "winter": "invented"}]]}, "valid winter keys"),
        ({"rotations": [[{"crop": "corn"}, {"crop": "winter_wheat"}]]}, "autumn sowing"),
        ({"rotations": [[{"crop": "alfalfa"}, {"crop": "corn"}]]}, "2 adjacent years"),
        ({"rotations": [STORY], "rain_change_pct": float("nan")}, "finite number"),
    ]:
        with pytest.raises((InputError, farms.FarmError), match=match):
            runtime.simulate(ids["story"], body)


def test_temperature_awc_and_farmer_constraints(context):
    root, ids, runtime = context
    modified = farms.add({"adm_id": "US-19-169",
                          "overrides": {"temp_change_c": 2, "awc_mm_per_m": 100},
                          "priorities": "Prioritize infiltration",
                          "constraints": ["no new equipment", "keep a pollinator strip"]}, root)
    with runtime.lock:
        farm = runtime.farm(modified)
        assert farm.site["soil"]["awc_mm_per_m"] == 100
        r, t = farm.changes()
        p = farm.preparation(r, t)
        d = farm.weather()
        assert t == 2 and r == 0
        assert torch.allclose(p["weather"][0, 0, :365, 2],
                              torch.tensor(d.loc[d.date.dt.year.eq(1981), "tmin"].to_numpy("float32"),
                                           device=p["device"]))
        assert not farm.options()["winter_options"]
    brief = build(modified, runtime)
    assert "Prioritize infiltration" in brief and "keep a pollinator strip" in brief
    assert "100.0 mm/m soil" in brief
    with pytest.raises(InputError, match="valid winter keys"):
        runtime.simulate(modified, {"rotations": [RYE]})
    forage = farms.add({"adm_id": "US-19-169", "constraints": ["must keep a forage crop"]}, root)
    with pytest.raises(InputError, match="forage"):
        runtime.simulate(forage, {"rotations": [[{"crop": "corn"}]]})
    response = runtime.simulate(ids["story"], {"rotations": [STORY], "rain_change_pct": -20, "temp_change_c": 1})
    assert response["applied_weather_changes"]["temp_change_c_total"] == 1


def test_http_auth_errors_brief_options_simulate_and_log(context, tmp_path):
    _, ids, runtime = context
    token = "test-token-do-not-deploy"
    logfile = tmp_path / "requests.jsonl"
    server = make_server(token, "127.0.0.1", 0, runtime, logfile)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def request(method, route, body=None, auth=True):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=120)
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + token
        connection.request(method, f"/farm/{ids['story']}/{route}", body, headers)
        response = connection.getresponse()
        data, status = response.read().decode(), response.status
        connection.close()
        return status, data

    try:
        assert request("GET", "options", auth=False)[0] == 401
        assert request("GET", "options")[0] == 200
        assert "Story County" in request("GET", "brief")[1]
        status, data = request("POST", "simulate", json.dumps({"rotations": [STORY, RYE]}))
        assert status == 200 and len(json.loads(data)["results"]) == 2
        # The service adds US-unit copies beside the unchanged metric values.
        served = json.loads(data)["results"][0]["metrics"]
        direct = runtime.simulate(ids["story"], {"rotations": [STORY]})["results"][0]["metrics"]
        assert served["margin_usd_ha_yr"]["median"] == pytest.approx(direct["margin_usd_ha_yr"]["median"])
        assert served["margin_usd_acre_yr"]["median"] == round(direct["margin_usd_ha_yr"]["median"] / 2.471053814671653)
        assert served["field_water_use_in_yr"]["median"] == round(direct["field_water_use_mm_yr"]["median"] / 25.4, 1)
        assert "nitrate_leached_lb_n_acre_yr" in served and "margin_usd_acre_yr" not in direct
        assert "us_units_note" in json.loads(data)
        status, data = request("POST", "scoreboard", json.dumps({"weights": {"income": 5}, "limits": {}, "top": 3}))
        board = json.loads(data)
        assert status == 200 and "margin_usd_acre_yr" in board["existing"]["metrics"]
        assert all("erosion_us_tons_acre_yr" in r["metrics"] for r in board["rows"])
        assert "margin_usd_acre_yr" not in board["metric_definitions"]
        assert request("POST", "simulate", "{not json}")[0] == 422
        status, data = request("POST", "simulate", '{"rotations":[[{"crop":"invented"}]]}')
        assert status == 422 and "valid main keys" in data and "Traceback" not in data
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    rows = [json.loads(v) for v in logfile.read_text().splitlines()]
    assert len(rows) == 7 and any(v["rotations"] for v in rows)
    assert token not in logfile.read_text()
