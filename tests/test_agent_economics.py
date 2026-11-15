import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest
import torch

from fieldshift.agent import farms, scoreboard
from fieldshift.agent.runtime import Runtime
from fieldshift.sim.components import accelerator, water_step
from fieldshift.sim.economics import (
    ACRES_HA,
    ECON_FIELDS,
    context,
    income_evidence,
    reprice,
    resolved,
)
from fieldshift.sim.traits import CATALOG

live_data = pytest.mark.skipif(not Path("data/processed/nass_county_crops.csv.gz").exists(),
                               reason="Requires local archives")


@pytest.mark.parametrize("value", [
    {"labour_usd_hr": -1}, {"fertilizer_n_usd_kg": True}, {"mystery": 10},
    {"crop_price_usd_t": {"millet": 3}}, {"crop_cost_usd_ha": {"corn": {"magic": 10}}},
    {"usd_per_acre_foot": 100, "water_charge_usd_per_mm_ha": 1},
])
def test_economic_validation(value):
    with pytest.raises(farms.FarmError):
        farms.validate_economics(value)


def test_nonmain_conditional_repeat_fees_are_not_silently_ignored():
    for crop in ("cereal_rye_cover","double_crop_soybean"):
        with pytest.raises(farms.FarmError,match="main crops"):
            farms.validate_economics({"crop_cost_usd_ha":{crop:{"repeat_crop_fixed":10}}})


def test_acre_foot_conversion_and_farmer_zero():
    x = farms.validate_economics({"usd_per_acre_foot": 100, "labour_usd_hr": 0})
    assert x["water_charge_usd_per_mm_ha"] == pytest.approx(100 * ACRES_HA / 304.8)
    assert x["labour_usd_hr"] == 0


@live_data
def test_kansas_operating_fixed_land_and_actual_n():
    # K-State2026 rotation wheat PDF pp2–4:62bu*$5.45, direct206.19,
    # capital recovery53.82, cash rent37/ac; N61.30 for101lb/ac.
    ctx = context({"adm_id":"US-20-191"},[CATALOG["winter_wheat"]],[],None,False)
    dev=accelerator()
    f={k:torch.zeros(1,device=dev) for k in ECON_FIELDS}
    f["harvest_t_ha"][:]=62*.06725
    f["n_fertilizer_kg_ha"][:]=101*.45359237*ACRES_HA
    quote={"crop_price_usd_t":{"winter_wheat":5.45*ACRES_HA/.06725}}
    op,listed,_,_=reprice(f,ctx,quote)
    assert float(op[0]) == pytest.approx(131.71*ACRES_HA,abs=.02)
    assert float(listed[0]) == pytest.approx((131.71-53.82)*ACRES_HA,abs=.02)
    rented,net,_,_=reprice(f,ctx,quote|{"land_rent_usd_ha":37*ACRES_HA})
    assert torch.equal(op,rented)
    assert float(net[0]) == pytest.approx(40.89*ACRES_HA,abs=.02)
    f["n_fertilizer_kg_ha"]-=10
    less,_,_,_=reprice(f,ctx,quote)
    assert float(less[0]-op[0]) == pytest.approx(10*61.30/(101*.45359237),abs=.002)


def test_every_foreign_cost_budget_is_marked_transfer():
    tx=context({"adm_id":"US-48-303"},[CATALOG["alfalfa"]],[],None,True)
    ca=context({"adm_id":"US-06-019"},[CATALOG["cotton"]],[],None,True)
    ks=context({"adm_id":"US-20-191"},[CATALOG["soybean"]],[],None,False)
    ms=context({"adm_id":"US-28-011"},[CATALOG["corn"]],[],None,True)
    assert not tx["crops"][0]["transfer"] and not ks["crops"][0]["transfer"]
    for c in (ca["crops"][0],ms["crops"][0]):
        assert c["transfer"]
        assert c["parts"]["seed"]["confidence"]=="low"
        assert c["parts"]["seed"]["transfer"]


@live_data
def test_isu_repeat_corn_uses_own_operating_ownership_split_without_double_n():
    # ISU2025 pp2–3, middle yield: continuous corn total926.75 less
    # land286; variable478.58 + labour56.42 is operating535/ac.
    ctx=context({"adm_id":"US-19-169"},[CATALOG["corn"]],[],None,False)
    f={k:torch.zeros(1,device=accelerator()) for k in ECON_FIELDS}
    f["economic_same_crop"][:]=1
    f["n_fertilizer_kg_ha"][:]=181*.45359237*ACRES_HA
    op,listed,_,_=reprice(f,ctx,{"crop_price_usd_t":{"corn":0}})
    assert float(op[0])==pytest.approx(-535*ACRES_HA,abs=.03)
    assert float(listed[0])==pytest.approx(-(926.75-286)*ACRES_HA,abs=.03)


def test_provenance_and_confidence_rise_with_main_farmer_inputs():
    ctx = context({"adm_id": "US-48-303"}, [CATALOG["cotton"]], [], None, True)
    quote = resolved(ctx, {"crop_price_usd_t": {"cotton": 1600},
                           "crop_cost_usd_ha": {"cotton": {"seed": 200}},
                           "fertilizer_n_usd_kg": 2})
    assert quote["crops"][0]["parts"]["seed"]["source"] == "farmer"
    assert quote["crops"][0]["n_price"]["confidence"] == "high"
    assert income_evidence(ctx, ["cotton"])[0] is None
    assert income_evidence(quote, ["cotton"])[0] == "medium"


def test_economics_endpoint_forks_immutably(tmp_path):
    from fieldshift.agent.service import make_server
    old = farms.add({"adm_id": "US-19-169"}, tmp_path)
    server = make_server("test-economics-only", "127.0.0.1", 0, Runtime(tmp_path), tmp_path/"http.jsonl")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/farm/{old}/economics"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer test-economics-only"}
        request = urllib.request.Request(url, json.dumps({"labour_usd_hr": 25}).encode(), headers)
        with urllib.request.urlopen(request) as response:
            new = json.load(response)["farm_id"]
        assert new != old and farms.get(new, tmp_path)["economics"]["labour_usd_hr"] == 25
        assert farms.get(old, tmp_path)["economics"] == {}
        bad = urllib.request.Request(url, b'{"unknown":1}', headers)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 422
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_official_state_fallback_before_county_state_mean(monkeypatch):
    from fieldshift.planner import data
    # Synthetic coordinates/values test hierarchy, not default agronomic numbers.
    target, other = "US-06-019", "US-06-001"
    table = pd.DataFrame({"alfalfa": [float("nan"), 5.]}, index=[target, other])
    static = pd.DataFrame({"centroid_lat": [36., 42.], "centroid_lon": [-119., -110.]}, index=[target, other])
    monkeypatch.setattr(data, "practice_normals", lambda practice: (table, {}))
    monkeypatch.setattr(data, "_static", lambda: static)
    official = (15.624649, "NASS official STATE ALL: irrigation_checks.json, CA 2017–2026")
    monkeypatch.setattr(data, "state_normal", lambda state,key,practice: official)
    assert data.practice_normal(target, "alfalfa", "ALL PRODUCTION PRACTICES") == official
    monkeypatch.setattr(data, "state_normal", lambda state,key,practice: None)
    assert data.practice_normal(target, "alfalfa", "ALL PRODUCTION PRACTICES")[0] == 5


def test_cotton_practice_class_and_lint_units(monkeypatch):
    from fieldshift.planner import data
    # Synthetic 1000 lb/acre verifies the conversion, not a county yield estimate.
    rows = pd.DataFrame({
        "commodity_desc": ["COTTON"]*3, "class_desc": ["UPLAND", "UPLAND", "PIMA"],
        "unit_desc": ["LB / ACRE"]*3, "util_practice_desc": ["ALL UTILIZATION PRACTICES"]*3,
        "prodn_practice_desc": ["IRRIGATED", "ALL PRODUCTION PRACTICES", "IRRIGATED"],
        "adm_id": ["US-48-303"]*3, "year": [2020]*3, "value": [1000., 2000., 3000.]})
    monkeypatch.setattr(data, "practice_yields", lambda: rows)
    monkeypatch.setattr(data, "_static", lambda: pd.DataFrame(index=["US-48-303"]))
    table, _ = data.practice_normals.__wrapped__("IRRIGATED")
    assert table.loc["US-48-303", "cotton"] == pytest.approx(1.12085)


@live_data
def test_budget_reference_no_double_irrigation_charge():
    # TAMU 2026 D2 pivot cotton p1:1000lb lint@.68, seed170.40,
    # total963.29, rent100, pumping100, system55. Net excluding rent=-12.89/ac.
    crop = CATALOG["cotton"]
    ctx = context({"adm_id": "US-48-303"}, [crop], [], None, True)
    dev = accelerator()
    fields = {k: torch.zeros(1, device=dev) for k in ECON_FIELDS}
    fields.update(harvest_t_ha=torch.tensor([1000*.00112085], device=dev),
                  irrigation_mm=torch.tensor([254.], device=dev),
                  n_fertilizer_kg_ha=torch.tensor([90*1.12085], device=dev))
    income, listed, irrigation, _ = reprice(fields, ctx, {"crop_price_usd_t": {"cotton": .68/.00045359237}})
    assert float(income[0]) == pytest.approx((850.40-758.04) * ACRES_HA, abs=.02)
    assert float(listed[0]) == pytest.approx(-12.89 * ACRES_HA, abs=.02)
    assert float(irrigation[0]) == pytest.approx((100+55)*ACRES_HA, abs=.01)


@live_data
def test_custom_baling_tracks_harvested_mass_and_can_be_overridden():
    ctx = context({"adm_id": "US-48-303"}, [CATALOG["alfalfa"]], [], None, True)
    dev = accelerator()
    fields = {k: torch.zeros(2, device=dev) for k in ECON_FIELDS}
    reference = ctx["crops"][0]["harvest_reference_yield_t_ha"]["value"]
    fields["harvest_t_ha"] = torch.tensor([reference/2, reference], device=dev)
    income, _, _, _ = reprice(fields, ctx, {"crop_price_usd_t": {"alfalfa": 0}})
    assert float(income[0]-income[1]) == pytest.approx(390*ACRES_HA/2, abs=.01)
    flat, _, _, _ = reprice(fields, ctx, {"crop_price_usd_t": {"alfalfa": 0},
                                      "crop_cost_usd_ha": {"alfalfa": {"harvest": 100}}})
    assert float(flat[0]) == float(flat[1])  # explicit farmer USD/ha bill, not USD/t


@live_data
def test_water_step_cap_mass_balance_and_deficit():
    dev = accelerator()
    z = torch.zeros(2, device=dev)
    hundred = torch.full_like(z, 100.)
    new, f = water_step(z, z, torch.full_like(z, 10), torch.ones_like(z), hundred,
                        torch.full_like(z, .5), True, True, torch.tensor([0., 5.], device=dev))
    assert f["irrigation_mm"].tolist() == [0., 5.]
    assert torch.allclose(new+f["et_mm"]+f["drainage_mm"], f["irrigation_mm"])
    assert float(f["ks"][0]) == 0


@live_data
def test_price_edit_reuses_field_and_scoreboard(tmp_path, monkeypatch):
    r = Runtime(tmp_path)
    a = farms.add({"adm_id": "US-19-169"}, tmp_path)
    f = r.farm(a)
    before = r.simulate(a, {"rotations": [f.existing]})["results"][0]
    board = r.scoreboard(a, {"weights": {"income": 1}, "top": 5})
    price = next(x for x in f.options()["main_crops"] if x["crop"] == "corn")["price"]["value"]
    b = farms.edit_economics(a, {"crop_price_usd_t": {"corn": price*1.2}}, tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Economic edit re-simulated the field")
    import fieldshift.agent.runtime as rt
    monkeypatch.setattr(rt, "water_pass", forbidden)
    monkeypatch.setattr(scoreboard, "simulate_pool", forbidden)
    after = r.simulate(b, {"rotations": [f.existing]})["results"][0]
    changed = r.scoreboard(b, {"weights": {"income": 1}, "top": 5})
    assert after["metrics"]["margin_usd_ha_yr"]["median"] > before["metrics"]["margin_usd_ha_yr"]["median"]
    assert before["metrics"].keys() == board["existing"]["metrics"].keys()
    for q in ("p10","median","p90"):
        assert before["metrics"]["soil_carbon_input_t_c_ha_yr"][q] == pytest.approx(
            board["existing"]["metrics"]["soil_carbon_input_t_c_ha_yr"][q], rel=1e-4, abs=1e-6)
    for key in before["metrics"].keys() - {"margin_usd_ha_yr","return_over_listed_costs_usd_ha_yr"}:
        assert before["metrics"][key] == after["metrics"][key]
    assert changed["existing"]["metrics"]["margin_usd_ha_yr"]["median"] > board["existing"]["metrics"]["margin_usd_ha_yr"]["median"]
    quote = next(x for x in r.farm(b).options()["main_crops"] if x["crop"] == "corn")["price"]
    assert quote["source"] == "farmer" and quote["confidence"] == "high"
    assert farms.get(a, tmp_path)["economics"] == {}
    assert a != b and json.loads((tmp_path/"farms.json").read_text())[a]["economics"] == {}
    rented=farms.edit_economics(b,{"land_rent_usd_ha":500},tmp_path)
    rent_board=r.scoreboard(rented,{"weights":{"income":1},"top":5})
    assert [(x["label"],x["score"]) for x in rent_board["rows"]]==[(x["label"],x["score"]) for x in changed["rows"]]
    assert rent_board["existing"]["metrics"]["return_over_listed_costs_usd_ha_yr"]["median"]==pytest.approx(
        changed["existing"]["metrics"]["return_over_listed_costs_usd_ha_yr"]["median"]-500,abs=.001)


@live_data
def test_annual_capacity_is_physical_not_price(tmp_path):
    r = Runtime(tmp_path)
    a = farms.add({"adm_id": "US-48-303", "irrigated": True}, tmp_path)
    f = r.farm(a)
    before = r.simulate(a, {"rotations": [f.existing]})["results"][0]
    b = farms.edit_economics(a, {"irrigation_capacity_mm_yr": 5}, tmp_path)
    after = r.simulate(b, {"rotations": [f.existing]})["results"][0]
    assert after["metrics"]["applied_irrigation_mm_yr"]["p90"] <= 5.001
    assert after["relative_yield_by_crop"]["cotton"]["median"] < before["relative_yield_by_crop"]["cotton"]["median"]
    for ledger in r.farm(b).physical_results.values():
        assert bool((ledger["irrigation_mm"] <= 5.001).all())
        assert bool((ledger["irrigation_mm"][:, -1] > 0).any())  # annual reset, not a lifetime cap
