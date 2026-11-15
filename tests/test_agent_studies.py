import json
import os
import threading

import pytest
import torch

from fieldshift.agent import checks, evidence, farms, studies
from fieldshift.agent.runtime import Runtime
from fieldshift.agent.service import make_server

ROOT = evidence.default_root()
STORY = {"state": "IA", "irrigated": False, "annual_rain_mm": 909, "texture": "loam", "texture_group": "medium",
         **studies.farm_climate_class(909, 999, 9.6, 171, .35)}
AKRON = {"state": "CO", "irrigated": False, "annual_rain_mm": 433, "texture": "loam", "texture_group": "medium",
         **studies.farm_climate_class(433, 1480, 9.9, 164, .4)}
LUBBOCK = {"state": "TX", "irrigated": False, "annual_rain_mm": 477, "texture": "sandy clay loam",
           "texture_group": "medium", **studies.farm_climate_class(477, 1878, 15.5, 223, .35)}


def ids(query, farm, **kw):
    return [r["id"] for r in studies.search(ROOT, {"query": query, "top": 5}, farm, **kw)["results"]]


def test_tokens_stem_and_synonyms():
    assert studies.tokens("Reduced reduces REDUCE") == ["reduc"] * 3
    assert studies.tokens("cover-crops") == ["cover", "crop"]
    weights, added = studies.expand("lucerne before corn")
    assert weights["lucern"] == 1 and weights["alfalfa"] == studies.SYNONYM_WEIGHT and "alfalfa" in added
    assert studies.expand("lucerne before corn", synonyms=False)[1] == []


def test_farm_climate_and_texture_classes():
    assert (STORY["aridity"], STORY["thermal"]) == ("humid", "continental")
    assert (LUBBOCK["aridity"], LUBBOCK["thermal"]) == ("semiarid", "warm")
    assert studies.texture_class(40, 22) == ("loam", "medium")
    assert studies.texture_class(10, 30)[0] == "silty clay loam"
    assert studies.texture_class(80, 5) == ("loamy sand", "coarse")
    assert studies.texture_class(float("nan"), 10) == (None, None)


def test_farm_fit_labels_are_explained():
    cards = {c.id: c for c in evidence.load_cards(ROOT)}
    kaspar = studies.fit(cards["kaspar2007-rye-nitrate"], STORY)
    assert kaspar["label"] == "close" and kaspar["region"] == "same" and kaspar["climate"] == "match"
    assert any("Iowa" in r for r in kaspar["reasons"])
    assert studies.fit(cards["mitchell2015-sjv-covercrop-water"], STORY)["label"] == "far"
    assert studies.fit(cards["abdalla2019-cover-meta"], STORY)["label"] == "partial"
    assert studies.fit(cards["nielsen2010-fallow-storage"], AKRON)["label"] == "close"
    assert studies.fit(cards["nielsen2010-fallow-storage"], STORY)["label"] == "far"
    # neighbouring state with a similar climate
    near = studies.fit(cards["trlica2017-rotation-profit"], STORY)
    assert near["region"] == "neighbour" and near["label"] == "close"
    # a stated soil texture that differs blocks "close"
    tarkalson = studies.fit(cards["tarkalson2006-dryland-tillage"], AKRON | {"texture": "loamy sand",
                                                                            "texture_group": "coarse"})
    assert tarkalson["soil"] == "mismatch" and tarkalson["label"] != "close"


def test_search_ranking_snippets_and_filters():
    assert ids("rye nitrate Iowa", STORY)[0] == "kaspar2007-rye-nitrate"
    assert {"davis2012-marsden", "hunt2019-marsden-nutrient"} & set(
        ids("longer rotation Iowa fertilizer herbicide", STORY)[:3])
    assert ids("fallow wheat Colorado water", AKRON)[:2] == ["nielsen2005-legume-fallow", "nielsen2010-fallow-storage"]
    assert "lucerne" not in json.dumps([c.sections for c in evidence.load_cards(ROOT)]).casefold()
    assert ids("lucerne in the rotation, nitrogen for the next corn", STORY, synonyms=False)[:1] != \
        ids("lucerne in the rotation, nitrogen for the next corn", STORY)[:1]
    result = studies.search(ROOT, {"query": "cereal rye nitrate", "top": 3}, STORY)
    first = result["results"][0]
    assert first["cite_as"] == "card:kaspar2007-rye-nitrate" and first["file"] == "evidence/kaspar2007-rye-nitrate.md"
    assert 1 <= len(first["snippets"]) <= studies.SNIPPETS
    assert all(s["section"] in ("Findings", "Conditions and limits") for s in first["snippets"])
    assert "61%" in json.dumps(first["snippets"]) and "## Quotes" not in json.dumps(first)
    assert first["farm_fit"]["label"] == "close" and "read tool" in result["note"]
    assert not result["only_far_evidence"]
    filtered = studies.search(ROOT, {"query": "water", "crops": ["cotton"], "top": 15}, None)["results"]
    assert filtered and all("cotton" in r["citation"].casefold() or "cotton" in
                            {c.id: c for c in evidence.load_cards(ROOT)}[r["id"]].meta["crops"] for r in filtered)
    assert studies.search(ROOT, {"query": "fallow wheat", "top": 3}, STORY)["only_far_evidence"] in (True, False)


@pytest.mark.parametrize("bad", [{"query": ""}, {"query": "x", "top": 0}, {"query": "x", "top": 16},
                                 {"query": "x", "crops": "corn"}, {"query": "x", "web": True}, []])
def test_search_rejects_bad_requests(bad):
    with pytest.raises(ValueError):
        studies.validate(bad)


def test_eval_set_hits_at_3():
    profiles = {"story": STORY, "akron": AKRON, "lubbock": LUBBOCK,
                "fresno": {"state": "CA", "irrigated": True, "annual_rain_mm": 231, "texture": "clay loam",
                           "texture_group": "fine", **studies.farm_climate_class(231, 1693, 17.5, 297, .01)},
                "york": {"state": "NE", "irrigated": True, "annual_rain_mm": 721, "texture": "silty clay loam",
                         "texture_group": "fine", **studies.farm_climate_class(721, 1220, 10.4, 180, .35)}}
    result = studies.evaluate(ROOT, profiles)
    assert len(result["queries"]) >= 8
    full = result["hits_at_k"]["bm25+synonyms+fit"]
    assert full["rate"] >= .9 and full["hits"] >= result["hits_at_k"]["bm25"]["hits"]
    assert full["close_fit_in_top_k"] >= result["hits_at_k"]["bm25"]["close_fit_in_top_k"]


def test_note_wording_grounds_exclusion():
    assert checks.text_limits("My landlord does not allow alfalfa.")[0] == {"exclude_crops": ["alfalfa"]}
    assert checks.text_limits("The landlord forbids alfalfa")[0] == {"exclude_crops": ["alfalfa"]}
    assert checks.text_limits("I don't permit corn on the hill")[0] == {"exclude_crops": ["corn"]}
    assert checks.text_limits("We allow alfalfa")[0] == {}


def test_list_limit_grounded_by_several_quotes():
    form = {"limits": {}, "free_text": "Avoid oats.\n\nApproved notes:\n- My landlord does not allow alfalfa."}
    one = {"limits": {"exclude_crops": ["alfalfa"]},
           "limit_reasons": {"exclude_crops": [{"quote": "My landlord does not allow alfalfa."}]}}
    assert checks.ground_limits(one, form) == []
    two = {"limits": {"exclude_crops": ["oats", "alfalfa"]},
           "limit_reasons": {"exclude_crops": [{"quote": "Avoid oats"}, "My landlord does not allow alfalfa"]}}
    assert checks.ground_limits(two, form) == []
    more = {"limits": {"exclude_crops": ["oats", "alfalfa", "corn"]}, "limit_reasons": two["limit_reasons"]}
    assert checks.ground_limits(more, form) == [("exclude_crops: quoted words do not unambiguously specify this "
                                                 "limit; ask farmer to use the form")]
    invented = {"limits": {"exclude_crops": ["alfalfa"]},
                "limit_reasons": {"exclude_crops": [{"quote": "My landlord does not allow alfalfa."},
                                                    {"quote": "never alfalfa"}]}}
    assert checks.ground_limits(invented, form) == ["exclude_crops: missing exact quote from free_text"]
    scalar = {"limits": {"no_new_equipment": True}, "limit_reasons": {"no_new_equipment": [{"quote": "Avoid oats"}]}}
    assert checks.ground_limits(scalar, form) == ["no_new_equipment: missing exact quote from free_text"]


@pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE", "").strip().lower() == "cpu"
         or torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Select FIELDSHIFT_DEVICE=cpu or provide MPS/CUDA")
def test_studies_route(tmp_path):
    import http.client
    fid = farms.add({"adm_id": "US-19-169"}, tmp_path)
    runtime = Runtime(tmp_path)
    server = make_server("t-token", "127.0.0.1", 0, runtime, tmp_path / "log.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(body):
        c = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=120)
        c.request("POST", f"/farm/{fid}/studies", json.dumps(body),
                  {"Content-Type": "application/json", "Authorization": "Bearer t-token"})
        r = c.getresponse()
        out = r.status, json.loads(r.read())
        c.close()
        return out
    try:
        status, data = post({"query": "Would cereal rye cut my nitrate losses?", "top": 3})
        assert status == 200 and data["results"][0]["id"] == "kaspar2007-rye-nitrate"
        assert data["farm"]["state"] == "IA" and data["farm"]["aridity"] == "humid"
        assert data["results"][0]["farm_fit"]["label"] == "close"
        status, data = post({"query": "x", "top": 99})
        assert status == 422 and "top" in data["message"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        runtime.cache.clear()
