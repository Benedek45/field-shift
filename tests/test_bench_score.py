import json

import pytest

from fieldshift.bench.grounding import (
    call_requests,
    classify_text,
    numeric_grounding,
    report_sentences,
)
from fieldshift.bench.score import (
    analyse,
    call_outputs,
    numeric_faithfulness,
    partial_report,
    rounded_match,
    rubric_pass,
    summarise,
)


def evidence(claim, value=940, path="rows.0.metrics.margin.median", source="sim:1", scale=1):
    return {"evidence": [{"claim": claim, "source": source,
                          "numeric_values": [{"path": path, "value": value, "scale": scale}]}]}


def test_numeric_paths_rounding_and_wrong_metric():
    outputs = {1: {"rows": [{"metrics": {"margin": {"median": 940.24}, "water": {"median": 500}}}]}}
    assert not numeric_faithfulness(evidence("Margin $940/ha."), outputs)
    assert numeric_faithfulness(evidence("Margin $950/ha.", value=950), outputs)
    assert numeric_faithfulness(evidence("Margin $940/ha.", path="rows.0.metrics.water.median"), outputs)
    assert numeric_faithfulness(evidence("Margin $940/ha and water 500 mm."), outputs)
    assert numeric_faithfulness(evidence("Margin $940/ha.", source="sim:2"), outputs)
    assert not rounded_match(True, 1)
    assert not rounded_match(1, float("nan"))


def test_numeric_missing_paths_fail_closed_and_percentage():
    assert numeric_faithfulness({"evidence": [{"claim": "Income 940", "source": "sim:1"}]}, {})
    assert not numeric_faithfulness(evidence("Loss 23.4%.", 23.4, "loss", scale=100), {1: {"loss": .234}})


def test_rubrics_and_control():
    missing = {"kind": "missing_data", "terms": ["buyer", "eligibility", "approval"]}
    d = {"confidence": "low", "information_gaps": ["Buyer approval is missing"]}
    assert rubric_pass(d, missing, True)
    assert not rubric_pass(d, missing, False)
    d["confidence"] = "high"
    assert not rubric_pass(d, missing, True)
    unsupported = {"kind": "infeasible", "terms": ["millet"], "unsupported_crops": ["millet"]}
    d = {"information_gaps": ["Millet is not supported by this tool"], "recommended": {"rotation": [{"crop": "corn"}]}}
    assert rubric_pass(d, unsupported, True)
    d["evidence"] = [{"source": "sim:1", "claim": "Millet returns $600/ha"}]
    assert not rubric_pass(d, unsupported, True)
    d["evidence"] = []
    d["alternatives"] = [{"rotation": [{"crop": "millet"}], "metrics": {"margin": 600}}]
    assert not rubric_pass(d, unsupported, True)


def test_transcript_complete_and_receipt_fallback(tmp_path):
    (tmp_path / "work").mkdir()
    payload = {"rows": [{"score": 12.34}]}
    output = "Simulator call sim:1 (cite these results as sim:1)\n" + json.dumps(payload)
    transcript = {"parts": [{"type": "tool", "output": output, "output_chars": len(output)}]}
    (tmp_path / "transcript.jsonl").write_text(json.dumps(transcript) + "\n")
    calls, problems, fallback = call_outputs(tmp_path)
    assert calls == {1: payload}
    assert not problems and not fallback
    transcript["parts"][0]["output"] = output[:30]
    (tmp_path / "transcript.jsonl").write_text(json.dumps(transcript) + "\n")
    (tmp_path / "work/.sim-response-1.json").write_text(json.dumps(payload))
    calls, _, fallback = call_outputs(tmp_path)
    assert calls == {1: payload}
    assert fallback == [1]


def test_pass3_is_all_three_not_majority():
    rows = [{"pair": "p", "rep": rep, "planted": True, "pass": p, "correct": p, "trap": not p, "rubric": False}
            for rep, p in enumerate((True, True, False))]
    result = summarise(rows)
    assert result["pass^3"] == 0
    assert result["mean_accuracy"] == pytest.approx(2 / 3)
    assert result["trap_rate"] == pytest.approx(1 / 3)
    assert summarise(rows[:2])["pass^3"] is None


def test_paired_flip_requires_both_twins(tmp_path):
    pair = {"id": "abc", "family": "hidden_limit", "key": [{"crop": "soybean"}],
            "naive": [{"crop": "corn"}], "rubric": None}
    for suffix, crop in (("a", "corn"), ("b", "soybean")):
        run = tmp_path / suffix
        (run / "work/evidence").mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"case": "abc" + suffix, "model": "test", "rep": 0,
                                                  "status": "completed", "exit_code": 0,
                                                  "trace": {"simulate_calls": 0}}))
        (run / "work/DECISION.json").write_text(json.dumps({"recommended": {"rotation": [{"crop": crop}]}}))
    report = analyse(tmp_path, {"pairs": [pair]})
    assert report["test"]["twin_flip_rate"] == 1
    assert report["test"]["pass^3"] is None


def test_missing_cells_are_reported_not_scored(tmp_path):
    pair = {"id": "abc", "family": "hidden_limit", "key": [{"crop": "soybean"}],
            "naive": [{"crop": "corn"}], "rubric": None}
    (tmp_path / "contract.json").write_text(json.dumps({"models": ["test"], "pairs": ["abc"], "reps": 3}))
    report = analyse(tmp_path, {"pairs": [pair]})["test"]
    assert len(report["missing_episodes"]) == 6
    assert report["mean_accuracy"] is None
    assert report["episodes"] == 0
    assert report["pass^3"] is None
    assert report["pass^3_denominator"] == 0


def test_negative_weather_wording_and_boolean_annotations():
    d = {"evidence": [{"claim": "With 20% less rain it remains tied.",
                       "source": "sim:1", "numeric_values": [{"path": "rain", "value": -20},
                                                          {"path": "tied", "value": True}]}]}
    assert not numeric_faithfulness(d, {1: {"rain": -20, "tied": True}})
    d["evidence"][0]["claim"] = "With 20% more rain it remains tied."
    assert numeric_faithfulness(d, {1: {"rain": -20, "tied": True}})
    d["evidence"][0]["claim"] = "With 20% less rain it remains tied."
    assert numeric_faithfulness(d, {1: {"rain": -20, "tied": False}})


def test_string_warning_is_a_valid_exact_source_path():
    d = evidence("Overlap in 77% of years.", 77, "pruned.0")
    assert not numeric_faithfulness(d, {1: {"pruned": ["harvest overlaps sowing in 77% of years"]}})
    assert numeric_faithfulness(d, {1: {"pruned": ["harvest overlaps sowing in 70% of years"]}})


def test_no_double_rounding_or_percentage_guessing():
    assert numeric_faithfulness(evidence("Water 1.70 mm", 1.7, "water"), {1: {"water": 1.73}})
    assert not numeric_faithfulness(evidence("Water 1.7 mm", 1.7, "water"), {1: {"water": 1.73}})
    assert numeric_faithfulness(evidence("Loss 23.4%", 23.4, "loss"), {1: {"loss": .234}})


def test_bad_receipt_does_not_crash_or_replace_transcript(tmp_path):
    (tmp_path / "work").mkdir()
    output = 'Simulator call sim:1\n{"score":42,"latency_seconds":1}'
    (tmp_path / "transcript.jsonl").write_text(json.dumps(
        {"parts": [{"type": "tool", "name": "shell", "output": output}]}) + "\n")
    receipt = tmp_path / "work/.sim-response-1.json"
    receipt.write_text('{"score":99}\nstale tail')
    calls, problems, _ = call_outputs(tmp_path)
    assert calls[1]["score"] == 42
    assert any("malformed receipt" in p for p in problems)
    receipt.write_text('{"score":42,"latency_seconds":2}')
    _, problems, _ = call_outputs(tmp_path)
    assert not problems


def test_conflicting_label_cannot_be_recovered_by_matching_claim(tmp_path):
    (tmp_path / "work").mkdir()
    parts = [{"type": "tool", "name": "shell", "output": f'Simulator call sim:1\n{{"score":{n}}}'}
             for n in (42, 99)]
    (tmp_path / "transcript.jsonl").write_text(json.dumps({"parts": parts}) + "\n")
    (tmp_path / "work/.sim-response-1.json").write_text('{"score":99}')
    calls, problems, _ = call_outputs(tmp_path)
    assert 1 not in calls
    assert any("conflicting" in p for p in problems)


def test_partial_excludes_archive_infra_and_other_repeats(tmp_path):
    pair = {"id": "abc", "family": "hidden_limit", "key": [{"crop": "soybean"}],
            "naive": [{"crop": "corn"}], "rubric": None}
    (tmp_path / "contract.json").write_text(json.dumps({"models": ["test"], "pairs": ["abc"], "reps": 3}))
    entries = [("a", "a", 0, "completed", "corn"), ("b", "b", 0, "completed", "soybean"),
               ("_infra_provider_limit_old", "a", 0, "completed", "soybean"),
               ("killed", "b", 0, "infra", "corn"), ("repeat1", "a", 1, "completed", "soybean")]
    for name, suffix, rep, status, crop in entries:
        run = tmp_path / name
        (run / "work/evidence").mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"case": "abc" + suffix, "model": "test", "rep": rep,
                                                  "status": status, "exit_code": 0, "seconds": 100,
                                                  "trace": {"simulate_calls": 0, "tokens": {"input": 50}}}))
        (run / "work/DECISION.json").write_text(json.dumps({"recommended": {"rotation": [{"crop": crop}]}}))
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete/meta.json").write_text(json.dumps(
        {"case": "abca", "model": "test", "rep": 0, "status": "completed"}))
    report = partial_report(tmp_path, {"pairs": [pair]})["models"]["test"]
    assert report["overall"]["n"] == 2
    assert report["overall"]["accuracy"] == 1
    assert report["overall"]["twin_pairs_n"] == 1
    assert report["overall"]["twin_pair_accuracy"] == 1
    assert report["overall"]["pass^3"] is None


SCOREBOARD = {
    "n_after_limits": 457, "n_cycles_scored": 458,
    "existing": {"rank": 311, "metrics": {"margin_usd_ha_yr": {"median": 224.035},
                                          "days_without_living_cover_yr": {"median": 178.625}},
                 "violates_limits": ["soybean → winter_wheat: harvest overlaps autumn sowing in 77% of years"]},
    "rows": [{"rank": 1, "metrics": {"margin_usd_ha_yr": {"median": 195.317, "p10": 26.76},
                                     "nitrate_leached_kg_n_ha_yr": {"median": 27.4},
                                     "days_without_living_cover_yr": {"median": 25.625},
                                     "cover_establishment_fraction": {"median": .75}},
              "tied_with_best": True},
             {"rank": 2, "metrics": {"margin_usd_ha_yr": {"median": 134.2}}, "tied_with_best": False}],
}


def classes(claim, outputs=None, requests=None, annotations=()):
    rows = classify_text(claim, [1], {1: SCOREBOARD} if outputs is None else outputs, requests or {},
                         annotations=annotations)
    return {r["token"]: r["class"] for r in rows}


def test_grounding_found_including_strings_shares_and_rounding():
    assert classes("Median return 195 USD/ha/yr against 224 USD/ha/yr.") == {"195": "found", "224": "found"}
    assert classes("Soybean is harvested after wheat sowing in 77% of years.") == {"77": "found"}
    assert classes("Rye establishes in 75% of sowings.") == {"75": "found"}
    assert classes("Bare for 25.6 days a year.") == {"25.6": "found"}
    assert classes("Bare for 25.7 days a year.") == {"25.7": "unfounded"}


def test_grounding_unit_aware_matching_catches_cross_metric_coincidence():
    # 27 exists only as nitrate (kg N); a 27 USD claim may not borrow it, but P10 26.76 USD can support it.
    assert classes("Nitrate leaching is 27 kg N/ha/yr.") == {"27": "found"}
    assert classes("It earns 27.4 USD/ha/yr.") == {"27.4": "unfounded"}


def test_grounding_request_class_and_sign_wording():
    requests = {1: [{"command": "scoreboard", "request": {"rain_change_pct": -20, "temp_change_c": 1.5},
                     "economics": []}]}
    assert classes("Under 20% less rain and +1.5 °C it drops to rank 2.", requests=requests) == {
        "20": "request", "+1.5": "request", "2": "structural"}
    assert classes("Under 20% more rain the order holds.", requests=requests) == {"20": "unfounded"}
    assert classes("Under −20% rain the order holds.", requests=requests) == {"−20": "request"}


def test_grounding_negative_money_and_ranges():
    outputs = {1: {"rows": [{"metrics": {"margin_usd_ha_yr": {"median": -6.2, "p10": 912.3, "p90": 1249.6}}}]}}
    assert classes("Median return is -$6/ha/yr.", outputs) == {"6": "found"}
    assert classes("P10–P90 $912–$1,250 per ha.", outputs) == {"912": "found", "1,250": "found"}
    assert classes("Median return is about –6 USD/ha.", outputs) == {"6": "found"}


def test_grounding_derived_needs_both_inputs_in_claim():
    assert classes("153 days more living cover (25.6 vs 178.6 days).") == {
        "153": "derived", "25.6": "found", "178.6": "found"}
    assert classes("About 29 USD/ha/yr below the existing cycle.") == {"29": "unfounded"}
    assert classes("Return falls from 224 to 195 USD/ha/yr, 13% lower.")["13"] == "derived"


def test_grounding_structural_ranks_counts_and_cycle_years():
    assert classes("It ranks 1 of 457 feasible cycles; the existing cycle ranks 311.") == {
        "1": "structural", "457": "structural", "311": "structural"}
    assert classes("A 4-year cycle with 2 years of soybean.") == {"4": "structural", "2": "structural"}
    assert classes("It ranks 9 of 457.")["9"] == "unfounded"


def test_grounding_annotations_are_optional_and_rescue_only():
    path = {"path": "rows.0.metrics.margin_usd_ha_yr.median", "value": 195}
    rows = classify_text("Return 195 USD/ha/yr.", [1], {1: SCOREBOARD}, {}, annotations=[path])
    assert rows[0]["class"] == "found" and rows[0]["via"] == "annotation"
    typo = {"path": "rows.0.margin_usd_ha_yr.median", "value": 195}
    rows = classify_text("Return 195 USD/ha/yr.", [1], {1: SCOREBOARD}, {}, annotations=[typo])
    assert rows[0]["class"] == "found" and rows[0]["annotation_error"]
    wrong = {"path": "rows.0.metrics.margin_usd_ha_yr.median", "value": 240}
    rows = classify_text("Return 240 USD/ha/yr.", [1], {1: SCOREBOARD}, {}, annotations=[wrong])
    assert rows[0]["class"] == "unfounded"


def test_grounding_detects_planted_fabrication(tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work/.sim-response-1.json").write_text(json.dumps(SCOREBOARD))
    outputs, _, _ = call_outputs(tmp_path)
    honest = {"evidence": [{"source": "sim:1", "claim": "Return 195 vs 224 USD/ha/yr; 25.6 vs 178.6 bare days."}]}
    result = numeric_grounding(tmp_path, honest, outputs, {})
    assert result["grounded_pass"] and result["counts"]["found"] == 4
    planted = {"evidence": [{"source": "sim:1", "claim": "Return 241 vs 224 USD/ha/yr; 25.6 vs 178.6 bare days."}]}
    result = numeric_grounding(tmp_path, planted, outputs, {})
    assert not result["grounded_pass"]
    assert [(r["token"], r["reason"]) for r in result["unfounded"]] == [("241", "absent_from_run")]
    (tmp_path / "work/REPORT.md").write_text("You would earn 241 USD/ha a year. [sim:1]\n")
    result = numeric_grounding(tmp_path, honest, outputs, {})
    assert result["grounded_pass"] and result["report_counts"]["unfounded"] == 1
    unavailable = {"evidence": [{"source": "sim:2", "claim": "Return 195 USD/ha/yr."}]}
    assert numeric_grounding(tmp_path, unavailable, outputs, {})["counts"]["unverifiable"] == 1


def test_call_requests_follows_client_counter(tmp_path):
    def shell(command, output, status="completed"):
        return {"type": "tool", "name": "shell", "status": status, "input": {"command": command}, "output": output}
    parts = [
        shell("./farmsim options", "{}"),
        shell("./farmsim scoreboard '{\"top\": 5}'", "", status="error"),  # never ran: no label consumed
        shell("./farmsim scoreboard '{\"top\": 5}'", '..."rank": 3}\n[showing the last 50 KB]'),  # header lost
        shell("./farmsim economics '{\"crop_price_usd_t\": {\"hay\": 3000}}'", "Farmer economics applied."),
        shell("./farmsim scoreboard '{\"rain_change_pct\": -20}'", 'Simulator call sim:2\n{"rows": []}'),
        shell("./farmsim economics-reset", "Back to the farm's original economics."),
        shell("./farmsim simulate '{\"rotations\": []}'", "{}"),
        shell("./farmsim simulate '{\"rotations\": [], \"temp_change_c\": 2}'", 'Simulator call sim:3\n{}'),
    ]
    (tmp_path / "transcript.jsonl").write_text(json.dumps({"parts": parts}) + "\n")
    requests, problems = call_requests(tmp_path)
    assert sorted(requests) == [1, 2, 3]
    assert requests[1][0]["request"] == {"top": 5}
    assert requests[2][0]["economics"] == [{"crop_price_usd_t": {"hay": 3000}}]
    assert [r["request"] for r in requests[3]] == [{"rotations": []}, {"rotations": [], "temp_change_c": 2}]
    assert requests[3][0]["economics"] == []
    assert problems == ["sim:3: label after sim:3"]
    rows = classify_text("At the quoted $3,000/t hay price hay ranks first.", [2], {2: {"rows": []}}, requests)
    assert rows[0]["class"] == "request"


def test_report_sentences_keep_trailing_citation():
    text = "First claim [weights; sim:3] Second sentence costs $61.9/ha. [sim:2]\n1. List item 5 kg (sim:1)."
    sentences = list(report_sentences(text))
    assert any("$61.9" in s and "sim:2" in s for s in sentences)
    rows = classify_text(sentences[-1], [1], {1: {"x_kg": 5}}, {})
    assert [r["token"] for r in rows] == ["5"]
