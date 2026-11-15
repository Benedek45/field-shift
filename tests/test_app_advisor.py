"""Questionnaire app: price table and form building, the background worker with a scripted agent
(real private simulator service, no container, no provider) and the Streamlit page (AppTest)."""

import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fieldshift.agent import chat, checks, form, pilot
from fieldshift.app import advisor_core as core
from fieldshift.app import advisor_worker as worker
from fieldshift.sim.economics import ACRES_HA

STORY = "US-19-169"
STORY_TEXT = "Story County, IA · US-19-169"
PAGE = str(Path(__file__).resolve().parents[1] / "src/fieldshift/app/advisor.py")
CARD = "seifert2017-yield-penalty"

def test_app_worker_import_does_not_load_proj():
    # A fresh interpreter avoids unrelated tests' optional geospatial imports.
    subprocess.run(
        [sys.executable, "-c",
         ("import sys; from fieldshift.app import advisor, advisor_core, advisor_worker; "
          "assert 'pyproj' not in sys.modules; assert 'geopandas' not in sys.modules")],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def story():
    return core.county_defaults(STORY, False)


def answers(**changes):
    values, risk = core.preset_values("steady")
    q = {"irrigated": False, "rotation_mode": "unknown", "rotation": [], "own_soil": {}, "soil": {},
         "preset": "steady", "weights": values, "risk": risk, "limits": {}, "price_changes": {},
         "general_changes": {}, "weather_view": "record", "free_text": ""}
    return q | changes


def test_county_defaults_and_price_table(story):
    assert story["label"] == "Story County, IA" and story["rotation"] and "basis" not in story["rotation_basis"][:0]
    assert {"corn", "soybean"} <= set(story["main_crops"])
    assert 1 < story["soil"]["awc_in_ft"] < 3  # 140 mm/m is about 1.7 in/ft
    rows = core.price_rows(story)
    corn = next(r for r in rows if r["key"] == "corn|price")
    assert corn["unit"] == "$/bu" and corn["form_unit"] == "usd/bu" and 3 < corn["default"] < 8
    assert corn["source"] and corn["confidence"] in ("low", "medium", "high")
    assert not any(r["item"] in ("repeat_crop", "repeat_crop_fixed") for r in rows)
    assert not any(r["item"] == "irrigation_fixed" for r in rows)  # rainfed farm, zero system cost
    hay = next(r for r in rows if r["key"] == "hay|price")
    assert hay["form_unit"] == "usd/ton"
    general = {r["key"]: r for r in core.general_rows(story)}
    assert general["fertilizer_n_usd_lb"]["unit"] == "$/lb N" and general["cover_seed_usd_ha"]["default"] > 0

    # Edits: a changed cell becomes a farmer value; set back to the default it is dropped again.
    i = rows.index(corn)
    changes = core.apply_edits(rows, {}, {i: {"Value": 4.6}})
    assert changes == {"corn|price": 4.6}
    frame = core.table_frame(rows, changes)
    assert frame.loc[i, "Source"] == "farmer" and frame.loc[i, "Value"] == 4.6 and frame.loc[i, "Default"] == corn["default"]
    assert frame.loc[i + 1, "Source"] != "farmer"
    assert core.apply_edits(rows, changes, {i: {"Value": corn["default"]}}) == {}
    seed = next(r for r in rows if r["key"] == "corn|seed")
    changes = core.apply_edits(rows, changes, {rows.index(seed): {"Value": 120}})
    block = core.economics_block(rows, changes, core.general_rows(story), {"fertilizer_n_usd_lb": .5,
                                                                         "cover_seed_usd_ha": 30})
    assert block == {"crop_prices": {"corn": {"value": 4.6, "unit": "usd/bu"}},
                     "crop_cost_parts_usd_acre": {"corn": {"seed": 120}},
                     "fertilizer_n_usd_lb": .5, "cover_seed_usd_ha": pytest.approx(30 * ACRES_HA)}
    si = form.economics(block)
    assert si["crop_price_usd_t"]["corn"] == pytest.approx(4.6 * ACRES_HA / 0.06277, rel=1e-3)  # 1 bu/ac = 0.06277 t/ha
    assert si["crop_cost_usd_ha"]["corn"]["seed"] == pytest.approx(120 * ACRES_HA)
    assert si["fertilizer_n_usd_kg"] == pytest.approx(.5 / .45359237)
    # Defaults shown in the table round-trip to the simulator's SI values.
    assert corn["default"] * core.price_unit("corn")[2] == pytest.approx(
        next(c for c in story["options"]["economics"]["crops"] if c["crop"] == "corn")["price"]["value"], rel=1e-3)


def test_build_form_and_error_sections(story):
    q = answers(rotation_mode="own", rotation=[{"crop": "corn", "winter": "cereal_rye_cover"}, {"crop": "soybean"}],
                own_soil={"slope_pct": True, "awc_in_ft": True}, soil={"slope_pct": 4.0, "awc_in_ft": 2.0},
                limits={"income_floor": True, "income_floor_pct": 80, "no_new_machinery": True,
                        "exclude_crops": ["alfalfa"], "max_nitrate_kg_n_ha_yr": 40.},
                price_changes={"corn|price": 4.6}, free_text="  Keep it simple. ")
    public = core.build_form(q, story)
    assert public["location"] == {"adm_id": STORY, "irrigated": False}
    assert public["current_rotation"] == [{"crop": "corn", "winter": "cereal_rye_cover"}, {"crop": "soybean"}]
    assert public["field_facts"] == {"slope_pct": 4.0, "water_holding_capacity": {"value": 2.0, "unit": "in/ft"}}
    assert public["limits"]["minimum_share_current_income"] == {"value": 80.0, "unit": "%"}
    assert public["free_text"] == "Keep it simple."
    normal = form.validate(public)
    assert normal["weights"] == form.validate({"location": {"adm_id": STORY, "irrigated": False},
                                               "priorities": {"preset": "steady"}})["weights"]
    assert normal["limits"]["no_new_equipment"] and normal["limits"]["min_margin_share_of_existing"] == .8
    assert normal["farm"]["overrides"]["awc_mm_per_m"] == pytest.approx(2.0 * 25.4 / .3048)
    assert normal["limits"]["max_nitrate_kg_n_ha_yr"] == pytest.approx(40 * .45359237 * ACRES_HA)
    # Confirming the county estimate states it; "not sure" leaves the inference to the simulator.
    assert core.build_form(answers(rotation_mode="inferred"), story)["current_rotation"] == story["rotation"]
    assert "current_rotation" not in core.build_form(answers(), story)
    _, normal, errors = core.validate(answers(rotation_mode="own", rotation=[{"crop": "corn"}, {"crop": "winter_wheat"}]),
                                      story)
    assert normal is None and "infeasible" in errors["rotation"]
    _, _, errors = core.validate(answers(limits={"include_crops": ["oats"], "exclude_crops": ["oats"]}), story,
                                 feasibility=False)
    assert list(errors) == ["limits"]
    _, _, errors = core.validate(answers(weather_view="tomorrow"), story, feasibility=False)
    assert list(errors) == ["weather"]
    _, _, errors = core.validate(answers(irrigated=True, well_known=True, well_in=-3), story, feasibility=False)
    assert list(errors) == ["farm"]
    assert core.section_for("economics.crop_prices.corn must be between 0 and 1e9") == "prices"


def test_job_state_and_busy(tmp_path):
    session = core.new_session("Story County, IA", "fake/model", root=tmp_path)
    assert session.name.endswith("-story-ia") and (session.stat().st_mode & 0o777) == 0o700
    launched = []
    core.start_job(session, "solve", launcher=lambda s, k, **kw: launched.append(k))
    assert launched == ["solve"] and core.job_state(session)["running"]  # starting, inside the grace period
    with pytest.raises(core.JobBusy):
        core.start_job(session, "ask", launcher=lambda s, k, **kw: launched.append(k))
    with core.job_lock(session):
        assert core.job_state(session)["running"]
        with pytest.raises(core.JobBusy), core.job_lock(session):
            pass
    core.write_json(session / "job.json", {"kind": "solve", "state": "running", "started": 0})
    state = core.job_state(session)  # lock released, worker gone: a readable failure, not "running" forever
    assert not state["running"] and state["state"] == "failed" and "stopped unexpectedly" in state["error"]
    assert worker.readable(ValueError("bad corn price")).startswith("The questionnaire values were rejected")
    assert "Traceback" not in worker.readable(RuntimeError("x"))


# -- the worker with a scripted agent ----------------------------------------------------------

def call(port, token, fid, route, body=None):
    try:
        return json.loads(pilot.http(f"http://127.0.0.1:{port}/farm/{fid}/{route}", token, body))
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, **json.loads(exc.read().decode())}


class ScriptedAgent(chat.OpenCodeAgent):
    """The real run-folder preparation; the container turn is replaced by scripted service calls."""

    def __init__(self):
        super().__init__("opencode-go/fake-model", 60, prompt=worker.SOLVE_PROMPT)
        self.messages, self.refused = [], []

    def sim(self, run_dir, where, route, body):
        work = run_dir / "work"
        n = int((work / ".farmsim_calls").read_text()) + 1 if (work / ".farmsim_calls").exists() else 1
        (work / ".farmsim_calls").write_text(str(n))
        data = call(where["port"], where["token"], where["farm_id"], route, body)
        (work / f".sim-response-{n}.json").write_text(json.dumps(data))
        return n, data

    def write_decision(self, run_dir, normal, n, board, extra=()):
        choice = checks.choose(board)
        others = [r for r in board["rows"] if checks.canonical(r["rotation"]) != checks.canonical(choice["rotation"])]
        decision = {"existing": {"rotation": board["existing"]["rotation"]}, "weights": normal["weights"],
                    "weight_reasons": dict.fromkeys(normal["weights"], "form"), "limits": normal["limits"],
                    "limit_reasons": dict.fromkeys(normal["limits"], "form"), "scoreboard": f"sim:{n}",
                    "recommended": {"rotation": choice["rotation"], "rank": choice["rank"],
                                    "from_tied_set": bool(choice.get("tied_with_best")), "override_reason": None},
                    "alternatives": [{"rotation": r["rotation"], "aim": "next in the ranking"} for r in others[:2]],
                    "why": f"Top of the ranking under your answers [sim:{n}].",
                    "evidence": [{"claim": "Rotation effects on yield", "source": f"card:{CARD}"}, *extra],
                    "information_gaps": ["Field soil tests"], "risks_for_local_review": ["Local prices"],
                    "benefits": ["Familiar crops"], "confidence": "medium"}
        (run_dir / "work/DECISION.json").write_text(json.dumps(decision))
        (run_dir / "work/REPORT.md").write_text(f"Keep {choice['rotation'][0]['crop']} first [sim:{n}]. "
                                                "This supports, does not replace your judgment.")

    def turn(self, run_dir, message, *, session_id, keep_session, **where):
        self.messages.append({"message": message, "session_id": session_id, "keep_session": keep_session})
        normal = json.loads((run_dir.parent / "advice/form.json").read_text())
        agent_file = (run_dir / "cfg/opencode/agents/farm-advisor.md").read_text()
        if "## Farmer's question" not in message:  # the solution episode
            assert "Product form contract" in message and "Choose the crop rotation" in message
            assert "Explains the farmer form" in agent_file
            n, board = self.sim(run_dir, where, "scoreboard",
                                {"weights": normal["weights"], "limits": normal["limits"], "top": 15})
            self.write_decision(run_dir, normal, n, board)
            reply = "Done."
        else:
            assert "Answers one farmer's questions" in agent_file  # farm-qa.md swapped in for question time
            question = message.rsplit("## Farmer's question", 1)[1].strip()
            if "dry" in question:
                self.refused.append(call(where["port"], where["token"], where["farm_id"], "form",
                                         {"economics": {"crop_prices": {"corn": {"value": 9, "unit": "usd/bu"}}}}))
                self.refused.append(call(where["port"], where["token"], where["farm_id"], "economics",
                                         {"crop_price_usd_t": {"corn": 400}}))
                rotation = json.loads((run_dir / "work/DECISION.json").read_text())["recommended"]["rotation"]
                n, data = self.sim(run_dir, where, "simulate", {"rotations": [rotation], "rain_change_pct": -30})
                assert "results" in data, data
                reply = f"With 30% less rain in every past year the typical return falls [sim:{n}]."
            elif "alfalfa" in question:
                n, data = self.sim(run_dir, where, "simulate", {"rotations": [[{"crop": "alfalfa"}]]})
                m, board = self.sim(run_dir, where, "scoreboard",
                                    {"weights": normal["weights"], "limits": normal["limits"], "top": 15})
                self.write_decision(run_dir, normal, m, board,
                                    extra=[{"claim": "Alfalfa compared", "source": f"sim:{n}"}])
                reply = f"Alfalfa earns less here [sim:{n}]; the recommendation stands [sim:{m}]."
            elif "earn" in question:
                m = json.loads((run_dir / "work/.sim-response-1.json").read_text())["existing"]["metrics"]
                reply = (f"A typical year returns ${m['margin_usd_acre_yr']['median']} per acre and the crops use "
                         f"{m['field_water_use_in_yr']['median']} inches of water [sim:1].")
            elif "wrong" in question:
                reply = "A typical year returns $99999 per acre and uses 193 inches of water [sim:1]."
            else:
                reply = "Nitrate leaching is nitrate washing out below the roots [own-knowledge]."
        (run_dir / "data/opencode").mkdir(parents=True, exist_ok=True)
        (run_dir / "data/opencode/opencode.db").write_text("conversation store stand-in")
        (run_dir / "events.jsonl").write_text(json.dumps({"type": "text", "sessionID": "ses_fake",
                                                          "part": {"text": reply}}) + "\n")
        (run_dir / "transcript.jsonl").write_text(json.dumps({"summary": {"messages": len(self.messages)}}) + "\n")
        return {"status": "completed", "exit_code": 0, "session_id": "ses_fake", "reply": reply, "agent_seconds": 0.1,
                "trace": {"tokens": {"input": 1, "output": 1}, "steps": 1,
                          "read_paths": [f"/work/evidence/{CARD}.md"]},
                "credential_scrub": chat.scrub_credentials(run_dir, [], keep_session=keep_session)}


class Phases:
    def __init__(self):
        self.seen = []

    def phase(self, name):
        self.seen.append(name)


@pytest.fixture(scope="module")
def solved(tmp_path_factory, story):
    root = tmp_path_factory.mktemp("app")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
        session = core.new_session("Story County, IA", "opencode-go/fake-model", root=root)
        q = answers(limits={"no_new_machinery": True}, price_changes={"corn|price": 4.6})
        public, _, errors = core.validate(q, story)
        assert errors is None
        core.write_json(session / "form.json", public)
        agent, job = ScriptedAgent(), Phases()
        receipt = worker.solve(session, job, agent=agent)
        solution = json.loads((session / "latest.json").read_text())
        asked = []
        for question in ("What if next summer is dry?", "Why not alfalfa?"):
            n = len(core.qa_rows(session)) + 1
            core.write_json(session / "qa/pending.json", {"n": n, "question": question})
            asked.append(worker.ask(session, job, agent=agent))
    return {"session": session, "agent": agent, "receipt": receipt, "job": job, "asked": asked, "root": root,
            "solution": solution}


def test_worker_solution_keeps_session_and_locks_settings(solved):
    session, receipt, agent = solved["session"], solved["receipt"], solved["agent"]
    assert solved["job"].seen[:4] == ["ranking", "agent", "checks", "comparison"]
    assert receipt["checks"]["all_passed"], receipt["checks"]
    assert receipt["model_used"] == "opencode-go/fake-model"
    state = json.loads((session / "session.json").read_text())
    assert state["status"] == "solved" and state["qa_available"] and state["farm_id"]
    assert state["continuity"] == "session" and state["opencode_session_id"] == "ses_fake"
    for name in ("form.json", "advice/REPORT.md", "advice/DECISION.json", "advice/run.json", "advice/scoreboard.json",
                 "advice/transcript.jsonl", "advice/FARM.md", "latest.json", "qa.jsonl", "agent/work/DECISION.json"):
        assert (session / name).exists(), name
    normal = json.loads((session / "advice/form.json").read_text())
    assert normal["farm"]["economics"]["crop_price_usd_t"]["corn"] == pytest.approx(4.6 * ACRES_HA / 0.06277, rel=1e-3)
    latest = solved["solution"]
    roles = [r["role"] for r in latest["comparison"]["rows"]]
    assert latest["source"] == "solution" and roles[0].startswith("Recommended")
    assert sum(r.endswith(("Current", "= current")) for r in roles) == 1
    frame = core.comparison_frame(latest["comparison"])
    assert "Operating return typical ($/acre/yr)" in frame and "Nitrate leaching bad year (lb N/acre/yr)" in frame
    # The service refused both ways of changing the farmer's settings, by code.
    assert [r["http_status"] for r in agent.refused] == [422, 422]
    assert "questionnaire" in agent.refused[0]["message"]


def test_worker_questions_continue_session_and_recheck(solved):
    session, agent = solved["session"], solved["agent"]
    first, second = solved["asked"]
    rows = core.qa_rows(session)
    assert [r["n"] for r in rows] == [1, 2] and rows[0]["question"] == "What if next summer is dry?"
    assert first["sim_calls"] == ["sim:2"] and first["checks"] is None and first["continuity"] == "session"
    assert second["checks"]["all_passed"] and "Checks: 9/9 passed" in second["answer"]
    for key in ("question", "answer", "model", "seconds", "checks", "sim_calls", "credential_scrub"):
        assert all(key in r for r in rows), key
    # Questions continue the stored session: no brief again, same session id, Q&A context from the host.
    for m in agent.messages[1:]:
        assert m["session_id"] == "ses_fake" and "# Farm brief" not in m["message"]
        assert m["message"].startswith("## Question time")
    latest = json.loads((session / "latest.json").read_text())
    assert latest["source"] == "question 2" and (session / "qa/REPORT-2.md").exists()
    assert core.load_solution(session)["decision"]["evidence"][-1]["source"] == "sim:3"
    assert not (session / "qa/pending.json").exists()


def test_worker_replays_after_interrupted_turn(solved, tmp_path):
    session, agent = solved["session"], solved["agent"]
    (session / "agent/.turn-in-progress").write_text("x")
    core.write_json(session / "qa/pending.json", {"n": 3, "question": "What does nitrate leaching mean?"})
    row = worker.ask(session, Phases(), agent=agent)
    last = agent.messages[-1]
    assert last["session_id"] is None and not last["keep_session"] and row["continuity"] == "replay"
    assert "# Farm brief" in last["message"] and "## Earlier questions and answers" in last["message"]
    assert "What if next summer is dry?" in last["message"] and "sim:2 simulate (rain -30%" in last["message"]
    assert any("interrupted" in h for h in row["host"])


def test_worker_withholds_answer_with_unmatched_figures(solved):
    session, agent = solved["session"], solved["agent"]
    rows = []
    for question in ("What do I earn?", "Give me wrong numbers"):
        n = len(core.qa_rows(session)) + 1
        core.write_json(session / "qa/pending.json", {"n": n, "question": question})
        rows.append(worker.ask(session, Phases(), agent=agent))
    good, bad = rows
    assert not good["withheld"] and all(v["passed"] for v in good["figure_checks"].values())
    assert good["answer"].startswith("A typical year returns $") and "inches" in good["answer"]
    assert bad["withheld"] and bad["answer"].startswith("This answer was withheld:")
    assert "99999" in bad["answer"] and "193 inches" in bad["answer"]
    assert not bad["figure_checks"]["dollar_figures_match_cited_runs"]["passed"]
    assert not bad["figure_checks"]["us_unit_figures_match_cited_runs"]["passed"]
    assert (session / f"qa/withheld-{bad['n']}.md").read_text().startswith("A typical year returns $99999")
    assert core.qa_rows(session)[-1]["answer"] == bad["answer"]


# -- the page -------------------------------------------------------------------------------

def test_page_questionnaire_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", tmp_path)
    launched = []
    monkeypatch.setattr(core, "spawn", lambda session, kind, env=None: launched.append((Path(session), kind, env)))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    assert not at.exception
    assert at.text_input(key="model").value == "opencode-go/gpt-6-luna"
    assert any("OPENCODE_GO_API_KEY" in c.value for c in at.caption)
    assert "offline-test-key" not in str([c.value for c in at.caption])
    at.selectbox(key="q_county").set_value(STORY_TEXT).run()
    assert not at.exception and not at.error
    assert [s.value for s in at.subheader] == ["Farm", "Current rotation", "What matters", "Hard limits",
                                               "Price table", "Weather view", "Anything else"]
    assert any("County estimate" in m.value for m in at.markdown)
    at.button(key="preset_profit").click().run()
    assert at.slider(key="q_w_money").value == 5 and at.slider(key="q_w_water").value == 0
    at.button(key="preset_steady").click().run()
    assert at.slider(key="q_w_money").value == 4 and at.radio(key="q_risk").value == "protect me in bad years"
    at.checkbox(key="q_lim_no_new_machinery").check()
    at.session_state["q_price_changes"] = {"corn|price": 4.6}
    at.run()
    next(b for b in at.button if b.label == "Submit and run the advisor").click().run()
    assert not at.exception, at.exception
    assert len(launched) == 1 and launched[0][1] == "solve" and launched[0][2] == {}
    session = launched[0][0]
    public = json.loads((session / "form.json").read_text())
    assert public["limits"] == {"no_new_machinery": True}
    assert public["economics"] == {"crop_prices": {"corn": {"value": 4.6, "unit": "usd/bu"}}}
    assert public["priorities"]["preset"] == "steady" and public["priorities"]["money"] == 4
    assert json.loads((session / "questionnaire.json").read_text())["q_county"] == STORY_TEXT
    assert any(h.value == "2 · Solution" for h in at.header)
    assert any("Ranking" in m.value or "Starting" in m.value for m in at.markdown)


def test_page_model_picker_builds_the_worker_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", tmp_path)
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    at.selectbox(key="model_preset").set_value("OpenRouter · GPT-6 Luna").run()
    assert at.text_input(key="model").value == "openrouter/openai/gpt-6-luna"
    assert any("OPENROUTER_API_KEY" in w.value for w in at.warning)
    at.text_input(key="api_key").set_value("FAKE-PAGE-KEY-123").run()
    assert not at.warning and any("OPENROUTER_API_KEY from this page" in c.value for c in at.caption)
    assert at.session_state["api_key_provider"] == "openrouter"
    at.selectbox(key="model_preset").set_value("Custom endpoint (OpenAI-compatible)").run()
    at.text_input(key="model").set_value("custom/qwen3.5:9b").run()
    at.text_input(key="base_url").set_value("http://localhost:11434/v1").run()
    assert any("host.docker.internal" in c.value for c in at.caption)
    assert "FAKE-PAGE-KEY-123" not in str([c.value for c in at.caption] + [w.value for w in at.warning])
    assert not list(tmp_path.iterdir())


def test_page_shows_field_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", tmp_path)
    monkeypatch.setattr(core, "spawn", lambda *a, **kw: pytest.fail("must not launch"))
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    at.selectbox(key="q_county").set_value(STORY_TEXT).run()
    at.multiselect(key="q_lim_include_crops").select("oats")
    at.multiselect(key="q_lim_exclude_crops").select("oats")
    at.run()
    next(b for b in at.button if b.label == "Submit and run the advisor").click().run()
    assert not at.exception
    assert [e.value for e in at.error] == ["A crop cannot be both included and excluded"]
    assert not list(tmp_path.iterdir())


def test_page_shows_solution_and_questions(solved, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", solved["root"])
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.session_state["session"] = str(solved["session"])
    at.run()
    assert not at.exception, at.exception
    assert [h.value for h in at.header] == ["2 · Solution", "3 · Questions about this solution"]
    assert at.success and at.success[0].value.startswith("**Recommended:")
    assert len(at.dataframe) >= 2  # comparison and checks
    assert any("supports, and does not replace, your judgment" in i.value for i in at.info)
    assert any(e.label == "Glossary" for e in at.expander)
    assert [m.name for m in at.chat_message][:4] == ["user", "assistant", "user", "assistant"]
    assert at.chat_input and not at.chat_input[0].disabled
    text = " ".join(m.value for m in at.markdown)
    assert "Why not alfalfa?" in text and "Checks: 9/9 passed" in text


def test_page_hides_unchecked_answer_and_comparison(solved, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", solved["root"])
    failed = {"form_weights": {"passed": False, "detail": ["Weights differ from the form"]},
              "all_passed": False}
    solution = {"run": {"mode": "No recommendation", "episode": {"status": "completed"},
                        "no_recommendation_reason": "decision checks failed: form weights",
                        "recommendation": None, "timings": {"total_seconds": 1}},
                "decision": {"recommended": {"rotation": [{"crop": "corn"}]}},
                "checks": failed, "comparison": {"source": "none", "rows": []},
                "report_text": "# Your field rotation report\nThere is no recommendation this time.",
                "source": "question 1"}
    monkeypatch.setattr(core, "load_solution", lambda session: solution)
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.session_state["session"] = str(solved["session"])
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 1 and "No recommendation this time" in at.error[0].value
    assert "form weights" in at.error[0].value
    assert not at.tabs
    assert not any("Comparison" in m.value or "Recommended:" in m.value for m in at.markdown)
    assert at.chat_input == []
