"""Farmer memory: profiles, approved notes, history summaries, proposals, and the worker with a scripted agent."""

import json

import pytest
from streamlit.testing.v1 import AppTest
from test_app_advisor import PAGE, STORY, STORY_TEXT, Phases, ScriptedAgent, answers

from fieldshift.agent import checks
from fieldshift.app import advisor_core as core
from fieldshift.app import advisor_worker as worker
from fieldshift.app import profiles

NOTE = "My landlord does not allow alfalfa."


def test_profile_notes_are_bounded_and_need_a_drop_choice(tmp_path):
    name = profiles.create("Demo Iowa!", tmp_path)
    assert name == "demo-iowa" and profiles.names(tmp_path) == ["demo-iowa"]
    with pytest.raises(profiles.ProfileError):
        profiles.create("demo iowa", tmp_path)
    profiles.add_note(name, NOTE, tmp_path)
    assert profiles.notes(name, tmp_path) == [NOTE]
    assert "- " + NOTE in (tmp_path / "profiles/demo-iowa/PROFILE.md").read_text()
    long = "x" * 299
    for i in range(4):
        profiles.add_note(name, f"{i}{long}", tmp_path)
    with pytest.raises(profiles.ProfileError, match="drop"):
        profiles.add_note(name, "y" * 299, tmp_path)
    items = profiles.add_note(name, "y" * 299, tmp_path, drop="0" + long)
    assert "0" + long not in items and items[-1] == "y" * 299
    assert profiles.size(items) <= profiles.NOTES_LIMIT
    with pytest.raises(profiles.ProfileError):
        profiles.add_note(name, "z" * (profiles.NOTE_MAX + 1), tmp_path)
    assert profiles.remove_note(name, NOTE, tmp_path)[0] != NOTE
    text = profiles.merge_free_text("Three fields.", [NOTE])
    assert text.startswith("Three fields.") and profiles.NOTES_HEADER in text and NOTE in text
    assert profiles.merge_free_text("", []) == ""


@pytest.fixture(scope="module")
def remembered(tmp_path_factory):
    """Profile with one approved note and one earlier session; a solution and one question."""
    root = tmp_path_factory.mktemp("app")
    story = core.county_defaults(STORY, False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "APP_ROOT", root)
        mp.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
        name = profiles.create("demo-iowa")
        profiles.add_note(name, NOTE)
        earlier = core.new_session("Story County, IA", "opencode-go/fake-model", profile=name)
        core.write_json(earlier / "form.json", {"priorities": {"preset": "profit"}})
        with open(earlier / "qa.jsonl", "w") as out:
            out.write(json.dumps({"n": 1, "question": "Would rye help?", "answer": "Rye cuts nitrate.\nMore."}) + "\n")
        profiles.record_submission(name, earlier, {"q_county": STORY_TEXT}, {})

        session = core.new_session("Story County, IA", "opencode-go/fake-model", profile=name)
        q = answers(free_text=profiles.merge_free_text("We farm 3 fields we rent.", profiles.notes(name)))
        public, _, errors = core.validate(q, story)
        assert errors is None
        core.write_json(session / "form.json", public)
        profiles.record_submission(name, session, {"q_county": STORY_TEXT, "q_free_text": "We farm 3 fields we rent."},
                                   public)
        agent = MemoryAgent()
        receipt = worker.solve(session, Phases(), agent=agent)
        core.write_json(session / "qa/pending.json", {"n": 1, "question": "We bought a no-till drill last year. "
                                                                          "Would cereal rye cut my nitrate losses?"})
        row = worker.ask(session, Phases(), agent=agent)
        yield {"root": root, "name": name, "earlier": earlier, "session": session, "agent": agent,
               "receipt": receipt, "row": row}


class MemoryAgent(ScriptedAgent):
    def turn(self, run_dir, message, *, session_id, keep_session, **where):
        work = run_dir / "work"
        if "## Farmer's question" not in message:
            self.history = sorted(p.name for p in (work / "history").glob("*.md"))
            assert "approved notes" in message and NOTE in message
            normal = json.loads((run_dir.parent / "advice/form.json").read_text())
            limits = normal["limits"] | {"exclude_crops": ["alfalfa"]}
            reasons = dict.fromkeys(normal["limits"], "form") | {"exclude_crops": {"quote": NOTE}}
            n, board = self.sim(run_dir, where, "scoreboard", {"weights": normal["weights"], "limits": limits, "top": 15})
            self.write_decision(run_dir, normal, n, board)
            decision = json.loads((work / "DECISION.json").read_text())
            decision.update(limits=limits, limit_reasons=reasons)
            (work / "DECISION.json").write_text(json.dumps(decision))
            (work / profiles.PROPOSALS_FILE).write_text(json.dumps([
                {"note": "Farms 3 rented fields.", "source": 'farmer said "We farm 3 fields we rent."'},
                {"note": "Prefers corn.", "source": 'farmer said "I love corn"'}]))
            reply = "Done."
            self.messages.append({"message": message, "session_id": session_id, "keep_session": keep_session})
        else:
            m, found = self.sim(run_dir, where, "studies", {"query": "cereal rye nitrate", "top": 3})
            assert found["results"][0]["farm_fit"]["label"] == "close"
            (work / profiles.PROPOSALS_FILE).write_text(json.dumps(
                [{"note": "Owns a no-till drill.", "source": "farmer said 'We bought a no-till drill last year.'"}]))
            reply = f"Yes; see the Iowa study [card:{found['results'][0]['id']}] found via sim:{m}."
            self.messages.append({"message": message, "session_id": session_id, "keep_session": keep_session})
        (run_dir / "data/opencode").mkdir(parents=True, exist_ok=True)
        (run_dir / "data/opencode/opencode.db").write_text("stand-in")
        (run_dir / "events.jsonl").write_text(json.dumps({"type": "text", "sessionID": "ses_fake",
                                                          "part": {"text": reply}}) + "\n")
        (run_dir / "transcript.jsonl").write_text("{}\n")
        from fieldshift.agent import chat
        return {"status": "completed", "exit_code": 0, "session_id": "ses_fake", "reply": reply, "agent_seconds": 0.1,
                "trace": {"steps": 1, "read_paths": ["/work/evidence/seifert2017-yield-penalty.md",
                                                     "/work/evidence/kaspar2007-rye-nitrate.md"]},
                "credential_scrub": chat.scrub_credentials(run_dir, [], keep_session=keep_session)}


def test_note_becomes_quoted_limit_and_checks_pass(remembered):
    receipt = remembered["receipt"]
    assert receipt["checks"]["all_passed"], receipt["checks"]
    assert receipt["checks"]["limits_grounded_in_form_or_quote"]["passed"]
    decision = json.loads((remembered["session"] / "advice/DECISION.json").read_text())
    assert decision["limits"]["exclude_crops"] == ["alfalfa"]
    # Without the note in free_text the same quote is not grounded.
    normal = json.loads((remembered["session"] / "advice/form.json").read_text())
    assert checks.ground_limits(decision, normal | {"free_text": "We farm 3 fields we rent."})


def test_history_and_profile_records(remembered):
    agent, earlier, session = remembered["agent"], remembered["earlier"], remembered["session"]
    assert agent.history == [f"{earlier.name}.md"]
    text = (session / "agent/work/history" / f"{earlier.name}.md").read_text()
    assert "Would rye help?" in text and "Rye cuts nitrate." in text and "More." not in text
    assert "Outcome: not finished" in text
    state = json.loads((session / "session.json").read_text())
    assert state["profile"] == "demo-iowa" and state["history_sessions"] == [earlier.name]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "APP_ROOT", remembered["root"])
        assert profiles.sessions("demo-iowa") == [earlier.name, session.name]
        assert profiles.last_questionnaire("demo-iowa")["q_free_text"] == "We farm 3 fields we rent."
    question = agent.messages[-1]["message"]
    assert "history/" in question and "demo-iowa" in question


def test_proposals_need_farmer_words_and_a_click(remembered):
    session, root = remembered["session"], remembered["root"]
    items = profiles.proposals(session)
    assert [(p["note"], p["status"]) for p in items] == [
        ("Farms 3 rented fields.", "pending"), ("Prefers corn.", "rejected_by_code"),
        ("Owns a no-till drill.", "pending")]
    assert remembered["row"]["memory_proposals"] == 1 and remembered["row"]["sim_calls"] == ["sim:2"]
    assert not (session / "agent/work" / profiles.PROPOSALS_FILE).exists()
    assert profiles.notes("demo-iowa", root) == [NOTE]  # nothing saved before a click
    profiles.decide(session, 1, True, root)
    profiles.decide(session, 3, False, root)
    assert profiles.notes("demo-iowa", root) == [NOTE, "Farms 3 rented fields."]
    with pytest.raises(profiles.ProfileError):
        profiles.decide(session, 2, True, root)


def test_page_profile_prefill_and_approve(remembered, monkeypatch):
    monkeypatch.setattr(core, "APP_ROOT", remembered["root"])
    monkeypatch.setattr(core, "spawn", lambda *a: None)
    session = remembered["session"]
    items = profiles.proposals(session)
    for p in items:
        if p["status"] in ("approved", "rejected"):
            p["status"] = "pending"
    core.write_json(session / "memory_proposals.json", items)
    profiles.remove_note("demo-iowa", "Farms 3 rented fields.")
    at = AppTest.from_file(PAGE, default_timeout=180).run()
    assert not at.exception
    at.selectbox(key="profile_choice").set_value("demo-iowa").run()
    assert not at.exception, at.exception
    assert at.session_state["profile"] == "demo-iowa"
    assert at.session_state["q_free_text"] == "We farm 3 fields we rent."
    assert at.session_state["q_county"] == STORY_TEXT
    assert any(NOTE in m.value for m in at.markdown)
    at.session_state["session"] = str(session)
    at.run()
    assert not at.exception, at.exception
    assert any("Remember this for next time" in m.value for m in at.markdown)
    at.button(key="memory_ok_1").click().run()
    assert not at.exception, at.exception
    assert "Farms 3 rented fields." in profiles.notes("demo-iowa")
    assert profiles.proposals(session)[0]["status"] == "approved"
