"""Offline chat tests: real private service and simulator, a scripted fake agent, no provider."""

import json
import re
import sqlite3
import urllib.error

import pytest

from fieldshift.agent import chat, chat_forms, checks, evidence, farms, pilot, report
from fieldshift.agent.runtime import Runtime

STORY = "US-19-169"
TOKEN = "FAKE-TEST-CREDENTIAL-" + "Zq7" * 30  # synthetic, never a real credential


def call(port, token, fid, route, body=None):
    try:
        return json.loads(pilot.http(f"http://127.0.0.1:{port}/farm/{fid}/{route}", token, body))
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, **json.loads(exc.read().decode())}


class FakeAgent:
    """Stands in for the OpenCode container: same files, same service calls, scripted replies."""

    def __init__(self, wrong_dollar=False):
        self.messages, self.refused, self.wrong_dollar = [], [], wrong_dollar

    def prepare(self, run_dir, farm_brief):
        (run_dir / "work" / "evidence").mkdir(parents=True)
        (run_dir / "data" / "opencode").mkdir(parents=True)
        (run_dir / "FARM.md").write_text(farm_brief)

    def sim(self, run_dir, port, token, fid, route, body):
        work = run_dir / "work"
        n = int((work / ".farmsim_calls").read_text()) + 1 if (work / ".farmsim_calls").exists() else 1
        (work / ".farmsim_calls").write_text(str(n))
        data = call(port, token, fid, route, body)
        (work / f".sim-response-{n}.json").write_text(json.dumps(data))
        return n, data

    def turn(self, run_dir, message, *, session_id, keep_session, farm_id, port, token):
        self.messages.append({"message": message, "session_id": session_id, "keep_session": keep_session})
        farmer = message.rsplit("## Farmer's message", 1)[1].strip()
        work = run_dir / "work"
        if "County" in farmer and "irrigat" not in farmer.casefold():
            # Ranking before any confirmed form must be refused by the service.
            self.refused.append(call(port, token, farm_id, "scoreboard", {"weights": {"income": 1}}))
            reply = "Is the field irrigated, and what do you grow? Pick a preset: profit, steady, soil builder, water saver."
        elif "rainfed" in farmer:
            proposal = call(port, token, farm_id, "form", {"location": {"irrigated": False},
                                                           "current_rotation": [{"crop": "corn"}, {"crop": "soybean"}],
                                                           "priorities": {"preset": "profit"},
                                                           "limits": {"no_new_machinery": True}})
            assert proposal["confirmation_required"]
            self.refused.append(call(port, token, farm_id, "simulate", {"rotations": [[{"crop": "corn"}]]}))
            reply = "Thanks. Please check the table below."
        elif "weights" in message and "Use exactly these weights" in message:
            weights = json.loads(re.search(r"Use exactly these weights: (\{.*\})", message)[1])
            limits = json.loads(re.search(r"Use exactly these limits: (\{.*\})", message)[1])
            n, board = self.sim(run_dir, port, token, farm_id, "scoreboard",
                                {"weights": weights, "limits": limits, "top": 15})
            choice = checks.choose(board)
            decision = {"existing": {"rotation": board["existing"]["rotation"]}, "weights": weights,
                        "weight_reasons": {k: "form" for k in weights}, "limits": limits,
                        "limit_reasons": {k: "form" for k in limits}, "scoreboard": f"sim:{n}",
                        "recommended": {"rotation": choice["rotation"], "rank": choice["rank"],
                                        "from_tied_set": bool(choice.get("tied_with_best")), "override_reason": None},
                        "alternatives": [], "why": f"Top of the ranking [sim:{n}].", "evidence": [],
                        "confidence": "medium"}
            (work / "DECISION.json").write_text(json.dumps(decision))
            (work / "REPORT.md").write_text(f"Keep {report.label(choice['rotation'])} [sim:{n}]. "
                                            "This supports, does not replace your judgment.")
            reply = f"I recommend {report.label(choice['rotation'])} [sim:{n}]."
        elif "drier" in farmer:
            n, data = self.sim(run_dir, port, token, farm_id, "simulate",
                               {"rotations": [[{"crop": "corn"}, {"crop": "soybean"}]], "rain_change_pct": -30})
            assert "results" in data, data
            reply = f"With 30% less rain in every year the typical return falls [sim:{n}]."
            if self.wrong_dollar:
                reply = f"With 30% less rain a typical year returns about $99999 per acre [sim:{n}]."
        else:
            call(port, token, farm_id, "form", {"economics": {"crop_prices": {"corn": {"value": 4.9, "unit": "usd/bu"}}}})
            reply = "Water price cannot matter on a rainfed field. Please confirm the corn price in the table."
        (run_dir / "events.jsonl").write_text(json.dumps({"type": "text", "sessionID": "ses_fake",
                                                          "part": {"text": reply}}) + "\n")
        (run_dir / "transcript.jsonl").write_text(json.dumps({"summary": {"messages": len(self.messages)}}) + "\n")
        return {"status": "completed", "exit_code": 0, "session_id": "ses_fake", "reply": reply,
                "agent_seconds": 0.0, "trace": {"tokens": {"input": 1, "output": 1}, "steps": 1, "read_paths": []},
                "credential_scrub": chat.scrub_credentials(run_dir, [], keep_session=keep_session)}


SCRIPT = ["Hi, my farm is in Story County, Iowa.",
          "It's rainfed, corn then soybean, profit preset, no new machinery.",
          "Yes, that's right.",
          "What if next summer is 30% drier?",
          "My water is free and I can get $5 more per ton for corn; does that change it?"]


def test_confirmation_words():
    for yes in ("yes", "Yes, that's right.", "Looks good, go ahead", "ok", "Correct."):
        assert chat_forms.is_confirmation(yes), yes
    for no in ("Yes, but it is irrigated", "yes 80%", "No", "Correct except the rotation", "yes?",
               "Actually it's corn-corn-soy", "", "yes " * 60):
        assert not chat_forms.is_confirmation(no), no


def test_merge_and_changes():
    base = {"location": {"county": "Story County, IA", "irrigated": False}, "economics": {"labour_usd_hr": 20}}
    new = chat_forms.merge(base, {"location": {"irrigated": True}, "economics": None, "weather_view": "outlook"})
    assert new == {"location": {"county": "Story County, IA", "irrigated": True}, "weather_view": "outlook"}
    assert chat_forms.changes(base, new) == ["economics", "location.irrigated", "weather_view"]
    assert base["location"]["irrigated"] is False  # input untouched
    with pytest.raises(farms.FarmError):
        chat_forms.merge(base, ["not", "an", "object"])


def test_find_county():
    assert chat.find_county("Hi, my farm is in Story County, Iowa.") == (STORY, "Story County, IA")
    assert chat.find_county("story, ia please")[0] == STORY
    adm, candidates = chat.find_county("We farm in Washington County")
    assert adm is None and len(candidates) > 5
    assert chat.find_county("hello there") == (None, [])


def test_gate_versions_and_patch(tmp_path):
    runtime = Runtime(root=tmp_path / ".service")
    forms = chat_forms.ChatForms(runtime, tmp_path / "forms")
    runtime.chat_form = forms
    draft = forms.seed({"location": {"county": "Story County, IA", "irrigated": False},
                        "priorities": {"preset": "steady"}, "weather_view": "record"})
    with pytest.raises(farms.FarmError):
        forms.gate(draft)  # the draft is never rankable
    assert forms.confirm("yes") is None  # nothing proposed yet
    first = forms.propose(draft, {"current_rotation": [{"crop": "corn"}, {"crop": "soybean"}],
                                  "priorities": {"preset": "profit"}})
    assert first["confirmation_required"] and first["changed_fields"] == ["current_rotation", "priorities.preset"]
    table = forms.table()
    assert "corn → soybean (your words)" in table and "**(changed)**" in table
    assert forms.confirm("No, it's corn-corn-soybean") is None  # a correction is not a confirmation
    with pytest.raises(farms.FarmError):
        forms.gate(forms.pending["farm_id"])  # proposed values cannot be ranked
    second = forms.propose(draft, {"current_rotation": [{"crop": "corn"}, {"crop": "corn"}, {"crop": "soybean"}],
                                   "priorities": {"preset": "profit"}})
    assert second["proposal"] != first["proposal"]
    applied = forms.confirm("Yes, that's right.")
    assert applied["version"] == 1 and applied["confirmation"]["proposal"] == second["proposal"]
    assert any(s.startswith("## Cropping history") for s in applied["changed_brief_sections"])
    forms.gate(applied["farm_id"])
    with pytest.raises(farms.FarmError):
        forms.propose(draft, {"weather_view": "outlook"})  # stale version
    forms.propose(applied["farm_id"], {"economics": {"water_usd_acre_foot": 60}})
    forms.gate(applied["farm_id"])  # the confirmed version stays usable while a change waits
    v2 = forms.confirm("ok")
    assert v2["farm_id"] != applied["farm_id"] and v2["normalised"]["farm"]["economics"]["water_charge_usd_per_mm_ha"] > 0
    with pytest.raises(farms.FarmError):
        forms.gate(applied["farm_id"])  # superseded
    v3 = (forms.propose(v2["farm_id"], {"economics": None}), forms.confirm("yes"))[1]
    assert "economics" not in v3["input"]
    files = sorted(p.name for p in (tmp_path / "forms").iterdir())
    assert {"state.json", "v0-draft.json", "v1.json", "v2.json", "v3.json"} <= set(files)
    reloaded = chat_forms.ChatForms.load(runtime, tmp_path / "forms")
    assert reloaded.confirmed_farm_id == v3["farm_id"] and len(reloaded.versions) == 4
    runtime.cache.clear()


def test_fake_agent_chat_end_to_end_and_resume(tmp_path):
    agent = FakeAgent()
    session = chat.Chat(model="fake/agent", agent=agent, root=tmp_path)
    replies = []
    chat.run(session, iter(SCRIPT), out=replies.append)
    folder = session.dir
    assert folder.name.endswith("-story-ia")
    # Ranking was refused before confirmation, by the service, not the model.
    assert [r["http_status"] for r in agent.refused] == [422, 422]
    assert "confirmed" in agent.refused[0]["message"]
    turns = [json.loads(line) for line in (folder / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == 5 and [t["turn"] for t in turns] == [1, 2, 3, 4, 5]
    for key in ("farmer", "reply", "sim_calls", "checks", "seconds", "tokens", "credential_scrub", "form_version"):
        assert all(key in t for t in turns), key
    assert "Please check these values" in turns[1]["reply"] and "| Irrigated | no (rainfed)" in turns[1]["reply"]
    assert turns[2]["confirmed_version"] == 1 and turns[2]["sim_calls"] == ["sim:1"]
    assert turns[2]["checks"]["all_passed"] and "Checks: 9/9 passed" in turns[2]["reply"]
    assert turns[3]["sim_calls"] == ["sim:2"] and turns[3]["checks"] is None
    assert turns[3]["withheld"] is False and all(v["passed"] for v in turns[3]["answer_checks"].values())
    assert turns[4]["pending_proposal"] and "**(changed)**" in turns[4]["reply"]
    assert turns[4]["confirmed_version"] == 1  # the price change waits for a yes
    # One session: the brief is sent once, later turns continue the same session id.
    assert "# Farm brief" in agent.messages[0]["message"] and agent.messages[0]["session_id"] is None
    assert all(m["session_id"] == "ses_fake" and "# Farm brief" not in m["message"] for m in agent.messages[1:])
    assert "Use exactly these weights" in agent.messages[2]["message"]
    for name in ("conversation.md", "turns.jsonl", "session.json", "DECISION.json", "REPORT.md", "checks.json",
                 "FARM.md", "forms/state.json", "forms/v0-draft.json", "forms/v1.json",
                 "transcripts/turn-001.jsonl", "events/turn-005.jsonl", "agent/work/.sim-response-1.json"):
        assert (folder / name).exists(), name
    assert (folder.stat().st_mode & 0o777) == 0o700
    conversation = (folder / "conversation.md").read_text()
    assert conversation.count("**Farmer:**") == 5 and "Checks: 9/9 passed" in conversation
    assert "Checks" in (folder / "REPORT.md").read_text()
    state = json.loads((folder / "session.json").read_text())
    assert state["opencode_session_id"] == "ses_fake" and state["continuity"] == "session"
    assert state["decision"]["form_version"] == 1 and state["status"] == "closed"

    # Resume after an interrupted turn: the session store is dropped and the context replayed.
    (folder / "agent" / ".turn-in-progress").write_text("x")
    resumed = chat.Chat.resume(folder, agent=agent)
    assert not (folder / "agent" / "data").exists() and resumed.state["continuity"] == "replay"
    chat.run(resumed, iter(["Yes"]), out=replies.append)
    last = agent.messages[-1]
    assert last["session_id"] is None and not last["keep_session"]
    assert "## Conversation so far" in last["message"] and "sim:2 simulate (rain -30%" in last["message"]
    assert "plainly confirms proposal" in last["message"]
    turns = [json.loads(line) for line in (folder / "turns.jsonl").read_text().splitlines()]
    assert len(turns) == 6 and turns[-1]["confirmed_version"] == 2 and turns[-1]["continuity"] == "replay"


def test_chat_answer_with_wrong_dollar_is_withheld(tmp_path):
    session = chat.Chat(model="fake/agent", agent=FakeAgent(wrong_dollar=True), root=tmp_path)
    chat.run(session, iter(SCRIPT[:4]), out=lambda _: None)
    turn = [json.loads(line) for line in (session.dir / "turns.jsonl").read_text().splitlines()][3]
    assert turn["withheld"] and turn["reply"].startswith("This answer was withheld:")
    assert "99999" in (session.dir / "withheld-turn-004.md").read_text()


def test_host_asks_for_county_first(tmp_path):
    session = chat.Chat(model="fake/agent", agent=FakeAgent(), root=tmp_path)
    assert session.handle("hello") == chat.ASK_COUNTY
    assert "Which state" in session.handle("Washington County")
    assert session.dir is None and len(session.early) == 2
    session.close()


def _synthetic_store(run_dir, *, leak_into_conversation=False):
    (run_dir / "data" / "opencode").mkdir(parents=True)
    (run_dir / "data" / "claude").mkdir()
    (run_dir / "data" / "claude" / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {"accessToken": TOKEN}}))
    (run_dir / "data" / "opencode" / "auth.json").write_text(
        json.dumps({"anthropic": {"type": "oauth", "access": TOKEN, "refresh": "MLBENCH_INVALID_NOT_A_CREDENTIAL",
                                  "expires": 1}}))
    (run_dir / "work").mkdir()
    (run_dir / "work" / "anthropic_auth.jsonl").write_text("bootstrap ok\n")
    db = run_dir / "data" / "opencode" / "opencode.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("create table credential (id text primary key, value text)")
    con.execute("create table account (id text primary key, access_token text, refresh_token text)")
    con.execute("create table session_message (seq integer, type text, data text)")
    con.execute("insert into credential values ('c1', ?)", (json.dumps({"access": TOKEN}),))
    con.execute("insert into account values ('a1', ?, 'x')", (TOKEN,))
    con.execute("insert into session_message values (1, 'user', ?)", (json.dumps({"text": "my farm"}),))
    con.execute("insert into session_message values (2, 'assistant', ?)",
                (json.dumps({"text": "token " + TOKEN[:40] if leak_into_conversation else "hello"}),))
    con.commit()
    con.close()
    return db


def test_credential_scrub_keeps_conversation(tmp_path):
    db = _synthetic_store(tmp_path)
    values = chat.credential_strings(tmp_path / "data" / "opencode" / "auth.json")
    assert TOKEN in values and "MLBENCH_INVALID_NOT_A_CREDENTIAL" not in values
    assert chat.scan(tmp_path, values)  # present before
    result = chat.scrub_credentials(tmp_path, values)
    assert result["verified_clean"] and result["session_kept"]
    assert result["rows_deleted"] == {"credential": 1, "account": 1} and not result["other_rows_deleted"]
    assert {"data/opencode/auth.json", "data/claude"} <= set(result["removed"])
    assert chat.scan(tmp_path, values) == []
    assert TOKEN.encode() not in b"".join(p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from session_message").fetchone()[0] == 2
    con.close()
    assert TOKEN not in json.dumps(result)  # the report never contains the credential


def test_credential_in_conversation_drops_session(tmp_path):
    _synthetic_store(tmp_path, leak_into_conversation=True)
    values = chat.credential_strings(tmp_path / "data" / "opencode" / "auth.json")
    result = chat.scrub_credentials(tmp_path, values)
    assert result["other_rows_deleted"] == {"session_message": 1}
    assert not result["session_kept"] and not (tmp_path / "data").exists()
    assert result["verified_clean"] and chat.scan(tmp_path, values) == []


def test_unverifiable_scrub_is_not_called_clean(tmp_path):
    _synthetic_store(tmp_path)
    result = chat.scrub_credentials(tmp_path, [], keep_session=False)
    assert not result["verified_clean"] and not (tmp_path / "data").exists()


def test_checks_line_formats():
    ok = {"a": {"passed": True}, "b": {"passed": True}, "all_passed": True}
    assert chat.checks_line(ok, None) == "Checks: 2/2 passed"
    bad = {"form_weights": {"passed": False, "detail": ["Weights differ from the form"]},
           "simulator_numbers": {"passed": False, "advisory": True, "detail": ["x"], "numbers_unmatched": 2},
           "all_passed": False}
    board = {"rows": [{"rank": 1, "rotation": [{"crop": "corn"}, {"crop": "soybean"}], "tied_with_best": False}],
             "existing": {"rank": 3, "tied_with_best": False}}
    line = chat.checks_line(bad, board)
    assert line.startswith("No checked recommendation this time. Failed checks: priorities copied from your answers.")
    assert "Reference ranking" not in line and "corn → soybean" not in line


def test_prompt_and_client_exist():
    text = (pilot.VM / "agents" / chat.PROMPT).read_text()
    assert "./farmsim form" in text and "DECISION.json" in text
    assert "form '<json patch>'" in (pilot.VM / "farmsim").read_text()
    assert evidence.load_cards(chat.EVIDENCE_ROOT)
