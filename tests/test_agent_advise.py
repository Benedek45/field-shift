"""Offline product checks for the one-shot farmer harness."""

import json

import pytest

from fieldshift.agent import advise, checks, outlook, pilot, providers, report


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Offline test must not fetch an outlook")
    monkeypatch.setattr(outlook, "fetch", unavailable)
    monkeypatch.setattr(advise, "episode", unavailable)


def source_form(tmp_path):
    source = tmp_path / "form.json"
    source.write_text(json.dumps({"location": {"adm_id": "US-19-169", "irrigated": False},
                                  "current_rotation": [{"crop": "corn"}, {"crop": "soybean"}],
                                  "priorities": {"preset": "steady"}, "weather_view": "record"}))
    return source


def fake_episode(*, bad=False, raising=False):
    def run(out, normal, fid, farm_brief, token, port, model, key):
        if raising:
            raise RuntimeError("offline synthetic failure")
        board = json.loads(pilot.http(f"http://127.0.0.1:{port}/farm/{fid}/scoreboard", token,
                                      {"weights": normal["weights"], "limits": normal["limits"], "top": 40}))
        selected = checks.choose(board)
        decision = {
            "weights": dict(normal["weights"]),
            "limits": dict(normal["limits"]),
            "recommended": {"rotation": selected["rotation"], "override_reason": None},
            "alternatives": [], "why": "A brief explanation based on the checked simulator results.",
            "evidence": [], "information_gaps": [], "risks_for_local_review": [],
        }
        if bad:
            decision["weights"]["income"] = 0
        return decision, "A checked explanation with no unsupported numeric claims.", {
            "status": "completed", "exit_code": 0, "cards_opened": []}, {}
    return run


def assert_no_recommendation(out, receipt, scoreboard):
    run = json.loads((out / "run.json").read_text())
    assert run["recommendation"] is None and run["no_recommendation_reason"]
    assert (out / "scoreboard.json").is_file() and (out / "checked_scoreboard.json").is_file()
    assert (out / "DECISION.json").is_file()
    assert receipt["mode"] == "No recommendation"
    text = (out / "REPORT.md").read_text()
    assert "There is no recommendation this time" in text
    assert "submit again or change the form" in text.casefold()
    for row in scoreboard["rows"]:
        assert report.label(row["rotation"]) not in text


def test_episode_exception_produces_no_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
    out = tmp_path / "failed"
    receipt = advise.run(source_form(tmp_path), out, agent_episode=fake_episode(raising=True))
    board = json.loads((out / "scoreboard.json").read_text())
    assert "RuntimeError" in receipt["no_recommendation_reason"]
    assert_no_recommendation(out, receipt, board)


def test_failed_decision_check_produces_no_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
    out = tmp_path / "failed-check"
    receipt = advise.run(source_form(tmp_path), out, agent_episode=fake_episode(bad=True))
    board = json.loads((out / "scoreboard.json").read_text())
    assert "form weights" in receipt["no_recommendation_reason"]
    assert_no_recommendation(out, receipt, board)


def test_passing_decision_keeps_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "offline-test-key")
    out = tmp_path / "passed"
    receipt = advise.run(source_form(tmp_path), out, agent_episode=fake_episode())
    assert receipt["checks"]["all_passed"]
    assert receipt["recommendation"] == json.loads((out / "DECISION.json").read_text())["recommended"]["rotation"]
    assert receipt["mode"] != "No recommendation"
    text = (out / "REPORT.md").read_text()
    assert "Recommended:" in text
    assert "Operating return $/acre/year" in text and "Field water inches/year" in text and "$/ha" not in text


def test_go_model_without_credentials_fails_before_writing(tmp_path):
    out = tmp_path / "no-credentials"
    with pytest.raises(providers.MissingKeyError, match="OPENCODE_API_KEY") as caught:
        advise.run(source_form(tmp_path), out, model="opencode-go/fake")
    assert "OPENCODE_GO_API_KEY" in str(caught.value)
    assert not out.exists()
