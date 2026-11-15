"""Background jobs of the questionnaire app, one per session at a time:

    python -m fieldshift.app.advisor_worker solve <session>   # advise pipeline, session kept for questions
    python -m fieldshift.app.advisor_worker ask <session>     # answer qa/pending.json in the same session

The solution runs `agent.advise.run` with an episode hook that keeps the OpenCode session (credentials
scrubbed after the turn, as in the chat), so questions continue it with `chat.continue_turn`. The farm's
settings are locked: the service refuses form and economics edits from the agent.
"""

import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from fieldshift.agent import advise, chat, checks, pilot, providers
from fieldshift.app import advisor_core as core
from fieldshift.app import profiles

SOLVE_TIMEOUT = advise.EPISODE_TIMEOUT
QA_TIMEOUT = chat.TURN_TIMEOUT
SOLVE_PROMPT = "farm-advisor-v1.md"
QA_PROMPT = "farm-qa.md"


def make_agent(model, timeout, prompt):
    return chat.OpenCodeAgent(model, timeout, prompt=prompt)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class Job:
    """job.json for the page: phase, timings and a readable error (tracebacks go to worker.log only)."""

    def __init__(self, session, kind):
        self.path = Path(session) / "job.json"
        old = core.read_json(self.path, {}) or {}
        self.state = {"kind": kind, "state": "running", "phase": "starting", "started": old.get("started", time.time()),
                      "pid_started": time.time()}
        self.save()

    def phase(self, name):
        self.state["phase"] = name
        self.save()

    def save(self):
        core.write_json(self.path, self.state)

    def finish(self, error=None):
        self.state.update(state="failed" if error else "done", finished=time.time(), error=error)
        self.save()


def readable(exc):
    """A message a farmer can act on; never a traceback."""
    text = str(exc).strip()[:400]
    name = type(exc).__name__
    if isinstance(exc, providers.ProviderError):
        return f"The advisor model cannot run: {text}"
    if isinstance(exc, ValueError):
        return f"The questionnaire values were rejected: {text}"
    if isinstance(exc, OSError):
        return f"The advisor container or service could not run ({name}): {text}"
    return f"Unexpected failure ({name}); details are in the session's worker.log."


def _copy_clean(src, dst, scrub):
    """Copy an agent file into the session only if the credential search found nothing in it."""
    if src.exists() and src.name not in {Path(p).name for p in (scrub or {}).get("files_with_credential_after", [])}:
        shutil.copyfile(src, dst)
        dst.chmod(0o600)
        return True
    return False


def _cards(trace):
    return {Path(p).stem for p in (trace or {}).get("read_paths", []) if "evidence/" in p} - {"INDEX"}


def scrub_summary(scrub):
    scrub = scrub or {}
    return {"verified_clean": scrub.get("verified_clean"), "session_kept": scrub.get("session_kept"),
            "files_with_credential_after": len(scrub.get("files_with_credential_after", [])),
            "credential_strings_checked": scrub.get("credential_strings_checked")}


# -- solution -------------------------------------------------------------------------------

def solve(session, job, agent=None):
    session = Path(session)
    state = core.read_json(session / "session.json")
    model = state["model"]
    agent = agent or make_agent(model, SOLVE_TIMEOUT, SOLVE_PROMPT)
    agent_dir, out = session / "agent", session / "advice"

    def on_service(runtime, fid):
        runtime.chat_form = core.SettingsLock(fid)
        state["farm_id"] = fid
        core.write_json(session / "session.json", state)
        job.phase("ranking")

    def episode(out_, normal, fid, farm_brief, token, port, model_, key):
        job.phase("agent")
        agent.prepare(agent_dir, farm_brief)
        profile = state.get("profile")
        history = profiles.write_history(agent_dir / "work", profile, current=session)
        record = {"host": [f"history/ holds {len(history)} earlier session(s) of profile {profile}"] if profile else []}
        notes = profiles.notes(profile) if profile else []
        message = advise.episode_prompt(farm_brief, normal) + profiles.prompt_section(profile, notes) + pilot.TASK
        result, _ = chat.continue_turn(agent, model, agent_dir, state, lambda replay: message, record,
                                       farm_id=fid, port=port, token=token)
        job.phase("checks")
        scrub = result.get("credential_scrub")
        _copy_clean(agent_dir / "transcript.jsonl", out_ / "transcript.jsonl", scrub)
        _copy_clean(agent_dir / "events.jsonl", out_ / "events.jsonl", scrub)
        work = agent_dir / "work"
        decision = {}
        if (work / "DECISION.json").exists():
            try:
                decision = json.loads((work / "DECISION.json").read_text())
            except ValueError:
                decision = {}
        if not isinstance(decision, dict):
            decision = {}
        narrative = (work / "REPORT.md").read_text() if (work / "REPORT.md").exists() else ""
        trace = result.get("trace") or {}
        state["cards_opened"] = sorted(_cards(trace))
        state["decision_sha"] = _sha(work / "DECISION.json")
        state["history_sessions"] = history
        core.write_json(session / "session.json", state)
        proposed = profiles.harvest(session, "solution")
        core.write_json(session / "session.json", state)
        meta = {"status": result.get("status"), "exit_code": result.get("exit_code"),
                "seconds": result.get("agent_seconds"), "served_model": result.get("served_model"),
                "host_auth": result.get("host_auth"), "session_id": result.get("session_id"),
                "cards_opened": state["cards_opened"], "error": result.get("error"),
                "trace": {k: v for k, v in trace.items() if k != "read_paths"},
                "credential_scrub": scrub_summary(scrub), "continuity": state["continuity"],
                "host_notes": record["host"], "memory_proposals": len(proposed)}
        return decision, narrative, meta, checks.recorded_calls(agent_dir / "transcript.jsonl", work)

    receipt = advise.run(session / "form.json", out, model=model, agent_episode=episode, on_service=on_service)
    job.phase("comparison")
    decision = core.read_json(out / "DECISION.json", {}) or {}
    board = core.read_json(out / "scoreboard.json")
    with advise.local_service(out) as (runtime, _, _):
        runtime.chat_form = core.SettingsLock(state["farm_id"])
        comp = core.comparison(runtime, state["farm_id"], decision, board, receipt["checks"])
    core.write_json(session / "latest.json", {"source": "solution", "decision": "advice/DECISION.json",
                                              "report": "advice/REPORT.md", "checks": receipt["checks"],
                                              "comparison": comp, "mode": receipt["mode"]})
    episode_meta = receipt.get("episode") or {}
    state.update(status="solved", solved=datetime.now(UTC).isoformat(),
                 qa_available=receipt.get("model_used") is not None and episode_meta.get("status") == "completed"
                 and receipt["checks"].get("all_passed")
                 and (session / "agent/work").exists())
    core.write_json(session / "session.json", state)
    return receipt


# -- questions ------------------------------------------------------------------------------

def host_state(state, n, decision_checks, calls, normal):
    passed = sum(v["passed"] for v in decision_checks.values() if isinstance(v, dict) and "passed" in v)
    total = sum(isinstance(v, dict) and "passed" in v for v in decision_checks.values())
    return "\n".join([
        "## Question time (from the host program; not the farmer's words)",
        (f"- Question {n} about the checked solution for {state['county']}. The farmer's settings are fixed by the "
        "questionnaire; `./farmsim form` and `./farmsim economics` are refused."),
        "- Weights used for the solution: " + json.dumps(normal["weights"]),
        "- Hard limits used for the solution: " + json.dumps(normal["limits"]),
        f"- Weather view: {normal['weather_view']}.",
        f"- Recommendation on file: DECISION.json; host checks {passed}/{total} passed.",
        f"- Simulator calls so far: {f'sim:1 to sim:{calls}' if calls else 'none'}.",
        ("- Questionnaire sections: Farm, Current rotation, What matters, Hard limits, Price table, Weather view, "
        "Anything else."),
        (f"- Farmer profile `{state['profile']}`: approved notes are at the end of the form's free_text; earlier "
         f"sessions are summarised in history/ ({len(state.get('history_sessions') or [])} file(s)).\n- "
         + profiles.MEMORY_TASK
         if state.get("profile") else "- No farmer profile: do not write MEMORY_PROPOSALS.json.")])


def compose(session, state, n, question, replay, normal, decision_checks):
    work = session / "agent/work"
    calls = int((work / ".farmsim_calls").read_text()) if (work / ".farmsim_calls").exists() else 0
    head = host_state(state, n, decision_checks, calls, normal)
    if not replay:
        return f"{head}\n\n## Farmer's question\n\n{question}"
    parts = [(session / "advice/FARM.md").read_text(),
             "## Product form contract (validated questionnaire)", json.dumps(normal, indent=1),
             "## Your earlier solution (replayed by the host; your earlier session is not available)",
             "DECISION.json and REPORT.md in your working directory hold the checked solution; read them first.",
             "## Earlier questions and answers"]
    rows = core.qa_rows(session)
    parts += [f"**Farmer:** {r['question']}\n\n**Advisor:** {r['answer']}" for r in rows] or ["(none)"]
    summaries = chat.sim_summaries(work)
    if summaries:
        parts += ["## Earlier simulator results (full responses: .sim-response-<n>.json in your workspace)",
                  "\n".join(summaries)]
    return "\n\n".join([*parts, head, "## Farmer's question", question])


def ask(session, job, agent=None):
    session = Path(session)
    state = core.read_json(session / "session.json")
    pending = core.read_json(session / "qa/pending.json")
    if not pending:
        raise ValueError("No pending question")
    model, fid = state["model"], state["farm_id"]
    if not state.get("qa_available"):
        raise ValueError("This solution has no agent session to ask (deterministic template or failed episode)")
    agent = agent or make_agent(model, QA_TIMEOUT, QA_PROMPT)
    agent_dir, work = session / "agent", session / "agent/work"
    started = time.perf_counter()
    record = {"host": []}
    if (agent_dir / ".turn-in-progress").exists():
        # An interrupted turn may have left a credential copy: drop the whole session store.
        chat.scrub_credentials(agent_dir, [], keep_session=False)
        (agent_dir / ".turn-in-progress").unlink()
        state.update(continuity="replay", opencode_session_id=None)
        record["host"].append("An earlier turn was interrupted; its session store was deleted and the context replayed.")
    if state["continuity"] == "session" and not (agent_dir / "data/opencode/opencode.db").exists():
        state.update(continuity="replay", opencode_session_id=None)
    # Question time uses the Q&A agent instructions; the agent name stays farm-advisor.
    shutil.copyfile(pilot.VM / "agents" / QA_PROMPT, agent_dir / "cfg/opencode/agents/farm-advisor.md")
    normal = core.read_json(session / "advice/form.json")
    latest = core.read_json(session / "latest.json")
    n = pending["n"]
    job.phase("service")
    with advise.local_service(session / "advice") as (runtime, token, port):
        runtime.chat_form = core.SettingsLock(fid)
        calls_before = int((work / ".farmsim_calls").read_text()) if (work / ".farmsim_calls").exists() else 0
        decision_before = state.get("decision_sha")
        (work / ".farmsim_farm").unlink(missing_ok=True)
        job.phase("agent")
        result, replay = chat.continue_turn(
            agent, model, agent_dir, state,
            lambda replay: compose(session, state, n, pending["question"], replay, normal, latest["checks"]),
            record, farm_id=fid, port=port, token=token)
        reply = result.get("reply") or ""
        if result.get("status") == "infra":
            answer = ("The advisor is not available right now (provider or login problem: "
                      f"{result.get('error')}). Nothing was retried. Your question is saved; ask again later.")
        else:
            answer = reply or f"(The advisor gave no answer this time: {result.get('status')}.)"
        figures = None
        if reply and result.get("status") != "infra":
            figures, answer = checks.withhold_failing_answer(reply, checks.recorded_calls(agent_dir / "transcript.jsonl", work), normal)
            if figures and not all(v["passed"] for v in figures.values()):
                (session / f"qa/withheld-{n}.md").write_text(reply)
                (session / f"qa/withheld-{n}.md").chmod(0o600)
        trace = result.get("trace") or {}
        opened = _cards(trace)
        state["cards_opened"] = sorted(set(state.get("cards_opened", [])) | opened)
        calls_after = int((work / ".farmsim_calls").read_text()) if (work / ".farmsim_calls").exists() else 0
        checked, line, t = None, None, time.perf_counter()
        if (work / "DECISION.json").exists() and _sha(work / "DECISION.json") != decision_before:
            job.phase("checks")
            decision, checked, reference, text = chat.recheck(
                runtime, fid, normal, agent_dir, reply, state["cards_opened"],
                mode=f"LLM explanation (updated in question {n}) with deterministic form ranking")
            line = chat.checks_line(checked, reference)
            answer += "\n\n" + line
            (session / f"qa/REPORT-{n}.md").write_text(text)
            core.write_json(session / f"qa/DECISION-{n}.json", decision)
            core.write_json(session / f"qa/checks-{n}.json", checked)
            comp = core.comparison(runtime, fid, decision, reference, checked)
            core.write_json(session / "latest.json", {"source": f"question {n}", "decision": f"qa/DECISION-{n}.json",
                                                      "report": f"qa/REPORT-{n}.md", "checks": checked,
                                                      "comparison": comp, "mode": latest.get("mode")})
            recommendation = decision.get("recommended")
            if not checked.get("all_passed") or not (
                    recommendation.get("rotation") if isinstance(recommendation, dict) else None):
                state["qa_available"] = False
            state["decision_sha"] = _sha(work / "DECISION.json")
        check_seconds = time.perf_counter() - t
    proposed = profiles.harvest(session, f"question {n}", extra=pending["question"])
    scrub = result.get("credential_scrub")
    _copy_clean(agent_dir / "events.jsonl", session / f"qa/events-{n}.jsonl", scrub)
    _copy_clean(agent_dir / "transcript.jsonl", session / f"qa/transcript-{n}.jsonl", scrub)
    row = {"n": n, "time": datetime.now(UTC).isoformat(), "question": pending["question"], "answer": answer,
           "model": model, "served_model": result.get("served_model"), "status": result.get("status"),
           "exit_code": result.get("exit_code"), "error": result.get("error"),
           "seconds": {"agent": result.get("agent_seconds"), "checks": round(check_seconds, 2),
                       "total": round(time.perf_counter() - started, 1)},
           "sim_calls": [f"sim:{i}" for i in range(calls_before + 1, calls_after + 1)],
           "cards_opened_this_turn": sorted(opened), "checks_line": line, "figure_checks": figures,
           "withheld": bool(figures) and not all(v["passed"] for v in figures.values()),
           "checks": None if checked is None else {
               "all_passed": checked["all_passed"],
               "passed": sum(v["passed"] for v in checked.values() if isinstance(v, dict)),
               "total": sum(isinstance(v, dict) for v in checked.values()),
               "failed": [k for k, v in checked.items() if isinstance(v, dict) and not v["passed"]]},
           "continuity": "replay" if replay else "session", "session_id": result.get("session_id"),
           "tokens": trace.get("tokens"), "steps": trace.get("steps"), "credential_scrub": scrub_summary(scrub),
           "host": record["host"], "memory_proposals": len(proposed)}
    with open(session / "qa.jsonl", "a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    (session / "qa.jsonl").chmod(0o600)
    (session / "qa/pending.json").unlink(missing_ok=True)
    core.write_json(session / "session.json", state)
    return row


def run(kind, session):
    session = Path(session).resolve()
    try:
        with core.job_lock(session):
            job = Job(session, kind)
            try:
                if kind == "solve":
                    solve(session, job)
                elif kind == "ask":
                    ask(session, job)
                else:
                    raise ValueError(f"Unknown job {kind!r}")
            except Exception as exc:  # noqa: BLE001 — the page shows a readable message, the log the traceback
                traceback.print_exc()
                job.finish(readable(exc))
                if kind == "solve":
                    state = core.read_json(session / "session.json", {}) or {}
                    core.write_json(session / "session.json", dict(state, status="failed"))
                return 1
            job.finish()
            return 0
    except core.JobBusy:
        print("Another job of this session is running; nothing done.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(run(sys.argv[1], sys.argv[2]))
