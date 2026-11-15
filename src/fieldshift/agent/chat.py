"""Terminal chat with the farm advisor: several turns, host-checked confirmations, code checks.

    uv run python -m fieldshift.agent.chat [--model M] [--county "Story, IA"] [--resume DIR] [--script FILE]

The farmer may start with only a county. The advisor (an OpenCode agent in the Colima container)
asks for the rest, proposes form values with `./farmsim form`, and this host program shows them as
a table and applies them only when the farmer's next message is a plain "yes". After every new or
changed recommendation the host re-runs the product checks and appends a one-line result.

Continuity: one OpenCode session per chat (`opencode run --session <id>` against the same run
directory). Provider credentials are prepared fresh for each turn and scrubbed afterwards, keeping
the conversation: credential files and credential-table rows are deleted, SQLite is vacuumed, and
every remaining file is searched for the credential strings. If anything is found, or a turn was
interrupted, the session store is deleted and the chat falls back to a fresh episode per turn that
replays the conversation, the confirmed form and earlier simulator results.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from fieldshift.agent import (
    brief,
    chat_forms,
    checks,
    evidence,
    farms,
    pilot,
    report,
    transcripts,
)
from fieldshift.agent.advise import audit_decision, local_service
from fieldshift.planner.data import county_options

DEFAULT_MODEL = "opencode-go/gpt-6-luna"
VERSION = "farm-chat-v1"
PROMPT = "farm-chat.md"
TURN_TIMEOUT = 900
REPO = Path(__file__).resolve().parents[3]
CHAT_ROOT = REPO / "data/processed/chat"
EVIDENCE_ROOT = REPO / "data/evidence"
CREDENTIAL_TABLES = ("credential", "account", "control_account", "account_state")
CREDENTIAL_PATHS = ("data/opencode/auth.json", "data/claude")
SQLITE_MAGIC = b"SQLite format 3\x00"
PLACEHOLDERS = {"MLBENCH_INVALID_NOT_A_CREDENTIAL"}  # known non-secret refresh placeholder
ASK_COUNTY = "Hello! Which county is your farm in? For example: Story County, IA."
CHECK_NAMES = {"form_weights": "priorities copied from your answers",
               "limits_grounded_in_form_or_quote": "limits match your answers",
               "all_limits_and_feasibility": "choice meets your limits",
               "independent_ranking_or_opened_card_override": "choice is top-ranked or tied (or a study-backed override)",
               "cards_exist_and_opened": "cited studies exist and were read",
               "sim_references_recorded": "cited simulator runs exist",
               "simulator_numbers": "numbers found in the cited runs",
               "dollar_figures_match_cited_runs": "dollar figures match the cited runs",
               "us_unit_figures_match_cited_runs": "inch and lb figures match the cited runs"}
STATES = {"alabama": "AL", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
          "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "idaho": "ID",
          "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
          "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
          "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
          "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
          "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
          "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
          "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
          "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"}
WORD = re.compile(r"[a-z][a-z.']*|,")


# -- county from the farmer's words ------------------------------------------------------------

def _county_index():
    index = {}
    for adm, text in county_options().items():
        name = text.split(" · ")[0]
        if "," not in name:
            continue
        county, state = name.rsplit(",", 1)
        words = tuple(WORD.findall(re.sub(r"\s+(county|parish)$", "", county.strip().casefold())))
        index.setdefault(words, []).append((state.strip().upper(), adm, name))
    return index


def find_county(text):
    """(adm_id, label) named in the text, or (None, candidate labels) if missing or ambiguous."""
    words = WORD.findall(text.casefold())
    alphas = set(STATES.values())
    candidates = []
    for key, rows in _county_index().items():
        n = len(key)
        for i in range(len(words) - n + 1):
            if tuple(words[i:i + n]) != key:
                continue
            j, said_county = i + n, False
            if j < len(words) and words[j] in ("county", "parish"):
                j, said_county = j + 1, True
            j += j < len(words) and words[j] == ","
            state = None
            if j < len(words) and words[j].upper() in alphas:
                state = words[j].upper()
            for length in (2, 1):
                name = " ".join(words[j:j + length])
                if state is None and name in STATES:
                    state = STATES[name]
            if state:
                match = [r for r in rows if r[0] == state]
                if match:
                    return match[0][1], match[0][2]
            elif said_county:
                candidates += rows
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][2]
    return None, sorted({r[2] for r in candidates})


# -- credentials -----------------------------------------------------------------------------

def credential_strings(path):
    """Secret strings (and distinctive fragments) the provider adapter wrote; never print them."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    out = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, str) and len(v) >= 16 and v not in PLACEHOLDERS:
            out.append(v)
    walk(value)
    return fragments(out)


def fragments(values):
    out = []
    for v in values:
        out.append(v)
        if len(v) >= 48:  # truncated copies in logs
            out += [v[:32], v[-24:]]
    return list(dict.fromkeys(out))


def _is_sqlite(path):
    try:
        with open(path, "rb") as stream:
            return stream.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def scan(root, values):
    """Relative paths of files under root that contain any of the values (bytes)."""
    needles = [v.encode() for v in values if v]
    hits = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if any(n in data for n in needles):
            hits.append(str(path.relative_to(root)))
    return hits


def scrub_credentials(run_dir, values, *, keep_session=True):
    """Delete every credential copy in run_dir while keeping the conversation; verify by search.

    Removes any provider auth file, deletes all rows of OpenCode's credential and
    account tables plus any other row containing a credential string (secure_delete, WAL truncated,
    VACUUM), then byte-searches every remaining file. If a credential is still found, or
    keep_session is False, the whole data tree is deleted. Returns a report without the values.
    """
    run_dir = Path(run_dir)
    data = run_dir / "data"
    out = {"removed": [], "rows_deleted": {}, "other_rows_deleted": {}, "databases": 0}
    for rel in CREDENTIAL_PATHS:
        path = run_dir / rel
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            out["removed"].append(rel)
        elif path.exists() or path.is_symlink():
            path.unlink()
            out["removed"].append(rel)
    needles = [v.encode() for v in values if v]
    if keep_session and data.is_dir():
        for path in sorted(data.rglob("*")):
            if path.is_symlink() or not path.is_file() or not _is_sqlite(path):
                continue
            out["databases"] += 1
            con = sqlite3.connect(path)
            try:
                con.execute("PRAGMA secure_delete=ON")
                tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
                for table in tables:
                    quoted = '"' + table.replace('"', '""') + '"'
                    if table in CREDENTIAL_TABLES:
                        n = con.execute(f"DELETE FROM {quoted}").rowcount
                        if n:
                            out["rows_deleted"][table] = out["rows_deleted"].get(table, 0) + n
                        continue
                    for column in [r[1] for r in con.execute(f"PRAGMA table_info({quoted})")]:
                        col = '"' + column.replace('"', '""') + '"'
                        for needle in needles:
                            n = con.execute(f"DELETE FROM {quoted} WHERE instr(CAST({col} AS BLOB), ?) > 0",
                                            (needle,)).rowcount
                            if n:
                                out["other_rows_deleted"][table] = out["other_rows_deleted"].get(table, 0) + n
                con.commit()
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.execute("VACUUM")
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                con.close()
    hits = scan(run_dir, values) if values else []
    out["files_with_credential"] = hits
    # Rows outside the credential tables mean the conversation itself held a credential.
    out["session_kept"] = bool(keep_session and data.is_dir() and not hits and not out["other_rows_deleted"])
    if not out["session_kept"] and data.exists():
        shutil.rmtree(data)
        out["removed"].append("data (whole session store)")
    out["files_with_credential_after"] = scan(run_dir, values) if values else []
    out["verified_clean"] = bool(values) and not out["files_with_credential_after"]
    out["credential_strings_checked"] = len(values)
    return out


def auth_errors(model):
    return (pilot.SubscriptionModelError,)


# -- the agent ----------------------------------------------------------------------------------

def read_events(path):
    events = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def reply_text(events):
    """The agent's text after its last tool call (its answer to the farmer)."""
    last_tool = max((i for i, e in enumerate(events) if e.get("type") == "tool_use"), default=-1)
    texts = [e.get("part", {}).get("text", "") for i, e in enumerate(events)
             if e.get("type") == "text" and i > last_tool]
    if not any(t.strip() for t in texts):
        texts = [e.get("part", {}).get("text", "") for e in events if e.get("type") == "text"]
    return "\n\n".join(t.strip() for t in texts if t.strip())


class OpenCodeAgent:
    """One OpenCode turn in the pinned container; credentials live only for the turn."""

    def __init__(self, model=DEFAULT_MODEL, timeout=TURN_TIMEOUT, prompt=PROMPT):
        self.model, self.timeout, self.prompt = model, timeout, prompt

    def prepare(self, run_dir, farm_brief):
        pilot.prepare(run_dir, {"id": "chat"}, "full", farm_brief, EVIDENCE_ROOT, agent_prompt=self.prompt)

    def turn(self, run_dir, message, *, session_id, keep_session, farm_id, port, token):
        run_dir = Path(run_dir)
        pilot.auth_module(self.model)  # refuses subscription-login models
        marker = run_dir / ".turn-in-progress"
        marker.write_text(str(time.time()))
        name = f"fieldshift-chat-{secrets.token_hex(6)}"
        env = dict(os.environ, FARMSIM_URL=f"http://host.docker.internal:{port}", FARMSIM_FARM=farm_id,
                   FARMSIM_TOKEN=token, FIELDSHIFT_CHAT="1", FIELDSHIFT_CONTAINER_NAME=name)
        values, result = [], {"status": "failed", "exit_code": None, "session_id": None}
        started = time.perf_counter()
        finished = False
        try:
            load_dotenv(REPO / ".agent-env", override=False)
            key = os.environ.get("OPENCODE_GO_API_KEY", "")
            values = fragments([key] if key else [])
            env.update(OPENCODE_API_KEY=key,
                       FIELDSHIFT_PASS_ENV="OPENCODE_API_KEY FARMSIM_URL FARMSIM_FARM FARMSIM_TOKEN FIELDSHIFT_CHAT")
            command = [str(pilot.VM / "oc.sh"), str(run_dir), "run", "--standalone", "--agent", "farm-advisor",
                       "--model", self.model, "--format", "json", "--auto",
                       *(["--session", session_id] if session_id else []), message]
            with open(run_dir / "events.jsonl", "w") as out, open(run_dir / "stderr.log", "w") as err:
                try:
                    code = subprocess.run(command, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                          timeout=self.timeout, check=False).returncode
                    result.update(status="completed" if code == 0 else "failed", exit_code=code)
                except subprocess.TimeoutExpired:
                    result["status"] = "timeout"
                    subprocess.run(["docker", "--context", os.environ.get("FIELDSHIFT_DOCKER_CONTEXT", "colima-rosetta"),
                                    "stop", name], capture_output=True, check=False)
            finished = True
            result["agent_seconds"] = round(time.perf_counter() - started, 1)
            events = read_events(run_dir / "events.jsonl")
            result["session_id"] = next((e["sessionID"] for e in events if e.get("sessionID")), None)
            result["reply"] = reply_text(events)
            result["trace"] = pilot.parse_events(run_dir / "events.jsonl")
            limit = transcripts.OUTPUT_CHARS
            transcripts.OUTPUT_CHARS = 10_000_000
            try:
                result["transcript"] = transcripts.export(run_dir)
            except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
                result["transcript"] = {"error": str(exc)}
            finally:
                transcripts.OUTPUT_CHARS = limit
            return result
        finally:
            if not finished:
                subprocess.run(["docker", "--context", os.environ.get("FIELDSHIFT_DOCKER_CONTEXT", "colima-rosetta"),
                                "stop", name], capture_output=True, check=False)
            result["credential_scrub"] = scrub_credentials(run_dir, values, keep_session=keep_session)
            values = None
            marker.unlink(missing_ok=True)


def continue_turn(agent, model, run_dir, state, compose, record, **where):
    """One agent turn in the stored OpenCode session, falling back once to a fresh replayed episode.

    state holds "continuity" ("session" or "replay") and "opencode_session_id" and is updated in
    place; compose(replay) returns the message; record["host"] collects host notes; where holds
    farm_id, port and token. Returns (result, replay) where replay is the mode the turn started in.
    """
    replay = state["continuity"] != "session"
    session_id = None if replay else state["opencode_session_id"]
    try:
        result = agent.turn(run_dir, compose(replay), session_id=session_id,
                            keep_session=state["continuity"] == "session", **where)
    except auth_errors(model) as exc:
        result = {"status": "infra", "error": str(exc), "reply": ""}
    if (result.get("status") != "infra" and not result.get("reply") and session_id
            and state["continuity"] == "session"):
        # A continued session that produced nothing: fall back to a fresh, replayed episode once.
        record["host"].append(f"Session turn failed ({result.get('status')}, exit {result.get('exit_code')}); "
                              "retried once as a fresh episode with replayed context.")
        record["failed_session_attempt"] = {k: result.get(k) for k in ("status", "exit_code", "agent_seconds",
                                                                     "credential_scrub")}
        scrub_credentials(run_dir, [], keep_session=False)
        state.update(continuity="replay", opencode_session_id=None)
        try:
            result = agent.turn(run_dir, compose(True), session_id=None, keep_session=False, **where)
        except auth_errors(model) as exc:
            result = {"status": "infra", "error": str(exc), "reply": ""}
    update_continuity(state, result, session_id, record)
    return result, replay


def update_continuity(state, result, session_id, record):
    scrub = result.get("credential_scrub") or {}
    if state["continuity"] != "session":
        return
    if not scrub.get("session_kept"):
        state.update(continuity="replay", opencode_session_id=None)
        record["host"].append("Session store deleted after the turn (credential scrub could not keep it); "
                              "later turns replay the conversation.")
    elif session_id and result.get("session_id") not in (None, session_id):
        state.update(continuity="replay", opencode_session_id=None)
        record["host"].append("OpenCode answered in a different session; later turns replay the conversation.")
    elif result.get("session_id"):
        state["opencode_session_id"] = result["session_id"]


# -- the chat -------------------------------------------------------------------------------

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _money(row):
    metrics = row.get("metrics", {})
    m = metrics.get("margin_usd_ha_yr", {})
    text = f"{m.get('median', float('nan')):.0f} $/ha typical, {m.get('p10', float('nan')):.0f} bad year"
    us = metrics.get("margin_usd_acre_yr")
    if isinstance(us, dict):
        text += f" (${us.get('median')}/acre typical, ${us.get('p10')}/acre bad year)"
    return text


def summarise_call(n, data):
    """One line per earlier simulator call, for replayed context."""
    try:
        if not isinstance(data, dict):
            return f"- sim:{n}: unreadable response"
        if "error" in data:
            return f"- sim:{n}: error: {str(data.get('message', data['error']))[:200]}"
        if "rows" in data and "existing" in data:
            rows = "; ".join(f"#{r['rank']} {report.label(r['rotation'])} (score {r['score']:.1f}, {_money(r)}"
                             f"{', tied with best' if r.get('tied_with_best') else ''})" for r in data["rows"][:3])
            ex = data["existing"]
            return (f"- sim:{n} scoreboard: {rows}; current {report.label(ex['rotation'])} rank {ex.get('rank')}"
                    f"{' (tied with best)' if ex.get('tied_with_best') else ''}")
        if "applied_weather_changes" in data:
            w = data["applied_weather_changes"]
            items = "; ".join(f"{report.label(r['rotation'])}: {_money(r)}" for r in data["results"])
            return (f"- sim:{n} simulate (rain {w['request_rain_change_pct']:+g}%, temperature "
                    f"{w['request_temp_change_c']:+g} C): {items}; current under these changes: "
                    f"{_money(data['existing_under_request_weather'])}")
        if "results" in data:
            return f"- sim:{n} forecast: " + "; ".join(report.label(r["rotation"]) for r in data["results"])
    except (KeyError, TypeError, ValueError):
        pass
    return f"- sim:{n}: see .sim-response-{n}.json"


def sim_summaries(work):
    """One line per saved simulator response in the agent workspace, in call order."""
    out = []
    for path in sorted(Path(work).glob(".sim-response-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = None
        out.append(summarise_call(path.stem.rsplit("-", 1)[1], data))
    return out


def recheck(runtime, fid, normal, run_dir, reply, opened, *, mode):
    """Product checks of the agent's current DECISION.json and REPORT.md against a validated form.

    Re-runs the form's reference ranking and the decision independently. Returns
    (decision, checked, reference, report text)."""
    work = Path(run_dir) / "work"
    notes = []
    try:
        decision = json.loads((work / "DECISION.json").read_text())
        if not isinstance(decision, dict):
            raise TypeError("DECISION.json must be an object")
    except (ValueError, TypeError) as exc:
        decision = {}
        notes.append(f"DECISION.json unreadable: {exc}")
    narrative = (work / "REPORT.md").read_text() if (work / "REPORT.md").exists() else ""
    available = {c.id for c in evidence.load_cards(EVIDENCE_ROOT)}
    calls = checks.recorded_calls(Path(run_dir) / "transcript.jsonl", work)
    reference = runtime.scoreboard(fid, {"weights": normal["weights"], "limits": normal["limits"], "top": 40})
    audit_board, simulated, audit_notes = audit_decision(runtime, fid, decision, reference)
    notes += audit_notes
    checked = checks.evaluate(decision, normal, audit_board, simulated, narrative + "\n" + reply,
                              available, opened, calls)
    farm = runtime.farm(fid)
    farm.evidence_root = EVIDENCE_ROOT
    choice = (decision.get("recommended") or {}).get("rotation")
    text = report.render(normal, reference, farm, checked, mode=mode, agent_report=narrative, notes=notes,
                         agent_choice=choice)
    return decision, checked, reference, text


def checks_line(checked, reference):
    items = {k: v for k, v in checked.items() if isinstance(v, dict) and "passed" in v}
    passed = sum(v["passed"] for v in items.values())
    line = f"Checks: {passed}/{len(items)} passed"
    advisory = [v for v in items.values() if not v["passed"] and v.get("advisory")]
    if advisory:
        line += f" (advisory: {advisory[0].get('numbers_unmatched', '?')} number(s) not found in the cited simulator runs)"
    failed = [(k, v) for k, v in items.items() if not v["passed"] and not v.get("advisory")]
    if failed:
        names = [CHECK_NAMES.get(k, k.replace("_", " ")) for k, _ in failed]
        line = "No checked recommendation this time. Failed checks: " + ", ".join(names) + "."
    elif not checked.get("all_passed"):
        line = "No checked recommendation this time. The decision checks did not all pass."
    return line


class Chat:
    def __init__(self, model=DEFAULT_MODEL, agent=None, root=CHAT_ROOT):
        self.model, self.root = model, Path(root)
        self.agent = agent or OpenCodeAgent(model)
        self.dir = self.forms = self.runtime = self.token = self.port = None
        self.stack = contextlib.ExitStack()
        self.state = {"version": VERSION, "model": model, "created": datetime.now(UTC).isoformat(),
                      "county": None, "adm_id": None, "opencode_session_id": None, "continuity": "session",
                      "turns": 0, "decision": None, "cards_opened": [], "status": "open"}
        self.early = []  # host-only turns before a county is known

    # -- lifecycle ------------------------------------------------------------------------
    @property
    def agent_dir(self):
        return self.dir / "agent"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.dir is not None:
            self.state["status"] = "closed"
            self.save()
        self.stack.close()

    def _open_service(self):
        self.runtime, self.token, self.port = self.stack.enter_context(local_service(self.dir))

    def start(self, adm_id, label):
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", label.split(",")[0].casefold().replace(" county", "")).strip("-")
        self.dir = self.root / f"{stamp}-{slug}-{label.rsplit(',', 1)[-1].strip().lower()}"
        self.dir.mkdir(parents=True, mode=0o700)
        self.dir.chmod(0o700)
        for sub in ("transcripts", "events"):
            (self.dir / sub).mkdir()
        self._open_service()
        self.forms = chat_forms.ChatForms(self.runtime, self.dir / "forms")
        self.runtime.chat_form = self.forms
        fid = self.forms.seed({"location": {"county": label, "irrigated": False},
                               "priorities": {"preset": "steady"}, "weather_view": "record"})
        farm_brief = brief.build(fid, self.runtime, include_outlook=False)
        (self.dir / "FARM.md").write_text(farm_brief)
        self.agent.prepare(self.agent_dir, farm_brief)
        self.state.update(county=label, adm_id=adm_id)
        (self.dir / "conversation.md").write_text(f"# Chat with the farm advisor — {label}\n\nModel: {self.model}\n\n")
        for record in self.early:
            self._record(record)
        self.early = []
        self.save()

    @classmethod
    def resume(cls, folder, model=None, agent=None):
        folder = Path(folder).resolve()
        state = json.loads((folder / "session.json").read_text())
        self = cls(model or state["model"], agent, folder.parent)
        self.dir, self.state = folder, state
        self.state["model"] = self.model
        self._open_service()
        self.forms = chat_forms.ChatForms.load(self.runtime, folder / "forms")
        self.runtime.chat_form = self.forms
        if (self.agent_dir / ".turn-in-progress").exists():
            # An interrupted turn may have left a credential behind: drop the whole session store.
            report_ = scrub_credentials(self.agent_dir, [], keep_session=False)
            (self.agent_dir / ".turn-in-progress").unlink()
            self.state.update(continuity="replay", opencode_session_id=None,
                              interrupted_turn_scrub=report_["removed"])
        if self.state["continuity"] == "session" and not (self.agent_dir / "data/opencode/opencode.db").exists():
            self.state.update(continuity="replay", opencode_session_id=None)
        self.state["status"] = "open"
        self.save()
        return self

    def save(self):
        if self.dir is not None:
            chat_forms.write(self.dir / "session.json", self.state)

    def turns(self):
        path = self.dir / "turns.jsonl" if self.dir else None
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path and path.exists() else []
        return rows

    # -- one turn ---------------------------------------------------------------------------
    def handle(self, text):
        started = time.perf_counter()
        text = text.strip()
        n = self.state["turns"] + len(self.early) + 1
        record = {"turn": n, "time": datetime.now(UTC).isoformat(), "farmer": text, "host": []}
        if self.forms is None:
            adm, found = find_county(text)
            if adm is None:
                reply = (ASK_COUNTY if not found else
                         "I found more than one county with that name: " + ", ".join(found[:8])
                         + ". Which state is it in?")
                record.update(reply=reply, kind="host-only", seconds={"total": round(time.perf_counter() - started, 2)})
                self.early.append(record)
                return reply
            self.start(adm, found)
            record["host"].append(f"County from the farmer's words: {found} ({adm}); draft form v0 registered.")
        pending_before = (self.forms.pending or {}).get("id")
        applied = self.forms.confirm(text) if pending_before else None
        if applied:
            record["host"].append(f"Plain confirmation of proposal {pending_before}: applied as form v{applied['version']}.")
            (self.dir / "FARM.md").write_text(applied["brief"])
        elif pending_before:
            record["host"].append(f"Not a plain confirmation; proposal {pending_before} still waiting.")
        work = self.agent_dir / "work"
        calls_before = int((work / ".farmsim_calls").read_text()) if (work / ".farmsim_calls").exists() else 0
        decision_before = _sha(work / "DECISION.json")
        (work / ".farmsim_farm").unlink(missing_ok=True)
        result, replay = continue_turn(
            self.agent, self.model, self.agent_dir, self.state,
            lambda replay: self.compose(text, applied, pending_before, replay=replay), record,
            farm_id=self.forms.current["farm_id"], port=self.port, token=self.token)
        reply = result.get("reply") or ""
        if result.get("status") == "infra":
            reply = ("The advisor is not available right now (provider or login problem: "
                     f"{result.get('error')}). Nothing was retried. Your answers so far are saved.")
            self.state["status"] = "blocked"
        elif not reply:
            reply = f"(The advisor gave no answer this turn: {result.get('status')}.)"
        trace = result.get("trace") or {}
        opened = {Path(p).stem for p in trace.get("read_paths", []) if "evidence/" in p} - {"INDEX"}
        self.state["cards_opened"] = sorted(set(self.state["cards_opened"]) | opened)
        calls_after = int((work / ".farmsim_calls").read_text()) if (work / ".farmsim_calls").exists() else 0
        extras, checked, figures = [], None, None
        t = time.perf_counter()
        if _sha(work / "DECISION.json") != decision_before and (work / "DECISION.json").exists():
            checked, line = self.check(reply)
            extras.append(line)
        if result.get("status") != "infra" and result.get("reply"):
            form = (self.forms.confirmed or self.forms.current or {}).get("normalised")
            figures, reply = checks.withhold_failing_answer(
                reply, checks.recorded_calls(self.agent_dir / "transcript.jsonl", work), form)
            if not all(v["passed"] for v in figures.values()):
                (self.dir / f"withheld-turn-{n:03d}.md").write_text(result["reply"])
                (self.dir / f"withheld-turn-{n:03d}.md").chmod(0o600)
        elif self.state["decision"] and self.forms.confirmed and \
                self.state["decision"]["form_version"] < self.forms.confirmed["version"]:
            extras.append(f"Note: the recommendation on file was made with your earlier answers "
                          f"(v{self.state['decision']['form_version']}); ask me to rerun it.")
        check_seconds = time.perf_counter() - t
        pending = self.forms.pending
        if pending and pending["id"] != pending_before:
            extras.append("Please check these values:\n\n" + self.forms.table() +
                          "\n\nReply **yes** if they are right, or tell me what to change. "
                          "Nothing is ranked with them until you confirm.")
        elif pending:
            extras.append("(The values in the last table are still waiting for your yes.)")
        reply = "\n\n".join([reply, *extras])
        record.update(reply=reply, kind="agent", sim_calls=[f"sim:{i}" for i in range(calls_before + 1, calls_after + 1)],
                      checks=None if checked is None else {
                          "all_passed": checked["all_passed"],
                          "passed": sum(v["passed"] for v in checked.values() if isinstance(v, dict)),
                          "total": sum(isinstance(v, dict) for v in checked.values()),
                          "failed": [k for k, v in checked.items() if isinstance(v, dict) and not v["passed"]]},
                      answer_checks=figures,
                      withheld=bool(figures) and not all(v["passed"] for v in figures.values()),
                      seconds={"agent": result.get("agent_seconds"), "checks": round(check_seconds, 2),
                               "total": round(time.perf_counter() - started, 1)},
                      tokens=trace.get("tokens"), steps=trace.get("steps"), tools=trace.get("tools"),
                      agent_status=result.get("status"), exit_code=result.get("exit_code"),
                      served_model=result.get("served_model"), host_auth=result.get("host_auth"),
                      continuity="replay" if replay else "session", session_id=result.get("session_id"),
                      farm_id=self.forms.current["farm_id"], form_version=self.forms.current["version"],
                      confirmed_version=(self.forms.confirmed or {}).get("version"),
                      pending_proposal=(pending or {}).get("id"), cards_opened_this_turn=sorted(opened),
                      credential_scrub=result.get("credential_scrub"), error=result.get("error"))
        if (self.agent_dir / "events.jsonl").exists():
            shutil.copyfile(self.agent_dir / "events.jsonl", self.dir / "events" / f"turn-{n:03d}.jsonl")
        if (self.agent_dir / "transcript.jsonl").exists():
            shutil.copyfile(self.agent_dir / "transcript.jsonl", self.dir / "transcripts" / f"turn-{n:03d}.jsonl")
        self._record(record)
        self.save()
        return reply

    def _record(self, record):
        record = dict(record, turn=self.state["turns"] + 1)
        self.state["turns"] += 1
        with open(self.dir / "turns.jsonl", "a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")
        os.chmod(self.dir / "turns.jsonl", 0o600)
        seconds = (record.get("seconds") or {}).get("total")
        meta = [f"{seconds:.1f} s" if seconds is not None else None,
                ", ".join(record.get("sim_calls") or []) or None,
                f"form v{record['form_version']}" if record.get("form_version") is not None else None]
        with open(self.dir / "conversation.md", "a") as out:
            out.write(f"## Turn {record['turn']}\n\n**Farmer:** {record['farmer']}\n\n**Advisor:**\n\n{record['reply']}\n\n"
                      + (f"_{' · '.join(m for m in meta if m)}_\n\n" if any(meta) else ""))

    # -- messages -----------------------------------------------------------------------------
    def compose(self, text, applied, pending_before, *, replay):
        f = self.forms
        current, confirmed = f.current, f.confirmed
        calls = int((self.agent_dir / "work/.farmsim_calls").read_text()) \
            if (self.agent_dir / "work/.farmsim_calls").exists() else 0
        lines = ["## Chat state (from the host program; not the farmer's words)",
                 (f"- Turn {self.state['turns'] + 1}. Farm: {self.state['county']}. Current form version v{current['version']} "
                 f"({'confirmed by the farmer' if current['confirmed'] else 'draft, NOT confirmed'})."),
                 f"- Simulator calls so far: {f'sim:1 to sim:{calls}' if calls else 'none'}."]
        if applied:
            lines += [(f"- The farmer's message below plainly confirms proposal {applied['confirmation']['proposal']} "
                      f"(host check). The host applied it as form v{applied['version']}; ranking tools now use it."),
                      "- Use exactly these weights: " + json.dumps(applied["normalised"]["weights"]),
                      "- Use exactly these limits: " + json.dumps(applied["normalised"]["limits"]),
                      f"- Weather view: {applied['normalised']['weather_view']}.",
                      "- Rank, check studies, write DECISION.json and REPORT.md, then reply with the recommendation."]
            if not replay and applied["changed_brief_sections"]:
                lines += ["", "### Brief sections that changed with this version", *applied["changed_brief_sections"]]
        elif f.pending:
            lines.append(f"- Proposal {f.pending['id']} is waiting. The host did NOT read the farmer's message as a plain "
                         "yes. If they corrected or added something, send a new ./farmsim form patch; otherwise "
                         "answer and ask them to confirm the table.")
        elif confirmed is None:
            lines.append("- No form is confirmed yet: gather the answers, then propose them with ./farmsim form. "
                         "Ranking and simulation refuse until the farmer confirms.")
        else:
            lines.append(f"- Confirmed weights: {json.dumps(confirmed['normalised']['weights'])}; limits: "
                         f"{json.dumps(confirmed['normalised']['limits'])}; weather view: "
                         f"{confirmed['normalised']['weather_view']}.")
        decision = self.state["decision"]
        lines.append("- Recommendation on file: " + ("none." if not decision else
                     f"DECISION.json for form v{decision['form_version']}, host checks "
                     f"{decision['passed']}/{decision['total']} passed."))
        state = "\n".join(lines)
        if not replay:
            head = [(self.dir / "FARM.md").read_text()] if self.state["opencode_session_id"] is None else []
            return "\n\n".join([*head, state, "## Farmer's message", text])
        parts = [brief.build(current["farm_id"], self.runtime, include_outlook=False),
                 "## Conversation so far (replayed by the host; your earlier session is not available)"]
        for row in self.turns():
            parts.append(f"**Farmer:** {row['farmer']}\n\n**Advisor:** {row['reply']}")
        if len(parts) == 2:
            parts.append("(This is the first message.)")
        if confirmed:
            parts += ["## Confirmed form (farmer units)", json.dumps(confirmed["input"], indent=1)]
        if f.pending and not applied:
            parts += ["## Proposal waiting for the farmer's yes", json.dumps(f.pending["input"], indent=1)]
        summaries = sim_summaries(self.agent_dir / "work")
        if summaries:
            parts += ["## Earlier simulator results (full responses: .sim-response-<n>.json in your workspace)",
                      "\n".join(summaries)]
        return "\n\n".join([*parts, state, "## Farmer's message", text])

    # -- checks -------------------------------------------------------------------------------
    def check(self, reply):
        work = self.agent_dir / "work"
        confirmed = self.forms.confirmed
        if confirmed is None:
            checked = {"confirmed_form": {"passed": False, "detail": [("No confirmed form: a recommendation needs "
                                                                      "the farmer's confirmed answers")]},
                       "all_passed": False}
            line = checks_line(checked, None)
            chat_forms.write(self.dir / "checks.json", checked)
            return checked, line
        fid, normal = confirmed["farm_id"], confirmed["normalised"]
        decision, checked, reference, text = recheck(
            self.runtime, fid, normal, self.agent_dir, reply, self.state["cards_opened"],
            mode="LLM chat explanation with deterministic form ranking")
        line = checks_line(checked, reference)
        (self.dir / "REPORT.md").write_text(text)
        chat_forms.write(self.dir / "DECISION.json", decision)
        chat_forms.write(self.dir / "checks.json", checked)
        items = [v for v in checked.values() if isinstance(v, dict)]
        self.state["decision"] = {"form_version": confirmed["version"], "farm_id": fid,
                                  "sha256": _sha(work / "DECISION.json"), "all_passed": checked["all_passed"],
                                  "passed": sum(v["passed"] for v in items), "total": len(items)}
        return checked, line


# -- command line ----------------------------------------------------------------------------

def script_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def run(chat, messages, out=print, county=None):
    """Feed farmer messages (an iterable) through the chat; print replies."""
    with chat:
        if county and chat.forms is None:
            adm, found = find_county(county if "," in county or "county" in county.casefold() else county + " County")
            if adm is None:
                raise farms.FarmError(f"Unknown or ambiguous county {county!r}; use 'County, ST'")
            chat.start(adm, found)
            out(f"Advisor: Hello! I have your farm in {found}. Tell me a little about it, or just say hello.")
        elif chat.forms is None:
            out("Advisor: " + ASK_COUNTY)
        for text in messages:
            if not text.strip():
                continue
            if text.strip().casefold() in ("quit", "exit", "bye"):
                break
            started = time.perf_counter()
            reply = chat.handle(text)
            out(f"\nAdvisor ({time.perf_counter() - started:.0f} s):\n{reply}\n")
            if chat.state["status"] == "blocked":
                break
    return chat


def interactive():
    while True:
        try:
            yield input("You: ")
        except EOFError:
            return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help=f"default {DEFAULT_MODEL}")
    p.add_argument("--county", help='start with a county, e.g. "Story, IA"')
    p.add_argument("--resume", type=Path, help="continue a saved chat folder")
    p.add_argument("--script", type=Path, help="one farmer message per line (non-interactive)")
    p.add_argument("--turn-timeout", type=int, default=TURN_TIMEOUT)
    args = p.parse_args()
    chat = None
    try:
        if args.resume:
            chat = Chat.resume(args.resume, args.model)
            chat.agent.timeout = args.turn_timeout
            for row in chat.turns()[-2:]:
                print(f"You: {row['farmer']}\n\nAdvisor:\n{row['reply']}\n")
        else:
            chat = Chat(args.model or DEFAULT_MODEL, OpenCodeAgent(args.model or DEFAULT_MODEL, args.turn_timeout))
        if args.script:
            lines = script_lines(args.script)

            def echo():
                for line in lines:
                    print(f"You: {line}")
                    yield line
            run(chat, echo(), county=args.county)
        else:
            run(chat, interactive(), county=args.county)
    except (farms.FarmError, ValueError, OSError) as exc:
        p.exit(2, f"Chat error: {exc}\n")
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    if chat is not None and chat.dir is not None:
        print(f"Session saved: {chat.dir}")


if __name__ == "__main__":
    main()
