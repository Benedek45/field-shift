"""Farmer profiles for the questionnaire app: approved notes, the last questionnaire and past sessions.

    data/processed/app/profiles/<slug>/PROFILE.md       approved notes, one "- " line each, <= NOTES_LIMIT chars
    data/processed/app/profiles/<slug>/last_form.json   last submitted questionnaire (page state and public form)
    data/processed/app/profiles/<slug>/sessions.json    the profile's session folder names, oldest first

Notes are the farmer's own statements. Nothing is written without a click on the page: the agent can only
propose notes (MEMORY_PROPOSALS.json in its workspace), and a proposal is listed only when its quoted words
appear in what the farmer typed. Past sessions are summarised into the agent's workspace as history/*.md.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from fieldshift.agent import report
from fieldshift.app import advisor_core as core

NOTES_LIMIT = 1500
NOTE_MAX = 300
PROPOSALS_FILE = "MEMORY_PROPOSALS.json"
NOTES_HEADER = "Approved notes from earlier sessions (the farmer's own words):"
HISTORY_SESSIONS = 8
QUOTE = re.compile(r"[\"“”']([^\"“”]{3,})[\"“”']")


class ProfileError(ValueError):
    pass


def root(base=None):
    return Path(base or core.APP_ROOT) / "profiles"


def slug(name):
    text = re.sub(r"[^a-z0-9]+", "-", str(name).casefold()).strip("-")[:40]
    if not text:
        raise ProfileError("A profile name needs at least one letter or digit")
    return text


def path(name, base=None):
    return root(base) / slug(name)


def create(name, base=None):
    p = path(name, base)
    if p.exists():
        raise ProfileError(f"Profile {slug(name)!r} already exists")
    p.mkdir(parents=True, mode=0o700)
    write_notes(p, [])
    core.write_json(p / "sessions.json", [])
    return p.name


def names(base=None):
    r = root(base)
    return sorted(p.name for p in r.iterdir() if (p / "PROFILE.md").exists()) if r.exists() else []


# -- notes ------------------------------------------------------------------------------------

def notes(name, base=None):
    text = (path(name, base) / "PROFILE.md").read_text() if (path(name, base) / "PROFILE.md").exists() else ""
    return [line[2:].strip() for line in text.splitlines() if line.startswith("- ") and line[2:].strip()]


def size(items):
    return sum(len(n) for n in items)


def write_notes(p, items):
    text = (f"# Farmer profile: {Path(p).name}\n\nApproved notes (the farmer's own statements; at most "
            f"{NOTES_LIMIT} characters in total):\n\n" + "".join(f"- {n}\n" for n in items))
    tmp = Path(p) / "PROFILE.md.part"
    tmp.write_text(text)
    tmp.chmod(0o600)
    tmp.replace(Path(p) / "PROFILE.md")


def clean_note(note):
    note = " ".join(str(note).split())
    if not note:
        raise ProfileError("A note cannot be empty")
    if len(note) > NOTE_MAX:
        raise ProfileError(f"A note can have at most {NOTE_MAX} characters")
    return note


def add_note(name, note, base=None, drop=None):
    """Append an approved note; `drop` (an existing note) is removed first. Raises if over the limit."""
    note = clean_note(note)
    items = notes(name, base)
    if drop is not None:
        if drop not in items:
            raise ProfileError("The note to drop is not in the profile")
        items.remove(drop)
    if note in items:
        return items
    if size(items) + len(note) > NOTES_LIMIT:
        raise ProfileError(f"The notes would exceed {NOTES_LIMIT} characters; choose an old note to drop")
    items.append(note)
    write_notes(path(name, base), items)
    return items


def remove_note(name, note, base=None):
    items = [n for n in notes(name, base) if n != note]
    write_notes(path(name, base), items)
    return items


def notes_block(items):
    """Appended to the form's free_text, so limits quoted from a note pass the same grounding check."""
    return (NOTES_HEADER + "\n" + "".join(f"- {n}\n" for n in items)).strip() if items else ""


def merge_free_text(free_text, items):
    block = notes_block(items)
    return "\n\n".join(t for t in (free_text.strip(), block) if t)


def prompt_section(name, items):
    if not name:
        return ("\n\n## Farmer profile\nNo farmer profile is selected: do not write MEMORY_PROPOSALS.json. "
                "There is no history/ folder.")
    lines = [f"\n\n## What the farmer told us before (approved notes, profile `{name}`)",
             ("The farmer approved these notes in earlier sessions. They are farmer statements, like free_text, "
              "and are also copied at the end of the form's free_text: add a hard limit only where a note states "
              "it explicitly, quoting the note's exact words in limit_reasons.")]
    lines += [f"- {n}" for n in items] or ["(no notes yet)"]
    lines.append("Summaries of this farmer's earlier sessions, if any, are in history/ (grep them; cite as "
                 "history:<file>).")
    lines.append(MEMORY_TASK)
    return "\n".join(lines)


MEMORY_TASK = ("Memory task, before you finish: if the farmer's own words in this session (free_text above the "
               "approved notes, or the question) state a lasting fact about the farm (land, landlord, fields, "
               "equipment, buyers, water) that is not already a note, write MEMORY_PROPOSALS.json as "
               '[{"note":"short note","source":"farmer said \\"exact words\\""}], one entry per fact, quoting '
               "the farmer exactly. Never propose your own inferences or results. The farmer approves or rejects "
               "each; nothing is saved otherwise.")


# -- sessions and the last questionnaire --------------------------------------------------------

def sessions(name, base=None):
    return core.read_json(path(name, base) / "sessions.json", []) or []


def record_submission(name, session, questionnaire, public, base=None):
    p = path(name, base)
    items = [s for s in sessions(name, base) if s != Path(session).name] + [Path(session).name]
    core.write_json(p / "sessions.json", items)
    core.write_json(p / "last_form.json", {"saved": datetime.now(UTC).isoformat(), "session": Path(session).name,
                                           "questionnaire": questionnaire, "form": public})


def last_questionnaire(name, base=None):
    return (core.read_json(path(name, base) / "last_form.json", {}) or {}).get("questionnaire")


def _first_line(text):
    for line in str(text).splitlines():
        if line.strip():
            line = line.strip()
            return line if len(line) <= 240 else line[:237] + "…"
    return ""


def summary(session):
    """Short markdown summary of one earlier session: date, county, outcome, questions and first answer lines."""
    session = Path(session)
    state = core.read_json(session / "session.json", {}) or {}
    latest = core.read_json(session / "latest.json")
    outcome = "no recommendation"
    if latest:
        decision = core.read_json(session / latest["decision"], {}) or {}
        rec = (decision.get("recommended") or {}).get("rotation") if isinstance(decision.get("recommended"), dict) else None
        if rec and (latest.get("checks") or {}).get("all_passed"):
            try:
                outcome = "recommended " + report.label(rec)
            except (KeyError, TypeError, ValueError):
                outcome = "recommended (unreadable rotation)"
    elif state.get("status") not in ("solved", "failed"):
        outcome = "not finished"
    form = core.read_json(session / "form.json", {}) or {}
    lines = [f"# Earlier session {session.name}", "",
             f"- Date: {str(state.get('created', '?'))[:10]}", f"- County: {state.get('county', '?')}",
             f"- Outcome: {outcome}",
             f"- Priorities preset: {(form.get('priorities') or {}).get('preset', '?')}; hard limits: "
             + (json.dumps(form.get("limits")) if form.get("limits") else "none")]
    rows = core.qa_rows(session)
    if rows:
        lines += ["", "## Farmer's questions and the first line of each answer"]
        for r in rows:
            lines += [f"- Q{r.get('n')}: {_first_line(r.get('question', ''))}",
                      f"  A: {_first_line(r.get('answer', ''))}"]
    return "\n".join(lines) + "\n"


def write_history(work, name, current=None, base=None):
    """history/<session>.md for the profile's earlier sessions (newest HISTORY_SESSIONS), into the workspace."""
    out = Path(work) / "history"
    if out.exists():
        for old in out.glob("*.md"):
            old.unlink()
    written = []
    if not name:
        return written
    app = Path(base or core.APP_ROOT)
    earlier = [s for s in sessions(name, base) if s != (Path(current).name if current else None)]
    for s in earlier[-HISTORY_SESSIONS:]:
        if (app / s / "session.json").exists():
            out.mkdir(exist_ok=True)
            (out / f"{s}.md").write_text(summary(app / s))
            written.append(s)
    if written:
        out.chmod(0o777)
        for f in out.glob("*.md"):
            f.chmod(0o666)
    return written


# -- proposals --------------------------------------------------------------------------------

def farmer_words(session):
    """Everything the farmer typed in this session: questionnaire notes (without the approved-notes block)
    and the questions."""
    form = core.read_json(Path(session) / "form.json", {}) or {}
    text = str(form.get("free_text", "")).split(NOTES_HEADER)[0]
    return "\n".join([text, *(r.get("question", "") for r in core.qa_rows(session))])


def harvest(session, turn, base=None, extra=""):
    """Move the agent's MEMORY_PROPOSALS.json into memory/proposals.json as pending items.

    A proposal is kept only if its source quotes words the farmer actually typed (case-insensitive);
    others are stored as rejected by code with the reason. Returns the new items."""
    session = Path(session)
    src = session / "agent/work" / PROPOSALS_FILE
    if not src.exists():
        return []
    try:
        raw = json.loads(src.read_text())
    except ValueError:
        raw = []
    src.unlink()
    if isinstance(raw, dict):
        raw = raw.get("proposals", [])
    if not isinstance(raw, list):
        raw = []
    words = " ".join((farmer_words(session) + "\n" + extra).casefold().split())
    state = core.read_json(session / "session.json", {}) or {}
    existing = set(notes(state["profile"], base)) if state.get("profile") else set()
    stored = proposals(session)
    seen = {p["note"] for p in stored} | existing
    new = []
    for item in raw[:10]:
        if not isinstance(item, dict):
            continue
        try:
            note = clean_note(item.get("note", ""))
        except ProfileError:
            continue
        source = " ".join(str(item.get("source", "")).split())[:400]
        if note in seen:
            continue
        quotes = [q.strip() for q in QUOTE.findall(source)]
        found = [q for q in quotes if " ".join(q.casefold().split()) in words]
        entry = {"id": len(stored) + len(new) + 1, "turn": turn, "note": note, "source": source,
                 "status": "pending" if found else "rejected_by_code",
                 "reason": None if found else "the quoted words were not found in what the farmer typed"}
        new.append(entry)
        seen.add(note)
    core.write_json(session / "memory_proposals.json", stored + new)
    return new


def proposals(session):
    return core.read_json(Path(session) / "memory_proposals.json", []) or []


def decide(session, proposal_id, approve, base=None, drop=None):
    """Approve (append the note to the session's profile) or reject one pending proposal."""
    session = Path(session)
    items = proposals(session)
    item = next((p for p in items if p["id"] == proposal_id), None)
    if item is None or item["status"] != "pending":
        raise ProfileError("This proposal is not pending")
    if approve:
        state = core.read_json(session / "session.json", {}) or {}
        if not state.get("profile"):
            raise ProfileError("This session has no farmer profile")
        add_note(state["profile"], item["note"], base, drop=drop)
        item["status"] = "approved"
    else:
        item["status"] = "rejected"
    item["decided"] = datetime.now(UTC).isoformat()
    core.write_json(session / "memory_proposals.json", items)
    return item
