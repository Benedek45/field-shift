"""Export an episode's model messages (reasoning, text, tool calls) from OpenCode's SQLite store.

OpenCode's `--format json` stream carries no reasoning, but the session database does. Only the
`session_message` table is read; account and credential tables are never touched.

    uv run python -m fieldshift.agent.transcripts backfill <runs-dir>
    uv run python -m fieldshift.agent.transcripts watch <runs-dir> --until-pid <pid>
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

OUTPUT_CHARS = 4000


def _db(run_dir):
    return Path(run_dir) / "data" / "opencode" / "opencode.db"


def _part(part):
    kind = part.get("type")
    if kind == "reasoning":
        text = part.get("text") or ""
        native = part.get("native") or {}
        hidden = "reasoningEncryptedContent" in native or "signature" in native
        return {"type": kind, "text": text, "encrypted_only": not text and hidden}
    if kind == "text":
        return {"type": kind, "text": part.get("text") or ""}
    if kind == "tool":
        state = part.get("state") or {}
        output = "".join(c.get("text", "") for c in state.get("content") or [] if isinstance(c, dict))
        return {"type": kind, "name": part.get("name"), "status": state.get("status"), "input": state.get("input"),
                "output": output[:OUTPUT_CHARS], "output_chars": len(output)}
    return {"type": kind}


def read_messages(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        rows = con.execute("select seq, type, data from session_message "
                           "where type in ('user', 'assistant') order by seq").fetchall()
    finally:
        con.close()
    messages = []
    for seq, kind, data in rows:
        data = json.loads(data)
        content = data.get("content")
        parts = [_part(p) for p in content] if isinstance(content, list) else []
        if kind == "user" and not parts:
            parts = [{"type": "text", "text": str(data.get("text") or "")[:OUTPUT_CHARS]}]
        messages.append({"seq": seq, "role": kind, "model": (data.get("model") or {}).get("id"),
                         "time": data.get("time"), "parts": parts})
    return messages


def summarise(messages):
    reasoning = [p for m in messages for p in m["parts"] if p["type"] == "reasoning"]
    return {"messages": len(messages), "reasoning_parts": len(reasoning),
            "reasoning_chars": sum(len(p["text"]) for p in reasoning),
            "reasoning_encrypted_only": sum(p["encrypted_only"] for p in reasoning)}


def export(run_dir, snapshot=False):
    """Write <run>/transcript.jsonl and return its summary. Raises if the database is missing."""
    run_dir = Path(run_dir)
    messages = read_messages(_db(run_dir))
    summary = dict(summarise(messages), complete=not snapshot, exported=time.time())
    tmp = run_dir / "transcript.jsonl.part"
    with open(tmp, "w") as out:
        out.write(json.dumps({"summary": summary}) + "\n")
        out.writelines(json.dumps(message) + "\n" for message in messages)
    os.replace(tmp, run_dir / "transcript.jsonl")
    return summary


def _complete(run_dir):
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        return False
    with open(path) as handle:
        return json.loads(handle.readline())["summary"].get("complete", False)


def backfill(runs_dir):
    done = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if run_dir.is_dir() and (run_dir / "meta.json").exists() and _db(run_dir).exists() and not _complete(run_dir):
            done.append((run_dir.name, export(run_dir)))
    return done


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def watch(runs_dir, until_pid, interval):
    """Snapshot running episodes whose database disappears at the end (Anthropic cleanup)."""
    seen = {}
    while _alive(until_pid):
        for run_dir in Path(runs_dir).iterdir():
            db = _db(run_dir)
            if not run_dir.is_dir() or (run_dir / "meta.json").exists() or not db.exists():
                continue
            wal = db.with_name(db.name + "-wal")
            stamp = max(db.stat().st_mtime, wal.stat().st_mtime if wal.exists() else 0)
            if seen.get(run_dir.name) != stamp:
                try:
                    export(run_dir, snapshot=True)
                    seen[run_dir.name] = stamp
                except (sqlite3.Error, OSError, json.JSONDecodeError):
                    pass
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["backfill", "watch"])
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--until-pid", type=int)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.command == "backfill":
        for name, summary in backfill(args.runs_dir):
            print(name, summary)
    else:
        if args.until_pid is None:
            parser.error("watch needs --until-pid")
        watch(args.runs_dir, args.until_pid, args.interval)


if __name__ == "__main__":
    main()
