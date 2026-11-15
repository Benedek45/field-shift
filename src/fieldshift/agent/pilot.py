"""Run the farm-advisor agent in containers and collect its decisions.

One run = one farm case x one setup (llm | evidence | full) x one model x one repeat,
in its own run directory with a fresh OpenCode data directory. The simulator service
must already be running on the host; the container reaches it through
host.docker.internal.
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fieldshift.agent import evidence, farms, transcripts

REPO = Path(__file__).resolve().parents[3]
VM = REPO / "vm"
SETUPS = ("llm", "evidence", "full")
TASK = (
    "\n\n---\n\nThe farm state above is everything known about this field. Choose the crop rotation "
    "for it as described in your instructions, using the tools present in the working directory, "
    "then write DECISION.json and REPORT.md."
)
CARD_RE = re.compile(r"card:([A-Za-z0-9_\-]+)")
SIM_RE = re.compile(r"sim:(\d+)")


def slug(text):
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


def http(url, token, body=None, timeout=900):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode())
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def farm_ids(cases, out_root):
    """Register each case as a farm once; the mapping is kept next to the runs."""
    path = out_root / "farms.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    for case in cases:
        if case["id"] in known:
            continue
        record = {
            "adm_id": case["adm_id"],
            "irrigated": bool(case.get("irrigated", False)),
            "overrides": case.get("overrides", {}),
            "priorities": {"text": case["priority"], "weights": case.get("weights", {})},
            "constraints": case.get("constraints", []),
            "anonymous": case.get("anonymous", True),
        }
        if case.get("economics"):
            record["economics"] = case["economics"]
        known[case["id"]] = farms.add(record)
        path.write_text(json.dumps(known, indent=2))
    return known


def canonical(rotation):
    """Rotation-invariant form of a cycle: the lexicographically smallest shift."""
    years = [(str(y.get("crop")), y.get("winter") or None) for y in rotation]
    if not years:
        return ()
    shifts = [tuple(years[i:] + years[:i]) for i in range(len(years))]
    return min(shifts, key=lambda s: [(c, w or "") for c, w in s])


def prepare(run_dir, case, setup, brief, evidence_root, agent_prompt="farm-advisor.md"):
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work = run_dir / "work"
    (run_dir / "cfg" / "opencode" / "agents").mkdir(parents=True)
    (run_dir / "data").mkdir()
    work.mkdir()
    (run_dir / "cfg" / "opencode" / "opencode.json").write_text(
        json.dumps({"$schema": "https://opencode.ai/config.json"}, indent=2) + "\n"
    )
    shutil.copy(VM / "agents" / agent_prompt, run_dir / "cfg" / "opencode" / "agents" / "farm-advisor.md")
    cards = []
    if setup in ("evidence", "full"):
        evidence.export(evidence_root, work / "evidence", exclude_groups=tuple(case.get("holdout_groups", ())))
        cards = sorted(p.stem for p in (work / "evidence").glob("*.md") if p.name != "INDEX.md")
    if setup == "full":
        shutil.copy(VM / "farmsim", work / "farmsim")
        (work / "farmsim").chmod(0o755)
        shutil.copy(VM / "SIMULATOR_NOTES.md", work / "SIMULATOR_NOTES.md")
    (run_dir / "FARM.md").write_text(brief)
    subprocess.run(["chmod", "-R", "a+rwX", str(run_dir)], check=True)
    return cards


def parse_events(path):
    out = {"tools": {}, "read_paths": [], "simulate_calls": 0, "scoreboard_calls": 0, "options_calls": 0, "denied": 0,
           "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0}, "cost": 0.0, "steps": 0,
           "errors": []}
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part", {})
        if event.get("type") == "tool_use":
            tool = part.get("tool", "?")
            out["tools"][tool] = out["tools"].get(tool, 0) + 1
            state = part.get("state", {})
            given = state.get("input") or {}
            if state.get("status") == "error":
                out["denied"] += "Permission denied" in str(state.get("error"))
                continue
            if tool == "read":
                out["read_paths"].append(str(given.get("path", "")))
            if tool == "shell":
                command = str(given.get("command", ""))
                out["simulate_calls"] += command.count("farmsim simulate") + command.count("farmsim scoreboard")
                out["scoreboard_calls"] += command.count("farmsim scoreboard")
                out["options_calls"] += command.count("farmsim options")
        elif event.get("type") == "step_finish":
            tokens = part.get("tokens", {})
            out["steps"] += 1
            out["tokens"]["input"] += tokens.get("input", 0)
            out["tokens"]["output"] += tokens.get("output", 0)
            out["tokens"]["reasoning"] += tokens.get("reasoning", 0)
            out["tokens"]["cache_read"] += tokens.get("cache", {}).get("read", 0)
            out["cost"] += part.get("cost", 0) or 0
        elif event.get("type") == "error":
            out["errors"].append(json.dumps(event)[:500])
    return out


def score(run_dir, cards, trace):
    result = {"valid": False}
    path = run_dir / "work" / "DECISION.json"
    if not path.exists():
        result["problem"] = "no DECISION.json"
        return result
    try:
        decision = json.loads(path.read_text())
        recommended = canonical(decision["recommended"]["rotation"])
        existing = canonical(decision["existing"]["rotation"])
        alternatives = [canonical(a["rotation"]) for a in decision.get("alternatives", [])]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        result["problem"] = f"unreadable decision: {error!r}"
        return result
    items = decision.get("evidence", [])
    sources = [str(item.get("source", "")) if isinstance(item, dict) else "" for item in items]
    text = path.read_text()
    cited = sorted(set(CARD_RE.findall(text)))
    opened = {Path(p).stem for p in trace["read_paths"] if "/evidence/" in p or p.startswith("evidence/")}
    sims = [int(n) for n in SIM_RE.findall(text)]
    result.update(
        valid=True,
        recommended=[list(y) for y in recommended],
        existing=[list(y) for y in existing],
        is_existing=recommended == existing,
        alternatives=len(alternatives),
        existing_included=existing == recommended or existing in alternatives,
        confidence=decision.get("confidence"),
        weights=decision.get("weights"),
        limits=decision.get("limits"),
        from_tied_set=decision.get("recommended", {}).get("from_tied_set"),
        override_reason=decision.get("recommended", {}).get("override_reason"),
        evidence_items=len(items),
        sourced_share=(sum(s.startswith(("card:", "sim:", "brief:")) for s in sources) / len(sources))
        if sources else None,
        own_knowledge_items=sum("own-knowledge" in s for s in sources),
        cards_cited=cited,
        cards_missing=[c for c in cited if c not in cards],
        cards_not_opened=[c for c in cited if c in cards and c not in opened],
        sim_refs_beyond_calls=[n for n in sims if n > trace["simulate_calls"]],
        report=(run_dir / "work" / "REPORT.md").exists(),
    )
    return result


SUBSCRIPTION_PREFIXES = ("anthropic/", "openai/")


class SubscriptionModelError(ValueError):
    """The model needs a subscription-login adapter, which is not part of this repository."""


def auth_module(model):
    """None for API-key providers; subscription-login models are refused."""
    if model.startswith(SUBSCRIPTION_PREFIXES):
        raise SubscriptionModelError(
            f"{model} needs a subscription-login adapter, which is not part of this repository. "
            "Use an API-key model such as opencode-go/gpt-6-luna and set OPENCODE_GO_API_KEY.")


def run_one(job, args, brief, token, key):
    auth_module(job["model"])
    return _run_one(job, args, brief, token, key)


def _run_one(job, args, brief, token, key):
    case, setup, model, rep = job["case"], job["setup"], job["model"], job["rep"]
    name = f"{case['id']}__{setup}__{slug(model)}__r{rep}"
    run_dir = args.out / name
    meta_path = run_dir / "meta.json"
    if meta_path.exists() and not args.redo:
        return json.loads(meta_path.read_text())
    cards = prepare(run_dir, case, setup, brief, args.evidence_root, args.agent_prompt)
    env = dict(os.environ, OPENCODE_API_KEY=key, FARMSIM_URL=args.container_service,
               FARMSIM_FARM=job["farm_id"], FARMSIM_TOKEN=token,
               FIELDSHIFT_PASS_ENV="OPENCODE_API_KEY FARMSIM_URL FARMSIM_FARM FARMSIM_TOKEN",
               FIELDSHIFT_CONTAINER_NAME=f"fieldshift-{slug(name).lower()}")
    if setup != "full":
        env["FIELDSHIFT_PASS_ENV"] = "OPENCODE_API_KEY"
    command = [str(VM / "oc.sh"), str(run_dir), "run", "--standalone", "--agent", "farm-advisor",
               "--model", model, "--format", "json", "--auto", brief + TASK]
    started = time.time()
    status = "completed"
    with open(run_dir / "events.jsonl", "w") as out, open(run_dir / "stderr.log", "w") as err:
        try:
            code = subprocess.run(command, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                  timeout=args.timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            code, status = None, "timeout"
            subprocess.run(["docker", "--context", os.environ.get("FIELDSHIFT_DOCKER_CONTEXT", "colima-rosetta"),
                            "stop", env["FIELDSHIFT_CONTAINER_NAME"]], capture_output=True, check=False)
    trace = parse_events(run_dir / "events.jsonl")
    try:
        transcript = transcripts.export(run_dir)
    except (sqlite3.Error, OSError, json.JSONDecodeError) as error:
        transcript = {"error": str(error)}
    meta = {"run": name, "case": case["id"], "setup": setup, "model": model, "rep": rep, "farm_id": job["farm_id"],
            "status": status, "exit_code": code, "seconds": round(time.time() - started, 1),
            "cards_available": len(cards), "agent_prompt": args.agent_prompt, "trace": {k: v for k, v in trace.items() if k != "read_paths"},
            "cards_opened": sorted({Path(p).stem for p in trace["read_paths"] if "evidence/" in p} - {"INDEX"}),
            "transcript": transcript, "score": score(run_dir, cards, trace)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def rescore(runs):
    """Recompute `score` from saved files only; the first earlier score is kept as `score_before_rescore`."""
    for meta_path in sorted(Path(runs).glob("*/meta.json")):
        run_dir = meta_path.parent
        if run_dir.name.startswith("_"):
            continue
        meta = json.loads(meta_path.read_text())
        cards = {p.stem for p in (run_dir / "work" / "evidence").glob("*.md")} - {"INDEX"}
        trace = parse_events(run_dir / "events.jsonl")
        new = score(run_dir, cards, trace)
        meta.setdefault("score_before_rescore", meta["score"])
        meta["score"] = new
        meta_path.write_text(json.dumps(meta, indent=2))
        before = meta["score_before_rescore"]
        if {k: before.get(k) for k in ("cards_missing", "cards_not_opened")} != {
                k: new.get(k) for k in ("cards_missing", "cards_not_opened")}:
            print(run_dir.name, "cards_missing", before.get("cards_missing"), "->", new.get("cards_missing"),
                  "| not_opened", before.get("cards_not_opened"), "->", new.get("cards_not_opened"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", type=Path, help="re-score saved runs in this directory and exit")
    parser.add_argument("--cases", type=Path, default=VM / "pilot_cases.json")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--setups", nargs="*", choices=SETUPS, help="override the setups listed in the cases")
    parser.add_argument("--model", default="opencode-go/deepseek-v4.1-flash")
    parser.add_argument("--reps", type=int, help="override the repeats listed in the cases")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=Path.home() / ".opencode-vm" / "fieldshift" / "runs" / "a1")
    parser.add_argument("--evidence-root", type=Path, default=REPO / "data" / "evidence")
    parser.add_argument("--service", default="http://127.0.0.1:8601")
    parser.add_argument("--container-service", default="http://host.docker.internal:8601")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--agent-prompt", default="farm-advisor.md", help="prompt file in vm/agents/")
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore)
        return
    token = os.environ["FARMSIM_TOKEN"]
    auth_module(args.model)
    key = os.environ["OPENCODE_GO_API_KEY"]
    cases = [c for c in json.loads(args.cases.read_text()) if not args.only or c["id"] in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    ids = farm_ids(cases, args.out)
    briefs = {c["id"]: http(f"{args.service}/farm/{ids[c['id']]}/brief", token) for c in cases}
    jobs = [{"case": c, "setup": s, "model": args.model, "rep": r, "farm_id": ids[c["id"]]}
            for c in cases for s in (args.setups or c.get("setups", ["full"]))
            for r in range(args.reps or c.get("reps", 1))]
    print(f"{len(jobs)} runs -> {args.out}", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        for meta in pool.map(lambda j: run_one(j, args, briefs[j["case"]["id"]], token, key), jobs):
            s = meta["score"]
            print(meta["run"], meta["status"], f"{meta['seconds']}s", "sim", meta["trace"]["simulate_calls"],
                  "cards", len(meta["cards_opened"]), s.get("recommended") if s["valid"] else s.get("problem"),
                  flush=True)


if __name__ == "__main__":
    main()
