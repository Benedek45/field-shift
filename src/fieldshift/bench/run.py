"""Run planted twins through the unchanged pilot machinery (new prompt/client).

uv run python -m fieldshift.bench.run --smoke
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from fieldshift.agent import farms, pilot, transcripts
from fieldshift.agent.scoreboard import _fingerprint
from fieldshift.bench.common import DATA, ROOT, RUNS, local_service

ORIGINAL_PREPARE = pilot.prepare
PROVIDER_LIMIT = re.compile(r"provider\.go-limit|GoUsageLimitError|rate_limit_error|overloaded_error"
                            r'|"status":\s*(429|529)')


def anonymous_brief(text):
    """Remove location-bearing and year-mangled source URLs from blind briefs."""
    text = re.sub(r"(?ms)^## Weather outlook\s.*?(?=^## |\Z)",
                  "## Weather outlook\n\nNot included in this benchmark; see Open-Meteo / ECMWF SEAS5 separately.\n\n", text)
    return re.sub(r"https?://[^\s<>|)\]]+", "[source URL withheld in blind brief]", text)


def prepare(*args, **kwargs):
    cards = ORIGINAL_PREPARE(*args, **kwargs)
    run_dir = args[0]
    shutil.copy(ROOT / "vm/farmsim", run_dir / "work/.farmsim-client")
    (run_dir / "work/.farmsim-client").chmod(0o755)
    shutil.copy(ROOT / "bench/farmsim", run_dir / "work/farmsim")
    (run_dir / "work/farmsim").chmod(0o755)
    prompt = (ROOT / "vm/agents/farm-advisor-a3.md").read_text()
    (run_dir / "work/PROCEDURE.md").write_text(prompt.split("---", 2)[-1].strip() + "\n")
    return cards


def cases_for(pairs):
    """Only legitimate farm inputs; no answers, family labels or expected limits."""
    cases = []
    for pair in pairs:
        for suffix, rec in (("a", pair["base"]), ("b", pair["planted"])):
            cases.append({"id": pair["id"] + suffix, "record": rec})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=Path, default=DATA / "keys.json")
    parser.add_argument("--models", nargs="+", default=["opencode-go/deepseek-v4.1-flash"])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--port", type=int, default=8611)
    parser.add_argument("--smoke", action="store_true", help="One pair, one repeat, DeepSeek only")
    parser.add_argument("--pair", help="Optional opaque pair ID")
    parser.add_argument("--per-family", type=int, default=3, help="First N fixed pairs per family; 0 = all")
    args = parser.parse_args()
    load_dotenv(ROOT / ".agent-env", override=False)
    keys = json.loads(args.keys.read_text())
    pairs = [p for p in keys["pairs"] if not args.pair or p["id"] == args.pair]
    if not args.pair and args.per_family:
        selected, counts = [], {}
        for pair in pairs:
            family = pair["family"]
            if counts.get(family, 0) < args.per_family:
                selected.append(pair)
                counts[family] = counts.get(family, 0) + 1
        pairs = selected
    if args.smoke:
        pairs = [next(p for p in pairs if p["family"] == "hidden_limit")]
        args.models, args.reps = ["opencode-go/deepseek-v4.1-flash"], 1
        args.out = RUNS.with_name("bench_smoke")
    if not pairs:
        parser.error("No matching pairs")
    fingerprint = _fingerprint()
    if any(p["fingerprint"] != fingerprint for p in pairs):
        parser.error("Simulator/data fingerprint changed; regenerate keys with --recompute")
    if any(not p.get("rubric") and p["board"].get("all_pool_unique") is not True for p in pairs):
        parser.error("Legacy keys lack full-pool uniqueness certificates; regenerate with --recompute")
    if args.reps < 1 or args.parallel < 1:
        parser.error("reps and parallel must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out / ".runner.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A benchmark runner already owns this output directory")
    args.evidence_root, args.agent_prompt, args.redo = ROOT / "data/evidence", "farm-advisor-bench.md", False
    args.container_service = f"http://host.docker.internal:{args.port}"
    cases = cases_for(pairs)
    # Pilot's exporter normally truncates tool outputs at 4,000 chars; this benchmark
    # must retain complete numeric evidence. Bounds are checked by score.py.
    transcripts.OUTPUT_CHARS = 10_000_000
    pilot.prepare = prepare
    receipt = {"keys_sha256": hashlib.sha256(args.keys.read_bytes()).hexdigest(),
               "pairs": [p["id"] for p in pairs], "models": args.models, "reps": args.reps,
               "fingerprint": fingerprint, "smoke": args.smoke,
               "harness_version": "OpenCode 2.3.0",
               "image_id": subprocess.check_output(
                   ["docker", "--context", os.environ.get("FIELDSHIFT_DOCKER_CONTEXT", "colima-rosetta"),
                    "image", "inspect", os.environ.get("FIELDSHIFT_IMAGE", "fieldshift-agent:oc2.3.0-arm64"),
                    "--format", "{{.Id}}"], text=True).strip(),
               "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("vm/oc.sh", "vm/agents/farm-advisor-a3.md",
                                              "vm/agents/farm-advisor-bench.md", "vm/farmsim", "bench/farmsim")}}
    receipt_path = args.out / "contract.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        parser.error("Output directory contains a different run contract; choose a fresh --out")
    receipt_path.write_text(json.dumps(receipt, indent=2))
    with local_service(args.port) as (runtime, token):
        registered = {c["id"]: farms.add(c["record"], runtime.root) for c in cases}
        briefs = {c["id"]: anonymous_brief(pilot.http(
            f"http://127.0.0.1:{args.port}/farm/{registered[c['id']]}/brief", token)) for c in cases}
        jobs = [{"case": {"id": c["id"]}, "setup": "full", "model": model, "rep": rep,
                 "farm_id": registered[c["id"]]}
                for model in args.models for rep in range(args.reps) for c in cases]
        print(f"{len(jobs)} episodes -> {args.out}", flush=True)

        stopped = threading.Event()
        retry_after = []

        def execute_job(job):
            if stopped.is_set():
                return None
            if _fingerprint() != fingerprint:
                raise RuntimeError("Simulator/data changed during run; preserve existing episodes and regenerate keys")
            meta = pilot.run_one(job, args, briefs[job["case"]["id"]], token, os.environ["OPENCODE_GO_API_KEY"])
            errors = " ".join(meta.get("trace", {}).get("errors", []))
            if PROVIDER_LIMIT.search(errors):
                # Quota exhaustion is infrastructure: keep the evidence, free the slot for a rerun.
                run_dir = args.out / meta["run"]
                run_dir.rename(args.out / f"_infra_provider_limit_{meta['run']}_{int(time.time())}")
                retry_after.extend(int(m) for m in re.findall(r'"retryAfterMs":\s*(\d+)', errors))
                stopped.set()
                meta = {**meta, "status": "infra_provider_limit"}
            return meta

        with ThreadPoolExecutor(args.parallel) as pool:
            for meta in pool.map(execute_job, jobs):
                if meta is not None:
                    print(meta["run"], meta["status"], meta["score"].get("recommended"), flush=True)
    print("Service on owned port stopped.", flush=True)
    if stopped.is_set():
        print(f"PROVIDER_LIMIT retry_after_ms={max(retry_after, default=0)}", flush=True)
        raise SystemExit(75)


if __name__ == "__main__":
    main()
