"""Planted-fabrication test for scorer v2 number grounding (offline; writes only to --out).

Copies of real decisions get one invented number each: a number the classifier found in the cited output
is replaced by a value 15-50% away at the same displayed precision, and the item's `numeric_values` are
dropped (prompt v2 makes them optional). A fabrication is detected if that token is classified
`unfounded`. Run directories are only read.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from fieldshift.bench.common import RUNS
from fieldshift.bench.grounding import call_requests, numeric_grounding
from fieldshift.bench.score import call_outputs


def completed_runs(root, repeat=0):
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for run in sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            try:
                meta = json.loads((run / "meta.json").read_text())
                decision = json.loads((run / "work/DECISION.json").read_text())
            except (OSError, ValueError):
                continue
            if meta.get("status") == "completed" and meta.get("exit_code") == 0 and meta.get("rep") == repeat:
                yield model_dir.name, run, decision


def fabricate(value, unit, rng):
    for _ in range(100):
        factor = 1 + rng.choice((-1, 1)) * rng.uniform(.15, .5)
        new = round(value * factor / unit) * unit
        if abs(new - value) >= unit:
            return new
    return value + 3 * unit


def render(value, unit):
    decimals = max(0, -round(math.log10(unit)))
    return f"{value:.{decimals}f}"


def plant(root, out, n=20, seed=2026):
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for model, run, decision in completed_runs(root):
        outputs, _, _ = call_outputs(run)
        requests, _ = call_requests(run)
        result = numeric_grounding(run, decision, outputs, requests)
        for number in result["numbers"]:
            if number["class"] == "found" and number["value"] != 0:
                candidates[model].append((run, decision, number))
    models = sorted(candidates)
    picks, used = [], set()
    for _ in range(100 * n):
        if len(picks) == n:
            break
        model = models[len(picks) % len(models)]
        run, decision, number = rng.choice(candidates[model])
        key = (run, number["evidence"], number["context"])
        if key in used:
            continue
        used.add(key)
        picks.append((model, run, decision, number))
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for k, (model, run, decision, number) in enumerate(picks):
        planted = json.loads(json.dumps(decision))
        item = planted["evidence"][number["evidence"]]
        new_value = fabricate(number["value"], number["unit"], rng)
        token = render(new_value, number["unit"])
        claim = str(item["claim"])
        start = claim.find(number["context"].split("⟨")[1].split("⟩")[0])
        item["claim"] = claim[:start] + token + claim[start + len(number["token"]):]
        item.pop("numeric_values", None)
        (out / f"{k:02d}__{run.name}.DECISION.json").write_text(json.dumps(planted, indent=1))
        outputs, _, _ = call_outputs(run)
        requests, _ = call_requests(run)
        result = numeric_grounding(run, planted, outputs, requests)
        hit = [r for r in result["numbers"] if r["evidence"] == number["evidence"] and r["token"] == token]
        label = hit[0]["class"] if hit else "not_tokenised"
        records.append({"k": k, "model": model, "run": run.name, "evidence": number["evidence"],
                        "original": number["token"], "planted": token, "quantity": number["quantity"],
                        "class": label, "detected": label == "unfounded",
                        "matched": hit[0].get("via") if hit else None, "claim": item["claim"]})
    summary = {"seed": seed, "n": len(records), "detected": sum(r["detected"] for r in records),
               "by_class": dict(Counter(r["class"] for r in records)),
               "unit_stated": {"n": sum(r["quantity"] is not None for r in records),
                               "detected": sum(r["detected"] for r in records if r["quantity"] is not None)},
               "records": records}
    (out / "planted_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, required=True, help="Scratch directory, never a run directory")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.runs.resolve()):
        parser.error("--out must not be inside the run directory")
    summary = plant(args.runs, args.out, args.n, args.seed)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    for r in summary["records"]:
        print(f"{r['k']:2d} {r['model'][:22]:22s} {r['original']:>9s} -> {r['planted']:>9s} "
              f"{r['quantity']!s:11s} {r['class']:11s} {r['matched'] or ''}")


if __name__ == "__main__":
    main()
