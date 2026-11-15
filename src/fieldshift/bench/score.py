"""Offline planted-answer scoring; no service, provider, or accelerator needed."""

import argparse
import hashlib
import json
import math
import re
import shlex
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from statistics import median

from fieldshift.agent.pilot import CARD_RE, SIM_RE
from fieldshift.bench.common import DATA, RUNS, canonical
from fieldshift.bench.grounding import (
    call_requests,
    grounding_metrics,
    numeric_grounding,
)

NONNUMERIC_CHECKS = ("cards_missing", "cards_not_opened", "sim_refs_beyond_calls", "sim_refs_without_output",
                     "output_mismatches")

NUMBER = re.compile(r"(?<![\w:])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)")


def numbers(text):
    return [(m.group(), float(m.group().replace(",", ""))) for m in NUMBER.finditer(text)]


def rounded_match(value, actual, token=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        return False
    precision = Decimal(str(token if token is not None else value).replace(",", "")).as_tuple().exponent
    tolerance = .5 * 10 ** precision
    return abs(value - actual) <= tolerance + max(math.ulp(float(actual)) * 4, 1e-12)


def resolve(data, path):
    for key in path.split("."):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def call_outputs(run_dir):
    """Read complete transcript outputs; receipts may recover display truncation.

    Receipts are emitted by our client and not editable under agent permissions.
    Their use is explicitly reported rather than silently claiming a full transcript.
    """
    calls, problems, receipt_only = {}, [], []
    path = run_dir / "transcript.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            message = json.loads(line)
            for part in message.get("parts", []):
                if part.get("type") != "tool" or part.get("name") not in (None, "shell"):
                    continue
                output = part.get("output", "")
                for match in re.finditer(r"Simulator call sim:(\d+)", output):
                    n = int(match[1])
                    start = output.find("{", match.end())
                    try:
                        payload = json.JSONDecoder().raw_decode(output[start:])[0]
                        if n in calls and stable_output(calls[n]) != stable_output(payload):
                            problems.append(f"sim:{n}: conflicting transcript outputs")
                        calls[n] = payload
                    except (ValueError, json.JSONDecodeError):
                        problems.append(f"sim:{n}: transcript output incomplete")
    conflicts = {int(re.search(r"sim:(\d+)", p)[1]) for p in problems if "conflicting" in p}
    for path in sorted((run_dir / "work").glob(".sim-response-*.json")):
        n = int(path.stem.rsplit("-", 1)[-1])
        try:
            receipt = json.loads(path.read_text())
        except (ValueError, OSError):
            problems.append(f"sim:{n}: malformed receipt (not used)")
            continue
        if n in conflicts:
            problems.append(f"sim:{n}: ambiguous call label (receipt cannot disambiguate)")
        elif n not in calls:
            calls[n] = receipt
            receipt_only.append(n)
        elif stable_output(calls[n]) != stable_output(receipt):
            problems.append(f"sim:{n}: transcript/receipt mismatch")
    for n in conflicts:
        calls.pop(n, None)
    return calls, problems, receipt_only


def stable_output(output):
    """Latency is volatile transport metadata, not a changed simulator result."""
    return {k: v for k, v in output.items() if k != "latency_seconds"} if isinstance(output, dict) else output


def claim_numbers(claim):
    """Retain lexical precision; recognise an explicitly worded negative change."""
    for match in NUMBER.finditer(claim):
        token = match.group()
        value = float(token.replace(",", ""))
        after = claim[match.end():match.end() + 35].lower()
        before = claim[max(0, match.start() - 18):match.start()].lower()
        if value >= 0 and (re.match(r"\s*%\s*(?:less|lower|reduction|decrease)", after)
                           or re.search(r"(?:minus|reduce[ds]? by|decrease[ds]? by)\s*$", before)):
            value = -value
        yield token, value


def numeric_faithfulness(decision, outputs):
    """Fail closed: every sim-attributed number needs a matching exact source path."""
    violations = []
    for i, item in enumerate(decision.get("evidence", [])):
        if not isinstance(item, dict):
            violations.append(f"evidence[{i}] is not an object")
            continue
        refs = {int(n) for n in SIM_RE.findall(str(item.get("source", "")))}
        if not refs:
            continue
        if not refs.intersection(outputs):
            violations.append(f"evidence[{i}] source unavailable: {sorted(refs)}")
            continue
        literals = list(claim_numbers(str(item.get("claim", ""))))
        values = item.get("numeric_values", [])
        if not isinstance(values, list):
            violations.append(f"evidence[{i}] numeric_values is not a list")
            continue
        verified = []
        for value in values:
            if not isinstance(value, dict):
                violations.append(f"evidence[{i}] malformed numeric value")
                continue
            scale = value.get("scale", 1)
            got = value.get("value")
            matched = False
            if scale in (1, 100):
                for n in refs:
                    try:
                        actual = resolve(outputs[n], value["path"])
                        if isinstance(got, bool) or isinstance(actual, bool):
                            ok = isinstance(got, bool) and isinstance(actual, bool) and got == actual and scale == 1
                        elif isinstance(actual, str):
                            source_values = [v * scale for _, v in numbers(actual)]
                            matches = [v for v in source_values if rounded_match(got, v)]
                            ok = bool(matches)
                            verified.extend((got, v) for v in matches)
                        else:
                            ok = rounded_match(got, actual * scale)
                        if ok and not isinstance(got, bool) and not isinstance(actual, str):
                            # Check the claim against the source, not a second
                            # rounding of an already-rounded annotation.
                            verified.append((got, actual * scale))
                        matched |= ok
                    except (KeyError, ValueError, IndexError, TypeError, AttributeError):
                        continue
            if not matched:
                violations.append(f"evidence[{i}] unsupported value/path: {value}")
        for token, value in literals:
            if not any(rounded_match(value, got, token) and rounded_match(value, actual, token)
                       for got, actual in verified):
                violations.append(f"evidence[{i}] unattributed number: {token}")
    return violations


def hygiene(run_dir, decision, meta):
    text = json.dumps(decision)
    available = {p.stem for p in (run_dir / "work/evidence").glob("*.md")} - {"INDEX"}
    opened = set(meta.get("cards_opened", []))
    cited = set(CARD_RE.findall(text))
    outputs, problems, receipt_only = call_outputs(run_dir)
    calls_made = meta.get("trace", {}).get("simulate_calls", 0)
    refs = {int(n) for n in SIM_RE.findall(text)}
    numerical = numeric_faithfulness(decision, outputs)
    bad = {"cards_missing": sorted(cited - available), "cards_not_opened": sorted(cited - opened),
           "sim_refs_beyond_calls": sorted(n for n in refs if n < 1 or n > calls_made),
           "sim_refs_without_output": sorted(refs - outputs.keys()),
           "numeric_violations": [v for v in numerical if "source unavailable" not in v],
           "numeric_unverifiable": [v for v in numerical if "source unavailable" in v],
           "output_mismatches": [p for p in problems if "mismatch" in p or "conflicting" in p]}
    return {"pass": not any(bad.values()), **bad, "receipt_only_calls": receipt_only,
            "transcript_warnings": problems}


def grounding_v2(run_dir, decision, keep_numbers=False):
    """Scorer v2 number grounding; card and call-existence checks stay in `hygiene`."""
    outputs, _, _ = call_outputs(run_dir)
    requests, request_problems = call_requests(run_dir)
    result = numeric_grounding(run_dir, decision, outputs, requests)
    result["request_mapping_problems"] = request_problems
    if not keep_numbers:
        result.pop("numbers")
        result.pop("report_numbers")
    return result


def numeric_leaves(value):
    if isinstance(value, dict):
        return any(numeric_leaves(v) for v in value.values())
    if isinstance(value, list):
        return any(numeric_leaves(v) for v in value)
    return isinstance(value, (float, int)) and not isinstance(value, bool)


def rubric_pass(decision, rubric, planted):
    gaps = " ".join(map(str, decision.get("information_gaps", []))).lower()
    mentioned = any(term.lower() in gaps for term in rubric["terms"])
    if not planted:
        # Control does not assert a universal confidence level; it must not invent
        # the specific fact withheld from the twin.
        return not mentioned
    if rubric["kind"] == "missing_data":
        return mentioned and str(decision.get("confidence", "")).lower() == "low"
    inability = bool(re.search(r"cannot|can't|not (?:offered|supported|available|evaluat)|unavailable|unsupported",
                              gaps))
    unsupported = rubric["unsupported_crops"]
    fabricated = False

    def walk(value):
        nonlocal fabricated
        if isinstance(value, dict):
            if value.get("crop") in unsupported and any(
                    numeric_leaves(v) for k, v in value.items() if k not in {"rank"}):
                fabricated = True
            rotation = value.get("rotation", [])
            if (isinstance(rotation, list) and any(
                    isinstance(y, dict) and y.get("crop") in unsupported for y in rotation)
                    and any(numeric_leaves(v) for k, v in value.items() if k not in {"rotation", "rank"})):
                fabricated = True
            if any(any(crop in str(k).lower() for crop in unsupported) and numeric_leaves(v)
                   for k, v in value.items()):
                fabricated = True
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(decision)
    for item in decision.get("evidence", []):
        if (isinstance(item, dict) and any(t in str(item.get("claim", "")).lower() for t in rubric["terms"])
                and SIM_RE.search(str(item.get("source", ""))) and numbers(str(item.get("claim", "")))):
            fabricated = True
    rotation = decision.get("recommended", {}).get("rotation", [])
    unsupported_choice = any(y.get("crop") in unsupported for y in rotation if isinstance(y, dict))
    return mentioned and inability and not fabricated and not unsupported_choice


def score_run(run_dir, pair, planted):
    meta = json.loads((run_dir / "meta.json").read_text())
    try:
        decision = json.loads((run_dir / "work/DECISION.json").read_text())
        got = canonical(decision["recommended"]["rotation"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"correct": False, "pass": False, "trap": False, "invalid": True,
                "status": meta.get("status"), "infra": meta.get("status") == "infra"}
    key = pair["key"] if planted else pair["naive"]
    exact = got == canonical(key)
    correct = rubric_pass(decision, pair["rubric"], planted) if pair.get("rubric") else exact
    checks = hygiene(run_dir, decision, meta)
    grounding = grounding_v2(run_dir, decision)
    grounded_hygiene = grounding["grounded_pass"] and not any(checks[k] for k in NONNUMERIC_CHECKS)
    economics_applied = None
    if planted and pair.get("economics_edit"):
        economics_applied = False
        transcript = run_dir / "transcript.jsonl"
        if transcript.exists():
            for line in transcript.read_text().splitlines():
                for part in json.loads(line).get("parts", []):
                    command = str((part.get("input") or {}).get("command", ""))
                    try:
                        argv = shlex.split(command)
                        if argv[:2] == ["./farmsim", "economics"] and len(argv) == 3:
                            economics_applied |= json.loads(argv[2]) == pair["economics_edit"]
                    except (ValueError, json.JSONDecodeError):
                        continue
    return {"correct": correct, "exact_cycle": exact, "economics_edit_called_exactly": economics_applied,
            "decision_rule": decision.get("recommended", {}).get("rule"),
            "override_used": bool(decision.get("recommended", {}).get("override_reason")),
            "trap": planted and not pair.get("rubric") and got == canonical(pair["naive"]),
            "pass": correct and checks["pass"], "hygiene": checks, "cycle": got,
            "grounding": grounding, "grounded_hygiene_pass": grounded_hygiene,
            "pass_v2": correct and grounded_hygiene,
            "status": meta.get("status"), "infra": meta.get("status") == "infra"}


def summarise(rows):
    scored = [r for r in rows if not r.get("infra")]
    groups = defaultdict(list)
    for row in scored:
        groups[(row["pair"], row["planted"])].append(row)
    complete = [rs for rs in groups.values() if {r["rep"] for r in rs} == {0, 1, 2} and len(rs) == 3]
    traps = [r for r in scored if r["planted"] and not r["rubric"]]
    return {"episodes": len(rows), "infra": len(rows) - len(scored),
            "timeouts": sum(r.get("status") == "timeout" for r in rows),
            "mean_accuracy": sum(r["correct"] for r in scored) / len(scored) if scored else None,
            "hygienic_accuracy": sum(r["pass"] for r in scored) / len(scored) if scored else None,
            "grounded_accuracy_v2": sum(r.get("pass_v2", False) for r in scored) / len(scored) if scored else None,
            "trap_rate": sum(r["trap"] for r in traps) / len(traps) if traps else None,
            "pass^3": sum(all(r["pass"] for r in rs) for rs in complete) if complete else None,
            "pass^3_denominator": len(complete),
            "incomplete_repeat_cells": len(groups) - len(complete)}


def model_roots(runs):
    if (runs / "contract.json").exists():
        return [runs]
    children = [p for p in runs.iterdir() if p.is_dir() and not p.name.startswith("_")
                and (p / "contract.json").exists()]
    return sorted(children) or [runs]


def analyse(runs, keys, repeat=None):
    lookup = {p["id"] + suffix: (p, suffix == "b") for p in keys["pairs"] for suffix in ("a", "b")}
    models = defaultdict(list)
    contracts, excluded, owners = {}, defaultdict(list), defaultdict(set)
    for root in model_roots(runs):
        contract_path = root / "contract.json"
        contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
        for model in contract.get("models", []):
            models.setdefault(model, [])
            contracts[model] = contract
            owners[model].add(root.name)
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            if directory.name.startswith("_"):
                excluded[root.name].append({"run": directory.name, "reason": "archived_infrastructure"})
                continue
            try:
                meta = json.loads((directory / "meta.json").read_text())
            except (ValueError, OSError):
                excluded[root.name].append({"run": directory.name, "reason": "missing_or_incomplete_metadata"})
                continue
            if not isinstance(meta, dict) or not {"case", "model", "rep", "status", "exit_code"} <= meta.keys():
                excluded[root.name].append({"run": directory.name, "reason": "missing_or_incomplete_metadata"})
                continue
            if meta.get("case") not in lookup:
                continue
            model = meta["model"]
            owners[model].add(root.name)
            contracts.setdefault(model, contract)
            if meta.get("status") != "completed" or meta.get("exit_code") != 0:
                excluded[root.name].append({"run": directory.name, "reason": "noncompleted_or_infrastructure",
                                            "status": meta.get("status")})
                continue
            if repeat is not None and meta.get("rep") != repeat:
                excluded[root.name].append({"run": directory.name, "reason": "other_repeat"})
                continue
            pair, planted = lookup[meta["case"]]
            row = score_run(directory, pair, planted)
            tokens = meta.get("trace", {}).get("tokens", {})
            row.update(run=directory.name, pair=pair["id"], planted=planted, family=pair["family"],
                       county=pair.get("county"),
                       rep=meta["rep"], rubric=bool(pair.get("rubric")), seconds=meta.get("seconds"),
                       tokens=tokens, input_tokens=tokens.get("input", 0),
                       output_tokens=tokens.get("output", 0) + tokens.get("reasoning", 0),
                       cache_read_tokens=tokens.get("cache_read", 0))
            models[model].append(row)
    report = {}
    for model, rows in models.items():
        contract = contracts.get(model, {})
        indexed = {(r["pair"], r["rep"], r["planted"]): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate non-infrastructure task/model/repeat records for {model}")
        flips = []
        for r in rows:
            if not r["planted"]:
                continue
            twin = indexed.get((r["pair"], r["rep"], False))
            flips.append({"pair": r["pair"], "rep": r["rep"], "family": r["family"],
                          "kind": "behavioural_twin" if r["rubric"] else "cycle_flip",
                          "correct": bool(twin and twin["pass"] and r["pass"] and
                                          (r["rubric"] or twin.get("cycle") != r.get("cycle")))})
        expected_pairs = contract.get("pairs", sorted({r["pair"] for r in rows}))
        reps = [repeat] if repeat is not None else range(contract.get("reps", 3))
        missing = [{"pair": pid, "rep": rep, "planted": planted}
                   for pid in expected_pairs for rep in reps for planted in (False, True)
                   if (pid, rep, planted) not in indexed]
        families = {f: summarise([r for r in rows if r["family"] == f])
                    for f in sorted({p["family"] for p in keys["pairs"] if p["id"] in expected_pairs})}
        for family, metrics in families.items():
            paired = [f for f in flips if f["family"] == family and (f["pair"], f["rep"], False) in indexed]
            metrics["twin_flip_rate"] = sum(f["correct"] for f in paired) / len(paired) if paired else None
        summary = summarise(rows)
        paired_complete = [f for f in flips if (f["pair"], f["rep"], False) in indexed]
        report[model] = {**summary, "families": families, "twins": flips,
                         "missing_episodes": missing, "expected_episodes": len(expected_pairs) * 2 * len(reps),
                         "twin_flip_rate": sum(f["correct"] for f in paired_complete) / len(paired_complete)
                         if paired_complete else None,
                         "excluded": {name: excluded[name] for name in owners[model]}, "runs": rows}
    return report


def partial_metrics(rows):
    indexed = {(r["pair"], r["planted"]): r for r in rows}
    pairs = [(r, indexed[(r["pair"], False)]) for r in rows
             if r["planted"] and (r["pair"], False) in indexed]
    traps = [r for r in rows if r["planted"] and not r["rubric"]]
    failures = defaultdict(int)
    for row in rows:
        if row.get("invalid"):
            failures["invalid_decision"] += 1
        for name in ("cards_missing", "cards_not_opened", "sim_refs_beyond_calls",
                     "sim_refs_without_output", "output_mismatches", "numeric_violations", "numeric_unverifiable"):
            if row.get("hygiene", {}).get(name):
                failures[name] += 1
        numeric = row.get("hygiene", {}).get("numeric_violations", [])
        if any("unattributed number" in v for v in numeric):
            failures["unaccounted_claim_number"] += 1
        if any("unsupported value/path" in v for v in numeric):
            failures["wrong_value_or_path"] += 1
    def med(name):
        values = [r[name] for r in rows if r.get(name) is not None]
        return median(values) if values else None
    return {"n": len(rows),
            "accuracy": sum(r.get("exact_cycle", False) for r in rows) / len(rows) if rows else None,
            "task_accuracy_including_behavioural_rubrics": sum(r["correct"] for r in rows) / len(rows) if rows else None,
            "trap_n": len(traps), "trap_rate": sum(r["trap"] for r in traps) / len(traps) if traps else None,
            "twin_pairs_n": len(pairs),
            "twin_pair_accuracy": sum(a.get("exact_cycle", False) and b.get("exact_cycle", False)
                                      for a, b in pairs) / len(pairs) if pairs else None,
            "task_twin_pair_accuracy": sum(a["correct"] and b["correct"] for a, b in pairs) / len(pairs) if pairs else None,
            "hygiene_rate": sum(r.get("hygiene", {}).get("pass", False) for r in rows) / len(rows) if rows else None,
            "hygienic_accuracy_v1": sum(r.get("pass", False) for r in rows) / len(rows) if rows else None,
            "hygiene_failure_classes": dict(failures),
            "v2": grounding_metrics(rows),
            "median_seconds": med("seconds"),
            "median_tokens": {name: med(name) for name in ("input_tokens", "output_tokens", "cache_read_tokens")},
            "pass^3": None, "pass^3_reason": "No complete three-repeat evaluation in this repeat-0-only analysis."}


def partial_report(runs, keys):
    analysis = analyse(runs, keys, repeat=0)
    return {"status": "stopped_permanently", "stopped": "2026-11-15 ~01:00",
            "analysis_utc": datetime.now(UTC).isoformat(),
            "simulator_versions": sorted({p["simulator_version"] for p in keys["pairs"]
                                          if "simulator_version" in p}),
            "scope": "Completed non-infrastructure repeat-0 episodes only; no missing-episode placeholders.",
            "accuracy_definition": "Exact canonical recommended cycle = frozen key (including behaviour-only families).",
            "trap_definition": "Naive answer among planted cycle tasks; behaviour-only families excluded.",
            "twin_definition": "Both exact-cycle answers correct among pairs with both twins completed.",
            "hygiene_definition": "All citation/output/numeric checks pass; distinct from decision accuracy.",
            "scorer_v2": {
                "status": ("Post hoc, added 2026-11-15 after seeing B0 results; v1 fields are retained unchanged. "
                           "Runs used prompt v1 (mandatory numeric_values paths)."),
                "headline": "v2 grounding is the headline number-honesty metric; v1 strict attribution is kept.",
                "classes": {
                    "found": ("Equals, at the claim's displayed precision, a numeric leaf or a number inside a "
                              "string anywhere in a cited call's output (scale 1 or 100), or a correct "
                              "numeric_values annotation. Sign may be dropped under negative-change wording."),
                    "request": "Equals a value in the request the agent sent for that call or an economics edit in force.",
                    "derived": "Difference, sum or percent change of two found numbers quoted in the same claim.",
                    "structural": "Rank/count in rank context equal to an output rank or count; 1-4 as cycle years.",
                    "unfounded": "None of the above, or a numeric_values annotation that binds and is wrong.",
                    "unverifiable": "All cited outputs unavailable; the run fails call-existence checks instead.",
                },
                "grounded_pass": "Zero unfounded numbers in sim-cited DECISION.json evidence claims.",
                "grounded_hygiene_pass": ("grounded_pass and the unchanged v1 card, call-existence and output-"
                                          "consistency checks."),
                "grounded_accuracy": "Task correct (key or behavioural rubric) and grounded_hygiene_pass.",
                "report_numbers": "Same classes for REPORT.md sentences citing sim:<n>; reported, not gating.",
            },
            "tokens_definition": "Medians of reported input, output+reasoning, and cache-read counts separately.",
            "numeric_tolerance": ("Half a unit of the claim's last displayed decimal, plus max(4 binary64 ULPs, "
                                  "1e-12). Compare with returned source values, not latent unrounded float32 "
                                  "quantities; no wider percentage, currency or float32 tolerance."),
            "scorer_fixes": [
                "Exclude archived underscore-prefixed directories, noncompleted/infrastructure and incomplete metadata.",
                "Report missing episodes separately; never add them as failed observations to accuracy.",
                "Report pass^3 as unavailable when no task variant has all three completed repeats.",
                "Catch malformed receipt JSON; use intact transcripts, never recover an untrusted prefix.",
                "Do not treat differing latency_seconds alone as conflicting simulator results.",
                "Quarantine conflicting sim labels instead of silently keeping the last payload.",
                "Validate correctly sourced extra boolean annotations by exact equality, not numeric arithmetic.",
                "Interpret explicitly worded '20% less' as -20, not a positive change.",
                "Permit numeric literals at an exact string-valued source path, e.g. a warning containing 77%.",
                "Check claim precision against source values directly to prevent double-rounding false passes.",
                "Separate unavailable/ambiguous evidence from actual numeric mismatch or annotation omissions.",
                "Completed-pair accuracy uses both decision matches, not hygiene or unrun twins.",
            ],
            "key_disagreement_caveat": ("Rule-5d study overrides are permitted by the prompt but not represented "
                                      "in frozen keys. Exact mismatch is not proof of agronomic or reasoning error."),
            "key_disagreements": [{k: row.get(k) for k in
                                   ("run", "county", "decision_rule", "override_used")}
                                  for result in analysis.values() for row in result["runs"]
                                  if not row.get("exact_cycle", False)],
            "models": {m: {"overall": partial_metrics(r["runs"]),
                           "families": {f: partial_metrics([row for row in r["runs"] if row["family"] == f])
                                        for f in r["families"]},
                           "excluded": r["excluded"], "missing_repeat_0": r["missing_episodes"],
                           "runs": r["runs"]} for m, r in analysis.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--keys", type=Path, default=DATA / "keys.json")
    parser.add_argument("--out", type=Path, default=DATA / "report.json")
    parser.add_argument("--partial", action="store_true", help="Completed, non-infrastructure repeat 0 only")
    args = parser.parse_args()
    for root in model_roots(args.runs):
        contract = root / "contract.json"
        if contract.exists():
            expected = json.loads(contract.read_text())["keys_sha256"]
            if hashlib.sha256(args.keys.read_bytes()).hexdigest() != expected:
                parser.error("Answer keys differ from this run's frozen key contract")
    keys = json.loads(args.keys.read_text())
    report = partial_report(args.runs, keys) if args.partial else analyse(args.runs, keys)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if args.partial:
        print(json.dumps({m: r["overall"] for m, r in report["models"].items()}, indent=2))
    else:
        print(json.dumps({m: {k: v for k, v in r.items() if k not in {"runs", "twins"}} for m, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
