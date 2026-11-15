"""Score pilot A3 (farmer form and personalisation) against the rules registered in EXPERIMENTS.md.

uv run python -m fieldshift.agent.a3_analysis [--runs DIR] [--service URL]
Needs FARMSIM_TOKEN; personalisation and procedure checks call the simulator service.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from fieldshift.agent.a1_analysis import label, modal, paired
from fieldshift.agent.a2_analysis import (
    WEIGHT_KEYS,
    Board,
    citations,
    load,
    procedure,
    weight_vector,
)

RUNS = Path.home() / ".opencode-vm/fieldshift/runs/a3"
CASES = Path(__file__).resolve().parents[3] / "vm" / "a3_cases.json"
FARMS = ("story", "lubbock", "york")
PROFILES = ("p1", "p2", "p3", "p4")
MAIN_METRIC = {"p3": ("soil_carbon_input_t_c_ha_yr", "higher"), "p4": ("field_water_use_mm_yr", "lower")}


def limits_exact(decision, expected):
    got = (decision or {}).get("limits")
    if not isinstance(got, dict):
        return False, "no limits object"
    norm = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in got.items()}
    want = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in expected.items()}
    if norm == want:
        return True, None
    return False, {"got": got, "expected": expected}


def weights_match(decision, expected):
    return weight_vector(decision) == tuple(expected.get(k, 0) for k in WEIGHT_KEYS)


def analyse_model(rows, cases, service):
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["case"]].append(r)
    rep = {"runs": len(rows), "status": dict(Counter(r["status"] for r in rows))}

    cells = {}
    for cid, rs in sorted(by_cell.items()):
        cycles = [r["cycle"] for r in rs if r["cycle"]]
        top, n, counts = modal(cycles) if cycles else (None, 0, Counter())
        cells[cid] = {"modal": top, "modal_count": n, "of": len(rs), "all": {label(k): v for k, v in counts.items()}}
    agree = [c for c, v in cells.items() if v["of"] >= 3 and v["modal_count"] == v["of"]]
    rep["A3-1"] = {"cells_all_reps_agree": len(agree), "of": len(cells),
                   "cells": {c: {k: v for k, v in d.items() if k != "modal"} | {"modal": label(d["modal"]) if d["modal"]
                                                                                else None}
                             for c, d in cells.items()},
                   "pass": len(cells) == 12 and len(agree) >= 10}

    lim = {}
    for r in rows:
        ok, why = limits_exact(r["decision"], cases[r["case"]]["expected_limits"])
        if not ok:
            lim[r["run"]] = why
    share = 1 - len(lim) / len(rows) if rows else 0
    wrong_weights = [r["run"] for r in rows if not weights_match(r["decision"], cases[r["case"]]["weights"])]
    rep["A3-2"] = {"share_exact": round(share, 3), "wrong": lim, "pass": share >= 0.9,
                   "weights_not_copied_from_form": wrong_weights}

    pers = {}
    for farm in FARMS:
        modal_by_profile = {p: cells.get(f"{farm}-{p}", {}).get("modal") for p in PROFILES}
        distinct = len({m for m in modal_by_profile.values() if m})
        checks = {}
        p1 = modal_by_profile["p1"]
        p1_case = by_cell.get(f"{farm}-p1")
        for p, (metric_name, direction) in MAIN_METRIC.items():
            mine = modal_by_profile[p]
            if not (mine and p1 and p1_case):
                checks[p] = {"error": "missing modal"}
                continue
            # Compare both cycles on P1's farm record so economics forks (P2) and limits do not enter.
            c = paired(service, p1_case[0]["farm_id"], mine, p1, metric_name)
            if "error" in c:
                checks[p] = c
                continue
            better = c["a"] > c["b"] if direction == "higher" else c["a"] < c["b"]
            checks[p] = c | {"cycle": label(mine), "p1": label(p1), "better": better,
                             "same_cycle": mine == p1}
        pers[farm] = {"modal": {p: label(m) if m else None for p, m in modal_by_profile.items()},
                      "distinct": distinct, "main_metric": checks,
                      "pass": distinct >= 3 and all(v.get("better") for v in checks.values())}
    rep["A3-3"] = {"farms": pers, "pass": all(v["pass"] for v in pers.values())}

    proc = {r["run"]: procedure(service, r) for r in rows}
    share = sum(p["ok"] for p in proc.values()) / len(proc) if proc else 0
    rep["A3-4"] = {"share": round(share, 3), "not_following": {k: v for k, v in proc.items() if not v["ok"]},
                   "overrides": {k: v["override_reason"] for k, v in proc.items() if v.get("override_reason")},
                   "pass": share >= 0.9}

    viol, short = {}, {}
    for r in rows:
        bad, shortened = citations(r)
        if bad:
            viol[r["run"]] = bad
        if shortened:
            short[r["run"]] = shortened
    rep["A3-5"] = {"violations": viol, "shortened_ids": short, "pass": not viol}

    rep["usable"] = all(rep[k]["pass"] for k in ("A3-1", "A3-2", "A3-3", "A3-4", "A3-5"))
    rep["usage"] = {
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None,
        "input_tokens": sum(r["trace"]["tokens"]["input"] for r in rows),
        "cache_read_tokens": sum(r["trace"]["tokens"]["cache_read"] for r in rows),
        "output_tokens": sum(r["trace"]["tokens"]["output"] + r["trace"]["tokens"]["reasoning"] for r in rows),
        "denied_tool_calls": sum(r["trace"]["denied"] for r in rows),
        "scoreboard_calls": sum(r["trace"].get("scoreboard_calls", 0) for r in rows),
        "simulate_calls": sum(r["trace"]["simulate_calls"] for r in rows),
    }
    rep["modal_by_cell"] = {c: d["modal"] for c, d in cells.items()}
    rep["tied_by_run"] = {r["run"]: proc[r["run"]].get("tied") for r in rows}
    return rep


def across_models(report):
    """Share of farm-profile cells where every model's modal choice is identical or in each other's tied sets."""
    models = [m for m, r in report.items() if isinstance(r, dict) and "modal_by_cell" in r]
    cells = sorted({c for m in models for c in report[m]["modal_by_cell"]})
    out = {}
    for c in cells:
        modals = {m: report[m]["modal_by_cell"].get(c) for m in models}
        labels = {m: label(v) if v else None for m, v in modals.items()}
        identical = len(set(labels.values())) == 1
        tied = {}
        for m in models:
            sets = [set(t or []) for run, t in report[m]["tied_by_run"].items() if f"{c}__" in run]
            tied[m] = set().union(*sets) if sets else set()
        mutual = all(labels[a] in tied[b] or labels[a] == labels[b] for a in models for b in models)
        out[c] = {"modal": labels, "identical": identical, "within_tied_sets": mutual}
    n = len(out)
    return {"cells": out, "identical_share": round(sum(v["identical"] for v in out.values()) / n, 3) if n else None,
            "identical_or_tied_share": round(sum(v["identical"] or v["within_tied_sets"] for v in out.values()) / n, 3)
            if n else None}


def analyse(runs, service, cases):
    rows = load(runs)
    models = defaultdict(list)
    for r in rows:
        models[r["model"]].append(r)
    report = {m: analyse_model(rs, cases, service) for m, rs in sorted(models.items())}
    report["across_models"] = across_models(report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--service", default="http://127.0.0.1:8601")
    ap.add_argument("--out", type=Path, default=Path("data/processed/agent/a3_report.json"))
    args = ap.parse_args()
    cases = {c["id"]: c for c in json.loads(args.cases.read_text())}
    report = analyse(args.runs, Board(args.service, os.environ["FARMSIM_TOKEN"]), cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    summary = {m: {k: v["pass"] for k, v in r.items() if isinstance(v, dict) and "pass" in v} | {"usable": r["usable"]}
               for m, r in report.items() if isinstance(r, dict) and "usable" in r}
    summary["across_models"] = {k: v for k, v in report["across_models"].items() if k != "cells"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
