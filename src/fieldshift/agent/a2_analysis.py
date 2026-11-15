"""Score pilot A2 (scoreboard procedure, several models) against the rules registered in EXPERIMENTS.md.

uv run python -m fieldshift.agent.a2_analysis [--runs DIR] [--service URL]
Needs FARMSIM_TOKEN; sensitivity and procedure checks call the simulator service.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import requests

from fieldshift.agent.a1_analysis import (
    BASE_FARMS,
    FORAGE_LEGUMES,
    SMALL_GRAINS,
    Service,
    cycle,
    feasible,
    label,
    modal,
    paired,
)

RUNS = Path.home() / ".opencode-vm/fieldshift/runs/a2"
WEIGHT_KEYS = ("income", "water", "soil", "nitrogen", "erosion", "cover", "forage", "resilience", "market")


class Board(Service):
    def scoreboard(self, farm, weights, limits):
        body = {"weights": weights or {}, "limits": limits or {}, "top": 40}
        r = requests.post(f"{self.url}/farm/{farm}/scoreboard", headers=self.headers, json=body, timeout=600)
        if r.status_code != 200:
            return {"error": r.status_code, "detail": r.text[:500]}
        return r.json()


def load(runs):
    out = []
    for meta_path in sorted(runs.glob("*/meta.json")):
        if meta_path.parent.name.startswith("_"):
            continue
        meta = json.loads(meta_path.read_text())
        path = meta_path.parent / "work" / "DECISION.json"
        try:
            decision = json.loads(path.read_text()) if path.exists() else None
            meta["cycle"] = cycle(decision["recommended"]["rotation"]) if decision else None
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            decision, meta["cycle"] = None, None
        meta["decision"] = decision
        out.append(meta)
    return out


def weight_vector(decision):
    w = (decision or {}).get("weights") or {}
    return tuple(w.get(k, 0) for k in WEIGHT_KEYS)


def procedure(service, run):
    """Expected choice from the run's own weights and limits: existing if tied with best, else rank 1."""
    d = run["decision"]
    if not d:
        return {"ok": False, "why": "no decision"}
    board = service.scoreboard(run["farm_id"], d.get("weights"), d.get("limits"))
    if "error" in board:
        return {"ok": False, "why": f"scoreboard error {board}"}
    rows = [(cycle(r["rotation"]), r) for r in board["rows"]]
    tied = [c for c, r in rows if r.get("tied_with_best")]
    existing = cycle(d["existing"]["rotation"])
    expected = existing if existing in tied else (rows[0][0] if rows else None)
    override = (d.get("recommended") or {}).get("override_reason")
    ok = run["cycle"] == expected or bool(override)
    return {"ok": ok, "expected": label(expected) if expected else None, "got": label(run["cycle"]),
            "override_reason": override, "tied": [label(c) for c in tied]}


def citations(run):
    s = run["score"]
    opened = set(run.get("cards_opened", []))
    missing = s.get("cards_missing") or []
    shortened = [m for m in missing if any(o.startswith(m) or m in o for o in opened)]
    truly_missing = [m for m in missing if m not in shortened]
    bad = {}
    if truly_missing:
        bad["missing"] = truly_missing
    if s.get("cards_not_opened"):
        bad["not_opened"] = s["cards_not_opened"]
    if len(shortened) > 1:
        bad["shortened"] = shortened
    if s.get("sim_refs_beyond_calls"):
        bad["sim_refs_beyond_calls"] = s["sim_refs_beyond_calls"]
    if not s.get("valid"):
        bad["invalid_decision"] = True
    return bad, shortened


def analyse_model(rows, service):
    by = {(r["case"], r["rep"]): r for r in rows}
    rep = {"runs": len(rows), "status": dict(Counter(r["status"] for r in rows))}
    if not all((f, 0) in by and by[(f, 0)]["cycle"] for f in BASE_FARMS):
        rep["incomplete"] = sorted(f for f in BASE_FARMS if (f, 0) not in by or not by[(f, 0)]["cycle"])
    base = {f: by[(f, 0)] for f in BASE_FARMS if (f, 0) in by and by[(f, 0)]["cycle"]}
    opts = {f: service.options(base[f]["farm_id"]) for f in base}
    infeasible = {f: feasible(base[f], opts[f]) for f in base}
    distinct = len({base[f]["cycle"] for f in base})
    rep["A2-1"] = {"recommended": {f: label(base[f]["cycle"]) for f in base}, "distinct": distinct,
                   "infeasible": {f: v for f, v in infeasible.items() if v},
                   "pass": len(base) == 6 and distinct >= 4 and not any(infeasible.values())}

    reps = {f: [by[(f, i)]["cycle"] for i in range(5) if (f, i) in by and by[(f, i)]["cycle"]]
            for f in ("story", "lubbock")}
    stab = {}
    for f, cycles in reps.items():
        if cycles:
            top, n, counts = modal(cycles)
            stab[f] = {"modal": label(top), "modal_count": n, "of": len(cycles),
                       "all": {label(k): v for k, v in counts.items()}}
    rep["A2-3"] = {"farms": stab, "pass": len(stab) == 2 and all(s["modal_count"] >= 4 for s in stab.values())}

    checks, verdicts, pairs = {}, {}, {}

    def verdict(c):
        if "error" in c:
            return "error"
        if c["a"] < c["b"]:
            return "expected"
        return "same" if abs(c["a"] - c["b"]) < 1e-9 else "reversed"

    story_modal = modal(reps["story"])[0] if reps["story"] else None
    lubbock_modal = modal(reps["lubbock"])[0] if reps["lubbock"] else None
    slope = by.get(("story-slope12", 0))
    if slope and slope["cycle"] and story_modal:
        checks["a_slope"] = {n: paired(service, slope["farm_id"], slope["cycle"], story_modal, n)
                             for n in ("erosion_t_ha_yr", "days_without_living_cover_yr")}
        v = [verdict(c) for c in checks["a_slope"].values()]
        verdicts["a_slope"] = "expected" if "expected" in v else ("reversed" if all(x == "reversed" for x in v)
                                                                  else "same")
        pairs["a_slope"] = [label(slope["cycle"]), label(story_modal)]
    nitrate, income = by.get(("story-nitrate", 0)), by.get(("story-income", 0))
    if nitrate and income and nitrate["cycle"] and income["cycle"]:
        checks["b_nitrate"] = paired(service, nitrate["farm_id"], nitrate["cycle"], income["cycle"],
                                     "nitrate_leached_kg_n_ha_yr")
        verdicts["b_nitrate"] = verdict(checks["b_nitrate"])
        pairs["b_nitrate"] = [label(nitrate["cycle"]), label(income["cycle"])]
    dry = by.get(("lubbock-dry", 0))
    if dry and dry["cycle"] and lubbock_modal:
        checks["c_dry"] = paired(service, dry["farm_id"], dry["cycle"], lubbock_modal, "field_water_use_mm_yr")
        verdicts["c_dry"] = verdict(checks["c_dry"])
        pairs["c_dry"] = [label(dry["cycle"]), label(lubbock_modal)]
    rep["A2-2"] = {"pairs": pairs, "values": checks, "verdicts": verdicts,
                   "pass": len(verdicts) == 3 and sum(v == "expected" for v in verdicts.values()) >= 2
                   and "reversed" not in verdicts.values()}

    low_sim = [r["run"] for r in rows if r["trace"]["simulate_calls"] < 1]
    low_src = [r["run"] for r in rows if (r["score"].get("sourced_share") or 0) < 0.7]
    rep["A2-4"] = {"without_simulator": low_sim, "sourced_below_70": low_src, "pass": not low_sim and not low_src}

    known = {}
    rye = by.get(("known-rye", 0))
    if rye and rye["cycle"]:
        known["a_rye"] = {"recommended": label(rye["cycle"]),
                          "pass": any(w == "cereal_rye_cover" for _, w in rye["cycle"])}
    marsden = by.get(("known-marsden", 0))
    if marsden and marsden["cycle"]:
        crops = {c for c, _ in marsden["cycle"]}
        known["b_marsden"] = {"recommended": label(marsden["cycle"]),
                              "pass": len(marsden["cycle"]) >= 3 and bool(crops & (SMALL_GRAINS | FORAGE_LEGUMES))}
    rep["A2-5"] = {"cases": known, "pass": sum(k["pass"] for k in known.values()) >= 1}

    viol, short = {}, {}
    for r in rows:
        bad, shortened = citations(r)
        if bad:
            viol[r["run"]] = bad
        if shortened:
            short[r["run"]] = shortened
    rep["A2-6"] = {"violations": viol, "shortened_ids": short, "pass": not viol}

    proc = {r["run"]: procedure(service, r) for r in rows}
    share = sum(p["ok"] for p in proc.values()) / len(proc) if proc else 0
    rep["A2-7"] = {"share": round(share, 3), "not_following": {k: v for k, v in proc.items() if not v["ok"]},
                   "overrides": {k: v["override_reason"] for k, v in proc.items() if v.get("override_reason")},
                   "pass": share >= 0.9}

    story_w = [weight_vector(by[("story", i)]["decision"]) for i in range(5) if ("story", i) in by]
    wtop, wn, wcounts = modal(story_w) if story_w else (None, 0, Counter())
    rep["A2-8"] = {"modal_weights": dict(zip(WEIGHT_KEYS, wtop)) if wtop else None, "modal_count": wn,
                   "distinct": len(wcounts), "pass": wn >= 4}

    rep["usable"] = all(rep[k]["pass"] for k in ("A2-1", "A2-2", "A2-3", "A2-6"))
    rep["usage"] = {
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None,
        "input_tokens": sum(r["trace"]["tokens"]["input"] for r in rows),
        "cache_read_tokens": sum(r["trace"]["tokens"]["cache_read"] for r in rows),
        "output_tokens": sum(r["trace"]["tokens"]["output"] + r["trace"]["tokens"]["reasoning"] for r in rows),
        "cost_reported": sum(r["trace"]["cost"] for r in rows),
        "denied_tool_calls": sum(r["trace"]["denied"] for r in rows),
        "scoreboard_calls": sum(r["trace"].get("scoreboard_calls", 0) for r in rows),
        "simulate_calls": sum(r["trace"]["simulate_calls"] for r in rows),
    }
    rep["weights_by_case"] = {f"{r['case']}__r{r['rep']}": dict(zip(WEIGHT_KEYS, weight_vector(r["decision"])))
                              for r in rows}
    return rep


def analyse(runs, service):
    rows = load(runs)
    models = defaultdict(list)
    for r in rows:
        models[r["model"]].append(r)
    report = {m: analyse_model(rs, service) for m, rs in sorted(models.items())}
    cases = sorted({k for m in report.values() for k in m["weights_by_case"]})
    report["weights_agree_across_models"] = {
        c: len({tuple(sorted(m["weights_by_case"][c].items())) for m in report.values()
                if isinstance(m, dict) and c in m.get("weights_by_case", {})}) == 1
        for c in cases
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--service", default="http://127.0.0.1:8601")
    ap.add_argument("--out", type=Path, default=Path("data/processed/agent/a2_report.json"))
    args = ap.parse_args()
    report = analyse(args.runs, Board(args.service, os.environ["FARMSIM_TOKEN"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    summary = {m: {k: v["pass"] for k, v in r.items() if isinstance(v, dict) and "pass" in v} | {"usable": r["usable"]}
               for m, r in report.items() if isinstance(r, dict) and "usable" in r}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
