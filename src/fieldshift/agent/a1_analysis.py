"""Score pilot A1 against the pass rules registered in EXPERIMENTS.md.

uv run python -m fieldshift.agent.a1_analysis [--runs DIR] [--service URL]
Needs FARMSIM_TOKEN; the paired-sensitivity rule re-simulates recommendations.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import requests

RUNS = Path.home() / ".opencode-vm/fieldshift/runs/a1"
BASE_FARMS = ("story", "york", "sumner", "lubbock", "bolivar", "fresno")
SMALL_GRAINS = {"oats", "winter_wheat", "barley", "spring_wheat", "rye"}
FORAGE_LEGUMES = {"alfalfa", "clover", "red_clover"}


def winter_key(value):
    if not value:
        return None
    text = str(value).lower()
    if "rye" in text:
        return "cereal_rye_cover"
    if "vetch" in text:
        return "hairy_vetch_cover"
    if "wheat" in text:
        return "winter_wheat"
    return "other:" + text


def cycle(rotation):
    years = [(str(y.get("crop")), winter_key(y.get("winter"))) for y in rotation]
    n = len(years)
    period = next(p for p in range(1, n + 1) if n % p == 0 and years == years[:p] * (n // p))
    years = years[:period]
    shifts = [tuple(years[i:] + years[:i]) for i in range(period)]
    return min(shifts, key=lambda s: [(c, w or "") for c, w in s])


def label(c):
    return " / ".join(crop + (f"+{w.replace('_cover', '')}" if w else "") for crop, w in c)


def load(runs):
    out = []
    for meta_path in sorted(runs.glob("*/meta.json")):
        if meta_path.parent.name.startswith("_"):
            continue
        meta = json.loads(meta_path.read_text())
        decision_path = meta_path.parent / "work" / "DECISION.json"
        decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
        meta["decision"] = decision
        meta["cycle"] = cycle(decision["recommended"]["rotation"]) if decision else None
        out.append(meta)
    return out


class Service:
    def __init__(self, url, token):
        self.url, self.headers = url.rstrip("/"), {"Authorization": f"Bearer {token}"}

    def options(self, farm):
        r = requests.get(f"{self.url}/farm/{farm}/options", headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def simulate(self, farm, rotations):
        body = {"rotations": [[{"crop": c, **({"winter": w} if w else {})} for c, w in rot] for rot in rotations]}
        r = requests.post(f"{self.url}/farm/{farm}/simulate", headers=self.headers, json=body, timeout=300)
        if r.status_code != 200:
            return {"error": r.status_code, "detail": r.text[:500]}
        return r.json()


def feasible(run, options):
    crops = {c["crop"] for c in options["main_crops"]}
    winters = {o["crop"] if isinstance(o, dict) else o for o in options["winter_options"]} | {None}
    bad = [(c, w) for c, w in run["cycle"] if c not in crops or w not in winters]
    return bad


def modal(cycles):
    counts = Counter(cycles)
    top, n = counts.most_common(1)[0]
    return top, n, counts


def metric(result, name):
    return result["metrics"][name]["median"]


def paired(service, farm, cycle_a, cycle_b, name):
    res = service.simulate(farm, [cycle_a, cycle_b])
    if "error" in res:
        return {"error": res}
    a, b = (metric(r, name) for r in res["results"])
    return {"metric": name, "a": a, "b": b}


def analyse(runs, service):
    rows = load(runs)
    by = {(r["case"], r["setup"], r["rep"]): r for r in rows}
    report = {"runs": len(rows), "status": dict(Counter(r["status"] for r in rows))}

    full0 = {f: by[(f, "full", 0)] for f in BASE_FARMS}
    opts = {f: service.options(full0[f]["farm_id"]) for f in BASE_FARMS}
    infeasible = {f: feasible(full0[f], opts[f]) for f in BASE_FARMS}
    distinct = len({full0[f]["cycle"] for f in BASE_FARMS})
    report["A1-1"] = {
        "recommended": {f: label(full0[f]["cycle"]) for f in BASE_FARMS},
        "distinct": distinct,
        "infeasible": {f: v for f, v in infeasible.items() if v},
        "pass": distinct >= 4 and not any(infeasible.values()),
    }

    story_reps = [by[("story", "full", i)]["cycle"] for i in range(5)]
    lubbock_reps = [by[("lubbock", "full", i)]["cycle"] for i in range(5)]
    story_modal = modal(story_reps)[0]
    checks = {}
    slope = by[("story-slope12", "full", 0)]
    for name in ("erosion_t_ha_yr", "days_without_living_cover_yr"):
        checks.setdefault("a_slope", {})[name] = paired(service, slope["farm_id"], slope["cycle"], story_modal, name)
    nitrate, income = by[("story-nitrate", "full", 0)], by[("story-income", "full", 0)]
    checks["b_nitrate"] = paired(service, nitrate["farm_id"], nitrate["cycle"], income["cycle"],
                                 "nitrate_leached_kg_n_ha_yr")
    dry = by[("lubbock-dry", "full", 0)]
    lubbock_modal = modal(lubbock_reps)[0]
    checks["c_dry"] = paired(service, dry["farm_id"], dry["cycle"], lubbock_modal, "field_water_use_mm_yr")

    def verdict(c):
        if "error" in c:
            return "error"
        if c["a"] < c["b"]:
            return "expected"
        return "same" if abs(c["a"] - c["b"]) < 1e-9 else "reversed"

    v_slope = [verdict(c) for c in checks["a_slope"].values()]
    verdicts = {
        "a_slope": "expected" if "expected" in v_slope else ("reversed" if all(v == "reversed" for v in v_slope)
                                                              else "same"),
        "b_nitrate": verdict(checks["b_nitrate"]),
        "c_dry": verdict(checks["c_dry"]),
    }
    report["A1-2"] = {
        "pairs": {
            "a_slope": [label(slope["cycle"]), label(story_modal)],
            "b_nitrate": [label(nitrate["cycle"]), label(income["cycle"])],
            "c_dry": [label(dry["cycle"]), label(lubbock_modal)],
        },
        "values": checks,
        "verdicts": verdicts,
        "pass": sum(v == "expected" for v in verdicts.values()) >= 2 and "reversed" not in verdicts.values(),
    }

    stab = {}
    for farm, reps in (("story", story_reps), ("lubbock", lubbock_reps)):
        top, n, counts = modal(reps)
        stab[farm] = {"modal": label(top), "modal_count": n, "all": {label(k): v for k, v in counts.items()}}
    report["A1-3"] = {"farms": stab, "pass": all(s["modal_count"] >= 4 for s in stab.values())}

    full = [r for r in rows if r["setup"] == "full"]
    low_sim = [r["run"] for r in full if r["trace"]["simulate_calls"] < 1]
    low_src = [r["run"] for r in full if r["score"].get("sourced_share", 0) < 0.7]
    differs = {}
    for f in BASE_FARMS:
        cycles = {s: label(by[(f, s, 0)]["cycle"]) for s in ("llm", "evidence", "full")}
        differs[f] = cycles
    n_diff_llm = sum(differs[f]["llm"] != differs[f]["full"] for f in BASE_FARMS)
    n_diff_ev = sum(differs[f]["evidence"] != differs[f]["full"] for f in BASE_FARMS)
    report["A1-4"] = {
        "full_runs": len(full), "without_simulate": low_sim, "sourced_below_70": low_src,
        "by_setup": differs, "full_differs_from_llm": n_diff_llm, "full_differs_from_evidence": n_diff_ev,
        "pass": not low_sim and not low_src,
    }

    rye = by[("known-rye", "full", 0)]["cycle"]
    marsden = by[("known-marsden", "full", 0)]["cycle"]
    fallow_run = by[("known-fallow", "full", 0)]
    fallow_opts = service.options(fallow_run["farm_id"])
    has_fallow_option = any("fallow" in c["crop"] for c in fallow_opts["main_crops"])
    marsden_crops = {c for c, _ in marsden}
    known = {
        "a_rye": {"recommended": label(rye), "pass": any(w == "cereal_rye_cover" for _, w in rye)},
        "b_marsden": {"recommended": label(marsden),
                      "pass": len(marsden) >= 3 and bool(marsden_crops & (SMALL_GRAINS | FORAGE_LEGUMES))},
        "c_fallow": {"recommended": label(fallow_run["cycle"]),
                     "existing": label(cycle(fallow_run["decision"]["existing"]["rotation"])),
                     "fallow_in_options": has_fallow_option,
                     "options": [c["crop"] for c in fallow_opts["main_crops"]],
                     "pass": None if not has_fallow_option else False},
    }
    testable = [k for k in known.values() if k["pass"] is not None]
    report["A1-5"] = {"cases": known, "testable": len(testable), "passed": sum(bool(k["pass"]) for k in testable),
                      "pass": sum(bool(k["pass"]) for k in testable) >= 2}

    viol = {}
    for r in rows:
        s = r["score"]
        bad = {k: s.get(k) for k in ("cards_missing", "cards_not_opened", "sim_refs_beyond_calls") if s.get(k)}
        if not s.get("valid"):
            bad["invalid_decision"] = True
        if bad:
            viol[r["run"]] = bad
    report["A1-6"] = {"violations": viol, "pass": not viol}

    report["usage"] = {
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2],
        "input_tokens": sum(r["trace"]["tokens"]["input"] for r in rows),
        "cache_read_tokens": sum(r["trace"]["tokens"]["cache_read"] for r in rows),
        "output_tokens": sum(r["trace"]["tokens"]["output"] + r["trace"]["tokens"]["reasoning"] for r in rows),
        "cost_reported": sum(r["trace"]["cost"] for r in rows),
        "denied_tool_calls": sum(r["trace"]["denied"] for r in rows),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--service", default="http://127.0.0.1:8601")
    ap.add_argument("--out", type=Path, default=Path("data/processed/agent/a1_report.json"))
    args = ap.parse_args()
    report = analyse(args.runs, Service(args.service, os.environ["FARMSIM_TOKEN"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
