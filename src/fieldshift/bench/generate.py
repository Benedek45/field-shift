"""Compute planted keys: uv run python -m fieldshift.bench.generate --recompute."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from fieldshift.agent import farms
from fieldshift.agent import scoreboard as scoring
from fieldshift.agent.scoreboard import _fingerprint
from fieldshift.bench.common import DATA, ROOT, canonical, choose, local_service, unique
from fieldshift.planner import search


def certify(runtime, result):
    """Check every candidate, not merely the API's top-40 presentation window."""
    rec = farms.get(result["farm_id"], runtime.root)
    key = scoring._hash([scoring.VERSION, search.VERSION, _fingerprint(), farms.physical_record(rec), 0.0, 0.0])
    root = Path(runtime.root) / "scoreboard"
    data = json.loads((root / (key + ".json")).read_text())
    with np.load(root / (key + ".npz")) as arrays:
        runs = arrays["runs"]
    edits = dict(rec.get("economics", {}))
    if "irrigation_capacity_mm_yr" in rec["overrides"]:
        edits["irrigation_capacity_mm_yr"] = rec["overrides"]["irrigation_capacity_mm_yr"]
    full = scoring.ranking(data, runs, result["weights_normalised"], result["limits_applied"],
                           len(data["rows"]), edits)
    result["all_pool_unique"] = unique(full)
    result["uniqueness_certificate"] = {
        "n_candidates_checked": len(full["rows"]),
        "distinct_tied_cycles": len({canonical(r["rotation"]) for r in full["rows"] if r["tied_with_best"]}),
        "smallest_lead_beyond_tolerance": min(
            (full["rows"][0]["score"] - r["score"] - r["tie_tolerance"]
             for r in full["rows"][1:]
             if canonical(r["rotation"]) != canonical(full["rows"][0]["rotation"])),
            default=None),
    }
    return result["all_pool_unique"]


def record(county, weights, prose=""):
    return {"adm_id": county["adm_id"], "irrigated": county["irrigated"], "anonymous": True,
            "overrides": {}, "economics": {}, "constraints": [],
            "priorities": {"text": prose, "weights": weights}}


def generate(runtime, specs):
    pairs, rejected = [], []
    fingerprint = _fingerprint()

    def board(rec, weights, limits=None):
        fid = farms.add(rec, runtime.root)
        result = runtime.scoreboard(fid, {"weights": weights, "limits": limits or {}, "top": 40})
        return result

    for county in specs["counties"]:
        weights = {"income": 5}
        base = record(county, weights, "Maximise return over operating costs.")
        naive = board(base, weights)
        naive_rotation = choose(naive)
        options = runtime.farm(farms.add(base, runtime.root)).options()
        crops = [c["crop"] for c in options["main_crops"]]
        absent = [c for c in crops if c not in {y["crop"] for y in naive_rotation}]

        def publish(family, planted, result, prose, limits=None, rubric=None, *,
                    county=county, base=base, naive=naive, naive_rotation=naive_rotation):
            key = choose(result)
            why = None
            if not rubric:
                if canonical(key) == canonical(naive_rotation):
                    why = "answer equals naive"
                elif not unique(result):
                    why = "planted winner is inside tie tolerance"
                elif not certify(runtime, result):
                    why = "winner tied with a candidate outside the top-40 display"
            if why:
                rejected.append({"county": county["name"], "family": family, "reason": why})
                return False
            # Deterministic opaque identifiers; no task-family labels in model inputs.
            pid = hashlib.sha256(f"{county['adm_id']}:{family}".encode()).hexdigest()[:12]
            planted = copy.deepcopy(planted)
            planted["priorities"]["text"] = base["priorities"]["text"] + "\n" + prose
            pairs.append({"id": pid, "family": family, "county": county["adm_id"],
                          "simulator_version": result["simulator_version"], "fingerprint": fingerprint,
                          "restricted_input_dependencies": [
                              ("Brief rotation evidence may include QDANN-derived county aggregates "
                               "(CC-BY-NC-SA 4.0); raw points/maps are not distributed.")
                          ],
                          "base": copy.deepcopy(base), "planted": planted,
                          "key": key, "naive": naive_rotation, "limits": limits or {}, "rubric": rubric,
                          "board": result, "naive_board": naive})
            return True

        for family in ("hidden_limit", "priority_order"):
            accepted = False
            for crop in absent:
                # One decisive restriction, not a list of unrelated facts.
                limits = {"exclude_crops": [c for c in crops if c != crop]}
                prose = (f"My lease permits only {crop.replace('_', ' ')} as a harvested main crop."
                         if family == "hidden_limit" else
                         f"My first priority is to honour the buyer's crop specification: only "
                         f"{crop.replace('_', ' ')} may be harvested here. My second priority is income; "
                         "the income form does not override that specification.")
                result = board(base, weights, limits)
                if publish(family, base, result, prose, limits):
                    accepted = True
                    break
            if not accepted and not absent:
                rejected.append({"county": county["name"], "family": family, "reason": "no absent feasible crop"})

        accepted = False
        for crop in absent:
            edited = copy.deepcopy(base)
            edited["economics"] = {"crop_price_usd_t": {crop: 3000}}
            result = board(edited, weights)
            # The agent receives BASE economics; must explicitly apply the prose quotation.
            if publish("quoted_price", base, result,
                       f"I have a signed contract selling {crop.replace('_', ' ')} for $3000 per tonne. "
                       "Use this quote, not the regional default price."):
                pairs[-1]["economics_edit"] = edited["economics"]
                accepted = True
                break
        if not accepted and not absent:
            rejected.append({"county": county["name"], "family": "quoted_price", "reason": "no absent feasible crop"})

        field_found = False
        changes = ([{"irrigation_capacity_mm_yr": 0}, {"irrigation_capacity_mm_yr": 50}]
                   if county["irrigated"] else [])
        changes += [{"awc_mm_per_m": 10}, {"slope_pct": 30}, {"rain_change_pct": -70}]
        for overrides in changes:
            edited = copy.deepcopy(base)
            edited["overrides"] = overrides
            result = board(edited, weights)
            if publish("field_fact", edited, result,
                       "The field-specific measurement in the field state supersedes the county default: "
                       + json.dumps(overrides) + "."):
                field_found = True
                break
        # Physical changes often leave income-only candidates tied. Search a fixed,
        # prespecified collection of mixed farmer forms on the same physical pools;
        # never tune after model outputs. Each twin has exactly the SAME form.
        profiles = [
            {"income": 5, "water": 1}, {"income": 1, "water": 5},
            {"income": 5, "erosion": 2}, {"income": 1, "soil": 5},
            {"income": 1, "nitrogen": 5}, {"income": 1, "cover": 5},
            {"water": 5}, {"erosion": 5}, {"resilience": 5},
        ]
        if not field_found:
            for profile in profiles:
                field_base = record(county, profile, "Use the priorities on my form.")
                field_naive = board(field_base, profile)
                for overrides in changes:
                    edited = copy.deepcopy(field_base)
                    edited["overrides"] = overrides
                    result = board(edited, profile)
                    if publish("field_fact", edited, result,
                               "The measured field condition supersedes the county default: "
                               + json.dumps(overrides) + ".",
                               base=field_base, naive=field_naive, naive_rotation=choose(field_naive)):
                        field_found = True
                        break
                if field_found:
                    break
        if not field_found and not changes:
            rejected.append({"county": county["name"], "family": "field_fact", "reason": "no override available"})

        publish("infeasible", base, naive,
                "Please evaluate a pearl millet rotation as well. If your tool cannot evaluate it, "
                "tell me explicitly rather than substituting another crop or making up its economics.",
                rubric={"kind": "infeasible", "terms": ["millet"],
                        "unsupported_crops": ["millet", "pearl_millet"]})
        publish("missing_data", base, naive,
                "I cannot confirm whether the crop buyer will accept this field's harvest; "
                "the buyer eligibility approval is missing. Treat any recommendation as provisional "
                "with low confidence until that approval is known.",
                rubric={"kind": "missing_data", "terms": ["buyer", "eligibility", "approval"]})
        print(county["name"], len([p for p in pairs if p["county"] == county["adm_id"]]), "valid pairs", flush=True)
    return {"schema_version": 1, "specs_sha256": hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest(),
            "pairs": pairs, "discarded": rejected, "excluded_families": specs["excluded_families"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="Replace previously computed keys")
    parser.add_argument("--specs", type=Path, default=ROOT / "bench/specs.json")
    parser.add_argument("--out", type=Path, default=DATA / "keys.json")
    parser.add_argument("--port", type=int, default=8611)
    args = parser.parse_args()
    if args.out.exists() and not args.recompute:
        parser.error("Keys already exist; use --recompute explicitly")
    with local_service(args.port) as (runtime, _):
        result = generate(runtime, json.loads(args.specs.read_text()))
    if any(p["fingerprint"] != _fingerprint() for p in result["pairs"]):
        raise SystemExit("Simulator/data changed during generation; no new keys published. Recompute after edits settle.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    public = [{"id": p["id"], "family": p["family"], "county": p["county"],
               "base": p["base"], "planted": p["planted"], "economics_edit": p.get("economics_edit"),
               "restricted_input_dependencies": p["restricted_input_dependencies"]}
              for p in result["pairs"]]
    (ROOT / "bench/tasks.json").write_text(json.dumps(public, indent=2, allow_nan=False))
    counts = {f: sum(p["family"] == f for p in result["pairs"])
              for f in json.loads(args.specs.read_text())["families"]}
    print(json.dumps({"valid_pairs": counts, "discarded_candidates": len(result["discarded"])}))
    if any(n < 3 for n in counts.values()):
        raise SystemExit("INCOMPLETE: at least one family has fewer than three valid counties; see keys.json")


if __name__ == "__main__":
    main()
