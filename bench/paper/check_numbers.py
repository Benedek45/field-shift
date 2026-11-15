"""Cross-check every results table in bench/paper/b0_v0.5.md against b0_partial.json.

uv run python bench/paper/check_numbers.py            # verify; exit 1 on any mismatch
uv run python bench/paper/check_numbers.py --print    # print the expected tables

Each checked table in the paper follows a marker line ``<!-- check:<name> -->``. The
table is re-rendered from the JSON and compared cell by cell. Counts are rate x n and
must be integers. Numbers quoted in the paper from documents rather than the JSON
(pilot results, the planted-fabrication audit) are checked to appear verbatim in their
source document as well as in the paper.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "bench/paper/b0_v0.5.md"
PARTIAL = ROOT / "data/processed/bench/b0_partial.json"

MODELS = [
    ("anthropic/claude-opus-5-5", "Claude Opus 5.5"),
    ("opencode-go/gpt-6-luna", "GPT-6 Luna"),
    ("opencode-go/glm-5.3-flash", "GLM 5.3 Flash"),
    ("opencode-go/deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
]
FAMILIES = [
    ("hidden_limit", "hidden limit"),
    ("quoted_price", "quoted price"),
    ("field_fact", "field fact"),
    ("priority_order", "priority order"),
    ("infeasible", "infeasible request"),
    ("missing_data", "missing data"),
]
CLASSES = ["found", "request", "derived", "structural", "unfounded", "unverifiable"]

# (phrase in the paper, source document, phrase in the source document)
DOC_FACTS = [
    ("247/300 (82%)", "bench/EXPERIMENTS.md", "247/300 (82%)"),
    ("208/242 (86%)", "bench/EXPERIMENTS.md", "208/242 (86%)"),
    ("39/58 (67%)", "bench/EXPERIMENTS.md", "39/58 (67%)"),
    ("13/27 (48%)", "bench/EXPERIMENTS.md", "13/27 (48%)"),
    ("14/20", "bench/EXPERIMENTS.md", "14/20 detected"),
    ("0/60", "bench/EXPERIMENTS.md", "0/60"),
    ("8/60", "bench/EXPERIMENTS.md", "8/60"),
    ("38/40", "bench/EXPERIMENTS.md", "38/40"),
    ("8/9", "bench/EXPERIMENTS.md", "8/9 correct"),
    ("903/906", "bench/EXPERIMENTS.md", "903/906"),
    ("501 of the 630", "bench/EXPERIMENTS.md", "501 of the 630"),
    ("Story 3 of 5", "EXPERIMENTS.md", "Story modal 3 of 5"),
    ("Lubbock 2 of 5", "EXPERIMENTS.md", "Lubbock 2 of 5"),
    ("42.2 vs 51.2 kg N/ha", "EXPERIMENTS.md", "42.2 vs 51.2 kg N/ha"),
    ("320.6 vs 319.7 mm", "EXPERIMENTS.md", "320.6 vs 319.7 mm"),
    ("83%", "EXPERIMENTS.md", "identical cycle in 83% of cases"),
    ("1,640 mm", "EXPERIMENTS.md", "1,640 mm/yr"),
    ("r = 0.51", "bench/README.md", "r = 0.51 corn / 0.53 soybean"),
]


def k_of(rate, n):
    k = rate * n
    if abs(k - round(k)) > 1e-6:
        raise ValueError(f"rate {rate} x n {n} is not an integer")
    return round(k)


def frac(rate, n):
    if not n:
        return "n/a"
    k = k_of(rate, n)
    return f"{k}/{n} ({100 * k / n:.1f}%)"


def unfounded(block):
    nums = block["v2"]["numbers"]
    checked = sum(nums[c] for c in CLASSES if c != "unverifiable")
    return f"{block['v2']['unfounded_per_100_numbers']:.2f} ({nums['unfounded']}/{checked})"


def seconds(x):
    return f"{x:.1f}"


def tokens(x):
    return f"{round(x):,}"


def overall_rows(d):
    rows = []
    for alias, name in MODELS:
        o = d["models"][alias]["overall"]
        rows.append([name, str(o["n"]), frac(o["accuracy"], o["n"]),
                     frac(o["task_accuracy_including_behavioural_rubrics"], o["n"]),
                     frac(o["v2"]["grounded_accuracy"], o["n"]),
                     frac(o["hygienic_accuracy_v1"], o["n"]),
                     frac(o["trap_rate"], o["trap_n"]),
                     frac(o["twin_pair_accuracy"], o["twin_pairs_n"]),
                     unfounded(o), seconds(o["median_seconds"]),
                     tokens(o["median_tokens"]["output_tokens"])])
    return rows


def family_rows(d):
    rows = []
    for alias, name in MODELS:
        fams = d["models"][alias]["families"]
        for fam, label in FAMILIES:
            b = fams[fam]
            rows.append([name, label, str(b["n"]), frac(b["accuracy"], b["n"]),
                         frac(b["task_accuracy_including_behavioural_rubrics"], b["n"]),
                         frac(b["v2"]["grounded_accuracy"], b["n"]),
                         frac(b["hygienic_accuracy_v1"], b["n"]),
                         frac(b["twin_pair_accuracy"], b["twin_pairs_n"]),
                         unfounded(b), seconds(b["median_seconds"]),
                         tokens(b["median_tokens"]["output_tokens"])])
    return rows


def classes_rows(d):
    rows = []
    for alias, name in MODELS:
        o = d["models"][alias]["overall"]
        v2 = o["v2"]
        rep = v2["report_numbers"]
        rows.append([name] + [str(v2["numbers"][c]) for c in CLASSES]
                    + [frac(v2["grounded_pass_rate"], o["n"]),
                       f"{v2['report_unfounded_per_100_numbers']:.2f} ({rep['unfounded']})",
                       str(v2["annotation_errors_not_penalised"])])
    return rows


def coverage_rows(d):
    rows = []
    for alias, name in MODELS:
        m = d["models"][alias]
        excluded = [e for es in m["excluded"].values() for e in es]
        reasons = collections.Counter(e["reason"] for e in excluded)
        rows.append([name, str(m["overall"]["n"]), str(len(m["missing_repeat_0"])),
                     str(reasons.get("archived_infrastructure", 0)),
                     str(reasons.get("noncompleted_or_infrastructure", 0)),
                     str(reasons.get("other_repeat", 0)),
                     str(len(excluded))])
    return rows


def disagreement_rows(d):
    keyed = {k["run"] for k in d["key_disagreements"]}
    counts = collections.Counter()
    for alias, name in MODELS:
        for r in d["models"][alias]["runs"]:
            if not r["exact_cycle"]:
                assert r["run"] in keyed and r["county"] == "US-48-303"
                assert r["decision_rule"] == "5d" and r["override_used"]
                counts[(name, "planted" if r["planted"] else "control")] += 1
    rows = []
    for _, name in MODELS:
        c, p = counts[(name, "control")], counts[(name, "planted")]
        rows.append([name, str(c), str(p), str(c + p)])
    tc = sum(v for (n, t), v in counts.items() if t == "control")
    tp = sum(v for (n, t), v in counts.items() if t == "planted")
    rows.append(["Total", str(tc), str(tp), str(tc + tp)])
    assert tc + tp == len(d["key_disagreements"])
    return rows


TABLES = {"overall": overall_rows, "families": family_rows, "classes": classes_rows,
          "coverage": coverage_rows, "disagreements": disagreement_rows}


def inline_facts(d):
    """Numbers stated in the prose of the paper, recomputed from the JSON."""
    runs = [r for alias, _ in MODELS for r in d["models"][alias]["runs"]]
    five_d = sum(r["decision_rule"] == "5d" for r in runs)
    planted_cycle = [r for r in runs if r["planted"] and not r["rubric"]]
    m = {name: d["models"][alias] for alias, name in MODELS}

    def v1_failed(name):
        o = m[name]["overall"]
        correct = k_of(o["task_accuracy_including_behavioural_rubrics"], o["n"])
        return correct - k_of(o["hygienic_accuracy_v1"], o["n"]), correct

    def missing(name):
        b = m[name]["families"]["missing_data"]
        return f"{k_of(b['task_accuracy_including_behavioural_rubrics'], b['n'])}/{b['n']}"

    luna = m["GPT-6 Luna"]["overall"]
    lowest_missing = all(
        min(fams, key=lambda f: fams[f]["task_accuracy_including_behavioural_rubrics"]) == "missing_data"
        for fams in (m[name]["families"] for name in m))
    return {
        "37 of 64 task-correct Opus": v1_failed("Claude Opus 5.5") == (37, 64),
        "14 of 60 for\nLuna": v1_failed("GPT-6 Luna") == (14, 60),
        "47 of 50 for GLM": v1_failed("GLM 5.3 Flash") == (47, 50),
        "15 of 30 for DeepSeek": v1_failed("DeepSeek V4.1 Flash") == (15, 30),
        "failed in 14\nepisodes": luna["n"] - k_of(luna["v2"]["grounded_hygiene_rate"], luna["n"]) == 14
        and luna["v2"]["numbers"]["unfounded"] == 0,
        "Opus 8/11, Luna 7/12, GLM 6/10 and DeepSeek\n3/6": lowest_missing and [missing(n) for _, n in MODELS]
        == ["8/11", "7/12", "6/10", "3/6"],
        "12 to 23 per model": sorted(m[n]["overall"]["trap_n"] for n in m) == [12, 20, 23, 23],
        "236 completed": len(runs) == 236,
        "21 exact-cycle": len(d["key_disagreements"]) == 21,
        "rule 5d was used in 21 of 236": five_d == 21 and len(runs) == 236,
        "78/78": len(planted_cycle) == 78 and all(r["exact_cycle"] for r in planted_cycle),
        "fieldshift-v25": d["simulator_versions"] == ["fieldshift-v25"],
    }


def key_facts():
    """Construction numbers stated in the paper, recomputed from the frozen keys file."""
    path = ROOT / "data/processed/bench/keys.json"
    if not path.exists():
        return {}
    keys = json.loads(path.read_text())
    pairs, discarded = keys["pairs"], keys["discarded"]
    fams = collections.Counter(p["family"] for p in pairs)
    reasons = collections.Counter(x["reason"] for x in discarded)
    certs = [p["board"].get("uniqueness_certificate", {}) for p in pairs if not p.get("rubric")]
    checked = [c["n_candidates_checked"] for c in certs if c]
    leads = [c["smallest_lead_beyond_tolerance"] for c in certs
             if c and c.get("smallest_lead_beyond_tolerance") is not None]
    field_counties = {p["county"] for p in pairs if p["family"] == "field_fact"}
    return {
        "35 certified twin pairs": len(pairs) == 35,
        "187 discarded": len(discarded) == 187 and all(x["family"] == "field_fact" for x in discarded),
        "125 because": reasons["answer equals naive"] == 125,
        "62 because": reasons["planted winner is inside tie tolerance"] == 62,
        "hidden limit 6, quoted price 6, field fact 5, priority order 6, infeasible request 6, missing data 6": (
            [fams[f] for f, _ in FAMILIES] == [6, 6, 5, 6, 6, 6]),
        "Bolivar": "US-28-011" not in field_counties,
        "1 to 550 candidate cycles": min(checked) == 1 and max(checked) == 550,
        "0.13 score points": f"{min(leads):.2f}" == "0.13",
        "93.58 score points": f"{max(leads):.2f}" == "93.58",
    }


def parse(md, name):
    lines = md.splitlines()
    try:
        start = lines.index(f"<!-- check:{name} -->")
    except ValueError:
        return None
    table = []
    for line in lines[start + 1:]:
        if not line.strip():
            if table:
                break
            continue
        if not line.startswith("|"):
            break
        table.append(line)
    body = [row for row in table[2:]]
    return [[c.strip().replace("**", "") for c in row.strip().strip("|").split("|")] for row in body]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", help="print expected tables")
    args = ap.parse_args()
    d = json.loads(PARTIAL.read_text())
    if args.print:
        for name, fn in TABLES.items():
            print(f"<!-- check:{name} -->")
            for row in fn(d):
                print("| " + " | ".join(row) + " |")
            print()
        return
    md = PAPER.read_text()
    errors, cells = [], 0
    for name, fn in TABLES.items():
        got, want = parse(md, name), fn(d)
        if got is None:
            errors.append(f"table {name}: marker missing")
            continue
        if len(got) != len(want):
            errors.append(f"table {name}: {len(got)} rows, expected {len(want)}")
            continue
        for i, (g, w) in enumerate(zip(got, want)):
            if len(g) != len(w):
                errors.append(f"table {name} row {i}: {len(g)} cells, expected {len(w)}")
                continue
            for j, (a, b) in enumerate(zip(g, w)):
                cells += 1
                if a != b:
                    errors.append(f"table {name} row {i} col {j}: paper '{a}' != json '{b}'")
    facts = {**inline_facts(d), **key_facts()}
    for phrase, ok in facts.items():
        if not ok:
            errors.append(f"inline fact false in JSON: {phrase}")
        elif phrase not in md:
            errors.append(f"inline fact not in paper: {phrase}")
    for phrase, src, src_phrase in DOC_FACTS:
        text = (ROOT / src).read_text()
        if src_phrase not in text:
            errors.append(f"source {src} lacks '{src_phrase}'")
        if phrase not in md:
            errors.append(f"paper lacks '{phrase}' (from {src})")
    for e in errors:
        print("MISMATCH", e)
    print(f"{cells} table cells, {len(facts)} JSON/keys facts, {len(DOC_FACTS)} document facts checked; "
          f"{len(errors)} mismatches")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
