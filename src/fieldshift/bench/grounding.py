"""Scorer v2 number grounding: does each simulator-attributed number exist in what the agent saw or sent?

The strict v1 check (`score.numeric_faithfulness`) requires a source path for every number. v2 asks a
narrower question, the one the check exists for: was the number invented? Each number in a sim-cited
claim gets one class:

- ``found``: at the claim's displayed precision it equals a numeric leaf, or a number inside a string,
  anywhere in a cited call's output (scale 1 or 100), or a correct ``numeric_values`` annotation.
- ``request``: it equals a value the agent sent for that call (or a farmer economics edit in force).
- ``derived``: difference, sum or percent change of two ``found`` numbers quoted in the same claim.
- ``structural``: a rank or count used as such that equals an output rank/count, or 1-4 as cycle years.
- ``unfounded``: none of these, or a bound annotation that is wrong. This is the failure.
- ``unverifiable``: every cited output is unavailable (harness problem, scored by call-existence checks).
"""

import bisect
import itertools
import json
import math
import re
import shlex
from collections import Counter
from decimal import Decimal

from fieldshift.agent.pilot import CARD_RE, SIM_RE

NUMBER = re.compile(r"(?<![\w:.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)(?!\.\d)")
NEGATIVE = re.compile(r"\b(?:less|lower|lowers|lowered|fewer|drops?|dropped|dropping|minus|reduc\w*|decreas\w*"
                      r"|falls?|fell|falling|declin\w*|cuts?|down|below|lose|loses|losing|lost|loss|losses|deficit|drier|dryer)\b")
APPROX = re.compile(r"(?:about|around|roughly|approximately|approx\.|nearly|almost|some|~|≈)[\s*_]*(?:USD|\$)?[\s*_]*$")
RANK_CONTEXT = re.compile(r"\b(?:rank\w*|place|position|tied|top|rows?|cycles?|rotations?|options?|"
                          r"candidates?|feasible|scored|alternatives?)\b|#\s*$|\d\s+of\s*$")
YEAR_COUNT = re.compile(r"^\s*-?\s*(?:years?|yr)\b")
CLASSES = ("found", "request", "derived", "structural", "unfounded", "unverifiable")
COUNT_KEYS = {"rank", "n_after_limits", "n_cycles_scored"}


def _unit(token, approximate):
    """Last displayed digit. "about 180" is read as rounded to tens, never coarser than ±5% of the value."""
    text = token.replace(",", "").lstrip("+-−")
    exponent = Decimal(text).as_tuple().exponent
    if approximate and exponent == 0 and len(text) > 1 and float(text) >= 100:
        trailing = len(text) - len(text.rstrip("0"))
        exponent = min(trailing, math.floor(math.log10(float(text) / 10)))
    return float(Decimal(1).scaleb(exponent))


CONTINUATION = re.compile(r"^(?:\s*(?:-|–|—|to|and|or|vs\.?|versus|against|/|,)\s*(?:about|around|~|≈)?\s*"
                          r"[-+−]?\s*\$?\s*"
                          r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)*\s*")
UNIT_AFTER = [("money", re.compile(r"^(?:USD|US\$|dollars?\b|\$)")),
              ("percent", re.compile(r"^(?:%|percent\b|per cent\b|pp\b|percentage points?\b)", re.IGNORECASE)),
              ("kg", re.compile(r"^kg\b")),
              ("tonnes", re.compile(r"^(?:t|tonnes?|tons?|Mg)\b")),
              ("days", re.compile(r"^(?:days?\b|d/)")),
              ("mm", re.compile(r"^mm\b")),
              ("temperature", re.compile(r"^(?:°\s*C?|degrees?\b|C\b)"))]
MONEY_BEFORE = re.compile(r"(?:\$|USD|US\$)\s*[-−]?\s*$")
SCORE_BEFORE = re.compile(r"\bscores?(?:\s+of)?\s*$", re.IGNORECASE)
COUNT_BEFORE = re.compile(r"\b(?:rank(?:ed|s)?|ranking|place[ds]?|position)\s*(?:#|no\.?)?\s*$|\d\s+of\s*$|#\s*$",
                          re.IGNORECASE)
COUNT_AFTER = re.compile(r"^\s*(?:of\s+[\d,]+|scored|feasible|(?:canonical\s+)?(?:cycles?|rotations?|candidates?|"
                         r"options?|rows?)\b)", re.IGNORECASE)
PATH_FAMILIES = [("count", re.compile(r"(?:^|\.)(?:rank|n_\w+|weather_runs|spin_years|scored_years_per_run)$")),
                 ("money", re.compile(r"usd")), ("kg", re.compile(r"kg")), ("tonnes", re.compile(r"_t(?:_|$)")),
                 ("days", re.compile(r"days")), ("mm", re.compile(r"_mm")), ("temperature", re.compile(r"temp")),
                 ("percent", re.compile(r"fraction|pct|share|percent|relative_yield|weights_normalised")),
                 ("score", re.compile(r"score|tie_tolerance"))]


def quantity(before, after):
    """Quantity family stated next to a number in prose (None when no unit is stated)."""
    if MONEY_BEFORE.search(before):
        return "money"
    rest = after[CONTINUATION.match(after).end():]
    stated = next((name for name, pattern in UNIT_AFTER if pattern.match(rest)), None)
    if stated:
        return stated
    if SCORE_BEFORE.search(before):
        return "score"
    if COUNT_BEFORE.search(before) or COUNT_AFTER.match(after):
        return "count"
    return None


def path_quantity(path):
    path = path.lower()
    return next((name for name, pattern in PATH_FAMILIES if pattern.search(path)), "other")


def claim_tokens(text):
    """Numbers with displayed precision, stated quantity and context; skips list markers and identifiers."""
    for match in NUMBER.finditer(text):
        token = match.group()
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        if re.fullmatch(r"\s*", text[line_start:start]) and re.match(r"\.\s", text[match.end():match.end() + 2]):
            continue  # markdown list marker "1. "
        value = float(token.replace(",", "").replace("−", "-"))
        before = text[max(0, start - 40):start]
        after = text[match.end():match.end() + 45]
        if value > 0 and re.search(r"(?:^|[\s(\[:,~*_])(?:[-−–]\s*\$\s*|–)$", before):
            value = -value  # "-$6/ha", or an en dash used as a minus sign: "P10 –167.5"
        yield {"token": token, "value": value, "unit": _unit(token, bool(APPROX.search(before))),
               "quantity": quantity(before, after),
               "negative_context": bool(NEGATIVE.search(before.lower()) or NEGATIVE.search(after.lower())),
               "before": before, "after": after}


def _close(target, actual, unit):
    return abs(target - actual) <= .5 * unit + max(math.ulp(float(actual)) * 4, 1e-9)


def _targets(tok):
    return {tok["value"], -tok["value"]} if tok["negative_context"] else {tok["value"]}


def _walk(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk(v, f"{path}.{i}" if path else str(i))
    else:
        yield path, value


class Haystack:
    """Source numbers indexed by quantity family for tolerance search.

    A claim number with a stated unit only matches JSON leaves whose key names the same quantity
    (e.g. `margin_usd_ha_yr` for "$195") or string numbers printed with the same unit. Percentages
    may match share/fraction leaves at scale 100. A unit-less claim number matches any source number
    at scale 1; this is the weaker part of the check and is counted separately (`unitless`).
    """

    def __init__(self):
        self.items = []

    def add_json(self, value, origin, skip=("latency_seconds",)):
        for path, leaf in _walk(value):
            if path.rsplit(".", 1)[-1] in skip:
                continue
            if isinstance(leaf, bool) or leaf is None:
                continue
            if isinstance(leaf, (int, float)) and math.isfinite(leaf):
                self.items.append((float(leaf), path_quantity(path), f"{origin}:{path}"))
            elif isinstance(leaf, str):
                self.add_text(leaf, f"{origin}:{path}")
        return self

    def add_text(self, text, origin, any_quantity=False):
        """String numbers carry the unit printed next to them; prose sources such as the brief or a card
        (tables with units in headers) are matched whatever the claim's quantity (`any_quantity`)."""
        for tok in claim_tokens(text):
            self.items.append((tok["value"], "*" if any_quantity else tok["quantity"], origin))
        return self

    def freeze(self):
        indexes = {None: [(v, o) for v, _, o in self.items]}
        for v, family, o in self.items:
            if family == "*":
                for key in ("money", "percent", "kg", "tonnes", "days", "mm", "temperature", "score", "count"):
                    indexes.setdefault(key, []).append((v, o))
                continue
            indexes.setdefault(family, []).append((v, o))
            if family == "percent":
                indexes.setdefault("percent100", []).append((v * 100, o))
        self.index = {}
        for key, entries in indexes.items():
            entries.sort(key=lambda x: x[0])
            self.index[key] = ([v for v, _ in entries], [o for _, o in entries])
        return self

    def find(self, tok):
        tol = .5 * tok["unit"] + 1e-9
        keys = [tok["quantity"]] + (["percent100"] if tok["quantity"] == "percent" else [])
        for target in sorted(_targets(tok), key=lambda t: t != tok["value"]):
            for key in keys:
                values, origins = self.index.get(key, ([], []))
                lo = bisect.bisect_left(values, target - tol - abs(target) * 1e-12)
                hi = bisect.bisect_right(values, target + tol + abs(target) * 1e-12)
                if lo < hi:
                    best = min(range(lo, hi), key=lambda i: abs(values[i] - target))
                    return values[best], origins[best]
        return None


def call_requests(run_dir):
    """Map each `sim:<n>` label to the request JSON the agent sent, following the client's counter.

    `vm/farmsim` increments its counter only for `simulate|scoreboard` with an argument, before the
    request is sent. Tool parts with status `error` never ran (no output, no label consumed). A visible
    `Simulator call sim:<n>` header overrides the running count; display truncation often removes it.
    Farmer economics edits apply to all later calls until `economics-reset`.
    """
    requests, problems, counter, economics = {}, [], 0, []
    path = run_dir / "transcript.jsonl"
    if not path.exists():
        return requests, ["no transcript"]
    for line in path.read_text().splitlines():
        for part in json.loads(line).get("parts", []):
            if part.get("type") != "tool" or part.get("name") not in (None, "shell"):
                continue
            if part.get("status") != "completed":
                continue
            command = str((part.get("input") or {}).get("command", ""))
            if "farmsim" not in command:
                continue
            try:
                argv = shlex.split(command)
            except ValueError:
                problems.append(f"unparseable command: {command[:60]}")
                continue
            labels = [int(n) for n in re.findall(r"Simulator call sim:(\d+)", part.get("output", ""))]
            for i, arg in enumerate(argv):
                if not arg.endswith("farmsim") or i + 1 >= len(argv):
                    continue
                sub, payload = argv[i + 1], argv[i + 2] if i + 2 < len(argv) else None
                try:
                    parsed = json.loads(payload) if payload is not None else None
                except json.JSONDecodeError:
                    parsed = None
                if sub == "economics-reset":
                    economics = []
                elif sub == "economics" and isinstance(parsed, dict):
                    economics.append(parsed)
                elif sub in ("simulate", "scoreboard") and payload is not None:
                    n = labels.pop(0) if labels else counter + 1
                    if n != counter + 1:
                        problems.append(f"sim:{n}: label after sim:{counter}")
                    counter = n
                    # Repeated labels (concurrent calls racing on the counter) keep every request.
                    requests.setdefault(n, []).append({"command": sub, "request": parsed,
                                                       "economics": list(economics)})
    return requests, problems


def _annotation_check(annotation, tok, outputs, refs):
    """Return (bound, correct, source_value) for one numeric_values entry against one token."""
    if not isinstance(annotation, dict):
        return False, False, None
    got, scale = annotation.get("value"), annotation.get("scale", 1)
    if isinstance(got, bool) or not isinstance(got, (int, float)) or scale not in (1, 100):
        return False, False, None
    if not any(_close(t, got, tok["unit"]) for t in _targets(tok)):
        return False, False, None
    for n in refs:
        try:
            actual = outputs[n]
            for key in str(annotation["path"]).split("."):
                actual = actual[int(key)] if isinstance(actual, list) else actual[key]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        if isinstance(actual, bool):
            continue
        candidates = ([v * scale for v in (t["value"] for t in claim_tokens(actual))] if isinstance(actual, str)
                      else [actual * scale] if isinstance(actual, (int, float)) else [])
        for value in candidates:
            if any(_close(t, value, tok["unit"]) for t in _targets(tok)):
                return True, True, value
    return True, False, None


def _structural_values(outputs, refs):
    values = set()
    for n in refs:
        out = outputs[n]
        for path, leaf in _walk(out):
            if path.rsplit(".", 1)[-1] in COUNT_KEYS and isinstance(leaf, int) and not isinstance(leaf, bool):
                values.add(leaf)
        if isinstance(out, dict):
            rows = out.get("rows")
            if isinstance(rows, list):
                values.add(len(rows))
                values.add(sum(isinstance(r, dict) and r.get("tied_with_best") is True for r in rows))
    return values


def classify_text(text, refs, outputs, requests, extra=None, annotations=(), elsewhere=None):
    """Classify every number in one claim or sentence. Returns a list of per-number dicts.

    `elsewhere` (all outputs of the run plus the brief) only labels why an unfounded number failed.
    """
    tokens = list(claim_tokens(text))
    available = [n for n in refs if n in outputs]
    if not available:
        return [{**_public(t), "class": "unverifiable"} for t in tokens]
    hay = Haystack()
    for n in available:
        hay.add_json(outputs[n], f"sim:{n}")
    for origin, extra_text in (extra or {}).items():
        hay.add_text(extra_text, origin, any_quantity=True)
    hay.freeze()
    req = Haystack()
    for n in refs:
        if n in requests:
            for sent in requests[n]:
                req.add_json(sent["request"], f"request sim:{n}")
                for i, edit in enumerate(sent["economics"]):
                    req.add_json(edit, f"economics[{i}] before sim:{n}")
    req.freeze()
    structural = _structural_values(outputs, available) - {0}
    results = []
    for tok in tokens:
        row = _public(tok)
        # Annotations are optional and rescue-only: a correct one proves the number; a wrong path, scale
        # or value is recorded as a bookkeeping error and the number is then searched like any other.
        bound = [b for b in (_annotation_check(a, tok, outputs, available) for a in annotations) if b[0]]
        ok = [b for b in bound if b[1]]
        if bound and not ok:
            row["annotation_error"] = True
        is_int = tok["unit"] >= 1 and float(tok["value"]).is_integer()
        if ok:
            row.update({"class": "found", "via": "annotation", "source_value": ok[0][2]})
        elif is_int and 1 <= abs(tok["value"]) <= 4 and YEAR_COUNT.match(tok["after"]):
            row.update({"class": "structural", "via": "cycle_years"})
        elif (is_int and abs(tok["value"]) in structural
              and (RANK_CONTEXT.search(tok["before"][-16:].lower()) or re.match(r"\s*of\s+\d", tok["after"]))):
            row.update({"class": "structural", "via": "rank_or_count"})
        elif hit := hay.find(tok):
            row.update({"class": "found", "via": hit[1], "source_value": hit[0]})
        elif hit := req.find(tok):
            row.update({"class": "request", "via": hit[1], "source_value": hit[0]})
        else:
            row["class"] = None
        results.append(row)
    found = [(r, r.get("source_value")) for r in results if r["class"] == "found"
             and (r["via"] == "annotation" or r["via"].startswith("sim:"))]
    for row, tok in zip(results, tokens):
        if row["class"] is not None:
            continue
        derived = _derived(row, found)
        if derived:
            row.update({"class": "derived", "via": derived})
            continue
        row["class"] = "unfounded"
        hit = elsewhere.find(tok) if elsewhere is not None else None
        row["reason"] = ("in_other_call" if hit and hit[1].startswith("sim:") else
                         "in_brief" if hit else "absent_from_run")
        if hit:
            row["elsewhere"] = hit[1]
    return results


def _compatible(*quantities):
    stated = {q for q in quantities if q is not None}
    return len(stated) <= 1


def _derived(row, found):
    """Difference, sum or percent change of two sim-output numbers quoted in the same claim, in the same
    quantity (a missing unit is a wildcard); a percent change must itself be a percentage or unit-less."""
    unit = row["unit"]
    targets = {row["value"], -row["value"]}
    for (a, sa), (b, sb) in itertools.permutations(found, 2):
        if not _compatible(a["quantity"], b["quantity"]):
            continue
        for x, y in {(a["value"], b["value"]), (sa, sb)}:
            if x is None or y is None:
                continue
            options = {}
            if _compatible(a["quantity"], b["quantity"], row["quantity"]):
                options.update(difference=x - y, sum=x + y)
            if y and row["quantity"] in (None, "percent"):
                options["percent_change"] = (x - y) / abs(y) * 100
            for name, value in options.items():
                if any(_close(t, value, unit) for t in targets):
                    return f"{name}({a['token']},{b['token']})"
    return None


def _public(tok):
    return {"token": tok["token"], "value": tok["value"], "unit": tok["unit"], "quantity": tok["quantity"],
            "context": (tok["before"][-30:] + "⟨" + tok["token"] + "⟩" + tok["after"][:30]).replace("\n", " ")}


def _cited_extra(run_dir, source):
    extra = {}
    for card in CARD_RE.findall(source):
        path = run_dir / "work/evidence" / f"{card}.md"
        if path.exists():
            extra[f"card:{card}"] = path.read_text()
    brief = _brief(run_dir)
    if "brief:" in source and brief is not None:
        extra["brief"] = brief
    return extra


def _brief(run_dir):
    """The farm brief is written to the run directory (and mounted into the container as FARM.md)."""
    for path in (run_dir / "FARM.md", run_dir / "work/FARM.md"):
        if path.exists():
            return path.read_text()
    return None


LEADING_CITATION = re.compile(r"^\s*([\[(][^\])]*\b(?:sim|card|brief):[^\])]*[\])]\.?)")


def report_sentences(text):
    """Sentences citing sim:<n>; a citation group after the full stop belongs to the sentence before it."""
    for line in text.splitlines():
        sentences = []
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(*\[`])", line):
            lead = LEADING_CITATION.match(piece)
            if lead and sentences:
                sentences[-1] += " " + lead.group(1)
                piece = piece[lead.end():]
            if piece.strip():
                sentences.append(piece)
        yield from (s for s in sentences if SIM_RE.search(s))


def numeric_grounding(run_dir, decision, outputs, requests):
    """Classify all sim-cited numbers in DECISION.json evidence and, separately, in REPORT.md."""
    elsewhere = Haystack()
    for n, output in outputs.items():
        elsewhere.add_json(output, f"sim:{n}")
    if (brief := _brief(run_dir)) is not None:
        elsewhere.add_text(brief, "brief", any_quantity=True)
    elsewhere.freeze()
    evidence = []
    for i, item in enumerate(decision.get("evidence", []) if isinstance(decision, dict) else []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        refs = sorted({int(n) for n in SIM_RE.findall(source)})
        if not refs:
            continue
        claim = str(item.get("claim", ""))
        annotations = item.get("numeric_values", [])
        annotations = annotations if isinstance(annotations, list) else []
        for row in classify_text(claim, refs, outputs, requests, _cited_extra(run_dir, f"{source} {claim}"),
                                 annotations, elsewhere):
            evidence.append({"evidence": i, **row})
    report = []
    path = run_dir / "work/REPORT.md"
    if path.exists():
        for sentence in report_sentences(path.read_text()):
            refs = sorted({int(n) for n in SIM_RE.findall(sentence)})
            for row in classify_text(sentence, refs, outputs, requests, _cited_extra(run_dir, sentence),
                                     elsewhere=elsewhere):
                report.append({"sentence": sentence[:200], **row})
    counts = Counter(r["class"] for r in evidence)
    report_counts = Counter(r["class"] for r in report)
    return {"grounded_pass": counts["unfounded"] == 0,
            "counts": {c: counts[c] for c in CLASSES},
            "report_counts": {c: report_counts[c] for c in CLASSES},
            "unfounded_reasons": dict(Counter(r["reason"] for r in evidence if r["class"] == "unfounded")),
            "annotation_errors": sum(bool(r.get("annotation_error")) for r in evidence),
            "unitless_numbers": sum(r["quantity"] is None for r in evidence if r["class"] != "unverifiable"),
            "unfounded": [r for r in evidence if r["class"] == "unfounded"],
            "report_unfounded": [r for r in report if r["class"] == "unfounded"],
            "numbers": evidence, "report_numbers": report}


def grounding_metrics(rows):
    """Aggregate v2 per-run grounding into rates per 100 sim-cited numbers."""
    counts, report_counts, reasons, extra = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        g = row.get("grounding") or {}
        counts.update(g.get("counts", {}))
        report_counts.update(g.get("report_counts", {}))
        reasons.update(g.get("unfounded_reasons", {}))
        extra.update({k: g.get(k, 0) for k in ("annotation_errors", "unitless_numbers")})
    def rate(c):
        checked = sum(c[k] for k in CLASSES if k != "unverifiable")
        return 100 * c["unfounded"] / checked if checked else None
    scored = [r for r in rows if r.get("grounding")]
    return {"numbers": {c: counts[c] for c in CLASSES},
            "unfounded_per_100_numbers": rate(counts),
            "unfounded_reasons": dict(reasons),
            "annotation_errors_not_penalised": extra["annotation_errors"],
            "unitless_numbers": extra["unitless_numbers"],
            "report_numbers": {c: report_counts[c] for c in CLASSES},
            "report_unfounded_per_100_numbers": rate(report_counts),
            "grounded_pass_rate": sum(r["grounding"]["grounded_pass"] for r in scored) / len(scored)
            if scored else None,
            "grounded_hygiene_rate": sum(r.get("grounded_hygiene_pass", False) for r in rows) / len(rows)
            if rows else None,
            "grounded_accuracy": sum(r.get("pass_v2", False) for r in rows) / len(rows) if rows else None,
            "grounded_exact_cycle_accuracy": sum(r.get("exact_cycle", False) and r.get("grounded_hygiene_pass", False)
                                                 for r in rows) / len(rows) if rows else None}

