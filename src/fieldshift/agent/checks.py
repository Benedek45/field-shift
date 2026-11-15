"""Product checks, independent of the concurrently developed benchmark scorer."""

import json
import math
import re
from pathlib import Path

from fieldshift.agent import farms, scoreboard, us_units
from fieldshift.agent.runtime import WINTER_ALIASES
from fieldshift.sim.economics import ACRES_HA
from fieldshift.sim.traits import CATALOG

NUMERIC_METHOD = ("Advisory: each number in a sentence citing sim:<n> (or uncited, on a line citing one) must appear, "
                  "exact or rounded, in that call's numeric or text fields, in the farmer form, or as a stated unit "
                  "conversion; sentences citing only a card, the brief or history are not checked against simulator "
                  "calls. Not a semantic/metric attribution check.")
CARD = re.compile(r"card:([A-Za-z0-9_-]+)")
SIM = re.compile(r"sim:(\d+)")
NUMBER = re.compile(r"(?<![\w:])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
# Citation tokens removed before numbers are read (their IDs may contain digits).
CITATION = re.compile(r"\b(?:sim|card|brief|history):[^\s,;()\[\]]*[^\s,;()\[\].]|\bown-knowledge\b")
OTHER_SOURCE = re.compile(r"\b(?:card|brief|history):|\bown-knowledge\b")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*\"'])")
NEGATIVE_MONEY = re.compile(r"[-\u2212\u2013]\s*\$\s*$")
US_UNIT = re.compile(r"\s*-?\s*(inch(?:es)?\b|lbs?\b|pounds?\b)", re.IGNORECASE)


def canonical(rotation):
    steps = farms.validate_rotation(rotation)
    values = [(s["crop"], s.get("winter", ""), s.get("double_crop", "")) for s in steps]
    n = len(values)
    period = next(i for i in range(1, n + 1) if n % i == 0 and values == values[:i] * (n // i))
    values = values[:period]
    return min(tuple(values[i:] + values[:i]) for i in range(period))


def choose(board):
    if not board.get("rows"):
        return None
    return board["existing"] if board["existing"].get("tied_with_best") else board["rows"][0]


def text_limits(text):
    """Deliberately narrow, auditable fallback/grounding parser; no inferred goals."""
    result, reasons = {}, {}
    patterns = {
        "no_new_equipment": r"\bno new (?:machinery|(?:crop )?equipment)\b",
        "keep_forage_crop": r"\bkeep (?:a |one )?forage crop\b",
        "require_winter_cover": r"\b(?:require|keep|include) (?:a )?winter cover\b",
    }
    for k, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[k], reasons[k] = True, m[0]
    m = re.search(r"(?:keep|retain|at least)[^.!?\n]{0,35}?(\d+(?:\.\d+)?)\s*%[^.!?\n]{0,35}?(?:income|margin)", text, re.IGNORECASE)
    if m:
        result["min_margin_share_of_existing"] = float(m[1]) / 100
        reasons["min_margin_share_of_existing"] = m[0]
    # Extra numeric caps must quote an explicit maximum and its unit.
    for match in re.finditer(r"(?:at most|no more than|maximum|max|under|below)\s+(\d+(?:\.\d+)?)\s*([^.!?\n;]+)", text, re.IGNORECASE):
        x, rest = float(match[1]), match[2].casefold()
        key, factor = None, 1.
        if "nitrate" in rest and (("kg" in rest and "ha" in rest) or
                                  ("lb" in rest and ("acre" in rest or "/ac" in rest))):
            key = "max_nitrate_kg_n_ha_yr"
            factor = .45359237 * ACRES_HA if "lb" in rest and ("acre" in rest or "/ac" in rest) else 1.
        elif ("water" in rest or "evapotranspiration" in rest) and ("mm" in rest or "inch" in rest):
            key, factor = "max_field_water_use_mm_yr", 25.4 if "inch" in rest else 1.
        elif "erosion" in rest and ("t/ha" in rest or "ton" in rest):
            key, factor = "max_erosion_t_ha_yr", .90718474 * ACRES_HA if "acre" in rest or "/ac" in rest else 1.
        elif "day" in rest and ("bare" in rest or "without living cover" in rest):
            key = "max_days_without_living_cover_yr"
        if key:
            result[key], reasons[key] = x * factor, match[0]
    exclude = (r"(?:do not plant|exclude|avoid|forbids?|(?:does not|doesn't|do not|don't|will not|won't)"
               r" (?:allow|permit))\s+")
    for key, prefix in (("exclude_crops", exclude),
                        ("require_crops", r"(?:require|include)\s+")):
        for crop in CATALOG:
            if re.search(prefix + re.escape(crop.replace("_", " ")) + r"\b", text, re.IGNORECASE):
                result.setdefault(key, []).append(crop)
                reasons[key] = text
    return result, reasons


def ground_limits(decision, form):
    given = decision.get("limits")
    if not isinstance(given, dict):
        return ["Decision has no limits object"]
    errors = [f"Form limit changed or omitted: {k}" for k, v in form["limits"].items() if given.get(k) != v]
    reasons = decision.get("limit_reasons") or {}
    reasons = reasons if isinstance(reasons, dict) else {}
    for k, v in given.items():
        if k in form["limits"]:
            continue
        reason = reasons.get(k, "")
        # A list limit may be grounded by several quotes, given as a list of quote objects or strings.
        items = reason if isinstance(reason, list) and k in scoreboard.LIST_LIMITS else [reason]
        quotes = []
        for item in items:
            quote = item.get("quote", "") if isinstance(item, dict) else item
            if isinstance(quote, str) and quote.startswith("farmer:"):
                quote = quote[7:].strip().strip("'\"")
            quotes.append(quote)
        if not quotes or any(not isinstance(q, str) or not q or q.casefold() not in form["free_text"].casefold()
                             for q in quotes):
            errors.append(f"{k}: missing exact quote from free_text")
            continue
        if k in scoreboard.LIST_LIMITS:
            found = sorted({c for q in quotes for c in text_limits(q)[0].get(k, [])})
            grounded = isinstance(v, list) and all(isinstance(c, str) for c in v) and found == sorted(v)
        else:
            grounded = len(quotes) == 1 and text_limits(quotes[0])[0].get(k) == v
        if not grounded:
            errors.append(f"{k}: quoted words do not unambiguously specify this limit; ask farmer to use the form")
    return errors


def limit_violations(rotation, metrics, baseline, existing, limits):
    steps = farms.validate_rotation(rotation)
    present = {s["crop"] for s in steps} | {s["winter"] for s in steps if s.get("winter")}
    if any(s.get("double_crop") for s in steps):
        present.add("double_crop_soybean")
    bad = []
    mapping = {"max_nitrate_kg_n_ha_yr": "nitrate_leached_kg_n_ha_yr",
               "max_field_water_use_mm_yr": "field_water_use_mm_yr",
               "max_erosion_t_ha_yr": "erosion_t_ha_yr",
               "max_days_without_living_cover_yr": "days_without_living_cover_yr"}
    for k, metric in mapping.items():
        if k in limits and (metrics.get(metric, {}).get("median") is None or metrics[metric]["median"] > limits[k]):
            bad.append(k)
    if "min_margin_share_of_existing" in limits:
        b = baseline["margin_usd_ha_yr"]["median"]
        floor = b * limits["min_margin_share_of_existing"] if b > 0 else b
        if metrics["margin_usd_ha_yr"]["median"] < floor:
            bad.append("min_margin_share_of_existing")
    for k in limits.get("require_crops", []):
        if k not in present:
            bad.append("require_crops:" + k)
    for k in limits.get("exclude_crops", []):
        if k in present:
            bad.append("exclude_crops:" + k)
    if limits.get("require_winter_cover") and not any(s.get("winter") for s in steps):
        bad.append("require_winter_cover")
    if limits.get("keep_forage_crop") and not any(CATALOG[s["crop"]].v("forage_value") > .5 for s in steps):
        bad.append("keep_forage_crop")
    if limits.get("no_new_equipment"):
        mains = {s["crop"] for s in existing}
        winters = {s.get("winter", "") for s in existing}
        if any(s["crop"] not in mains or WINTER_ALIASES.get(s.get("winter", ""), s.get("winter", "")) not in winters
               or (s.get("double_crop") and not any(e.get("double_crop") for e in existing)) for s in steps):
            bad.append("no_new_equipment")
    return bad


def _numbers(value, path=""):
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_numbers(v, path + "/" + k))
    elif isinstance(value, list):
        for v in value:
            out.extend(_numbers(v, path))
    elif isinstance(value, str):
        out.extend(float(raw.rstrip("%").replace(",", "")) for raw in NUMBER.findall(value))
    elif isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        out.append(float(value))
        if "usd_ha" in path or "kg_n_ha" in path:
            out.append(value / ACRES_HA)
        if "_mm" in path:
            out.append(value / 25.4)
        if "fraction" in path or "probability" in path or "share" in path:
            out.append(value * 100)
    return out


def _unit_numbers(value, unit, key=""):
    """Numbers of one quantity only: US-unit fields as printed plus the metric fields converted.
    `unit` is "in" (inches) or "lb" (lb N per acre)."""
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_unit_numbers(v, unit, key if k in us_units.SUMMARY_KEYS else k))
    elif isinstance(value, list):
        for v in value:
            out.extend(_unit_numbers(v, unit, key))
    elif isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        tokens = key.split("_")
        if "per" in tokens:
            return out
        if unit == "in":
            if "in" in tokens:
                out.append(float(value))
            if "mm" in tokens:
                out.append(value / us_units.MM_PER_INCH)
        elif unit == "lb":
            if "lb" in tokens:
                out.append(float(value))
            if "kg" in tokens:
                out.append(value * us_units.LB_PER_KG / (ACRES_HA if "ha" in tokens else 1))
    return out


def claim_segments(decision, report):
    """(text, sim refs) pairs to check. A report line is split into sentences: a sentence is checked
    against its own `sim:<n>` citations; one citing only a card, the brief, history or own knowledge is
    not checked against simulator calls; an uncited sentence falls back to the line's `sim:` refs."""
    claims = []
    for item in decision.get("evidence", []) or []:
        if isinstance(item, dict) and SIM.search(str(item.get("source", ""))):
            claims.append((str(item.get("claim", "")), SIM.findall(item["source"])))
    for line in report.splitlines():
        line_refs = SIM.findall(line)
        if not line_refs:
            continue
        for sentence in SENTENCE.split(line):
            refs = SIM.findall(sentence)
            if refs:
                claims.append((sentence, refs))
            elif not OTHER_SOURCE.search(sentence):
                claims.append((sentence, line_refs))
    return claims


def numeric_claims(decision, report, calls, form=None):
    errors, money, units, checked = [], [], [], 0
    given = _numbers(form or {})
    for text, refs in claim_segments(decision, report):
        sources = [calls.get(ref, {}) for ref in refs]
        candidates = [n for s in sources for n in _numbers(s)] + given
        text = CITATION.sub("", text)
        for match in NUMBER.finditer(text):
            raw = match[0]
            x = float(raw.rstrip("%").replace(",", ""))
            if raw[0] not in "+-" and NEGATIVE_MONEY.search(text[:match.start()]):
                x = -x  # "-$6" carries its sign before the dollar mark
            mantissa, *exponent = raw.rstrip("%").lower().split("e")
            places = len(mantissa.split(".")[1]) if "." in mantissa else 0
            tolerance = .5 * 10 ** ((int(exponent[0]) if exponent else 0) - places)
            unit = US_UNIT.match(text, match.end())
            if unit and not raw.endswith("%"):
                kind = "in" if unit[1].lower().startswith("inch") else "lb"
                pool = [n for s in sources for n in _unit_numbers(s, kind)] + given
                if not any(abs(x - n) <= tolerance + 1e-7 for n in pool):
                    label = "inches" if kind == "in" else "lb"
                    units.append(f"{raw} {label} not in the {label} fields of sim:{','.join(refs)}: {text[:160]}")
            if not any(abs(x - n) <= tolerance + 1e-7 for n in candidates):
                error = f"{raw} not supported by sim:{','.join(refs)}: {text[:160]}"
                errors.append(error)
                if text[:match.start()].rstrip().endswith("$"):
                    money.append(error)
            checked += 1
    return {"passed": not errors, "advisory": True, "detail": errors, "method": NUMERIC_METHOD,
            "numbers_checked": checked, "numbers_unmatched": len(errors), "dollar_errors": money,
            "us_unit_errors": units}


def withhold_failing_answer(reply, calls, form):
    """Gating dollar and inch/lb checks on a follow-up answer: (checks, text to show). No answer is
    better than a bad answer, so a failed check replaces the answer with the reason."""
    figures = figure_checks(reply, calls, form)
    failed = [d for v in figures.values() for d in v["detail"]]
    if not failed:
        return figures, reply
    reason = "; ".join(failed[:3]) + (f" (and {len(failed) - 3} more)" if len(failed) > 3 else "")
    return figures, (f"This answer was withheld: a figure does not match the simulator run it cites ({reason}). "
                     "Ask again, or ask for the figure directly.")


def figure_checks(text, calls, form=None):
    """Gating dollar and US-unit checks for free text such as a follow-up answer (no decision file)."""
    found = numeric_claims({}, text, calls, form)
    return {"dollar_figures_match_cited_runs": {"passed": not found["dollar_errors"], "detail": found["dollar_errors"]},
            "us_unit_figures_match_cited_runs": {"passed": not found["us_unit_errors"],
                                                 "detail": found["us_unit_errors"]}}


def sim_references(decision, report, calls):
    refs = set(SIM.findall(json.dumps(decision) + "\n" + report))
    missing = [f"sim:{ref} was not recorded" for ref in sorted(refs, key=int) if ref not in calls]
    return {"passed": not missing, "detail": missing}


def recorded_calls(transcript, work=None):
    """Full responses saved by `farmsim` in the workspace first; the harness may shorten long tool
    outputs in the transcript, dropping the call label. Transcript outputs fill any gap."""
    calls = {}
    for path in sorted(Path(work).glob(".sim-response-*.json")) if work else []:
        try:
            calls[path.stem.rsplit("-", 1)[1]] = json.loads(path.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
    if not Path(transcript).exists():
        return calls
    for line in Path(transcript).read_text().splitlines():
        message = json.loads(line)
        for part in message.get("parts", []):
            text = part.get("output", "")
            match = re.search(r"Simulator call sim:(\d+)", text)
            if match and match[1] not in calls:
                try:
                    calls[match[1]] = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
                except (ValueError, json.JSONDecodeError):
                    continue
    return calls


def evaluate(decision, form, board, simulated, report="", available=(), opened=(), calls=None):
    """board is independently re-run with decision weights/limits, never agent data."""
    def result(errors):
        return {"passed": not errors, "detail": errors}
    out = {}
    weights = decision.get("weights")
    out["form_weights"] = result([] if weights == form["weights"] else ["Weights differ from the form"])
    out["limits_grounded_in_form_or_quote"] = result(ground_limits(decision, form))
    try:
        scoreboard.validate({"weights": weights, "limits": decision.get("limits")})
        rotation = decision["recommended"]["rotation"]
        key = canonical(rotation)
        union = dict(form["limits"]) | decision["limits"]
        bad = limit_violations(rotation, simulated["metrics"], board["existing"]["metrics"],
                               board["existing"]["rotation"], union)
        out["all_limits_and_feasibility"] = result(bad)
        rows = board["rows"] + [board["existing"]]
        matching = [r for r in rows if canonical(r["rotation"]) == key]
        accepted = any(r.get("tied_with_best") or r.get("rank") == 1 for r in matching)
        override = decision["recommended"].get("override_reason") or ""
        cited = set(CARD.findall(str(override)))
        if not accepted:
            accepted = (bool(cited) and cited <= set(opened) & set(available)
                        and any(r.get("rank", 999) <= 15 for r in matching))
        out["independent_ranking_or_opened_card_override"] = result(
            [] if accepted else ["Not top/tied; no valid opened-card override within the top 15"])
    except (ValueError, KeyError, TypeError) as exc:
        out["all_limits_and_feasibility"] = result([str(exc)])
        out["independent_ranking_or_opened_card_override"] = result(["Invalid/unverifiable recommendation"])
    cited = set(CARD.findall(json.dumps(decision) + "\n" + report))
    out["cards_exist_and_opened"] = result(
        [f"card:{c}: {'missing' if c not in available else 'not opened'}" for c in sorted(cited)
         if c not in available or c not in opened])
    out["sim_references_recorded"] = sim_references(decision, report, calls or {})
    out["simulator_numbers"] = numeric_claims(decision, report, calls or {}, form)
    out["dollar_figures_match_cited_runs"] = result(out["simulator_numbers"].pop("dollar_errors"))
    out["us_unit_figures_match_cited_runs"] = result(out["simulator_numbers"].pop("us_unit_errors"))
    out["all_passed"] = all(v["passed"] for v in out.values() if not v.get("advisory"))
    return out
