"""US-unit copies of the simulator's metric results, so the agent quotes numbers instead of converting.

`annotate(data)` returns a copy of a response in which every metric value whose key names a per-hectare
or millimetre unit (a plain number or a `{p10, median, p90}` summary) gains a sibling key in US units,
placed right after it. Metric keys and values are never changed or removed. Prices, rates (`_per_`),
percentages, days and ranks have no such unit in their names and are left alone.
"""

from fieldshift.sim.economics import ACRES_HA

MM_PER_INCH = 25.4
LB_PER_KG = 2.20462262
US_TON_T = 1.10231131  # US short tons per metric tonne

# (metric token run, US token run, factor, decimals or None for whole numbers); longest runs first.
RULES = (
    (("kg", "n", "ha"), ("lb", "n", "acre"), LB_PER_KG / ACRES_HA, "lb"),
    (("t", "dm", "ha"), ("us", "tons", "dm", "acre"), US_TON_T / ACRES_HA, "tons"),
    (("t", "c", "ha"), ("us", "tons", "c", "acre"), US_TON_T / ACRES_HA, "tons"),
    (("t", "ha"), ("us", "tons", "acre"), US_TON_T / ACRES_HA, "tons"),
    (("usd", "ha"), ("usd", "acre"), 1 / ACRES_HA, "usd"),
    (("mm",), ("in",), 1 / MM_PER_INCH, "in"),
)
SUMMARY_KEYS = {"p10", "p25", "median", "mean", "p75", "p90", "min", "max"}
NOTE = ("US-unit copies sit next to each metric: *_usd_acre* = US dollars per acre (whole dollars), *_in* = "
        "inches (1 decimal), *_lb_n_acre* = lb N per acre, *_us_tons_*_acre* = US short tons per acre. "
        "Quote these values as printed; do not convert the metric values yourself.")


def us_key(key):
    """US sibling name and its conversion, or None when the key carries no convertible unit."""
    if not isinstance(key, str):
        return None
    tokens = key.split("_")
    if "per" in tokens:  # prices and rates such as water_charge_usd_per_mm_ha
        return None
    for metric, us, factor, kind in RULES:
        n = len(metric)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == metric:
                return "_".join(tokens[:i] + list(us) + tokens[i + n:]), factor, kind
    return None


def _round(x, kind):
    if kind == "usd":
        y = round(x)
    elif kind == "in":
        y = round(x, 1)
    elif kind == "lb":
        y = round(x) if abs(x) >= 10 else round(x, 1)
    else:
        y = round(x, 2)
    return y + 0  # -0.0 -> 0.0


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _convert(value, factor, kind):
    if _number(value):
        return _round(value * factor, kind)
    if isinstance(value, dict) and value and set(value) <= SUMMARY_KEYS and all(
            v is None or _number(v) for v in value.values()):
        return {k: None if v is None else _round(v * factor, kind) for k, v in value.items()}
    return None


def annotate(data):
    """Copy of `data` with US-unit siblings added; never mutates the input."""
    if isinstance(data, list):
        return [annotate(v) for v in data]
    if not isinstance(data, dict):
        return data
    out = {}
    for key, value in data.items():
        out[key] = annotate(value)
        rule = us_key(key)
        if rule is None or rule[0] in data:
            continue
        converted = _convert(value, rule[1], rule[2])
        if converted is not None:
            out[rule[0]] = converted
    return out


def with_note(data):
    """Annotated response with a top-level note, added only when a US copy exists."""
    out = annotate(data)
    if isinstance(out, dict) and _changed(data, out):
        out["us_units_note"] = NOTE
    return out


def _changed(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        return len(after) != len(before) or any(_changed(before[k], after[k]) for k in before)
    if isinstance(before, list) and isinstance(after, list):
        return any(_changed(b, a) for b, a in zip(before, after, strict=True))
    return False
