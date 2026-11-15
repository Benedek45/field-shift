"""Study search over the fixed evidence library (`data/evidence/cards`), never the web.

Ranking: Okapi BM25 over each card's front matter and its Findings, Conditions and Quotes, with a small
agronomy synonym table that expands the query (expanded terms count at `SYNONYM_WEIGHT`), multiplied by a
farm-fit factor. Farm fit compares the card's stated region, climate and soil with the farm, and is reported
as `close` / `partial` / `far` with the reasons, not as an opaque number. No embedding model is used: the
library is 56 cards, and BM25 plus synonyms reaches the evaluation targets
(`data/processed/agent/study_search_eval.json`, `python -m fieldshift.agent.studies --eval`).
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from fieldshift.agent import evidence

K1, B = 1.2, 0.75
SYNONYM_WEIGHT = 0.4
FIT_FACTOR = {"close": 1.3, "partial": 1.1, "far": 1.0, None: 1.0}
MAX_TOP = 15
SNIPPETS = 3
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "without", "what", "which",
    "how", "does", "do", "my", "our", "we", "i", "me", "you", "your", "would", "will", "can", "could",
    "should", "if", "than", "then", "there", "their", "them", "they", "about", "after", "before", "over",
    "under", "between", "more", "less", "most"
}

# Each group: phrases that mean the same thing for a farmer's question. Phrases are tokenised like the cards.
SYNONYMS = (
    ("nitrate", "nitrate leaching", "n leaching", "nitrogen loss", "no3", "leaching", "tile drainage"),
    ("cover crop", "catch crop", "winter cover", "cereal rye", "rye cover"),
    ("erosion", "soil loss", "rusle", "runoff", "sediment"),
    ("irrigation", "water use", "evapotranspiration", "consumptive use", "irrigated"),
    ("alfalfa", "lucerne", "hay", "forage"),
    ("fallow", "summer fallow"),
    ("soil water", "soil moisture", "water storage", "precipitation storage", "stored water"),
    ("yield penalty", "continuous corn", "corn after corn", "monoculture"),
    ("rotation", "crop sequence", "cropping system", "diversified"),
    ("profit", "income", "net return", "economic", "margin", "profitability", "money"),
    ("soil carbon", "soil organic carbon", "soc", "organic matter", "carbon sequestration"),
    ("herbicide", "weed"),
    ("fertilizer", "nitrogen fertilizer", "n rate", "n credit", "nitrogen credit"),
    ("soybean cyst nematode", "scn", "nematode"),
    ("rootworm", "pest", "insect"),
    ("drought", "dry year", "water stress", "water limited"),
    ("double crop", "doublecrop", "double cropped"),
    ("no till", "conservation tillage", "reduced tillage", "strip till"),
    ("legume", "pulse", "pea", "vetch", "clover"),
    ("sorghum", "milo", "grain sorghum"),
    ("cotton", "lint"),
)

STATES = {
    "AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"}
NEIGHBOURS = {
    "AL": "FL GA MS TN", "AZ": "CA CO NM NV UT", "AR": "LA MO MS OK TN TX", "CA": "AZ NV OR",
    "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI", "DE": "MD NJ PA", "FL": "AL GA", "GA": "AL FL NC SC TN",
    "ID": "MT NV OR UT WA WY", "IL": "IA IN KY MO WI", "IN": "IL KY MI OH", "IA": "IL MN MO NE SD WI",
    "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV", "LA": "AR MS TX", "ME": "NH", "MD": "DE PA VA WV",
    "MA": "CT NH NY RI VT", "MI": "IN OH WI", "MN": "IA ND SD WI", "MS": "AL AR LA TN",
    "MO": "AR IA IL KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY", "NV": "AZ CA ID OR UT",
    "NH": "MA ME VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX UT", "NY": "CT MA NJ PA VT",
    "NC": "GA SC TN VA", "ND": "MN MT SD", "OH": "IN KY MI PA WV", "OK": "AR CO KS MO NM TX",
    "OR": "CA ID NV WA", "PA": "DE MD NJ NY OH WV", "RI": "CT MA", "SC": "GA NC",
    "SD": "IA MN MT ND NE WY", "TN": "AL AR GA KY MO MS NC VA", "TX": "AR LA NM OK", "UT": "AZ CO ID NM NV WY",
    "VT": "MA NH NY", "VA": "KY MD NC TN WV", "WA": "ID OR", "WV": "KY MD OH PA VA",
    "WI": "IA IL MI MN", "WY": "CO ID MT NE SD UT"}
# Named regions (longest phrases first, removed from the text before state names are matched).
REGIONS = (
    ("mississippi alluvial plain", "AR MS LA MO TN"), ("mississippi delta", "AR MS LA MO TN"),
    ("central great plains", "NE KS CO"), ("northern great plains", "ND SD MT NE"),
    ("southern great plains", "OK TX KS"), ("southern high plains", "TX NM"), ("texas high plains", "TX"),
    ("rolling plains", "TX OK"), ("high plains", "TX OK KS NE CO"), ("great plains", "ND SD NE KS OK TX CO WY MT NM"),
    ("upper midwest", "MN WI IA SD ND MI"), ("lower midwest", "MO IL IN KY KS"),
    ("midwest", "IA IL IN OH MI WI MN MO ND SD NE KS"), ("corn belt", "IA IL IN OH NE MN SD WI MO"),
    ("san joaquin valley", "CA"), ("coastal plain", "GA AL SC NC FL MS"),
    ("southeastern", "GA AL FL SC NC MS TN"), ("northeast", "NY PA VT NH ME MA CT RI NJ"),
    ("western usa", "WA OR ID CA NV UT MT WY CO AZ NM"))
PROVINCES = {"ontario": "MI NY OH PA MN", "alberta": "MT", "saskatchewan": "MT ND", "manitoba": "ND MN",
             "quebec": "NY VT NH ME"}
NATIONAL = ("united states", "usa and canada", "north america", "national")
GLOBAL = ("global", "worldwide", "many countries", "studies worldwide")
ABROAD = ("canada", "australia", "south africa", "europe", "china", "brazil", "india")

TEXTURES = (("very fine sandy loam", "medium"), ("silty clay loam", "fine"), ("sandy clay loam", "medium"),
            ("loamy sand", "coarse"), ("sandy loam", "coarse"), ("silt loam", "medium"), ("clay loam", "fine"),
            ("silty clay", "fine"), ("sandy clay", "fine"), ("fine-loamy", "medium"), ("coarse-textured", "coarse"),
            ("coarse textured", "coarse"), ("fine textured", "fine"), ("loam", "medium"), ("sand", "coarse"),
            ("clay", "fine"), ("silt", "medium"))
BROAD_SOIL = ("range of", "several soil", "all major", "varying", "coarse, medium and fine")
ARIDITY = ("arid", "semiarid", "subhumid", "humid")


# -- text --------------------------------------------------------------------------------------

SUFFIXES = ("ations", "ation", "ments", "ment", "ions", "ion", "ing", "ies", "ied", "es", "ed", "s")


def _stem(token):
    """Light suffix stripping (reduce/reduced/reduces -> reduc); keeps stems of at least 4 letters."""
    if token.isdigit() or token.endswith(("ss", "us", "is")):
        return token
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            token = token[:-len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
            break
    return token[:-1] if len(token) > 4 and token.endswith("e") else token


def tokens(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("-", " ").replace("_", " ")
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text) if t not in STOP]


SYNONYM_TOKENS = [[tuple(tokens(p)) for p in group] for group in SYNONYMS]


def _contains(seq, phrase):
    n = len(phrase)
    return n > 0 and any(tuple(seq[i:i + n]) == phrase for i in range(len(seq) - n + 1))


def expand(query, synonyms=True):
    """Query term weights: 1 for the farmer's own words, SYNONYM_WEIGHT for synonym-table additions."""
    words = tokens(query)
    weights = dict.fromkeys(words, 1.0)
    added = []
    if synonyms:
        for group in SYNONYM_TOKENS:
            if any(_contains(words, phrase) for phrase in group):
                for phrase in group:
                    for t in phrase:
                        if t not in weights:
                            weights[t] = SYNONYM_WEIGHT
                            added.append(t)
    return weights, added


# -- index -------------------------------------------------------------------------------------

def _card_text(card):
    m = card.meta
    head = [m["id"], m["citation"], m["region"], m["climate"], m["soil"], m["study_type"],
            m["crops"], m["crops"], m["topics"], m["topics"], m["topics"]]
    return " ".join(head + [card.sections.get(s, "") for s in evidence.SECTIONS])


class Index:
    def __init__(self, cards):
        self.cards = list(cards)
        self.docs = [Counter(tokens(_card_text(c))) for c in self.cards]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}

    def bm25(self, weights):
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.
            for t, w in weights.items():
                f = doc.get(t, 0)
                if f:
                    s += w * self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * length / self.avg))
            out.append(s)
        return out


_CACHE = {}


def index_for(root):
    root = Path(root)
    stamp = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in (root / "cards").glob("*.md")))
    key = str(root.resolve())
    if key not in _CACHE or _CACHE[key][0] != stamp:
        _CACHE[key] = (stamp, Index(evidence.load_cards(root)))
    return _CACHE[key][1]


# -- farm fit ----------------------------------------------------------------------------------

def texture_class(sand, clay):
    """USDA texture group of the farm's upper-soil sand and clay percentages."""
    if sand is None or clay is None or not (math.isfinite(sand) and math.isfinite(clay)):
        return None, None
    silt = 100 - sand - clay
    if clay >= 40:
        name = "clay" if silt < 40 and sand <= 45 else ("silty clay" if silt >= 40 else "sandy clay")
    elif clay >= 27:
        name = "silty clay loam" if sand <= 20 else ("clay loam" if sand <= 45 else "sandy clay loam")
    elif silt >= 80 and clay < 12:
        name = "silt"
    elif silt >= 50:
        name = "silt loam"
    elif sand >= 85:
        name = "sand"
    elif sand >= 70:
        name = "loamy sand"
    elif clay < 20 and sand > 52 or clay < 7 and silt < 50:
        name = "sandy loam"
    elif clay >= 20 and sand > 45:
        name = "sandy clay loam"
    else:
        name = "loam"
    return name, dict(TEXTURES)[name]


def farm_climate_class(annual_rain_mm, annual_et0_mm, mean_temp_c, frost_free_days, summer_rain_share):
    """Aridity by UNEP index P/ET0 (<0.2 arid, <0.5 semiarid, <0.65 dry sub-humid, else humid);
    thermal by frost-free days and mean temperature; Mediterranean = dry summer."""
    ai = annual_rain_mm / annual_et0_mm if annual_et0_mm else float("nan")
    aridity = "arid" if ai < .2 else "semiarid" if ai < .5 else "subhumid" if ai < .65 else "humid"
    if frost_free_days is not None and frost_free_days < 130:
        thermal = "cold"
    elif mean_temp_c >= 15 or (frost_free_days or 0) >= 210:
        thermal = "warm"
    else:
        thermal = "continental"
    return {"aridity": aridity, "aridity_index": round(ai, 2), "thermal": thermal,
            "mediterranean": summer_rain_share < .08}


def farm_profile(farm):
    """Region, climate and soil of a loaded simulator farm (runtime.farm(fid)); reductions on its device."""
    import pandas as pd
    import torch

    from fieldshift.agent.brief import climate
    from fieldshift.planner.data import STATE_ALPHA, _static
    adm = farm.record["adm_id"]
    daily = farm.weather()
    device = farm.base["device"]
    dates = pd.DatetimeIndex(daily.date)
    values = torch.tensor(daily[["prec", "et0", "tmin", "tmax"]].to_numpy("float32"), device=device)
    months = torch.tensor(dates.month.to_numpy(), device=device)
    years = dates.year.nunique()
    rain = float(values[:, 0].sum()) / years
    et0 = float(values[:, 1].sum()) / years
    tmean = float(((values[:, 2] + values[:, 3]) / 2).mean())
    summer = float(values[(months >= 6) & (months <= 8), 0].sum()) / years
    c = climate(farm)
    ffd = c["frost_free_days"]
    ffd = float(ffd[torch.isfinite(ffd)].median()) if bool(torch.isfinite(ffd).any()) else None
    raw = _static().loc[adm]
    sand, clay = (float(raw.get(k)) if raw.get(k) is not None else None for k in ("sand_0_30", "clay_0_30"))
    texture, group = texture_class(sand, clay)
    out = {"state": STATE_ALPHA.get(adm[3:5]), "label": farm.site["label"].split(" · ")[0],
           "irrigated": bool(farm.irrigated), "annual_rain_mm": round(rain), "annual_et0_mm": round(et0),
           "mean_temp_c": round(tmean, 1), "frost_free_days": None if ffd is None else round(ffd),
           "texture": texture, "texture_group": group, "sand_pct": sand, "clay_pct": clay}
    out.update(farm_climate_class(rain, et0, tmean, ffd, summer / rain if rain else 0))
    return out


def card_states(region):
    """(level-agnostic) US states a card's region names, and its scope: states, national, global or abroad."""
    text = region.casefold()
    found, scope = set(), "states"
    for phrase, states in REGIONS:
        if phrase in text:
            found |= set(states.split())
            text = text.replace(phrase, " ")
    for abbr, name in sorted(STATES.items(), key=lambda kv: -len(kv[1])):
        if re.search(r"\b" + re.escape(name.casefold()) + r"\b", text):
            found.add(abbr)
            text = text.replace(name.casefold(), " ")
    near = set()
    for province, states in PROVINCES.items():
        if province in text:
            near |= set(states.split())
    if not found:
        if any(w in text for w in GLOBAL):
            scope = "global"
        elif near or any(w in text for w in ABROAD):
            scope = "abroad"
        elif any(w in text for w in NATIONAL) or re.search(r"\b(us|usa)\b", text):
            scope = "national"
    return found, near, scope


def card_climate(text):
    t = text.casefold().replace("semi-arid", "semiarid").replace("sub-humid", "subhumid").replace(
        "semi-humid", "subhumid")
    aridity = set()
    if re.search(r"\bsemiarid\b", t):
        aridity.add("semiarid")
    if re.search(r"(?<!semi)\barid\b", t):
        aridity.add("arid")
    if "subhumid" in t:
        aridity.add("subhumid")
    if re.search(r"(?<!sub)(?<!semi)\bhumid\b", t):
        aridity.add("humid")
    if "dryland" in t and not aridity:
        aridity.add("semiarid")
    if "mediterranean" in t:
        aridity.add("semiarid")
    thermal = set()
    if "subtropical" in t:
        thermal.add("warm")
    if "continental" in t:
        thermal.add("continental")
    if "cold" in t or "short growing season" in t:
        thermal.add("cold")
    if "temperate" in t:
        thermal |= {"continental", "warm"}
        aridity |= set() if aridity else {"humid", "subhumid"}
    rain = None
    m = re.search(r"(\d[\d,]*)\s*mm", t)
    if m:
        rain = float(m[1].replace(",", ""))
    else:
        inches = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*inch", t)]
        if inches:
            rain = 25.4 * sum(inches) / len(inches)
    broad = any(w in t for w in ("several climatic", "all us regions", "many countries"))
    return aridity, thermal, rain, broad, "mediterranean" in t


def card_soil(text):
    t = text.casefold()
    if any(w in t for w in BROAD_SOIL):
        return "broad", []
    names, rest = [], t
    for name, group in TEXTURES:
        if name in rest:
            names.append((name, group))
            rest = rest.replace(name, " ")
    return ("stated" if names else "unknown"), names


def fit(card, farm):
    """`close` / `partial` / `far` with plain reasons, from the card's stated region, climate and soil."""
    if not farm:
        return None
    reasons = []
    m = card.meta
    states, near, scope = card_states(m["region"])
    st = farm.get("state")
    neighbours = set(NEIGHBOURS.get(st, "").split())
    if st in states:
        region = "same"
        reasons.append(f"region: covers {STATES.get(st, st)} ({m['region']})")
    elif states & neighbours or st in near:
        region = "neighbour"
        reasons.append(f"region: neighbouring {', '.join(sorted(states & neighbours) or ['province'])} ({m['region']})")
    elif states:
        region = "distant"
        reasons.append(f"region: other US states ({m['region']})")
    elif scope in ("national", "global"):
        region = "broad"
        reasons.append(f"region: {scope} synthesis ({m['region']})")
    else:
        region = "abroad"
        reasons.append(f"region: outside the US ({m['region']})")

    aridity, thermal, rain, broad, med = card_climate(m["climate"])
    fa, ft = farm["aridity"], farm["thermal"]
    order = ARIDITY
    climate = "unknown"
    words = f"farm {fa}, {ft}, about {farm['annual_rain_mm']} mm rain/yr, P/ET0 {farm['aridity_index']}"
    if rain:
        ratio = rain / max(farm["annual_rain_mm"], 1)
        climate = "match" if .7 <= ratio <= 1.4 else ("partial" if .5 <= ratio <= 2 else "mismatch")
        reasons.append(f"climate: study about {rain:.0f} mm rain/yr vs {words}")
    elif broad:
        climate = "partial"
        reasons.append(f"climate: several climates ({m['climate']}); {words}")
    elif aridity or thermal:
        gap = min((abs(order.index(a) - order.index(fa)) for a in aridity), default=0)
        thermal_ok = not thermal or ft in thermal
        climate = ("match" if gap == 0 and thermal_ok else "mismatch" if gap >= 2 else "partial")
        if med and not farm.get("mediterranean"):
            climate = "partial" if climate == "match" else climate
        reasons.append(f"climate: study {m['climate']!r} vs {words}")
    else:
        reasons.append(f"climate: not stated ({m['climate']})")
    if farm.get("irrigated") and "irrigated" not in m["climate"].casefold() and climate == "match":
        reasons.append("water: farm irrigated; the study does not say it was")

    kind, names = card_soil(m["soil"])
    soil = "unknown"
    if kind == "broad":
        soil = "broad"
        reasons.append(f"soil: range of soils ({m['soil']}); farm {farm.get('texture') or 'unknown'}")
    elif kind == "stated" and farm.get("texture_group"):
        groups = {g for _, g in names}
        soil = "match" if farm["texture_group"] in groups else "mismatch"
        reasons.append(f"soil: study {', '.join(n for n, _ in names)} vs farm {farm['texture']}")

    if region in ("same", "neighbour") and climate in ("match", "unknown") and soil != "mismatch":
        label = "close"
    elif climate == "mismatch" or (region in ("distant", "abroad") and climate != "match"):
        label = "far"
    else:
        label = "partial"
    return {"label": label, "region": region, "climate": climate, "soil": soil, "reasons": reasons}


# -- search ------------------------------------------------------------------------------------

def _lines(card):
    out = []
    for section in ("Findings", "Conditions and limits"):
        for text in evidence._bullet_text(card.sections.get(section, "")):
            out.append((section, text))
    return out


def snippets(card, weights, idf, n=SNIPPETS):
    scored = []
    for i, (section, text) in enumerate(_lines(card)):
        words = set(tokens(text))
        s = sum(w * idf.get(t, 0) for t, w in weights.items() if t in words)
        scored.append((s, -i, section, text))
    best = sorted(scored, reverse=True)[:n]
    return [{"section": section, "text": text if len(text) <= 360 else text[:357] + "…"}
            for s, _, section, text in best if s > 0] or [
        {"section": section, "text": text[:360]} for _, _, section, text in scored[:1]]


def _match_any(field, wanted):
    """Crop/topic filter: a wanted word matches a card's list entry when either contains the other."""
    have = [" ".join(tokens(x)) for x in evidence._split_list(field)]
    want = [" ".join(tokens(w)) for w in wanted]
    return any(w and h and (w in h or h in w) for w in want for h in have)


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("studies body must be a JSON object")  # noqa: TRY004 — the service maps ValueError to 422
    unknown = set(payload) - {"query", "crops", "topics", "top"}
    if unknown:
        raise ValueError(f"Unknown studies keys: {sorted(unknown)}; use query, crops, topics, top")
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must be a nonempty string of at most 500 characters")
    out = {"query": query.strip(), "top": payload.get("top", 5)}
    if not isinstance(out["top"], int) or isinstance(out["top"], bool) or not 1 <= out["top"] <= MAX_TOP:
        raise ValueError(f"top must be an integer from 1 to {MAX_TOP}")
    for key in ("crops", "topics"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value) or len(value) > 10:
            raise ValueError(f"{key} must be a list of at most 10 nonempty strings")
        out[key] = value
    return out


def search(root, payload, farm=None, *, synonyms=True, use_fit=True):
    """Ranked cards for a query. `farm` is a farm_profile() dict or None."""
    request = validate(payload)
    index = index_for(root)
    weights, added = expand(request["query"], synonyms)
    scores = index.bm25(weights)
    rows = []
    for card, raw in zip(index.cards, scores, strict=True):
        if raw <= 0:
            continue
        if request.get("crops") and not _match_any(card.meta["crops"], request["crops"]):
            continue
        if request.get("topics") and not _match_any(card.meta["topics"], request["topics"]):
            continue
        f = fit(card, farm) if farm else None
        factor = FIT_FACTOR[f["label"] if (f and use_fit) else None]
        rows.append((raw * factor, raw, card, f))
    rows.sort(key=lambda r: (-r[0], r[2].id))
    results = []
    for rank, (score, raw, card, f) in enumerate(rows[:request["top"]], 1):
        m = card.meta
        results.append({"rank": rank, "id": card.id, "cite_as": f"card:{card.id}", "file": f"evidence/{card.id}.md",
                        "citation": m["citation"], "access": m["access"], "study_type": m["study_type"],
                        "region": m["region"], "climate": m["climate"], "soil": m["soil"], "years": m["years"],
                        "score": round(score, 3), "bm25": round(raw, 3), "farm_fit": f,
                        "snippets": snippets(card, weights, index.idf)})
    labels = Counter((r["farm_fit"] or {}).get("label") for r in results)
    return {"query": request["query"], "filters": {k: request[k] for k in ("crops", "topics") if k in request},
            "synonyms_added": added,
            "method": ("BM25 (k1 1.2, b 0.75) over the fixed study library, query expanded with an agronomy "
                       f"synonym table (weight {SYNONYM_WEIGHT}), score x farm fit (close 1.3, partial 1.1, far 1.0). "
                       "No web search, no embedding model."),
            "farm": farm, "matches": len(rows), "results": results,
            "only_far_evidence": bool(results) and labels.get("close", 0) + labels.get("partial", 0) == 0,
            "note": ("Snippets only. Open the card file (evidence/<id>.md) with the read tool before citing "
                     "card:<id>; a card that was not opened fails the checks.")}


# -- evaluation --------------------------------------------------------------------------------

EVAL_FARMS = {"story": ("US-19-169", False), "akron": ("US-08-121", False), "lubbock": ("US-48-303", False),
              "fresno": ("US-06-019", True), "york": ("US-31-185", True)}
EVAL = (
    ("rye nitrate Iowa", "story", ["kaspar2007-rye-nitrate"]),
    ("longer rotation Iowa fertilizer herbicide", "story", ["davis2012-marsden", "hunt2019-marsden-nutrient"]),
    ("fallow wheat Colorado water", "akron", ["nielsen2005-legume-fallow", "nielsen2010-fallow-storage"]),
    ("Would cereal rye cut my nitrate losses?", "story", ["kaspar2007-rye-nitrate", "thapa2018-cover-leach"]),
    ("corn after corn yield penalty", "story", ["seifert2017-yield-penalty"]),
    ("catch crop soil carbon", "story", ["poeplau2015-cover-carbon", "abdalla2019-cover-meta"]),
    ("N leaching under catch crops", "story", ["thapa2018-cover-leach", "abdalla2019-cover-meta",
                                               "kaspar2007-rye-nitrate"]),
    ("lucerne in the rotation, nitrogen for the next corn", "story",
     ["cornell-factsheet35-n-credits", "hunt2019-marsden-nutrient", "davis2012-marsden"]),
    ("milo cotton rotation dryland", "lubbock", ["shp-sorghum-cotton", "bilbro1972-cotton-sorghum-dryland",
                                                 "krieg2004-rotations-limiting-water"]),
    ("cover crop water use before cotton", "lubbock", ["baxter2021-shp-covercrop-water",
                                                       "mcdonald2020-cotton-cover-notill"]),
    ("soybean cyst nematode rotation", "story", ["scn-longterm-rotation", "long2001-scn-rotation"]),
    ("corn rootworm continuous corn", "story", ["mech-rootworm-rotation"]),
    ("soil loss and crop residue", "story", ["mchunu2011-residue-carbon-erosion"]),
    ("cover crop water and tomato cotton tillage", "fresno", ["mitchell2015-sjv-covercrop-water",
                                                              "mitchell2007-sjv-transition",
                                                              "mitchell2016-ca-cotton-notill",
                                                              "veenstra2006-sjv-conservation-tillage"]),
    ("does a cover crop reduce the next corn yield", "york", ["deines2023-cover-yield",
                                                             "chahal2023-cover-crop-yield-meta",
                                                             "miguez2005-cover-corn-meta"]),
    ("rotation profit and income", "story", ["trlica2017-rotation-profit", "davis2012-marsden"]),
)


def evaluate(root, profiles, k=3):
    variants = {"bm25": {"synonyms": False, "use_fit": False},
                "bm25+synonyms": {"synonyms": True, "use_fit": False},
                "bm25+synonyms+fit": {"synonyms": True, "use_fit": True}}
    out = {"k": k, "queries": [], "hits_at_k": {}}
    for query, farm_key, expected in EVAL:
        row = {"query": query, "farm": farm_key, "expected_any": expected}
        for name, opts in variants.items():
            res = search(root, {"query": query, "top": 5}, profiles[farm_key], **opts)
            ids = [r["id"] for r in res["results"]]
            row[name] = {"top": ids[:k], "hit": any(e in ids[:k] for e in expected),
                         "fit_of_top": [r["farm_fit"]["label"] for r in res["results"][:k]]}
        out["queries"].append(row)
    for name in variants:
        hits = sum(q[name]["hit"] for q in out["queries"])
        close = sum(q[name]["fit_of_top"].count("close") for q in out["queries"])
        out["hits_at_k"][name] = {"hits": hits, "queries": len(out["queries"]),
                                  "rate": round(hits / len(out["queries"]), 3),
                                  "close_fit_in_top_k": close, "top_k_slots": k * len(out["queries"])}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true", help="Run the fixed query set and write the JSON result")
    p.add_argument("--query")
    p.add_argument("--out", type=Path, default=Path("data/processed/agent/study_search_eval.json"))
    args = p.parse_args()
    root = evidence.default_root()
    if args.query:
        print(json.dumps(search(root, {"query": args.query, "top": 5}), indent=1))
        return
    if not args.eval:
        p.error("use --eval or --query")
    import tempfile

    from fieldshift.agent import farms
    from fieldshift.agent.runtime import Runtime
    profiles = {}
    with tempfile.TemporaryDirectory() as tmp:
        runtime = Runtime(root=Path(tmp))
        for key, (adm, irrigated) in EVAL_FARMS.items():
            fid = farms.add({"adm_id": adm, "irrigated": irrigated}, tmp)
            profiles[key] = farm_profile(runtime.farm(fid))
            runtime.cache.clear()
    result = evaluate(root, profiles)
    result["farms"] = profiles
    result["method"] = search(root, {"query": "x"}, None)["method"]
    result["embedding_model"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["hits_at_k"], indent=1))
    for q in result["queries"]:
        print(f"{q['query'][:45]:45s} bm25={q['bm25']['hit']!s:5} syn={q['bm25+synonyms']['hit']!s:5} "
              f"fit={q['bm25+synonyms+fit']['hit']!s:5} {q['bm25+synonyms+fit']['top']} "
              f"{q['bm25+synonyms+fit']['fit_of_top']}")


if __name__ == "__main__":
    main()
