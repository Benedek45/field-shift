"""Logic behind the questionnaire app (`advisor.py`): county defaults, the price table, building the
farmer form, session folders, background jobs and the comparison table. No Streamlit here."""

import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from fieldshift.agent import chat, checks, farms, form, report
from fieldshift.sim.economics import ACRES_HA, PARTS

REPO = Path(__file__).resolve().parents[3]
APP_ROOT = Path(os.environ.get("FIELDSHIFT_APP_ROOT", REPO / "data/processed/app"))
MODELS = (chat.DEFAULT_MODEL, "opencode-go/deepseek-v4.1-flash", "opencode-go/glm-5.3-flash")
VERSION = "farm-questionnaire-v1"
START_GRACE_S = 120  # a job file in "starting" state counts as running until its worker takes the lock

ASPECTS = (
    ("money", "Money", ("Return over operating costs in a typical year: sales minus seed, fertilizer, chemicals, "
                       "fuel, labour and water. Land rent and machinery ownership are not included.")),
    ("water", "Water", ("Use less water in total: what the crops and soil evaporate from the field, rain and "
                       "irrigation together, not only what you pump.")),
    ("soil", "Soil", "Return more plant carbon (roots and residue) to the soil each year."),
    ("nitrogen_loss", "Nitrogen loss", ("Keep nitrate from washing out below the roots into tile drains and "
                                       "groundwater.")),
    ("erosion", "Erosion", "Lose less soil to water and wind."),
    ("ground_cover", "Ground cover", "Fewer days each year with no living crop or cover crop in the field."),
    ("forage", "Forage", "Grow more hay or other feed for livestock."),
    ("local_market", "Local market", ("Favour crops that are widely grown in your county, so buyers and services "
                                     "are close. This is a county production share, not a buyer contract.")),
)
RISK_HELP = {"accept bad years": "Judge rotations by the typical year only.",
             "balanced": "Give the bad year (1 in 10) some weight.",
             "protect me in bad years": "Give the bad year (1 in 10) a lot of weight."}
PRESET_LABELS = {"profit": "Profit", "steady": "Steady", "soil_builder": "Soil builder", "water_saver": "Water saver"}
SOIL_FACTS = (("slope_pct", "Slope", "%"), ("organic_matter_pct", "Organic matter", "% of soil mass"),
              ("awc_in_ft", "Water-holding capacity", "inches per foot of soil"))
AWC_IN_FT = 25.4 / .3048  # mm/m per in/ft, as in the form's unit list
COST_PARTS = tuple(p for p in PARTS if p not in ("repeat_crop", "repeat_crop_fixed"))
PART_LABELS = {"seed": "Seed", "fertilizer_other": "Fertilizer other than N", "chemicals": "Chemicals",
               "fuel_repairs": "Fuel and repairs", "labour": "Labour", "other_operating": "Other operating",
               "harvest": "Harvest", "fixed_overhead": "Machinery ownership and overhead (not in operating return)",
               "irrigation_fixed": "Irrigation system ownership (not in operating return)"}
PRICE_UNITS = (("usd/bu", "$/bu"), ("usd/lb", "$/lb"), ("usd/ton", "$/ton (US short ton)"))
# (form key, label, unit, SI key in the options economics); cover seed has no farmer-unit alias in the form.
GENERAL = (("labour_usd_hr", "Farm wage", "$/hour", "labour_usd_hr"),
           ("fertilizer_n_usd_lb", "Nitrogen fertilizer", "$/lb N", "fertilizer_n_usd_kg"),
           ("water_usd_acre_foot", "Water charge (irrigated only)", "$/acre-foot", "water_charge_usd_per_mm_ha"),
           ("energy_usd_acre_inch", "Pumping energy (irrigated only)", "$/acre-inch", "irrigation_energy_usd_per_mm_ha"),
           ("cover_seed_usd_ha", "Cover crop seed", "$/acre sown", None),
           ("land_rent_usd_acre", "Land rent (not in operating return)", "$/acre/year", "land_rent_usd_ha"))
SECTION_WORDS = (("rotation", ("current_rotation", "existing_rotation", "rotation")),
                 ("limits", ("limit", "included", "exclude", "require_crops")),
                 ("prices", ("economics", "price", "usd", "cost", "budget", "labour", "fertilizer", "rent",
                             "cover_seed", "water charge")),
                 ("priorities", ("priorities", "risk_attitude", "preset", "weight")),
                 ("farm", ("location", "county", "irrigat", "well", "slope", "organic", "water_holding", "awc",
                           "field_facts")),
                 ("weather", ("weather_view",)),
                 ("notes", ("free_text",)))
# metric, label, farmer unit, factor from SI, which tail is the bad year, decimals shown
METRICS = (("margin_usd_ha_yr", "Operating return", "$/acre/yr", 1 / ACRES_HA, "p10", 0),
           ("field_water_use_mm_yr", "Field water use", "in/yr", 1 / 25.4, "p90", 1),
           ("nitrate_leached_kg_n_ha_yr", "Nitrate leaching", "lb N/acre/yr", 1 / (.45359237 * ACRES_HA), "p90", 1),
           ("erosion_t_ha_yr", "Erosion", "ton/acre/yr", 1 / (.90718474 * ACRES_HA), "p90", 2),
           ("days_without_living_cover_yr", "Days without living cover", "days/yr", 1., "p90", 0),
           ("soil_carbon_input_t_c_ha_yr", "Soil carbon input", "lb C/acre/yr", 1000 / (.45359237 * ACRES_HA), "p10", 0))
GLOSSARY = (
    ("Operating return", ("Sales minus operating costs (seed, fertilizer, chemicals, fuel, labour, water), per acre "
                         "and year. Land rent and machinery ownership are not subtracted, so it is not profit.")),
    ("Typical year / bad year", ("Each rotation is run through every historical weather year since 1981. "
                                "Typical is the middle (median) result; bad year is the 1-in-10 worst result "
                                "(lowest return, or highest water use, nitrate, erosion and bare days). "
                                "It describes weather variation, not a confidence range.")),
    ("Field water use", "Water the crop and soil evaporate (evapotranspiration), from rain and irrigation together."),
    ("Nitrate leaching", ("Nitrate nitrogen washed below the root zone, towards tile drains and groundwater. "
                         "Lost fertilizer value and a water-quality problem.")),
    ("Erosion", "Soil lost to water and wind, estimated from slope, soil and cover."),
    ("Days without living cover", ("Days per year with no growing crop or cover crop. Residue may still protect "
                                  "the soil on some of these days.")),
    ("Soil carbon input", ("Plant carbon from roots and residue that stays in the soil each year. "
                          "An input, not a measured change in soil carbon.")),
    ("Tied with the best", "The lead of the best rotation over this one is within year-to-year weather noise."),
    ("sim:<n> and card:<id>", "Labels for the simulator run (sim) or the study (card) behind a statement."),
    ("Checks", ("Code re-runs the ranking from your answers and tests the advisor's choice: your priorities and "
               "limits copied exactly, the choice top-ranked or tied, studies opened, simulator runs real. "
               "A pass means the advice is consistent with your answers, not that it is agronomically proven.")),
)


class JobBusy(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


# -- questionnaire defaults -------------------------------------------------------------------

def county_choices():
    """Display text ('Story County, IA · US-19-169') -> adm_id."""
    from fieldshift.planner.data import county_options
    return {text: adm for adm, text in county_options().items()}


def county_defaults(adm, irrigated):
    """What the simulator already knows for this county: options (crops, prices, costs), soil, rotation."""
    from fieldshift.agent.runtime import Runtime
    with tempfile.TemporaryDirectory() as tmp:
        runtime = Runtime(root=Path(tmp))
        fid = farms.add({"adm_id": adm, "irrigated": bool(irrigated)}, tmp)
        farm = runtime.farm(fid)
        options = farm.options()
        soil = {k: float(farm.site["soil"][k]) for k in ("slope_pct", "organic_matter_pct", "awc_mm_per_m")}
        label = farm.site["label"].split(" · ")[0]
        runtime.cache.clear()
    return {"adm_id": adm, "label": label, "irrigated": bool(irrigated), "options": options,
            "soil": {"slope_pct": soil["slope_pct"], "organic_matter_pct": soil["organic_matter_pct"],
                     "awc_in_ft": soil["awc_mm_per_m"] / AWC_IN_FT},
            "soil_source": "County cropland means: OpenLandMap soil, SRTM slope (brief: Site and soil). "
                           "County confidence medium, field confidence low.",
            "rotation": options["existing_rotation"], "rotation_basis": options["existing_rotation_source"],
            "main_crops": [m["crop"] for m in options["main_crops"]],
            "winter_options": [w["crop"] for w in options["winter_options"]]}


def preset_values(preset):
    """Slider values and risk attitude of one of the form's presets."""
    weights = form.PRESETS[preset]
    values = {name: weights[key] for name, key in form.PRIORITY_KEYS.items()}
    risk = next(name for name, level in form.RISK.items() if level == weights["resilience"])
    return values, risk


def check_feasible(normal):
    """Load the validated farm once so an infeasible stated rotation or unusable budget edit is caught
    before a long run starts. Raises ValueError with the simulator's message."""
    from fieldshift.agent.runtime import Runtime
    with tempfile.TemporaryDirectory() as tmp:
        runtime = Runtime(root=Path(tmp))
        fid = farms.add(normal["farm"], tmp)
        runtime.farm(fid).economics_context()
        runtime.cache.clear()


# -- price table ----------------------------------------------------------------------------

def price_unit(crop):
    """(form unit, display unit, USD/t per 1 display unit), taken from the form's own conversions."""
    for unit, text in PRICE_UNITS:
        try:
            factor = form.economics({"crop_prices": {crop: {"value": 1, "unit": unit}}})["crop_price_usd_t"][crop]
        except farms.FarmError:
            continue
        return unit, ("$/lb lint" if crop == "cotton" else text), factor
    return "usd/t", "$/t (metric tonne)", 1.


def _unit_text(si_unit):
    return si_unit.replace("USD/ha", "$/acre").replace("/year", "/yr")


def price_rows(defaults):
    """One row per crop price and per cost part, in the crop's usual US unit, with source and confidence."""
    econ = defaults["options"]["economics"]
    by_crop = {c["crop"]: c for c in econ["crops"]}
    rows = []
    for crop in defaults["main_crops"]:
        item = by_crop.get(crop)
        if item is None:
            continue
        unit, text, factor = price_unit(crop)
        p = item["price"]
        rows.append({"key": f"{crop}|price", "crop": crop, "item": "price", "what": "Price",
                     "default": round(p["value"] / factor, 2), "unit": text, "form_unit": unit,
                     "source": p["source"], "confidence": p["confidence"]})
        for part in COST_PARTS:
            e = item["parts"].get(part)
            if e is None or (part == "irrigation_fixed" and not defaults["irrigated"] and not e["value"]):
                continue
            rows.append({"key": f"{crop}|{part}", "crop": crop, "item": part, "what": PART_LABELS[part],
                         "default": round(e["value"] / ACRES_HA, 2), "unit": _unit_text(e["unit"]),
                         "form_unit": "usd/acre", "source": e["source"], "confidence": e["confidence"]})
    return rows


def general_factor(form_key, si_key):
    """SI value per one farmer-unit value, by the form's own conversion (cover seed: per acre)."""
    if si_key is None:
        return ACRES_HA
    return form.economics({form_key: 1})[si_key]


def general_rows(defaults):
    econ = defaults["options"]["economics"]
    covers = {c["crop"]: c for c in econ["covers"]}
    rows = []
    for key, what, unit, si_key in GENERAL:
        if si_key is None:
            cover = covers.get("cereal_rye_cover") or next(iter(covers.values()), None)
            if cover is None:
                continue
            entry = cover["seed"]
            what = f"{what} ({cover['crop'].replace('_cover', '').replace('_', ' ')} default)"
        else:
            entry = econ[si_key]
        rows.append({"key": key, "crop": "", "item": key, "what": what, "unit": unit, "form_unit": key,
                     "default": round(entry["value"] / general_factor(key, si_key), 2),
                     "source": entry["source"], "confidence": entry["confidence"]})
    return rows


def apply_edits(rows, changes, edited_rows):
    """Merge data-editor edits ({row index: {"Value": x}}) into the farmer's changes {row key: value}.
    A cell set back to its default, or cleared, is no longer a farmer value."""
    out = dict(changes)
    for index, edit in edited_rows.items():
        if "Value" not in edit:
            continue
        row = rows[int(index)]
        value = edit["Value"]
        if value is None or abs(float(value) - row["default"]) < 1e-9:
            out.pop(row["key"], None)
        else:
            out[row["key"]] = float(value)
    return out


def table_frame(rows, changes):
    import pandas as pd
    return pd.DataFrame([{"Crop": r["crop"].replace("_", " "), "Item": r["what"],
                          "Value": changes.get(r["key"], r["default"]), "Unit": r["unit"],
                          "Default": r["default"],
                          "Source": "farmer" if r["key"] in changes else r["source"],
                          "Confidence": "farmer" if r["key"] in changes else r["confidence"]} for r in rows])


def economics_block(price_rows_, price_changes, general_rows_, general_changes):
    """The form's economics block: only cells the farmer changed, in the form's US units."""
    out = {}
    for row in price_rows_:
        if row["key"] not in price_changes:
            continue
        value = price_changes[row["key"]]
        if row["item"] == "price":
            out.setdefault("crop_prices", {})[row["crop"]] = {"value": value, "unit": row["form_unit"]}
        else:
            out.setdefault("crop_cost_parts_usd_acre", {}).setdefault(row["crop"], {})[row["item"]] = value
    for row in general_rows_:
        if row["key"] in general_changes:
            value = general_changes[row["key"]]
            out[row["key"]] = value * ACRES_HA if row["key"] == "cover_seed_usd_ha" else value
    return out


# -- the form -------------------------------------------------------------------------------

def build_form(q, defaults):
    """Public farmer form (fieldshift.agent.form schema) from the questionnaire answers."""
    location = {"adm_id": defaults["adm_id"], "irrigated": bool(q["irrigated"])}
    if q["irrigated"] and q.get("well_known"):
        location["well_capacity"] = {"value": float(q["well_in"]), "unit": "in/yr"}
    out = {"location": location}
    if q["rotation_mode"] == "inferred":
        out["current_rotation"] = [dict(s) for s in defaults["rotation"]]
    elif q["rotation_mode"] == "own":
        out["current_rotation"] = [{"crop": s["crop"], **({"winter": s["winter"]} if s.get("winter") else {})}
                                   for s in q["rotation"] if s.get("crop")]
    facts = {}
    for key, _, _ in SOIL_FACTS:
        if q.get("own_soil", {}).get(key):
            value = float(q["soil"][key])
            if key == "awc_in_ft":
                facts["water_holding_capacity"] = {"value": value, "unit": "in/ft"}
            else:
                facts[key] = value
    if facts:
        out["field_facts"] = facts
    out["priorities"] = {"preset": q["preset"], **{k: int(q["weights"][k]) for k, _, _ in ASPECTS},
                         "risk_attitude": q["risk"]}
    lim, limits = q.get("limits", {}), {}
    if lim.get("income_floor"):
        limits["minimum_share_current_income"] = {"value": float(lim["income_floor_pct"]), "unit": "%"}
    for key in ("no_new_machinery", "keep_forage_crop", "require_winter_cover"):
        if lim.get(key):
            limits[key] = True
    for key in ("include_crops", "exclude_crops"):
        if lim.get(key):
            limits[key] = list(lim[key])
    for key, unit in (("max_nitrate_kg_n_ha_yr", "lb_n/ac/yr"), ("max_field_water_use_mm_yr", "in/yr"),
                      ("max_erosion_t_ha_yr", "ton/ac/yr"), ("max_days_without_living_cover_yr", "days/yr")):
        if lim.get(key) is not None:
            limits[key] = {"value": float(lim[key]), "unit": unit}
    if limits:
        out["limits"] = limits
    economics = economics_block(price_rows(defaults), q.get("price_changes", {}),
                                general_rows(defaults), q.get("general_changes", {}))
    if economics:
        out["economics"] = economics
    out["weather_view"] = q["weather_view"]
    if q.get("free_text", "").strip():
        out["free_text"] = q["free_text"].strip()
    return out


def section_for(message):
    """Questionnaire section an error message belongs to."""
    text = message.casefold()
    for section, words in SECTION_WORDS:
        if any(w in text for w in words):
            return section
    return "submit"


def validate(q, defaults, *, feasibility=True):
    """(public form, normalised form, None) or (public form, None, {section: message})."""
    public = build_form(q, defaults)
    try:
        normal = form.validate(public)
        if feasibility:
            check_feasible(normal)
    except ValueError as exc:  # FarmError, InputError and budget errors are ValueErrors
        message = str(exc)
        return public, None, {section_for(message): message}
    return public, normal, None


# -- sessions -------------------------------------------------------------------------------

def new_session(label, model, root=None, profile=None):
    root = Path(root or APP_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    county, _, state = label.rpartition(",")
    slug = re.sub(r"[^a-z0-9]+", "-", county.casefold().replace(" county", "")).strip("-")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}-{slug}-{state.strip().lower()}"
    n = 1
    while path.exists():
        n += 1
        path = root / f"{stamp}-{slug}-{state.strip().lower()}-{n}"
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    (path / "qa").mkdir(mode=0o700)
    write_json(path / "session.json", {"version": VERSION, "model": model, "county": label,
                                       "created": datetime.now(UTC).isoformat(), "status": "new",
                                       "farm_id": None, "continuity": "session", "opencode_session_id": None,
                                       "cards_opened": [], "profile": profile})
    return path


def list_sessions(root=None):
    root = Path(root or APP_ROOT)
    out = []
    for path in sorted(root.glob("*/session.json"), reverse=True):
        s = read_json(path, {})
        out.append({"path": path.parent, "name": path.parent.name, "county": s.get("county", "?"),
                    "model": s.get("model", "?"), "status": s.get("status", "?"), "profile": s.get("profile"),
                    "questions": len(qa_rows(path.parent))})
    return out


def qa_rows(session):
    path = Path(session) / "qa.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# -- background jobs ------------------------------------------------------------------------

def _lock_held(session):
    path = Path(session) / ".job.lock"
    if not path.exists():
        return False
    with open(path, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def job_state(session):
    job = read_json(Path(session) / "job.json", {}) or {}
    running = _lock_held(session) or (job.get("state") == "starting"
                                      and time.time() - job.get("started", 0) < START_GRACE_S)
    if job.get("state") in ("starting", "running") and not running:
        job = dict(job, state="failed", error=job.get("error") or
                   "The background run stopped unexpectedly. Your answers are saved; try again.")
    return dict(job, running=running)


def spawn(session, kind):
    log = open(Path(session) / "worker.log", "a")  # noqa: SIM115 — handed to the child process
    os.chmod(log.name, 0o600)
    log.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} launching {kind} job with {sys.executable}\n")
    log.flush()
    # Record the exit status even when the worker fails before writing job.json.
    script = '"$0" -u -m fieldshift.app.advisor_worker "$1" "$2"; echo "worker exited with status $?"'
    proc = subprocess.Popen(["/bin/sh", "-c", script, sys.executable, kind, str(session)], cwd=REPO,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.write(f"worker pid {proc.pid}\n")
    log.close()


def start_job(session, kind, launcher=None):
    """Start one background job; refuses while another job of this session is running."""
    if job_state(session)["running"]:
        raise JobBusy("The advisor is still working on this session; wait for it to finish.")
    write_json(Path(session) / "job.json", {"kind": kind, "state": "starting", "started": time.time()})
    (launcher or spawn)(session, kind)


@contextlib.contextmanager
def job_lock(session):
    """Held by the worker for the whole job; the kernel releases it if the worker dies."""
    stream = open(Path(session) / ".job.lock", "a")  # noqa: SIM115
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise JobBusy("Another run of this session is active") from None
    try:
        yield
    finally:
        fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def progress(session):
    """Live view of the agent episode: steps, simulator calls and recent tool calls (no tool output)."""
    agent = Path(session) / "agent"
    events = chat.read_events(agent / "events.jsonl")
    steps = sum(e.get("type") == "step_finish" for e in events)
    tools = []
    for e in events:
        if e.get("type") != "tool_use":
            continue
        part = e.get("part", {})
        given = (part.get("state") or {}).get("input") or {}
        detail = given.get("command") or given.get("path") or given.get("filePath") or given.get("pattern") or ""
        detail = str(detail).replace("/work/", "")
        tools.append({"tool": part.get("tool", "?"), "detail": detail[:110] + ("…" if len(detail) > 110 else ""),
                      "status": (part.get("state") or {}).get("status", "")})
    calls_path = agent / "work" / ".farmsim_calls"
    calls = int(calls_path.read_text()) if calls_path.exists() and calls_path.read_text().strip() else 0
    return {"steps": steps, "tools": tools, "sim_calls": calls,
            "study_searches": sum("farmsim studies" in t["detail"] for t in tools),
            "cards": sorted({Path(t["detail"]).stem for t in tools if t["tool"] == "read" and "evidence/" in t["detail"]}
                            - {"INDEX"})}


# -- the agent's limits in question time ------------------------------------------------------

class SettingsLock:
    """Installed as the service's chat_form: the agent may simulate this farm but never change its form."""

    MESSAGE = ("The farmer's settings come from the questionnaire and cannot be changed here. "
               "Tell the farmer which questionnaire section and field to change; resubmitting starts a new solution.")

    def __init__(self, farm_id):
        self.farm_id = farm_id

    def gate(self, farm_id):
        if farm_id != self.farm_id:
            raise farms.FarmError("Only the farm from the farmer's questionnaire can be simulated in this session")

    def propose(self, farm_id, payload):
        raise farms.FarmError(self.MESSAGE)


# -- comparison table -----------------------------------------------------------------------

def comparison(runtime, fid, decision, board, checked):
    """Simulate recommendation, alternatives and current rotation only for a checked choice."""
    recommendation = decision.get("recommended") if isinstance(decision, dict) else None
    rec = recommendation.get("rotation") if isinstance(recommendation, dict) else None
    picks = []
    if checked.get("all_passed") and rec:
        source = "advisor"
        picks.append(("Recommended", rec))
        for i, alt in enumerate((decision.get("alternatives") or [])[:2], 1):
            if isinstance(alt, dict) and alt.get("rotation"):
                picks.append((f"Alternative {i}", alt["rotation"]))
    else:
        return {"source": "none", "rows": [], "errors": []}
    picks.append(("Current", board["existing"]["rotation"]))
    ranked = {}
    for row in board["rows"] + [board["existing"]]:
        with contextlib.suppress(ValueError):
            ranked.setdefault(checks.canonical(row["rotation"]), row)
    rows, seen, errors = [], {}, []
    for role, rotation in picks:
        try:
            key = checks.canonical(rotation)
        except ValueError as exc:
            errors.append(f"{role}: {exc}")
            continue
        if key in seen:
            if role == "Current" and seen[key] is not None:
                seen[key]["role"] += " = current"
            continue
        seen[key] = None
        try:
            result = runtime.simulate(fid, {"rotations": [farms.validate_rotation(rotation)]})["results"][0]
        except ValueError as exc:
            errors.append(f"{role} ({report.label(rotation)}): {exc}")
            continue
        board_row = ranked.get(key, {})
        rows.append({"role": role, "rotation": rotation, "label": report.label(rotation),
                     "rank": board_row.get("rank"), "tied_with_best": board_row.get("tied_with_best"),
                     "metrics": {m: result["metrics"].get(m) for m, *_ in METRICS},
                     "warnings": result.get("warnings", [])})
        seen[key] = rows[-1]
    return {"source": source, "rows": rows, "errors": errors}


def comparison_frame(comp):
    """Farmer-unit table: one row per rotation, typical and bad-year columns per metric."""
    import pandas as pd
    out = []
    for row in comp["rows"]:
        item = {"Rotation": f"{row['role']}: {row['label']}",
                "Rank": row["rank"] if row["rank"] is not None else "—"}
        for metric, label, unit, factor, bad, places in METRICS:
            value = row["metrics"].get(metric) or {}
            for name, q in (("typical", "median"), ("bad year", bad)):
                x = value.get(q)
                item[f"{label} {name} ({unit})"] = None if x is None else round(x * factor, places)
        out.append(item)
    return pd.DataFrame(out)


def checks_rows(checked):
    rows = []
    for key, value in checked.items():
        if not isinstance(value, dict) or "passed" not in value:
            continue
        verdict = "PASS" if value["passed"] else ("WARN (advisory)" if value.get("advisory") else "FAIL")
        detail = "; ".join(map(str, value.get("detail") or []))
        rows.append({"Check": chat.CHECK_NAMES.get(key, key.replace("_", " ")), "Result": verdict,
                     "Detail": detail[:400] + ("…" if len(detail) > 400 else "")})
    return rows


def load_solution(session):
    """Everything the page shows about the latest checked solution, or None before the first finishes."""
    session = Path(session)
    latest = read_json(session / "latest.json")
    if not latest:
        return None
    out = dict(latest)
    out["decision"] = read_json(session / latest["decision"], {}) or {}
    out["report_text"] = (session / latest["report"]).read_text() if (session / latest["report"]).exists() else ""
    out["run"] = read_json(session / "advice/run.json", {}) or {}
    out["reference"] = read_json(session / "advice/scoreboard.json", {}) or {}
    return out
