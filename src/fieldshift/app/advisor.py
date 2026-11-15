"""Questionnaire app: uv run streamlit run src/fieldshift/app/advisor.py

The farmer fills one questionnaire (farm, current rotation, what matters, hard limits, price table,
weather view, notes). Submitting validates it with `fieldshift.agent.form`, then a background worker
runs the advise pipeline (code ranking, one agent episode, code checks). Questions about the solution
continue the same OpenCode session. Sessions live in data/processed/app/<time>-<county>/.
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from fieldshift.agent import form, report
from fieldshift.app import advisor_core as core
from fieldshift.app import profiles

JUDGMENT = ("This tool supports, and does not replace, your judgment. Check the plan with a local agronomist "
            "and confirm prices, costs and water access before changing anything.")
ROT_MODES = {"inferred": "Yes, that is my rotation", "own": "No, I will enter my own",
             "unknown": "Not sure: use the county estimate"}
WEATHER = {"record": "Long-term record: historical weather 1981–2025",
           "outlook": "Seasonal outlook as well (low-skill scenarios, not predictions)"}
PHASES = {"starting": "Starting the advisor", "ranking": "Ranking every rotation with your answers and writing the farm brief",
          "agent": "The advisor is reading the brief and the studies and running the simulator",
          "checks": "Code checks of the advisor's choice", "comparison": "Simulating the comparison table",
          "service": "Starting the simulator for your farm"}
LIMIT_MAXIMUMS = (("max_nitrate_kg_n_ha_yr", "Most nitrate leaching", "lb N/acre/yr", 30.),
                  ("max_field_water_use_mm_yr", "Most field water use", "in/yr", 25.),
                  ("max_erosion_t_ha_yr", "Most erosion", "ton/acre/yr", 2.),
                  ("max_days_without_living_cover_yr", "Most days without living cover", "days/yr", 200.))
SUGGESTIONS = ("Why not alfalfa?", "What if next summer is dry?", "What does nitrate leaching mean?")
EDITOR_PREFIXES = ("ed_", "FormSubmitter")
NO_PROFILE = "(no profile)"

ss = st.session_state


@st.cache_data(show_spinner=False)
def county_choices():
    return core.county_choices()


@st.cache_data(show_spinner="Loading what we know about this county…", max_entries=16)
def defaults_for(adm, irrigated):
    return core.county_defaults(adm, irrigated)


def init_state():
    values, risk = core.preset_values("steady")
    base = {"session": None, "model": core.chat.DEFAULT_MODEL, "model_preset": next(iter(core.MODEL_PRESETS)),
            "api_key": "", "api_key_provider": None, "base_url": "", "profile": None, "profile_choice": NO_PROFILE,
            "profile_new": "", "profile_note": "", "profile_msg": None, "q_water": "Rainfed", "q_well_known": False,
            "q_well_in": 12.0, "q_rot_mode": "inferred", "q_rotation": [], "q_rot_init": None, "q_preset": "steady",
            "q_risk": risk, "q_lim_floor": False, "q_lim_floor_pct": 80, "q_lim_no_new_machinery": False,
            "q_lim_keep_forage_crop": False, "q_lim_require_winter_cover": False, "q_lim_include_crops": [],
            "q_lim_exclude_crops": [], "q_price_changes": {}, "q_general_changes": {}, "q_table_v": 0,
            "q_weather": "record", "q_free_text": "", "q_errors": {}, "q_defaults_for": None}
    base |= {f"q_w_{k}": values[k] for k, _, _ in core.ASPECTS}
    for key, _, _, default in LIMIT_MAXIMUMS:
        base |= {f"q_max_on_{key}": False, f"q_max_{key}": default}
    for key, _, _ in core.SOIL_FACTS:
        base |= {f"q_soil_own_{key}": False, f"q_soil_{key}": 0.0}
    for key, value in base.items():
        ss.setdefault(key, value)


def snapshot():
    return {k: v for k, v in ss.items() if k.startswith("q_") and k not in ("q_errors",)}


def apply_preset(name):
    values, risk = core.preset_values(name)
    for key, value in values.items():
        ss[f"q_w_{key}"] = value
    ss.q_risk, ss.q_preset = risk, name


def open_session(path):
    saved = core.read_json(Path(path) / "questionnaire.json", {}) or {}
    for key, value in saved.items():
        ss[key] = value
    ss.q_table_v = ss.get("q_table_v", 0) + 1
    ss.q_errors, ss.session = {}, str(path)


def new_questionnaire():
    for key in [k for k in ss if k.startswith(("q_", "ed_"))]:
        del ss[key]
    ss.session = None


def on_table_edit(editor_key, rows, changes_key):
    ss[changes_key] = core.apply_edits(rows, ss[changes_key], ss[editor_key].get("edited_rows", {}))
    ss.q_table_v += 1


def section_error(name):
    if name in ss.q_errors:
        st.error(ss.q_errors[name])


def sync_defaults(defaults):
    """When the county or irrigation changes: county soil values, the inferred rotation, valid crops."""
    tag = f"{defaults['adm_id']}|{defaults['irrigated']}"
    if ss.q_defaults_for == tag:
        return
    for key, _, _ in core.SOIL_FACTS:
        if not ss[f"q_soil_own_{key}"] or ss.q_defaults_for is None and not ss[f"q_soil_{key}"]:
            ss[f"q_soil_{key}"] = round(defaults["soil"][key], 2)
    if ss.q_rot_init is None or ss.q_defaults_for is not None:
        ss.q_rot_init = [{"Crop": s["crop"], "Winter cover": s.get("winter") or "none"} for s in defaults["rotation"]]
    crops = set(defaults["main_crops"])
    for key in ("q_lim_include_crops", "q_lim_exclude_crops"):
        ss[key] = [c for c in ss[key] if c in crops]
    keys = {r["key"] for r in core.price_rows(defaults)}
    ss.q_price_changes = {k: v for k, v in ss.q_price_changes.items() if k in keys}
    ss.q_table_v += 1
    ss.q_defaults_for = tag


# -- questionnaire ---------------------------------------------------------------------------

def farm_section():
    st.subheader("Farm")
    section_error("farm")
    choices = county_choices()
    st.selectbox("County", list(choices), index=None, key="q_county",
                 placeholder="Type to search, for example Story County, IA")
    st.radio("Water", ["Rainfed", "Irrigated"], horizontal=True, key="q_water")
    irrigated = ss.q_water == "Irrigated"
    if irrigated:
        st.checkbox("I know my well capacity", key="q_well_known")
        if ss.q_well_known:
            st.number_input("Well capacity (inches of water per year)", 0.0, 196.0, step=0.5, key="q_well_in")
        else:
            st.caption("Without a well capacity, irrigation is unlimited in the simulation (a scenario, not your supply).")
    if not ss.q_county:
        st.info("Pick your county to see what we already know about it.")
        return None
    try:
        defaults = defaults_for(choices[ss.q_county], irrigated)
    except Exception as exc:  # noqa: BLE001 — shown as a message, not a stack trace
        st.error(f"Could not load this county ({type(exc).__name__}): {exc}")
        return None
    sync_defaults(defaults)
    st.markdown("**What we already know** (county values; confirm or use your own)")
    st.caption(defaults["soil_source"])
    cols = st.columns(len(core.SOIL_FACTS))
    for col, (key, label, unit) in zip(cols, core.SOIL_FACTS, strict=True):
        with col:
            own = st.checkbox(f"My own {label.lower()}", key=f"q_soil_own_{key}")
            st.number_input(f"{label} ({unit})", min_value=0.0, step=0.1, format="%.2f", key=f"q_soil_{key}",
                            disabled=not own, help=f"County value {defaults['soil'][key]:.2f} {unit}")
    return defaults


def rotation_section(defaults):
    st.subheader("Current rotation")
    section_error("rotation")
    st.markdown(f"County estimate: **{report.label(defaults['rotation'])}**")
    st.caption("Basis: " + defaults["rotation_basis"])
    st.radio("Is this your current rotation?", list(ROT_MODES), format_func=ROT_MODES.get, key="q_rot_mode")
    if ss.q_rot_mode != "own":
        return
    st.caption("One row per year, in order (1 to 6 years). Add or delete rows at the bottom of the table.")
    edited = st.data_editor(
        pd.DataFrame(ss.q_rot_init or [{"Crop": defaults["main_crops"][0], "Winter cover": "none"}]),
        key=f"ed_rotation_{ss.q_table_v}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"Crop": st.column_config.SelectboxColumn("Crop", options=defaults["main_crops"], required=True),
                       "Winter cover": st.column_config.SelectboxColumn(
                           "Winter cover", options=["none", *defaults["winter_options"]], required=True)})
    ss.q_rotation = [{"crop": r["Crop"], "winter": "" if r["Winter cover"] in (None, "none") else r["Winter cover"]}
                     for r in edited.to_dict("records") if r.get("Crop")]


def priorities_section():
    st.subheader("What matters")
    section_error("priorities")
    st.caption("0 = does not matter, 5 = matters most. A preset fills the sliders; you can then move any of them.")
    cols = st.columns(len(core.PRESET_LABELS))
    for col, (key, label) in zip(cols, core.PRESET_LABELS.items(), strict=True):
        col.button(label + (" ✓" if ss.q_preset == key else ""), key=f"preset_{key}", on_click=apply_preset,
                   args=(key,), width="stretch")
    left, right = st.columns(2)
    for i, (key, label, text) in enumerate(core.ASPECTS):
        with left if i % 2 == 0 else right:
            st.slider(label, 0, 5, key=f"q_w_{key}")
            st.caption(text)
    st.radio("Bad years (1 in 10)", list(form.RISK), key="q_risk", horizontal=True,
             captions=[core.RISK_HELP[k] for k in form.RISK])


def limits_section(defaults):
    st.subheader("Hard limits")
    section_error("limits")
    st.caption("The code removes every rotation that breaks one of these before ranking.")
    a, b = st.columns(2)
    with a:
        st.checkbox("Keep at least this share of my current income", key="q_lim_floor")
        st.slider("Share of current operating return (%)", 0, 100, step=5, key="q_lim_floor_pct",
                  disabled=not ss.q_lim_floor)
        st.checkbox("No new machinery (only crops and covers I already grow)", key="q_lim_no_new_machinery")
        st.checkbox("Keep a forage crop", key="q_lim_keep_forage_crop")
        st.checkbox("Require a winter cover crop", key="q_lim_require_winter_cover")
    with b:
        st.multiselect("Crops that must be in the rotation", defaults["main_crops"], key="q_lim_include_crops")
        st.multiselect("Crops to exclude", defaults["main_crops"], key="q_lim_exclude_crops")
    with st.expander("Maximums (optional; checked on the typical year)"):
        for key, label, unit, _ in LIMIT_MAXIMUMS:
            c1, c2 = st.columns([1, 2])
            c1.checkbox(label, key=f"q_max_on_{key}")
            c2.number_input(f"{label} ({unit})", min_value=0.0, key=f"q_max_{key}",
                            disabled=not ss[f"q_max_on_{key}"], label_visibility="collapsed")


def price_section(defaults):
    st.subheader("Price table")
    section_error("prices")
    st.caption("Defaults come from county and state price records and published crop budgets; each row shows "
               "its source and confidence. Change a value in the **Value** column: changed rows are marked "
               "**farmer**. Only your changes are used. Costs are per acre.")
    rows = core.price_rows(defaults)
    key = f"ed_prices_{ss.q_table_v}"
    st.data_editor(core.table_frame(rows, ss.q_price_changes), key=key, hide_index=True, height=420,
                   width="stretch", on_change=on_table_edit, args=(key, rows, "q_price_changes"),
                   disabled=["Crop", "Item", "Unit", "Default", "Source", "Confidence"],
                   column_config={"Value": st.column_config.NumberColumn("Value", min_value=0.0, step=0.01, format="%.2f"),
                                  "Default": st.column_config.NumberColumn("Default", format="%.2f"),
                                  "Source": st.column_config.TextColumn("Source", width="large")})
    st.markdown("**General costs**")
    general = core.general_rows(defaults)
    key = f"ed_general_{ss.q_table_v}"
    st.data_editor(core.table_frame(general, ss.q_general_changes).drop(columns="Crop"), key=key, hide_index=True,
                   width="stretch", on_change=on_table_edit, args=(key, general, "q_general_changes"),
                   disabled=["Item", "Unit", "Default", "Source", "Confidence"],
                   column_config={"Value": st.column_config.NumberColumn("Value", min_value=0.0, step=0.01, format="%.2f"),
                                  "Default": st.column_config.NumberColumn("Default", format="%.2f"),
                                  "Source": st.column_config.TextColumn("Source", width="large")})
    st.caption("A new wage reprices the labour hours in each crop budget unless you also change that crop's "
               "labour row. Machinery ownership and land rent are shown beside, not inside, operating return.")
    changed = len(ss.q_price_changes) + len(ss.q_general_changes)
    if changed:
        st.button(f"Reset {changed} changed value(s) to the defaults", on_click=reset_prices)


def reset_prices():
    ss.q_price_changes, ss.q_general_changes = {}, {}
    ss.q_table_v += 1


def collect():
    return {"irrigated": ss.q_water == "Irrigated", "well_known": ss.q_well_known, "well_in": ss.q_well_in,
            "rotation_mode": ss.q_rot_mode, "rotation": ss.q_rotation,
            "own_soil": {k: ss[f"q_soil_own_{k}"] for k, _, _ in core.SOIL_FACTS},
            "soil": {k: ss[f"q_soil_{k}"] for k, _, _ in core.SOIL_FACTS},
            "preset": ss.q_preset, "weights": {k: ss[f"q_w_{k}"] for k, _, _ in core.ASPECTS}, "risk": ss.q_risk,
            "limits": {"income_floor": ss.q_lim_floor, "income_floor_pct": ss.q_lim_floor_pct,
                       "no_new_machinery": ss.q_lim_no_new_machinery, "keep_forage_crop": ss.q_lim_keep_forage_crop,
                       "require_winter_cover": ss.q_lim_require_winter_cover,
                       "include_crops": ss.q_lim_include_crops, "exclude_crops": ss.q_lim_exclude_crops,
                       **{k: ss[f"q_max_{k}"] if ss[f"q_max_on_{k}"] else None for k, *_ in LIMIT_MAXIMUMS}},
            "price_changes": ss.q_price_changes, "general_changes": ss.q_general_changes,
            "weather_view": ss.q_weather, "free_text": ss.q_free_text}


def profile_notes():
    return profiles.notes(ss.profile) if ss.profile else []


def submit(defaults):
    answers = collect()
    answers["free_text"] = profiles.merge_free_text(answers["free_text"], profile_notes())
    with st.spinner("Checking your answers against the simulator…"):
        public, _normal, errors = core.validate(answers, defaults)
    if errors:
        ss.q_errors = errors
        st.rerun()
    access = model_access(ss.model)
    if access["error"]:
        ss.q_errors = {"submit": "The advisor model cannot run: " + access["error"]}
        st.rerun()
    path = core.new_session(defaults["label"], ss.model, profile=ss.profile)
    core.write_json(path / "form.json", public)
    core.write_json(path / "questionnaire.json", snapshot())
    if ss.profile:
        profiles.record_submission(ss.profile, path, snapshot(), public)
    core.start_job(path, "solve", env=access["env"])
    ss.q_errors, ss.session = {}, str(path)
    st.rerun()


def questionnaire():
    defaults = farm_section()
    if defaults is None:
        return
    rotation_section(defaults)
    priorities_section()
    limits_section(defaults)
    price_section(defaults)
    st.subheader("Weather view")
    section_error("weather")
    st.radio("Compare rotations over", list(WEATHER), format_func=WEATHER.get, key="q_weather")
    st.subheader("Anything else")
    section_error("notes")
    block = profiles.notes_block(profile_notes())
    st.text_area("Notes for the advisor (equipment, buyers, plans, worries)", key="q_free_text",
                 max_chars=8000 - len(block) - 2)
    if block:
        st.caption(f"Your approved profile notes ({len(profile_notes())}) are added after these notes, so the "
                   "advisor can use them like anything you write here.")
    section_error("submit")
    if ss.session:
        st.caption("Submitting again starts a new solution in a new session; the current one is kept.")
    if st.button("Submit and run the advisor", type="primary"):
        submit(defaults)


# -- solution --------------------------------------------------------------------------------

@st.fragment(run_every=2)
def live_progress(session):
    job = core.job_state(session)
    if not job["running"]:
        st.rerun()
    elapsed = time.time() - job.get("started", time.time())
    st.markdown(f"**{PHASES.get(job.get('phase'), PHASES['starting'])}…** ({elapsed:.0f} s)")
    if job.get("phase") in ("agent", "checks"):
        p = core.progress(session)
        st.caption(f"Model steps: {p['steps']} · simulator calls: {p['sim_calls']} · study searches: "
                   f"{p['study_searches']} · studies opened: {len(p['cards'])}")
        if p["tools"]:
            st.dataframe(pd.DataFrame(p["tools"][-8:]), hide_index=True, width="stretch")
    st.caption("Usually 2 to 5 minutes for a solution and 1 to 3 minutes for a question. You can leave this page "
               "open; the run continues in the background.")


def list_block(title, items):
    st.markdown(f"**{title}**")
    items = [i for i in (items or []) if i]
    if not items:
        st.caption("None given.")
    for item in items:
        if isinstance(item, dict):
            st.markdown(f"- {item.get('claim', '')} `{item.get('source', '')}`")
        else:
            st.markdown(f"- {item}")


def solution_section(session, job):
    st.header("2 · Solution")
    state = core.read_json(Path(session) / "session.json", {}) or {}
    st.caption(f"Session `{Path(session).name}` · {state.get('county')} · model {state.get('model')}")
    if job["running"] and job.get("kind") == "solve":
        live_progress(session)
        return
    solution = core.load_solution(session)
    if solution is None:
        st.error(job.get("error") or "This session has no finished solution.")
        return
    run, decision, checked = solution["run"], solution["decision"], solution["checks"]
    episode = run.get("episode") or {}
    st.caption(f"{solution.get('mode')} · {run.get('timings', {}).get('total_seconds', 0):.0f} s"
               + (f" · served model {episode['served_model']}" if episode.get("served_model") else "")
               + (f" · updated in {solution['source']}" if solution["source"] != "solution" else ""))
    recommendation = decision.get("recommended")
    rec = recommendation.get("rotation") if isinstance(recommendation, dict) else None
    no_recommendation = (not checked.get("all_passed") or not rec
                         or run.get("recommendation") is None
                         or episode.get("error") is not None
                         or episode.get("status") not in (None, "completed"))
    if no_recommendation:
        failed = [row["Check"] for row in core.checks_rows(checked) if row["Result"].startswith("FAIL")]
        reason = run.get("no_recommendation_reason") or (
            "decision checks failed: " + ", ".join(failed) if failed else
            ("the advisor did not finish" if episode.get("error") else "the advisor did not provide a checked recommendation"))
        st.error("**No recommendation this time.** " + reason +
                 ". No answer is shown because an unchecked answer could be wrong. Submit the questionnaire again, "
                 "or change a setting." +
                 ("\n\nFailed checks: " + ", ".join(failed) if failed else ""))
        with st.expander("Code checks"):
            st.dataframe(pd.DataFrame(core.checks_rows(checked)), hide_index=True, width="stretch")
        with st.expander("Full report (REPORT.md)"):
            st.markdown(solution["report_text"])
        with st.expander("Glossary"):
            for term, text in core.GLOSSARY:
                st.markdown(f"**{term}.** {text}")
        return
    st.success(f"**Recommended: {report.label(rec)}**")
    if decision.get("why"):
        st.markdown(decision["why"])
    comp = solution["comparison"]
    st.markdown("**Comparison** (typical year and bad year, 1 in 10, over historical weather)")
    st.dataframe(core.comparison_frame(comp), hide_index=True, width="stretch")
    st.caption("Operating return excludes land rent and machinery ownership. Bad year = lowest 1-in-10 return or "
               "soil carbon, highest 1-in-10 water use, nitrate, erosion and bare days. Rank is in the code ranking "
               "of your answers. " + " ".join(comp.get("errors", [])))
    alternatives = [a for a in decision.get("alternatives") or [] if isinstance(a, dict)]
    if alternatives and checked.get("all_passed"):
        st.markdown("**Alternatives**")
        for a in alternatives[:2]:
            st.markdown(f"- {report.label(a.get('rotation') or [])}: {a.get('aim', '')}")
    tabs = st.tabs(["Why", "Evidence", "Gaps", "Risks for local review"])
    with tabs[0]:
        st.markdown(decision.get("why") or "No explanation given.")
        list_block("Benefits", decision.get("benefits"))
        list_block("Opportunities", decision.get("opportunities"))
    with tabs[1]:
        list_block("Evidence (cited study cards and simulator runs)", decision.get("evidence"))
    with tabs[2]:
        list_block("What we do not know", decision.get("information_gaps"))
    with tabs[3]:
        list_block("Check locally", decision.get("risks_for_local_review"))
        list_block("Bare periods", decision.get("bare_periods") if isinstance(decision.get("bare_periods"), list)
                   else [decision.get("bare_periods")])
        list_block("Vulnerable seasons", decision.get("vulnerable_seasons")
                   if isinstance(decision.get("vulnerable_seasons"), list) else [decision.get("vulnerable_seasons")])
    st.markdown("**Code checks**")
    st.dataframe(pd.DataFrame(core.checks_rows(checked)), hide_index=True, width="stretch")
    with st.expander("Glossary"):
        for term, text in core.GLOSSARY:
            st.markdown(f"**{term}.** {text}")
    with st.expander("Full report (REPORT.md)"):
        st.markdown(solution["report_text"])
    st.info(JUDGMENT)


# -- questions -------------------------------------------------------------------------------

def ask(session, question):
    n = len(core.qa_rows(session)) + 1
    core.write_json(Path(session) / "qa/pending.json", {"n": n, "question": question.strip()[:4000],
                                                        "asked": time.time()})
    state = core.read_json(Path(session) / "session.json", {}) or {}
    access = model_access(state.get("model") or ss.model)
    if access["error"]:
        st.warning("Questions cannot run: " + access["error"])
        return
    try:
        core.start_job(session, "ask", env=access["env"])
    except core.JobBusy as exc:
        st.warning(str(exc))
        return
    st.rerun()


def questions_section(session, job):
    st.header("3 · Questions about this solution")
    state = core.read_json(Path(session) / "session.json", {}) or {}
    if core.load_solution(session) is None:
        return
    solution = core.load_solution(session)
    if solution is None:
        return
    recommendation = solution["decision"].get("recommended")
    if not state.get("qa_available") or not solution["checks"].get("all_passed") \
            or not (recommendation.get("rotation") if isinstance(recommendation, dict) else None):
        st.info("Questions are unavailable because there is no checked recommendation. Submit the questionnaire again "
                "to try another run.")
        return
    st.caption("Answers continue the advisor session that made this solution, so it keeps its brief, studies and "
               "simulator results. The advisor cannot change your answers: it tells you which questionnaire field "
               "to change. " + (f"Questions use {state['model']}; the sidebar model applies to new submissions."
                                if ss.model != state.get("model") else ""))
    for row in core.qa_rows(session):
        with st.chat_message("user"):
            st.markdown(row["question"])
        with st.chat_message("assistant"):
            st.markdown(row["answer"])
            meta = [f"{(row.get('seconds') or {}).get('total', 0):.0f} s", ", ".join(row.get("sim_calls") or []),
                    row.get("served_model") or row.get("model")]
            st.caption(" · ".join(m for m in meta if m))
    busy = job["running"]
    if busy and job.get("kind") == "ask":
        pending = core.read_json(Path(session) / "qa/pending.json", {}) or {}
        with st.chat_message("user"):
            st.markdown(pending.get("question", ""))
        with st.chat_message("assistant"):
            live_progress(session)
    elif job.get("kind") == "ask" and job.get("state") == "failed":
        st.error("Your last question was not answered: " + str(job.get("error")))
    cols = st.columns(len(SUGGESTIONS))
    for col, text in zip(cols, SUGGESTIONS, strict=True):
        if col.button(text, disabled=busy, key=f"suggest_{text}"):
            ask(session, text)
    question = st.chat_input("Ask about the solution", disabled=busy)
    if question:
        ask(session, question)


# -- farmer memory ---------------------------------------------------------------------------

def choose_profile():
    choice = ss.profile_choice
    ss.profile = None if choice == NO_PROFILE else choice
    ss.profile_msg = None
    saved = profiles.last_questionnaire(ss.profile) if ss.profile else None
    if saved:
        new_questionnaire()
        for key, value in saved.items():
            ss[key] = value
        ss.q_table_v = ss.get("q_table_v", 0) + 1
        ss.q_errors = {}
        ss.profile_msg = "Questionnaire filled from this profile's last submission."


def create_profile():
    try:
        name = profiles.create(ss.profile_new)
    except profiles.ProfileError as exc:
        ss.profile_msg = str(exc)
        return
    ss.profile = ss.profile_choice = name
    ss.profile_new, ss.profile_msg = "", f"Profile {name} created."


def add_own_note():
    try:
        profiles.add_note(ss.profile, ss.profile_note)
        ss.profile_note, ss.profile_msg = "", "Note saved."
    except profiles.ProfileError as exc:
        ss.profile_msg = str(exc)


def profile_sidebar():
    st.markdown("**Farmer profile**")
    options = [NO_PROFILE, *profiles.names()]
    if ss.profile_choice not in options:
        ss.profile_choice = NO_PROFILE
    st.selectbox("Farmer profile", options, key="profile_choice", on_change=choose_profile,
                 label_visibility="collapsed",
                 help="A profile keeps your approved notes, your last questionnaire and your past sessions.")
    with st.expander("New profile"):
        st.text_input("Profile name", key="profile_new")
        st.button("Create profile", on_click=create_profile, width="stretch")
    if ss.profile_msg:
        st.caption(ss.profile_msg)
    if not ss.profile:
        return
    items = profiles.notes(ss.profile)
    st.caption(f"Approved notes ({profiles.size(items)}/{profiles.NOTES_LIMIT} characters). The advisor reads "
               "them as your own words.")
    for i, note in enumerate(items):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"- {note}")
        c2.button("✕", key=f"note_rm_{i}", help="Remove this note", on_click=profiles.remove_note,
                  args=(ss.profile, note))
    st.text_input("Add a note in your own words", key="profile_note", max_chars=profiles.NOTE_MAX,
                  placeholder="My landlord does not allow alfalfa.")
    st.button("Save note", on_click=add_own_note, width="stretch")
    st.caption(f"{len(profiles.sessions(ss.profile))} past session(s) in this profile.")


def decide_proposal(session, pid, approve, drop=None):
    try:
        profiles.decide(session, pid, approve, drop=drop)
    except profiles.ProfileError as exc:
        ss[f"memory_err_{pid}"] = str(exc)


def memory_section(session):
    """The advisor's proposed notes; nothing is saved without the farmer's click."""
    state = core.read_json(Path(session) / "session.json", {}) or {}
    items = profiles.proposals(session)
    pending = [p for p in items if p["status"] == "pending"]
    if not pending:
        return
    st.markdown("**Remember this for next time?** The advisor suggests saving these notes from what you said. "
                "Nothing is saved unless you approve it.")
    if not state.get("profile"):
        st.caption("This session has no farmer profile, so notes cannot be saved.")
    current = profiles.notes(state["profile"]) if state.get("profile") else []
    for p in pending:
        with st.container(border=True):
            st.markdown(f"**{p['note']}**")
            st.caption(f"Source: {p['source']} · from " + ("the solution" if p["turn"] == "solution" else p["turn"]))
            over = profiles.size(current) + len(p["note"]) > profiles.NOTES_LIMIT
            drop = None
            if over and current:
                drop = st.radio("Your notes are full. Which old note should be dropped to make room?", current,
                                key=f"memory_drop_{p['id']}")
            c1, c2 = st.columns(2)
            c1.button("Approve", key=f"memory_ok_{p['id']}", disabled=not state.get("profile"), type="primary",
                      on_click=decide_proposal, args=(session, p["id"], True, drop))
            c2.button("Reject", key=f"memory_no_{p['id']}", on_click=decide_proposal, args=(session, p["id"], False))
            if ss.get(f"memory_err_{p['id']}"):
                st.error(ss[f"memory_err_{p['id']}"])
    dropped = [p for p in items if p["status"] == "rejected_by_code"]
    if dropped:
        st.caption(f"{len(dropped)} suggestion(s) were not shown because their quoted words were not found in "
                   "what you wrote.")


# -- model and key --------------------------------------------------------------------------

def model_access(model):
    """Worker environment and status for a model; the typed key only counts for the provider it was typed for."""
    key_model = f"{ss.api_key_provider}/x" if ss.api_key_provider else model
    return core.model_environment(model, ss.api_key, ss.base_url, key_model=key_model)


def apply_model_preset():
    model = core.MODEL_PRESETS[ss.model_preset]
    if model:
        ss.model = model


def remember_key_provider():
    ss.api_key_provider = ss.model.split("/", 1)[0] if ss.api_key else None


def model_picker():
    st.selectbox("Provider", list(core.MODEL_PRESETS), key="model_preset", on_change=apply_model_preset,
                 help="Pick a preset, then edit the model if you like. Used for new submissions.")
    st.text_input("Model (provider/model id)", key="model",
                  help="For example openrouter/anthropic/claude-sonnet-5.5, anthropic/claude-sonnet-5-5, "
                       "custom/qwen3.5:9b. See README 'Choosing a model and API key'.")
    if ss.model.split("/", 1)[0] in ("custom", "ollama", "lmstudio"):
        st.text_input("Endpoint URL (FIELDSHIFT_BASE_URL)", key="base_url", placeholder="http://localhost:11434/v1",
                      help="An OpenAI-compatible server. localhost is rewritten to host.docker.internal so the "
                           "advisor container can reach this machine.")
    variable = core.key_variable(ss.model)
    st.text_input(f"API key ({variable})" if variable else "API key", key="api_key", type="password",
                  on_change=remember_key_provider,
                  help="Kept only in this browser session's memory and handed to the advisor process through its "
                       "environment. Never saved to disk. Leave empty to use the environment or fieldshift.env.")
    access = model_access(ss.model)
    st.caption("Keys found in the environment or key files: " + (", ".join(access["detected"]) or "none"))
    if access["error"]:
        st.warning(access["error"])
    else:
        route = access["route"]
        source = ", ".join(access["from_page"]) + " from this page" if access["from_page"] else \
            (", ".join(route["key_names"]) or "no key")
        st.caption(f"Ready: {route['provider']} ({source}). " + " ".join(route["notes"]))


# -- page ------------------------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        profile_sidebar()
        st.divider()
        model_picker()
        st.button("New questionnaire", on_click=new_questionnaire, width="stretch")
        sessions = core.list_sessions()
        if sessions:
            st.markdown("**Past sessions**")
            labels = {str(s["path"]): f"{s['name'][:15]} · {s['county']} · {s['status']} · {s['questions']} Q"
                      for s in sessions}
            choice = st.selectbox("Session", list(labels), format_func=labels.get, label_visibility="collapsed")
            st.button("Open", on_click=open_session, args=(choice,), width="stretch")


def main():
    st.set_page_config(page_title="Field Shift · rotation advisor", page_icon="🌱", layout="wide")
    init_state()
    sidebar()
    st.title("Field Shift · rotation advisor")
    st.caption(JUDGMENT)
    session = ss.session if ss.session and Path(ss.session).is_dir() else None
    with st.expander("1 · Questionnaire", expanded=session is None):
        questionnaire()
    if session:
        job = core.job_state(session)
        solution_section(session, job)
        questions_section(session, job)
        if not job["running"]:
            memory_section(session)


try:
    main()
except Exception as exc:  # noqa: BLE001 — never show a stack trace (st.rerun/stop are BaseExceptions)
    st.error(f"Something went wrong on this page ({type(exc).__name__}): {str(exc)[:400]}")
