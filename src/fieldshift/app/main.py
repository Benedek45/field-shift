"""Run with: uv run streamlit run src/fieldshift/app/main.py (theme in .streamlit/config.toml)."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from fieldshift.planner.data import county_options
from fieldshift.planner.observations import county_observations
from fieldshift.planner.search import (
    DEFAULT_WEIGHTS,
    GOALS,
    comparison_pool,
    explain,
    plan,
    rank,
    select_comparisons,
)

st.set_page_config(page_title="Field Shift · NASA-powered rotation explorer", page_icon="🌱", layout="wide")
st.caption("NASA SPACE APPS 2026  /  FIELD SHIFT")
st.title("A healthier field starts with the next rotation")
st.write("Compare four-year crop cycles, with or without a winter cover, against 45 years of NASA weather. "
         "Weigh the trade-offs, then take the questions to your agronomist.")
st.warning("Decision support, not a prescription. This tool supports and does not replace the farmer’s judgment.")

LABELS = {"income": "Margin over operating costs", "water": "Lower field water use", "soil": "Soil carbon input",
          "nitrogen": "Nitrogen carry-over", "nitrate": "Less nitrate leaching", "erosion": "Less erosion",
          "cover": "Living cover (fewer bare days)", "forage": "Forage", "resilience": "Bad-year margin",
          "market": "Crops already grown locally"}
# (label, column stem, unit, digits)
MEASURES = (
    ("Margin over operating costs", "income", "$/ha/yr", 0),
    ("Field water use (crops, covers, bare soil)", "et_total_mm", "mm/yr", 0),
    ("Cash-crop water use", "et_mm", "mm/yr", 0),
    ("Soil water at planting (2 m profile)", "water_at_planting_mm", "mm", 0),
    ("Humified soil carbon input", "soil_carbon_input_t_ha", "t C/ha/yr", 2),
    ("Erosion (screening)", "erosion_t_ha", "t/ha/yr", 2),
    ("Nitrate leached", "n_leached_kg_ha", "kg N/ha/yr", 0),
    ("Nitrogen fixed", "n_fixed_kg_ha", "kg N/ha/yr", 0),
    ("Applied fertilizer nitrogen", "n_fertilizer_kg_ha", "kg N/ha/yr", 0),
    ("Additional cover fertilizer saving", "n_fertilizer_saving_usd_ha", "$/ha/yr", 0),
    ("Days without living cover", "bare_days", "days/yr", 0),
    ("Winter cover biomass", "cover_biomass_t_ha", "t/ha", 1),
    ("Winter cover cost", "cover_cost_usd_ha", "$/ha/yr", 0),
    ("Relative yield (cash crops)", "yield_relative", "", 2),
)
KIND_COLORS = {"corn": "#d9a520", "soybean": "#6b8e23", "winter wheat": "#c48a3a", "oats": "#b9a26a",
               "alfalfa": "#3f7f5f", "hay": "#7fa86b", "sorghum": "#a0522d", "spring wheat": "#d2b48c",
               "barley": "#c2b280", "cereal rye cover": "#4f9fbf", "hairy vetch cover": "#8a6fbf",
               "double crop soybean": "#9acd32", "fallow": "#cccccc"}


@st.cache_data(show_spinner=False)
def choices():
    return county_options()


@st.cache_data(show_spinner=False)
def location(county):
    path = Path("data/processed/static/county_static.parquet")
    if not path.exists():
        return None
    static = pd.read_parquet(path, columns=["adm_id", "centroid_lat", "centroid_lon"])
    row = static.loc[static.adm_id.eq(county)]
    if not len(row) or row.isna().any(axis=None):
        return None
    return {"lat": float(row.centroid_lat.iloc[0]), "lon": float(row.centroid_lon.iloc[0])}


@st.cache_data(show_spinner=False)
def site_context(county):
    return county_observations(county)


def accept_water_suggestion(county, suggested):
    st.session_state.update({"weight_water": suggested, "accepted_nasa_" + county: True})


def run_plan(scenario, bar):
    return plan(*scenario, progress=lambda p, text: bar.progress(min(max(p, 0.), 1.), text))


with st.sidebar:
    st.header("Your field")
    options = choices()
    ids = list(options)
    with st.form("field_inputs"):
        county = st.selectbox("County (type to search)", ids, index=ids.index("US-19-169"), format_func=options.get)
        irrigated = st.toggle("Irrigated", value=False,
                              help="Ideal irrigation, no water-right or pump limits. Net application is shown separately.")
        no_equipment = st.checkbox("No new crop equipment",
                                   help="Only crops already in the detected rotation; no winter covers.")
        forage = st.checkbox("Must include a forage year")
        st.subheader("What if the weather changes?")
        rainfall = st.slider("Rainfall change (%)", -50, 50, 0, 5)
        temperature = st.slider("Temperature change (°C)", -3., 5., 0., .5)
        submitted = st.form_submit_button("Explore rotations", type="primary")
    if "scenario" not in st.session_state or submitted:
        st.session_state.scenario = (county, irrigated, rainfall / 100, temperature, no_equipment, forage)
    st.header("What matters to you?")
    st.caption("These sliders only re-rank stored results; they do not re-run the field simulation.")
    weights = {key: st.slider(label, 0, 10, DEFAULT_WEIGHTS[key], key="weight_" + key) for key, label in LABELS.items()}
    use_erosion = st.checkbox("Hard limit on erosion (screening estimate)")
    max_erosion = st.number_input("Maximum P90 erosion (t/ha/year)", min_value=0., value=10.) if use_erosion else None
    use_margin = st.checkbox("Hard limit on bad-year margin")
    min_margin = st.number_input("Minimum P10 margin ($/ha/year)", value=-500.) if use_margin else None
    margin_share = st.slider("Minimum share of existing median margin (%)", 0, 100, 70, 5) / 100

bar = st.empty()
scenario = st.session_state.scenario
if st.session_state.get("plan_key") != scenario:
    try:
        with st.spinner("Opening the cached county plan, or simulating every rotation on the accelerator…"):
            st.session_state.plan = run_plan(scenario, bar)
            st.session_state.baseline_plan = (run_plan(scenario[:2] + (0., 0.) + scenario[4:], bar)
                                              if scenario[2] or scenario[3] else None)
            st.session_state.plan_key = scenario
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        st.error(f"Could not compute the plan: {error}")
        st.stop()
result = st.session_state.plan
baseline_plan = st.session_state.baseline_plan
bar.empty()

st.subheader(result["label"])
centroid = location(result["adm_id"])
if centroid is not None:
    st.map(pd.DataFrame([centroid]), latitude="lat", longitude="lon", zoom=5, height=180)
st.caption(f"{result['search_note']} Each rotation: {result['trajectories_per_rotation']} runs (every cycle phase × "
           f"consecutive weather windows), {result['spin_years']} spin-up years, then one scored cycle. "
           f"{result['device']} · first computation {result['total_seconds']:.1f} s.")
what_if = bool(result["rain_change"] or result["temp_change"])
st.info(f"Yield: **{result['yield_path']}** on historical weather. "
        + ("A weather what-if multiplies that baseline by the FAO-33 water-response ratio (edited ÷ historical weather) "
           "for the same crop, year and field state, so a zero edit reproduces the baseline exactly. " if what_if else "")
        + "P10–P90 ranges span historical weather windows and cycle phases; they are not confidence intervals.")
if what_if:
    st.caption(f"Active what-if: rainfall {result['rain_change']:+.0%}, temperature {result['temp_change']:+.1f} °C. "
               "Warming changes phenology and scales ET₀ with the Hargreaves temperature factor; humidity, radiation "
               "and CO₂ are unchanged. Cells show the change against historical weather.")

ranked = comparison_pool(rank(result["table"], weights, max_erosion, min_margin), margin_share, weights)
comparisons, ties = select_comparisons(ranked, margin_share=margin_share, weights=weights,
                                       runs=result["arrays"].get("runs"),
                                       windows=len(result["arrays"]["window_start"]), details=True)
if ranked.allowed.sum() < 2:
    st.error("Fewer than two rotations meet the current limits. Relax a limit to compare alternatives.")
if not sum(weights.values()):
    st.warning("All priorities are zero; no strategy is preferred.")
baseline_row = ranked.loc[ranked.is_existing].iloc[0]
base_table = pd.DataFrame(baseline_plan["table"]).set_index("id") if baseline_plan else None

st.header("What NASA satellites show for this county")
context = site_context(result["adm_id"])
ndvi = context["ndvi"]
month_axis_obs = alt.Axis(values=[0, 59, 120, 181, 243, 304],
                         labelExpr="['Jan','Mar','May','Jul','Sep','Nov'][indexof([0,59,120,181,243,304],datum.value)]")
obs_x = alt.X("day:Q", title=None, scale=alt.Scale(domain=[0, 366]), axis=month_axis_obs)
left, right = st.columns(2)
with left:
    st.subheader("Greenness and the existing calendar")
    bands = alt.Chart(context["low"]).mark_rect(color="#d8cfb0", opacity=.4).encode(
        x=obs_x, x2="end:Q")
    green = alt.Chart(ndvi).mark_line(color="#2f6f4f").encode(
        x=obs_x, y=alt.Y("ndvi:Q", title="NDVI", scale=alt.Scale(domain=[0, 1])))
    st.altair_chart(bands + green, width="stretch")
    calendar = []
    for j, ci in enumerate(result["rotations"][0][0]):
        sow = result["arrays"]["sow"][ci]
        end = result["arrays"]["operation"][ci]
        prev = result["arrays"]["prev_len"]
        valid = (sow >= 0) & (end >= 0) & (np.asarray(result["years"]) >= 2016)
        if valid.any():
            start = int(np.median(np.where(sow >= prev, sow - prev, sow)[valid]))
            stop = int(np.median(end[valid]))
            for a, b in ([(start, stop)] if start < stop else [(0, stop), (start, 365)]):
                calendar.append({"year": f"Year {j + 1}", "day": a, "end": b,
                                 "crop": result["crops"][ci].replace("_", " ")})
    if calendar:
        st.altair_chart(alt.Chart(pd.DataFrame(calendar)).mark_bar().encode(
            x=obs_x, x2="end:Q", y=alt.Y("year:N", title=None, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color("crop:N", title=None)).properties(height=110),
            width="stretch")
    st.caption(f"NASA Terra MODIS MOD13A2, county cropland NDVI, 2016–2025 mean. Shading: "
               f"{context['low_days']} low-greenness days/year (NDVI <0.3), NOT proven bare soil: snow and dormancy "
               "also lower NDVI. Calendar: sowing to harvest operation using USDA state dates where available "
               "(otherwise maturity proxy), not satellite-detected field dates. "
               "Pre-2000 adjusted NOAA AVHRR is deliberately excluded.")
with right:
    st.subheader("Root-zone moisture")
    moisture = context["moisture"]
    if len(moisture):
        st.altair_chart(alt.Chart(moisture).mark_line(opacity=.7).encode(
            x=alt.X("month:O", title="Month"), y=alt.Y("smap_rootzone:Q", title="Root-zone moisture [m³/m³]"),
            detail="year:N", color=alt.Color("summer:N", title=None,
                                            scale=alt.Scale(domain=["D2+ summer", "No D2+ summer"],
                                                            range=["#c0392b", "#7c9b9a"])),
            tooltip=["year", "month", "smap_rootzone", "summer"]), width="stretch")
    else:
        st.info("SMAP root-zone observations are missing here.")
    st.caption("NASA SMAP SPL4SMGP Level-4 root-zone moisture (radiometer observations + land model), "
               "2016–2025 monthly means; red years had D2+ drought somewhere in this county during summer.")
    share = context["drought_share"]
    st.write(f"**Summer drought: {share:.0%} of years**" if np.isfinite(share) else "Summer drought history missing.")
    drought = context["drought"]
    if len(drought):
        st.altair_chart(alt.Chart(drought).mark_bar(color="#c48a3a").encode(
            x=alt.X("year:O", title="Year"), y=alt.Y("max_d2_area_pct:Q", title="Maximum summer D2+ area [%]")),
            width="stretch")
    st.caption("US Drought Monitor (USDA/NOAA/NDMC), 2000–2025: any June–August week with any county area "
               "in D2 or worse. D2 is cumulative; D3 and D4 are not added again. Not a NASA-only product.")
trend = context["grace_trend"]
if len(context["grace"]):
    st.altair_chart(alt.Chart(context["grace"]).mark_line(color="#4f9fbf").encode(
        x=alt.X("date:T", title=None), y=alt.Y("grace_tws:Q", title="Total water storage anomaly [cm]")),
        width="stretch")
if np.isfinite(trend):
    st.write(f"**GRACE total water-storage trend: {trend:+.2f} cm/year.** "
             + ("Storage has declined over the observation period." if trend < 0 else
                "Storage has increased over the observation period."))
st.caption("NASA GRACE/GRACE-FO gravity satellites, JPL mascons, 2002–2024; only observed months. "
           "Roughly 3° mascons: nearby counties share a signal. Includes groundwater, soil moisture, surface water "
           "and snow; this is not a field well measurement or proof of groundwater depletion alone.")
if context["suggest_water"] and st.session_state.get("accepted_nasa_" + result["adm_id"]):
    st.caption(f"You accepted a water-priority suggestion. Current setting: {weights['water']}/10; "
               "you can override it at any time.")
elif context["suggest_water"] and not st.session_state.get("ignore_nasa_" + result["adm_id"]):
    suggested = min(10, max(weights["water"] + 2, 4))
    water_reasons = []
    if np.isfinite(trend) and trend < -.5:
        water_reasons.append(f"GRACE storage trend {trend:+.2f} cm/year (screen below −0.5)")
    if np.isfinite(share) and share >= .3:
        water_reasons.append(f"D2+ in {share:.0%} of summers (screen at least 30%)")
    st.info("Consider a higher water-use priority because " + "; ".join(water_reasons)
            + ". These are explicit screening thresholds, not significance tests. "
            f"Suggested slider: {suggested}/10; nothing changes unless you accept.")
    st.button("Use suggested water priority", disabled=weights["water"] >= 10,
              on_click=accept_water_suggestion, args=(result["adm_id"], suggested))
    st.button("Ignore this suggestion", on_click=lambda: st.session_state.update(
        {"ignore_nasa_" + result["adm_id"]: True}))
if context["low_days"] > 120:
    cover_shown = any(row.cover_option and row.cycle == baseline_row.cycle
                      for row in ranked.loc[ranked.id.isin([i for i, _ in comparisons])].itertuples())
    st.info(f"{context['low_days']} low-greenness days suggest checking the winter gap against field records. "
            + ("Compare the 'Existing crops plus a winter cover' strategy below." if cover_shown else
               "No cover strategy meets the current display limits; inspect all candidates or relax a limit."))


def cell(row, stem, unit, digits):
    med, low, high = row[stem + "_median"], row[stem + "_p10"], row[stem + "_p90"]
    if not np.isfinite(med):
        return "—"
    fmt = f",.{digits}f"
    text = f"{med:{fmt}} ({low:{fmt}} to {high:{fmt}})"
    if not row.is_existing:
        text += f" · Δ {med - baseline_row[stem + '_median']:+{fmt}}"
    if base_table is not None:
        text += f" · weather Δ {med - base_table.loc[int(row.id), stem + '_median']:+{fmt}}"
    return text


st.caption("Existing rotation as detected: " + result["rotation_evidence"])
for note in dict.fromkeys(result.get("baseline_warnings", [])):
    st.warning(note)
st.caption(f"Alternatives must be Pareto-optimal on all outcomes with a nonzero priority and keep at least "
           f"{margin_share:.0%} of the existing median margin (${ranked.margin_floor.iloc[0]:,.0f}/ha/year). "
           "If the existing margin is negative, alternatives must not lose more. The existing rotation remains a reference.")
for goal, goal_weights in GOALS:
    if not isinstance(goal_weights, dict):
        continue
    pool = ranked.loc[ranked.allowed]
    if len(pool):
        winner = rank(pool.to_dict("records"), goal_weights).iloc[0]
        if not bool(ranked.loc[ranked.id.eq(winner.id), "display_allowed"].iloc[0]):
            st.caption(f"{goal}: the unrestricted winner fails the margin floor or is dominated. "
                       "This label is awarded only among candidates passing both checks.")
for goal, test in ties.items():
    if test["tie"]:
        winner = ranked.loc[ranked.id.eq(test["winner"]), "rotation"].iloc[0]
        st.caption(f"{goal}: the top candidate ({winner}) scores {test['gap']:.1f} points above the existing "
                   f"rotation, inside the weather-noise tolerance of {test['tolerance']:.1f} points (1.96 × SD of the "
                   f"paired score difference over {test['draws']:,} resamples of the {test['windows']} weather windows). "
                   "The existing rotation is as good a match as any candidate under these priorities, so no separate "
                   "winner is shown.")
st.markdown("### How the strategies compare")
columns = {}
shown = ranked.loc[ranked.id.isin([i for i, _ in comparisons])]
for rotation_id, reasons in comparisons:
    row = ranked.loc[ranked.id.eq(rotation_id)].iloc[0]
    header = " / ".join(reasons) + ": " + row.rotation
    values = {"Priority match (0–100)": f"{row.score:.1f}" + ("" if row.allowed else " (fails your limits)"),
              "Margin confidence": row.income_confidence.upper()}
    for name, stem, unit, digits in MEASURES:
        if all(shown[stem + suffix].abs().lt(1e-8).all() for suffix in ("_median", "_p10", "_p90")):
            continue
        values[f"{name} [{unit}]" if unit else name] = cell(row, stem, unit, digits)
    values["Organic matter trend [%-points/decade]"] = f"{row.om_trend_pct_per_decade:+.2f}"
    if row.cover_option:
        values["Cover establishment / survival"] = f"{row.cover_established_median:.0%} (failures still cost money)"
    else:
        values["Cover establishment / survival"] = "Not sown"
    columns[header] = values
st.table(pd.DataFrame(columns))
st.caption("Each cell: median over runs (10th to 90th percentile) · Δ difference from the existing rotation. "
           "Margins exclude land and are reference scenarios, not quotes.")

st.subheader("Explore the trade-offs")
outcomes = {"Nitrate leached [kg N/ha/yr]": "n_leached_kg_ha_median",
            "Humified soil carbon input [t C/ha/yr]": "soil_carbon_input_t_ha_median",
            "Field water use [mm/yr]": "et_total_mm_median", "Erosion [t/ha/yr]": "erosion_t_ha_median",
            "Days without living cover": "bare_days_median"}
outcome = st.selectbox("Outcome on the vertical axis", list(outcomes))
scatter = ranked.copy()
scatter["strategy"] = "Other candidates"
scatter.loc[scatter.pareto, "strategy"] = "Pareto front (all active outcomes)"
scatter.loc[scatter.id.isin([i for i, _ in comparisons]), "strategy"] = "Displayed alternative"
scatter.loc[scatter.is_existing, "strategy"] = "Existing rotation"
st.altair_chart(alt.Chart(scatter).mark_point(filled=True, size=70, opacity=.8).encode(
    x=alt.X("income_median:Q", title="Median operating margin [$/ha/yr]", scale=alt.Scale(zero=False)),
    y=alt.Y(outcomes[outcome] + ":Q", title=outcome, scale=alt.Scale(zero=False)),
    color=alt.Color("strategy:N", title=None, scale=alt.Scale(
        domain=["Other candidates", "Pareto front (all active outcomes)", "Displayed alternative", "Existing rotation"],
        range=["#cccccc", "#4f9fbf", "#c48a3a", "#2f6f4f"]), legend=alt.Legend(labelLimit=400, columns=2)),
    shape="strategy:N", tooltip=["rotation", "income_median", outcomes[outcome], "pareto", "display_allowed"]),
    width="stretch")
st.caption("All candidates, stored results only. Pareto means no other allowed cycle is at least as good on "
           "every active scored outcome and strictly better on one. The multi-outcome front need not form a line "
           "in this two-dimensional view. 'Most soil carbon input' does not mean best for nitrate, profit or all soil health.")

st.markdown("### Why each strategy is here")
tabs = st.tabs([" / ".join(reasons) for _, reasons in comparisons])
for tab, (rotation_id, reasons) in zip(tabs, comparisons, strict=True):
    row = ranked.loc[ranked.id.eq(rotation_id)].iloc[0]
    with tab:
        st.markdown(f"**{row.rotation}**")
        for heading, content in explain(result, row.to_dict(), reasons, baseline_row.to_dict()).items():
            with st.expander(heading, expanded=heading in ("Why suggested", "Benefits and opportunities")):
                for item in (content if isinstance(content, list) else [content]):
                    st.markdown("- " + item)

st.header("Follow the field through one cycle")
labels_by_id = {i: " / ".join(r) + ": " + ranked.loc[ranked.id.eq(i), "rotation"].iloc[0] for i, r in comparisons}
chosen = st.selectbox("Strategy", list(labels_by_id), format_func=labels_by_id.get)
arrays = result["arrays"]
years = result["years"]
starts = [int(s) for s in arrays["window_start"]]
spin = result["spin_years"]
scored_first = [years[s + spin] for s in starts]
first_year = st.select_slider("Replay an observed weather cycle starting in", options=scored_first,
                              value=scored_first[len(scored_first) // 2])
window = scored_first.index(first_year)
cycle, option, double = result["rotations"][chosen]
crops = result["crops"]
prev_len = arrays["prev_len"]


def to_calendar(y, offset):
    """Harvest-year window offset -> (calendar year, day of year starting at 0)."""
    if offset < prev_len[y]:
        return years[y] - 1, int(offset)
    return years[y], int(offset - prev_len[y])


segments, flowers = [], []
for step in range(-1, len(cycle)):
    y = starts[window] + spin + step
    ci = cycle[step % len(cycle)]
    name = crops[ci].replace("_", " ")
    sow, end = int(arrays["sow"][ci, y]), int(arrays["end"][ci, y])
    if sow >= 0 and end >= 0:
        (y0, d0), (y1, d1) = to_calendar(y, sow), to_calendar(y, end)
        if y0 < y1:
            segments += [(y0, d0, 365, name, "crop"), (y1, 0, d1, name, "crop")]
        else:
            segments.append((y1, d0, d1, name, "crop"))
        if int(arrays["operation"][ci, y]) > d1:
            segments.append((years[y], d1, int(arrays["operation"][ci, y]), name + " (mature, awaiting harvest)",
                             "standing"))
    fs, fe = int(arrays["flower_start"][ci, y]), int(arrays["flower_end"][ci, y])
    if fs >= 0:
        flowers.append({"year": str(to_calendar(y, fs)[0]), "start": to_calendar(y, fs)[1],
                        "end": to_calendar(y, fe)[1], "what": f"{name} flowering"})
    nxt = cycle[(step + 1) % len(cycle)]
    if option and arrays["cover_alive"].size and y + 1 < len(years):
        cover = result["covers"][option - 1]
        traits = result["crop_traits"]
        eligible = (not traits[crops[ci]]["perennial"]["value"] and not traits[crops[nxt]]["perennial"]["value"]
                    and not traits[crops[nxt]]["autumn_sown"]["value"] and not traits[crops[ci]]["fallow"]["value"])
        if eligible and arrays["cover_alive"][option - 1, ci, y]:
            gap = int(traits[crops[nxt]]["cover_termination_gap_days"]["value"])
            start = int(arrays["operation"][ci, y]) + 1
            stop = int(arrays["plant"][nxt, y + 1]) - gap
            label = cover.replace("_", " ")
            segments += [(years[y], start, 365, label, "cover"), (years[y] + 1, 0, max(stop, 0), label, "cover")]
        elif eligible:
            segments.append((years[y], int(arrays["operation"][ci, y]) + 1, 365, "cover sown, failed to establish",
                             "failed"))
frame = pd.DataFrame(segments, columns=["year", "start", "end", "what", "kind"])
frame = frame.loc[frame.year.between(first_year, first_year + len(cycle) - 1)]
frame["year"] = frame.year.astype(str)
months = pd.DataFrame({"start": [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
                       "end": [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], "month": range(1, 13)})
losses = pd.DataFrame(result.get("loss_months", []))
months["insured loss share"] = (months.month.map(losses.groupby("month").indemnity.sum() / losses.indemnity.sum())
                                .fillna(0) if len(losses) else 0.)
month_axis = alt.Axis(values=months.start.tolist(), title=None, labelExpr=(
    "['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][indexof(" + str(months.start.tolist())
    + ", datum.value)]"))
day_scale = alt.Scale(domain=[0, 366], nice=False)
base = alt.Chart(months).mark_rect(color="#c0392b").encode(
    x=alt.X("start:Q", scale=day_scale, axis=month_axis), x2="end:Q",
    opacity=alt.Opacity("insured loss share:Q", scale=alt.Scale(range=[0, .35]), legend=None),
    tooltip=["month", alt.Tooltip("insured loss share:Q", format=".0%")])
bars = alt.Chart(frame).mark_bar(height=18).encode(
    y=alt.Y("year:N", title=None), x=alt.X("start:Q", scale=day_scale, axis=month_axis), x2="end:Q",
    color=alt.Color("what:N", scale=alt.Scale(domain=list(dict.fromkeys(frame.what)),
                                               range=[KIND_COLORS.get(w.split(" (")[0], "#999999")
                                                      if k != "standing" else "#d8cfb0"
                                                      for w, k in dict.fromkeys(zip(frame.what, frame.kind, strict=True))]),
                     legend=alt.Legend(title=None, orient="bottom", labelLimit=600, columns=1)),
    tooltip=["what", "year", "start", "end"])
layers = [base, bars]
flower_frame = pd.DataFrame(flowers)
if len(flower_frame):
    flower_frame = flower_frame.loc[flower_frame.year.astype(int).between(first_year, first_year + len(cycle) - 1)]
    layers.append(alt.Chart(flower_frame).mark_tick(color="#c0392b", thickness=3, size=24).encode(
        y="year:N", x="start:Q", tooltip=["what", "start", "end"]))
st.altair_chart(alt.layer(*layers).properties(height=60 * len(cycle) + 40), width="stretch")
st.caption("Bars: living crops and winter covers; white gaps are days without living cover (residue may still "
           "protect the soil). Red ticks: start of flowering, the most drought- and heat-sensitive stage. Red shading: "
           "share of this county's USDA RMA insured crop losses (2000+) by month. Planting follows USDA state 50%-planted "
           "dates where reported; covers are sown after the state 50%-harvested date and stopped before the next planting.")

per = len(starts)
row_index = chosen * per + window
ledger = pd.DataFrame(arrays["ledger"][row_index], columns=result["ledger_columns"])
ledger["year"] = [str(first_year + i) for i in range(len(ledger))]
tab_water, tab_soil, tab_n, tab_erosion = st.tabs(["Soil water", "Soil carbon", "Nitrogen", "Erosion"])
with tab_water:
    trace = pd.DataFrame({"date": pd.date_range(f"{first_year}-01-01", periods=12 * len(cycle), freq="MS"),
                          "Soil water in the 2 m profile (mm)": arrays["water_month"][row_index].reshape(-1)})
    field = "Soil water in the 2 m profile (mm)"
    trace = trace.replace([np.inf, -np.inf], np.nan).dropna()
    if len(trace):
        st.altair_chart(alt.Chart(trace).mark_line().encode(
            x=alt.X("date:T", scale=alt.Scale(domain=[trace.date.min().isoformat(), trace.date.max().isoformat()])),
            y=alt.Y(field + ":Q", scale=alt.Scale(domain=[0, max(1., float(trace[field].max()) * 1.05)]))),
            width="stretch")
    else:
        st.info("No finite soil-water trace for this weather window.")
with tab_soil:
    st.bar_chart(ledger, x="year", y="soil_carbon_input_t_ha")
    st.caption("Humified plant carbon added per year (t C/ha): residues, roots and root exudates (Bolinder 2007), "
               "roots humified 2.3× more efficiently than residue (Kätterer 2011).")
with tab_n:
    st.bar_chart(ledger, x="year", y=["n_fertilizer_kg_ha", "n_fixed_kg_ha", "n_leached_kg_ha"], stack=False)
    st.caption("kg N/ha. Internal unmet demand is not an agronomic outcome and is not shown: it does not affect "
               "yield. Corn N timing uses the published two-thirds uptake by VT/R1 (Bender); the monthly mixing, "
               "mineralisation and demand priors are not a calibrated field N budget.")
with tab_erosion:
    st.bar_chart(ledger, x="year", y="erosion_t_ha")
    st.caption("Screening RUSLE t/ha/yr: annual-rain erosivity proxy, county soil K and slope, 100 m slope length.")

with st.expander("All candidates and search limits"):
    st.dataframe(ranked[["rotation", "allowed", "score", "income_median", "income_p10", "et_total_mm_median",
                         "soil_carbon_input_t_ha_median", "erosion_t_ha_median", "n_leached_kg_ha_median",
                         "bare_days_median"]], hide_index=True)
    for item in result["pruned"]:
        st.write("- " + item)
    st.download_button("Download the comparison table (CSV)", ranked.to_csv(index=False).encode(),
                       file_name=result["adm_id"] + "_rotations.csv")

with st.expander("Science in plain language"):
    st.markdown("""
- **Evapotranspiration (ET):** water leaving the field through soil evaporation and plant leaves.
- **ET₀:** atmospheric water demand over a reference grass surface.
- **Kc / Ks:** how a crop changes that demand / how dry soil limits actual water use.
- **Winter cover crop:** a crop sown after harvest to cover the soil over winter and stopped before the next planting; it is not harvested.
- **Steady state:** each rotation is run for one full cycle first, so the scored cycle starts from the soil water, nitrogen and pest state the rotation itself leaves behind.
- **NDVI:** satellite greenness; it cannot identify a bare field reliably on its own.
- **P10:** one in ten historical weather runs was worse; not a forecast or a confidence bound.
- **Soil carbon input:** plant carbon that becomes stable soil organic matter; a slow one-pool estimate, not a lab test.
""")

with st.expander("Data sources, assumptions and validation"):
    st.markdown("""
**NASA:** POWER MERRA-2 daily temperature, precipitation and humidity (POWER daily rain is rescaled to PRISM
4 km gauge-based 8-day totals for the water balance; the learned yield model keeps the POWER rain it was trained on);
POWER SYN1deg/SRB radiation (Hargreaves fill
1981–1983); Terra MODIS MOD13A2 greenness, MOD15A2H leaf area and MOD16A2GF actual ET; SMAP SPL4SMGP root-zone
moisture; GRACE/GRACE-FO JPL mascon total-water storage. Adjusted NOAA AVHRR before 2000 is a weak signal and is
not used in the visible recent greenness curve.

**Local context:** USDA NASS county yields and areas (neighbouring counties or the state where a county has no series),
state crop-progress planting, maturity and harvest dates, 2020–2025 prices received; USDA CDL crop history at 34,866
sample points; USDA RMA cause-of-loss records; US Drought Monitor; Iowa State University 2025 crop budgets,
2025 Iowa rye costs and a low-confidence mixed-date vetch budget; county soil and terrain.

**Methods:** FAO-56 daily Kc/Ks bucket; FAO-33 season-level water response; cover growth from simulated transpiration
(AquaCrop water productivity); Bolinder 2007 plant carbon inputs, Kätterer 2011 humification, RothC slow-pool decay;
RUSLE; Seifert 2017 rotation penalties; low-confidence Davis 2012 diverse-system yield transfer, not a
reproduction of the Marsden experiment's manure/underseeding/input reductions.

**Limits:** no water table or runoff; ideal irrigation; cover-crop yield effect on the next crop set to zero (grass
covers: no effect in a 65-study meta-analysis); legume-cover N is credited against fertilizer; double-crop soybean only
where it matures before frost in at least 60% of years. Results are historical scenarios, not forecasts.
""")
st.header("Validation: mixed yield evidence, failed ET check")
with st.container():
    validation = Path("data/processed/planner/validation.json")
    if validation.exists():
        report = json.loads(validation.read_text())
        multi = report["multi_county"]
        et = multi["et"]
        rows = [{"Check": "Bucket ET vs MODIS MOD16", "Held-out / anomaly r": f"{et['median_summer_anomaly_r']:.2f}",
                 "Evidence": f"{et['counties']} counties; bucket {et['mean_bucket_season_mm']:.0f} vs MODIS "
                             f"{et['mean_modis_season_mm']:.0f} mm", "Verdict": et["verdict"]}]
        for crop, check in multi["yield"].items():
            rows.append({"Check": crop.title() + " yield vs NASS", "Held-out / anomaly r": f"{check['median_r_odd']:.2f}",
                         "Evidence": f"{check['county_series']} series; even-year trends, odd-year test",
                         "Verdict": check["verdict"]})
        st.table(pd.DataFrame(rows).set_index("Check"))
        drought_rows = []
        for crop, check in multi["yield"].items():
            for year, values in check["drought_years"].items():
                drought_rows.append({"Crop": crop, "Year": year, "Observed loss [%]": values["observed_loss_pct"],
                                     "Simulated loss [%]": values["simulated_loss_pct"]})
        st.table(pd.DataFrame(drought_rows))
        tests = report.get("weather_tests")
        if tests:
            st.markdown("**Tested alternatives (300 counties, rules fixed before running)**")
            st.caption(tests["rules"])
            def test_row(name, score, adopted):
                cells = {"Variant": name, "Decision": adopted, "Corn odd-year r": f"{score['corn']['median_r_odd']:.3f}",
                         "Soybean odd-year r": f"{score['soybean']['median_r_odd']:.3f}"}
                for year in ("1983", "1988", "2012"):
                    cells[f"Corn {year} obs/sim %"] = "{:.0f} / {:.0f}".format(
                        score["corn"]["drought_years"][year]["observed_loss_pct"],
                        score["corn"]["drought_years"][year]["simulated_loss_pct"])
                    cells[f"Soy {year} obs/sim %"] = "{:.0f} / {:.0f}".format(
                        score["soybean"]["drought_years"][year]["observed_loss_pct"],
                        score["soybean"]["drought_years"][year]["simulated_loss_pct"])
                cells["Mean drought error [pts]"] = f"{score['mean_abs_drought_error_pct_points']:.1f}"
                return cells
            rain, heat = tests["rain"], tests["heat"]
            rows = [test_row("Rain: NASA POWER", rain["scores"]["power"], "reference"),
                    test_row("Rain: POWER rescaled to PRISM", rain["scores"]["prism"],
                             "ADOPTED" if rain["prism_adopted"] else "rejected")]
            for name, score in heat["scores"].items():
                decision = "reference" if name.startswith("current") else (
                    "ADOPTED" if heat["adopted"][name] else "rejected")
                rows.append(test_row(f"Yield ({heat['rain_used'].upper()} rain): {name}", score, decision))
            st.dataframe(pd.DataFrame(rows).set_index("Variant"))
            st.caption(rain["method"] + " Heat: " + heat["method"] + " Fitted slopes (per degree day): "
                       + json.dumps(heat["fitted"]["schlenker_roberts_slope_per_degree_day"]) + " and "
                       + json.dumps(heat["fitted"]["season_variant_slope_per_degree_day"]) + ". "
                       + heat["fitted"]["published_reference"])
        if "dry_wet" in et:
            split = et["dry_wet"]
            st.write(f"Dry/wet summer ET anomaly r: {split['dry_summer_anomaly_r']:.2f} / "
                     f"{split['wet_summer_anomaly_r']:.2f}. Wet-year MODIS ET vs LAI r "
                     f"{split['wet_modis_lai_r']:.2f}, versus humidity {split['wet_modis_humidity_r']:.2f}; "
                     f"bucket ET vs reference ET r {split['wet_bucket_reference_et_r']:.2f}.")
            st.caption(split["method"] + " The LAI relationship is consistent with a canopy-versus-reference-demand "
                       "difference; the humidity relationship is weak. This does not establish causality.")
        st.caption("MOD16 is an independent satellite-driven ET product, not direct field truth. "
                   "Crop mixing and missing capillary rise remain limitations; a leaf-area/humidity explanation "
                   "is a hypothesis, not an established cause of the failed anomaly check.")
        st.json(report, expanded=False)
    else:
        st.write("The validation report has not been generated yet.")
