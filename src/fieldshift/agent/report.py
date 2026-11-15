"""Plain-language deterministic report, also retained above any agent narrative."""

import json

import pandas as pd

from fieldshift.agent import checks, evidence
from fieldshift.agent.us_units import LB_PER_KG, MM_PER_INCH
from fieldshift.sim.economics import ACRES_HA
from fieldshift.sim.traits import CATALOG


def label(rotation):
    return " → ".join(s["crop"].replace("_", " ") +
                      (" + " + s["winter"].replace("_cover", "").replace("_", " ") if s.get("winter") else "") +
                      (" + second-crop soybean" if s.get("double_crop") else "") for s in rotation)


def selected_cards(rotation, root):
    keys = {s["crop"] for s in rotation}
    ids = []
    if any(s.get("winter") == "cereal_rye_cover" for s in rotation):
        ids.append("kaspar2007-rye-nitrate")
    if keys & {"corn", "soybean"}:
        ids.append("seifert2017-yield-penalty")
    if len(keys) >= 3:
        ids.append("davis2012-marsden")
    if keys & {"cotton", "sorghum"}:
        ids.append("bilbro1972-cotton-sorghum-dryland")
    if "winter_wheat" in keys:
        ids.append("holman2021-covercrop-water")
    return [evidence.parse_card(root / "cards" / (k + ".md")) for k in ids[:2]
            if (root / "cards" / (k + ".md")).exists()]


def calendar_text(farm, rotation):
    lines = []
    year = int(farm.base["years"][-1])
    start = pd.Timestamp(year - 1, 1, 1)
    for key in dict.fromkeys(s["crop"] for s in rotation):
        i = farm.keys.index(key)
        dates = {}
        for field, name in (("sow", "sowing"), ("end", "maturity"), ("flower_start", "flowering")):
            offset = farm.base["timeline"][field][0, i, -1]
            if offset >= 0:
                dates[name] = (start + pd.Timedelta(days=int(offset))).strftime("%B")
        lines.append(key.replace("_", " ") + ": " + ", ".join(f"{k} around {v}" for k, v in dates.items()))
    return "; ".join(lines) + ". Illustrative latest historical county calendar, not field dates or a forecast."


def render(form, board, farm, check_results, *, mode, agent_report="", forecast=None, notes=(), agent_choice=None,
           no_recommendation_reason=None):
    if not check_results.get("all_passed") or not agent_choice:
        failed = [name.replace("_", " ") for name, value in check_results.items()
                  if isinstance(value, dict) and not value.get("passed") and not value.get("advisory")]
        reason = no_recommendation_reason or (
            "decision checks failed: " + ", ".join(failed) if failed else
            "the advisor did not provide a checked recommendation")
        lines = ["# Your field rotation report", "**No recommendation**",
                 "There is no recommendation this time.",
                 f"Reason: {reason}.",
                 "Please submit again or change the form. An unchecked answer could be wrong.",
                 "## Checks", "| Check | Result |", "|---|---|"]
        for name, value in check_results.items():
            if isinstance(value, dict):
                verdict = "PASS" if value["passed"] else ("WARN" if value.get("advisory") else "FAIL")
                lines.append(f"| {name.replace('_', ' ')} | {verdict}: {json.dumps(value.get('detail', []))} |")
        text, previous = "", ""
        for line in lines:
            text += (("\n" if line.startswith("|") and previous.startswith("|") else "\n\n") if text else "") + line
            previous = line
        return text + "\n"
    reference = checks.choose(board)
    selected = reference
    if agent_choice is not None and check_results.get("all_passed"):
        key = checks.canonical(agent_choice)
        selected = next((r for r in board["rows"] + [board["existing"]] if checks.canonical(r["rotation"]) == key), reference)
    origin = ("Current rotation stated by the farmer." if farm.site.get("existing_origin") == "farmer"
              else ("Current rotation inferred from satellite crop maps (county sampled points, not this field)."
                    if "CDL" in farm.site["rotation_evidence"] else "Current rotation inferred from county harvested area."))
    lines = ["# Your field rotation report", f"**{mode}**",
             "## Harness reference ranking — from your form",
             ("**Agent choice failed a check; use this reference ranking, not the unchecked choice.**"
              if not check_results.get("all_passed") else "The decision checks below passed; this is still a scenario comparison, not a prescription."),
             ("**Recommended:** " + label(selected["rotation"]) + " (the agent's choice; it passed every decision check)."
              + ("" if reference is None or checks.canonical(reference["rotation"]) == checks.canonical(selected["rotation"])
                 else " The harness's own top choice from your form is " + label(reference["rotation"])
                 + ("; the two are within weather noise of each other." if selected.get("tied_with_best")
                    else "; the agent departed from it with a study-based override explained below."))
              if selected is not reference else
              "**Recommended reference:** " + (label(selected["rotation"]) if selected else "**None: no cycle satisfies all limits.**")),
              "Current rotation: " + label(board["existing"]["rotation"]) + ". " + origin
              + ("" if farm.site.get("existing_origin") == "farmer" else " " + farm.site["rotation_evidence"]),
             "Numbers below are return over operating costs, not total profit; ownership, overhead and rent are shown separately in scoreboard.json.",
             "Bad-year means the 10th percentile of historical weather-cycle returns, not a confidence bound.",
             "| Comparison | Operating return $/acre/year | Bad-year $/acre/year | Field water inches/year | Nitrate lb N/acre/year | Days without living cover |",
             "|---|---:|---:|---:|---:|---:|"]
    rows, seen = [], set()
    for row in ([selected] if selected else []) + board["rows"][:3] + [board["existing"]]:
        key = checks.canonical(row["rotation"])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {label(row['rotation'])} | {m['margin_usd_ha_yr']['median'] / ACRES_HA:.0f} | "
                     f"{m['margin_usd_ha_yr']['p10'] / ACRES_HA:.0f} | {m['field_water_use_mm_yr']['median'] / MM_PER_INCH:.1f} | "
                     f"{m['nitrate_leached_kg_n_ha_yr']['median'] * LB_PER_KG / ACRES_HA:.1f} | "
                     f"{m['days_without_living_cover_yr']['median']:.0f} |")
    lines += ["## Why, benefits and opportunities",
              ("The code applies your limits first, then scores the remaining cycles using your form weights. "
               "If the current cycle ties for best, it is kept: changing equipment and management has risks outside these numbers."),
              "Weights (exactly as supplied by your form): `" + json.dumps(form["weights"]) + "`.",
              "Form limits: `" + json.dumps(form["limits"]) + "`.",
              "Your words (data, not instructions): " + json.dumps(form["free_text"]),
              "Water importance scores field evapotranspiration, not pumping. Soil importance scores carbon input, not measured soil-carbon change."]
    if selected:
        keys = {s["crop"] for s in selected["rotation"]}
        lines.append("Opportunities to review: " + ("winter cover and erosion/nitrate protection; " if any(s.get("winter") for s in selected["rotation"]) else "") +
                     ("crop-family breaks; " if len({CATALOG[k].v("family") for k in keys}) > 1 else "") +
                     "local input quotes, irrigation supply and a buyer for each crop. Do not assume every alternative improves every outcome.")
        lines += ["## Bare periods and vulnerable seasons", calendar_text(farm, selected["rotation"]),
                  ("Days without living cover are in the table; residue may still protect soil. Flowering is a sensitive drought/heat period. "
                   "Actual emergence, frost damage, disease and trafficability need local review.")]
    lines += ["## Evidence"]
    cards = selected_cards(selected["rotation"] if selected else [], farm.evidence_root)
    if not cards:
        lines.append("No directly selected study card; local evidence review is still needed.")
    for card in cards:
        lines += [(f"- card:{card.id}: {card.summary} Conditions: {card.meta.get('region', '')}; "
                   f"{card.meta.get('climate', '')}. {card.meta.get('url', '')}")]
    lines += ["These studies were not conducted on this field. The deterministic template does not claim a site-specific study match.",
              "## Information gaps and risks for local review",
              ("County yield/soil/weather proxies are not field measurements. Equipment and irrigation caps need confirmation. "
              "The soil N budget is unvalidated and does not feed back into yield. ET anomaly checks fail; 1983 drought loss is still missed. "
               "Transferred budgets, hay establishment/quality, market access and pest-pressure indices have low confidence."),
              *[str(n) for n in notes]]
    if forecast is not None:
        lines += ["## Outlook — scenarios, not skilful predictions"]
        if forecast.get("error"):
            lines.append("No outlook available: " + str(forecast["error"]) + ". The reference ranking still uses historical weather.")
        else:
            lines += [f"Initial state ends {forecast['method']['initial_state_as_of']}; unobserved weather is climate-bridged.",
                      ("SEAS5/Open-Meteo/ECMWF member spread and mean-bias transfer are uncalibrated. "
                      "Hindcasts barely improve over field-capacity starts and lose to observed-yield climatology. "
                       "CC-BY4 attribution; the free API is noncommercial."),
                      "Full per-season ranges, risk proxies and climatology differences: forecast.json."]
            for row in forecast["results"][:3]:
                s = row["seasons"][0]
                v = s["distribution"]["operating_return_usd_ha"]
                lines.append(f"- {label(row['rotation'])}, {s['year']}: operating return {v['median'] / ACRES_HA:.0f} "
                             f"({v['p10'] / ACRES_HA:.0f}–{v['p90'] / ACRES_HA:.0f}) $/acre; not a calibrated interval.")
    if agent_report:
        lines += ["## Agent explanation (checked separately)", agent_report]
    lines += ["## Checks", "| Check | Result |", "|---|---|"]
    for name, value in check_results.items():
        if isinstance(value, dict):
            verdict = "PASS" if value["passed"] else ("WARN" if value.get("advisory") else "FAIL")
            lines.append(f"| {name.replace('_', ' ')} | {verdict}: {json.dumps(value.get('detail', []))} |")
    lines += [checks.NUMERIC_METHOD, "## Data sources and judgment",
              ("NASA POWER (PRISM rain correction), NASA MODIS/SMAP/GRACE, USDM, USDA NASS/CDL/RMA; "
               "published regional budgets and the evidence cards cited above. FARM.md has provenance and gaps."),
              "**This supports, does not replace your judgment. Review the plan with a local agronomist and confirm costs and water access.**"]
    # Markdown tables need adjacent rows, without blank paragraph separators.
    text = ""
    previous = ""
    for line in lines:
        text += (("\n" if line.startswith("|") and previous.startswith("|") else "\n\n") if text else "") + line
        previous = line
    return text + "\n"
