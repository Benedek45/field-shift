"""Cited budget scenarios and a pure accelerator-side re-pricing of field outputs.

Budget costs exclude land. Variable irrigation in a source budget is REMOVED
before simulator-applied water is priced; system ownership/repairs are retained
once, reported in the irrigation subtotal. Regional transfers are not farm quotes.
"""
import copy

import torch

from fieldshift.sim.alfalfa import POLICY as ALFALFA_POLICY
from fieldshift.sim.alfalfa import PRODUCTION_N, establishment
from fieldshift.sim.kansas import FILES as KS_FILES
from fieldshift.sim.kansas import ROOT as KS_ROOT
from fieldshift.sim.kansas import budget as kansas_budget

ACRES_HA = 2.471053814671653
MM_FOOT = 304.8
TX = "https://agecoext.tamu.edu/wp-content/uploads/2026/02/2026D2"
NE = "https://cap.unl.edu/sites/unl.edu.ianr.agecon.center-for-ag-profitability/files/media/file/2026-nebraska-crop-budgets.pdf"
NE25 = "https://cap.unl.edu/sites/unl.edu.ianr.agecon.center-for-ag-profitability/files/media/file/2025-crop-budgets-final.pdf"
CA = "https://coststudyfiles.ucdavis.edu/uploads/cs_public/1c/e2/1ce256d0-957e-4bd4-b17e-18fef4efcedd/16alfalfasjv300acfinal_41916.pdf"
ISU = "https://www.extension.iastate.edu/agdm/crops/pdf/a1-20-2025.pdf"
HPWD = "https://www.hpwd.org/files/5fa25ede6/Amended+Rules+by+HB+2443+-+Rules+-+11%7B2%7D15%7B2%7D2023+%28ID+49614%29.pdf"
WESTLANDS = "https://wwd.ca.gov/wwd-notice/2026-2027-groundwater-allocation-application-agreement-789/"
FIXED_PARTS = ("fixed_overhead", "irrigation_fixed")
PARTS = ("seed", "fertilizer_other", "chemicals", "fuel_repairs", "labour",
         "other_operating", "fixed_overhead", "irrigation_fixed", "repeat_crop", "repeat_crop_fixed", "harvest")
SCALARS = {
    "labour_usd_hr": (0, 500, "USD/hour"),
    "fertilizer_n_usd_kg": (0, 100, "USD/kg N"),
    "water_charge_usd_per_mm_ha": (0, 1000, "USD/(mm ha)"),
    "usd_per_acre_foot": (0, 100000, "USD/acre-foot"),
    "irrigation_energy_usd_per_mm_ha": (0, 1000, "USD/(mm ha)"),
    "irrigation_capacity_mm_yr": (0, 5000, "mm/year"),
    "cover_seed_usd_ha": (0, 10000, "USD/ha sown"),
    "land_rent_usd_ha": (0, 100000, "USD/ha/year"),
}


def entry(value, unit, source, confidence="low", allowed=None):
    out = {"value": value, "unit": unit, "source": source, "confidence": confidence}
    if allowed is not None:
        out["allowed_range"] = list(allowed)
    return out


def capacity(adm):
    if adm == "US-48-303":
        return entry(1.5 * MM_FOOT, "mm/year",
                     HPWD + " Rule 5.2(a): 1.50 acre-feet/contiguous acre/year; "
                     "Lubbock permitted-well proxy, not measured well capacity", "low")
    return entry(None, "mm/year", "No verified farm-wide allocation or well capacity; unlimited until farmer supplies a cap. "
                 + (WESTLANDS + " gives 1.0 AF/acre groundwater ONLY, not total Fresno supply; surface water unknown."
                    if adm.startswith("US-06-") else "Regional well capacity is not a farm allocation."), "low")


def context(site, crops, covers, double, irrigated):
    state = site.get("adm_id", "")[3:5]
    ctx = {"crops": [], "covers": [], "main_keys": [c.key for c in crops], "irrigated": irrigated, "capacity": capacity(site.get("adm_id", "")),
           "cottonseed_price": entry(240/.90718474, "USD/t seed", TX+"CottonPivot.pdf p1: $240/US ton; LOW transfer outside South Plains", "low", (0, 1000000)),
           "land_rent_usd_ha": entry(0., "USD/ha/year", "Excluded by accounting convention; no rent charged", "high", (0, 100000))}
    # TX energy is the explicit $10/acre-inch budget line. Elsewhere this is
    # a clearly labelled transfer, not an invented zero-energy fallback.
    energy, wage, nprice, water, rate_source = 10 * ACRES_HA / 25.4, 16., .725 / .45359237, 0., TX + "CottonPivot.pdf"
    if state == "31":
        # UNL 2025: 1.88 gal diesel/acre-inch at125ft/35psi, x14.12 kWh/gal;
        # UNL2026 electricity .12/kWh. Engineering proxy at NPC performance.
        energy, wage, nprice = 1.88 * 14.12 * .12 * ACRES_HA / 25.4, 27., .74 / .45359237
        rate_source = NE + " p9 electricity; " + NE25 + " diesel example/conversion table; wage27 from 2025 (hours proxy)"
    if state == "06":
        energy, water, wage = 0., 130 * ACRES_HA / MM_FOOT, 20.44
        rate_source = CA + " pp3–4: $130/AF combined water+pumping; no separate energy surcharge; labour operator $20.44/hr p6"
    if not irrigated and crops:
        nprice = crops[0].v("fertilizer_price_usd_kg")
        wage = 20.15
    if state == "20":
        from urllib.parse import quote
        rate_source = KS_ROOT+quote(KS_FILES["wheat"], safe="")+" 2026 FMG pp3–4; operator24/additional22 USD/hour"
        wage, nprice = 24., 61.30/(101*.45359237)
        energy = (24.73/14.5)*ACRES_HA/25.4
    for k, v, unit in (("labour_usd_hr", wage, "USD/hour"), ("fertilizer_n_usd_kg", nprice, "USD/kg N"),
                        ("irrigation_energy_usd_per_mm_ha", energy if irrigated else 0., "USD/(mm ha)"),
                        ("water_charge_usd_per_mm_ha", water if irrigated else 0., "USD/(mm ha)")):
        source = rate_source
        if k == "fertilizer_n_usd_kg":
            source = (NE + " p118: .74 USD/lb N" if state == "31" and irrigated
                      else TX + "CottonPivot.pdf p1: $65.25/90lb N"
                      if irrigated else crops[0].traits["fertilizer_price_usd_kg"].source if crops else "No crop")
        if k == "labour_usd_hr" and not irrigated:
            source = ISU+" p1:20.15 USD/hour; transfer outside Iowa; wage cannot reprice unidentified hours in aggregate budgets"
        if k == "water_charge_usd_per_mm_ha" and state != "06":
            source = TX + "CottonPivot.pdf: pumping energy only; zero separate water fee assumed, unverified farm quote"
        if state == "20":
            source = rate_source + "; native per-crop wage blends/N/energy rates override this regional reference; separate water fee unverified"
        ctx[k] = entry(v, unit, source + ("; regional transfer" if state not in ("48", "31", "06", "20") else ""),
                       allowed=SCALARS[k][:2])
    # TX: total specified, land rent, pumping line, labour dollars, seed,
    # N dollars, P/other fertilizer, chemicals, fuel/repairs excluding irrigation,
    # irrigation system ownership+repairs. Residual is cited other/fixed costs.
    tx = {
        "cotton": (963.29, 100, 100, 41.6, 63.6, 65.25, 24.28, 109.5, 24.92, 74, "CottonPivot"),
        "sorghum": (616.94, 100, 90, 32.8, 10.8, 60.32, 48.55, 46.12, 22.8, 70.75, "SorghumIrrigated"),
        "alfalfa": (1235.45, 130, 240, 36., 0., 21.75, 64.56, 47.9, 19.65, 97., "AlfalfaHay"),
        "winter_wheat": (695.40, 100, 140, 32., 37.5, 83.23, 54.38, 11., 20.68, 80.25, "WheatIrrigated"),
    }
    # NE2026 pp44–46 (#10),116–118 (#33),205–207 (#62).
    # Remove land opportunity/tax and each irrigation energy/system line.
    ne = {
        "corn": (1203.96-122.22-261.90, 32.83, 19.03, 148.03, 148., 24., 123.90,
                 42.67-32.83+38.13-23.70, 23.70+37.11+24.50, "pp116–118 budget33"),
        "soybean": (887.21-122.22-261.90, 52.29, 18.87, 82., 0., 0., 74.20,
                    63.94-52.29+37.79-20., 20.+38.89+25.17, "pp205–207 budget62 diesel (rate uses electric proxy)"),
        "alfalfa": (1043.76-136.22-261.90, 58.36, 61.15, 0., 0., 34.50, 11.99,
                   90.50-58.36+50.90-23.70, 23.70+37.11+24.50, "pp44–46 budget10"),
    }
    for crop in [*crops, *([double] if double is not None else [])]:
        parts = dict.fromkeys(PARTS, 0.)
        source = crop.traits["cost_usd_ha"].source
        legacy, labour_hours, harvest_ref = True, 0., 0.
        origin, native_labour, region_transfer, native_energy, ks = None, False, False, None, None
        if state == "20":
            ks = kansas_budget(crop.key, irrigated)
        if ks is not None:
            source, legacy, origin = ks["source"], False, "20"
            parts.update(ks["parts"])
            parts = {k:v*ACRES_HA for k,v in parts.items()}
            labour_hours = ks["hours"]*ACRES_HA
            harvest_ref = ks["harvest_ref"]*.90718474*ACRES_HA
            native_labour, region_transfer = True, ks["region_transfer"]
            if ks["energy"] is not None:
                native_energy = ks["energy"]*ACRES_HA/25.4
        elif irrigated and crop.key in tx and (state == "48" or (state == "06" and crop.key != "alfalfa")):
            total, land, pumping, labour, seed, nitrogen, fert, chem, fuel, fixed, name = tx[crop.key]
            source, legacy = TX + name + ".pdf p1; excludes rent and replaces pumping, system cost retained once", False
            parts.update(seed=seed, fertilizer_other=fert, chemicals=chem, fuel_repairs=fuel,
                         irrigation_fixed=fixed, labour=labour)
            parts["fixed_overhead"] = {"cotton": 50.25, "sorghum": 55.25, "alfalfa": 150.25, "winter_wheat": 50.25}[crop.key]
            parts["harvest"] = {"cotton": 221.56, "sorghum": 33., "alfalfa": 390., "winter_wheat": 30.}[crop.key]
            harvest_ref = {"cotton": 1000*.00112085, "sorghum": 55*.112085,
                           "alfalfa": 6*.90718474*ACRES_HA, "winter_wheat": 60*.06277}[crop.key]
            parts["other_operating"] = total-land-pumping-sum(parts.values())-nitrogen
            parts = {k: v * ACRES_HA for k, v in parts.items()}
            labour_hours = labour / 16 * ACRES_HA
            origin = "48"
            repair = {"cotton":19., "sorghum":15.75, "alfalfa":42., "winter_wheat":25.25}[crop.key]*ACRES_HA
            parts["irrigation_fixed"] -= repair
            parts["fuel_repairs"] += repair
            if state != "48":
                source += "; LOW cross-region South Plains budget transfer, not a California budget"
        elif irrigated and state == "31" and crop.key in ne:
            total, pumping, labour, seed, nitrogen, fert, chem, fuel, fixed, loc = ne[crop.key]
            source, legacy = NE + " " + loc + "; land tax/opportunity excluded, irrigation replaced once", False
            parts.update(seed=seed, fertilizer_other=fert, chemicals=chem, fuel_repairs=fuel,
                         irrigation_fixed=fixed, labour=labour)
            parts["fixed_overhead"] = {"corn": 107.96, "soybean": 111.57, "alfalfa": 207.44}[crop.key]
            parts["other_operating"] = total-pumping-sum(parts.values())-nitrogen
            parts = {k: v * ACRES_HA for k, v in parts.items()}
            labour_hours = labour / 27 * ACRES_HA  # explicitly a 2025-wage hours proxy
            origin = "31"
            repair = (20. if crop.key == "soybean" else 23.70)*ACRES_HA
            parts["irrigation_fixed"] -= repair
            parts["fuel_repairs"] += repair
        elif irrigated and state == "06" and crop.key == "alfalfa":
            source, legacy = CA + " p13: total2409 less land capital488 and tax157; replace water693 once. Nominal2016 dollars, LOW", False
            parts.update(fertilizer_other=74, chemicals=125, fuel_repairs=6, labour=54, irrigation_fixed=16, fixed_overhead=402)
            # UC p5:90%DM hay; express its billed mass on the model's explicit
            # 88%DM working convention. NASS sample moisture itself is unknown.
            parts["harvest"], harvest_ref = 221., 9*.90718474*ACRES_HA*.90/.88
            parts["other_operating"] = 2409-488-157-693-sum(parts.values())
            parts = {k: v * ACRES_HA for k, v in parts.items()}
            labour_hours = 54 / 20.44 * ACRES_HA  # mixed-wage bill converted to operator-equivalent hours
            origin = "06"
        elif crop.key in ("corn", "soybean", "alfalfa", "double_crop_soybean"):
            # Reconcile rounded source lines, then remove embedded N so that
            # the actual simulated fertilizer quantity is charged exactly once.
            source = ISU+" corn209bu/soy60bu/alfalfa6US-ton reference; excludes land. Other includes reference N/insurance/interest; actual-N marginal pricing retains legacy baseline"
            if crop.key == "corn":
                parts.update(seed=114.3, fertilizer_other=45.24+22.68+6.41, chemicals=47,
                             fuel_repairs=22.4+66.19, fixed_overhead=34.8+67.25)
                labour_hours = 2.55 * 2.47105
            elif crop.key in ("soybean", "double_crop_soybean"):
                parts.update(seed=62.9, fertilizer_other=27.84+32.4+6.41, chemicals=65,
                             fuel_repairs=24.2+13.5, fixed_overhead=37.6+32.32)
                labour_hours = 2.2 * 2.47105
            else:
                parts.update(seed=62.7/3, fertilizer_other=161.89+22/3, chemicals=26.3/3,
                             fuel_repairs=81.70+20.60/3, fixed_overhead=138.12+7+35.3/3)
                labour_hours = (107.47+20.15/3) / 20.15 * 2.47105
            parts = {k: v * 2.47105 for k,v in parts.items()}
            parts["labour"] = labour_hours * 20.15
            residual = crop.v("cost_usd_ha")-sum(parts.values())
            parts["other_operating"] = max(residual, 0.)
            if residual < 0:
                parts["fixed_overhead"] += residual  # source table rounding, not a new expense
            # N is now priced at the actual simulated rate, not left embedded
            # in the old aggregate. ISU's corn reference includes138lb ×$.50.
            if crop.key == "corn":
                parts["other_operating"] -= 69*2.47105
            legacy, origin = False, "19"
            source = ISU+" 2025: ownership machinery fixed; insurance/working-capital interest operating; N reference bill removed and actual N repriced"
        else:
            parts["other_operating"] = crop.v("cost_usd_ha")
            source += "; aggregate reference retained in other; no defensible hours/component split available"
        parts["repeat_crop"] = crop.v("same_crop_extra_cost_usd_ha") if legacy or origin == "19" else 0.
        if origin=="19" and crop.key=="corn":
            # ISU2025 pp2–3: total640.75-595.03=45.72/ac; embedded
            # N90.50-69=21.50 is now charged via actualN, NOT again here.
            # Ownership(41.20+64.55)-(34.80+67.25)=3.70/ac.
            parts["repeat_crop"]=(45.72-21.50-3.70)*2.47105
            parts["repeat_crop_fixed"]=3.70*2.47105
        np_value, np_source = nprice, ctx["fertilizer_n_usd_kg"]["source"]
        if legacy:
            np_value, np_source = crop.v("fertilizer_price_usd_kg"), crop.traits["fertilizer_price_usd_kg"].source
            source += "; unsplit aggregate treated OPERATING, N remains bundled; documented N-credit savings retained once, farmer N-price changes use actual-N marginal adjustment"
        elif ks is not None:
            np_value, np_source = ks["n_price"], source+"; N line dollars / N kg; incidental N in P/K fertilizer not purchased separately"
        elif origin == "19":
            np_value, np_source = .50/.45359237, ISU+" 2025 corn .50USD/lb N"
        elif irrigated and state in ("48", "06") and crop.key in ("sorghum", "winter_wheat"):
            np_value = (60.32/(100*.45359237) if crop.key == "sorghum" else 83.23/(140*.45359237))
            np_source = source + "; explicit N line dollars / nitrogen kg"
        budget_transfer = (origin is not None and origin != state) or region_transfer
        energy_transfer = irrigated and (state not in ("48","31","06","20"))
        transfer = budget_transfer or energy_transfer
        if budget_transfer:
            source += "; transfer: budget state/region/crop differs from this farm"
        cost_parts = {k: entry(v, "USD/ha/year", source, allowed=(0, 100000)) |
                      {"classification":"fixed" if k in (*FIXED_PARTS,"repeat_crop_fixed") else "operating", "transfer":budget_transfer}
                      for k,v in parts.items()}
        ctx["crops"].append({
            "crop": crop.key, "legacy": legacy, "legacy_price": crop.v("price_usd_t"),
            "n_in_aggregate":legacy, "origin_state":origin, "transfer":transfer,
            "native_labour":native_labour,
            "energy_rate":entry(native_energy if native_energy is not None else ctx["irrigation_energy_usd_per_mm_ha"]["value"],
                                "USD/(mm ha)", source if native_energy is not None else ctx["irrigation_energy_usd_per_mm_ha"]["source"], "low") | {"transfer":energy_transfer},
            "price": entry(crop.v("price_usd_t"), "USD/t", crop.traits["price_usd_t"].source,
                           crop.traits["price_usd_t"].confidence, (0, 1000000)),
            "parts": cost_parts,
            "labour_hours_ha": entry(labour_hours, "hours/ha", source + "; bill divided by cited wage; hours proxy", "low"),
            "cottonseed_usd_per_lint_t": entry(.71 * 2000 / 1000 * 240 / .90718474
                    if crop.key == "cotton" else 0., "USD/t lint",
                    TX + "CottonPivot.pdf: .71 US ton seed per1000lb lint, $240/US ton" if crop.key == "cotton" else "Not applicable"),
            "default_n_price": crop.v("fertilizer_price_usd_kg"),
            "repeat_labour_hours_ha": .25 * 2.47105 if origin == "19" and crop.key == "corn" else 0.,
            "harvest_reference_yield_t_ha": entry(harvest_ref, "t/ha", source+"; custom harvest billed quantity (zero means area-priced)", "low"),
            "n_price": entry(np_value, "USD/kg N", np_source, "low", SCALARS["fertilizer_n_usd_kg"][:2]),
        })
        if crop.key == "alfalfa":
            c = ctx["crops"][-1]
            # Remove source ANNUALISED establishment before charging a cash event.
            if state == "48" and irrigated:
                c["parts"]["fixed_overhead"]["value"] -= 100*ACRES_HA
            elif state == "06" and irrigated:
                c["parts"]["fixed_overhead"]["value"] -= 315*ACRES_HA
            elif state == "20":
                for part, removal in {"seed":28.20, "fuel_repairs":3.64, "labour":1.74}.items():
                    c["parts"][part]["value"] -= removal*ACRES_HA
                c["labour_hours_ha"]["value"] -= (.36/22+(.47+.71+.20)/24)*ACRES_HA
            elif origin == "19":
                for part, removal in {"seed":62.7/3, "fertilizer_other":22/3, "chemicals":26.3/3,
                                      "fuel_repairs":20.6/3, "labour":20.15/3,
                                      "fixed_overhead":35.3/3}.items():
                    c["parts"][part]["value"] -= removal*2.47105
                c["labour_hours_ha"]["value"] -= (1/3)*2.47105
            for p in c["parts"].values():
                p["source"] += "; v25 removes identified annualised seeding before explicit establishment. "
                if state == "20":
                    p["source"] += "Unsplit Kansas ownership retained; may retain allocated drill ownership (LOW listed-cost uncertainty)."
            c["stand_policy"] = ALFALFA_POLICY
            c["production_n_kg_ha"] = entry(PRODUCTION_N.get(state,0.)*1.12085, "kg N/ha/production year",
                                           source+"; explicit N line or N content of phosphate, included once")
            c["establishment"] = {}
            for mode in ("fall", "spring"):
                b = establishment(state, mode == "fall", irrigated)
                seed, labour = b["seed"], b["labour"]
                # Retain unsplit inputs in OTHER; never fabricate component quantities.
                parts = dict.fromkeys(PARTS, 0.)
                parts.update(seed=seed, labour=labour, fixed_overhead=b["fixed"]-b["irrigation_fixed"],
                             irrigation_fixed=b["irrigation_fixed"])
                parts.update({k:b[k] for k in ("fertilizer_other","chemicals","fuel_repairs")})
                parts["other_operating"] = b["operating"]-sum(v for k,v in parts.items() if k not in FIXED_PARTS)
                c["establishment"][mode] = {
                    "parts": {k:entry(v*ACRES_HA, "USD/ha/new stand", b["source"])
                              for k,v in parts.items()},
                    "labour_hours_ha":entry(labour/b["wage"]*ACRES_HA, "hours/ha/new stand", b["source"]),
                    "labour_rate":entry(b["wage"], "USD/hour", b["source"]),
                    "n_price":entry(b["n_price"] or c["n_price"]["value"], "USD/kg N",
                                    b["source"]+"; marginal quote only when N is embedded in phosphate"),
                    "n_in_aggregate":b["n_price"] == 0,
                    "starter_n_kg_ha":entry(b["nitrogen"]*1.12085, "kg N/ha/new stand", b["source"]),
                    "source":b["source"],
                }
            if state in ("31","06","20"):
                c["n_in_aggregate"] = True
                c["default_n_price"] = c["n_price"]["value"]
                c["n_price"]["source"] += "; alfalfa N embedded in phosphate; only marginal farmer N-price changes billed"
    for c in covers:
        parts = {k: entry(0., "USD/ha sown", c.traits["cost_usd_ha"].source+"; unallocated contractor costs in other_operating",
                          "low", (0, 100000)) for k in PARTS}
        parts["seed"] = entry(c.v("seed_cost_usd_ha"), "USD/ha sown", c.traits["seed_cost_usd_ha"].source, "low", (0, 10000))
        parts["other_operating"] = entry(c.v("cost_usd_ha")-c.v("seed_cost_usd_ha"), "USD/ha sown",
                                           c.traits["cost_usd_ha"].source+"; drilling/termination bundle", "low", (0, 100000))
        for k,p in parts.items():
            p.update(classification="fixed" if k in (*FIXED_PARTS,"repeat_crop_fixed") else "operating", transfer=state != "19" or c.key=="hairy_vetch_cover")
        ctx["covers"].append({"crop": c.key, "seed": parts["seed"], "parts": parts, "transfer":state != "19" or c.key=="hairy_vetch_cover"})
    ctx["fertilizer_n_usd_kg"]["scope"] = "Regional reference only; per-crop budget blends/marginal rates shown separately. Farmer override replaces all crop N rates."
    return ctx


def resolved(base, overrides):
    ctx = copy.deepcopy(base)
    for key, val in overrides.items():
        if key in SCALARS and key not in ("usd_per_acre_foot", "cover_seed_usd_ha"):
            if key == "irrigation_capacity_mm_yr":
                ctx["capacity"] = entry(val, "mm/year", "farmer", "high", SCALARS[key][:2])
            else:
                ctx[key] = entry(val, SCALARS[key][2], "farmer", "high", SCALARS[key][:2])
    if "usd_per_acre_foot" in overrides:
        ctx["water_charge_usd_per_mm_ha"] = entry(overrides["usd_per_acre_foot"] * ACRES_HA / MM_FOOT,
                                                   "USD/(mm ha)", "farmer", "high", SCALARS["water_charge_usd_per_mm_ha"][:2])
    for c in ctx["crops"]:
        changed_parts = overrides.get("crop_cost_usd_ha", {}).get(c["crop"], {})
        if (c["legacy"] and "no defensible" in c["parts"]["other_operating"]["source"]
                and changed_parts.keys()-{"other_operating", "repeat_crop"}
                and "other_operating" not in changed_parts):
            raise ValueError(f"{c['crop']} budget is unallocated: supply revised other_operating with component edits to avoid double counting")
        if "fertilizer_n_usd_kg" in overrides:
            c["n_price"] = entry(overrides["fertilizer_n_usd_kg"], "USD/kg N", "farmer", "high", SCALARS["fertilizer_n_usd_kg"][:2])
        if c["crop"] in overrides.get("crop_price_usd_t", {}):
            c["price"] = entry(overrides["crop_price_usd_t"][c["crop"]], "USD/t", "farmer", "high", (0, 1000000))
        for part, val in overrides.get("crop_cost_usd_ha", {}).get(c["crop"], {}).items():
            c["parts"][part] = entry(val, "USD/ha/year", "farmer", "high", (0, 100000)) | {"classification":"fixed" if part in (*FIXED_PARTS,"repeat_crop_fixed") else "operating", "transfer":False}
        if "labour" not in overrides.get("crop_cost_usd_ha", {}).get(c["crop"], {}) and (
                "labour_usd_hr" in overrides or not c.get("native_labour")):
            c["parts"]["labour"]["value"] = c["labour_hours_ha"]["value"] * ctx["labour_usd_hr"]["value"]
            if c["labour_hours_ha"]["value"]:
                c["parts"]["labour"].update(source=ctx["labour_usd_hr"]["source"]+" × " + c["labour_hours_ha"]["source"],
                                             confidence="low")  # wage known, hours still uncertain
        if "repeat_crop" not in overrides.get("crop_cost_usd_ha", {}).get(c["crop"], {}):
            c["parts"]["repeat_crop"]["value"] += c["repeat_labour_hours_ha"] * (
                ctx["labour_usd_hr"]["value"] - 20.15)
        if "irrigation_energy_usd_per_mm_ha" in overrides:
            c["energy_rate"] = ctx["irrigation_energy_usd_per_mm_ha"]
        for profile in c.get("establishment", {}).values():
            for part, value in changed_parts.items():
                if part not in ("repeat_crop", "repeat_crop_fixed"):
                    profile["parts"][part] = entry(value, "USD/ha/new stand", "farmer", "high", (0, 100000))
            if "labour_usd_hr" in overrides and "labour" not in changed_parts:
                profile["parts"]["labour"] = entry(
                    profile["labour_hours_ha"]["value"]*overrides["labour_usd_hr"],
                    "USD/ha/new stand", "farmer wage × "+profile["labour_hours_ha"]["source"], "low")
            if "fertilizer_n_usd_kg" in overrides:
                profile["n_price"] = c["n_price"]
        if "establishment" in c:
            c["parts"]["seed"]["unit"] = "USD/ha/new stand (not annually)"
        c["transfer"] = any(p.get("transfer",False) and p["value"] for p in c["parts"].values()) or c["energy_rate"].get("transfer",False)
    if "cottonseed" in overrides.get("crop_price_usd_t", {}):
        ctx["cottonseed_price"] = entry(overrides["crop_price_usd_t"]["cottonseed"], "USD/t seed", "farmer", "high", (0, 1000000))
    for c, original in zip(ctx["crops"], base["crops"], strict=True):
        c["cottonseed_usd_per_lint_t"]["value"] = original["cottonseed_usd_per_lint_t"]["value"] * (
            ctx["cottonseed_price"]["value"]/base["cottonseed_price"]["value"])
        if c["cottonseed_usd_per_lint_t"]["value"]:
            c["cottonseed_usd_per_lint_t"]["source"] = (TX+"CottonPivot.pdf:1.42t seed/t lint × current seed price "
                + f"{ctx['cottonseed_price']['value']:,.6g}"+" USD/t ["+ctx["cottonseed_price"]["source"]+"]")
    for c in ctx["covers"]:
        if "cover_seed_usd_ha" in overrides:
            c["parts"]["seed"] = entry(overrides["cover_seed_usd_ha"], "USD/ha sown", "farmer", "high", (0, 10000))
        for part, val in overrides.get("crop_cost_usd_ha", {}).get(c["crop"], {}).items():
            c["parts"][part] = entry(val, "USD/ha sown", "farmer", "high", (0, 100000))
        c["seed"] = c["parts"]["seed"]
    return ctx


def income_evidence(ctx, keys):
    used = [c for c in ctx["crops"] if c["crop"] in keys]
    quotes = sum(c["price"]["source"] == "farmer" for c in used)
    costs = sum(any(p["source"] == "farmer" for p in c["parts"].values()) for c in used)
    rates = [f"{k}={ctx[k]['value']:,.6g} {ctx[k]['unit']} [{ctx[k]['confidence']}; {ctx[k]['source']}]"
             for k in ("labour_usd_hr", "fertilizer_n_usd_kg", "water_charge_usd_per_mm_ha",
                       "irrigation_energy_usd_per_mm_ha", "land_rent_usd_ha")]
    crop_sources = []
    for c in used:
        groups = {}
        for k, p in c["parts"].items():
            groups.setdefault((p["source"], p["confidence"]), []).append(f"{k}={p['value']:,.6g}")
        crop_sources.append(f"{c['crop']} cost transfer={c['transfer']}; price {c['price']['value']:,.6g} USD/t [{c['price']['confidence']}; {c['price']['source']}]; "
                            + "; ".join(f"cost USD/ha {', '.join(values)} [{conf}; {source}]"
                                        for (source, conf), values in groups.items()))
        crop_sources.append(f"{c['crop']} N unit price {c['n_price']['value']:,.6g} [{c['n_price']['confidence']}; {c['n_price']['source']}]")
        for mode, b in c.get("establishment", {}).items():
            crop_sources.append(f"{c['crop']} {mode} establishment replaces production costs: "
                                + ", ".join(f"{k}={p['value']:,.6g} [{p['confidence']}; {p['source']}]"
                                            for k,p in b["parts"].items() if p["value"])
                                + f"; N quote {b['n_price']['value']:,.6g} [{b['n_price']['source']}]; "
                                + c["stand_policy"])
    for c in ctx["covers"]:
        if c["crop"] in keys:
            crop_sources.append(f"{c['crop']} costs: " + "; ".join(
                f"{k}={p['value']:,.6g} USD/ha [{p['confidence']}; {p['source']}]"
                for k, p in c["parts"].items() if p["value"] or p["source"] == "farmer"))
    if "cotton" in keys:
        p = ctx["cottonseed_price"]
        crop_sources.append(f"Cottonseed price {p['value']:,.6g} USD/t [{p['confidence']}; {p['source']}]")
    confidence = "medium" if quotes == len(used) and quotes and (costs or any(
        ctx[k]["source"] == "farmer" for k in ("labour_usd_hr", "irrigation_energy_usd_per_mm_ha", "water_charge_usd_per_mm_ha"))) else None
    return confidence, ("Economics inputs: " + "; ".join(crop_sources + rates)
                        + ". Farmer inputs are high-confidence quotes; hours, yields, budget transfers and irrigation supply remain uncertain.")


ECON_FIELDS = ("income_legacy", "harvest_t_ha", "double_crop_t_ha", "n_fertilizer_kg_ha", "irrigation_mm",
               "cover_cost_usd_ha", "cover_planted", "n_fertilizer_saving_usd_ha",
               "economic_crop_index", "economic_cover_index", "economic_double_planted", "economic_same_crop",
               "economic_establishment", "economic_establishment_fall")


def reprice(fields, base, overrides=None):
    """Operating return, listed-cost return, irrigation subtotal, provenance.

    No field simulation. Fixed lines never affect the headline operating return.
    Unsplit source bundles stay operating; we do not invent an ownership share.
    """
    overrides = overrides or {}
    ctx = resolved(base, overrides)
    index = fields["economic_crop_index"].long()
    device = index.device
    arr = lambda values: torch.tensor(values, device=device)[index]
    price = arr([c["price"]["value"] for c in ctx["crops"]])
    bundled_n = arr([c.get("n_in_aggregate",c["legacy"]) for c in base["crops"]]).bool()
    subtotal = arr([sum(p["value"] for k, p in c["parts"].items() if k not in ("repeat_crop","repeat_crop_fixed")) for c in ctx["crops"]])
    subtotal -= arr([c["parts"]["seed"]["value"] if "establishment" in c else 0. for c in ctx["crops"]])
    repeat = arr([c["parts"]["repeat_crop"]["value"] for c in ctx["crops"]]) * fields["economic_same_crop"]
    ref = arr([c["harvest_reference_yield_t_ha"]["value"] for c in ctx["crops"]])
    scale_harvest = arr([c["parts"]["harvest"]["source"] != "farmer" for c in ctx["crops"]]).bool() & (ref > 0)
    harvest_bill = arr([c["parts"]["harvest"]["value"] for c in ctx["crops"]])
    subtotal += torch.where(scale_harvest, harvest_bill*(fields["harvest_t_ha"]/ref.clamp_min(.0001)-1), 0)
    irrfixed = arr([c["parts"]["irrigation_fixed"]["value"] for c in ctx["crops"]])
    variable_water = fields["irrigation_mm"] * (ctx["water_charge_usd_per_mm_ha"]["value"]
                        + arr([c["energy_rate"]["value"] for c in ctx["crops"]]))
    irr = variable_water + irrfixed
    n = fields["n_fertilizer_kg_ha"]
    np = arr([c["n_price"]["value"] for c in ctx["crops"]])
    # Only genuinely UNSPLIT generic bundles retain embedded N. Known source
    # budgets remove their reference N bill and charge actual fertilizer once.
    n_delta = n * (np - arr([c["default_n_price"] for c in base["crops"]])) if "fertilizer_n_usd_kg" in overrides else 0.
    seed_credit = fields["harvest_t_ha"] * arr([c["cottonseed_usd_per_lint_t"]["value"] for c in ctx["crops"]])
    fixed = arr([sum(c["parts"][k]["value"] for k in FIXED_PARTS) for c in ctx["crops"]])
    # Substitute the actual seeding-year budget, not production PLUS seeding.
    # Zero defaults retain compatibility with synthetic non-stand field fixtures.
    establishing = fields.get("economic_establishment", torch.zeros_like(n)).bool()
    fall = fields.get("economic_establishment_fall", torch.zeros_like(n)).bool()
    for mode, selected in (("fall", establishing & fall), ("spring", establishing & ~fall)):
        profiles = [c.get("establishment", {}).get(mode) for c in ctx["crops"]]
        op = arr([sum(p["value"] for k,p in b["parts"].items() if k not in FIXED_PARTS)
                  if b else 0. for b in profiles])
        fx = arr([sum(b["parts"][k]["value"] for k in FIXED_PARTS) if b else 0. for b in profiles])
        starter_price = arr([b["n_price"]["value"] if b else 0. for b in profiles])
        # Embedded phosphate-N is already in the cash fertilizer line.
        starter_price -= arr([base["crops"][i]["establishment"][mode]["n_price"]["value"]
                              if b and b["n_in_aggregate"] else 0. for i,b in enumerate(profiles)])
        subtotal = torch.where(selected, op+fx, subtotal)
        fixed = torch.where(selected, fx, fixed)
        np = torch.where(selected, starter_price, np)
        # The production crop may be bundled while establishment is not (TX),
        # or vice versa. The event-specific price above is already marginal.
        bundled_n = torch.where(selected, False, bundled_n)
        irrfixed = torch.where(selected, arr([b["parts"]["irrigation_fixed"]["value"] if b else 0.
                                             for b in profiles]), irrfixed)
    irr = variable_water+irrfixed
    income = fields["harvest_t_ha"]*price + seed_credit - (subtotal-fixed) - repeat - variable_water - torch.where(bundled_n,n_delta,n*np)
    income += torch.where(bundled_n,fields["n_fertilizer_saving_usd_ha"],0)
    fixed += arr([c["parts"]["repeat_crop_fixed"]["value"] for c in ctx["crops"]])*fields["economic_same_crop"]
    if len(ctx["crops"]) > len(base["main_keys"]):
        dc = ctx["crops"][-1]
        planted = fields["economic_double_planted"]
        dc_fixed = sum(dc["parts"][k]["value"] for k in FIXED_PARTS)
        income += fields["double_crop_t_ha"]*dc["price"]["value"] - planted*sum(
            p["value"] for k,p in dc["parts"].items() if k not in (*FIXED_PARTS,"repeat_crop","repeat_crop_fixed"))
        fixed += planted*dc_fixed
    if ctx["covers"]:
        ci = (fields["economic_cover_index"].long()-1).clamp_min(0)
        planted = fields["economic_cover_index"] > 0
        bills = torch.tensor([sum(p["value"] for k,p in c["parts"].items() if k not in (*FIXED_PARTS,"repeat_crop","repeat_crop_fixed"))
                              for c in ctx["covers"]], device=device)
        ownership = torch.tensor([sum(c["parts"][k]["value"] for k in FIXED_PARTS) for c in ctx["covers"]], device=device)
        income -= bills[ci]*planted
        fixed += ownership[ci]*planted
    listed = income-fixed-ctx["land_rent_usd_ha"]["value"]
    return income, listed, irr, ctx
