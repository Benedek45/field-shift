"""In-memory farm preparation and exact requested-cycle evaluation on MPS/CUDA."""

import copy
import json
import math
import re
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

from fieldshift.agent import farms
from fieldshift.planner.data import epic_k, load_site
from fieldshift.planner.search import VERSION, candidate_crops, summarize
from fieldshift.sim.engine import (
    _edited,
    cover_eligible,
    fao_reference,
    ledger_pass,
    prepare,
    trajectories,
    water_pass,
)
from fieldshift.sim.traits import CATALOG

FAMILIES = {1: "grass", 2: "legume", 3: "aster", 4: "mallow", 5: "nightshade",
            6: "amaranth", 7: "flax"}
WINTER_ALIASES = {"cereal_rye": "cereal_rye_cover", "hairy_vetch": "hairy_vetch_cover",
                  "none": "", "": ""}
METRICS = {
    "margin_usd_ha_yr": "income", "field_water_use_mm_yr": "et_total_mm",
    "return_over_listed_costs_usd_ha_yr": "return_over_listed_costs_usd_ha",
    "soil_carbon_input_t_c_ha_yr": "soil_carbon_input_t_ha", "forage_fraction": "forage",
    "soil_water_at_planting_mm": "water_at_planting_mm", "nitrate_leached_kg_n_ha_yr": "n_leached_kg_ha",
    "fertilizer_n_kg_n_ha_yr": "n_fertilizer_kg_ha", "nitrogen_fixed_kg_n_ha_yr": "n_fixed_kg_ha",
    "erosion_t_ha_yr": "erosion_t_ha", "days_without_living_cover_yr": "bare_days",
    "cover_biomass_t_dm_ha_yr": "cover_biomass_t_ha", "cover_establishment_fraction": "cover_established",
    "soil_mineral_n_kg_n_ha": "mineral_n_kg_ha", "mineralized_n_kg_n_ha_yr": "n_mineralized_kg_ha",
    "double_crop_yield_t_ha_yr": "double_crop_t_ha",
    "applied_irrigation_mm_yr": "irrigation_mm", "irrigation_cost_usd_ha_yr": "irrigation_cost_usd_ha",
}
METRIC_DEFINITIONS = {
    "margin_usd_ha_yr": "Return over operating costs (USD/ha/year): revenue including cottonseed minus operating inputs, actual N, labour and applied-water charges; excludes ownership/overhead and land.",
    "return_over_listed_costs_usd_ha_yr": "Return over all listed costs (USD/ha/year): operating return minus separately listed ownership/overhead and farmer-set land rent; not used by default weights.",
    "soil_carbon_input_t_c_ha_yr": "Humified/retained plant carbon input (t C/ha/year), not gross biomass carbon or observed soil carbon change; cycle means across weather windows and phases.",
    "local_market_fraction": "Cycle mean county NASS harvested-area share among supported crops (v25), not binary presence. Production-depth proxy only: no verified hay buyer, grade, delivered price or market liquidity.",
}


class InputError(ValueError):
    pass


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def constraints(record):
    texts = {t.lower().strip().rstrip(".") for t in record["constraints"]}
    equipment = bool(texts & {"no new equipment", "no new crop equipment"})
    forage = bool(texts & {"must keep a forage crop", "must include forage", "must include a forage crop"})
    recognized = {"no new equipment", "no new crop equipment", "must keep a forage crop",
                  "must include forage", "must include a forage crop"}
    return equipment, forage, [t for t in record["constraints"]
                               if t.lower().strip().rstrip(".") not in recognized]


def quantiles(values):
    q = torch.nanquantile(values, torch.tensor([.1, .5, .9], device=values.device))
    return dict(zip(("p10", "median", "p90"), q.tolist(), strict=True))


def private_text(text, site):
    """Remove local identity and calendar years from free text in anonymous output."""
    # County label is 'Story County, IA'; sources can also spell out a state.
    from fieldshift.planner.data import STATE_ALPHA
    state_names = (
        "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|"
        "Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|"
        "Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
        "New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|"
        "Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
        "West Virginia|Wisconsin|Wyoming"
    )
    label = site["label"].split(" County")[0].split(",")[0]
    text = re.sub(re.escape(site["adm_id"]), "[location withheld]", text, flags=re.IGNORECASE)
    fips = site["adm_id"][3:5] + site["adm_id"][6:]
    text = re.sub(re.escape(fips), "[location withheld]", text)
    text = re.sub(r"\b(?:" + re.escape(label) + "|" + state_names + r")\b", "[place withheld]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:" + "|".join(STATE_ALPHA.values()) + r")\b", "[state withheld]", text)
    text = re.sub(r"\b(?:19|20)\d{2}\b", lambda m: f"year {int(m[0]) - 2025}", text)
    return text


class Farm:
    def __init__(self, record):
        self.record = copy.deepcopy(record)
        self.site = load_site(record["adm_id"], record["irrigated"])
        if "existing_rotation" in record:
            self.site["existing_steps"] = copy.deepcopy(record["existing_rotation"])
            self.site["existing"] = tuple(s["crop"] for s in record["existing_rotation"])
            self.site["rotation_evidence"] = "Current rotation stated by the farmer; not inferred from satellite maps or county areas."
            self.site["existing_origin"] = "farmer"
        for k in ("slope_pct", "organic_matter_pct", "awc_mm_per_m"):
            if k in record["overrides"]:
                self.site["soil"][k] = record["overrides"][k]
        # OM changes erodibility as well as organic-N stocks, using the same EPIC formula.
        if "organic_matter_pct" in record["overrides"]:
            from fieldshift.planner.data import _static
            raw = _static().loc[record["adm_id"]]
            texture = [raw.get(k, np.nan) for k in ("sand_0_30", "silt_0_30", "clay_0_30")]
            if np.isfinite(texture).all():
                self.site["soil"]["rusle_k"] = epic_k(*texture, self.site["soil"]["organic_matter_pct"])
        self.no_equipment, self.must_forage, self.unchecked = constraints(record)
        self.crops, self.covers, self.double, self.excluded = candidate_crops(self.site, self.no_equipment)
        self.keys = [c.key for c in self.crops]
        self.cover_keys = [c.key for c in self.covers]
        self.irrigated = record["irrigated"]
        physical = farms.physical_record(record)
        if "irrigation_capacity_mm_yr" in physical["overrides"]:
            self.site["soil"]["irrigation_capacity_mm_yr"] = physical["overrides"]["irrigation_capacity_mm_yr"]
        self.reference_soils = [dict(self.site["soil"])]
        self.reference_soils[0].pop("irrigation_capacity_mm_yr", None)
        self.base = prepare([self.site], self.crops, self.covers, self.double, irrigated=self.irrigated)
        self.prepared = OrderedDict()
        self.results = OrderedDict()
        self.physical_results = OrderedDict()
        self.normals = np.array([[self.site["normals"].get(c.key, c.v("normal_t_ha")) for c in self.crops]],
                                dtype=np.float32)
        self.existing = copy.deepcopy(record.get("existing_rotation", [{"crop": k} for k in self.site["existing"]]))
        self.score_years = math.lcm(4, len(self.existing)) if "existing_rotation" in record else 4
        if "existing_rotation" in record:
            self.existing = self.parse(self.existing)
            _, invalid = self.warnings(self.existing, self.base)
            # An explicit history must pass the same maturity check as a new crop.
            invalid += [f"{s['crop']}: maturity reached in fewer than 60% of weather years"
                        for s in self.existing if float(self.base["maturity"][0, self.keys.index(s["crop"])]) < .6]
            if invalid:
                raise InputError("Farmer's current_rotation is infeasible: " + "; ".join(dict.fromkeys(invalid)))
        # Reference independent of what cycles the caller requests.
        rt = trajectories([((i,), 0, False) for i in range(len(self.crops))], self.crops,
                          len(self.base["years"]))
        water = water_pass(self.base, self.reference_soils, self.crops, self.covers, self.double, rt, self.irrigated)
        actual = (water_pass(self.base, [self.site["soil"]], self.crops, self.covers, self.double, rt, self.irrigated)
                  if "irrigation_capacity_mm_yr" in self.site["soil"] else water)
        v = actual["irrigation_mm"][:, rt["spin"]:].mean(-1).reshape(len(self.crops), -1)
        self.irrigation_by_crop = dict(zip(self.keys, torch.quantile(v, .5, dim=1).tolist(), strict=True))
        self.reference = fao_reference(water, rt, 1, len(self.crops), rt["spin"])
        autumn = [i for i, c in enumerate(self.crops) if c.v("autumn_sown")]
        if self.double is not None and autumn:
            dt = trajectories([((i,), 0, True) for i in autumn], self.crops, len(self.base["years"]))
            dw = water_pass(self.base, self.reference_soils, self.crops, self.covers, self.double, dt, self.irrigated)
            self.reference[:, -1] = fao_reference(dw, dt, 1, len(self.crops), dt["spin"])[:, -1]
        self.base_summary = None

    def changes(self, rain=0., temp=0.):
        rain = farms.number(rain, "rain_change_pct", -100, 200)
        temp = farms.number(temp, "temp_change_c", -15, 15)
        o = self.record["overrides"]
        r = (1 + o.get("rain_change_pct", 0) / 100) * (1 + rain / 100) - 1
        t = o.get("temp_change_c", 0) + temp
        return r, t

    def weather(self):
        r, t = self.changes()
        return _edited(self.site["daily"], r, t) if (r or t) else self.site["daily"].copy()

    def preparation(self, rain, temp):
        if not rain and not temp:
            return self.base
        key = (rain, temp)
        if key not in self.prepared:
            p = prepare([self.site], self.crops, self.covers, self.double, rain, temp, self.irrigated)
            # Preserve POWER-trained yield features; all edits act through the v22 FAO ratio.
            p["learned"] = self.base["learned"]
            self.prepared[key] = p
            if len(self.prepared) > 3:
                self.prepared.popitem(last=False)
        self.prepared.move_to_end(key)
        return self.prepared[key]

    def options(self):
        from fieldshift.agent.scoreboard import DEFAULT_WEIGHTS, WEIGHTS_HELP
        r, t = self.changes()
        p = self.preparation(r, t)
        main, winter, excluded = [], [], list(self.excluded)
        for i, c in enumerate(self.crops):
            maturity = float(p["maturity"][0, i])
            if maturity < .6 and c.key not in self.site["existing"]:
                excluded.append(f"{c.key}: matures in {maturity:.0%} of weather years (<60%)")
                continue
            main.append(self.traits(c) | {"maturity_fraction": maturity,
                                        "warning": "Generic cultivar maturity is poorly matched" if maturity < .6 else None})
        eligible = cover_eligible(self.crops)
        for i, c in enumerate(self.covers):
            shares = p["covers"]["alive"][0, i].float().mean(-1).tolist()
            possible = [s for j, s in enumerate(shares) if eligible[j].any()]
            if not possible or max(possible) < .2:
                excluded.append(f"{c.key}: establishment/winter survival below 20% after every eligible main crop")
                continue
            winter.append(self.traits(c) | {"api_alias": c.key.removesuffix("_cover"),
                                           "survival_fraction_after_crop": dict(zip(self.keys, shares, strict=True))})
        for key, crop in CATALOG.items():
            if key not in self.keys + self.cover_keys and key != "double_crop_soybean" and not crop.v("cover_crop"):
                excluded.append(f"{key}: outside this farm's current local-crop/equipment screen; not available in this service")
        double = None
        if p["double"] is not None and any(c.v("autumn_sown") and float(p["double"]["maturity"][0, i]) >= .6
                                           for i, c in enumerate(self.crops)):
            double = self.traits(self.double) | {"api_field": "double_crop", "api_value": "soybean"}
        elif self.double is not None:
            excluded.append("double_crop_soybean: matures in fewer than 60% of years after every supported autumn-sown crop")
        from fieldshift.sim.economics import SCALARS
        economics = self.economics_context()
        return clean({"main_crops": main, "winter_options": winter, "double_crop_option": double, "excluded": excluded,
                      "economics": economics, "metric_definitions": METRIC_DEFINITIONS,
                      "existing_economic_metrics": {k:v for k,v in self.evaluate(self.existing,reference=True)["metrics"].items()
                                                   if k in METRIC_DEFINITIONS},
                      "economic_overrides": {"scalars": {k: {"unit": u, "allowed_range": [lo, hi]}
                                                         for k, (lo, hi, u) in SCALARS.items()},
                                             "crop_price_usd_t": "Per-crop price entries include current values, provenance and ranges",
                                             "crop_cost_usd_ha": "Per-crop parts include current values, provenance and ranges; crop labour bill overrides global wage",
                                             "capacity": "Physical input: changing it reruns the field. Prices/costs only reprice cached physical outputs"},
                      "weights_help": WEIGHTS_HELP, "default_weights": DEFAULT_WEIGHTS,
                       "existing_rotation": self.existing,
                       "existing_rotation_source": self.site["rotation_evidence"],
                      "constraints_not_machine_enforced": self.unchecked,
                      "notes": ["County proxy, not an observed field. Rotation-level calendar/stand checks still apply.",
                                "winter is an unharvested cover; double_crop='soybean' is a harvested second crop after autumn-sown grain.",
                                 ("Typical yield and income are reference scenarios; irrigation is priced from applied mm; land excluded unless farmer supplies rent. "
                                  "Annual caps are allocation proxies, not measured well capacity; net application omits distribution losses.")]})

    def traits(self, c):
        def trait(k, unit):
            v = c.traits[k]
            return {"value": v.value, "unit": unit, "confidence": v.confidence,
                    "source": v.source if not self.record["anonymous"] else "Published crop trait / reference budget"}
        econ = self.economics_context()
        item = next((x for x in econ["crops"] if x["crop"] == c.key), None)
        money = {}
        if item is not None:
            reference_n = item.get("production_n_kg_ha",{}).get("value",c.v("recommended_n_kg_ha"))
            applied = self.irrigation_by_crop.get(c.key, 0.)
            rate = econ["water_charge_usd_per_mm_ha"]["value"] + item["energy_rate"]["value"]
            fixed = item["parts"]["irrigation_fixed"]["value"]
            money = {"price": item["price"], "cost_parts": item["parts"], "transfer": item["transfer"],
                      "cost_origin_state":item["origin_state"],
                     "labour_hours_ha": item["labour_hours_ha"], "fertilizer_n_price": item["n_price"],
                     "custom_harvest_reference_yield": item["harvest_reference_yield_t_ha"],
                     "irrigation_cost": {"value": applied*rate+fixed, "unit": "USD/ha/year",
                                         "source": "Simulated uniform-crop median applied irrigation × cited rates + system fixed/repairs once",
                                         "confidence": "low", "applied_mm_yr": applied,
                                         "water_charge": econ["water_charge_usd_per_mm_ha"],
                                         "energy": econ["irrigation_energy_usd_per_mm_ha"],
                                         "fixed_system": item["parts"]["irrigation_fixed"]},
                      "cost": {"value": sum(p["value"] for k, p in item["parts"].items() if k not in ("repeat_crop","repeat_crop_fixed")) + applied*rate
                               + (0 if item["n_in_aggregate"] else reference_n * item["n_price"]["value"]),
                              "unit": "USD/ha/year reference", "source": "Sum of cited parts, reference N rate and simulated uniform-crop irrigation; rotation uses actual N/water",
                              "confidence": "low"}}
            if item["n_in_aggregate"] and "fertilizer_n_usd_kg" in self.record.get("economics", {}):
                reference_n -= c.v("previous_legume_credit_kg_ha")
                money["cost"]["value"] += max(reference_n, 0) * (item["n_price"]["value"]-item["default_n_price"])
            ownership = sum(item["parts"][k]["value"] for k in ("fixed_overhead","irrigation_fixed"))
            if "establishment" in item:
                money["cost"]["value"] -= item["parts"]["seed"]["value"]
                money["establishment"] = item["establishment"]
                money["stand_policy"] = item["stand_policy"]
                money["production_n_rate"] = item["production_n_kg_ha"]
                money["hay_basis"] = "As-reported baled hay, grade unspecified. 88% dry matter is a carbon convention, not observed NASS moisture."
            money["operating_cost"] = money["cost"] | {"value":money["cost"]["value"]-ownership,
                                                        "unit":"USD/ha/year reference operating cost"}
            money["listed_fixed_cost"] = {"value":ownership+econ["land_rent_usd_ha"]["value"],
                                          "unit":"USD/ha/year reference fixed+land cost",
                                          "source":"Cited fixed_overhead and irrigation ownership parts plus farmer rent; excluded from operating return",
                                          "confidence":"low"}
        else:
            cover = next((x for x in econ["covers"] if x["crop"] == c.key), None)
            if cover is not None:
                money = {"cost_parts": cover["parts"], "cover_seed": cover["seed"],
                         "cost": {"value": sum(p["value"] for k,p in cover["parts"].items() if k != "repeat_crop"),
                                  "unit": "USD/ha sown", "confidence": "low",
                                  "source": "Sum of cited cover cost parts; farmer seed/crop quotes replace defaults"}}
        return {"crop": c.key, "family": FAMILIES[int(c.v("family"))],
                "season": "perennial" if c.v("perennial") else ("autumn-sown" if c.v("autumn_sown") else "spring/summer"),
                "water_demand": trait("relative_water_demand", "relative index"),
                "nitrogen_fixed": trait("fixed_n_kg_ha", "kg N/ha/year reference"),
                "typical_yield": {"value": self.site["normals"].get(c.key, c.v("normal_t_ha")),
                                   "unit": "t/ha", "confidence": "low" if c.key not in self.site["observed"] or "LOW" in self.site["normal_sources"].get(c.key, "") else "medium",
                                  "source": self.site["normal_sources"].get(c.key, "Generic reference yield")
                                  if not self.record["anonymous"] else "County/neighbour/reference yield proxy"},
                "price": trait("price_usd_t", "USD/t"), "cost": trait("cost_usd_ha", "USD/ha/year"),
                 "traits": {k: {"value": tr.value, "confidence": tr.confidence,
                                "source": tr.source if not self.record["anonymous"] else "Published crop trait / reference budget"}
                            for k, tr in c.traits.items() if k not in ("price_usd_t", "cost_usd_ha", "fertilizer_price_usd_kg")},
                 **money}

    def economics_context(self):
        from fieldshift.sim.economics import resolved
        overrides = dict(self.record.get("economics", {}))
        if "irrigation_capacity_mm_yr" in self.record["overrides"]:
            overrides["irrigation_capacity_mm_yr"] = self.record["overrides"]["irrigation_capacity_mm_yr"]
        return resolved(self.base["economics"][0], overrides)

    def parse(self, cycle):
        if not isinstance(cycle, list) or not 1 <= len(cycle) <= 6:
            raise InputError("Each rotation must be a list of 1 to 6 crop-year objects")
        out = []
        for step in cycle:
            if not isinstance(step, dict) or step.keys() - {"crop", "winter", "double_crop"}:
                raise InputError('A crop-year has crop, optional winter cover, and optional double_crop="soybean" after autumn grain')
            crop, winter = step.get("crop"), step.get("winter", "")
            if not isinstance(crop, str) or crop not in self.keys:
                raise InputError(f"Unknown or excluded crop {crop!r}; valid main keys: {self.keys}. See /options for exclusions.")
            if not isinstance(winter, str):
                raise InputError("winter must be a crop key string (omit it for no winter cover)")
            winter = WINTER_ALIASES.get(winter, winter)
            if winter and winter not in self.cover_keys:
                raise InputError(f"Unknown or excluded winter {winter!r}; valid winter keys: {self.cover_keys} (or omit winter)")
            double = step.get("double_crop", "")
            if not isinstance(double, str) or double not in ("", "soybean", "double_crop_soybean"):
                raise InputError('double_crop must be "soybean" (or omit it)')
            if double and self.double is None:
                raise InputError("Double cropping is excluded by the farm's equipment constraint")
            if double and winter:
                raise InputError("A crop-year cannot have both an unharvested winter cover and double-crop soybean")
            out.append({"crop": crop, **({"winter": winter} if winter else {}),
                        **({"double_crop": "soybean"} if double else {})})
        return out

    def warnings(self, cycle, p, checks=None):
        notes, invalid, cover_shares, double_shares = [], [], [], []
        if self.irrigated:
            cap = self.site["soil"].get("irrigation_capacity_mm_yr")
            notes.append(f"Annual applied-irrigation cap {cap:g} mm; net bucket ignores distribution losses; source in /options."
                         if cap is not None else "Water availability UNKNOWN: no verified total allocation/well capacity; unlimited irrigation remains a scenario, not available supply.")
        eligible = cover_eligible(self.crops)
        for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            i, j = self.keys.index(a["crop"]), self.keys.index(b["crop"])
            c = self.crops[j]
            if c.v("autumn_sown") and not self.crops[i].v("fallow"):
                sow = p["timeline"]["sow"][0, j, 1:]
                operation = (checks["operation"][i][:-1] if checks is not None
                             else p["operation"][0, i, :-1].tolist())
                overlap = float(np.mean(np.asarray(operation) >= sow))
                if overlap > .4:
                    invalid.append(f"{a['crop']} → {b['crop']}: previous harvest overlaps autumn sowing in {overlap:.0%} of years")
            if a.get("winter"):
                if not eligible[i, j]:
                    invalid.append(f"Winter cover after {a['crop']} before {b['crop']} has no supported cover window")
                k = self.cover_keys.index(a["winter"])
                share = (checks["cover_survival"][k][i] if checks is not None
                         else float(p["covers"]["alive"][0, k, i].float().mean()))
                cover_shares.append(share)
                notes.append(f"{a['winter']} after {a['crop']}: {1-share:.0%} establishment/winter failure; failed sowings still cost money")
            if a.get("double_crop"):
                if not self.crops[i].v("autumn_sown") or p["double"] is None:
                    invalid.append(f"double_crop soybean after {a['crop']}: requires a supported autumn-sown main crop")
                else:
                    share = checks["double_maturity"][i] if checks is not None else float(p["double"]["maturity"][0, i])
                    double_shares.append(share)
                    notes.append(f"Double-crop soybean after {a['crop']}: reaches maturity in {share:.0%} of years")
            maturity = checks["maturity"][i] if checks is not None else float(p["maturity"][0, i])
            if maturity < .6:
                message = f"{a['crop']}: trait maturity reached in only {maturity:.0%} of years; cultivar uncertain"
                if a["crop"] in self.site["existing"]:
                    notes.append(message)
                else:
                    invalid.append(message)
        if cover_shares and np.mean(cover_shares) < .2:
            invalid.append("Winter cover cycle-average survival is below 20%; try earlier harvest or another cover")
        if double_shares and max(double_shares) < .6:
            invalid.append("Double-crop soybean matures in fewer than 60% of years after every requested main crop")
        for i, step in enumerate(cycle):
            if (self.crops[self.keys.index(step["crop"])].v("perennial")
                    and len({s["crop"] for s in cycle}) > 1
                    and cycle[i - 1]["crop"] != step["crop"] and cycle[(i + 1) % len(cycle)]["crop"] != step["crop"]):
                invalid.append(f"{step['crop']}: a perennial stand requires at least 2 adjacent years of the same crop")
        if self.must_forage and not any(self.crops[self.keys.index(s["crop"])].v("forage_value") > .5 for s in cycle):
            invalid.append("Farmer constraint: must keep a forage crop")
        if len(cycle) > 4:
            notes.append("v22 diversity bonus checks four consecutive crops; 5–6-year-cycle transfer is LOW confidence")
        if any(s["crop"] == "alfalfa" for s in cycle):
            from fieldshift.sim.alfalfa import POLICY
            notes.append(POLICY)
            notes.append("Alfalfa state yields, ungraded prices and stand/harvest-window transfers are LOW confidence; "
                         "hay market score is county production-area share, not a buyer commitment.")
        elif len(cycle) == 1 and self.crops[self.keys.index(cycle[0]["crop"])].v("perennial"):
            notes.append("v22 renews a uniform perennial stand once per listed cycle; a 1-year cycle renews annually")
        notes.extend(f"Constraint requires agent review: {s}" for s in self.unchecked)
        return list(dict.fromkeys(notes)), list(dict.fromkeys(invalid))

    def evaluate(self, cycle, rain=0., temp=0., reference=False):
        """One exact cycle; explicit cover indices replace only trajectory input."""
        r, t = self.changes(rain, temp)
        key = json.dumps([cycle, r, t, reference], sort_keys=True)
        if key in self.results:
            self.results.move_to_end(key)
            return copy.deepcopy(self.results[key])
        p = self.preparation(r, t)
        notes, invalid = self.warnings(cycle, p)
        if invalid and not reference:
            raise InputError("Infeasible sequence: " + "; ".join(invalid))
        notes.extend("Existing reference is not feasible: " + s for s in invalid)
        requested_cycle = cycle
        if all(s["crop"] == "alfalfa" for s in cycle):
            # A×1/A×2/A×6 are the SAME four-calendar-year stand enterprise,
            # not different stand lives or different weather-age sampling.
            cycle = [cycle[0]]*4
        indices = tuple(self.keys.index(s["crop"]) for s in cycle)
        # Keep planner's 4-year spin/score windows for 1/2/4-year cycles. Other
        # lengths spin and score a whole cycle (minimum 4-year spin).
        n = len(cycle)
        score, spin = (4, 4) if n in (1, 2, 4) else (n, max(4, n))
        phases = None
        if "existing_rotation" in self.record and self.score_years % n == 0:
            score, spin, phases = self.score_years, 4, self.score_years
        tr = trajectories([(indices, 0, False)], self.crops, len(p["years"]), spin=spin, score=score, phases=phases)
        covers = [self.cover_keys.index(s["winter"]) + 1 if s.get("winter") else 0 for s in cycle]
        tr["cover"] = np.asarray([[covers[pos] for pos in row] for row in tr["position"]], dtype=np.int64)
        doubles = [bool(s.get("double_crop")) for s in cycle]
        tr["double"] = np.asarray([[doubles[pos] for pos in row] for row in tr["position"]], dtype=bool)
        if key not in self.physical_results:
            water = water_pass(p, [self.site["soil"]], self.crops, self.covers, self.double, tr, self.irrigated)
            base_fao = None
            if r or t or "irrigation_capacity_mm_yr" in self.site["soil"]:
                bw = water_pass(self.base, self.reference_soils, self.crops, self.covers, self.double, tr, self.irrigated)
                base_fao = {"fao_main": bw["fao_main"]}
            ledger = ledger_pass(p, water, [self.site["soil"]], self.normals, self.crops, self.covers,
                                  self.double, tr, self.irrigated, fao_reference=self.reference, base_fao=base_fao)
            self.physical_results[key] = ledger
            if len(self.physical_results) > 96:
                self.physical_results.popitem(last=False)
        ledger = dict(self.physical_results[key])
        from fieldshift.sim.economics import reprice
        ledger["income"], ledger["return_over_listed_costs_usd_ha"], ledger["irrigation_cost_usd_ha"], econ = reprice(
            ledger, p["economics"][0], self.record.get("economics", {}))
        # summarize supplies the planner's income confidence and canonical stats.
        option = max(covers)
        row = summarize(ledger, tr, [(indices, option, any(doubles))], self.crops, self.covers, self.site)[0]
        metrics = {name: quantiles(ledger[source][:, spin:].nanmean(-1)) for name, source in METRICS.items()}
        om = (ledger["organic_matter_pct"][:, -1] - ledger["organic_matter_pct"][:, spin - 1]) / score * 10
        metrics["organic_matter_trend_pct_points_decade"] = quantiles(om)
        from fieldshift.planner.search import market_support
        metrics["local_market_fraction"] = dict.fromkeys(("p10","median","p90"),
                                                         sum(market_support(s["crop"], self.site) for s in cycle)/n)
        c = torch.as_tensor(tr["crop"][:, 1:-1][:, spin:], device=p["device"])
        relative = {}
        for i in set(indices):
            mask = c == i
            v = torch.where(mask, ledger["yield_relative"][:, spin:], float("nan")).nanmean(-1)
            relative[self.keys[i]] = quantiles(v[torch.isfinite(v)])
        # Reproduce the engine's generic main-crop family-pressure recurrence.
        pressure = torch.zeros((len(c), 8), device=p["device"])
        seq = torch.as_tensor(tr["crop"][:, 1:-1], device=p["device"])
        family = torch.tensor([int(cr.v("family")) for cr in self.crops], device=p["device"])
        samples = []
        for j in range(tr["horizon"]):
            pressure *= .5
            cr = seq[:, j]
            active = torch.tensor([1 - crp.v("fallow") for crp in self.crops],
                                  device=p["device"], dtype=torch.float32)[cr]
            pressure.scatter_add_(1, family[cr, None], active[:, None])
            if j >= spin:
                samples.append(pressure.clone())
        pest = {name: quantiles(torch.stack(samples, 1)[..., f].mean(1))
                for f, name in FAMILIES.items() if f in {int(self.crops[i].v("family")) for i in indices}}
        reason = "Yield proxies: "+ "; ".join(f"{k}: {self.site.get('normal_sources',{}).get(k,'generic prior')}" for k in sorted({s["crop"] for s in cycle}))
        from fieldshift.sim.economics import income_evidence
        confidence, economics_drivers = income_evidence(econ, [s["crop"] for s in cycle] + [s.get("winter", "") for s in cycle])
        if confidence:
            row["income_confidence"] = confidence
        reason += "; " + economics_drivers
        if any(doubles):
            row["income_confidence"] = "low"
            reason += "; double-crop yield fraction and costs are low-confidence transfers"
        if self.record["anonymous"]:
            reason = private_text(reason, self.site)
        result = clean({"rotation": requested_cycle, "metrics": metrics, "relative_yield_by_crop": relative,
                        "family_pressure_index": pest, "warnings": notes,
                        "income_confidence": row["income_confidence"], "income_confidence_drivers": reason,
                        "weather_runs": len(tr["rotation"]), "spin_years": spin, "scored_years_per_run": score})
        self.results[key] = result
        if len(self.results) > 96:
            self.results.popitem(last=False)
        return copy.deepcopy(result)


class Runtime:
    """Bounded LRU; serialize accelerator work and avoid thread-unsafe model loads."""
    def __init__(self, root=None, max_farms=3):
        self.root, self.max_farms = root, max_farms
        self.lock = threading.RLock()
        self.cache = OrderedDict()

    def farm(self, farm_id):
        record = farms.get(farm_id, self.root)
        key = (farm_id, json.dumps(record, sort_keys=True))
        if key not in self.cache:
            match = next((f for f in self.cache.values() if farms.physical_record(f.record) == farms.physical_record(record)), None)
            if match is not None:
                clone = copy.copy(match)
                clone.record, clone.results, clone.base_summary = copy.deepcopy(record), OrderedDict(), None
                self.cache[key] = clone
            else:
                self.cache[key] = Farm(record)
            if len(self.cache) > self.max_farms:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def scoreboard(self, farm_id, payload):
        from fieldshift.agent.scoreboard import build
        return build(self, farm_id, payload)

    def simulate(self, farm_id, payload):
        started = time.perf_counter()
        if not isinstance(payload, dict) or payload.keys() - {"rotations", "rain_change_pct", "temp_change_c"}:
            raise InputError("Body must have rotations and optional rain_change_pct / temp_change_c only")
        rotations = payload.get("rotations")
        if not isinstance(rotations, list) or not 1 <= len(rotations) <= 12:
            raise InputError("rotations must contain 1 to 12 rotations")
        rain, temp = payload.get("rain_change_pct", 0.), payload.get("temp_change_c", 0.)
        with self.lock:
            farm = self.farm(farm_id)
            farm.changes(rain, temp)
            cycles = [farm.parse(c) for c in rotations]
            # Validate every cycle before spending time evaluating the batch.
            r, t = farm.changes(rain, temp)
            p = farm.preparation(r, t)
            for cycle in cycles:
                _, invalid = farm.warnings(cycle, p)
                is_existing = cycle == farm.existing
                if invalid and not is_existing:
                    raise InputError("Infeasible sequence: " + "; ".join(invalid))
            existing = farm.evaluate(farm.existing, reference=True)
            existing_what_if = farm.evaluate(farm.existing, rain, temp, reference=True)
            values = [farm.evaluate(c, rain, temp, reference=(c == farm.existing)) for c in cycles]
            return clean({"farm_id": farm_id, "simulator_version": VERSION,
                          "device": str(farm.base["device"]), "existing": existing,
                          "existing_under_request_weather": existing_what_if, "results": values,
                          "applied_weather_changes": {"rain_change_pct_total": 100 * r, "temp_change_c_total": t,
                                                      "rain_rule": "farm multiplier × request multiplier",
                                                      "request_rain_change_pct": rain, "request_temp_change_c": temp},
                          "metric_definitions": METRIC_DEFINITIONS,
                          "method": "Median/P10/P90 of cycle means over historical weather windows × phases; not confidence intervals. "
                                     "v22 MLP/FAO response retained. Annual applied-irrigation cap where known; unknown supply remains unlimited with warning. "
                                     "Applied water × cited energy/water rates, system cost once; actual fertilizer N priced. No N-yield feedback; ET anomalies fail validation. "
                                      "Farmer prices/costs reprice stored physical outputs. Headline return excludes ownership/overhead and land; return over listed costs subtracts fixed lines and quoted rent. Payouts/subsidies excluded.",
                          "units": {"relative_yield_by_crop": "fraction of crop normal", "family_pressure_index": "uncalibrated index",
                                    "metrics": "Units encoded in metric names; cover establishment is fraction of scored crop-years, "
                                               "including years without cover. Cover biomass is all-year mean."},
                          "latency_seconds": time.perf_counter() - started})


DEFAULT_RUNTIME = Runtime()
