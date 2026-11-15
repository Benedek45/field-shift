"""Farmer-facing JSON form; all returned farm/economic quantities use SI units."""

import json
import re
from pathlib import Path

from fieldshift.agent import farms, scoreboard
from fieldshift.planner.data import MAPPING, county_options
from fieldshift.sim.economics import ACRES_HA
from fieldshift.sim.traits import CATALOG

VERSION = "farmer-form-v1"
PRIORITY_KEYS = {"money": "income", "water": "water", "soil": "soil", "nitrogen_loss": "nitrogen",
                 "erosion": "erosion", "ground_cover": "cover", "forage": "forage", "local_market": "market"}
RISK = {"accept bad years": 1, "balanced": 3, "protect me in bad years": 5}
# The four A3 profiles, vm/a3_cases.json p1..p4; no economic quotes are implied.
# The A3 profiles, with aspects a profile does not name set to 0 ("does not matter") instead of A3's 1:
# eight aspects at 1 together outweighed money 5 under min-max scaling, so the "profit" preset picked a
# rotation earning less than the farm's current one (York County NE, 2026-11-15).
PRESETS = {
    "profit": dict(zip(scoreboard.KEYS, (5, 0, 0, 0, 0, 0, 0, 1, 0), strict=True)),
    "steady": dict(zip(scoreboard.KEYS, (4, 2, 2, 0, 0, 0, 0, 5, 0), strict=True)),
    "soil_builder": dict(zip(scoreboard.KEYS, (3, 0, 5, 0, 5, 4, 0, 3, 0), strict=True)),
    "water_saver": dict(zip(scoreboard.KEYS, (3, 5, 0, 4, 0, 0, 0, 3, 0), strict=True)),
}


def obj(value, name, allowed):
    if not isinstance(value, dict):
        raise farms.FarmError(f"{name} must be a JSON object")
    if value.keys() - set(allowed):
        raise farms.FarmError(f"{name}: unknown fields {sorted(value.keys() - set(allowed))}")
    return value


def quantity(value, name, factors):
    """Numeric input uses the first (SI) unit; an object names another unit."""
    if isinstance(value, dict):
        obj(value, name, {"value", "unit"})
        unit = value.get("unit")
        if not isinstance(unit, str) or unit not in factors:
            raise farms.FarmError(f"{name}.unit must be one of {list(factors)}")
        value, factor = value.get("value"), factors[unit]
    else:
        factor = next(iter(factors.values()))
    return farms.number(value, name, 0, 1e9) * factor


def _county(location):
    def normal(s):
        return re.sub(r"\s+", " ", s.split("·")[0].casefold().replace(" county", "").strip())
    adm = location.get("adm_id")
    name = location.get("county")
    if name is not None:
        if not isinstance(name, str):
            raise farms.FarmError("location.county must be 'County, ST'")
        matches = [k for k, v in county_options().items() if normal(v) == normal(name)]
        if len(matches) != 1:
            raise farms.FarmError(f"Unknown/ambiguous county {name!r}; use 'County, ST' or adm_id")
        if adm and adm != matches[0]:
            raise farms.FarmError("location.adm_id and county identify different counties")
        adm = matches[0]
    if not adm:
        raise farms.FarmError("location requires adm_id or county ('County, ST')")
    if not isinstance(adm, str) or adm not in county_options():
        raise farms.FarmError(f"Unknown/unsupported county adm_id {adm!r}")
    return adm


def economics(value):
    aliases = {"fertilizer_n_usd_lb", "water_usd_acre_foot", "land_rent_usd_acre",
               "energy_usd_acre_inch", "crop_prices", "crop_cost_parts_usd_acre"}
    from fieldshift.sim.economics import SCALARS
    obj(value, "economics", set(SCALARS) | {"crop_price_usd_t", "crop_cost_usd_ha"} | aliases)
    out = {k: v for k, v in value.items() if k not in aliases}
    for src, dst, factor in (("fertilizer_n_usd_lb", "fertilizer_n_usd_kg", 1 / .45359237),
                             ("water_usd_acre_foot", "water_charge_usd_per_mm_ha", ACRES_HA / 304.8),
                             ("land_rent_usd_acre", "land_rent_usd_ha", ACRES_HA),
                             ("energy_usd_acre_inch", "irrigation_energy_usd_per_mm_ha", ACRES_HA / 25.4)):
        if src in value:
            if dst in out or (src == "water_usd_acre_foot" and "usd_per_acre_foot" in out):
                raise farms.FarmError(f"economics: specify {dst} in one unit only")
            out[dst] = farms.number(value[src], src, 0, 1e9) * factor
    if "crop_prices" in value:
        if "crop_price_usd_t" in out:
            raise farms.FarmError("Use crop_prices OR crop_price_usd_t")
        prices = obj(value["crop_prices"], "economics.crop_prices", CATALOG.keys() | {"cottonseed"})
        converted = {}
        bushel_factors = {key: factor for key, factor in MAPPING.values()
                         if key in {"corn", "soybean", "winter_wheat", "oats", "barley", "sorghum", "rye", "flaxseed"}}
        bushel_factors["spring_wheat"] = bushel_factors["durum"] = bushel_factors["winter_wheat"]
        for key, v in prices.items():
            factors = {"usd/t": 1.}
            if key in bushel_factors:
                factors["usd/bu"] = ACRES_HA / bushel_factors[key]
            if key in {"cotton", "sunflower", "peanuts", "rice"}:
                factors["usd/lb"] = 1000 / .45359237
            if key in {"hay", "alfalfa", "corn_silage", "cottonseed", "sugarbeets"}:
                factors["usd/ton"] = 1 / .90718474  # US short ton, as-reported mass
            converted[key] = quantity(v, f"economics.crop_prices.{key}", factors)
        out["crop_price_usd_t"] = converted
    if "crop_cost_parts_usd_acre" in value:
        if "crop_cost_usd_ha" in out:
            raise farms.FarmError("Use crop_cost_parts_usd_acre OR crop_cost_usd_ha")
        parts = value["crop_cost_parts_usd_acre"]
        if not isinstance(parts, dict) or any(not isinstance(v, dict) for v in parts.values()):
            raise farms.FarmError("crop_cost_parts_usd_acre maps crop keys to cost-part objects")
        out["crop_cost_usd_ha"] = {
            c: {k: farms.number(v, f"{c}.{k} USD/ac", 0, 100000) * ACRES_HA for k, v in p.items()}
            for c, p in parts.items()}
    return farms.validate_economics(out)


def validate(value):
    obj(value, "form", {"location", "current_rotation", "field_facts", "priorities", "limits",
                        "economics", "weather_view", "free_text", "anonymous"})
    location = obj(value.get("location"), "location", {"adm_id", "county", "irrigated", "well_capacity"})
    if not isinstance(location.get("irrigated"), bool):
        raise farms.FarmError("location.irrigated must explicitly be true or false")
    overrides = {}
    if "well_capacity" in location:
        if not location["irrigated"]:
            raise farms.FarmError("well_capacity requires irrigated: true")
        overrides["irrigation_capacity_mm_yr"] = quantity(location["well_capacity"], "well_capacity",
                                                        {"mm/yr": 1, "in/yr": 25.4})
    facts = obj(value.get("field_facts", {}), "field_facts",
                {"slope_pct", "organic_matter_pct", "water_holding_capacity", "awc_mm_per_m"})
    for k in ("slope_pct", "organic_matter_pct", "awc_mm_per_m"):
        if k in facts:
            overrides[k] = farms.number(facts[k], k, *farms.OVERRIDES[k][:2])
    if "water_holding_capacity" in facts:
        if "awc_mm_per_m" in facts:
            raise farms.FarmError("Supply water-holding capacity in one unit only")
        overrides["awc_mm_per_m"] = quantity(facts["water_holding_capacity"], "water_holding_capacity",
                                            {"mm/m": 1, "in/ft": 25.4 / .3048})
    p = obj(value.get("priorities", {}), "priorities", {*PRIORITY_KEYS, "risk_attitude", "preset"})
    preset = p.get("preset", "steady")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise farms.FarmError(f"priorities.preset must be one of {list(PRESETS)}")
    weights = dict(PRESETS[preset])
    for src, dst in PRIORITY_KEYS.items():
        if src in p:
            weights[dst] = farms.number(p[src], f"priorities.{src}", 0, 5)
    if "risk_attitude" in p:
        risk = p["risk_attitude"]
        if not isinstance(risk, str) or risk not in RISK:
            raise farms.FarmError(f"risk_attitude must be one of {list(RISK)}")
        weights["resilience"] = RISK[risk]
    limits = dict(obj(value.get("limits", {}), "limits",
                      {*scoreboard.NUMERIC_LIMITS, *scoreboard.BOOL_LIMITS, *scoreboard.LIST_LIMITS,
                       "min_margin_share_of_existing", "minimum_share_current_income",
                       "no_new_machinery", "include_crops"}))
    for src, dst in (("minimum_share_current_income", "min_margin_share_of_existing"),
                     ("no_new_machinery", "no_new_equipment"), ("include_crops", "require_crops")):
        if src in limits:
            if dst in limits:
                raise farms.FarmError(f"limits: duplicate {src}/{dst}")
            limits[dst] = limits.pop(src)
    units = {"max_field_water_use_mm_yr": {"mm/yr": 1, "in/yr": 25.4},
             "max_nitrate_kg_n_ha_yr": {"kg_n/ha/yr": 1, "lb_n/ac/yr": .45359237 * ACRES_HA},
             "max_erosion_t_ha_yr": {"t/ha/yr": 1, "ton/ac/yr": .90718474 * ACRES_HA},
             "max_days_without_living_cover_yr": {"days/yr": 1},
             "min_margin_share_of_existing": {"fraction": 1, "%": .01}}
    for k, factors in units.items():
        if k in limits:
            limits[k] = quantity(limits[k], f"limits.{k}", factors)
    _, limits, _, _, _ = scoreboard.validate({"weights": weights, "limits": limits})
    for k in scoreboard.LIST_LIMITS:
        if any(c not in CATALOG for c in limits.get(k, [])):
            raise farms.FarmError(f"limits.{k} contains an unknown catalogue crop")
    if set(limits.get("require_crops", [])) & set(limits.get("exclude_crops", [])):
        raise farms.FarmError("A crop cannot be both included and excluded")
    view = value.get("weather_view", "record")
    if view not in ("record", "outlook"):
        raise farms.FarmError("weather_view must be record or outlook")
    text = value.get("free_text", "")
    if not isinstance(text, str) or len(text) > 8000:
        raise farms.FarmError("free_text must be a string of at most 8000 characters")
    record = {"adm_id": _county(location), "irrigated": location["irrigated"], "overrides": overrides,
              "priorities": {"text": text, "weights": weights}, "constraints": [],
              "economics": economics(value.get("economics", {})), "anonymous": value.get("anonymous", False)}
    if "current_rotation" in value:
        record["existing_rotation"] = farms.validate_rotation(value["current_rotation"])
    return {"schema": VERSION, "farm": farms.validate(record), "weights": weights, "limits": limits,
            "weather_view": view, "free_text": text,
            "unit_conventions": "SI; USD/t grain/hay as reported (no extra moisture adjustment); cotton is lint. "
                                "USD/bu uses planner.data.MAPPING NASS mass conversions; ton means US short ton."}


def load(path):
    try:
        return validate(json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
            farms.FarmError(f"Nonfinite JSON value {x}"))))
    except (json.JSONDecodeError, OSError) as exc:
        raise farms.FarmError(f"Cannot read form {path}: {exc}") from exc
