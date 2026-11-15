"""Opaque farm registry. Run ``python -m fieldshift.agent.farms add --help``."""

import argparse
import json
import math
import os
import re
import secrets
import string
from pathlib import Path

from fieldshift.locking import file_lock

ROOT = Path("data/processed/agent")
OVERRIDES = {
    "rain_change_pct": (-100, 200, "% relative to historical precipitation"),
    "temp_change_c": (-15, 15, "degrees C added to daily temperature"),
    "slope_pct": (0, 100, "% rise/run"),
    "organic_matter_pct": (.05, 30, "% soil mass, upper 0.3 m"),
    "awc_mm_per_m": (10, 500, "mm available water per m of soil"),
    "irrigation_capacity_mm_yr": (0, 5000, "mm applied irrigation per calendar year"),
}
WEIGHTS = {"income", "water", "soil", "nitrogen", "nitrate", "erosion", "cover",
           "forage", "resilience", "market"}


class FarmError(ValueError):
    """Invalid farm definition or missing opaque ID."""


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FarmError(f"{name} must be a finite number")
    if not low <= value <= high:
        raise FarmError(f"{name} must be between {low} and {high}")
    return float(value)


def validate(record):
    if not isinstance(record, dict):
        raise FarmError("Farm must be a JSON object")
    allowed = {"adm_id", "irrigated", "overrides", "priorities", "constraints", "anonymous", "economics", "existing_rotation"}
    if record.keys() - allowed:
        raise FarmError(f"Unknown farm fields: {sorted(record.keys() - allowed)}")
    adm = record.get("adm_id")
    if not isinstance(adm, str) or not re.fullmatch(r"US-\d{2}-\d{3}", adm):
        raise FarmError("adm_id must look like US-19-169")
    from fieldshift.data.power import load_county_daily
    _, meta = load_county_daily()
    if adm not in meta["counties"]:
        raise FarmError(f"Unknown county {adm}; choose an adm_id in the local POWER archive")
    for flag in ("irrigated", "anonymous"):
        if not isinstance(record.get(flag, False), bool):
            raise FarmError(f"{flag} must be true or false")
    overrides = record.get("overrides", {})
    if not isinstance(overrides, dict) or overrides.keys() - OVERRIDES.keys():
        raise FarmError(f"Valid overrides: {list(OVERRIDES)}")
    overrides = {k: number(v, k, *OVERRIDES[k][:2]) for k, v in overrides.items()}
    priorities = record.get("priorities", {"text": "", "weights": {}})
    if isinstance(priorities, str):
        priorities = {"text": priorities}
    if not isinstance(priorities, dict) or priorities.keys() - {"text", "weights"}:
        raise FarmError('priorities must be {"text": "...", "weights": {...}}')
    text, weights = priorities.get("text", ""), priorities.get("weights", {})
    if not isinstance(text, str) or len(text) > 8000:
        raise FarmError("Priority text must be a string, at most 8000 characters")
    if not isinstance(weights, dict) or weights.keys() - WEIGHTS:
        raise FarmError(f"Valid weight names: {sorted(WEIGHTS)}")
    weights = {k: number(v, f"weight {k}", 0, 100) for k, v in weights.items()}
    constraints = record.get("constraints", [])
    if (not isinstance(constraints, list) or len(constraints) > 50
            or any(not isinstance(c, str) or len(c) > 2000 for c in constraints)):
        raise FarmError("constraints must be a list of at most 50 strings, each <=2000 characters")
    economics = validate_economics(record.get("economics", {}))
    if ("irrigation_capacity_mm_yr" in economics and "irrigation_capacity_mm_yr" in overrides
            and economics["irrigation_capacity_mm_yr"] != overrides["irrigation_capacity_mm_yr"]):
        raise FarmError("Conflicting irrigation_capacity_mm_yr in economics and overrides")
    if (adm.startswith("US-06-") and record.get("irrigated", False)
            and "irrigation_energy_usd_per_mm_ha" in economics and "water_charge_usd_per_mm_ha" not in economics):
        raise FarmError("California default water price includes pumping: supply both water charge and energy to avoid double counting")
    out = {"adm_id": adm, "irrigated": record.get("irrigated", False), "overrides": overrides, "economics": economics,
             "priorities": {"text": text, "weights": weights}, "constraints": constraints,
              "anonymous": record.get("anonymous", False)}
    if "existing_rotation" in record:
        out["existing_rotation"] = validate_rotation(record["existing_rotation"])
    return out


def validate_rotation(value):
    """Catalogue validation; local calendar/stand feasibility is checked by Farm."""
    from fieldshift.agent.runtime import WINTER_ALIASES
    from fieldshift.sim.traits import CATALOG
    if not isinstance(value, list) or not 1 <= len(value) <= 6:
        raise FarmError("existing_rotation must contain 1–6 seasons")
    out = []
    for i, step in enumerate(value):
        if not isinstance(step, dict) or step.keys() - {"crop", "winter", "double_crop"}:
            raise FarmError(f"existing_rotation[{i}]: expected crop and optional winter/double_crop")
        crop = step.get("crop")
        if not isinstance(crop, str) or crop not in CATALOG or CATALOG[crop].v("cover_crop") or crop == "double_crop_soybean":
            raise FarmError(f"existing_rotation[{i}]: unknown main crop {crop!r}")
        winter = step.get("winter", "")
        winter = "" if winter is None else winter
        if not isinstance(winter, str):
            raise FarmError(f"existing_rotation[{i}].winter must be a crop key")
        winter = WINTER_ALIASES.get(winter, winter)
        if winter and (winter not in CATALOG or not CATALOG[winter].v("cover_crop")):
            raise FarmError(f"existing_rotation[{i}]: unknown winter cover {winter!r}")
        double = step.get("double_crop", "")
        double = "" if double is None else double
        if not isinstance(double, str) or double not in ("", "soybean", "double_crop_soybean") or (double and winter):
            raise FarmError(f"existing_rotation[{i}]: use winter OR double_crop soybean")
        out.append({"crop": crop, **({"winter": winter} if winter else {}),
                    **({"double_crop": "soybean"} if double else {})})
    return out


def validate_economics(value):
    from fieldshift.sim.economics import ACRES_HA, MM_FOOT, PARTS, SCALARS
    from fieldshift.sim.traits import CATALOG
    if not isinstance(value, dict) or value.keys() - (set(SCALARS) | {"crop_price_usd_t", "crop_cost_usd_ha"}):
        raise FarmError(f"Valid economics keys: {sorted(set(SCALARS) | {'crop_price_usd_t', 'crop_cost_usd_ha'})}")
    out = {}
    for key, val in value.items():
        if key in SCALARS:
            out[key] = number(val, key, *SCALARS[key][:2])
        else:
            if not isinstance(val, dict) or val.keys() - (CATALOG.keys() | ({"cottonseed"} if key == "crop_price_usd_t" else set())):
                raise FarmError(f"{key} must map supported crop keys to values")
            out[key] = {}
            for crop, v in val.items():
                if key == "crop_price_usd_t":
                    if crop != "cottonseed" and CATALOG[crop].v("cover_crop"):
                        raise FarmError("Unharvested covers have no sale price; use cover_seed_usd_ha or cover cost parts")
                    out[key][crop] = number(v, f"{crop} price USD/t", 0, 1000000)
                else:
                    if not isinstance(v, dict) or v.keys() - set(PARTS):
                        raise FarmError(f"Valid crop cost parts: {PARTS}; N is priced via fertilizer_n_usd_kg")
                    if (CATALOG[crop].v("cover_crop") or crop=="double_crop_soybean") and any(
                        v.get(p,0)!=0 for p in ("repeat_crop","repeat_crop_fixed")
                    ):
                        raise FarmError("Repeat-crop cost parts apply only to main crops")
                    out[key][crop] = {p: number(x, f"{crop}.{p} USD/ha", 0, 100000) for p, x in v.items()}
    if "usd_per_acre_foot" in out:
        if "water_charge_usd_per_mm_ha" in out:
            raise FarmError("Specify water charge in one unit only")
        out["water_charge_usd_per_mm_ha"] = out.pop("usd_per_acre_foot") * ACRES_HA / MM_FOOT
    return out


def edit_economics(farm_id, edits, root=None):
    """Immutable fork; nested crop/part dictionaries merge without losing old values."""
    record = get(farm_id, root)
    edits = validate_economics(edits)
    current = record["economics"]
    for key, value in edits.items():
        if key in ("crop_price_usd_t", "crop_cost_usd_ha"):
            target = current.setdefault(key, {})
            for crop, val in value.items():
                if isinstance(val, dict):
                    target.setdefault(crop, {}).update(val)
                else:
                    target[crop] = val
        else:
            current[key] = value
    if "irrigation_capacity_mm_yr" in edits:
        record["overrides"].pop("irrigation_capacity_mm_yr", None)
    return add(record, root)


def physical_record(record):
    """Prices/wages are not field inputs; capacity is."""
    from fieldshift.sim.economics import capacity
    out = {k: record[k] for k in ("adm_id", "irrigated", "constraints")}
    out["overrides"] = dict(record["overrides"])
    if "existing_rotation" in record:
        out["existing_rotation"] = record["existing_rotation"]
    cap = record.get("economics", {}).get("irrigation_capacity_mm_yr",
                  out["overrides"].get("irrigation_capacity_mm_yr", capacity(record["adm_id"])["value"]))
    if cap is not None and record["irrigated"]:
        out["overrides"]["irrigation_capacity_mm_yr"] = cap
    return out


def get(farm_id, root=None):
    path = Path(root or ROOT) / "farms.json"
    if not isinstance(farm_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{12,80}", farm_id):
        raise FarmError("Invalid farm_id; create one with python -m fieldshift.agent.farms add")
    records = json.loads(path.read_text()) if path.exists() else {}
    if farm_id not in records:
        raise FarmError("Unknown farm_id; register the farm before using this service")
    return validate(records[farm_id])


def add(record, root=None):
    record = validate(record)
    root = Path(root or ROOT)
    root.mkdir(parents=True, exist_ok=True)
    with file_lock(root / ".registry.lock"):
        path = root / "farms.json"
        records = json.loads(path.read_text()) if path.exists() else {}
        # 150 bits of entropy; letters only so an opaque ID cannot accidentally
        # introduce calendar-year digits or uppercase state codes into a blind brief.
        farm_id = "".join(secrets.choice(string.ascii_lowercase) for _ in range(32))
        while farm_id in records:
            farm_id = "".join(secrets.choice(string.ascii_lowercase) for _ in range(32))
        records[farm_id] = record
        temporary = root / "farms.json.part"
        with temporary.open("w") as f:
            os.chmod(temporary, 0o600)
            json.dump(records, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    return farm_id


def _pairs(values):
    result = {}
    for item in values:
        try:
            k, v = item.split("=", 1)
            if k in result:
                raise ValueError(f"Duplicate key {k}")
            result[k] = float(v)
        except ValueError as exc:
            raise FarmError(f"Expected name=number, got {item!r}") from exc
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add")
    a.add_argument("--adm-id", required=True)
    a.add_argument("--irrigated", action="store_true")
    a.add_argument("--anonymous", action="store_true")
    a.add_argument("--override", action="append", default=[], metavar="NAME=NUMBER")
    a.add_argument("--priority", action="append", default=[])
    a.add_argument("--weight", action="append", default=[], metavar="NAME=NUMBER")
    a.add_argument("--constraint", action="append", default=[])
    args = parser.parse_args()
    try:
        print(add({"adm_id": args.adm_id, "irrigated": args.irrigated, "anonymous": args.anonymous,
                   "overrides": _pairs(args.override),
                   "priorities": {"text": "\n".join(args.priority), "weights": _pairs(args.weight)},
                   "constraints": args.constraint}))
    except FarmError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
