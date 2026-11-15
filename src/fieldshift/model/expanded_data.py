"""Multi-crop cached physical features and accelerator-ready weather layout."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data.crop_labels import ROOT, build_labels
from fieldshift.features.stress import (
    STAGES,
    aggregate_stress,
    daily_stages,
    sowing_dates,
)
from fieldshift.features.traits import CROPS, numeric_traits
from fieldshift.model.dataset import ROOT as OLD_ROOT
from fieldshift.model.dataset import soil_table

VARS = ("tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "cwb", "rsm")
FEATURE_VERSION = 2  # moisture calendar weights count only observed RSM days
METADATA = {
    "crop", "adm_id", "year", "series_id", "state_alpha", "yield_t_ha",
    "harvested_acres", "prodn_practice_desc", "irrigation_source", "primary",
    "season_start_ordinal", "season_end_ordinal", "county_index", "start_index",
    "season_length",
}


def feature_columns(data):
    return sorted(c for c in data if c not in METADATA and not c.startswith("climo__"))


def canonical_memmap(root=ROOT):
    """Materialise one county-major array from existing Parquet, not raw CSVs."""
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "cybench_weather.json"
    path = root / "cybench_weather.f32"
    if meta_path.exists():
        return path, json.loads(meta_path.read_text())
    first = pd.read_parquet(OLD_ROOT / "daily" / "2001.parquet")
    counties = sorted(first.adm_id.unique())
    start, end = pd.Timestamp("2001-01-01"), pd.Timestamp("2023-12-31")
    shape = (len(counties), (end - start).days + 1, len(VARS))
    array = np.memmap(path, mode="w+", dtype="float32", shape=shape)
    array[:] = np.nan
    for year in range(2001, 2024):
        d = first if year == 2001 else pd.read_parquet(OLD_ROOT / "daily" / f"{year}.parquet")
        c = pd.Index(counties).get_indexer(d.adm_id)
        day = (d.date - start).dt.days.to_numpy()
        array[c, day] = d[list(VARS)].to_numpy(dtype=np.float32)
    array.flush()
    meta = {"shape": shape, "start_date": str(start.date()), "counties": counties,
            "variables": list(VARS), "source": "cybench", "rsm_scale": 500.}
    meta_path.write_text(json.dumps(meta))
    return path, meta


def weather_spec(source):
    if source == "cybench":
        return canonical_memmap()
    if source not in {"power", "power_matched"}:
        raise ValueError(source)
    path = Path("data/processed/weather/power_county_daily.f32")
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    variables = meta["variables"]
    if isinstance(variables, dict):
        variables = list(variables)
    if variables and isinstance(variables[0], dict):
        expected_units = {
            "tmin": "degC", "tmax": "degC", "tavg": "degC", "prec": "mm/day",
            "rad": "MJ/m2/day", "vpd": "kPa", "et0": "mm/day", "cwb": "mm/day",
            "rh": "%", "wind": "m/s", "gwetroot": "1", "gwettop": "1",
            "gwetprof": "1", "rad_filled": "1",
        }
        for variable in variables:
            name = variable.get("name", variable.get("variable"))
            if name in expected_units and variable.get("units") != expected_units[name]:
                raise ValueError(f"Unexpected POWER units for {name}: {variable.get('units')}")
        variables = [v.get("name", v.get("variable")) for v in variables]
    meta["variable_records"] = meta["variables"]
    meta["variables"] = variables
    meta["source"], meta["rsm_scale"] = source, 1.
    meta["vpd_definition"] = "FAO-56 mean saturation minus actual vapour pressure"
    if source == "power_matched":
        meta["vpd_to_tmax"] = True
        meta["vpd_definition"] = "es(Tmax) minus actual vapour pressure, matched to CY-Bench"
    shape = meta["shape"]
    if (len(shape) != 3 or shape[0] != len(meta["counties"]) or shape[2] != len(variables)
            or len(set(meta["counties"])) != shape[0] or len(set(variables)) != shape[2]):
        raise ValueError("Inconsistent POWER metadata shape, counties or variables")
    required = set(VARS) - {"rsm"}
    if not required.issubset(variables) or "gwetroot" not in variables:
        raise ValueError("Missing required POWER weather variables")
    expected = np.prod(meta["shape"]) * 4
    if path.stat().st_size != expected:
        raise ValueError("POWER weather is not a complete float32 memmap")
    if "end_date" in meta:
        last = pd.Timestamp(meta["start_date"]) + pd.Timedelta(days=shape[1] - 1)
        if last != pd.Timestamp(meta["end_date"]):
            raise ValueError("POWER date range does not match its tensor shape")
    return path, meta


def yearly_frame(array, meta, year):
    start = pd.Timestamp(meta["start_date"])
    lo = max(0, (pd.Timestamp(year, 1, 1) - start).days)
    hi = min(array.shape[1], (pd.Timestamp(year + 1, 1, 1) - start).days)
    if hi <= lo:
        return pd.DataFrame()
    values = np.asarray(array[:, lo:hi])
    n, days, _ = values.shape
    d = pd.DataFrame(values.reshape(-1, len(meta["variables"])), columns=meta["variables"])
    d["adm_id"] = np.repeat(meta["counties"], days)
    d["date"] = np.tile(pd.date_range(start + pd.Timedelta(days=lo), periods=days), n)
    if "rsm" not in d:
        d["rsm"] = d["gwetroot"]
    inversion = d.tmin.gt(d.tmax)
    if (d.tmin - d.tmax).max() > 1:
        raise ValueError("Temperature inversion >1 C in weather")
    if inversion.any():
        d = d.copy()
        mid = (d.loc[inversion, "tmin"] + d.loc[inversion, "tmax"]) / 2
        d.loc[inversion, "tmin"] = mid
        d.loc[inversion, "tmax"] = mid
    if meta.get("vpd_to_tmax"):
        d = align_power_vpd_to_tmax(d)
    return d


def align_power_vpd_to_tmax(daily):
    """Convert FAO-56 mean-saturation VPD to CY-Bench's Tmax definition."""
    d = daily.copy()
    es_max = .6108 * np.exp(17.27 * d.tmax / (d.tmax + 237.3))
    es_min = .6108 * np.exp(17.27 * d.tmin / (d.tmin + 237.3))
    d["vpd"] = d.vpd + .5 * (es_max - es_min)
    return d


def build_expanded(source="cybench"):
    path, meta = weather_spec(source)
    fingerprint = hashlib.sha256(json.dumps(
        {"version": FEATURE_VERSION, "crops": {c: numeric_traits(t) for c, t in CROPS.items()}}, sort_keys=True
    ).encode()).hexdigest()[:12]
    root = ROOT / f"{source}_{fingerprint}"
    root.mkdir(parents=True, exist_ok=True)
    final = root / "features.parquet"
    if final.exists():
        return pd.read_parquet(final), path, meta, root
    labels = build_labels()
    array = np.memmap(path, mode="r", dtype="float32", shape=tuple(meta["shape"]))
    start = pd.Timestamp(meta["start_date"])
    first_year = start.year
    last_year = (start + pd.Timedelta(days=array.shape[1] - 1)).year
    labels = labels.loc[labels.adm_id.isin(meta["counties"]) & labels.year.between(first_year, last_year)]
    soil = soil_table()
    previous = pd.DataFrame()
    features, monthly = [], []
    for year in range(first_year, last_year + 1):
        current = yearly_frame(array, meta, year)
        current["month"] = current.date.dt.month
        monthly.append(current.groupby(["adm_id", "month"]).rsm.agg(["sum", "count"]).assign(year=year).reset_index())
        target_year = labels.loc[labels.year.eq(year)]
        cache = root / f"features_{year}.parquet"
        if cache.exists():
            features.append(pd.read_parquet(cache))
        else:
            parts = []
            for crop, target in target_year.groupby("crop"):
                traits = CROPS[crop]
                if numeric_traits(traits)["autumn_sown"] and previous.empty:
                    continue  # no fabricated previous-autumn weather at archive boundary
                ids = target.adm_id.unique()
                weather = current.loc[current.adm_id.isin(ids)]
                if numeric_traits(traits)["autumn_sown"]:
                    weather = pd.concat([previous.loc[previous.adm_id.isin(ids)], weather], ignore_index=True)
                sow = sowing_dates(weather, traits, year)
                d = daily_stages(weather, traits, sow, year)
                d.attrs.update(numeric_traits(traits))
                f = aggregate_stress(d).join(soil)
                counts = d.loc[d.rsm.notna()].groupby(["adm_id", "stage", "month"]).size()
                climo_columns = {}
                for stage, name in list(enumerate(STAGES)) + [(-1, "forage_regrowth")]:
                    for month in range(1, 13):
                        if stage in counts.index.get_level_values("stage"):
                            values = counts.xs(stage, level="stage")
                            values = values.xs(month, level="month") if month in values.index.get_level_values("month") else pd.Series(dtype=float)
                        else:
                            values = pd.Series(dtype=float)
                        climo_columns[f"climo__{name}__{month}"] = values.reindex(f.index).fillna(0)
                # Trait columns are added together to avoid a fragmented frame.
                trait_frame = pd.DataFrame(
                    {f"trait__{k}": v for k, v in numeric_traits(traits).items()}, index=f.index
                )
                f = pd.concat([f, trait_frame, pd.DataFrame(climo_columns, index=f.index)], axis=1)
                f["crop"], f["year"] = crop, year
                parts.append(f.reset_index())
            if parts:
                f = pd.concat(parts, ignore_index=True).merge(
                    target_year, on=["crop", "adm_id", "year"], validate="one_to_many"
                ).copy()
                f["county_index"] = pd.Index(meta["counties"]).get_indexer(f.adm_id)
                f["start_index"] = f.season_start_ordinal - start.toordinal()
                f["season_length"] = f.season_end_ordinal - f.season_start_ordinal + 1
                f.to_parquet(cache, index=False)
                features.append(f)
                print(f"{source} {year}: {len(f):,} label-strata, {f.crop.nunique()} crops", flush=True)
        previous = current
    pd.concat(monthly, ignore_index=True).to_parquet(root / "monthly_rsm.parquet", index=False)
    data = pd.concat(features, ignore_index=True)
    data.to_parquet(final, index=False)
    (root / "weather.json").write_text(json.dumps(meta, indent=2))
    return data, path, meta, root


def apply_climatology(data, fit_years, root):
    """Training-calendar-year-only county/month climatology for each stage window."""
    weather = pd.read_parquet(root / "monthly_rsm.parquet")
    weather = weather.loc[weather.year.isin(fit_years)]
    climo = weather.groupby(["adm_id", "month"])[["sum", "count"]].sum()
    values = (climo["sum"] / climo["count"].where(climo["count"].gt(0))).unstack("month")
    month_values = values.reindex(data.adm_id).reindex(columns=range(1, 13)).to_numpy()
    out = data.copy()
    for name in STAGES + ("forage_regrowth",):
        weights = data[[f"climo__{name}__{m}" for m in range(1, 13)]].to_numpy()
        valid = np.isfinite(month_values)
        denominator = (weights * valid).sum(axis=1)
        mean = np.nansum(weights * month_values, axis=1) / np.where(denominator > 0, denominator, np.nan)
        out[f"{name}__rsm_anomaly"] = data[f"{name}__rsm"].to_numpy() - mean
    weights = sum(
        data[[f"climo__{name}__{m}" for m in range(1, 13)]].to_numpy()
        for name in STAGES + ("forage_regrowth",)
    )
    denominator = (weights * np.isfinite(month_values)).sum(axis=1)
    mean = np.nansum(weights * month_values, axis=1) / np.where(denominator > 0, denominator, np.nan)
    out["season__rsm_anomaly"] = data["season__rsm"].to_numpy() - mean
    return out
