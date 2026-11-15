"""Streaming raw-data conversion and vectorised county-season feature cache."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.features.stress import compute_stress, sowing_dates
from fieldshift.features.traits import CROPS, numeric_traits

RAW = Path("data/raw/cybench/cybench-data")
ROOT = Path("data/processed/yield_stress")
KEYS = ["adm_id", "date"]
WEATHER = ["tmin", "tmax", "prec", "rad", "tavg", "et0", "vpd", "cwb"]


def convert_csvs(root=ROOT, raw=RAW):
    """Each multi-GB CSV is read exactly once; completed inputs are reusable."""
    for crop in ("maize", "wheat"):
        for kind, values in (("meteo", WEATHER), ("soil_moisture", ["ssm", "rsm"])):
            dest = root / "source" / crop / kind
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            source = raw / crop / "US" / f"{kind}_{crop}_US.csv"
            count = 0
            for i, chunk in enumerate(pd.read_csv(
                source, usecols=KEYS + values,
                dtype={**{v: "float32" for v in values}, "date": "int32"},
                chunksize=500_000,
            )):
                chunk["date"] = pd.to_datetime(chunk.date.astype(str), format="%Y%m%d")
                for year, part in chunk.groupby(chunk.date.dt.year):
                    folder = dest / str(year)
                    folder.mkdir(exist_ok=True)
                    part.to_parquet(folder / f"{i:04d}.parquet", index=False)
                count += len(chunk)
            (dest / "complete.json").write_text(json.dumps({"rows": count}))
            print(f"Converted {source.name}: {count:,} rows", flush=True)


def _read_source(root, crop, kind, year):
    files = sorted((root / "source" / crop / kind / str(year)).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def canonical_daily(root=ROOT):
    """Use the same crop-mask support for all labels, maize preferred."""
    dest = root / "daily"
    dest.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for year in range(2001, 2024):
        out = dest / f"{year}.parquet"
        audit = dest / f"{year}.json"
        if out.exists() and audit.exists():
            comparisons.append(json.loads(audit.read_text()))
            continue
        maize = _read_source(root, "maize", "meteo", year)
        wheat = _read_source(root, "wheat", "meteo", year)
        overlap = maize.merge(wheat, on=KEYS, suffixes=("_m", "_w"), validate="one_to_one")
        comparison = {"year": year, "overlap_days": len(overlap)}
        for col in WEATHER:
            diff = (overlap[f"{col}_m"] - overlap[f"{col}_w"]).abs()
            comparison[col + "_mean_abs_difference_raw"] = float(diff.mean())
        comparison["identical_weather_fraction"] = float(
            np.column_stack([overlap[f"{v}_m"].eq(overlap[f"{v}_w"]) for v in WEATHER]).all(axis=1).mean()
        )
        del overlap
        # Preserve all available counties, but never change mask by target crop.
        maize_ids = set(maize.adm_id)
        d = pd.concat([maize, wheat.loc[~wheat.adm_id.isin(maize_ids)]], ignore_index=True)
        del maize, wheat
        moisture = []
        for crop in ("maize", "wheat"):
            m = _read_source(root, crop, "soil_moisture", year)
            if len(m):
                keep = m.adm_id.isin(maize_ids)
                moisture.append(m.loc[keep if crop == "maize" else ~keep, KEYS + ["rsm"]])
        if moisture:
            d = d.merge(pd.concat(moisture), on=KEYS, how="left", validate="one_to_one")
        else:
            d["rsm"] = np.nan
        d["rad"] /= 1e6
        d["vpd"] /= 10
        d["cwb"] = d.prec - d.et0
        d.to_parquet(out, index=False)
        audit.write_text(json.dumps(comparison, indent=2))
        comparisons.append(comparison)
        print(f"Canonical weather {year}: {len(d):,} days", flush=True)
    (root / "weather_audit.json").write_text(json.dumps(comparisons, indent=2))


def soil_table(raw=RAW):
    parts = [
        pd.read_csv(raw / c / "US" / f"soil_{c}_US.csv").drop(columns="crop_name")
        for c in ("maize", "wheat")
    ]
    return pd.concat(parts).drop_duplicates("adm_id", keep="first").set_index("adm_id")


def labels(root=ROOT, path=Path("data/processed/nass_county_crops.csv.gz")):
    out = root / "labels.parquet"
    if out.exists():
        return pd.read_parquet(out)
    parts = []
    cols = [
        "commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc",
        "statisticcat_desc", "unit_desc", "domain_desc", "domaincat_desc",
        "reference_period_desc", "state_fips_code", "county_code", "state_ansi",
        "county_ansi", "state_alpha", "year", "value",
    ]
    for d in pd.read_csv(path, usecols=cols, dtype={
        c: "string" for c in ["state_fips_code", "county_code", "state_ansi", "county_ansi"]
    }, chunksize=500_000):
        crop = np.select([
            d.commodity_desc.eq("CORN") & d.class_desc.eq("ALL CLASSES") & d.util_practice_desc.eq("GRAIN"),
            d.commodity_desc.eq("SOYBEANS") & d.class_desc.eq("ALL CLASSES"),
            d.commodity_desc.eq("WHEAT") & d.class_desc.eq("WINTER"),
        ], ["corn", "soybean", "winter_wheat"], default="")
        valid_county = (
            d.county_ansi.str.zfill(3).eq(d.county_code.str.zfill(3))
            & d.state_ansi.str.zfill(2).eq(d.state_fips_code.str.zfill(2))
            & pd.to_numeric(d.county_code, errors="coerce").between(1, 899)
        )
        mask = (
            (crop != "") & valid_county.fillna(False) & d.year.between(2003, 2023)
            & d.domain_desc.eq("TOTAL") & d.domaincat_desc.eq("NOT SPECIFIED")
            & d.reference_period_desc.eq("YEAR") & d.value.gt(0)
            & d.prodn_practice_desc.isin(["NON-IRRIGATED", "ALL PRODUCTION PRACTICES", "IRRIGATED"])
            & ((d.statisticcat_desc.eq("YIELD") & d.unit_desc.eq("BU / ACRE"))
               | (d.statisticcat_desc.eq("AREA HARVESTED") & d.unit_desc.eq("ACRES")))
        )
        d = d.loc[mask].copy()
        d["crop"] = crop[mask]
        d["adm_id"] = "US-" + d.state_fips_code.str.zfill(2) + "-" + d.county_code.str.zfill(3)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    key = ["crop", "adm_id", "year"]
    if d.duplicated(key + ["statisticcat_desc", "prodn_practice_desc"]).any():
        raise ValueError("Unexpected duplicate NASS series after explicit domain filters")
    area = d.loc[d.statisticcat_desc.eq("AREA HARVESTED")].pivot(
        index=key, columns="prodn_practice_desc", values="value"
    )
    share = (area["IRRIGATED"] / area["ALL PRODUCTION PRACTICES"]).rename("irrigated_share")
    y = d.loc[d.statisticcat_desc.eq("YIELD")].copy()
    non_counts = y.loc[y.prodn_practice_desc.eq("NON-IRRIGATED")].groupby(["crop", "adm_id"]).size()
    preferred = pd.MultiIndex.from_frame(y[["crop", "adm_id"]]).map(non_counts).fillna(0) >= 8
    y = y.loc[
        (preferred & y.prodn_practice_desc.eq("NON-IRRIGATED"))
        | (~preferred & y.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES"))
    ].merge(share, on=key, how="left")
    y = y.loc[y.prodn_practice_desc.eq("NON-IRRIGATED") | y.irrigated_share.isna() | y.irrigated_share.le(.1)]
    # 56 lb/bu maize, 60 lb/bu soybean/wheat; lb to kg, acres to hectares.
    y["yield_t_ha"] = y.value * y.crop.map({"corn": 56, "soybean": 60, "winter_wheat": 60}) * .45359237 / .40468564224 / 1000
    y["irrigation_unknown"] = y.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES") & y.irrigated_share.isna()
    count = y.groupby(["crop", "adm_id"]).year.transform("size")
    y = y.loc[count.ge(8), key + ["state_alpha", "yield_t_ha", "prodn_practice_desc", "irrigated_share", "irrigation_unknown"]]
    y.to_parquet(out, index=False)
    (root / "label_audit.json").write_text(json.dumps({
        "rows_per_crop": y.groupby("crop").size().to_dict(),
        "unknown_irrigation_per_crop": y.groupby("crop").irrigation_unknown.mean().to_dict(),
        "practice_counts": y.prodn_practice_desc.value_counts().to_dict(),
        "rule": "Prefer NON-IRRIGATED county series with >=8 years; else ALL. Exclude ALL when known irrigated harvested-area share >10%; keep unknown with explicit warning.",
    }, indent=2))
    return y


def build_features(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    final = root / "features.parquet"
    if final.exists():
        return pd.read_parquet(final)
    convert_csvs(root)
    canonical_daily(root)
    y = labels(root)
    soil = soil_table()
    # Label-free leave-calendar-year-out calendar-day climatology, month/day
    # rather than ordinal day so leap years do not shift windows.
    climatology = root / "rsm_climatology.parquet"
    if not climatology.exists():
        aggregates = []
        for year in range(2003, 2024):
            d = pd.read_parquet(root / "daily" / f"{year}.parquet", columns=KEYS + ["rsm"])
            d["calendar_day"] = d.date.dt.month * 100 + d.date.dt.day
            aggregates.append(d.groupby(["adm_id", "calendar_day"]).rsm.agg(["sum", "count"]))
        pd.concat(aggregates).groupby(level=[0, 1]).sum().to_parquet(climatology)
    clim = pd.read_parquet(climatology).reset_index()
    parts = []
    previous = None
    temperature_qc = []
    for year in range(2002, 2024):
        current = pd.read_parquet(root / "daily" / f"{year}.parquet")
        inversion = current.tmin - current.tmax
        if inversion.max() > 1:
            raise ValueError("CY-Bench minimum exceeds maximum by >1 C")
        bad = inversion.gt(0)
        temperature_qc.append({"year": year, "inverted_days": int(bad.sum()),
                               "max_inversion_c": float(max(0, inversion.max()))})
        midpoint = (current.loc[bad, "tmin"] + current.loc[bad, "tmax"]) / 2
        current.loc[bad, "tmin"] = midpoint
        current.loc[bad, "tmax"] = midpoint
        current["calendar_day"] = current.date.dt.month * 100 + current.date.dt.day
        current = current.merge(clim, on=["adm_id", "calendar_day"], how="left", validate="many_to_one")
        count = current["count"] - current.rsm.notna().astype(int)
        mean = (current["sum"] - current.rsm.fillna(0)) / count.where(count.gt(0))
        current["rsm_anomaly"] = current.rsm - mean
        current = current.drop(columns=["sum", "count", "calendar_day"])
        if year >= 2003:
            cache = root / f"features_{year}.parquet"
            if cache.exists():
                parts.append(pd.read_parquet(cache))
            else:
                features = []
                for crop, traits in CROPS.items():
                    target = y.loc[y.year.eq(year) & y.crop.eq(crop)]
                    ids = set(target.adm_id)
                    weather = current.loc[current.adm_id.isin(ids)]
                    if numeric_traits(traits)["autumn_sown"]:
                        weather = pd.concat([previous.loc[previous.adm_id.isin(ids)], weather], ignore_index=True)
                    sow = sowing_dates(weather, traits, year)
                    f = compute_stress(weather, traits, sow, year).join(soil)
                    for k, v in numeric_traits(traits).items():
                        f[f"trait__{k}"] = v
                    f["crop"], f["year"] = crop, year
                    features.append(f.reset_index())
                f = pd.concat(features, ignore_index=True)
                f = f.merge(y, on=["crop", "adm_id", "year"], validate="one_to_one")
                f.to_parquet(cache, index=False)
                parts.append(f)
                print(f"Features {year}: {len(f):,} seasons", flush=True)
        previous = current
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(final, index=False)
    (root / "temperature_qc.json").write_text(json.dumps(temperature_qc, indent=2))
    return result
