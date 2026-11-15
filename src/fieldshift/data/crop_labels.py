"""Expanded survey labels and census-derived irrigation, separate from NASS IO."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/processed/yield_expanded")
RAW = Path("data/raw/nass/qs.crops_20261103.txt.gz")
SURVEY = Path("data/processed/nass_county_crops.csv.gz")

# Crop identity is used for data selection and evaluation, never as a predictor.
# Bushel weights are USDA standard pounds per bushel, not volume-to-mass guesses.
SPECS = {
    "corn": ("CORN", "ALL CLASSES", "GRAIN", "BU / ACRE", 56),
    "corn_silage": ("CORN", "ALL CLASSES", "SILAGE", "TONS / ACRE", None),
    "soybean": ("SOYBEANS", "ALL CLASSES", None, "BU / ACRE", 60),
    "winter_wheat": ("WHEAT", "WINTER", None, "BU / ACRE", 60),
    "spring_wheat": ("WHEAT", "SPRING, (EXCL DURUM)", None, "BU / ACRE", 60),
    "durum": ("WHEAT", "SPRING, DURUM", None, "BU / ACRE", 60),
    "alfalfa": ("HAY", "ALFALFA", None, "TONS / ACRE", None),
    "hay": ("HAY", "(EXCL ALFALFA)", None, "TONS / ACRE", None),
    "oats": ("OATS", "ALL CLASSES", None, "BU / ACRE", 32),
    "sorghum": ("SORGHUM", "ALL CLASSES", "GRAIN", "BU / ACRE", 56),
    "barley": ("BARLEY", "ALL CLASSES", None, "BU / ACRE", 48),
    "cotton": ("COTTON", "UPLAND", None, "LB / ACRE", None),
    "peanuts": ("PEANUTS", "ALL CLASSES", None, "LB / ACRE", None),
    "dry_beans": ("BEANS", "DRY EDIBLE, INCL CHICKPEAS", None, "LB / ACRE", None),
    "rye": ("RYE", "ALL CLASSES", None, "BU / ACRE", 56),
    "potatoes": ("POTATOES", "ALL CLASSES", None, "CWT / ACRE", None),
    "sunflower": ("SUNFLOWER", "OIL TYPE", None, "LB / ACRE", None),
    "sugarbeets": ("SUGARBEETS", "ALL CLASSES", None, "TONS / ACRE", None),
    "rice": ("RICE", "ALL CLASSES", None, "LB / ACRE", None),
    "flaxseed": ("FLAXSEED", "ALL CLASSES", None, "BU / ACRE", 56),
}
KEY = ["crop", "adm_id", "year"]


def identify_crops(d):
    names = pd.Series("", index=d.index, dtype="string")
    for crop, (commodity, cls, util, _, _) in SPECS.items():
        mask = d.commodity_desc.str.strip().eq(commodity) & d.class_desc.str.strip().eq(cls)
        mask &= d.util_practice_desc.str.strip().eq(util or "ALL UTILIZATION PRACTICES")
        names.loc[mask] = crop
    return names


def county_ids(d):
    valid = (
        d.county_ansi.str.zfill(3).eq(d.county_code.str.zfill(3))
        & d.state_ansi.str.zfill(2).eq(d.state_fips_code.str.zfill(2))
        & pd.to_numeric(d.county_code, errors="coerce").between(1, 899)
    ).fillna(False)
    ids = "US-" + d.state_fips_code.str.zfill(2) + "-" + d.county_code.str.zfill(3)
    return ids, valid


def extract_census(raw=RAW, root=ROOT):
    """One streaming pass, retaining harvested-area rows before narrow filtering."""
    root.mkdir(parents=True, exist_ok=True)
    out = root / "census_harvested.parquet"
    if out.exists():
        return pd.read_parquet(out)
    columns = [
        "SOURCE_DESC", "COMMODITY_DESC", "CLASS_DESC", "UTIL_PRACTICE_DESC",
        "PRODN_PRACTICE_DESC", "STATISTICCAT_DESC", "UNIT_DESC", "DOMAIN_DESC",
        "DOMAINCAT_DESC", "AGG_LEVEL_DESC", "STATE_ANSI", "COUNTY_ANSI",
        "STATE_FIPS_CODE", "COUNTY_CODE", "YEAR", "VALUE", "SHORT_DESC",
    ]
    parts, counts = [], {}
    for d in pd.read_csv(raw, sep="\t", usecols=columns, dtype="string", chunksize=500_000):
        counts = (pd.Series(counts, dtype="int64").add(d.SOURCE_DESC.value_counts(), fill_value=0)
                  .astype("int64").to_dict())
        mask = (
            d.SOURCE_DESC.eq("CENSUS") & d.AGG_LEVEL_DESC.eq("COUNTY")
            & d.STATISTICCAT_DESC.eq("AREA HARVESTED") & d.UNIT_DESC.eq("ACRES")
        )
        if mask.any():
            p = d.loc[mask].copy()
            p.columns = p.columns.str.lower()
            parts.append(p)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[c.lower() for c in columns]
    )
    result.to_parquet(out, index=False)
    (root / "census_audit.json").write_text(json.dumps({
        "raw": str(raw), "source_counts": counts, "census_harvested_rows": len(result),
    }, indent=2))
    print(f"Census extraction: {counts}, harvested rows={len(result):,}", flush=True)
    return result


def interpolate_shares(observed, query):
    """County/crop linear interpolation; nearest endpoint outside census range."""
    values = []
    groups = {k: v.sort_values("year") for k, v in observed.groupby(["crop", "adm_id"])}
    # Loops over small county-crop time series, never weather county-days.
    for (crop, county), q in query.groupby(["crop", "adm_id"], sort=False):
        o = groups.get((crop, county))
        p = q[KEY].copy()
        if o is None:
            p["irrigated_share"] = np.nan
            p["irrigation_source"] = "unknown"
        else:
            p["irrigated_share"] = np.interp(q.year, o.year, o.irrigated_share)
            p["irrigation_source"] = "census_linear_nearest_endpoint"
        values.append(p)
    return pd.concat(values, ignore_index=True)


def census_shares(census):
    if census.empty:
        return pd.DataFrame(columns=KEY + ["irrigated_share"])
    d = census.copy()
    d["crop"] = identify_crops(d)
    d["adm_id"], valid = county_ids(d)
    d["year"] = pd.to_numeric(d.year)
    d["value"] = pd.to_numeric(d.value.str.replace(",", ""), errors="coerce")
    d = d.loc[valid & d.crop.ne("") & d.year.ge(1978) & d.value.ge(0)].copy()
    total = d.domain_desc.eq("TOTAL") & d.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")
    irrigated = d.prodn_practice_desc.eq("IRRIGATED") | (
        d.domain_desc.str.contains("IRRIGATION", case=False, na=False)
        & d.domaincat_desc.str.fullmatch("IRRIGATED|IRRIGATION STATUS: IRRIGATED", case=False, na=False)
    )
    tables = []
    for flag, name in ((total, "total"), (irrigated, "irrigated")):
        rows = d.loc[flag]
        if rows.duplicated(KEY).any():
            # Different census domains are not silently collapsed.
            totals = rows.groupby(KEY).value.agg(["min", "max"])
            if (totals["max"] - totals["min"]).abs().gt(1).any():
                raise ValueError(f"Ambiguous census {name} harvested acres; inspect census_harvested.parquet")
        tables.append(rows.groupby(KEY).value.first().rename(name))
    joined = pd.concat(tables, axis=1)
    joined["irrigated_share"] = joined.irrigated / joined.total.where(joined.total.gt(0))
    return joined.loc[joined.irrigated_share.between(0, 1)].reset_index()[KEY + ["irrigated_share"]]


def build_labels(root=ROOT, survey=SURVEY, raw=RAW):
    root.mkdir(parents=True, exist_ok=True)
    out = root / "labels.parquet"
    if out.exists():
        return pd.read_parquet(out)
    census = extract_census(raw, root)
    shares = census_shares(census)
    shares.to_parquet(root / "census_shares.parquet", index=False)
    parts = []
    columns = [
        "commodity_desc", "class_desc", "util_practice_desc", "prodn_practice_desc",
        "statisticcat_desc", "unit_desc", "domain_desc", "domaincat_desc", "year",
        "state_fips_code", "county_code", "state_ansi", "county_ansi", "state_alpha",
        "reference_period_desc", "value",
    ]
    dtype = {k: "string" for k in ("state_fips_code", "county_code", "state_ansi", "county_ansi")}
    inventory = []
    for d in pd.read_csv(survey, usecols=columns, dtype=dtype, chunksize=500_000):
        d["crop"] = identify_crops(d)
        d["adm_id"], valid = county_ids(d)
        mask = (
            valid & d.crop.ne("") & d.year.between(1981, 2025) & d.value.ge(0)
            & d.domain_desc.eq("TOTAL") & d.domaincat_desc.eq("NOT SPECIFIED")
            & d.reference_period_desc.eq("YEAR")
            & d.prodn_practice_desc.isin(["ALL PRODUCTION PRACTICES", "IRRIGATED", "NON-IRRIGATED"])
        )
        d = d.loc[mask].copy()
        inventory.append(d.groupby(["commodity_desc", "class_desc", "util_practice_desc", "unit_desc"]).size())
        valid_unit = d.unit_desc.eq(d.crop.map({c: s[3] for c, s in SPECS.items()}))
        mask = ((d.statisticcat_desc.eq("YIELD") & valid_unit & d.value.gt(0))
                | (d.statisticcat_desc.eq("AREA HARVESTED") & d.unit_desc.eq("ACRES")))
        parts.append(d.loc[mask])
    d = pd.concat(parts, ignore_index=True)
    pd.concat(inventory).groupby(level=[0, 1, 2, 3]).sum().to_csv(root / "series_inventory.csv")
    full_key = KEY + ["prodn_practice_desc"]
    if d.duplicated(full_key + ["statisticcat_desc"]).any():
        raise ValueError("Duplicate expanded survey series")
    area = d.loc[d.statisticcat_desc.eq("AREA HARVESTED"), full_key + ["value"]].rename(columns={"value": "harvested_acres"})
    y = d.loc[d.statisticcat_desc.eq("YIELD")].merge(area, on=full_key, how="left", validate="one_to_one")
    y = y.merge(interpolate_shares(shares, y[KEY].drop_duplicates()), on=KEY, how="left", validate="many_to_one")
    y.loc[y.prodn_practice_desc.eq("IRRIGATED"), ["irrigated_share", "irrigation_source"]] = [1., "survey_explicit_irrigated"]
    y.loc[y.prodn_practice_desc.eq("NON-IRRIGATED"), ["irrigated_share", "irrigation_source"]] = [0., "survey_explicit_nonirrigated"]
    factor = {}
    for crop, (_, _, _, unit, bushel_weight) in SPECS.items():
        pounds = bushel_weight if unit == "BU / ACRE" else {"TONS / ACRE": 2000, "LB / ACRE": 1, "CWT / ACRE": 100}[unit]
        factor[crop] = pounds * .45359237 / .40468564224 / 1000
    y["yield_t_ha"] = y.value * y.crop.map(factor)
    y["series_id"] = y.crop + "|" + y.prodn_practice_desc
    count = y.groupby(["series_id", "adm_id"]).year.transform("size")
    y = y.loc[count.ge(8)].copy()
    y["primary"] = y.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")
    y["irrigation_missing"] = y.irrigated_share.isna()
    y = y[KEY + ["series_id", "state_alpha", "yield_t_ha", "harvested_acres",
                 "prodn_practice_desc", "irrigated_share", "irrigation_source",
                 "irrigation_missing", "primary"]]
    y.to_parquet(out, index=False)
    (root / "label_audit.json").write_text(json.dumps({
        "rows": len(y), "rows_per_crop": y.groupby("crop").size().to_dict(),
        "irrigation_known_fraction": y.irrigated_share.notna().mean(),
        "area_known_fraction": y.harvested_acres.notna().mean(),
        "conversion_source": "https://www.ers.usda.gov/webdocs/publications/41880/33132_ah697_002.pdf",
        "census_share_rows": len(shares),
    }, indent=2))
    return y


if __name__ == "__main__":
    build_labels()
