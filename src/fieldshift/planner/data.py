"""Source adapters only: crop identifiers may be mapped here, never in ledgers."""

import csv
import gzip
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.data.power import county_frame, load_county_daily
from fieldshift.data.prism import load_prism
from fieldshift.sim.traits import CATALOG

ROOT = Path("data/processed/planner")
STATE_ALPHA = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "12": "FL",
               "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA",
               "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
               "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
               "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN",
               "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
MAPPING = {
    "CORN": ("corn", .06277), "SOYBEANS": ("soybean", .06725),
    "WHEAT": ("winter_wheat", .06725), "OATS": ("oats", .03587),
    "BARLEY": ("barley", .05380), "SORGHUM": ("sorghum", .06277),
    "HAY": ("hay", 2.2417), "COTTON": ("cotton", .00112085),
    "SUNFLOWER": ("sunflower", .00112085), "PEANUTS": ("peanuts", .00112085),
    "POTATOES": ("potatoes", .112085), "SUGARBEETS": ("sugarbeets", 2.2417),
    "BEANS": ("dry_beans", .112085), "FLAXSEED": ("flaxseed", .06277),
    "RICE": ("rice", .00112085), "RYE": ("rye", .06277),
}
CDL = {1: "corn", 5: "soybean", 24: "winter_wheat", 23: "spring_wheat",
       21: "barley", 28: "oats", 36: "alfalfa", 37: "hay", 4: "sorghum",
       6: "sunflower", 2: "cotton", 3: "rice", 43: "potatoes", 41: "sugarbeets"}


def epic_k(sand, silt, clay, organic_matter):
    """EPIC soil erodibility (Sharpley & Williams 1990), percent inputs; SI t ha h/(ha MJ mm).

    0.1317 converts US customary K to SI. Organic carbon = organic matter / 1.724.
    """
    carbon = organic_matter / 1.724
    sn1 = 1 - sand / 100
    k = ((.2 + .3 * np.exp(-.256 * sand * (1 - silt / 100)))
         * (silt / max(clay + silt, 1e-6)) ** .3
         * (1 - .25 * carbon / (carbon + np.exp(3.72 - 2.95 * carbon)))
         * (1 - .7 * sn1 / (sn1 + np.exp(-5.51 + 22.9 * sn1))))
    return float(k * .1317)


def build_index():
    """One streaming scan, ~200 MB compressed. Keep source-identifying columns."""
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "nass_index_v2.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pieces = []
    source = Path("data/processed/nass_county_crops.csv.gz")
    for d in pd.read_csv(source, chunksize=200_000, dtype={
        "state_ansi": "string", "county_ansi": "string",
    }, low_memory=False):
        keep = (d.commodity_desc.isin(MAPPING) & d.statisticcat_desc.isin(["YIELD", "AREA HARVESTED"])
                & d.reference_period_desc.eq("YEAR") & d.domain_desc.eq("TOTAL")
                & d.prodn_practice_desc.isin(["ALL PRODUCTION PRACTICES", "IRRIGATED", "NON-IRRIGATED"])
                & (d.year.ge(2016) | (d.statisticcat_desc.eq("YIELD") & d.commodity_desc.eq("CORN"))))
        d = d.loc[keep].copy()
        d = d.loc[d.state_ansi.notna() & d.county_ansi.notna()]
        d["adm_id"] = "US-" + d.state_ansi.str.zfill(2) + "-" + d.county_ansi.str.zfill(3)
        pieces.append(d)
    result = pd.concat(pieces, ignore_index=True)
    result.to_parquet(path, index=False)
    return result


@lru_cache(maxsize=1)
def practice_yields():
    """Full historical practice yields; the legacy area/normal index starts in 2016."""
    path = ROOT / "nass_practice_yields_v1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    pieces = []
    for d in pd.read_csv("data/processed/nass_county_crops.csv.gz", chunksize=200_000,
                         dtype={"state_ansi": "string", "county_ansi": "string"}, low_memory=False,
                         usecols=["commodity_desc", "class_desc", "statisticcat_desc", "util_practice_desc",
                                  "prodn_practice_desc", "unit_desc", "state_ansi", "county_ansi", "year",
                                  "value", "reference_period_desc", "domain_desc"]):
        keep = (d.commodity_desc.isin(MAPPING) & d.statisticcat_desc.eq("YIELD") & d.value.gt(0)
                & d.reference_period_desc.eq("YEAR") & d.domain_desc.eq("TOTAL")
                & d.prodn_practice_desc.isin(["IRRIGATED", "NON-IRRIGATED", "ALL PRODUCTION PRACTICES"])
                & d.state_ansi.notna() & d.county_ansi.notna() & d.year.ge(1981))
        d = d.loc[keep].copy()
        d["adm_id"] = "US-" + d.state_ansi.str.zfill(2) + "-" + d.county_ansi.str.zfill(3)
        pieces.append(d)
    result = pd.concat(pieces, ignore_index=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


@lru_cache(maxsize=64)
def practice_normals(practice):
    d, pieces, dates = practice_yields(), [], {}
    d = d.loc[d.adm_id.isin(_static().index)]
    for commodity, klass, key, factor in SERIES:
        x = _crop_rows(d, commodity, klass)
        expected = {"COTTON": "LB / ACRE", "HAY": "TONS / ACRE"}.get(commodity)
        if expected:
            x = x.loc[x.unit_desc.eq(expected)]
        x = x.loc[x.prodn_practice_desc.eq(practice)].drop_duplicates(["adm_id", "year"]).sort_values("year")
        recent = x.groupby("adm_id").tail(10)
        pieces.append((recent.groupby("adm_id").value.mean() * factor).rename(key))
        dates[key] = recent.groupby("adm_id").year.agg(["min", "max", "count"])
    return pd.concat(pieces, axis=1), dates


def practice_normal(adm_id, key, practice):
    """County, then neighbouring counties, then same-state county means; same crop/practice only."""
    table, dates = practice_normals(practice)
    if key not in table:
        return None
    values = table[key].dropna()
    if adm_id in values:
        row = dates[key].loc[adm_id]
        return float(values[adm_id]), (f"NASS county {practice}, {int(row['count'])} reported years "
                                      f"{int(row['min'])}–{int(row['max'])}; direct mean, t/ha; "
                                      "https://quickstats.nass.usda.gov/; local nass_county_crops.csv.gz"
                                      + ("; LOW: old or fewer than 3 reports" if row["max"] < 2016 or row["count"] < 3 else ""))
    static = _static()
    if adm_id in static.index:
        cols = ["centroid_lat", "centroid_lon"]
        here = np.radians(static.loc[adm_id, cols].to_numpy(float))
        other = np.radians(static.reindex(values.index)[cols].to_numpy(float))
        distance = 6371 * 2 * np.arcsin(np.sqrt(np.sin((other[:, 0] - here[0]) / 2)**2
                    + np.cos(here[0]) * np.cos(other[:, 0]) * np.sin((other[:, 1] - here[1]) / 2)**2))
        near = values[(distance < 100) & values.index.to_series().ne(adm_id)]
        if len(near):
            return float(near.median()), f"NASS {practice}, median of {len(near)} county normals within 100 km; LOW confidence transfer"
    official = state_normal(adm_id[3:5], key, practice)
    if official is not None:
        return official
    state = values.loc[values.index.str[:6] == adm_id[:6]]
    if len(state):
        return float(state.mean()), f"NASS {practice}, mean of {len(state)} same-state county normals; LOW confidence transfer"
    return None


@lru_cache(maxsize=1)
def state_yields():
    """Official STATE rows are absent from the county-only processed extract."""
    path = ROOT / "nass_state_yields_v3.parquet"
    if path.exists():
        return pd.read_parquet(path)
    fields = ["commodity_desc", "class_desc", "statisticcat_desc", "util_practice_desc",
              "prodn_practice_desc", "unit_desc", "state_ansi", "year", "value",
              "reference_period_desc", "domain_desc", "source_desc"]
    archives = sorted(Path("data/raw/nass").glob("qs.crops_*.txt.gz"))
    if not archives:
        return pd.DataFrame(columns=fields)
    records = []
    with gzip.open(archives[-1], "rt", newline="") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        columns = {k: header.index(k.upper()) for k in fields}
        for line in stream:
            if "\tSTATE\t" not in line or not any(f"\t{s}\t" in line for s in ("YIELD","AREA HARVESTED","PRODUCTION")):
                continue
            row = next(csv.reader([line], delimiter="\t"))
            r = {k: row[i] for k, i in columns.items()}
            survey = (r["source_desc"] == "SURVEY" and r["statisticcat_desc"] == "YIELD" and r["prodn_practice_desc"] in
                      ("IRRIGATED", "NON-IRRIGATED", "ALL PRODUCTION PRACTICES"))
            census = (r["source_desc"] == "CENSUS" and r["commodity_desc"] == "HAY"
                      and r["class_desc"] == "ALFALFA" and r["prodn_practice_desc"] in
                      ("IRRIGATED, ENTIRE CROP", "IRRIGATED, NONE OF CROP", "ALL PRODUCTION PRACTICES"))
            if (r["commodity_desc"] in MAPPING and (survey or census)
                    and r["reference_period_desc"] == "YEAR" and r["domain_desc"] == "TOTAL"
                    and r["state_ansi"] and int(r["year"]) >= 1981
                    and row[header.index("AGG_LEVEL_DESC")] == "STATE"):
                records.append(r)
    result = pd.DataFrame(records, columns=fields)
    result["value"] = pd.to_numeric(result.value.str.replace(",", ""), errors="coerce")
    result["year"] = pd.to_numeric(result.year)
    result = result.loc[result.value.gt(0)]
    ROOT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def alfalfa_normal(state, irrigated):
    """Recent official practice-specific state observations, never a 1980s county proxy.

    Census ENTIRE/NONE crop cohorts are NOT a yield attributed to the irrigated
    portion of a partly-irrigated crop. Suppressed values are never reconstructed.
    """
    practice = "IRRIGATED" if irrigated else "NON-IRRIGATED"
    raw = state_yields()
    raw = raw.loc[raw.state_ansi.eq(state) & raw.commodity_desc.eq("HAY")
                  & raw.class_desc.eq("ALFALFA") & raw.year.between(2016, 2025)]
    d = raw.loc[raw.statisticcat_desc.eq("YIELD") & raw.unit_desc.eq("TONS / ACRE")]
    survey = d.loc[d.source_desc.eq("SURVEY") & d.prodn_practice_desc.eq(practice)]
    census_practice = "IRRIGATED, ENTIRE CROP" if irrigated else "IRRIGATED, NONE OF CROP"
    census = d.loc[d.source_desc.eq("CENSUS") & d.prodn_practice_desc.eq(census_practice)]
    # A newer Census with a suppressed practice yield is not evidence for
    # transplanting an older tiny irrigated cohort into an entire county.
    totals = raw.loc[raw.source_desc.eq("CENSUS") & raw.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")]
    latest = totals.year.max()
    if pd.notna(latest) and (not len(census) or census.year.max() < latest) and not len(survey):
        p = totals.loc[totals.year.eq(latest) & totals.statisticcat_desc.eq("PRODUCTION") & totals.unit_desc.eq("TONS")]
        a = totals.loc[totals.year.eq(latest) & totals.statisticcat_desc.eq("AREA HARVESTED") & totals.unit_desc.eq("ACRES")]
        if len(p) == 1 and len(a) == 1:
            return float(p.value.iloc[0]/a.value.iloc[0]*2.2417), (
                f"NASS official STATE {state} CENSUS {int(latest)} ALL PRODUCTION PRACTICES: "
                f"{p.value.iloc[0]:g} short tons / {a.value.iloc[0]:g} harvested acres ×2.2417→t/ha; "
                f"{practice} yield unavailable/suppressed in the latest Census. "
                "LOW all-practice/spatial/small-market proxy; as-reported hay, moisture/grade unspecified; "
                "https://quickstats.nass.usda.gov/")
    x = survey.sort_values("year").tail(10)
    if len(census) and (not len(x) or census.year.max() > x.year.max()):
        x = census.sort_values("year").tail(1)
    if not len(x):
        x = d.loc[d.source_desc.eq("SURVEY")
                  & d.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")].sort_values("year").tail(10)
    if not len(x):
        return None
    desc = x.iloc[-1].prodn_practice_desc
    return float(x.value.mean()*2.2417), (
        f"NASS official STATE {state}, {x.iloc[-1].source_desc}, {desc}, "
        f"{len(x)} reports {int(x.year.min())}–{int(x.year.max())}; "
        "TONS/ACRE short tons of harvested hay ×2.2417→t/ha. As-reported baled-hay mass, "
        "not oven-dry biomass; NASS does not supply sample moisture. "
        "LOW single-census-year/spatial/practice transfer; "
        "https://quickstats.nass.usda.gov/; raw Quick Stats crops bulk.")


@lru_cache(maxsize=4096)
def state_normal(state, key, practice):
    d = state_yields()
    d = d.loc[d.statisticcat_desc.eq("YIELD")]
    for commodity, klass, crop, factor in SERIES:
        if crop != key:
            continue
        x = _crop_rows(d, commodity, klass)
        x = x.loc[x.state_ansi.eq(state) & x.prodn_practice_desc.eq(practice)]
        x = x.drop_duplicates("year").sort_values("year").tail(10)
        if len(x):
            lo, hi = int(x.year.min()), int(x.year.max())
            return float(x.value.mean()*factor), (f"NASS official STATE {state} {practice}, {len(x)} reported years "
                                                  f"{lo}–{hi}; https://quickstats.nass.usda.gov/; "
                                                  "local raw Quick Stats crops bulk; LOW spatial/age transfer")
    return None


def county_options():
    _, meta = load_county_daily()
    path = ROOT / "nass_index_v2.parquet"
    names = {}
    if path.exists():
        d = pd.read_parquet(path, columns=["adm_id", "county_name", "state_alpha"]).drop_duplicates("adm_id")
        names = {r.adm_id: f"{r.county_name.title()} County, {r.state_alpha}" for r in d.itertuples()}
    return {c: names.get(c, c) + f" · {c}" for c in meta["counties"]}


# NASS series per planner crop: commodity, class, key, (bu|ton|cwt|lb)/acre -> t/ha.
SERIES = [(commodity, {"WHEAT": "WINTER", "HAY": "(EXCL ALFALFA)", "COTTON": "UPLAND"}.get(commodity, "ALL CLASSES"), key, factor)
          for commodity, (key, factor) in MAPPING.items()] + [("HAY", "ALFALFA", "alfalfa", 2.2417)]


def _crop_rows(d, commodity, klass=None):
    x = d.loc[d.commodity_desc.eq(commodity)].copy()
    unit = {"HAY": "TONS / ACRE", "SUGARBEETS": "TONS / ACRE",
            "COTTON": "LB / ACRE", "SUNFLOWER": "LB / ACRE", "PEANUTS": "LB / ACRE",
            "RICE": "LB / ACRE", "POTATOES": "CWT / ACRE", "BEANS": "CWT / ACRE"}.get(commodity, "BU / ACRE")
    if "unit_desc" in x:
        x = x.loc[x.unit_desc.isin([unit, "ACRES"])]
    if commodity == "CORN":
        x = x.loc[x.util_practice_desc.eq("GRAIN") & x.unit_desc.isin(["BU / ACRE", "ACRES"])]
    klass = klass or {"WHEAT": "WINTER", "HAY": "(EXCL ALFALFA)", "COTTON": "UPLAND"}.get(commodity, "ALL CLASSES")
    return x.loc[x.class_desc.eq(klass)]


@lru_cache(maxsize=1)
def _recent_normals():
    """All-practice recent (last 10 reported years) county yield normal per crop, t/ha."""
    d = build_index()
    d = d.loc[d.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES") & d.statisticcat_desc.eq("YIELD") & d.value.gt(0)]
    pieces = []
    for commodity, klass, key, factor in SERIES:
        x = _crop_rows(d, commodity, klass).drop_duplicates(["adm_id", "year"]).sort_values("year")
        normal = x.groupby("adm_id").tail(10).groupby("adm_id").value.mean() * factor
        pieces.append(normal.rename(key))
    return pd.concat(pieces, axis=1)


@lru_cache(maxsize=1)
def _static():
    path = Path("data/processed/static/county_static.parquet")
    if not path.exists():
        return pd.DataFrame()
    static = pd.read_parquet(path)
    if "adm_id" not in static and static.index.name == "adm_id":
        static = static.reset_index()
    return static.set_index("adm_id")


def fallback_normal(adm_id, key, radius_km=100, minimum=3):
    """County normal from neighbouring counties (centroids within ``radius_km``),
    else the state's county mean; None if neither exists."""
    table, static = _recent_normals(), _static()
    if key not in table:
        return None
    values = table[key].dropna()
    if adm_id in static.index and {"centroid_lat", "centroid_lon"} <= set(static.columns):
        here = static.loc[adm_id, ["centroid_lat", "centroid_lon"]].astype(float)
        if here.notna().all():
            other = static.reindex(values.index)[["centroid_lat", "centroid_lon"]].astype(float)
            lat0, lon0 = np.radians(here.iloc[0]), np.radians(here.iloc[1])
            lat, lon = np.radians(other.centroid_lat), np.radians(other.centroid_lon)
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat - lat0) / 2) ** 2
                                                    + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2))
            near = values[distance.lt(radius_km) & values.index.to_series().ne(adm_id)]
            if len(near) >= minimum:
                return float(near.median()), f"median of {len(near)} counties within {radius_km} km (NASS, recent 10 years)"
    state = values[values.index.str[:6] == adm_id[:6]]
    if len(state):
        return float(state.mean()), f"mean of {len(state)} counties in the state (NASS, recent 10 years)"
    return None


def county_soil(adm_id, gaps):
    """County soil and terrain scenario; missing values are listed in ``gaps``."""
    soil = {"awc_mm_per_m": 150., "bulk_density_kg_m3": 1300.,
            "organic_matter_pct": 3., "slope_pct": 2., "slope_length_m": 100.,
            "rusle_k": .032, "profile_depth_m": 2.0}
    gaps.extend([
        "Slope length 100 m and the annual-rain R proxy are unmeasured RUSLE scenarios, not a field erosion estimate.",
        "Root-zone profile 2.0 m with 0–100 cm water-holding capacity extended below 1 m; no capillary rise from a water table.",
        ("Each trial spins up one full rotation cycle before the scored cycle. Plant C inputs follow Bolinder 2007; "
         "humification 0.15 (residue) / 0.35 (roots, Kätterer 2011) and 2%/year decay of one slow pool are "
         "low-confidence simplifications."),
        ("Prices are 2020–2025 USDA NASS state/national averages where matched, otherwise reference scenarios. "
         "Corn, soybean and alfalfa costs are ISU 2025 Iowa budgets (machinery, inputs, labor; land excluded); "
         "other crops' costs are low-confidence scenarios. Irrigation energy and new equipment are not costed."),
        "Heat response 2% per flowering degree-day and frost response 5% per reproductive frost day are uncalibrated priors.",
        "Generic family pest carry-over and decay are low confidence; inspect local pest history.",
    ])
    static = _static()
    if len(static):
        if adm_id in static.index:
            raw = static.loc[adm_id].to_dict()
            aliases = {
                "awc_mm_per_m": [("awc_0_100", 1), ("awc", 10)],
                "bulk_density_kg_m3": [("bulk_density_0_30", 1000), ("bulk_density", 1000)],
                "organic_matter_pct": [("organic_matter_0_30", 1), ("om", 1), ("organic_matter", 1)],
                "slope_pct": [("slope_percent", 1), ("slope_pct", 1)],
            }
            for key, possibilities in aliases.items():
                for field, scale in possibilities:
                    if field in raw and pd.notna(raw[field]):
                        soil[key] = float(raw[field]) * scale
                        break
                else:
                    gaps.append(f"{key}: default {soil[key]}; field measurement missing.")
            texture = [raw.get(k) for k in ("sand_0_30", "silt_0_30", "clay_0_30", "organic_matter_0_30")]
            if all(pd.notna(v) for v in texture):
                soil["rusle_k"] = epic_k(*texture)
            else:
                gaps.append("Soil texture missing: RUSLE K 0.032 SI (medium-textured soil scenario).")
        else:
            gaps.append("No county static row; FAO-56 representative loam defaults used.")
    else:
        gaps.append("Static county soil absent: FAO-56 loam 150 mm/m, 1.3 Mg/m³, OM 3%, slope 2% assumed.")
    soil["awc_mm_per_m"] = float(np.clip(soil["awc_mm_per_m"], 40, 300))
    return soil


@lru_cache(maxsize=1)
def _progress():
    path = Path("data/processed/nass_crop_progress_dates.parquet")
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def state_planting(state):
    """USDA state 50%-planted and maturity dates per crop key; also the state's progress rows."""
    progress = _progress()
    if len(progress):
        progress = progress.loc[progress.state.eq(state)]
    planting = {}
    if len(progress):
        planted = progress.loc[progress.stage.eq("planted") & progress.date_50.notna()]
        for commodity, (key, _) in MAPPING.items():
            rows = planted.loc[planted.commodity.eq(commodity)]
            rows = rows.loc[rows.class_desc.eq("WINTER" if commodity == "WHEAT" else "ALL CLASSES")]
            if commodity == "HAY" or not len(rows):
                continue
            dates = pd.to_datetime(rows.date_50)
            planting[key] = {"by_year": {int(y): pd.Timestamp(v) for y, v in zip(rows.year, dates, strict=True)},
                             "median_doy": int(dates.dt.dayofyear.median()),
                             "source": f"USDA NASS {state} crop progress, 50% planted"}
    maturing = {}
    if len(progress):
        # Physiological-maturity proxies reported by NASS crop progress.
        for commodity, stage in (("CORN", "mature"), ("SOYBEANS", "dropping leaves")):
            rows = progress.loc[progress.commodity.eq(commodity) & progress.stage.eq(stage)
                                & progress.class_desc.eq("ALL CLASSES") & progress.date_50.notna()]
            if len(rows):
                dates = pd.to_datetime(rows.date_50)
                maturing[MAPPING[commodity][0]] = {
                    "by_year": {int(y): pd.Timestamp(v) for y, v in zip(rows.year, dates, strict=True)},
                    "median_doy": int(dates.dt.dayofyear.median()),
                    "source": f"USDA NASS {state} crop progress, 50% {stage}"}
    for key, record in planting.items():
        if key in maturing:
            record["maturity"] = maturing[key]
    if len(progress):
        # Harvest operations (grain dry-down follows maturity): cover crops are sown after these.
        harvested = progress.loc[progress.stage.eq("harvested") & progress.date_50.notna()]
        for commodity, (key, _) in MAPPING.items():
            rows = harvested.loc[harvested.commodity.eq(commodity)
                                 & harvested.class_desc.eq("WINTER" if commodity == "WHEAT" else "ALL CLASSES")]
            if "GRAIN" in set(rows.util_practice):
                rows = rows.loc[rows.util_practice.eq("GRAIN")]
            rows = rows.drop_duplicates("year")
            if key in planting and len(rows):
                dates = pd.to_datetime(rows.date_50)
                planting[key]["harvested"] = {
                    "by_year": {int(y): pd.Timestamp(v) for y, v in zip(rows.year, dates, strict=True)},
                    "median_doy": int(dates.dt.dayofyear.median()),
                    "source": f"USDA NASS {state} crop progress, 50% harvested"}
    return planting, progress


PRISM_RULE = (
    "POWER daily rain is rescaled inside each PRISM 8-day step so the step total equals PRISM ppt_sum; "
    "POWER's wet-day pattern within the step is kept. Where POWER has <0.1 mm in a step and PRISM has rain, "
    "PRISM's total is split in equal shares over round(wet_days) evenly spaced days of the step (all step "
    "days if wet_days is 0). That placement is an assumption with no better information, not an observation. "
    "PRISM days start 12:00 UTC; POWER days are local; the half-day offset is ignored. ET0, temperature and "
    "humidity stay POWER.")


@lru_cache(maxsize=1)
def _prism():
    return load_prism()


def prism_scaled_rain(frames, ids, device):
    """PRISM-corrected POWER daily rain [sites, days] on the accelerator."""
    cube, meta = _prism()
    names = [v["name"] for v in meta["variables"]]
    rows = [meta["counties"].index(a) for a in ids]
    block = torch.tensor(np.asarray(cube[rows][:, :, [names.index("ppt_sum"), names.index("wet_days")]]),
                         device=device)
    starts = pd.DatetimeIndex(meta["step_start_dates"])
    dates = pd.DatetimeIndex(frames[0]["date"])
    steps = len(starts)
    step = torch.as_tensor(np.searchsorted(starts.values, dates.values, side="right") - 1, device=device)
    rain = torch.tensor(np.stack([f["prec"].to_numpy("float32") for f in frames]), device=device)
    s, t = rain.shape
    day = torch.arange(t, device=device)
    power = torch.zeros((s, steps), device=device).index_add_(1, step, rain)
    length = torch.zeros(steps, device=device).index_add_(0, step, torch.ones(t, device=device))
    first = torch.full((steps,), t, device=device, dtype=torch.long).scatter_reduce(0, step, day, "amin")
    offset = (day - first[step]).float()
    prism, wet = block[..., 0], block[..., 1]
    k = torch.where(wet >= .5, wet.round(), length[None]).clamp(max=length[None])
    k_day, len_day = k[:, step], length[step][None]
    chosen = torch.remainder(offset[None] * k_day, len_day) < k_day
    count = torch.zeros((s, steps), device=device).index_add_(1, step, chosen.float())
    spread = prism[:, step] * chosen / count[:, step].clamp_min(1)
    scaled = rain * (prism / power.clamp_min(1e-6))[:, step]
    out = torch.where((power < .1)[:, step], spread, scaled)
    check = torch.zeros((s, steps), device=device).index_add_(1, step, out)
    error = float((check - prism).abs().max())
    if error > .05:
        raise ValueError(f"PRISM rescaling does not conserve step totals (max error {error:.3f} mm)")
    return out, {"max_step_error_mm": error,
                 "dry_power_wet_prism_steps": int(((power < .1) & (prism > 0)).sum())}



def with_prism_rain(daily, adm_id):
    """Adopted 2026-11-14 (validation_weather_tests.json): bucket rain = POWER
    rescaled to PRISM 8-day totals. ``prec_power`` keeps the original POWER rain,
    which the learned yield model was trained on."""
    from fieldshift.sim.components import accelerator
    scaled, _ = prism_scaled_rain([daily], [adm_id], accelerator())
    out = daily.copy()
    out["prec_power"] = out["prec"]
    out["prec"] = scaled[0].cpu().numpy()  # handoff into the pandas daily frame
    return out


def load_light(adm_id, rain="prism"):
    """Weather, soil and state planting dates only (multi-county validation)."""
    cube, meta = load_county_daily()
    daily = county_frame(cube, meta, adm_id)
    if rain == "prism":
        daily = with_prism_rain(daily, adm_id)
    gaps = []
    soil = county_soil(adm_id, gaps)
    state = STATE_ALPHA.get(adm_id[3:5])
    planting, _ = state_planting(state)
    return {"adm_id": adm_id, "daily": daily, "soil": soil, "planting": planting, "gaps": gaps, "state": state}


def load_site(adm_id, irrigated=None):
    """irrigated=None uses all-practice county normals; True/False prefer NASS
    IRRIGATED / NON-IRRIGATED county yields when at least 3 recent years exist."""
    cube, meta = load_county_daily()
    daily = with_prism_rain(county_frame(cube, meta, adm_id), adm_id)
    daily["adm_id"] = adm_id
    # Required features are never silently imputed.
    if not np.isfinite(daily[["tmin", "tmax", "tavg", "prec", "et0", "vpd", "rad"]]).all().all():
        raise ValueError(f"{adm_id}: nonfinite daily weather; repair source archive before simulation")
    d = build_index()
    local = d.loc[d.adm_id.eq(adm_id)]
    normals, areas, observed = {}, {}, {}
    gaps = []
    practice = {None: None, True: "IRRIGATED", False: "NON-IRRIGATED"}[irrigated]
    practice_gaps, practice_ratios = [], {}
    for commodity, klass, key, factor in SERIES:
        rows = _crop_rows(local, commodity, klass)
        x = rows.loc[rows.prodn_practice_desc.eq("ALL PRODUCTION PRACTICES")]
        y = x.loc[x.statisticcat_desc.eq("YIELD") & x.value.gt(0)].drop_duplicates("year").sort_values("year")
        if len(y):
            normals[key] = float(y.tail(10).value.mean() * factor)
            observed[key] = {int(r.year): float(r.value * factor) for r in y.itertuples()}
        if practice:
            # Practice series often stop years ago; transfer their ratio to the
            # all-practice series over overlapping years onto the recent normal.
            p = rows.loc[rows.prodn_practice_desc.eq(practice) & rows.statisticcat_desc.eq("YIELD")
                         & rows.value.gt(0)].drop_duplicates("year").set_index("year").value
            overlap = p.index.intersection(y.year)
            if len(overlap) >= 3 and key in normals:
                recent = sorted(overlap)[-10:]
                ratio = float((p.loc[recent] / y.set_index("year").value.loc[recent]).mean())
                normals[key] *= ratio
                practice_ratios[key] = (ratio, int(min(recent)), int(max(recent)))
            elif len(y):
                practice_gaps.append(key)
        a = x.loc[x.statisticcat_desc.eq("AREA HARVESTED") & x.value.gt(0)]
        if len(a):
            areas[key] = float(a.loc[a.year.ge(a.year.max() - 4)].value.mean())
    normal_sources = {k: "county NASS yield, mean of the 10 most recent years" for k in normals}
    for _, _, key, _ in SERIES:
        if key not in normals:
            found = fallback_normal(adm_id, key)
            if found:
                normals[key], normal_sources[key] = found[0], found[1] + "; no county series"
    for key, (ratio, first, last) in practice_ratios.items():
        gaps.append(f"{key} {practice.lower()} normal = recent all-practice normal × {ratio:.2f} "
                    f"(NASS {practice.lower()}/all ratio, {first}–{last}); assumes the ratio still holds.")
    # Transferring one crop's practice ratio to another was tested and rejected:
    # in Finney County KS the corn ratio (0.31) made mostly-dryland wheat 14 bu/acre.
    if practice_gaps and practice_ratios:
        gaps.append(f"No {practice.lower()} county yield series for {', '.join(practice_gaps)} although this county "
                    "has irrigated production: their all-practice normals mix irrigated and rainfed fields, so "
                    "margins involving them are LOW confidence.")
    elif practice_gaps:
        gaps.append("NASS publishes no irrigated/non-irrigated yield split for this county (usually a sign one "
                    "practice dominates); all-practice county normals are used.")
    if practice:
        for _, _, key, _ in SERIES:
            found = practice_normal(adm_id, key, practice)
            if found is None and irrigated:
                # Recover discontinued all-practice alfalfa county series too;
                # do not replace a published practice series with a generic trait.
                found = practice_normal(adm_id, key, "ALL PRODUCTION PRACTICES")
            if found:
                normals[key], normal_sources[key] = found
        gaps = [g for g in gaps if "normal = recent all-practice" not in g
                and "NASS publishes no irrigated/non-irrigated" not in g
                and "their all-practice normals mix" not in g]
        gaps.append(f"Yield hierarchy: county {practice}, neighbours within 100 km, same-state county means, "
                     "then all-practice. Direct historical means, no cross-crop irrigation ratios; old series and transfers are LOW confidence.")
    found = alfalfa_normal(adm_id[3:5], bool(irrigated))
    if found:
        normals["alfalfa"], normal_sources["alfalfa"] = found
    gaps.append("Alfalfa: recent practice-specific STATE survey/Census takes precedence over discontinued county "
                "series. Census yields can reflect a single drought year. Hay prices and yields are as-reported "
                "baled-hay mass, not oven-dry. Carbon uses an explicit 88% dry-matter convention; actual moisture is unknown.")
    points = Path("data/processed/qdann_points.csv.gz")
    existing = ()
    evidence = ""
    if points.exists():
        columns = ["county_fips"] + [f"cdl_{y}" for y in range(2019, 2023)]
        p = pd.read_csv(points, usecols=columns, dtype={"county_fips": "string"})
        p = p.loc[p.county_fips.str.zfill(5).eq(adm_id[3:5] + adm_id[6:9])]
        sequences = [tuple(CDL.get(int(v), "") for v in row)
                     for row in p[[f"cdl_{y}" for y in range(2019, 2023)]].itertuples(index=False, name=None)]
        counts = Counter(s for s in sequences if all(s) and all(k in CATALOG for k in s))
        if counts:
            existing, count = counts.most_common(1)[0]
            evidence = (f"Most common fully mapped 2019–2022 USDA CDL sequence among sampled county points: "
                        f"{count}/{len(p)} points. County proxy, not this farmer's field history.")
    if not existing:
        ordered = sorted(areas, key=areas.get, reverse=True)
        if not ordered:
            ordered = ["hay", "fallow"]
            gaps.append("No published county crop-area normal: forage/fallow reference baseline is not an observed rotation.")
        existing = tuple((ordered[:2] * 4)[:4])
        evidence = ("Sequence inferred by alternating the two largest supported NASS harvested-area crops; "
                    "actual order is unknown. Only supported crop classes are included (e.g. upland cotton, grain corn, "
                    "winter wheat; not orchards or corn silage).")
    soil = county_soil(adm_id, gaps)
    states = d.drop_duplicates("state_ansi").set_index("state_ansi").state_alpha
    state = states.get(adm_id[3:5])
    planting, progress = state_planting(state)
    if not planting:
        gaps.append("No state crop-progress planting dates; temperature sowing rule used (known to run 2–5 weeks early).")
    eo = None
    eo_path = Path("data/processed/eo/eo_county_8day.f32")
    if eo_path.exists():
        em = json.loads(eo_path.with_suffix(".json").read_text())
        if adm_id in em.get("counties", []):
            arr = np.memmap(eo_path, dtype="<f4", mode="r", shape=tuple(em["shape"]))
            names = [v["name"] if isinstance(v, dict) else v for v in em["variables"]]
            eo = pd.DataFrame(arr[em["counties"].index(adm_id)], columns=names)
            eo["date"] = pd.to_datetime(em["step_start_dates"])
    if eo is None:
        gaps.append("Observed NDVI bare-ground periods unavailable; calendar-derived uncovered days only.")
    loss_path = Path("data/processed/rma_cause_of_loss.parquet")
    loss_evidence, loss_months = [], pd.DataFrame()
    if loss_path.exists():
        losses = pd.read_parquet(loss_path, filters=[("adm_id", "==", adm_id), ("year", ">=", 2000)],
                                 columns=["commodity", "year", "month", "cause_group", "indemnity"])
        losses = losses.loc[losses.month.between(1, 12) & losses.indemnity.gt(0)]
        if len(losses):
            loss_months = losses.groupby(["commodity", "month", "cause_group"], as_index=False).indemnity.sum()
            loss_months["month"] = loss_months.month.astype(int)
            total = loss_months.indemnity.sum()
            for row in loss_months.nlargest(3, "indemnity").itertuples():
                loss_evidence.append(
                    f"USDA RMA insured losses 2000+: {row.commodity.title()}, month {row.month}, "
                    f"{row.cause_group} = {row.indemnity / total:.0%} of county indemnity. "
                    "Insured claims indicate reported vulnerable seasons, not field-level loss probability.")
        else:
            gaps.append("No county insured-loss records since 2000; vulnerable windows are weather-derived only.")
    else:
        gaps.append("Insured monthly cause-of-loss records absent: vulnerable windows are weather-derived, not reported losses.")
    prices = {}
    price_path = ROOT / "nass_reference_prices.json"
    if price_path.exists():
        records = json.loads(price_path.read_text())["records"]
        for commodity, (key, yield_factor) in MAPPING.items():
            target_class = {"WHEAT": "WINTER", "COTTON": "UPLAND"}.get(commodity, "ALL CLASSES")
            matches = [r for r in records if r["commodity"] == commodity
                       and r["class_desc"] == target_class and r["state"] in {state, "US"}
                       and (commodity != "CORN" or r["util_practice"] == "GRAIN")
                       and len(r["years"]) >= 3
                       and r["unit"] in {"$ / BU", "$ / TON", "$ / CWT", "$ / LB"}]
            if matches:
                match = min(matches, key=lambda r: r["state"] != state)
                tonnes_per_unit = {"$ / TON": .90718474, "$ / CWT": .045359237,
                                   "$ / LB": .00045359237}.get(match["unit"], yield_factor / 2.47105)
                prices[key] = {"value": match["mean"] / tonnes_per_unit,
                               "source": f"USDA NASS raw archive annual prices, {match['state']}, {match['years']}; {match['unit']}",
                               "confidence": "medium"}
        matches = [r for r in records if r["commodity"] == "HAY" and r["class_desc"] == "ALFALFA"
                   and r["state"] in {state, "US"} and r["unit"] == "$ / TON"]
        if matches:
            match = min(matches, key=lambda r: r["state"] != state)
            prices["alfalfa"] = {"value": match["mean"] / .90718474,
                                  "source": f"USDA NASS annual ALFALFA hay prices, {match['state']}, {match['years']}; "
                                  "$/short ton ÷0.90718474→$/t as-sold hay; grade unspecified "
                                  "(NASS annual PRICE RECEIVED is not grade-specific); thin market, buyer/quality unverified",
                                  "confidence": "medium" if match["state"] == state else "low"}
    if irrigated:
        from fieldshift.sim.economics import capacity
        cap = capacity(adm_id)
        if cap["value"] is not None:
            soil["irrigation_capacity_mm_yr"] = cap["value"]
        gaps.append("Irrigation availability: " + cap["source"] + ". Bucket mm are net application; distribution losses unmodelled.")
        gaps = [g for g in gaps if "Irrigation energy and new equipment are not costed" not in g]
    return {"adm_id": adm_id, "daily": daily, "soil": soil, "normals": normals,
            "areas": areas, "observed": observed, "existing": existing,
            "rotation_evidence": evidence, "gaps": gaps, "progress": progress, "eo": eo,
            "label": county_options().get(adm_id, adm_id), "prices": prices,
            "planting": planting, "loss_evidence": loss_evidence, "loss_months": loss_months,
             "mixed_practice_normals": [k for k in normals if practice and practice not in normal_sources[k]],
            "normal_sources": normal_sources}
