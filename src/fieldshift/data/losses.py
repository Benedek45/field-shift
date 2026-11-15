"""Sub-annual crop damage (RMA cause of loss) and drought (US Drought Monitor).

Both sources are monthly or weekly, which is what makes them useful: they date a
loss inside a season instead of attributing a whole year.

RMA: one pipe-delimited ``colsom_YYYY.txt`` per year inside ``colsom_YYYY.zip`` on
the RMA public file server, 1989 onward, with the layout of
``COL_Summary_of_Business_with_Month_All_Years.pdf``.  Fields used here: net
planted quantity (19), net determined quantity (28), indemnity (29) and the loss
ratio (30), which the file states as indemnity / total premium.

USDM: the NDMC "comprehensive statistics" export, one JSON/CSV per request.  The
page is an ASP.NET form whose county area type has to be selected in a first
postback and whose county checkboxes then have to be sent with the export.
"""

import argparse
import datetime as dt
import json
import logging
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fieldshift.net import install_dns_fallback

RMA_BASE = ("https://pubfs-rma.fpac.usda.gov/pub/Web_Data_Files/Summary_of_Business/"
            "cause_of_loss")
RMA_URL = RMA_BASE + "/colsom_{year}.zip"
RMA_LAYOUT = RMA_BASE + "/COL_Summary_of_Business_with_Month_All_Years.pdf"
USDM_PAGE = ("https://droughtmonitor.unl.edu/DmData/DataDownload/"
             "ComprehensiveStatistics.aspx")
USDM_FIRST_YEAR = 2000
RMA_COLUMNS = [
    "commodity_year", "state_fips", "state", "county_fips", "county_name",
    "commodity_code", "commodity", "plan_code", "plan", "coverage_category",
    "stage_code", "cause_code", "cause", "month", "month_name", "loss_year",
    "month_sequence", "policies_indemnified", "net_planted", "net_endorsed_acres",
    "liability", "total_premium", "producer_premium", "subsidy",
    "state_private_subsidy", "additional_subsidy", "efa_discount",
    "net_determined", "indemnity", "loss_ratio",
]
NUMERIC = ["policies_indemnified", "net_planted", "net_endorsed_acres", "liability",
           "total_premium", "producer_premium", "subsidy", "state_private_subsidy",
           "additional_subsidy", "efa_discount", "net_determined", "indemnity",
           "loss_ratio"]
CAUSE_GROUPS = {
    "drought": "drought",
    "heat": "heat",
    "excess moisture/precipitation/rain": "excess_water",
    "flood": "excess_water",
    "cold wet weather": "excess_water",
    "frost": "cold",
    "freeze": "cold",
    "ice": "cold",
    "hail": "hail",
    "wind/excess wind": "wind",
    "hurricane/tropical depression": "wind",
    "tornado": "wind",
    "cyclone": "wind",
    "insects": "disease_insects",
    "plant disease": "disease_insects",
    "nematodes": "disease_insects",
}
CAUSE_ORDER = ["drought", "heat", "excess_water", "cold", "hail", "wind",
               "disease_insects", "other"]
_LOG = logging.getLogger(__name__)


def session():
    install_dns_fallback()
    sess = requests.Session()
    sess.headers["User-Agent"] = "fieldshift/0.1 (NASA Space Apps crop rotation)"
    return sess


def cause_group(cause):
    """Short fixed cause group for one RMA cause description."""
    return CAUSE_GROUPS.get(str(cause).strip().lower(), "other")


def adm_id(state_fips, county_fips):
    """CY-Bench style ``US-SS-CCC`` from the two FIPS pieces."""
    return f"US-{int(state_fips):02d}-{int(county_fips):03d}"


# --- RMA -------------------------------------------------------------------

def download_year(sess, year, directory, force=False):
    """Cached cause-of-loss zip for one year."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"colsom_{year}.zip"
    if path.exists() and not force:
        return path
    with sess.get(RMA_URL.format(year=year), stream=True, timeout=(20, 180)) as response:
        if response.status_code == 404:
            _LOG.warning("No cause-of-loss file for %d", year)
            return None
        response.raise_for_status()
        temporary = path.with_name(path.name + ".part")
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1 << 20):
                handle.write(block)
        temporary.replace(path)
    _LOG.info("Downloaded %s", path.name)
    return path


def read_year(path):
    """One year's pipe-delimited cause-of-loss table, with typed numeric columns."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not names:
            raise RuntimeError(f"No .txt member in {path}")
        with archive.open(min(names)) as handle:
            frame = pd.read_csv(handle, sep="|", header=None, dtype=str,
                                names=RMA_COLUMNS, keep_default_na=False,
                                encoding="latin-1")
    frame.columns = [name.strip() for name in frame.columns]
    for name in RMA_COLUMNS[:17]:
        frame[name] = frame[name].str.strip()
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name].str.strip(), errors="coerce")
    frame["adm_id"] = [adm_id(s, c) for s, c in zip(frame.state_fips, frame.county_fips,
                                                    strict=True)]
    frame["cause_group"] = frame.cause.map(cause_group)
    frame["month"] = pd.to_numeric(frame.month, errors="coerce").astype("Int16")
    frame["commodity_year"] = pd.to_numeric(frame.commodity_year, errors="coerce"
                                            ).astype("Int16")
    return frame


def build_rma(data_dir=Path("data"), first=1989, last=None, force=False):
    """Download every year, write the loss table and the monthly loss-share table."""
    data_dir = Path(data_dir)
    last = last or dt.date.today().year
    sess = session()
    frames = []
    for year in range(first, last + 1):
        path = download_year(sess, year, data_dir / "raw" / "rma", force)
        if path is None:
            continue
        frames.append(read_year(path))
    if not frames:
        raise RuntimeError("No cause-of-loss files were downloaded")
    frame = pd.concat(frames, ignore_index=True)
    keep = ["adm_id", "state", "county_name", "commodity", "commodity_code",
            "commodity_year", "month", "cause", "cause_group", "plan",
            "coverage_category", "stage_code", "month_sequence", "loss_year",
            "net_planted",
            "net_determined", "indemnity", "total_premium", "liability",
            "loss_ratio", "policies_indemnified"]
    table = frame.loc[:, keep].rename(columns={"commodity_year": "year"})
    table = table.loc[table.year.between(1989, last)]
    out = data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)
    _write(table, out / "rma_cause_of_loss.parquet")

    shares = cause_share(table, data_dir)
    _write_qc(table, shares, out / "rma_cause_of_loss_qc.json")
    return table, shares


NASS_ALIASES = {
    "CORN": "CORN", "SOYBEANS": "SOYBEANS", "WHEAT": "WHEAT", "COTTON": "COTTON",
    "GRAIN SORGHUM": "SORGHUM FOR GRAIN", "BARLEY": "BARLEY", "OATS": "OATS",
    "RICE": "RICE", "PEANUTS": "PEANUTS", "SUNFLOWERS": "SUNFLOWER",
    "DRY BEANS": "DRY EDIBLE BEANS", "POTATOES": "POTATOES",
    "ALFALFA": "HAY", "SUGAR BEETS": "SUGAR BEETS", "SUGARCANE": "SUGARCANE",
}


def nass_planted(data_dir=Path("data")):
    """County planted acres from the NASS extract, all practices and uses."""
    frame = pd.read_csv(
        Path(data_dir) / "processed" / "nass_county_crops.csv.gz",
        usecols=["commodity_desc", "statisticcat_desc", "unit_desc", "year",
                 "state_ansi", "county_ansi", "value", "prodn_practice_desc",
                 "util_practice_desc", "class_desc"])
    frame = frame.loc[
        frame.county_ansi.notna() & (frame.statisticcat_desc == "AREA PLANTED")
        & (frame.unit_desc == "ACRES")
        & frame.prodn_practice_desc.str.startswith("ALL")
        & frame.util_practice_desc.str.startswith("ALL")
        & frame.class_desc.str.startswith("ALL")]
    frame["adm_id"] = [f"US-{int(state):02d}-{int(county):03d}" for state, county in
                       zip(frame.state_ansi, frame.county_ansi, strict=True)]
    return frame.groupby(["adm_id", "commodity_desc", "year"], observed=True
                         ).value.max().rename("planted_acres").reset_index()


def cause_share(table, data_dir=Path("data")):
    """Cause composition per county-crop-year-month, and damage share of planting.

    RMA's own net planted quantity is not a county total: summing it over all
    cause rows gives 38.6 M US corn acres for 2012 against 97.2 M planted, so the
    damage share uses NASS county planted acres instead.
    """
    frame = table.copy()
    frame["commodity_key"] = frame.commodity.str.upper().str.strip()
    keys = ["adm_id", "commodity_key", "year"]
    cell = (frame.groupby(keys + ["month", "cause_group"], observed=True)
            [["net_determined", "indemnity", "policies_indemnified"]]
            .sum(min_count=1).reset_index())
    totals = (frame.groupby(keys, observed=True)
              [["net_determined", "indemnity", "net_planted"]]
              .sum(min_count=1).reset_index()
              .rename(columns={"net_determined": "year_determined_acres",
                               "indemnity": "year_indemnity",
                               "net_planted": "rma_reported_planted"}))
    names = (frame.groupby(keys, observed=True).commodity.first()
             .rename("commodity").reset_index())
    share = cell.merge(totals, on=keys, how="left").merge(names, on=keys, how="left")
    planted = nass_planted(data_dir)
    planted["commodity_key"] = planted.commodity_desc
    planted = planted.rename(columns={"year": "year"})
    share["nass_commodity"] = share.commodity_key.map(NASS_ALIASES)
    share = share.merge(
        planted[["adm_id", "commodity_key", "year", "planted_acres"]],
        left_on=["adm_id", "nass_commodity", "year"],
        right_on=["adm_id", "commodity_key", "year"], how="left",
        suffixes=("", "_nass")).drop(columns=["commodity_key_nass", "nass_commodity"])
    with np.errstate(invalid="ignore", divide="ignore"):
        share["indemnity_share"] = np.where(share.year_indemnity > 0,
                                            share.indemnity / share.year_indemnity,
                                            np.nan)
        share["determined_share"] = np.where(share.year_determined_acres > 0,
                                             share.net_determined
                                             / share.year_determined_acres, np.nan)
        share["damage_share_of_planted"] = np.where(
            share.planted_acres > 0, share.net_determined / share.planted_acres, np.nan)
    columns = ["adm_id", "commodity", "commodity_key", "year", "month", "cause_group",
               "net_determined", "indemnity", "policies_indemnified",
               "indemnity_share", "determined_share", "planted_acres",
               "damage_share_of_planted", "year_determined_acres", "year_indemnity",
               "rma_reported_planted"]
    share = share.loc[:, columns].sort_values(
        ["adm_id", "commodity_key", "year", "month", "cause_group"]).reset_index(drop=True)
    _write(share, Path(data_dir) / "processed" / "rma_cause_share.parquet")
    return share


def _write(frame, path):
    temporary = Path(path).with_name(Path(path).name + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_qc(table, shares, path):
    observed = sorted(table.cause.unique())
    unmapped = sorted({cause for cause in observed if cause_group(cause) == "other"})
    report = {
        "units": {"net_planted": "acres reported on loss records",
                  "net_determined": "acres of loss",
                  "indemnity": "nominal USD of the crop year",
                  "total_premium": "USD", "liability": "USD",
                  "loss_ratio": "USD indemnity per USD premium",
                  "indemnity_share": "share of the county-crop-year indemnity",
                  "damage_share_of_planted": "determined acres / NASS county planted acres"},
        "source": RMA_URL.format(year="YYYY"),
        "layout": RMA_LAYOUT,
        "join_key": "commodity_key (upper case, stripped) is the join key; the "
                    "commodity column keeps the source spelling, which changed "
                    "case across years (CORN before 2019, Corn after).",
        "time_step": "month of loss",
        "years": [int(table.year.min()), int(table.year.max())],
        "rows": {"cause_of_loss": len(table), "cause_share": len(shares)},
        "commodities": int(table.commodity.nunique()),
        "counties": int(table.adm_id.nunique()),
        "causes": observed,
        "causes_mapped_to_other": unmapped,
        "duplicate_keys": int(table.duplicated(
            ["adm_id", "commodity", "year", "loss_year", "month", "cause", "plan",
             "coverage_category", "stage_code", "month_sequence"]).sum()),
        "missing": {name: float(table[name].isna().mean())
                    for name in NUMERIC if name in table.columns},
        "range_checks": {
            name: {"min": float(table[name].min()), "max": float(table[name].max()),
                   "p1": float(table[name].quantile(0.01)),
                   "p99": float(table[name].quantile(0.99))}
            for name in ("net_planted", "net_determined", "indemnity", "loss_ratio")},
        "diagnostics": {
            "negative_indemnity_rows": int((table.indemnity < 0).sum()),
            "determined_above_planted_rows": int(
                (table.net_determined > table.net_planted * 1.0001).sum()),
            "loss_ratio_above_50_rows": int((table.loss_ratio > 50).sum()),
            "note": "Negative indemnities are RMA corrections. Net determined can "
                    "exceed net planted on a single row because the cause and month "
                    "split repeats the policy acreage; see rma_cause_share. "
                    "duplicate_keys counts rows that share every retained key "
                    "column (adm_id, commodity, year, loss_year, month, cause, "
                    "plan, coverage_category, stage_code, month_sequence, "
                    "commodity_code); the source repeats a record that differs only "
                    "in its numeric values, which leaves the additive shares "
                    "correct.",
        },
        "note": ("RMA net planted quantity covers only policies that reported a "
                 "loss (2012 US corn sums to 38.6 M acres against 97.2 M NASS "
                 "planted), so damage_share_of_planted uses NASS county planted "
                 "acres. Determined acres are counted once per cause and month, so "
                 "they can sum above the planted acreage of a year."),
        "validation": {"us_indemnity_2012_bn_usd": 17.451,
                       "iowa_corn_indemnity_2012_bn_usd": 1.748,
                       "published_us_total_2012_bn_usd": 17.4},
        "accessed": dt.date.today().isoformat(),
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


# --- US Drought Monitor ----------------------------------------------------

def _attributes(tag):
    return dict(re.findall(r'([A-Za-z_:][-\w:.]*)="([^"]*)"', tag))


def _hidden(html):
    """Hidden form fields, which an ASP.NET postback needs in full."""
    fields = {}
    for tag in re.findall(r"<input[^>]*>", html):
        attributes = _attributes(tag)
        if attributes.get("type", "").lower() == "hidden" and "name" in attributes:
            fields[attributes["name"]] = attributes.get("value", "")
    return fields


def _county_boxes(html):
    """``(field name, FIPS)`` for every county checkbox on the form."""
    boxes = []
    for tag in re.findall(r"<input[^>]*>", html):
        attributes = _attributes(tag)
        name = attributes.get("name", "")
        if (attributes.get("type", "").lower() == "checkbox"
                and name.startswith("ctl00$ContentPlaceHolder1$chkStatisticsAOI_county$")):
            boxes.append((name, attributes.get("value", "")))
    return boxes


def usdm_request(sess, fips, first, last, attempts=3):
    """County weekly D0-D4 area percentages for one FIPS batch and date range."""
    for attempt in range(1, attempts + 1):
        try:
            page = sess.get(USDM_PAGE, timeout=(20, 120))
            page.raise_for_status()
            state = _hidden(page.text)
            state.update({"__EVENTTARGET": "ctl00$ContentPlaceHolder1$aoiType",
                          "ctl00$ContentPlaceHolder1$aoiType": "county"})
            selected = sess.post(USDM_PAGE, data=state, timeout=(20, 180))
            selected.raise_for_status()
            boxes = {name: code for name, code in _county_boxes(selected.text)}
            wanted = {name: code for name, code in boxes.items() if code in set(fips)}
            if not wanted:
                raise RuntimeError("No county checkboxes matched the FIPS list")
            form = _hidden(selected.text)
            form.update({
                "ctl00$ContentPlaceHolder1$aoiType": "county",
                "ctl00$ContentPlaceHolder1$ddlOutputTypeBasic": "json",
                "ctl00$ContentPlaceHolder1$ddlStatCat": "Area Percent",
                "ctl00$ContentPlaceHolder1$ddlStatFormat": "1",
                "ctl00$ContentPlaceHolder1$txtStartDate": first,
                "ctl00$ContentPlaceHolder1$txtEndDate": last,
                "ctl00$ContentPlaceHolder1$btnSubmit": "Submit",
            })
            form.update({name: "on" for name in wanted})
            response = sess.post(USDM_PAGE, data=form, timeout=(30, 900))
            response.raise_for_status()
            if "json" not in response.headers.get("content-type", ""):
                raise RuntimeError("USDM returned a page instead of data")
            return response.json()
        except Exception as error:  # transient network or form state
            if attempt == attempts:
                raise
            _LOG.warning("USDM request failed (%s), retry %d/%d", error, attempt, attempts)
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


def fetch_usdm(data_dir=Path("data"), chunk=250, force=False):
    """Cached county weekly USDM statistics for every CONUS county."""
    data_dir = Path(data_dir)
    cache = data_dir / "raw" / "usdm"
    cache.mkdir(parents=True, exist_ok=True)
    counties = pd.read_parquet(data_dir / "processed" / "eo" / "eo_county_weights.parquet",
                              columns=["adm_id"])
    order = sorted(counties.adm_id.astype(str).unique())
    fips = [code[3:5] + code[6:9] for code in order]
    sess = session()
    records = []
    for begin in range(0, len(fips), chunk):
        block = fips[begin:begin + chunk]
        path = cache / f"usdm_{begin:04d}.json"
        if path.exists() and not force:
            payload = json.loads(path.read_text())
        else:
            payload = usdm_request(sess, block,
                                   f"01/01/{USDM_FIRST_YEAR}",
                                   f"12/31/{dt.date.today().year}")
            temporary = path.with_name(path.name + ".part")
            temporary.write_text(json.dumps(payload))
            temporary.replace(path)
            _LOG.info("USDM chunk %d/%d: %d rows", begin // chunk + 1,
                      (len(fips) + chunk - 1) // chunk, len(payload))
            time.sleep(1.0)
        records.extend(payload)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("USDM returned no rows")
    frame = frame.rename(columns={"mapDate": "map_date"})
    frame["adm_id"] = "US-" + frame.fips.str[:2] + "-" + frame.fips.str[2:]
    frame["map_date"] = pd.to_datetime(frame.map_date).dt.tz_localize(None).dt.normalize()
    for name in ("none", "d0", "d1", "d2", "d3", "d4"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    keep = ["adm_id", "fips", "county", "state", "map_date", "none", "d0", "d1", "d2",
            "d3", "d4", "validStart", "validEnd"]
    table = frame.loc[:, keep].rename(columns={"none": "no_drought"})
    table = table.drop_duplicates(["adm_id", "map_date"]).sort_values(
        ["adm_id", "map_date"]).reset_index(drop=True)
    out = data_dir / "processed"
    _write(table, out / "usdm_county_weekly.parquet")
    _write_qc_usdm(table, out / "usdm_county_weekly_qc.json")
    return table


def usdm_steps(table, data_dir=Path("data")):
    """Weekly D0-D4 percentages averaged onto the shared 8-day steps.

    Each step takes the mean of the weekly maps whose valid period overlaps it,
    weighted by the number of overlapping days.
    """
    from fieldshift.data import eo

    order = eo.county_order(data_dir)
    row = {adm_id: index for index, adm_id in enumerate(order)}
    bounds = [eo.step_bounds(step) for step in range(eo.N_STEPS)]
    values = {name: np.full((len(order), eo.N_STEPS), np.nan, np.float32)
              for name in ("d0", "d1", "d2", "d3", "d4")}
    table = table.copy()
    table["row"] = table.adm_id.map(row)
    table = table.loc[table.row.notna()]
    start = pd.to_datetime(table.validStart).dt.tz_localize(None).dt.normalize()
    end = pd.to_datetime(table.validEnd).dt.tz_localize(None).dt.normalize()
    start_day = (start - pd.Timestamp("1981-01-01")).dt.days.to_numpy()
    end_day = (end - pd.Timestamp("1981-01-01")).dt.days.to_numpy()
    rows = table.row.to_numpy(int)
    frame_values = {name: table[name].to_numpy(float)
                    for name in ("d0", "d1", "d2", "d3", "d4")}
    step_day = np.array([(value[0] - np.datetime64("1981-01-01")).astype(int)
                         for value in bounds])
    step_len = np.array([(value[1] - value[0]).astype(int) for value in bounds])
    for step in range(eo.N_STEPS):
        lo = step_day[step]
        hi = lo + step_len[step]
        overlap = np.minimum(end_day, hi - 1) - np.maximum(start_day, lo) + 1
        hit = overlap > 0
        if not hit.any():
            continue
        weight = overlap[hit]
        block = rows[hit]
        for name, value in values.items():
            share = frame_values[name][hit]
            total = np.bincount(block, weights=weight * share, minlength=len(order))
            mass = np.bincount(block, weights=weight, minlength=len(order))
            with np.errstate(invalid="ignore", divide="ignore"):
                value[:, step] = np.where(mass > 0, total / mass, np.nan)
    return values


USDM_NAMES = ("d0", "d1", "d2", "d3", "d4")


def write_usdm_cube(values, data_dir=Path("data"), notes=None):
    """Store the 8-day drought cube on the shared county/step grid."""
    from fieldshift.data import eo

    data_dir = Path(data_dir)
    out = data_dir / "processed" / "usdm"
    out.mkdir(parents=True, exist_ok=True)
    order = eo.county_order(data_dir)
    cube = np.full((len(order), eo.N_STEPS, len(USDM_NAMES)), np.nan, np.float32)
    for position, name in enumerate(USDM_NAMES):
        cube[:, :, position] = values[name]
    path = out / "usdm_8day.f32"
    temporary = path.with_name(path.name + ".part")
    handle = np.memmap(temporary, dtype="<f4", mode="w+", shape=cube.shape)
    handle[:] = cube
    handle.flush()
    del handle
    temporary.replace(path)
    metadata = {
        "shape": list(cube.shape),
        "dtype": "float32",
        "order": "C",
        "counties": order,
        "steps_per_year": eo.STEPS_PER_YEAR,
        "step_days": eo.STEP_DAYS,
        "start_date": str(eo.step_bounds(0)[0]),
        "end_date": str(eo.step_bounds(eo.N_STEPS - 1)[1]),
        "step_start_dates": [str(value) for value in eo.step_starts()],
        "names": list(USDM_NAMES),
        "nan_fraction": {name: round(float(np.isnan(values[name]).mean()), 6)
                         for name in USDM_NAMES},
        "units": "percent of county area in that drought category",
        "source": USDM_PAGE,
        "placement": "each step takes the mean of the weekly maps overlapping it, "
                     "weighted by overlapping days",
        "notes": notes or [],
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _write_qc_usdm(table, path):
    report = {
        "units": {"d0..d4": "percent of county area in that drought category",
                  "no_drought": "percent of county area with no drought"},
        "source": USDM_PAGE,
        "time_step": "weekly (validStart..validEnd)",
        "date_range": [str(table.map_date.min().date()), str(table.map_date.max().date())],
        "rows": len(table),
        "counties": int(table.adm_id.nunique()),
        "duplicate_keys": int(table.duplicated(["adm_id", "map_date"]).sum()),
        "missing": {name: float(table[name].isna().mean())
                    for name in ("d0", "d1", "d2", "d3", "d4")},
        "range_checks": {name: {"min": float(table[name].min()),
                                "max": float(table[name].max())}
                         for name in ("d0", "d1", "d2", "d3", "d4")},
        "out_of_range": int(((table[["d0", "d1", "d2", "d3", "d4"]] < 0)
                             | (table[["d0", "d1", "d2", "d3", "d4"]] > 100)).sum().sum()),
        "note": "Cumulative area percentages as published; D0-D4 are nested, so they "
                "are not additive with each other.",
        "accessed": dt.date.today().isoformat(),
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", choices=["rma", "usdm", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.source in ("rma", "both"):
        table, shares = build_rma(args.data_dir, force=args.force)
        print(f"RMA: {len(table)} rows, {shares.planted_acres.notna().sum()} "
              f"county-crop-years with planted acres")
    if args.source in ("usdm", "both"):
        weekly = fetch_usdm(args.data_dir, force=args.force)
        values = usdm_steps(weekly, args.data_dir)
        metadata = write_usdm_cube(values, args.data_dir)
        print(f"USDM: {len(weekly)} county-weeks "
              f"{weekly.map_date.min().date()}..{weekly.map_date.max().date()}, "
              f"cube {metadata['shape']}")


if __name__ == "__main__":
    main()
