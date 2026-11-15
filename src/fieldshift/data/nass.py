"""Resumable keyless Quick Stats crops download and bounded-memory county extract."""

import argparse
import gzip
import json
import logging
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from fieldshift.net import install_dns_fallback

INDEX_URL = "https://sb.nass.usda.gov/datasets/"
MAX_BYTES = 5_000_000_000
STATISTICS = {"YIELD", "AREA HARVESTED", "AREA PLANTED", "PRODUCTION"}
IDENTIFIERS = [
    "commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc",
    "statisticcat_desc", "unit_desc", "short_desc", "state_fips_code",
    "county_code", "state_alpha", "county_name", "year", "reference_period_desc",
    "domain_desc", "domaincat_desc", "state_ansi", "county_ansi",
]
FILTERS = ["source_desc", "agg_level_desc", "freq_desc", "statisticcat_desc"]
_LOG = logging.getLogger(__name__)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def discover(session):
    response = session.get(INDEX_URL, timeout=(15, 60))
    response.raise_for_status()
    parser = _Links()
    parser.feed(response.text)
    candidates = [
        urljoin(INDEX_URL, href) for href in parser.hrefs
        if re.fullmatch(r"qs\.crops_\d{8}\.txt\.gz", Path(urlparse(href).path).name)
    ]
    if not candidates:
        raise RuntimeError(f"No crops bulk files listed at {INDEX_URL}")
    return max(candidates)


def download(session, url, directory):
    """Resume only when the remote validator and exact byte range match."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(urlparse(url).path).name
    partial = path.with_name(path.name + ".part")
    metadata = path.with_name(path.name + ".download.json")
    headers = {"Accept-Encoding": "identity"}
    with session.head(url, headers=headers, timeout=(15, 60)) as response:
        response.raise_for_status()
        try:
            size = int(response.headers["Content-Length"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("Bulk source has no valid Content-Length; refusing download") from error
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
    if size <= 0 or size > MAX_BYTES:
        raise RuntimeError(f"Bulk file is {size:,} bytes; allowed range is 1–{MAX_BYTES:,}")
    info = {"url": url, "bytes": size, "validator": validator}
    _LOG.info("Bulk source %s: %s bytes", url, f"{size:,}")
    old = json.loads(metadata.read_text()) if metadata.exists() else None
    if path.exists():
        if path.stat().st_size == size and old == info:
            return path, info
        raise RuntimeError(f"Existing file does not match source metadata: {path}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and (old != info or not validator or offset > size):
        raise RuntimeError(f"Cannot safely resume changed source: {partial}")
    metadata.write_text(json.dumps(info, indent=2) + "\n")
    if offset < size:
        if offset:
            headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
            _LOG.info("Resuming validated partial download at byte %s", f"{offset:,}")
        with session.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise RuntimeError("Unexpected transfer encoding; cannot resume safely")
            if offset:
                expected = f"bytes {offset}-{size - 1}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Server did not honor the requested resume range")
            elif response.status_code != 200:
                raise RuntimeError(f"Unexpected download HTTP {response.status_code}")
            with partial.open("ab" if offset else "wb") as output:
                try:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        output.write(block)
                except requests.RequestException as error:
                    raise RuntimeError(
                        f"Download interrupted; partial retained at {partial}. "
                        "Rerun the same command to resume."
                    ) from error
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download retained for resume: {partial}")
    partial.replace(path)
    return path, info


def process(path, output, source=None, chunksize=100_000):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    header = list(pd.read_csv(path, sep="\t", nrows=0).columns)
    names = {name.lower(): name for name in header}
    needed = IDENTIFIERS + FILTERS + ["value"]
    missing = set(needed) - names.keys()
    if missing:
        raise RuntimeError(f"Missing bulk columns: {sorted(missing)}")
    columns = needed + (["cv_%"] if "cv_%" in names else [])
    observed = defaultdict(set)
    practices = set()
    references = set()
    codes = Counter()
    yield_rows = Counter()
    rows = total = missing_values = 0
    min_year = max_year = None
    with gzip.open(temporary, "wt", newline="") as destination:
        first = True
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str, keep_default_na=False,
            usecols=[names[name] for name in columns], chunksize=chunksize,
        ):
            chunk.columns = chunk.columns.str.lower()
            total += len(chunk)
            for column in FILTERS:
                observed[column].update(chunk[column].unique())
            selected = chunk.loc[
                chunk["source_desc"].eq("SURVEY")
                & chunk["agg_level_desc"].eq("COUNTY")
                & chunk["freq_desc"].eq("ANNUAL")
                & chunk["statisticcat_desc"].isin(STATISTICS)
            ].copy()
            if selected.empty:
                continue
            selected["value_raw"] = selected["value"].str.strip()
            selected["value"] = pd.to_numeric(
                selected["value_raw"].str.replace(",", "", regex=False), errors="coerce"
            )
            if "cv_%" in selected:
                selected["cv_%"] = (
                    selected["cv_%"].str.replace("\x00", "", regex=False).str.strip()
                )
            missing_mask = selected["value"].isna()
            missing_values += int(missing_mask.sum())
            codes.update(selected.loc[missing_mask, "value_raw"])
            years = pd.to_numeric(selected["year"], errors="raise")
            lo, hi = int(years.min()), int(years.max())
            min_year = lo if min_year is None else min(min_year, lo)
            max_year = hi if max_year is None else max(max_year, hi)
            practices.update(selected["prodn_practice_desc"].unique())
            references.update(selected["reference_period_desc"].unique())
            yield_rows.update(
                selected.loc[selected["statisticcat_desc"].eq("YIELD"), "commodity_desc"]
                .value_counts().to_dict()
            )
            keep = IDENTIFIERS + ["value", "value_raw"]
            if "cv_%" in selected:
                keep.append("cv_%")
            selected[keep].to_csv(destination, index=False, header=first)
            first = False
            rows += len(selected)
        if first:
            raise RuntimeError("No rows matched county/annual/survey/statistic filters")
    temporary.replace(output)
    summary = {
        "source": source, "header": header, "input_rows": total,
        "rows": rows, "year_range": [min_year, max_year],
        "filter_values": {key: sorted(value) for key, value in observed.items()},
        "production_practices": sorted(practices),
        "reference_periods": sorted(references),
        "missing_values": missing_values, "non_numeric_codes": dict(codes),
        "top_15_commodities_by_yield_rows": yield_rows.most_common(15),
        "note": "Counts are observations, not unique county-years; practices, classes, "
        "utilizations and domains remain distinct. NASS residual county codes are retained.",
    }
    output.with_name(output.name.removesuffix(".csv.gz") + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    install_dns_fallback()
    with requests.Session() as session:
        url = discover(session)
        raw, source = download(session, url, args.data_dir / "raw" / "nass")
    summary = process(raw, args.data_dir / "processed" / "nass_county_crops.csv.gz", source)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
