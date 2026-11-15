"""One-pass recent NASS national/state price extraction; no API or credentials."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from fieldshift.planner.data import ROOT


def extract(path=Path("data/raw/nass/qs.crops_20261103.txt.gz")):
    ROOT.mkdir(parents=True, exist_ok=True)
    prices = defaultdict(list)
    count = 0
    with gzip.open(path, "rt", newline="") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        ids = {k: header.index(k) for k in ("STATISTICCAT_DESC", "AGG_LEVEL_DESC", "YEAR",
               "COMMODITY_DESC", "CLASS_DESC", "UTIL_PRACTICE_DESC", "UNIT_DESC",
               "STATE_ALPHA", "VALUE", "REFERENCE_PERIOD_DESC", "DOMAIN_DESC")}
        for line in stream:
            # Avoid parsing ~20 million non-price lines into Python dicts.
            if "\tPRICE RECEIVED\t" not in line:
                continue
            row = next(csv.reader([line], delimiter="\t"))
            if (row[ids["AGG_LEVEL_DESC"]] not in {"NATIONAL", "STATE"}
                    or int(row[ids["YEAR"]]) < 2020
                    or row[ids["REFERENCE_PERIOD_DESC"]] not in {"YEAR", "MARKETING YEAR"}
                    or row[ids["DOMAIN_DESC"]] != "TOTAL"):
                continue
            try:
                value = float(row[ids["VALUE"]].replace(",", ""))
            except ValueError:
                continue
            key = tuple(row[ids[k]] for k in ("COMMODITY_DESC", "CLASS_DESC", "UTIL_PRACTICE_DESC",
                                             "UNIT_DESC", "STATE_ALPHA"))
            prices[key].append((int(row[ids["YEAR"]]), value))
            count += 1
    rows = []
    for key, observations in prices.items():
        # Each year gets equal weight; repeated revised/reference rows don't.
        by_year = defaultdict(list)
        for year, value in observations:
            by_year[year].append(value)
        rows.append(dict(zip(("commodity", "class_desc", "util_practice", "unit", "state"), key)) |
                    {"mean": sum(sum(v) / len(v) for v in by_year.values()) / len(by_year),
                     "years": sorted(by_year), "confidence": "medium: observed historical, not a future farm quote"})
    result = {"source": str(path), "price_rows": count, "records": rows,
              "note": "Full raw archive scanned. Recent annual state/national observations, 2020 onward."}
    (ROOT / "nass_reference_prices.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    report = extract()
    print(json.dumps({"rows": report["price_rows"], "records": len(report["records"])}))
