"""USDA Soil Data Access tabular queries."""

import pandas as pd
import requests

from fieldshift.net import install_dns_fallback

URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


def query(sql: str) -> pd.DataFrame:
    """Run SDA SQL and return rows with USDA-provided column names."""
    install_dns_fallback()
    response = requests.post(
        URL, json={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=(15, 120)
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Soil Data Access HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        body = response.json()
    except ValueError as error:
        raise RuntimeError("Soil Data Access returned invalid JSON") from error
    if not isinstance(body, dict) or any(
        "error" in key.lower() for key in body
    ):
        raise RuntimeError(f"Soil Data Access API error: {str(body)[:500]}")
    if not body:
        # SDA omits both the table and column names when the query matches no rows.
        return pd.DataFrame()
    table = body.get("Table")
    if not isinstance(table, list) or not table or not isinstance(table[0], list):
        raise RuntimeError(f"Soil Data Access missing column header: {str(body)[:500]}")
    try:
        return pd.DataFrame(table[1:], columns=table[0])
    except (ValueError, TypeError) as error:
        raise RuntimeError("Soil Data Access returned malformed table rows") from error
