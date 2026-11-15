from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from fieldshift.data import nass


def test_chunk_filter_and_suppressed_values(tmp_path):
    base = dict.fromkeys(nass.IDENTIFIERS + nass.FILTERS + ["value", "cv_%"], "")
    base.update(
        source_desc="SURVEY", agg_level_desc="COUNTY", freq_desc="ANNUAL",
        statisticcat_desc="YIELD", commodity_desc="CORN", year="2024",
        state_fips_code="01", county_code="003",
        prodn_practice_desc="NON-IRRIGATED", domain_desc="TOTAL",
        **{"cv_%": "\x00"},
    )
    rows = [
        {**base, "value": "1,234.5"},
        {**base, "value": " (D) ", "year": "2023"},
        {**base, "value": "(Z)", "statisticcat_desc": "PRODUCTION"},
        {**base, "value": "55", "source_desc": "CENSUS"},
        {**base, "value": "55", "agg_level_desc": "STATE"},
        {**base, "value": "55", "freq_desc": "MONTHLY"},
        {**base, "value": "55", "statisticcat_desc": "SALES"},
    ]
    raw = tmp_path / "raw.txt.gz"
    pd.DataFrame(rows).rename(columns=str.upper).to_csv(raw, sep="\t", index=False)
    output = tmp_path / "county.csv.gz"
    summary = nass.process(raw, output, chunksize=2)
    result = pd.read_csv(output, dtype={"state_fips_code": str, "county_code": str})
    assert len(result) == summary["rows"] == 3
    assert result["state_fips_code"].tolist() == ["01"] * 3
    assert result["county_code"].tolist() == ["003"] * 3
    assert result["value"].iloc[0] == 1234.5
    assert result["value"].isna().sum() == 2
    assert result["cv_%"].isna().all()
    assert summary["non_numeric_codes"] == {"(D)": 1, "(Z)": 1}
    assert summary["year_range"] == [2023, 2024]
    assert summary["top_15_commodities_by_yield_rows"] == [("CORN", 2)]
    assert summary["production_practices"] == ["NON-IRRIGATED"]


def response(status=200, headers=None, content=b""):
    result = Mock(status_code=status, headers=headers or {})
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.iter_content.return_value = [content]
    return result


def test_download_size_guard(tmp_path):
    session = Mock()
    session.head.return_value = response(headers={"Content-Length": str(nass.MAX_BYTES + 1)})
    with pytest.raises(RuntimeError, match="allowed range"):
        nass.download(session, "https://example/crops.gz", tmp_path)
    session.get.assert_not_called()


def test_resume_and_completed_file(tmp_path):
    session = Mock()
    session.head.return_value = response(headers={"Content-Length": "6", "ETag": '"same"'})
    session.get.return_value = response(content=b"abc")
    url = "https://example/crops.gz"
    with pytest.raises(RuntimeError, match="Incomplete download"):
        nass.download(session, url, tmp_path)
    session.get.return_value = response(
        status=206, headers={"Content-Range": "bytes 3-5/6"}, content=b"def"
    )
    path, _ = nass.download(session, url, tmp_path)
    assert session.get.call_args.kwargs["headers"]["Range"] == "bytes=3-"
    assert path.read_bytes() == b"abcdef"
    session.get.reset_mock()
    nass.download(session, url, tmp_path)
    session.get.assert_not_called()


def test_resume_rejects_ignored_range(tmp_path):
    session = Mock()
    session.head.return_value = response(headers={"Content-Length": "6", "ETag": '"same"'})
    session.get.return_value = response(content=b"abc")
    url = "https://example/crops.gz"
    with pytest.raises(RuntimeError, match="Incomplete"):
        nass.download(session, url, tmp_path)
    session.get.return_value = response(content=b"abcdef")
    with pytest.raises(RuntimeError, match="resume range"):
        nass.download(session, url, tmp_path)
    assert (tmp_path / "crops.gz.part").read_bytes() == b"abc"


def test_interrupted_transfer_retains_partial(tmp_path):
    session = Mock()
    session.head.return_value = response(headers={"Content-Length": "6", "ETag": '"same"'})
    interrupted = response()

    def blocks(**kwargs):
        yield b"abc"
        raise requests.ConnectionError("connection reset")

    interrupted.iter_content.side_effect = blocks
    session.get.return_value = interrupted
    with pytest.raises(RuntimeError, match="Rerun the same command"):
        nass.download(session, "https://example/crops.gz", tmp_path)
    assert (tmp_path / "crops.gz.part").read_bytes() == b"abc"


def test_discovery_selects_latest(monkeypatch):
    session = Mock()
    session.get.return_value = Mock(text=(
        '<a href="/datasets/qs.crops_20261102.txt.gz">older</a>'
        '<a href="/datasets/qs.crops_20261103.txt.gz">newer</a>'
        '<a href="/datasets/qs.animals_20261104.txt.gz">unrelated</a>'
    ))
    assert nass.discover(session) == (
        "https://sb.nass.usda.gov/datasets/qs.crops_20261103.txt.gz"
    )
