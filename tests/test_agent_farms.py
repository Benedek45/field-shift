import json

import pytest

from fieldshift.agent import farms


def test_registry_opaque_roundtrip_and_validation(tmp_path):
    record = {"adm_id": "US-19-169", "overrides": {"rain_change_pct": -30, "awc_mm_per_m": 120},
              "priorities": {"text": "Protect the soil", "weights": {"soil": 5}},
              "constraints": ["no new equipment"], "anonymous": True}
    a, b = farms.add(record, tmp_path), farms.add(record, tmp_path)
    assert a != b and "169" not in a and len(a) >= 18
    r = farms.get(a, tmp_path)
    assert r["overrides"]["rain_change_pct"] == -30
    assert r["priorities"]["text"] == "Protect the soil"
    assert r["anonymous"] and not r["irrigated"]
    assert len(json.loads((tmp_path / "farms.json").read_text())) == 2
    assert (tmp_path / "farms.json").stat().st_mode & 0o077 == 0
    with pytest.raises(farms.FarmError, match="Unknown farm_id"):
        farms.get("UnknownOpaqueFarm", tmp_path)


@pytest.mark.parametrize("change", [
    {"overrides": {"rain_change_pct": -101}},
    {"overrides": {"organic_matter_pct": float("nan")}},
    {"overrides": {"bad_key": 1}},
    {"irrigated": "false"},
    {"adm_id": "US-99-999"},
    {"priorities": {"weights": {"imaginary": 1}}},
    {"constraints": "no new equipment"},
])
def test_registry_rejects_bad_records(change):
    with pytest.raises(farms.FarmError):
        farms.validate({"adm_id": "US-19-169"} | change)
