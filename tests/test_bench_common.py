import pytest

from fieldshift.bench.common import canonical, choose, unique
from fieldshift.bench.run import anonymous_brief, cases_for


def rot(*names):
    return [{"crop": n} for n in names]


def test_canonical_phase_repeats_and_double_crop():
    assert canonical(rot("corn", "soybean") * 2) == canonical(rot("soybean", "corn"))
    assert canonical([{"crop": "wheat", "double_crop": "soybean"}]) != canonical(rot("wheat"))
    assert canonical([{"crop": "corn", "winter": "cereal_rye"}]) == canonical(
        [{"crop": "corn", "winter": "cereal_rye_cover"}])
    with pytest.raises(ValueError):
        canonical([])


def test_unique_rejects_near_tie():
    board = {"existing": {"rotation": rot("corn"), "tied_with_best": False},
             "rows": [{"rotation": rot("soybean"), "tied_with_best": True},
                      {"rotation": rot("oats"), "tied_with_best": False}]}
    assert unique(board)
    assert choose(board) == rot("soybean")
    board["rows"][1]["tied_with_best"] = True
    assert not unique(board)
    board["existing"]["tied_with_best"] = True
    assert choose(board) == rot("corn")
    assert not unique(board)


def test_cases_do_not_contain_keys_or_family():
    pair = {"id": "abcdef", "base": {"priorities": {}}, "planted": {"priorities": {}},
            "key": rot("corn"), "naive": rot("soybean"), "family": "secret"}
    cases = cases_for([pair])
    assert len(cases) == 2
    assert all(set(c) == {"id", "record"} for c in cases)
    assert all("secret" not in str(c) for c in cases)


def test_blind_render_removes_urls_and_forecast_response():
    text = "Source https://example.org/IA/2,026.pdf\n## Weather outlook\nraw forecast data\n## Farmer\ncrop"
    got = anonymous_brief(text)
    assert "https://" not in got
    assert "raw forecast data" not in got
    assert "## Farmer\ncrop" in got
