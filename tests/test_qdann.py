import pandas as pd

from fieldshift.data import qdann


def point(**values):
    row = {"point_id": "IA-0-0", "state": "IA", "county_fips": 19169}
    for year in qdann.HISTORY_YEARS:
        row |= {f"cdl_{year}": 0, f"interior_{year}": 1}
    for year in qdann.YIELD_YEARS:
        row |= {f"corn_{year}": 0, f"soy_{year}": 0}
    return row | values


def test_long_rows_need_matching_cdl_class_and_carry_history():
    frame = pd.DataFrame([point(
        cdl_2009=5, cdl_2010=1, cdl_2011=1, corn_2011=11000,
        cdl_2012=5, soy_2012=3500, interior_2012=0,
        cdl_2013=1, soy_2013=3000,
    )])
    long = qdann.to_long(frame)
    assert long[["year", "crop", "yield_kg_ha", "prev1", "prev2"]].to_dict("records") == [
        {"year": 2011, "crop": "corn", "yield_kg_ha": 11000, "prev1": 1, "prev2": 5},
        {"year": 2012, "crop": "soy", "yield_kg_ha": 3500, "prev1": 1, "prev2": 1},
    ]
    assert long.set_index("year")["interior"].to_dict() == {2011: True, 2012: False}
    assert (long["county_fips"] == 19169).all()


def test_seed_is_stable_per_state_and_batch():
    assert qdann._seed("IA", 0) == qdann._seed("IA", 0)
    assert qdann._seed("IA", 0) != qdann._seed("IA", 1)
    assert qdann._seed("IA", 0) != qdann._seed("IL", 0)
