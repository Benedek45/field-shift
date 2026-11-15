import numpy as np
import pandas as pd

from fieldshift.analysis import rotation
from fieldshift.data import qdann


def make_points(n_per_county=40, counties=(19001, 19003), penalty=0.05, seed=1):
    """Half the points rotate corn/soy, half grow corn every year; continuous corn yields less."""
    rng = np.random.default_rng(seed)
    rows = []
    for county in counties:
        level = rng.uniform(9000, 12000)
        for i in range(n_per_county):
            continuous = i % 2 == 0
            row = {"point_id": f"{county}-{i}", "state": "IA", "county_fips": county}
            for year in qdann.HISTORY_YEARS:
                corn = continuous or year % 2 == 0
                row[f"cdl_{year}"] = 1 if corn else 5
                row[f"interior_{year}"] = 1
            for year in qdann.YIELD_YEARS:
                corn = row[f"cdl_{year}"] == 1
                year_effect = 1 + 0.1 * np.sin(year)
                factor = 1 - penalty if continuous else 1.0
                row[f"corn_{year}"] = round(level * year_effect * factor) if corn else 0
                row[f"soy_{year}"] = 0 if corn else 3500
            rows.append(row)
    return pd.DataFrame(rows)


def test_recovers_planted_penalty_and_run_length():
    points = make_points()
    corn = rotation.analyse(points)["interior_pixels"]["corn"]
    assert abs(corn["overall"]["penalty_pct"] - 5.0) < 0.05
    low, high = corn["overall"]["ci95_pct"]
    assert low <= 5.0 <= high
    # Continuous points have every earlier year in corn, so only long runs exist.
    assert corn["by_run_years"]["1"] is None
    assert corn["by_run_years"]["4+"]["penalty_pct"] > 4.9


def test_same_crop_run_counts_back_to_first_break():
    points = make_points(n_per_county=2, counties=(19001,))
    runs = rotation.same_crop_run(points, "corn")
    assert runs[2010].tolist() == [4, 0]
    assert runs[2008].tolist() == [2, 0]
