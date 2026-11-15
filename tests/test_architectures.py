import os

import numpy as np
import pandas as pd
import pytest
import torch

from fieldshift.features.stress import daily_stages
from fieldshift.features.traits import CROPS, numeric_traits
from fieldshift.model.neural import accelerator_device
from fieldshift.model.probabilistic import (
    ARCHITECTURES,
    DEVELOPMENT_BINS,
    THERMAL_TRAITS,
    ResidentRows,
    ResidentWeather,
    distribution,
    load_response,
    loss_summary,
    save_response,
    train_response,
)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE") == "cpu" or
         torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Requires an accelerator or explicit CPU mode",
)
VARIABLES = ["tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "cwb", "rsm"]


def _weather(tmp_path, daily):
    values = daily[VARIABLES].to_numpy(dtype=np.float32)[None]
    path = tmp_path / "weather.f32"
    values.tofile(path)
    meta = {"shape": list(values.shape), "variables": VARIABLES, "rsm_scale": 500.}
    return ResidentWeather(path, meta)


def _daily(start, days, seed):
    rng = np.random.default_rng(seed)
    date = pd.date_range(start, periods=days)
    seasonal = 12 - 14 * np.cos(2 * np.pi * (date.day_of_year - 15) / 365)
    tmin = seasonal - 5 + rng.normal(0, 3, days)
    tmax = tmin + 8 + rng.uniform(0, 6, days)
    # Quarter degrees are exact in the float16 device store, so GDD sums are exact.
    tmin, tmax = np.round(tmin * 4) / 4, np.round(tmax * 4) / 4
    return pd.DataFrame({
        "adm_id": "US-19-169", "date": date, "tmin": tmin, "tmax": tmax,
        "tavg": (tmin + tmax) / 2, "prec": rng.gamma(.5, 4, days), "rad": 18.,
        "vpd": 1., "et0": 3., "cwb": 0., "rsm": 300.,
    })


def _thermal_counts(tmp_path, crop, daily, sow, year):
    t = numeric_traits(CROPS[crop])
    weather = _weather(tmp_path, daily)
    device = weather.values.device
    start = (pd.Timestamp(sow) - daily.date.iloc[0]).days
    end = pd.Timestamp(year, int(t["season_end_month"]), 31)
    window = (end - pd.Timestamp(sow)).days + 1
    jan1 = (pd.Timestamp(year, 1, 1) - pd.Timestamp(sow)).days
    traits = torch.tensor([[t[k] for k in THERMAL_TRAITS]], dtype=torch.float32, device=device)
    as_tensor = lambda v: torch.tensor([v], dtype=torch.long, device=device)
    grid = weather.thermal_windows(as_tensor(0), as_tensor(start), as_tensor(window), as_tensor(jan1), traits)
    assert grid.device == device and grid.shape == (1, weather.thermal_channels, 128)
    return t, grid[0, 18].cpu().numpy() * 5, grid[0]


@pytest.mark.parametrize("crop,sow,year", [("corn", "2010-05-01", 2010), ("winter_wheat", "2009-10-01", 2010)])
def test_thermal_axis_matches_stage_feature_gdd(tmp_path, crop, sow, year):
    daily = _daily("2009-01-01", 730, 7)
    t, day_counts, grid = _thermal_counts(tmp_path, crop, daily, sow, year)
    stages = daily_stages(daily, CROPS[crop], pd.Timestamp(sow), year)
    width = t["maturity_gdd"] / DEVELOPMENT_BINS
    expected = np.bincount(np.floor(stages.gdd_before / width).astype(int), minlength=DEVELOPMENT_BINS)
    # Same GDD (incl. dormancy/vernalisation cap) as the named stage features.
    np.testing.assert_allclose(day_counts[:DEVELOPMENT_BINS], expected[:DEVELOPMENT_BINS], atol=1e-6)
    stage = grid[19:24].argmax(0).cpu().numpy()
    centre = (np.arange(128) + .5) * width
    bounds = [t["emergence_gdd"], t["flowering_gdd"], t["flowering_gdd"] + t["flowering_duration_gdd"], t["maturity_gdd"]]
    np.testing.assert_array_equal(stage, np.searchsorted(bounds, centre, side="right"))


def test_thermal_bins_fill_hot_days_and_record_speed(tmp_path):
    daily = _daily("2010-01-01", 365, 1)
    daily[["tmin", "tmax", "tavg"]] = 30.  # corn: 22 GDD/day, wider than one bin
    _, day_counts, grid = _thermal_counts(tmp_path, "corn", daily, "2010-05-01", 2010)
    assert torch.all(grid[9:18, :DEVELOPMENT_BINS] == 1)  # every bin observed
    assert np.all(day_counts[:DEVELOPMENT_BINS] <= 1)
    torch.testing.assert_close(grid[0, :DEVELOPMENT_BINS], torch.ones(DEVELOPMENT_BINS, device=grid.device))


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_reference_ensemble_trains_on_accelerator(tmp_path, name):
    daily = _daily("2009-01-01", 730, 3)
    weather = _weather(tmp_path, daily)
    rng = np.random.default_rng(0)
    n = 96
    traits = numeric_traits(CROPS["corn"])
    data = pd.DataFrame({
        "x": rng.normal(size=n), "z": np.where(rng.random(n) < .2, np.nan, rng.normal(size=n)),
        "county_index": 0, "start_index": 120 + rng.integers(0, 20, n), "season_length": 150,
        "crop": "corn", "adm_id": [str(i) for i in range(n)], "year": 2009,
        "season_start_ordinal": pd.Timestamp("2009-05-01").toordinal(),
        **{f"trait__{k}": v for k, v in traits.items()},
    })
    columns = ["x", "z"] + [c for c in data if c.startswith("trait__")]
    rows = ResidentRows(data, columns, rng.normal(0, .2, n))
    index = rows.indices(np.ones(n, dtype=bool))
    trait_indices = list(range(2, len(columns)))
    net, timing = train_response(rows, index, trait_indices, epochs=1, weather=weather,
                                 architecture=name, batch_size=32)
    device = torch.empty(0, device=accelerator_device()).device
    assert timing["device"] == device.type and len(timing["epoch_train_loss"]) == 1
    assert all(p.device == device for p in net.parameters())
    predicted = distribution(net, rows, index, weather)
    assert predicted.device == device and torch.isfinite(predicted).all()
    assert np.isfinite(loss_summary(net, rows, index, weather)["nll"])
    if name == "thermal_cnn":
        path = tmp_path / f"{name}.pt"
        save_response(net, columns, path, {})
        loaded, _, _ = load_response(path)
        torch.testing.assert_close(distribution(loaded, rows, index, weather), predicted)
