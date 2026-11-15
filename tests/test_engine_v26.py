"""New conditional paths, with the old steady-state contract explicitly frozen."""

import json
import math
from pathlib import Path

import pytest
import torch

from fieldshift.sim.components import accelerator
from fieldshift.sim.forecast import validate
from fieldshift.sim.state import PROTECTED_ASSETS, ROOT, kalman_update

gpu = pytest.mark.skipif(not (torch.backends.mps.is_available() or torch.cuda.is_available()),
                         reason="Simulator requires MPS/CUDA, not a CPU fallback")


@gpu
def test_scalar_square_root_update_has_known_kalman_mean_and_covariance():
    x = torch.tensor([[0.], [2.]], device=accelerator())
    out = kalman_update(x, x[:, 0], 2., 1.)
    assert float(out.mean()) == pytest.approx(5/3, abs=1e-6)
    assert float(out[:, 0].var()) == pytest.approx(2/3, abs=1e-6)


@pytest.mark.parametrize("payload", [
    {}, {"rotations": [], "years": 1}, {"rotations": [[]], "years": True},
    {"rotations": [[]], "years": 5}, {"rotations": [[]], "weather": "made_up"},
    {"rotations": [[]], "force": True},
])
def test_forecast_rejects_invalid_envelope(payload):
    with pytest.raises(ValueError):
        validate(payload)


@gpu
def test_grouped_ensrf_equals_scalar_update_per_group():
    from fieldshift.sim.hindcast import grouped_ensrf

    g = torch.Generator().manual_seed(0)
    x = torch.rand((3, 8), generator=g).to(accelerator()) * 100
    h = x * .7 + 5
    y = torch.tensor([40., 60., 10.], device=x.device)
    out = grouped_ensrf(x, h, y, 9.)
    for i in range(3):
        ref = kalman_update(x[i, :, None], h[i], y[i], 9.)[:, 0]
        assert torch.allclose(out[i], ref, atol=1e-4)


@gpu
def test_crps_known_values():
    from fieldshift.sim.hindcast import crps

    device = accelerator()
    ens = torch.tensor([[1., 0.], [0., 2.]], device=device)
    mask = torch.tensor([[True, False], [True, True]], device=device)
    out = crps(ens, mask, torch.tensor([3., 1.], device=device))
    # One member: absolute error. Two members {0,2}, y=1: 1 - 2/4 = .5.
    assert out.tolist() == pytest.approx([2., .5])


@gpu
def test_water_pass_chunks_match_single_pass_with_alfalfa_stands():
    import numpy as np
    import pandas as pd

    from fieldshift.sim.engine import prepare, trajectories, water_pass
    from fieldshift.sim.traits import CATALOG

    dates = pd.date_range("1999-01-01", "2004-12-31")
    day = dates.day_of_year.to_numpy()
    rng = np.random.default_rng(0)
    temp = 10 + 14 * np.sin(2 * np.pi * (day - 105) / 365) + rng.normal(0, 2, len(dates))
    rain = np.where(rng.random(len(dates)) < .3, rng.gamma(1.2, 7, len(dates)), 0)
    daily = pd.DataFrame({"date": dates, "tavg": temp, "tmin": temp - 6, "tmax": temp + 6, "prec": rain,
                          "et0": np.clip(1 + .18 * temp, .2, None), "vpd": 1., "rad": 15., "rsm": .6,
                          "gwetroot": .6})
    site = {"daily": daily, "planting": {}, "soil": {
        "awc_mm_per_m": 150., "profile_depth_m": 2.0, "bulk_density_kg_m3": 1300., "organic_matter_pct": 3.,
        "rusle_k": .03, "slope_pct": 2., "slope_length_m": 100.}}
    crops = [CATALOG["corn"], CATALOG["soybean"], CATALOG["alfalfa"]]
    prepared = prepare([site], crops, [], None, 0., 0., learned=False)
    traj = trajectories([((0, 1), 0, False), ((2, 2, 2, 2), 0, False), ((0, 2, 2, 2), 0, False)], crops,
                        len(prepared["years"]), spin=2, score=2)
    whole = water_pass(prepared, [site["soil"]], crops, [], None, traj)
    parts = water_pass(prepared, [site["soil"]], crops, [], None, traj, chunk=3)
    assert len(traj["crop"]) > 3
    for name in ("fao_main", "et_total_mm", "water_mm", "stand_harvest_fraction"):
        assert torch.allclose(whole[name], parts[name], atol=1e-4), name


def test_shared_fingerprinted_inputs_have_not_even_been_touched():
    snapshot = ROOT/"protected_assets.json"
    if not snapshot.exists():
        pytest.skip("Worktree acceptance snapshot unavailable")
    before = json.loads(snapshot.read_text())
    for name in PROTECTED_ASSETS:
        p = Path("data/processed")/name
        assert [p.stat().st_size, p.stat().st_mtime_ns] == before[name], name


def test_pooled_innovations_match_known_sample_moments():
    from fieldshift.sim.hindcast import _pool_innovations

    parts = [{"group_updates": 2, "assumed_observation_sd_mm": 60.,
              "innovation_mean_mm": mean, "innovation_sd_mm": math.sqrt(2),
              "expected_innovation_sd_mm": 61., "forecast_spread_mm": 10.,
              "analysis_spread_mm": 9.} for mean in (1., 5.)]
    out = _pool_innovations(parts)  # samples {0,2} and {4,6}
    assert out["innovation_mean_mm"] == 3.
    assert out["innovation_sd_mm"] == pytest.approx(math.sqrt(20 / 3))


def _assert_v25_result(old, new, path=""):
    # York independently varies at float32 rounding between fresh MPS runs:
    # engine_v26/regression_york_detail.json records <=5.93e-7 relative.
    metadata = {"latency_seconds", "seconds", "simulator_version", "version",
                "cache_hit", "fingerprint", "device", "applied_weather_changes"}
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        assert old.keys() - metadata == new.keys() - metadata, path
        for key in old.keys() - metadata:
            _assert_v25_result(old[key], new[key], f"{path}/{key}")
    elif isinstance(old, list):
        assert len(old) == len(new), path
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            _assert_v25_result(a, b, f"{path}[{i}]")
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        assert math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-5), (path, old, new)
    else:
        assert old == new, (path, old, new)


@gpu
@pytest.mark.parametrize("name", ["story", "york", "lubbock"])
def test_v25_default_simulate_and_scoreboard_regression(name, tmp_path):
    from fieldshift.agent import farms, scoreboard
    from fieldshift.agent.runtime import Runtime

    receipt = Path("data/processed/agent/alfalfa_fieldshift-v25.json")
    integration = Path("data/processed/agent/integration_checks.json")
    if not receipt.exists() or not integration.exists():
        pytest.skip("Private v25 acceptance fixtures unavailable")
    (tmp_path / "farms.json").write_bytes((farms.ROOT / "farms.json").read_bytes())
    runtime = Runtime(root=tmp_path)
    fid = json.loads(integration.read_text())["farms"][name]
    old = json.loads(receipt.read_text())["farms"][name]
    weights = dict.fromkeys(scoreboard.DEFAULT_WEIGHTS, 1)
    weights.update(income=5, water=3, soil=3, erosion=3, resilience=5)
    board = runtime.scoreboard(fid, {"weights": weights,
                                    "limits": {"min_margin_share_of_existing": .7}, "top": 40})
    sim = runtime.simulate(fid, {"rotations": [[{"crop": "alfalfa"}] * 4]})
    _assert_v25_result(old["board"], board, name + "/scoreboard")
    _assert_v25_result(old["alfalfa"], sim, name + "/simulate")
    assert board["applied_weather_changes"] == sim["applied_weather_changes"]
