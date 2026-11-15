import numpy as np
import pytest
import torch

from fieldshift.model.neural import accelerator_device
from fieldshift.model.recursive import (
    RecursiveField,
    load_recursive,
    save_recursive,
    water_balance,
)
from fieldshift.model.recursive_data import DRIVERS, calendar
from fieldshift.model.recursive_train import calibrate


class _StubRows:
    """Minimal row source with the FieldRows.batch contract (one county)."""

    def __init__(self, device, n=6, year=9):
        self.field = type("Field", (), {"device": device})()
        self.year = torch.full((n,), year, dtype=torch.long, device=device)
        generator = torch.Generator(device="cpu").manual_seed(0)
        self.target = (torch.randn(n, generator=generator) * .1).to(device)

    def batch(self, subset, climatology, years, crop_masks=True):
        device, b, steps = self.field.device, len(subset), years * 46
        start = (self.year[subset] - years + 1) * 46
        positions = start[:, None] + torch.arange(steps + 1, device=device)[None]
        driver = torch.zeros(b, steps, len(DRIVERS), device=device)
        driver[..., 3], driver[..., 6], driver[..., 23] = .5, .8, 1
        driver[..., 9], driver[..., 21], driver[..., 17] = 1., 1., 1.
        season = torch.zeros(b, steps, device=device)
        season[:, -30:-15] = 1
        observations = torch.full((b, steps + 1, 8), .4, device=device)
        observations[:, ::3] = torch.nan
        initial = torch.tensor([.5, .4, .6, .3, 1., .3, 2., 0.], device=device).repeat(b, 1)
        return {"initial": initial, "drivers": driver, "season": season,
                "observations": observations, "static": torch.zeros(b, 4, device=device),
                "capacity": torch.tensor([180., 40., 400.], device=device).repeat(b, 1),
                "share": torch.zeros(b, device=device),
                "county": torch.zeros(b, dtype=torch.long, device=device),
                "positions": positions}


def test_calibration_outside_inference_mode_sets_buffers():
    # Regression: the lead merge once wrote in place to inference tensors
    # returned by predict(), crashing every fit after training.
    device = accelerator_device()
    model = RecursiveField(4, hidden=32, layers=2).eval()
    rows = _StubRows(device)
    climatology = torch.full((1, 46, 8), .45, device=device)
    indices = torch.arange(6, device=device)
    assert not torch.is_inference_mode_enabled()
    record = calibrate(model, rows, indices, climatology, [1990])
    assert .25 <= record["inner_year_sd_multiplier"] <= 4
    assert torch.isfinite(model.state_error[:46]).any().item()
    assert torch.isfinite(model.state_error[46:]).any().item()


def test_modis_calendar_short_last_period_and_year_boundary():
    dates, step, first, counts = calendar("1981-01-01", "1984-12-31")
    assert len(counts) == 4 * 46
    assert counts[45] == 5 and counts[-1] == 6
    assert dates[first[46]].isoformat().startswith("1982-01-01")
    assert step[364] == 45 and step[365] == 46
    assert np.all(counts[np.arange(len(counts)) % 46 < 45] == 8)


def test_water_conservation_nonnegative_flux_and_irrigation():
    device = accelerator_device()
    torch.manual_seed(42)
    wetness = torch.rand(5, 32, 3, device=device)
    capacity = torch.tensor([180., 40., 400.], device=device).expand_as(wetness)
    logits = torch.randn(5, 32, 13, device=device, requires_grad=True)
    precipitation = torch.rand(5, 32, device=device) * 200
    et0 = torch.rand(5, 32, device=device) * 60
    days = torch.full((5, 32), 8., device=device)
    share = torch.zeros(5, 32, device=device)
    updated, flux = water_balance(wetness, precipitation, et0, days, share, capacity, logits)
    assert updated.min().item() >= -1e-6 and updated.max().item() <= 1 + 1e-6
    assert flux[..., :5].min().item() >= 0
    assert flux[..., -1].abs().max().item() < .0002
    assert flux[..., 1].abs().max().item() == 0
    updated.sum().backward()
    assert torch.isfinite(logits.grad).all().item()
    _, irrigated_flux = water_balance(
        wetness * .05, precipitation * 0, et0, days, share + 1, capacity, logits.detach())
    assert irrigated_flux[..., 1].mean().item() > 0


def test_recursive_free_rollout_harvest_causality_and_roundtrip(tmp_path):
    device = accelerator_device()
    model = RecursiveField(4, hidden=32, layers=2).eval()
    initial = torch.tensor([[.5, .4, .6, .3, 1., .3, 2., 0.]], device=device).repeat(2, 1)
    driver = torch.zeros(2, 6, len(DRIVERS), device=device)
    driver[..., 3], driver[..., 6], driver[..., 23] = .5, .8, 1
    driver[..., 9], driver[..., 21], driver[..., 17] = 1., 1., 1.
    static = torch.zeros(2, 4, device=device)
    capacity = torch.tensor([180., 40., 400.], device=device).repeat(2, 1)
    share = torch.zeros(2, device=device)
    season = torch.tensor([[1., 1., 1., 0., 0., 0.]], device=device).repeat(2, 1)
    result = model.rollout(initial, driver, static, capacity, share)
    mean, sd, _, _ = model.yield_distribution(result, driver, season)
    assert result["states"].shape == (5, 2, 7, 8)
    assert torch.isfinite(mean).all().item() and sd.min().item() > 0
    assert result["fluxes"][..., -1].abs().max().item() < .0002
    changed = driver.clone()
    changed[:, 3:, :9] += 2
    later = model.rollout(initial, changed, static, capacity, share)
    later_mean = model.yield_distribution(later, changed, season)[0]
    assert torch.allclose(mean, later_mean, atol=1e-6)
    # Inference without observations really is the same function as a free rollout.
    nonsense = torch.full((2, 7, 8), 100., device=device)
    ignored = model.rollout(initial, driver, static, capacity, share, nonsense, 0.)
    assert torch.equal(result["states"], ignored["states"])
    (mean.square().sum() + result["states"].square().mean()).backward()
    assert all(p.device == initial.device for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    path = tmp_path / "recursive.pt"
    save_recursive(model, path, {"source": "test"})
    restored, metadata = load_recursive(path)
    assert metadata["source"] == "test"
    assert torch.equal(restored.state_error, model.state_error)
    again = restored.rollout(initial, driver, static, capacity, share)
    assert torch.allclose(result["states"], again["states"])


def test_forcing_weather_driver_schema():
    assert "flowering_heat_critical" in DRIVERS
    assert not any("wetness" in name or "ndvi" in name for name in DRIVERS)
    assert len(DRIVERS) == 29


def test_simulate_multiyear_crop_sequence_and_edited_weather(tmp_path):
    import pandas as pd

    from fieldshift.features.traits import CROPS, numeric_traits
    from fieldshift.model.recursive_data import SLOTS
    from fieldshift.model.simulator import simulate

    traits = sorted(numeric_traits(CROPS["corn"]))
    soil = {"latitude": 42., "longitude": -93., "awc": 14., "bulk_density": 1.4, "drainage_class": 4.}
    names = list(soil)
    model = RecursiveField(len(names) + len(traits) + 1, hidden=32, layers=2).eval()
    path = tmp_path / "model.pt"
    save_recursive(model, path, {"trait_names": traits, "static_names": names})
    dates = pd.date_range("2026-01-01", "2027-12-31")
    weather = pd.DataFrame({
        "tmin": 8., "tmax": 24., "tavg": 16., "prec": 3., "rad": 18.,
        "vpd": 1., "et0": 2., "wind": 2., "rh": 60.,
    }, index=dates)
    state = dict(zip(SLOTS, [.5, .5, .5, .3, 1., .4, 2., 0.]))
    plan = [{"crop": "corn", "year": 2026, "sowing_date": "2026-03-15"},
            {"crop": "soybean", "year": 2027, "sowing_date": "2027-04-01"}]
    original = simulate(state, weather, plan, soil=soil, irrigated_share=0, model_path=path)
    dry = simulate(state, weather.assign(prec=2.1), plan, soil=soil, irrigated_share=0, model_path=path)
    assert original["states"].shape == (93, 8)
    assert len(original["yields"]) == 2
    assert original["water_fluxes"].balance_residual_mm.abs().max() < .0002
    assert original["states"].root_wetness.mean() > dry["states"].root_wetness.mean()
    assert len(original["final_state"]["members"]) == 5
    assert all(row["relative_yield_std"] > 0 for row in original["yields"])
    # Field state carries through the short final period; it is not reset Jan 1.
    assert not np.isclose(original["states"].root_wetness.iloc[46], state["root_wetness"])
    assert original["development_drivers"].crop_present.max() == 1


def test_report_merges_identical_rows_and_counts_climatology_leads():
    import pandas as pd
    import pytest

    from fieldshift.model.recursive_data import SLOTS
    from fieldshift.model.recursive_report import _merge, state_leads

    keys = {"crop": ["corn", "corn"], "adm_id": ["US-19-169"] * 2, "year": [2011, 2012],
            "series_id": ["corn|ALL"] * 2}
    recursive = pd.DataFrame(keys | {"target": [.1, -.4], "recursive": [0., -.2]})
    baseline = pd.DataFrame(keys | {"target": [.1, -.4], "trend": 0., "season_tree": 0.,
                                    "stage_tree": 0., "mlp": 0., "mlp_std": .2})
    assert _merge(recursive, baseline).mlp.notna().all()
    with pytest.raises(ValueError, match="targets differ"):
        _merge(recursive, baseline.assign(target=[.1, 0.]))
    rmse = np.ones((3, 92, len(SLOTS)))
    rmse[0, :6] = .5  # better than climatology for exactly six 8-day leads
    rmse[0, 6:] = 2.
    counts = np.ones((92, len(SLOTS)))
    counts[:, 7] = 0
    leads = state_leads([{"state_rmse": rmse.tolist(), "state_counts": counts.tolist()}])
    assert leads[SLOTS[0]]["leads_beating_climatology_from_start"] == 6
    assert leads[SLOTS[7]] == {"observations": 0}
@pytest.mark.parametrize("variable,bad", [
    ("lai", 25.5), ("fpar", 2.55), ("ndvi", 1.01), ("et_actual", -1.),
    ("lst_day", 101.), ("smap_rootzone", 1.01), ("ndvi", float("inf")),
])
def test_eo_loader_rejects_corrupt_targets(variable, bad):
    from fieldshift.model.recursive_data import validate_eo_values

    device = accelerator_device()
    validate_eo_values(variable, torch.tensor([float("nan"), 0.], device=device))
    with pytest.raises(ValueError, match="outside physical bounds"):
        validate_eo_values(variable, torch.tensor([bad], device=device))


def test_eo_ingestion_fails_before_assigning_fill_codes(tmp_path):
    import json

    from fieldshift.model.recursive_data import FieldData

    path = tmp_path / "eo.f32"
    np.array([[[25.5]]], dtype=np.float32).tofile(path)
    path.with_suffix(".json").write_text(json.dumps({
        "shape": [1, 1, 1], "counties": ["test"],
        "variables": [{"name": "lai", "units": "m2/m2"}],
    }))
    field = FieldData.__new__(FieldData)
    field.device, field.n_counties, field.n_steps = accelerator_device(), 1, 1
    field.meta = {"counties": ["test"]}
    with pytest.raises(ValueError, match="EO lai outside physical bounds"):
        field._eo(path)


def test_report_merge_repeat_ignores_its_aggregate(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from fieldshift.model import recursive_report as reporting
    from fieldshift.model.recursive_data import SLOTS

    out, base = tmp_path / "r3", tmp_path / "baseline"
    out.mkdir()
    base.mkdir()
    keys = {"crop": ["corn", "corn"], "adm_id": ["US-19-169"] * 2,
            "year": [2011, 2012], "series_id": ["corn|ALL"] * 2}
    predictions = pd.DataFrame(keys | {"target": [.1, -.4], "recursive": [0., -.2]})
    baseline = pd.DataFrame(keys | {"target": [.1, -.4], "trend": 0., "season_tree": 0.,
                                    "stage_tree": 0., "mlp": 0., "mlp_std": .2})
    report = {"fold": 0, "held": False, "held_crops": ["corn"], "seconds": 2,
              "state_rmse": np.ones((3, 92, len(SLOTS))).tolist(),
              "state_counts": np.ones((92, len(SLOTS))).tolist(),
              "timing": {"parameters": 100, "epoch_seconds": [1.]},
              "calibration": {"inner_year_sd_multiplier": 1.}}
    for stem, filename in (("fold_0", "predictions"), ("held_0", "held_crop_predictions")):
        predictions.to_parquet(out / f"{stem}.parquet", index=False)
        (out / f"{stem}.json").write_text(json.dumps(report | {"held": stem.startswith("held")}))
        baseline.to_parquet(base / f"{filename}.parquet", index=False)
    # Isolate file discovery/joins from the separately tested scoring functions.
    monkeypatch.setattr(reporting, "report_tables",
                        lambda ordinary, held: {"ordinary_rows": len(ordinary), "held_rows": len(held)})
    first = reporting.merge(out, base)
    second = reporting.merge(out, base)
    assert first["ordinary_rows"] == second["ordinary_rows"] == 2
    assert first["held_rows"] == second["held_rows"] == 2
    assert second["folds"] == [0]
