import math

import pytest
import torch

from fieldshift.planner.search import rank, select_comparisons
from fieldshift.sim.components import (
    carbon_step,
    kc_curve,
    nitrogen_step,
    pest_penalty,
    plant_carbon,
    rusle,
    water_step,
    water_stress,
)
from fieldshift.sim.traits import CATALOG, Crop


def t(x):
    return torch.tensor(float(x), dtype=torch.float64)


def test_daily_water_mass_balance_and_bounds():
    storage = t(100)
    for rain in (0, 8, 200, 0, 0):
        new, f = water_step(storage, t(rain), t(5), t(1.2), t(160), t(.4))
        assert float(new + f["et_mm"] + f["drainage_mm"]) == pytest.approx(float(storage + rain), abs=1e-10)
        assert 0 <= new <= 160
        storage = new


def test_fao56_example37_and_kc_points():
    # FAO-56 Ex37: TAW160, RAW64; day3 starts depletion67 => Ks=93/96.
    assert float(water_stress(t(93), t(160), t(.4))) == pytest.approx(93 / 96)
    assert float(water_stress(t(105), t(160), t(.4))) == 1
    for progress, expected in [(0, .3), (.2, .3), (.5, 1.2), (.8, 1.2), (1, .35)]:
        assert float(kc_curve(t(progress), t(.3), t(1.2), t(.35))) == pytest.approx(expected)


@pytest.mark.parametrize("fixation,harvest,residue", [(0, 120, 40), (200, 180, 50), (300, 0, 0), (0, 1000, 400)])
def test_nitrogen_conservation(fixation, harvest, residue):
    mineral, organic, flows = nitrogen_step(t(30), t(2000), t(fixation), t(100),
                                            t(residue), t(harvest), t(100), t(200))
    initial = 30 + 2000 + fixation + 100
    final = mineral + organic + flows["harvest_removal"] + flows["leaching"]
    assert float(final) == pytest.approx(initial, abs=1e-9)


def test_rusle_hand_computation():
    theta = math.atan(.02)
    expected = 100 * .03 * (100 / 22.13) ** .5 * (
        65.41 * math.sin(theta) ** 2 + 4.56 * math.sin(theta) + .065) * .2
    assert float(rusle(t(100), .03, t(2), 100, .2)) == pytest.approx(expected)


def test_epic_k_hand_case():
    from fieldshift.planner.data import epic_k

    # Story County IA texture: hand computation gives 0.2018*0.862*0.817*1.0 US = 0.0187 SI.
    assert epic_k(33.48, 40.61, 25.91, 2.54) == pytest.approx(.0187, rel=.02)


def test_pest_break_and_build_up():
    first = pest_penalty(t(0), .043, t(3), t(0))
    repeat = pest_penalty(t(1), .043, t(3), t(0))
    long_run = pest_penalty(t(5), .043, t(3), t(0))
    assert float(first) == 0
    assert float(repeat) == pytest.approx(.043)
    assert long_run > repeat
    assert pest_penalty(t(0), .043, t(3), t(.5)) < pest_penalty(t(0), .043, t(3), t(1))


def test_carbon_return_vs_removed():
    removed, _ = carbon_step(t(30), t(0), t(.5))
    returned, flows = carbon_step(t(30), t(3), t(.5))
    assert removed < 30
    assert float(returned) == pytest.approx(30 + 3 * .15 + .5 * .35 - .6)
    assert float(flows["retained_c"]) == pytest.approx(3 * .15 + .5 * .35)


def test_bolinder_plant_carbon():
    # Bolinder 2007 Eqs 2-5: corn 10 t/ha at 15.5% moisture, HI 0.5, S:R 5.6, extra-root 0.65.
    above, below = plant_carbon(t(10), t(.845), t(.5), t(5.6), t(.65))
    dry = 10 * .845
    assert float(above) == pytest.approx(dry * .45)
    assert float(below) == pytest.approx(dry / (5.6 * .5) * .45 * 1.65)


def _table_rows():
    rows = []
    for i in range(5):
        rows.append({"id": i, "feasible": True, "is_existing": i == 0, "income": [5, 4, 3, 1, 2][i],
                     "water": -i, "soil": i, "nitrogen": i, "nitrate": -i, "erosion": -i, "forage": 0, "cover": i,
                     "resilience": [5, 1, 1, 1, 1][i], "market": 1, "cycle": "corn → soybean" if i < 3 else "oats",
                     "cover_option": "" if i in (0, 3) else "cereal_rye_cover", "double_crop": False,
                     "rotation": f"r{i}", "erosion_t_ha_p90": 2, "income_p10": 0,
                     "income_median": [5, 4, 3, 1, 2][i]})
    return rows


def test_rank_is_pure_and_comparisons_are_distinct_goal_winners():
    rows = _table_rows()
    assert rank(rows, {"income": 1}).iloc[0].id == 0
    assert rank(rows, {"water": 1}).iloc[0].id == 0
    assert "score" not in rows[0]
    chosen = select_comparisons(rank(rows, {"soil": 1}))
    ids = [i for i, _ in chosen]
    assert ids[0] == 0 and "Existing rotation" in chosen[0][1]
    # The existing rotation wins margin and bad-year margin, so those goals label it instead of a weaker alternative.
    assert "Highest margin" in chosen[0][1] and "Highest bad-year margin" in chosen[0][1]
    # Only the 80%-margin cover passes the new default 70% floor; never fill
    # the page with inferior strategies merely to obtain four columns.
    assert len(set(ids)) == len(ids) == 2
    labels = dict(chosen)
    assert any("Existing crops plus a winter cover" in v for v in labels.values())
    cover_id = next(i for i, v in labels.items() if "Existing crops plus a winter cover" in v)
    assert rows[cover_id]["cycle"] == rows[0]["cycle"] and rows[cover_id]["cover_option"]


def test_canonical_cycle_representative():
    from fieldshift.planner.search import canonical

    assert canonical([1, 0, 1, 1]) == canonical([0, 1, 1, 1]) == (0, 1, 1, 1)
    assert canonical([1, 0, 1, 0]) == (0, 1, 0, 1)


def test_new_record_has_all_traits_without_identifier_logic():
    new = Crop("entirely_new_record", CATALOG["corn"].traits.copy())
    assert new.key not in CATALOG
    assert new.v("kc_mid") == 1.2
    changed = new.override(price_usd_t=100)
    assert changed.v("price_usd_t") == 100
    assert new.v("price_usd_t") == 165


def _synthetic_site(years=(1999, 2000, 2001, 2002, 2003, 2004)):
    import numpy as np
    import pandas as pd

    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    day = dates.day_of_year.to_numpy()
    rng = np.random.default_rng(0)
    temp = 10 + 14 * np.sin(2 * np.pi * (day - 105) / 365) + rng.normal(0, 2, len(dates))
    rain = np.where(rng.random(len(dates)) < .3, rng.gamma(1.2, 7, len(dates)), 0)
    daily = pd.DataFrame({"date": dates, "tavg": temp, "tmin": temp - 6, "tmax": temp + 6, "prec": rain,
                          "et0": np.clip(1 + .18 * temp, .2, None), "vpd": 1., "rad": 15., "rsm": .6,
                          "gwetroot": .6})
    soil = {"awc_mm_per_m": 150., "profile_depth_m": 2.0, "bulk_density_kg_m3": 1300., "organic_matter_pct": 3.,
            "rusle_k": .03, "slope_pct": 2., "slope_length_m": 100.}
    return {"daily": daily, "soil": soil, "planting": {}}


needs_accelerator = pytest.mark.skipif(not (torch.backends.mps.is_available() or torch.cuda.is_available()),
                                       reason="The simulator deliberately refuses CPU fallback")


@needs_accelerator
def test_vectorised_calendar_matches_pandas_reference():
    import numpy as np
    import pandas as pd

    from fieldshift.features.stress import compute_stress, sowing_dates
    from fieldshift.sim.calendar import DailyGrid, sowing_offsets, stress_features
    from fieldshift.sim.components import accelerator

    daily = _synthetic_site()["daily"]
    daily["adm_id"] = "field"
    clim = daily.groupby(daily.date.dt.strftime("%m-%d")).rsm.transform("mean")
    daily["rsm_anomaly"] = daily.rsm - clim
    grid = DailyGrid([daily], accelerator())
    for key in ("corn", "winter_wheat", "alfalfa"):
        traits = CATALOG[key].traits
        sow = sowing_offsets(grid, traits)
        features = stress_features(grid, traits, sow)
        for yi in (2, 4):
            year = grid.years[yi]
            wd = daily.loc[daily.date.between(f"{year - 1}-01-01", f"{year}-12-31")]
            date = sowing_dates(wd, traits, year).iloc[0]
            assert (date - pd.Timestamp(year - 1, 1, 1)).days == int(sow[0, yi])
            reference = compute_stress(wd, traits, date, year).iloc[0]
            for name, value in features.items():
                a, b = float(value[0, yi]), float(reference[name])
                assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-4, abs=1e-3), (key, name)


def _simulate(rotations, cover=True, rain_change=0.):
    import numpy as np

    from fieldshift.sim.engine import (
        fao_reference,
        ledger_pass,
        prepare,
        trajectories,
        water_pass,
    )

    site = _synthetic_site()
    crops = [Crop("brand_new_identifier", CATALOG["corn"].traits.copy()), CATALOG["soybean"]]
    covers = [CATALOG["cereal_rye_cover"]] if cover else []
    prepared = prepare([site], crops, covers, None, rain_change, 0., learned=False)
    traj = trajectories(rotations, crops, len(prepared["years"]), spin=2, score=2)
    water = water_pass(prepared, [site["soil"]], crops, covers, None, traj)
    reference = fao_reference(water, traj, 1, len(crops), traj["spin"])
    normals = np.array([[10., 3.]], dtype=np.float32)
    ledger = ledger_pass(prepared, water, [site["soil"]], normals, crops, covers, None, traj,
                         fao_reference=reference)
    return prepared, traj, water, ledger, reference, (site, crops, covers, normals)


@needs_accelerator
def test_water_pass_closes_the_mass_balance():
    _prepared, _traj, water, _, _, (site, *_) = _simulate([((0, 1, 0, 1), 1, False)])
    capacity = site["soil"]["awc_mm_per_m"] * site["soil"]["profile_depth_m"]
    inflow = water["rain_mm"] + water["irrigation_mm"] - water["et_total_mm"] - water["drainage_mm"]
    expected = capacity + inflow.sum(1)
    assert torch.allclose(water["water_mm"][:, -1], expected, atol=1e-2)
    assert (water["water_mm"] >= 0).all() and (water["water_mm"] <= capacity + 1e-3).all()


@needs_accelerator
def test_cyclic_shifts_score_identically():
    rotations = [((0, 0, 1, 0), 0, False), ((0, 1, 0, 0), 0, False)]
    _, traj, _, ledger, _, _ = _simulate(rotations, cover=False)
    first = traj["rotation"] == 0
    for name in ("income", "et_total_mm", "n_leached_kg_ha", "erosion_t_ha"):
        a = torch.sort(ledger[name][torch.as_tensor(first)][:, traj["spin"]:].mean(1)).values
        b = torch.sort(ledger[name][torch.as_tensor(~first)][:, traj["spin"]:].mean(1)).values
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-4), name


@needs_accelerator
def test_winter_cover_acts_through_the_ledgers():
    rotations = [((0, 1, 0, 1), 0, False), ((0, 1, 0, 1), 1, False)]
    _, traj, _water, ledger, _, (_, _crops, covers, _) = _simulate(rotations)
    without, with_cover = (torch.as_tensor(traj["rotation"] == r) for r in (0, 1))
    scored = slice(traj["spin"], None)
    mean = lambda name, mask: float(ledger[name][mask][:, scored].mean())
    assert mean("bare_days", with_cover) < mean("bare_days", without)
    assert mean("et_total_mm", with_cover) > mean("et_total_mm", without)
    assert mean("drainage_mm", with_cover) < mean("drainage_mm", without)
    assert mean("cover_biomass_t_ha", with_cover) > 0 == mean("cover_biomass_t_ha", without)
    assert mean("soil_carbon_input_t_ha", with_cover) > mean("soil_carbon_input_t_ha", without)
    assert mean("cover_cost_usd_ha", with_cover) == pytest.approx(covers[0].v("cost_usd_ha"), rel=1e-5)


@needs_accelerator
def test_zero_weather_edit_reproduces_baseline_exactly():
    from fieldshift.sim.engine import ledger_pass, prepare, water_pass

    rotations = [((0, 1, 0, 1), 1, False)]
    _, traj, water, ledger, reference, (site, crops, covers, normals) = _simulate(rotations)
    edited = prepare([site], crops, covers, None, 0., 0., learned=False)
    again = water_pass(edited, [site["soil"]], crops, covers, None, traj)
    replay = ledger_pass(edited, again, [site["soil"]], normals, crops, covers, None, traj,
                         fao_reference=reference, base_fao={"fao_main": water["fao_main"]})
    for name in ("income", "yield_relative", "erosion_t_ha", "n_leached_kg_ha"):
        assert torch.equal(replay[name], ledger[name]), name
    # A drier what-if lowers yield through the FAO-33 ratio against the same baseline.
    _, _, dry_water, _, _, _ = _simulate(rotations, rain_change=-.4)
    dry_prepared = prepare([site], crops, covers, None, -.4, 0., learned=False)
    dry = ledger_pass(dry_prepared, dry_water, [site["soil"]], normals, crops, covers, None, traj,
                      fao_reference=reference, base_fao={"fao_main": water["fao_main"]})
    assert float(dry["yield_relative"].mean()) < float(ledger["yield_relative"].mean())


@needs_accelerator
def test_monthly_nitrogen_mass_balance_and_nonnegative_stores():
    _, _, _, ledger, _, _ = _simulate([((0, 1, 0, 1), 1, False)])
    assert float(ledger["n_balance_error_kg_ha"].abs().max()) < .005
    assert bool((ledger["mineral_n_kg_ha"] >= -1e-4).all())
    assert float(ledger["n_cover_uptake_kg_ha"].sum()) > 0


@needs_accelerator
def test_cover_failure_uses_species_frost_and_hardiness_traits():
    from fieldshift.sim.engine import prepare
    site = _synthetic_site()
    crops = [CATALOG["corn"]]
    cover = CATALOG["cereal_rye_cover"]
    impossible_frost = cover.override(minimum_frost_free_days=500)
    impossible_cold = cover.override(winter_min_c=100)
    p = prepare([site], crops, [cover, impossible_frost, impossible_cold], learned=False)
    assert bool(p["covers"]["alive"][:, 0].any())
    assert not bool(p["covers"]["alive"][:, 1:].any())


@needs_accelerator
def test_legume_cover_credit_reduces_applied_n_and_is_in_income():
    from fieldshift.sim.engine import ledger_pass
    p, tr, water, _, ref, (site, crops, covers, normals) = _simulate([((0, 1, 0, 1), 1, False)])
    # Isolate the fertilizer-credit contract with a positive measured-stand
    # scenario. This cool synthetic calendar otherwise establishes cover only
    # before soybean (which correctly receives no fertilizer to credit).
    water["cover_established"] = torch.ones_like(water["cover_established"])
    water["cover_biomass_t_ha"] = torch.full_like(water["cover_biomass_t_ha"], 2.)
    rye = ledger_pass(p, water, [site["soil"]], normals, crops, covers, None, tr, fao_reference=ref)
    legume = covers[0].override(fixed_n_fraction=.65)
    vetch = ledger_pass(p, water, [site["soil"]], normals, crops, [legume], None, tr, fao_reference=ref)
    assert float(vetch["n_fertilizer_kg_ha"].sum()) < float(rye["n_fertilizer_kg_ha"].sum())
    saving = vetch["n_fertilizer_saving_usd_ha"]
    assert float(saving.sum()) > 0
    assert torch.allclose(vetch["income"] - rye["income"], saving, atol=.001)


def test_nass_cotton_uses_upland_class_not_nonexistent_all_classes():
    import pandas as pd

    from fieldshift.planner.data import _crop_rows
    frame = pd.DataFrame({"commodity_desc": ["COTTON", "COTTON", "OATS"],
                          "class_desc": ["UPLAND", "PIMA", "ALL CLASSES"]})
    assert list(_crop_rows(frame, "COTTON").class_desc) == ["UPLAND"]


@needs_accelerator
def test_pareto_selection_excludes_dominated_and_low_margin_candidates():
    from fieldshift.planner.search import comparison_pool
    rows = _table_rows()
    points = [(100, -100, 1), (80, -80, 1), (70, -90, .8), (60, -50, 3), (90, -120, .5)]
    for row, (income, water, soil) in zip(rows, points, strict=True):
        row.update(income=income, income_median=income, water=water, soil=soil)
    weights = {"income": 1, "water": 1, "soil": 1}
    pool = comparison_pool(rank(rows, weights), .7, weights)
    assert set(pool.loc[pool.pareto, "id"]) == {0, 1, 3}
    assert set(pool.loc[pool.display_allowed, "id"]) == {0, 1}
    assert "pareto" not in rows[0]


@needs_accelerator
def test_rye_does_not_die_from_zero_autumn_gdd_or_calendar_padding():
    from fieldshift.sim.engine import prepare
    site = _synthetic_site()
    # Isolate the mortality rule by suppressing thermal growth everywhere.
    cover = CATALOG["cereal_rye_cover"].override(base_c=99.)
    p = prepare([site], [CATALOG["corn"]], [cover], learned=False)
    assert float(p["covers"]["gdd"].sum()) == 0
    assert bool(p["covers"]["alive"].all())  # includes padded non-leap years
    assert CATALOG["cereal_rye_cover"].v("germination_min_c") == pytest.approx(10 / 9)
    assert CATALOG["cereal_rye_cover"].v("base_c") == pytest.approx(10 / 3)


def test_usdm_uses_cumulative_d2_without_double_counting():
    import pandas as pd

    from fieldshift.planner.observations import drought_summers
    weekly = pd.DataFrame({"map_date": pd.to_datetime(["2012-07-01", "2012-07-08", "2013-07-01"]),
                           "d2": [50., 75., 0.], "d3": [25., 40., 0.], "d4": [5., 10., 0.]})
    assert drought_summers(weekly).max_d2_area_pct.tolist() == [75., 0.]


@needs_accelerator
def test_diverse_cycle_gain_is_cited_trait_not_crop_identifier_logic():
    import numpy as np

    from fieldshift.sim.engine import (
        fao_reference,
        ledger_pass,
        prepare,
        trajectories,
        water_pass,
    )

    site = _synthetic_site(years=tuple(range(1998, 2006)))
    crops = [Crop("new_corn_name", CATALOG["corn"].traits), CATALOG["soybean"], CATALOG["oats"]]
    p = prepare([site], crops, learned=False)
    tr = trajectories([((0, 1, 2, 2), 0, False)], crops, len(p["years"]), starts=[0])
    water = water_pass(p, [site["soil"]], crops, [], None, tr)
    ref = fao_reference(water, tr, 1, len(crops), 4)
    normals = np.array([[10., 3., 2.5]], dtype=np.float32)
    gained = ledger_pass(p, water, [site["soil"]], normals, crops, [], None, tr, fao_reference=ref)
    control = ledger_pass(p, water, [site["soil"]], normals,
                          [c.override(diverse_rotation_yield_gain=0) for c in crops], [], None, tr, fao_reference=ref)
    seq = torch.as_tensor(tr["crop"][:, 1:-1], device=p["device"])
    relative = gained["yield_relative"] / control["yield_relative"]
    assert torch.allclose(relative[seq == 0], torch.full_like(relative[seq == 0], 1.04), atol=1e-5)
    assert torch.allclose(relative[seq == 1], torch.full_like(relative[seq == 1], 1.09), atol=1e-5)


def test_internal_unmet_n_demand_is_not_a_reported_or_scored_outcome():
    from fieldshift.planner.search import METRICS, SUMMARY

    assert "n_shortage_kg_ha" not in SUMMARY
    assert "n_shortage_kg_ha" not in METRICS


@needs_accelerator
def test_engine_uses_season_level_fao33_factor():
    _prepared, traj, water, _, _, (_, crops, _, _) = _simulate([((0, 1, 0, 1), 0, False)], cover=False)
    p = _prepared
    actual, potential = water["stage_actual_et"].sum(-1), water["stage_potential_et"].sum(-1)
    crop = torch.as_tensor(traj["crop"][:, 1:-1], device=actual.device)
    year = torch.as_tensor(traj["year0"], device=actual.device)[:, None] + torch.arange(traj["horizon"],
                                                                                        device=actual.device)
    ky = torch.tensor([c.v("ky") for c in crops], device=actual.device)[crop]
    deficit = torch.where(potential > 0, 1 - actual / potential.clamp_min(1e-6), 0).clamp(0, 1)
    expected = (1 - ky * deficit).clamp(0, 1) * torch.exp(-.02 * p["stage_heat"][0, crop, year]
                                                         - .05 * p["stage_frost"][0, crop, year])
    assert torch.allclose(water["fao_main"], expected, atol=1e-5)


@needs_accelerator
def test_prism_rescaling_keeps_wet_days_and_conserves_step_totals(monkeypatch):
    import numpy as np
    import pandas as pd

    from fieldshift.planner import data
    from fieldshift.sim.components import accelerator

    cube = np.zeros((1, 2, 2), np.float32)
    cube[0, 0] = (16, 2)   # PRISM step total, wet days
    cube[0, 1] = (9, 3)
    meta = {"counties": ["X"], "variables": [{"name": "ppt_sum"}, {"name": "wet_days"}],
            "step_start_dates": ["2001-01-01", "2001-01-09"]}
    monkeypatch.setattr(data, "_prism", lambda: (cube, meta))
    rain = np.zeros(16, np.float32)
    rain[[0, 3]] = (2, 6)
    frame = pd.DataFrame({"date": pd.date_range("2001-01-01", periods=16), "prec": rain})
    scaled, info = data.prism_scaled_rain([frame], ["X"], accelerator())
    scaled = scaled[0].cpu().numpy()
    assert scaled[:8] == pytest.approx([4, 0, 0, 12, 0, 0, 0, 0])
    # POWER dry, PRISM wet: equal shares on round(wet_days) evenly spaced days.
    assert (scaled[8:] > 0).sum() == 3 and scaled[8:].sum() == pytest.approx(9)
    assert scaled[8:][scaled[8:] > 0] == pytest.approx([3, 3, 3])
    assert info["dry_power_wet_prism_steps"] == 1


def _tie_case(advantage):
    import numpy as np

    from fieldshift.planner.search import RUN_COLUMNS, _metric_values

    rng = np.random.default_rng(0)
    phases, windows = 4, 10
    weather = rng.normal(800, 150, windows)
    runs = np.zeros((3, phases * windows, len(RUN_COLUMNS)), np.float32)
    for r, shift in enumerate((0., advantage, -300.)):
        income = weather[None] + shift + rng.normal(0, 30, (phases, windows))
        runs[r, :, RUN_COLUMNS.index("income")] = income.reshape(-1)
    rows = _table_rows()[:3]
    for r, row in enumerate(rows):
        values = _metric_values(torch.as_tensor(runs[r]), 1)
        row.update({k: float(v) for k, v in values.items()}, income_median=float(values["income"]),
                   cover_option="", cycle=f"c{r}")
    return rows, runs, windows


def test_near_tie_is_not_presented_as_a_better_match():
    weights = {"income": 1}
    rows, runs, windows = _tie_case(20.)
    chosen, ties = select_comparisons(rank(rows, weights), weights=weights, runs=runs, windows=windows,
                                      details=True)
    assert [i for i, _ in chosen] == [0]
    assert "As good a match as any candidate" in chosen[0][1]
    assert "Highest margin (tied within weather noise)" in chosen[0][1]
    test = ties["Best match to your priorities"]
    assert test["tie"] and 0 < test["gap"] <= test["tolerance"]
    rows, runs, windows = _tie_case(500.)
    chosen, ties = select_comparisons(rank(rows, weights), weights=weights, runs=runs, windows=windows,
                                      details=True)
    assert dict(chosen)[1][0] == "Best match to your priorities"
    assert not ties["Best match to your priorities"]["tie"]
