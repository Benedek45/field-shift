"""Rotation cycles in steady state: a daily water pass, then an annual ledger pass.

A rotation is a cycle of main crops plus per-cycle options for the period after
each harvest (a winter cover, or double-crop soybean after an autumn-sown crop).
Each trajectory replays consecutive observed weather years: the first cycle
spins up soil water, nitrogen, residue, previous-crop and pest state; the next
cycle is scored. Starting the cycle at every phase lets every crop meet every
weather year, and the wrap-around transition is counted.

The water pass (daily FAO-56 bucket, stage ET for FAO-33, cover growth) does not
depend on yields, prices or nutrient traits, so the ledger pass can be re-run for
trait sensitivity and weather what-ifs without repeating the daily loop.
"""

import time

import numpy as np
import torch

from fieldshift.features.traits import numeric_traits
from fieldshift.sim.calendar import (
    DailyGrid,
    gdd_between,
    observed_offsets,
    sowing_offsets,
    stages,
    stress_features,
)
from fieldshift.sim.components import (
    accelerator,
    carbon_step,
    fao_relative_yield,
    kc_curve,
    pest_penalty,
    plant_carbon,
    renard_freimund_r,
    rusle,
    water_step,
)
from fieldshift.sim.nitrogen import monthly_nitrogen
from fieldshift.sim.yield_response import YieldResponse

DEC_1 = 334
MONTH_OF_DAY = np.minimum(np.searchsorted(np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                                          np.arange(366), side="right"), 11)
TRACE_DAYS = {30: 0, 58: 1, 89: 2, 119: 3, 150: 4, 180: 5, 211: 6, 242: 7, 272: 8, 303: 9, 333: 10, 364: 11}
LEDGER_COLUMNS = (
    "income", "income_legacy", "return_over_listed_costs_usd_ha", "irrigation_cost_usd_ha", "economic_crop_index", "economic_cover_index",
    "economic_double_planted", "economic_same_crop", "economic_establishment", "economic_establishment_fall",
    "yield_relative", "et_mm", "et_total_mm", "irrigation_mm", "water_mm", "water_at_planting_mm",
    "mineral_n_kg_ha", "organic_matter_pct", "soil_carbon_input_t_ha", "erosion_t_ha", "bare_days",
    "forage", "n_fixed_kg_ha", "n_shortage_kg_ha", "n_leached_kg_ha", "drainage_mm", "pest_loss_pct",
    "predictive_sd", "n_credit_kg_ha", "cover_biomass_t_ha", "cover_planted", "cover_established",
    "cover_cost_usd_ha", "double_crop_t_ha", "harvest_t_ha",
    "n_fertilizer_kg_ha", "n_mineralized_kg_ha", "n_harvest_removed_kg_ha", "n_residue_return_kg_ha",
    "n_cover_uptake_kg_ha", "n_fertilizer_saving_usd_ha",
    "n_balance_error_kg_ha",
)


def _edited(daily, rain_change, temp_change):
    d = daily.copy()
    d["prec"] = d.prec * (1 + rain_change)
    for name in ("tmin", "tmax", "tavg"):
        d[name] = d[name] + temp_change
    # Hargreaves temperature factor with unchanged diurnal range and radiation.
    d["et0"] = d.et0 * ((d.tavg + 17.8) / (daily.tavg + 17.8).clip(lower=1)).clip(lower=0)
    return d


def _traits(crops, name, device):
    return torch.tensor([float(c.v(name)) for c in crops], device=device, dtype=torch.float32)


def prepare(sites, crops, covers=(), double_crop=None, rain_change=0., temp_change=0.,
            irrigated=False, learned=True, daily_values=None, base_daily_values=None, calendar_scales=None):
    """Calendars for every site, crop and harvest year, all on the accelerator.

    The bucket calendar uses USDA state 50%-planted dates where available and
    scales GDD thresholds so maturity matches the state maturity date (computed
    on unedited weather, so warming still shortens seasons). The saved yield
    model keeps the temperature sowing rule it was trained on and always sees
    unedited weather: edits act through the FAO-33 ratio in the ledger pass.
    """
    device = accelerator()
    base_frames, edit_frames = [], []
    for site in ([] if daily_values is not None else sites):
        d = site["daily"].copy()
        clim = d.groupby(d.date.dt.strftime("%m-%d")).rsm.transform("mean")
        d["rsm_anomaly"] = d.rsm - clim
        base_frames.append(d)
        edit_frames.append(_edited(d, rain_change, temp_change))
    if daily_values is not None:
        if rain_change or temp_change or learned:
            raise ValueError("Tensor weather must be pre-edited and uses the explicit FAO forecast path")
        base_frames = [sites[0]["daily"]] * len(sites)
    base = DailyGrid(base_frames, device, base_daily_values if base_daily_values is not None else daily_values)
    grid = (DailyGrid(base_frames, device, daily_values) if base_daily_values is not None else base) if daily_values is not None else (
        DailyGrid(edit_frames, device) if (rain_change or temp_change) else base)
    ns, nc, ny = len(sites), len(crops), len(base.years)
    shape = (ns, nc, ny, 366)
    main = torch.zeros(shape + (3,), device=device)
    lead = torch.zeros_like(main)
    main[..., 0] = -1
    lead[..., 0] = -1
    plant = torch.full((ns, nc, ny), -1, dtype=torch.long, device=device)
    harvest = torch.full_like(plant, -1)
    operation = torch.full_like(plant, -1)
    stage_heat = torch.zeros((ns, nc, ny), device=device)
    stage_frost = torch.zeros_like(stage_heat)
    maturity = torch.ones((ns, nc), device=device)
    timeline = {k: np.full((ns, nc, ny), -1, dtype=np.int32) for k in
                ("sow", "end", "flower_start", "flower_end")}
    timeline["gdd_scale"] = np.ones((ns, nc, ny), dtype=np.float32)
    timeline["observed_sowing"] = np.zeros((ns, nc), dtype=bool)
    scales = {}
    offset_cache = {}
    def record_offset(record):
        if daily_values is None:
            return observed_offsets(base, record)[0]
        if id(record) not in offset_cache:
            offset_cache[id(record)] = observed_offsets(base, record)[0]
        return offset_cache[id(record)]
    days = torch.arange(366, device=device)
    for ci, crop in enumerate(crops):
        t = numeric_traits(crop.traits)
        if crop.v("fallow"):
            continue
        sow = sowing_offsets(grid, crop.traits)
        scale = None
        records = [s.get("planting", {}).get(crop.key) for s in sites]
        if not t["autumn_sown"] and not t.get("perennial", 0) and any(records):
            observed = torch.stack([record_offset(r) if r else sow[i] for i, r in enumerate(records)])
            sow = observed
            timeline["observed_sowing"][:, ci] = [bool(r) for r in records]
            if any(r and "maturity" in r for r in records) and not (calendar_scales and crop.key in calendar_scales):
                mature = torch.stack([record_offset(r["maturity"]) if r and "maturity" in r
                                      else sow[i] for i, r in enumerate(records)])
                gdd = gdd_between(base, crop.traits, sow, mature)
                scale = torch.where(mature > sow, (gdd / t["maturity_gdd"]).clamp(.6, 1.6), 1.)
                scales[crop.key] = scale
        if calendar_scales and crop.key in calendar_scales:
            known = torch.as_tensor(calendar_scales[crop.key], device=device, dtype=torch.float32)
            scale = known.reshape(-1, 1).expand(ns, ny)
            scales[crop.key] = scale
        st = stages(grid, crop.traits, sow, scale)
        keep = st["keep"]
        if st["perennial"]:
            stage, progress = torch.where(keep, 1, -1), torch.full_like(st["gdd"], .5)
        else:
            mat = st["maturity"] if torch.is_tensor(st["maturity"]) else torch.tensor(st["maturity"], device=device)
            stage = torch.where(keep, st["stage"].clamp(0, 3), -1)
            progress = st["gdd_before"] / mat
            reached = ((st["gdd"] * keep).sum(-1) >= mat.squeeze(-1) if mat.dim() else
                       (st["gdd"] * keep).sum(-1) >= mat).float()
            usable = slice(1, None) if t["autumn_sown"] else slice(None)
            maturity[:, ci] = reached[:, usable].mean(-1)
        channels = torch.stack([stage.float(), progress, keep.float()], -1)
        current, preceding = grid.calendar_days(channels)
        # Days outside a season must read as bare, not as stage 0.
        current[..., 0] = torch.where(current[..., 2] > 0, current[..., 0], -1)
        preceding[..., 0] = torch.where(preceding[..., 2] > 0, preceding[..., 0], -1)
        main[:, ci] = current
        lead[:, ci, :-1] = preceding[:, 1:]
        active = current[..., 2] > 0
        plant[:, ci] = torch.where(active.any(-1), torch.where(active, days, 999).amin(-1), -1)
        harvest[:, ci] = torch.where(active.any(-1), torch.where(active, days, -1).amax(-1), -1)
        operation[:, ci] = harvest[:, ci]
        if any(r and "harvested" in r for r in records):
            prev = torch.tensor(base.prev_len, device=device)[None, :]
            done = torch.stack([record_offset(r["harvested"]) if r and "harvested" in r
                                else torch.full_like(sow[i], -999) for i, r in enumerate(records)]) - prev
            operation[:, ci] = torch.where(harvest[:, ci] >= 0, torch.maximum(harvest[:, ci], done.clamp(max=364)), -1)
        stage_heat[:, ci] = (st["heat_dd"].nan_to_num(0) * (st["stage"] == 2) * keep).sum(-1)
        stage_frost[:, ci] = (st["frost_days"] * (st["stage"] >= 2) * keep).sum(-1)
        o = torch.arange(keep.shape[-1], device=device)
        flower = keep & (st["stage"] == 2)
        timeline["sow"][:, ci] = torch.where(keep.any(-1), torch.where(keep, o, 9999).amin(-1), -1).cpu().numpy()
        timeline["end"][:, ci] = torch.where(keep.any(-1), torch.where(keep, o, -1).amax(-1), -1).cpu().numpy()
        timeline["flower_start"][:, ci] = torch.where(flower.any(-1), torch.where(flower, o, 9999).amin(-1),
                                                     -1).cpu().numpy()
        timeline["flower_end"][:, ci] = torch.where(flower.any(-1), torch.where(flower, o, -1).amax(-1),
                                                   -1).cpu().numpy()
        if scale is not None:
            timeline["gdd_scale"][:, ci] = scale.cpu().numpy()
    out = {"device": device, "years": base.years, "prev_len": base.prev_len, "main": main, "lead": lead,
           "plant": plant, "harvest": harvest, "operation": operation, "stage_heat": stage_heat, "stage_frost": stage_frost,
           "maturity": maturity, "timeline": timeline, "rain_change": rain_change, "temp_change": temp_change}
    weather = torch.stack([grid.var(v) for v in ("prec", "et0", "tmin", "tmax", "tavg")], -1).nan_to_num(0)
    out["weather"], _ = grid.calendar_days(weather)
    out["year_len"] = torch.tensor(base.year_len, device=device)
    # Double crop after an autumn-sown crop's harvest: same stage machinery.
    out["double"] = None
    if double_crop is not None:
        dtab = torch.zeros(shape + (3,), device=device)
        dtab[..., 0] = -1
        dharvest = torch.full_like(plant, -1)
        dheat = torch.zeros_like(stage_heat)
        dfrost = torch.zeros_like(stage_heat)
        dmature = torch.zeros((ns, nc), device=device)
        dyear = torch.zeros((ns, nc, ny), dtype=torch.bool, device=device)
        dt = numeric_traits(double_crop.traits)
        prev_len = torch.tensor(base.prev_len, device=device)[None, :]
        for ci, crop in enumerate(crops):
            if not crop.v("autumn_sown"):
                continue
            sow = torch.where(harvest[:, ci] >= 0, harvest[:, ci] + prev_len + int(dt["sow_after_harvest_days"]), 730)
            st = stages(grid, double_crop.traits, sow, scales.get("soybean"))
            keep = st["keep"]
            mat = st["maturity"] if torch.is_tensor(st["maturity"]) else torch.tensor(st["maturity"], device=device)
            reached = ((st["gdd"] * keep).sum(-1) >= (mat.squeeze(-1) if mat.dim() else mat)).float()
            dmature[:, ci] = reached[:, 1:].mean(-1)
            dyear[:, ci] = reached > 0
            channels = torch.stack([torch.where(keep, st["stage"].clamp(0, 3), -1).float(),
                                    st["gdd_before"] / mat, keep.float()], -1)
            current, _ = grid.calendar_days(channels)
            current[..., 0] = torch.where(current[..., 2] > 0, current[..., 0], -1)
            dtab[:, ci] = current
            active = current[..., 2] > 0
            dharvest[:, ci] = torch.where(active.any(-1), torch.where(active, days, -1).amax(-1), -1)
            dheat[:, ci] = (st["heat_dd"].nan_to_num(0) * (st["stage"] == 2) * keep).sum(-1)
            dfrost[:, ci] = (st["frost_days"] * (st["stage"] >= 2) * keep).sum(-1)
        out["double"] = {"table": dtab, "harvest": dharvest, "heat": dheat, "frost": dfrost, "maturity": dmature,
                         "mature_year": dyear}
    # Crop-specific harvest operation dates are used, not a shared corn date.
    out["covers"] = None
    if covers:
        gdd_daily, alive, germination = [], [], []
        weather_mean, _ = grid.calendar_days(grid.var("tavg"))
        weather_min, _ = grid.calendar_days(grid.var("tmin"))
        # The 366th slot in non-leap years can be NaN, not an extreme frost.
        weather_min = torch.where(torch.isfinite(weather_min) & (days < out["year_len"][:, None]),
                                  weather_min, 100.)
        # First killing frost (28 F) after midsummer, and the following winter's
        # minimum. Air temperature is a conservative screen: snow/crown
        # insulation and cultivar differences are explicitly not modelled.
        frost = torch.where((weather_min <= -2.2222) & (days >= 180), days, 366).amin(-1)
        autumn_min = torch.where(days >= 300, weather_min, 100.).amin(-1)
        spring_min = torch.where(days < 90, weather_min, 100.).amin(-1)
        winter_min = autumn_min.clone()
        winter_min[:, :-1] = torch.minimum(autumn_min[:, :-1], spring_min[:, 1:])
        for cover in covers:
            t = numeric_traits(cover.traits)
            # The cover's base-0 establishment/cold-growth rule uses daily
            # mean temperature. Flooring Tmin before averaging with Tmax
            # spuriously generated GDD on freezing-mean winter days.
            g = (grid.var("tavg").clamp(max=t["upper_c"]) - t["base_c"]).clamp_min(0).nan_to_num(0)
            g, _ = grid.calendar_days(g)
            gdd_daily.append(g)
            est = (operation + 1)[..., None]
            warm = (weather_mean[:, None] >= cover.v("germination_min_c")) & (days >= est)
            germination.append(torch.where(warm, days, 366).amin(-1))
            fall = (g[:, None] * ((days >= est) & (days <= DEC_1))).sum(-1)
            enough_time = frost[:, None] - operation - 1 >= cover.v("minimum_frost_free_days")
            if cover.v("minimum_frost_free_days") == 0:
                enough_time = torch.ones_like(operation, dtype=torch.bool)
            # A late-sown rye seedbed can overwinter with little autumn growth.
            # Germination/growth is delayed by cold, not replaced by a 100-GDD death switch.
            alive.append((operation >= 0) & (fall >= cover.v("cover_establishment_gdd")) & enough_time
                         & (winter_min[:, None] >= cover.v("winter_min_c")))
        out["covers"] = {"gdd": torch.stack(gdd_daily, 1), "alive": torch.stack(alive, 1),
                         "germination_day": torch.stack(germination, 1)}
    out["learned"] = None
    from fieldshift.sim.economics import context
    if daily_values is None:
        out["economics"] = [context(s, crops, covers, double_crop, irrigated) for s in sites]
    else:
        context_cache = {id(s): None for s in sites}
        for s in sites:
            if context_cache[id(s)] is None:
                context_cache[id(s)] = context(s, crops, covers, double_crop, irrigated)
        out["economics"] = [context_cache[id(s)] for s in sites]
    out["states"] = [s.get("adm_id", "")[3:5] for s in sites]
    out["backend"] = YieldResponse(not learned)
    if learned and out["backend"].net is not None:
        # The learned model was trained on POWER rain; keep its inputs on POWER.
        model_grid = base
        if any("prec_power" in s["daily"] for s in sites):
            model_grid = DailyGrid([f.assign(prec=f["prec_power"]) if "prec_power" in f else f
                                    for f in base_frames], device)
        out["learned"] = _learned(model_grid, sites, crops, out["backend"], irrigated)
    return out


def _learned(grid, sites, crops, backend, irrigated):
    """Batched deployment features (temperature sowing rule, unedited weather)."""
    rows = []
    ns, ny = len(sites), len(grid.years)
    for crop in crops:
        f = stress_features(grid, crop.traits, sowing_offsets(grid, crop.traits))
        t = numeric_traits(crop.traits)
        columns = []
        for name in backend.columns:
            if name in f:
                columns.append(f[name])
            elif name.startswith("trait__"):
                columns.append(torch.full((ns, ny), float(t.get(name[7:], float("nan"))), device=grid.device))
            elif name == "awc":
                columns.append(torch.tensor([s["soil"]["awc_mm_per_m"] / 10 for s in sites],
                                            device=grid.device)[:, None].expand(ns, ny))
            elif name == "bulk_density":
                columns.append(torch.tensor([s["soil"]["bulk_density_kg_m3"] / 1000 for s in sites],
                                            device=grid.device)[:, None].expand(ns, ny))
            elif name == "irrigated_share":
                columns.append(torch.full((ns, ny), float(irrigated), device=grid.device))
            elif name == "irrigation_missing":
                columns.append(torch.zeros((ns, ny), device=grid.device))
            elif name == "drainage_class":
                columns.append(torch.full((ns, ny), float("nan"), device=grid.device))
            else:
                raise ValueError(f"Deployment feature {name} has no batched source")
        rows.append(torch.stack(columns, -1))
    x = torch.stack(rows, 1)
    mean, sd = backend.batch_matrix(x.reshape(-1, x.shape[-1]))
    return torch.stack([mean.reshape(ns, len(crops), ny), sd.reshape(ns, len(crops), ny)], -1)


def cover_eligible(crops):
    """[C, C] where a winter cover fits between consecutive main crops."""
    ok = []
    for a in crops:
        row = []
        for b in crops:
            row.append(not a.v("perennial") and not a.v("fallow") and not b.v("perennial")
                       and not b.v("fallow") and not b.v("autumn_sown"))
        ok.append(row)
    return np.array(ok, dtype=bool)


def trajectories(rotations, crops, n_years, site=0, spin=4, score=4, starts=None, phases=None):
    """Expand rotations (cycle, cover option, double-crop flag) into trajectories.

    Every cycle phase × every window of ``spin + score`` consecutive years;
    windows end on the last year and step by the cycle length.
    """
    horizon = spin + score
    if starts is None:
        starts = list(range(n_years - horizon, -1, -score))[::-1]
    eligible = cover_eligible(crops)
    autumn = np.array([bool(c.v("autumn_sown")) for c in crops])
    seq, cover, double, year0, rot, position, uniform = [], [], [], [], [], [], []
    for r, (cycle, option, dc) in enumerate(rotations):
        n = len(cycle)
        at_dc = [bool(dc and autumn[cycle[i]]) for i in range(n)]
        at_cover = [option if option and eligible[cycle[i], cycle[(i + 1) % n]] and not at_dc[i] else 0
                    for i in range(n)]
        if phases is not None and phases % n:
            raise ValueError("Shared phase count must be a multiple of every cycle length")
        for phase in range(n if phases is None else phases):
            for start in starts:
                positions = [(phase + t) % n for t in range(-1, horizon + 1)]
                seq.append([cycle[p] for p in positions])
                position.append(positions)
                uniform.append(len(set(cycle)) == 1)
                cover.append([at_cover[p] for p in positions])
                double.append([at_dc[p] for p in positions])
                year0.append(start)
                rot.append(r)
    as_np = lambda x, dtype: np.array(x, dtype=dtype)
    return {"crop": as_np(seq, np.int64), "cover": as_np(cover, np.int64), "double": as_np(double, bool),
            "year0": as_np(year0, np.int64), "rotation": as_np(rot, np.int64),
            "site": np.full(len(rot), site, dtype=np.int64), "horizon": horizon, "spin": spin,
            "position": as_np(position, np.int64), "uniform": as_np(uniform, bool),
            "starts": starts}


@torch.inference_mode()
def water_pass(prepared, soils, crops, covers, double_crop, traj, irrigated=False, residue_return=1.,
               chunk=24000, daily_log=False, continuation=None):
    """Daily bucket for every trajectory; returns per-step tensors on the device."""
    started = time.perf_counter()
    n = len(traj["rotation"])
    parts = [_water_chunk(prepared, soils, crops, covers, double_crop, traj, slice(i, min(i + chunk, n)),
                          irrigated, residue_return, daily_log, continuation) for i in range(0, n, chunk)]
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    if prepared["device"].type == "mps":
        torch.mps.synchronize()
    out["seconds"] = time.perf_counter() - started
    return out


def _water_chunk(prepared, soils, crops, covers, double_crop, traj, part, irrigated, residue_return, daily_log,
                 continuation=None):
    device = prepared["device"]
    tensor = lambda x: torch.as_tensor(x[part], device=device)
    seq, cov, dbl = tensor(traj["crop"]), tensor(traj["cover"]), tensor(traj["double"])
    site, year0 = tensor(traj["site"]), tensor(traj["year0"])
    n, horizon = len(site), traj["horizon"]
    days = torch.arange(366, device=device)
    soil_site = site if continuation is None else continuation["soil_index"][part]
    depth = torch.tensor([s["profile_depth_m"] for s in soils], device=device)[soil_site]
    capacity = torch.tensor([s["awc_mm_per_m"] * s["profile_depth_m"] for s in soils], device=device)[soil_site]
    kc_i, kc_m, kc_e = (_traits(crops, k, device) for k in ("kc_initial", "kc_mid", "kc_end"))
    root, p_crop = _traits(crops, "bucket_root_depth_m", device), _traits(crops, "depletion_fraction", device)
    gap = _traits(crops, "cover_termination_gap_days", device).long()
    # Season-level FAO-33 factor (adopted 2026-11-14 over the stage product on the
    # 300-county odd-year test; validation_weather_tests.json). Stage sums are kept.
    ky = _traits(crops, "ky", device)[:, None]
    bare_kc = .2 * (1 - .5 * residue_return)
    bare_c = .5 - .3 * residue_return
    if covers:
        cv = {k: _traits(covers, k, device) for k in ("kc_initial", "kc_mid", "kc_end", "bucket_root_depth_m",
                                                       "depletion_fraction", "flowering_gdd", "pre_vernalization_gdd",
                                                       "water_productivity_g_m2", "cold_transpiration_gdd")}
        cgdd, calive = prepared["covers"]["gdd"], prepared["covers"]["alive"]
    if double_crop is not None:
        dc_t = {k: torch.tensor(float(double_crop.v(k)), device=device) for k in
                ("kc_initial", "kc_mid", "kc_end", "bucket_root_depth_m", "depletion_fraction")}
        dc_ky = torch.tensor([double_crop.v("ky")], device=device)
    water = capacity.clone()
    carry_a = torch.zeros((n, 4), device=device)
    carry_p = torch.zeros_like(carry_a)
    carry_bio = torch.zeros(n, device=device)
    carry_gdd = torch.zeros(n, device=device)
    if continuation is not None:
        initial = continuation["initial"]
        water = initial["water"][part].clone()
        for name, target in (("carry_a", carry_a), ("carry_p", carry_p),
                             ("carry_bio", carry_bio), ("carry_gdd", carry_gdd)):
            if name in initial:
                target.copy_(initial[name][part])
    names = ("fao_main", "fao_double", "et_mm", "et_total_mm", "irrigation_mm", "drainage_mm", "water_mm",
             "water_at_planting_mm", "r_weighted", "rain_mm", "bare_days", "cover_biomass_t_ha",
             "cover_planted", "cover_established", "double_active", "double_planted", "stand_harvest_fraction")
    out = {k: torch.zeros((n, horizon), device=device) for k in names}
    out["et_month"] = torch.zeros((n, horizon, 12), device=device)
    # Main-crop stage sums (FAO-33 inputs), retained for validation diagnostics.
    out["stage_actual_et"] = torch.zeros((n, horizon, 4), device=device)
    out["stage_potential_et"] = torch.zeros((n, horizon, 4), device=device)
    out["water_month"] = torch.zeros((n, horizon, 12), device=device)
    for name in ("drain_month", "temperature_month", "uptake_month", "cover_growth_month"):
        out[name] = torch.zeros((n, horizon, 12), device=device)
    out["plant_month"] = torch.zeros((n, horizon), dtype=torch.long, device=device)
    out["harvest_month"] = torch.zeros_like(out["plant_month"])
    out["termination_month"] = torch.zeros_like(out["plant_month"])
    if continuation is not None:
        out["water_state"] = torch.zeros((n, horizon, 11), device=device)
        out["month_active"] = torch.zeros((n, horizon, 12), device=device)
        out["water_at_next_sowing_mm"] = torch.full((n, horizon), float("nan"), device=device)
    from fieldshift.sim.alfalfa import schedule
    # Schedule covers the whole trajectory set; keep only this chunk's rows.
    stand = {k: v[part] for k, v in schedule(traj, crops, prepared.get(
        "states", [""] * prepared["weather"].shape[0]), device).items()}
    if daily_log:
        out["daily_et"] = torch.zeros((n, horizon, 366), device=device)
        out["daily_water"] = torch.zeros((n, horizon, 366), device=device)
    for t in range(horizon):
        k = t + 1  # seq columns start one step before the window
        c, cn, cp = seq[:, k], seq[:, k + 1], seq[:, k - 1]
        y = year0 + t
        if continuation is not None and "weather_year" in continuation:
            y = continuation["weather_year"][part, t]
        here = cov[:, k]
        prev = cov[:, k - 1] if t else torch.zeros_like(here)
        dc_here = dbl[:, k]
        main = prepared["main"][site, c, y]
        nxt = prepared["lead"][site, cn, y]
        weather = prepared["weather"][site, y]
        rain, et0 = weather[..., 0], weather[..., 1]
        time_mask = days[None] < prepared["year_len"][y][:, None]
        if continuation is not None:
            if t == 0 and "start_day" in continuation:
                time_mask &= days[None] >= continuation["start_day"][part, None]
            if t == horizon - 1 and "end_day" in continuation:
                time_mask &= days[None] <= continuation["end_day"][part, None]
            rain, et0 = rain * time_mask, et0 * time_mask
        harvest = prepared["harvest"][site, c, y]
        m_main = main[..., 2] > 0
        full_kc = kc_curve(main[..., 1], kc_i[c, None], kc_m[c, None], kc_e[c, None])
        full_potential = (et0 * full_kc * m_main).sum(-1)
        # Fall renewal leaves a full September15→September15 autotoxicity gap.
        # Spring seeding is an explicit April15 scenario (UMN spring window).
        leap = (prepared["year_len"][y] == 366).long()
        fall_day = stand["fall_day"][:, t]+leap
        sow_day = torch.where(stand["fall"][:, t], fall_day, 104 + leap)
        m_main &= (~stand["establish"][:, t, None]) | (days[None] >= sow_day[:, None])
        m_main &= (~stand["terminate"][:, t, None]) | (days[None] < fall_day[:, None])
        short_fraction = (et0 * full_kc * m_main).sum(-1) / full_potential.clamp_min(.001)
        # NRCS Alfalfa Irrigation Facts: approximately linear yield vs ET.
        # This harvest-window proxy is LOW, not a cutting/regrowth model.
        out["stand_harvest_fraction"][:, t] = stand["harvest_fraction"][:, t] * torch.where(
            stand["terminate"][:, t], short_fraction, 1.)
        harvest = torch.where(stand["terminate"][:, t], fall_day-1, harvest)
        if double_crop is not None and prepared["double"] is not None:
            dtab = prepared["double"]["table"][site, c, y]
            m_dc = (dtab[..., 2] > 0) & dc_here[:, None] & ~m_main
            last = torch.maximum(harvest, torch.where(dc_here, prepared["double"]["harvest"][site, c, y], -1))
        else:
            m_dc = torch.zeros_like(m_main)
            last = harvest
        m_lead = (nxt[..., 2] > 0) & (days[None] > last[:, None]) & ~m_main & ~m_dc
        kc = torch.full_like(rain, bare_kc)
        rf = (root[c] / depth).clamp(max=1)[:, None].expand_as(rain).clone()
        p = p_crop[c][:, None].expand_as(rain).clone()
        stage_idx = torch.full_like(rain, 12, dtype=torch.long)
        cfac = torch.full_like(rain, bare_c)
        # RUSLE-style canopy subfactor on the residue-protected bare value; canopy closes over
        # the FAO-56 development stage (progress 0.2-0.5) and stays until maturity.
        crop_c = lambda prog: bare_c * (1 - .9 * ((prog - .2) / .3).clamp(0, 1))
        # Order matters: later assignments win, main crop last.
        kc_lead = kc_curve(nxt[..., 1], kc_i[cn, None], kc_m[cn, None], kc_e[cn, None])
        kc = torch.where(m_lead, kc_lead, kc)
        rf = torch.where(m_lead, (root[cn] / depth).clamp(max=1)[:, None], rf)
        p = torch.where(m_lead, p_crop[cn][:, None], p)
        stage_idx = torch.where(m_lead, 4 + nxt[..., 0].long().clamp(0, 3), stage_idx)
        cfac = torch.where(m_lead, crop_c(nxt[..., 1]), cfac)
        cover_cs = torch.zeros_like(rain)
        cover_ca = torch.zeros_like(rain)
        established = torch.zeros(n, dtype=torch.bool, device=device)
        if covers:
            # Spring part of the cover sown after the previous main crop.
            kp = (prev - 1).clamp_min(0)
            alive_prev = (prev > 0) & (y > 0) & calive[site, kp, cp, (y - 1).clamp_min(0)]
            term = prepared["plant"][site, c, y] - gap[c]
            m_cs = alive_prev[:, None] & (days[None] <= term[:, None]) & ~m_main & ~m_lead
            germinated_fall = prepared["covers"]["germination_day"][site, kp, cp, (y - 1).clamp_min(0)] < 366
            warm_spring = (weather[..., 4] >= _traits(covers, "germination_min_c", device)[kp, None])
            m_cs &= germinated_fall[:, None] | (warm_spring.cumsum(-1) > 0)
            g = cgdd[site, kp, y] * m_cs
            # Canopy grows with GDD toward heading; autumn GDD before vernalization are capped
            # at the parent record's pre-vernalization GDD (winter cereal rule of features.stress).
            progress = (carry_gdd[:, None] + g.cumsum(-1) - g) / cv["flowering_gdd"][kp, None]
            cc = progress.clamp(0, 1)
            kc_cs = bare_kc + (cv["kc_mid"][kp, None] - bare_kc) * cc
            cold = (cgdd[site, kp, y] / cv["cold_transpiration_gdd"][kp, None]).clamp(0, 1)
            cover_cs = cv["water_productivity_g_m2"][kp, None] / 100 * cc * cv["kc_mid"][kp, None] * cold * m_cs
            kc = torch.where(m_cs, kc_cs, kc)
            rf = torch.where(m_cs, (cv["bucket_root_depth_m"][kp] / depth).clamp(max=1)[:, None], rf)
            p = torch.where(m_cs, cv["depletion_fraction"][kp, None], p)
            cfac = torch.where(m_cs, bare_c * (1 - .9 * cc), cfac)
            # Autumn part of the cover sown after this year's main crop.
            kh = (here - 1).clamp_min(0)
            established = (here > 0) & calive[site, kh, c, y]
            sown = prepared["operation"][site, c, y]
            m_ca = established[:, None] & (days[None] > sown[:, None]) & ~m_main & ~m_lead & ~m_dc
            m_ca &= days[None] >= prepared["covers"]["germination_day"][site, kh, c, y, None]
            g = cgdd[site, kh, y] * m_ca
            autumn_gdd = (g.cumsum(-1) - g).clamp(max=cv["pre_vernalization_gdd"][kh, None])
            progress = autumn_gdd / cv["flowering_gdd"][kh, None]
            cc = progress.clamp(0, 1)
            kc_ca = bare_kc + (cv["kc_mid"][kh, None] - bare_kc) * cc
            cold = (cgdd[site, kh, y] / cv["cold_transpiration_gdd"][kh, None]).clamp(0, 1)
            cover_ca = cv["water_productivity_g_m2"][kh, None] / 100 * cc * cv["kc_mid"][kh, None] * cold * m_ca
            kc = torch.where(m_ca, kc_ca, kc)
            rf = torch.where(m_ca, (cv["bucket_root_depth_m"][kh] / depth).clamp(max=1)[:, None], rf)
            p = torch.where(m_ca, cv["depletion_fraction"][kh, None], p)
            cfac = torch.where(m_ca, bare_c * (1 - .9 * cc), cfac)
            out["cover_planted"][:, t] = (here > 0).float()
            out["cover_established"][:, t] = alive_prev.float()
            next_gdd = g.sum(-1).clamp(max=cv["pre_vernalization_gdd"][kh])
            m_cover = m_cs | m_ca
        else:
            m_cover = torch.zeros_like(m_main)
            next_gdd = torch.zeros(n, device=device)
        if double_crop is not None and prepared["double"] is not None:
            kc = torch.where(m_dc, kc_curve(dtab[..., 1], dc_t["kc_initial"], dc_t["kc_mid"], dc_t["kc_end"]), kc)
            rf = torch.where(m_dc, (dc_t["bucket_root_depth_m"] / depth).clamp(max=1)[:, None], rf)
            p = torch.where(m_dc, dc_t["depletion_fraction"], p)
            stage_idx = torch.where(m_dc, 8 + dtab[..., 0].long().clamp(0, 3), stage_idx)
            cfac = torch.where(m_dc, crop_c(dtab[..., 1]), cfac)
        kc = torch.where(m_main, kc_curve(main[..., 1], kc_i[c, None], kc_m[c, None], kc_e[c, None]), kc)
        rf = torch.where(m_main, (root[c] / depth).clamp(max=1)[:, None], rf)
        p = torch.where(m_main, p_crop[c][:, None], p)
        stage_idx = torch.where(m_main, main[..., 0].long().clamp(0, 3), stage_idx)
        cfac = torch.where(m_main, crop_c(main[..., 1]), cfac)
        cash = m_main | m_dc | m_lead
        if continuation is not None:
            kc = kc * continuation["kc_scale"][part, None]
            rf = (rf * continuation["root_scale"][part, None]).clamp(max=1)
        out["r_weighted"][:, t] = (rain * cfac).sum(-1)
        out["rain_mm"][:, t] = rain.sum(-1)
        valid_day = days[None] < prepared["year_len"][y][:, None]
        out["bare_days"][:, t] = (~(cash | m_cover) & (et0 > 0) & valid_day).float().sum(-1)
        plant_day = prepared["plant"][site, c, y]
        if continuation is not None:
            next_plant_day = torch.where(nxt[..., 2] > 0, days, 999).amin(-1)
        plant_day = torch.where(stand["establish"][:, t], sow_day, plant_day)
        # A water-use curve is not a corn N-uptake curve. Bender: 2/3 by VT/R1.
        # Smooth interpolation has no fitted parameters and remains a process prior.
        nf = _traits(crops, "n_fraction_at_flowering", device)[c]
        flowering_progress = (_traits(crops, "flowering_gdd", device)[c]
                              / _traits(crops, "maturity_gdd", device)[c]).clamp(.05, .95)
        progress = main[..., 1].clamp(0, 1)
        u = (progress / flowering_progress[:, None]).clamp(0, 1)
        v = ((progress - flowering_progress[:, None]) / (1 - flowering_progress[:, None])).clamp(0, 1)
        n_cumulative = nf[:, None] * (3 * u.square() - 2 * u.pow(3)) + (1 - nf[:, None]) * (3 * v.square() - 2 * v.pow(3))
        n_increment = (n_cumulative - torch.nn.functional.pad(n_cumulative[:, :-1], (1, 0))).clamp_min(0) * m_main
        out["plant_month"][:, t] = torch.as_tensor(MONTH_OF_DAY, device=device)[plant_day.clamp(0, 365)]
        out["harvest_month"][:, t] = torch.as_tensor(MONTH_OF_DAY, device=device)[harvest.clamp(0, 365)]
        term_day = (plant_day - gap[c]).clamp(0, 365)
        out["termination_month"][:, t] = torch.as_tensor(MONTH_OF_DAY, device=device)[term_day]
        acc_a = torch.zeros((n, 13), device=device)
        acc_p = torch.zeros_like(acc_a)
        bio_cs = torch.zeros(n, device=device)
        bio_ca = torch.zeros(n, device=device)
        et_cash = torch.zeros(n, device=device)
        et_all = torch.zeros(n, device=device)
        irr = torch.zeros(n, device=device)
        limits = torch.tensor([s.get("irrigation_capacity_mm_yr", float("inf")) for s in soils],
                              device=device)[soil_site]
        drain = torch.zeros(n, device=device)
        at_planting = torch.full((n,), float("nan"), device=device)
        for d in range(366):
            r = rf[:, d]
            accessible = water * r
            new_root, flows = water_step(accessible, rain[:, d], et0[:, d], kc[:, d], capacity * r, p[:, d],
                                         irrigated, cash[:, d], limits-irr)
            lower = water - accessible + flows["drainage_mm"]
            deep = (lower - capacity * (1 - r)).clamp_min(0)
            water = new_root + lower - deep
            if continuation is not None and continuation.get("observe") is not None:
                water = continuation["observe"](t, d, water, capacity, part)
            drain += deep
            month = MONTH_OF_DAY[d]
            out["drain_month"][:, t, month] += deep
            # Q10=2 seasonal activity, reference 20 C. This is a labelled
            # process prior, not fitted to nitrate-loss observations.
            out["temperature_month"][:, t, month] += (
                2 ** ((weather[:, d, 4] - 20) / 10)).clamp(max=4) / 365 * (
                    time_mask[:, d] if continuation is not None else 1)
            if continuation is not None:
                out["month_active"][:, t, month] = torch.maximum(
                    out["month_active"][:, t, month], time_mask[:, d].float())
            out["uptake_month"][:, t, month] += torch.where(
                nf >= 0, n_increment[:, d], flows["potential_et_mm"] * cash[:, d])
            et = flows["et_mm"]
            et_all += et
            et_cash += et * cash[:, d]
            irr += flows["irrigation_mm"]
            idx = stage_idx[:, d:d + 1]
            acc_a.scatter_add_(1, idx, et[:, None])
            acc_p.scatter_add_(1, idx, flows["potential_et_mm"][:, None])
            if covers:
                bio_cs += flows["ks"] * cover_cs[:, d]
                bio_ca += flows["ks"] * cover_ca[:, d]
                out["cover_growth_month"][:, t, month] += flows["ks"] * (cover_cs[:, d] + cover_ca[:, d])
            out["et_month"][:, t, MONTH_OF_DAY[d]] += et
            at_planting = torch.where(plant_day == d, water, at_planting)
            if continuation is not None:
                out["water_at_next_sowing_mm"][:, t] = torch.where(
                    next_plant_day == d, water, out["water_at_next_sowing_mm"][:, t])
            if d in TRACE_DAYS:
                out["water_month"][:, t, TRACE_DAYS[d]] = water
            if daily_log:
                out["daily_et"][:, t, d] = et
                out["daily_water"][:, t, d] = water
        actual = acc_a[:, :4] + carry_a
        potential = acc_p[:, :4] + carry_p
        carry_a, carry_p = acc_a[:, 4:8], acc_p[:, 4:8]
        ratio, _ = fao_relative_yield({"actual_et": actual.sum(-1, keepdim=True),
                                       "potential_et": potential.sum(-1, keepdim=True), "ky": ky[c],
                                      "flowering_heat_dd": prepared["stage_heat"][site, c, y],
                                      "reproductive_frost_days": prepared["stage_frost"][site, c, y]}, {})
        out["fao_main"][:, t] = ratio
        out["stage_actual_et"][:, t], out["stage_potential_et"][:, t] = actual, potential
        if double_crop is not None and prepared["double"] is not None:
            dr, _ = fao_relative_yield({"actual_et": acc_a[:, 8:12].sum(-1, keepdim=True),
                                        "potential_et": acc_p[:, 8:12].sum(-1, keepdim=True),
                                       "ky": dc_ky.expand(n, -1),
                                       "flowering_heat_dd": prepared["double"]["heat"][site, c, y],
                                       "reproductive_frost_days": prepared["double"]["frost"][site, c, y]}, {})
            out["fao_double"][:, t] = torch.where(dc_here, dr, 0)
            planted = dc_here & (prepared["double"]["harvest"][site, c, y] >= 0)
            out["double_planted"][:, t] = planted.float()
            # A double crop that the first autumn frost kills before maturity yields nothing.
            out["double_active"][:, t] = (planted & prepared["double"]["mature_year"][site, c, y]).float()
        out["cover_biomass_t_ha"][:, t] = carry_bio + bio_cs
        carry_bio, carry_gdd = bio_ca, next_gdd
        if continuation is not None:
            out["water_state"][:, t] = torch.cat(
                (water[:, None], carry_a, carry_p, carry_bio[:, None], carry_gdd[:, None]), -1)
        out["et_mm"][:, t], out["et_total_mm"][:, t] = et_cash, et_all
        out["irrigation_mm"][:, t], out["drainage_mm"][:, t] = irr, drain
        out["water_mm"][:, t] = water
        out["water_at_planting_mm"][:, t] = at_planting
    return out


@torch.inference_mode()
def ledger_pass(prepared, water, soils, normals, crops, covers, double_crop, traj, irrigated=False,
                 residue_return=1., fao_reference=None, base_fao=None, nitrogen_log=False, continuation=None):
    """Annual yield, pest, nitrogen, carbon, erosion and money ledgers.

    ``fao_reference`` [S, C] (+ double crop [S]) normalises FAO-33 ratios when no
    learned model is available. ``base_fao`` (the same trajectories' FAO ratios
    under unedited weather) turns edited-weather ratios into a multiplier on
    the baseline yield, so a zero edit reproduces the baseline exactly.
    """
    device = prepared["device"]
    tensor = lambda x: torch.as_tensor(x, device=device)
    seq, cov = tensor(traj["crop"]), tensor(traj["cover"])
    site, year0 = tensor(traj["site"]), tensor(traj["year0"])
    position, uniform = tensor(traj["position"]), tensor(traj["uniform"])
    n, horizon = len(site), traj["horizon"]
    tr = lambda name: _traits(crops, name, device)
    soil_site = site if continuation is None else continuation["soil_index"]
    soil_mass = torch.tensor([s["bulk_density_kg_m3"] * .3 * 10 for s in soils], device=device)[soil_site]
    om0 = torch.tensor([s["organic_matter_pct"] for s in soils], device=device)[soil_site]
    carbon = soil_mass * om0 / 100 / 1.724
    capacity = torch.tensor([s["awc_mm_per_m"] * s["profile_depth_m"] for s in soils], device=device)[soil_site]
    k_soil = torch.tensor([s["rusle_k"] for s in soils], device=device)[soil_site]
    slope = torch.tensor([s["slope_pct"] for s in soils], device=device)[soil_site]
    length = torch.tensor([s["slope_length_m"] for s in soils], device=device)[soil_site]
    normal = torch.as_tensor(np.asarray(normals, dtype=np.float32), device=device)
    mineral = torch.full((n,), 30., device=device)
    organic = carbon * 1000 / 12
    fast, fast_rate_mass, live_cover = (torch.zeros(n, device=device) for _ in range(3))
    pressure = torch.zeros((n, 8), device=device)
    last = torch.full((n,), -1, dtype=torch.long, device=device)
    streak = torch.zeros(n, device=device)
    cumulative_erosion = torch.zeros(n, device=device)
    if continuation is not None:
        initial = continuation["initial"]
        carbon, mineral, organic, fast, fast_rate_mass, live_cover, pressure, last, streak, cumulative_erosion = (
            initial[k].clone() for k in ("carbon", "mineral", "organic", "fast", "fast_rate_mass",
                                        "live_cover", "pressure", "last", "streak", "erosion"))
    family = tr("family").long()
    # Davis 2012 is a whole-system experiment, not a universal causal coefficient.
    # Transfer the published yield differences as LOW-confidence scenarios only
    # to >=3-species cycles containing a small grain or perennial legume.
    sequence = seq[:, 1:5]
    diversity = torch.nn.functional.one_hot(sequence, len(crops)).any(1).sum(-1) >= 3
    diversity &= ((tr("small_grain")[sequence] > 0)
                  | ((tr("perennial")[sequence] > 0) & (tr("legume")[sequence] > 0))).any(-1)
    if covers:
        cvt = {k: _traits(covers, k, device) for k in ("residue_cn", "shoot_root_ratio", "extra_root_fraction",
                                                       "fixed_n_fraction", "cost_usd_ha", "cover_yield_effect")}
    out = {k: torch.zeros((n, horizon), device=device) for k in LEDGER_COLUMNS}
    if continuation is not None:
        out["ledger_state"] = torch.zeros((n, horizon, 18), device=device)
    n_log = [] if nitrogen_log else None
    if fao_reference is not None:
        fao_reference = tensor(fao_reference)
    from fieldshift.sim.alfalfa import schedule
    stand = schedule(traj, crops, prepared.get("states", [""] * prepared["weather"].shape[0]), device)
    for t in range(horizon):
        k = t + 1
        c, y = seq[:, k], year0 + t
        if continuation is not None and "weather_year" in continuation:
            y = continuation["weather_year"][:, t]
        here = cov[:, k]
        prev = cov[:, k - 1] if t else torch.zeros_like(here)
        fao = water["fao_main"][:, t]
        if prepared["learned"] is not None:
            base_ratio = prepared["learned"][site, c, y, 0]
            sd = prepared["learned"][site, c, y, 1]
        else:
            base_ratio = fao / fao_reference[site, c].clamp_min(.05)
            sd = torch.full_like(fao, float("nan"))
        if base_fao is not None:
            base_ratio = base_ratio * (fao / base_fao["fao_main"][:, t].clamp_min(.05)).clamp(0, 2)
        cover_done = water["cover_established"][:, t] > 0
        if covers:
            kp = (prev - 1).clamp_min(0)
            base_ratio = base_ratio * (1 + cvt["cover_yield_effect"][kp] * cover_done)
        previous_streak = streak
        streak = torch.where(last == c, streak + 1, 0)
        # Alfalfa has an explicit establishment event and harvest loss. Other
        # forage scenarios retain their existing ledger treatment.
        renew = ((uniform & (position[:, k] == 0) & (tr("perennial")[c] > 0) & ~stand["alfalfa"][:, t])
                 | stand["establish"][:, t])
        streak = torch.where(renew, 0, streak)
        fam = family[c]
        prior = pressure.gather(1, fam[:, None]).squeeze(1)
        penalty = pest_penalty(streak, tr("same_crop_penalty")[c], tr("penalty_years_to_plateau")[c],
                                torch.where(last == c, 0, prior), tr("family_penalty")[c])
        # Autotoxicity concerns RESEEDING; do not impose a generic annual-crop
        # same-crop penalty on an intact established alfalfa stand.
        penalty = torch.where(stand["alfalfa"][:, t], 0., penalty)
        ratio = base_ratio * (1 - penalty) * (1 + tr("diverse_rotation_yield_gain")[c] * diversity)
        ratio *= water["stand_harvest_fraction"][:, t]
        if continuation is not None and t == 0 and "start_day" in continuation:
            ratio *= prepared["harvest"][site, c, y] >= continuation["start_day"]
        fallow = tr("fallow")[c]
        harvested = normal[site, c] * ratio * (1 - fallow)
        above, below = plant_carbon(harvested, tr("dry_matter_fraction")[c], tr("harvest_index")[c],
                                    tr("shoot_root_ratio")[c], tr("extra_root_fraction")[c], residue_return)
        perennial = tr("perennial")[c] > 0
        # Bolinder 2007: perennial root C enters in the establishment year, extra-root C every year.
        extra = tr("extra_root_fraction")[c]
        # Move the one-time root pulse to first PRODUCTION after a no-hay fall
        # seeding year; continuing stands retain the existing extra-root input.
        first_production = stand["alfalfa"][:, t] & stand["fall"][:, t] & (streak == 1)
        below = torch.where(perennial & (streak > 0) & ~first_production, below * extra / (1 + extra), below)
        residue_n = (above + below) * 1000 / tr("residue_cn")[c]
        fixed = tr("fixed_n_kg_ha")[c] * ratio
        harvest_n = harvested * tr("harvest_n_kg_t")[c]
        # Mass-balance N rate: N removed in grain or hay at the county normal yield.
        previous_crop = seq[:, k - 1]
        legume_credit = tr("previous_legume_credit_kg_ha")[c] * (tr("legume")[previous_crop] > 0)
        recommendation = tr("recommended_n_kg_ha")[c] * (1 - fallow)
        fertilizer = (recommendation - legume_credit).clamp_min(0)
        after_forage = ((tr("perennial")[previous_crop] > 0) & (tr("legume")[previous_crop] > 0)
                        & (tr("after_perennial_legume_n_kg_ha")[c] > 0))
        # The existing 30lb/ac corn TARGET already implies a 151lb/ac credit
        # against continuous-corn181, not an extra 151lb of N entering soil.
        # UMN: one-year stands supply only30–60lb/ac; midpoint45 is a LOW scenario.
        forage_target = torch.where(previous_streak >= 1, tr("after_perennial_legume_n_kg_ha")[c],
                                    (recommendation - 45 * 1.12085).clamp_min(0))
        fertilizer = torch.where(after_forage, torch.minimum(recommendation, forage_target), fertilizer)
        # TX D3 establishment specifies40lb/ac starter N, charged once. Starter
        # N included in phosphate blends elsewhere remains in the source bundle.
        from fieldshift.sim.alfalfa import PRODUCTION_N, STARTER_N
        states = prepared.get("states", [""] * prepared["weather"].shape[0])
        starter = torch.tensor([STARTER_N.get(s, 0.)*1.12085 for s in states], device=device)[site]
        production_n = torch.tensor([PRODUCTION_N.get(s, 0.)*1.12085 for s in states], device=device)[site]
        fertilizer = torch.where(stand["alfalfa"][:, t],
                                 torch.where(stand["establish"][:, t], starter, production_n), fertilizer)
        # Corn cost already assumes corn after soybean; only the additional
        # reduction beyond that rate saves money, never count the soy credit twice.
        forage_saving = torch.where(after_forage, (recommendation - tr("previous_legume_credit_kg_ha")[c]
                                                  - fertilizer).clamp_min(0), 0)
        income = (harvested * tr("price_usd_t")[c] - tr("cost_usd_ha")[c] * (1 - fallow)
                  - tr("same_crop_extra_cost_usd_ha")[c] * (streak > 0))
        dc_t = torch.zeros(n, device=device)
        if double_crop is not None and prepared["double"] is not None:
            on = water["double_active"][:, t] > 0
            soy = normals_double(normals, crops, double_crop)
            dref = (fao_reference[:, -1][site] if fao_reference is not None and fao_reference.shape[1] > len(crops)
                    else torch.ones(n, device=device))
            d_ratio = water["fao_double"][:, t] / dref.clamp_min(.05)
            dc_t = torch.where(on, torch.as_tensor(soy, dtype=torch.float32, device=device)[site] * double_crop.v("double_crop_factor") * d_ratio.clamp(0, 1.5), 0)
            a2, b2 = plant_carbon(dc_t, torch.tensor(double_crop.v("dry_matter_fraction"), device=device),
                                  torch.tensor(double_crop.v("harvest_index"), device=device),
                                  torch.tensor(double_crop.v("shoot_root_ratio"), device=device),
                                  torch.tensor(double_crop.v("extra_root_fraction"), device=device), residue_return)
            above, below = above + a2, below + b2
            n2 = (a2 + b2) * 1000 / double_crop.v("residue_cn")
            residue_n = residue_n + n2
            fixed = fixed + double_crop.v("fixed_n_kg_ha") * torch.where(on, d_ratio.clamp(0, 1.5) *
                                                                       double_crop.v("double_crop_factor"), 0)
            harvest_n = harvest_n + dc_t * double_crop.v("harvest_n_kg_t")
            planted = water["double_planted"][:, t] > 0
            income = income + dc_t * double_crop.v("price_usd_t") - double_crop.v("cost_usd_ha") * planted
            pressure.scatter_add_(1, torch.full_like(fam, int(double_crop.v("family")))[:, None],
                                  on.float()[:, None])
        cover_bio = water["cover_biomass_t_ha"][:, t] * cover_done
        cover_cost = torch.zeros(n, device=device)
        cover_credit = torch.zeros_like(fertilizer)
        cover_cn = torch.full_like(fertilizer, 30)
        cover_root_factor = torch.ones_like(fertilizer)
        cover_fixed_fraction = torch.zeros_like(fertilizer)
        if covers:
            kh = (here - 1).clamp_min(0)
            ca = cover_bio * .45
            cb = cover_bio / cvt["shoot_root_ratio"][kp] * .45 * (1 + cvt["extra_root_fraction"][kp])
            above, below = above + ca, below + cb
            # SARE available N ~half top-growth N, bounded by UNL's observed
            # 88 lb/ac credit. No fertilizer credit to a non-fertilized legume.
            cover_credit = torch.minimum(fertilizer, torch.minimum(
                cover_bio * .45 * 1000 / cvt["residue_cn"][kp] * .5 * (cvt["fixed_n_fraction"][kp] > 0),
                torch.full_like(fertilizer, 88 * 1.12085)))
            fertilizer = fertilizer - cover_credit
            cover_cn = cvt["residue_cn"][kh]
            cover_root_factor = 1 + (1 + cvt["extra_root_fraction"][kh]) / cvt["shoot_root_ratio"][kh]
            cover_fixed_fraction = cvt["fixed_n_fraction"][kh]
            cover_cost = cvt["cost_usd_ha"][kh] * (here > 0)
            income = income - cover_cost
        # Cover N is handled in its own living store and returned at termination,
        # not counted a second time as cash-crop residue demand.
        rate = -torch.log1p(-(12.5 / tr("residue_cn")[c]).clamp(.1, .8))
        state, flows = monthly_nitrogen((mineral, organic, fast, fast_rate_mass, live_cover), water, t,
                                        fertilizer, fixed, harvest_n, residue_n, rate,
                                         cover_cn, cover_root_factor, cover_fixed_fraction, capacity,
                                          diagnostics=nitrogen_log, store_state=continuation is not None)
        if nitrogen_log:
            n_log.append(flows["monthly"])
        mineral, organic, fast, fast_rate_mass, live_cover = state
        returned = flows["residue_return"]
        credit = fast * (1 - torch.exp(-fast_rate_mass / fast.clamp_min(1e-6)))
        saving = (cover_credit + forage_saving) * tr("fertilizer_price_usd_kg")[c]
        income = income + saving
        carbon, cflows = carbon_step(carbon, above, below)
        rain = water["rain_mm"][:, t]
        erosion = rusle(renard_freimund_r(rain), k_soil, slope, length, water["r_weighted"][:, t] / rain.clamp_min(1))
        pressure *= .5
        pressure.scatter_add_(1, fam[:, None], (1 - fallow)[:, None])
        last = c
        if continuation is not None:
            cumulative_erosion += erosion
            out["ledger_state"][:, t] = torch.cat(
                (torch.stack((carbon, mineral, organic, fast, fast_rate_mass, live_cover,
                              last.float(), streak, cumulative_erosion,
                              fast * tr("residue_cn")[c] / 1000), -1), pressure), -1)
        values = {
            "income": income, "income_legacy": income, "economic_crop_index": c.float(),
            "economic_cover_index": here.float(), "economic_double_planted": water["double_planted"][:, t],
            "economic_same_crop": (streak > 0).float(),
            "economic_establishment": stand["establish"][:, t].float(),
            "economic_establishment_fall": stand["fall"][:, t].float(),
            "yield_relative": ratio, "et_mm": water["et_mm"][:, t],
            "et_total_mm": water["et_total_mm"][:, t], "irrigation_mm": water["irrigation_mm"][:, t],
            "water_mm": water["water_mm"][:, t], "water_at_planting_mm": water["water_at_planting_mm"][:, t],
            "mineral_n_kg_ha": mineral, "organic_matter_pct": carbon * 1.724 / soil_mass * 100,
            "soil_carbon_input_t_ha": cflows["retained_c"], "erosion_t_ha": erosion,
            "bare_days": water["bare_days"][:, t], "forage": tr("forage_value")[c] * (water["stand_harvest_fraction"][:, t] > 0),
            "n_fixed_kg_ha": fixed + flows["cover_fixed"], "n_shortage_kg_ha": flows["shortage"], "n_leached_kg_ha": flows["leaching"],
            "drainage_mm": water["drainage_mm"][:, t], "pest_loss_pct": penalty * 100, "predictive_sd": sd,
            "n_credit_kg_ha": credit, "cover_biomass_t_ha": cover_bio,
            "cover_planted": water["cover_planted"][:, t], "cover_established": cover_done.float(),
            "cover_cost_usd_ha": cover_cost, "double_crop_t_ha": dc_t, "harvest_t_ha": harvested,
            "n_fertilizer_kg_ha": fertilizer, "n_mineralized_kg_ha": flows["mineralization"],
            "n_harvest_removed_kg_ha": flows["harvest_removal"], "n_residue_return_kg_ha": returned,
            "n_cover_uptake_kg_ha": flows["cover_uptake"],
            "n_fertilizer_saving_usd_ha": saving, "n_balance_error_kg_ha": flows["balance_error"],
        }
        for name, value in values.items():
            out[name][:, t] = value
    if nitrogen_log:
        out["nitrogen_monthly"] = {k: torch.stack([v[k] for v in n_log], 1) for k in n_log[0]}
    from fieldshift.sim.economics import reprice
    if "economics" in prepared:
        groups = ((None, continuation["economics"]),) if continuation is not None and "economics" in continuation else enumerate(prepared["economics"])
        for i, context in groups:
            mask = (torch.ones(n, device=device, dtype=torch.bool) if i is None else
                    torch.as_tensor(traj["site"], device=device) == i)
            fields = {k: v[mask] for k, v in out.items() if isinstance(v, torch.Tensor)}
            out["income"][mask], out["return_over_listed_costs_usd_ha"][mask], out["irrigation_cost_usd_ha"][mask], _ = reprice(fields, context)
            ix = fields["economic_crop_index"].long()
            unit = torch.tensor([c["n_price"]["value"] for c in context["crops"]], device=device)[ix]
            prior = torch.tensor([c["default_n_price"] for c in context["crops"]], device=device)[ix]
            out["n_fertilizer_saving_usd_ha"][mask] *= unit/prior.clamp_min(1e-8)
    return out


def normals_double(normals, crops, double_crop):
    """Double-crop soybean uses the site's soybean normal, else the trait reference."""
    keys = [c.key for c in crops]
    normals = np.asarray(normals)
    if "soybean" in keys:
        return normals[:, keys.index("soybean")]
    return np.full(normals.shape[0], double_crop.v("normal_t_ha"))


def fao_reference(water, traj, n_sites, n_crops, spin):
    """Mean scored FAO-33 ratio per site and crop (+ double crop as the last column)."""
    device = water["fao_main"].device
    seq = torch.as_tensor(traj["crop"][:, 1:-1], device=device)
    site = torch.as_tensor(traj["site"], device=device)[:, None].expand_as(seq)
    scored = slice(spin, None)
    total = torch.zeros((n_sites, n_crops + 1), device=device)
    count = torch.zeros_like(total)
    flat = (site[:, scored] * (n_crops + 1) + seq[:, scored]).reshape(-1)
    total.view(-1).index_add_(0, flat, water["fao_main"][:, scored].reshape(-1))
    count.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    on = water["double_active"][:, scored].reshape(-1)
    dflat = site[:, scored].reshape(-1) * (n_crops + 1) + n_crops
    total.view(-1).index_add_(0, dflat, water["fao_double"][:, scored].reshape(-1) * on)
    count.view(-1).index_add_(0, dflat, on)
    return torch.where(count > 0, total / count.clamp_min(1), 1.)
