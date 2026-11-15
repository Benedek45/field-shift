"""Benchmark C: held-out, state-initialised hindcasts. No parameter is fitted.

Yield: for each rainfed county, crop and target year 2003-2024 the water state
is taken from a continuous 1981+ replay as of 31 October of the previous year
(SMAP L4 root-zone EnSRF from 2015-03-30; model-only before). Forecasts from
that state run 1 November -> harvest with (i) every other historical weather
year (1982-2025 minus the target) and (ii) the actual weather. The same runs
from field capacity show what the initial state adds. Scores are against NASS
trend-relative yields, with the trend fitted only on years before the target.

Soil water: from the analysis at the first SMAP step of each month
(2015-04..2025-09), open-loop runs with actual weather predict the next 1-12
eight-day SMAP root-zone values, compared with persistence and climatology.
"""
import calendar
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fieldshift.planner.data import _static, load_light
from fieldshift.planner.validate import json_safe, selected_labels
from fieldshift.sim.components import accelerator
from fieldshift.sim.engine import prepare, water_pass
from fieldshift.sim.state import OBSERVATION_ERROR, ROOT, design
from fieldshift.sim.traits import CATALOG

MEMBERS = 32
TARGETS = tuple(range(2003, 2025))
ASSIMILATED_FROM = 2016  # first target whose 31 Oct state follows SMAP updates
LEADS = 12
CROPS = ("corn", "soybean")


def grouped_ensrf(state, predicted, observation, error_sd):
    """``state.kalman_update`` for one scalar state per member, vectorised over
    groups: state/predicted [G, M], observation [G]. Same EnSRF algebra."""
    m = state.shape[1]
    mean, hmean = state.mean(1, keepdim=True), predicted.mean(1, keepdim=True)
    a, ha = state - mean, predicted - hmean
    total = ha.square().sum(1, keepdim=True) / (m - 1) + error_sd ** 2
    gain = (a * ha).sum(1, keepdim=True) / (m - 1) / total.clamp_min(1e-12)
    alpha = 1 / (1 + torch.sqrt(error_sd ** 2 / total))
    return mean + gain * (observation[:, None] - hmean) + a - alpha * ha * gain


def crps(ensemble, mask, observed):
    """Ensemble CRPS, E|X-y| - E|X-X'|/2, rows padded with ``mask``."""
    n = mask.sum(-1).clamp_min(1)
    first = torch.where(mask, (ensemble - observed[:, None]).abs(), 0).sum(-1) / n
    pair = (ensemble[:, :, None] - ensemble[:, None, :]).abs() * (mask[:, :, None] & mask[:, None, :])
    return first - pair.sum((-1, -2)) / (2 * n * n)


def _smap(ids):
    path = Path("data/processed/eo/eo_county_8day.f32")
    meta = json.loads(path.with_suffix(".json").read_text())
    names = [v["name"] if isinstance(v, dict) else v for v in meta["variables"]]
    cube = np.memmap(path, dtype="<f4", mode="r", shape=tuple(meta["shape"]))
    rows = [meta["counties"].index(a) for a in ids]
    return np.asarray(cube[rows, :, names.index("smap_rootzone")]), pd.to_datetime(meta["step_start_dates"])


def _steps(dates, years):
    """Step end dates (8-day composites truncated at 31 Dec) and their (year, day) slots."""
    ends = np.minimum(dates + pd.Timedelta(days=7), pd.to_datetime([f"{y}-12-31" for y in dates.year]))
    ends = pd.DatetimeIndex(ends)
    inside = np.isin(dates.year, years)
    slot = np.where(inside, np.searchsorted(years, dates.year), -1)
    return ends, slot, ends.day_of_year.to_numpy() - 1


def _frozen(sites, dates, ends):
    """True where the 8-day mean air temperature is <= 0 C (coarse frozen-ground proxy)."""
    out = np.ones((len(sites), len(dates)), dtype=bool)
    for i, s in enumerate(sites):
        d = s["daily"]
        cum = np.concatenate([[0.], np.cumsum(d.tavg.to_numpy(np.float64))])
        a = np.searchsorted(d.date.to_numpy(), dates.to_numpy())
        b = np.searchsorted(d.date.to_numpy(), ends.to_numpy(), side="right")
        ok = b > a
        out[i, ok] = (cum[b[ok]] - cum[a[ok]]) / (b[ok] - a[ok]) <= 0
    return out


class Observer:
    """Grouped SMAP EnSRF on profile water plus state snapshots.

    Trajectories are ordered [site, phase, member]; each (site, phase) is one
    ensemble. Snapshots: water after 31 October each year and after every SMAP
    step end (post-analysis), for later restarts."""

    def __init__(self, obs, slot, day, year_lengths, depth, wilting, n_years, device):
        self.obs, self.depth, self.wilting = obs, depth, wilting
        self.year_lengths = year_lengths
        self.lookup = {(int(t), int(d)): k for k, (t, d) in enumerate(zip(slot, day, strict=True)) if t >= 0}
        n = len(wilting)
        self.october = torch.zeros((n, n_years), device=device)
        self.analysis = torch.full((n, obs.shape[1]), float("nan"), device=device)
        self.total, self.count, self.log = None, 0, []

    def __call__(self, t, d, water, capacity, part):
        if part.start or part.stop < len(self.wilting):
            raise ValueError("Grouped EnSRF needs every member in one chunk")
        if d >= self.year_lengths[t]:
            return water
        self.total = water.clone() if self.total is None else self.total + water
        self.count += 1
        k = self.lookup.get((t, d))
        if k is not None:
            theta = self.obs[:, k]
            good = torch.isfinite(theta)
            forecast = self.total / self.count / self.depth + self.wilting * 1000
            g = (len(water) // MEMBERS, MEMBERS)
            sd = OBSERVATION_ERROR["sd_m3_m3"] * 1000
            updated = grouped_ensrf(water.view(g), forecast.view(g), torch.nan_to_num(theta) * 1000, sd)
            updated = torch.minimum(updated.reshape(-1).clamp_min(0), capacity)
            mask = good.repeat_interleave(MEMBERS)
            if bool(good.any()):
                prior = forecast.view(g).mean(1)
                self.log.append(torch.stack((torch.where(good, theta * 1000 - prior, torch.nan),
                                             torch.where(good, forecast.view(g).std(1), torch.nan),
                                             torch.where(good, water.view(g).std(1) / self.depth.view(g)[:, 0], torch.nan),
                                             torch.where(good, updated.view(g).std(1) / self.depth.view(g)[:, 0], torch.nan))))
            water = torch.where(mask, updated, water)
            self.analysis[:, k] = water
            self.total, self.count = None, 0
        if d == 303 + (self.year_lengths[t] == 366):
            self.october[:, t] = water
        return water


class Accumulator:
    """Open-loop 8-day means of modelled 0-1 m water content for restarts."""

    def __init__(self, start, length0, bounds, depth, wilting, year_len):
        self.start, self.length0, self.bounds = start, length0, bounds
        self.depth, self.wilting, self.year_len = depth, wilting, year_len
        self.sums = torch.zeros((len(start), LEADS), device=start.device)
        self.counts = torch.zeros_like(self.sums)

    def __call__(self, t, d, water, capacity, part):
        offset = (d - self.start[part]) if t == 0 else (self.length0[part] - self.start[part] + d)
        valid = (offset >= 0) & (d < self.year_len[part, t])
        lead = (offset[:, None] >= self.bounds[part, 1:]).sum(-1)
        valid &= lead < LEADS
        theta = water / self.depth[part] + self.wilting[part] * 1000
        index = lead.clamp(max=LEADS - 1)
        self.sums[part].scatter_add_(1, index[:, None], (theta * valid)[:, None])
        self.counts[part].scatter_add_(1, index[:, None], valid.float()[:, None])
        return water


def _trajectory(crop, weather_years=None):
    n, cols = crop.shape
    out = {"crop": crop, "cover": np.zeros((n, cols), dtype=np.int64), "double": np.zeros((n, cols), dtype=bool),
           "site": np.zeros(n, dtype=np.int64), "year0": np.zeros(n, dtype=np.int64),
           "rotation": np.zeros(n, dtype=np.int64), "position": np.tile(np.arange(cols) % 2, (n, 1)),
           "uniform": np.zeros(n, dtype=bool), "horizon": cols - 2, "spin": 0}
    return out


def _phase_crop(phase, year):
    """Phase 0 grows corn in even years, phase 1 in odd years (corn-soybean rotation)."""
    return np.where((np.asarray(year) % 2 == 0) == (np.asarray(phase) == 0), 0, 1)


@torch.inference_mode()
def batch(ids, labels, device):
    started = time.perf_counter()
    sites = [load_light(a) for a in ids]
    crops = [CATALOG[k] for k in CROPS]
    prepared = prepare(sites, crops, learned=False)
    years = np.asarray(prepared["years"])
    year_lengths = [366 if calendar.isleap(int(y)) else 365 for y in years]
    year_len = torch.as_tensor(year_lengths, device=device)
    values = design(MEMBERS, device)
    scales = torch.stack((values["awc_scale"], values["initial_available_water_fraction"]), -1).tolist()
    soils = [dict(s["soil"], awc_mm_per_m=s["soil"]["awc_mm_per_m"] * a) for s in sites for a, _ in scales]
    static = _static()
    wilting_site = torch.tensor([float(static.loc[a].wilting_point_0_100) for a in ids], device=device)
    capacity = torch.tensor([s["awc_mm_per_m"] * s["profile_depth_m"] for s in soils], device=device)
    depth_soil = torch.tensor([s["profile_depth_m"] for s in soils], device=device)
    S, Y = len(ids), len(years)
    # Replay trajectories [site, phase, member]; soils [site, member].
    site = np.repeat(np.arange(S), 2 * MEMBERS)
    phase = np.tile(np.repeat([0, 1], MEMBERS), S)
    member = np.tile(np.arange(MEMBERS), 2 * S)
    cols = np.arange(years[0] - 1, years[-1] + 2)
    replay_traj = _trajectory(_phase_crop(phase[:, None], cols[None]))
    replay_traj["site"], replay_traj["year0"] = site, np.zeros_like(site)
    soil_index = torch.as_tensor(site * MEMBERS + member, device=device)
    m_index = torch.as_tensor(member, device=device)
    smap, dates = _smap(ids)
    ends, slot, day = _steps(dates, years)
    frozen = _frozen(sites, dates, ends)
    obs = np.where((smap > 0) & (smap < 1) & ~frozen & np.isfinite(wilting_site.cpu().numpy())[:, None], smap, np.nan)
    obs_t = torch.as_tensor(obs.astype(np.float32), device=device)
    observer = Observer(obs_t[torch.as_tensor(site[::MEMBERS], device=device)], slot, day, year_lengths,
                        depth_soil[soil_index], wilting_site[torch.as_tensor(site, device=device)], Y, device)
    context = {"initial": {"water": capacity[soil_index] * values["initial_available_water_fraction"][m_index]},
               "soil_index": soil_index, "root_scale": values["root_scale"][m_index],
               "kc_scale": values["kc_scale"][m_index], "observe": observer}
    replay = water_pass(prepared, soils, crops, [], None, replay_traj, chunk=len(site), continuation=context)
    replay_seconds = time.perf_counter() - started
    fao = replay["fao_main"].view(S, 2, MEMBERS, Y)
    # Model-only normaliser: mean FAO ratio of the crop over replay years
    # 1983..target-1 (no NASS input); 1981-82 are spin-up.
    grows = torch.as_tensor(_phase_crop(np.array([0, 1])[:, None], years[None]), device=device)  # [2, Y]
    yield_part = _yield(prepared, soils, crops, years, year_len, observer, values, capacity, fao, grows,
                        S, ids, labels, device)
    soil_part = _soil_water(prepared, soils, crops, years, year_len, observer, values, ids, obs, dates, ends,
                            slot, day, wilting_site, depth_soil, S, device)
    del replay
    return {"yield": yield_part, "soil_water": soil_part, "replay_seconds": replay_seconds,
            "innovations": _innovations(observer, S), "seconds": time.perf_counter() - started}


def _innovations(observer, S):
    if not observer.log:
        return {"updates": 0}
    log = torch.stack(observer.log)  # [steps, 4, G]
    good = torch.isfinite(log[:, 0])
    sd = OBSERVATION_ERROR["sd_m3_m3"] * 1000
    pick = lambda i: log[:, i][good]
    return {"group_updates": int(good.sum()), "innovation_mean_mm": float(pick(0).mean()),
            "innovation_sd_mm": float(pick(0).std()), "assumed_observation_sd_mm": sd,
            "expected_innovation_sd_mm": float((pick(1).square() + sd ** 2).mean().sqrt()),
            "forecast_spread_mm": float(pick(2).mean()), "analysis_spread_mm": float(pick(3).mean())}


def _relative(labels, ids, device):
    """Observed yield / log-linear trend fitted on years strictly before each target."""
    obs = np.full((len(ids), 2, 45), np.nan, dtype=np.float32)
    for i, a in enumerate(ids):
        for c, k in enumerate(CROPS):
            for row in labels.loc[labels.adm_id.eq(a) & labels.crop.eq(k)].itertuples():
                if 1981 <= row.year <= 2025:
                    obs[i, c, row.year - 1981] = row.yield_t_ha
    o = torch.as_tensor(obs, device=device)
    x = torch.arange(1981, 2026, device=device, dtype=torch.float32) - 2000
    rel = torch.full((len(ids), 2, len(TARGETS)), float("nan"), device=device)
    prior = []
    for j, target in enumerate(TARGETS):
        train = torch.isfinite(o) & (o > 0) & (x < target - 2000)
        n = train.sum(-1)
        mx = torch.where(train, x, 0).sum(-1) / n.clamp_min(1)
        ly = o.clamp_min(.01).log()
        my = torch.where(train, ly, 0).sum(-1) / n.clamp_min(1)
        slope = (torch.where(train, (x - mx[..., None]) * (ly - my[..., None]), 0).sum(-1)
                 / torch.where(train, (x - mx[..., None]).square(), 0).sum(-1).clamp_min(1e-6))
        trend = (my[..., None] + slope[..., None] * (x - mx[..., None])).exp()
        ok = n >= 10
        rel[..., j] = torch.where(ok, o[..., target - 1981] / trend[..., target - 1981], torch.nan)
        # Prior-year residuals relative to the same trend: the observed-climatology reference.
        prior.append(torch.where(train & ok[..., None], o / trend, torch.nan))
    return rel, torch.stack(prior, -2)  # [S,2,T], [S,2,T,45]


def _yield(prepared, soils, crops, years, year_len, observer, values, capacity, fao, grows, S, ids, labels, device):
    T = len(TARGETS)
    donors = np.arange(1982, 2026)
    yi = {int(y): i for i, y in enumerate(years)}
    rows = []  # (site, crop, target index, kind, member, weather year)
    kinds = ("state_perfect", "state_climate", "capacity_climate", "capacity_perfect")
    for s in range(S):
        for c in range(2):
            for j, target in enumerate(TARGETS):
                for m in range(MEMBERS):
                    rows.append((s, c, j, 0, m, target))
                    rows.append((s, c, j, 3, m, target))
                for q, d in enumerate(donors[donors != target]):
                    m = (7 * q + target) % MEMBERS
                    rows.append((s, c, j, 1, m, d))
                    rows.append((s, c, j, 2, m, d))
    r = np.asarray(rows)
    s, c, j, kind, m, w = r.T
    target = np.asarray(TARGETS)[j]
    phase = np.where((target % 2 == 0) == (c == 0), 0, 1)
    cols = target[:, None] + np.arange(-2, 2)[None]
    crop = _phase_crop(phase[:, None], cols)
    assert (crop[:, 2] == c).all()
    traj = _trajectory(crop)
    traj["site"] = s
    weather_year = torch.as_tensor(np.stack([[yi[int(a) - 1], yi[int(a)]] for a in w]), device=device)
    first = torch.as_tensor(np.asarray([yi[int(t) - 1] for t in target]), device=device)
    st, ph, mm = (torch.as_tensor(a, device=device) for a in (s, phase, m))
    replay_index = st * 2 * MEMBERS + ph * MEMBERS + mm
    soil_index = st * MEMBERS + mm
    start_water = observer.october[replay_index, first]
    from_capacity = torch.as_tensor(kind >= 2, device=device)
    initial = torch.where(from_capacity, capacity[soil_index], start_water)
    start_day = 304 + (year_len[weather_year[:, 0]] == 366).long()
    context = {"initial": {"water": initial}, "soil_index": soil_index, "root_scale": values["root_scale"][mm],
               "kc_scale": values["kc_scale"][mm], "weather_year": weather_year, "start_day": start_day}
    out = water_pass(prepared, soils, crops, [], None, traj, chunk=60000, continuation=context)
    sim = out["fao_main"][:, 1]
    # Normaliser per (site, crop, target): replay mean FAO ratio, years 1983..target-1.
    year_t = torch.as_tensor(years, device=device)
    norm = torch.zeros((S, 2, T), device=device)
    for jj, t in enumerate(TARGETS):
        use = (year_t >= 1983) & (year_t < t)
        for cc in range(2):
            sel = (grows == cc) & use[None]  # [2, Y]
            norm[:, cc, jj] = (fao * sel[None, :, None]).sum((1, 2, 3)) / (sel.sum() * MEMBERS)
    rel_sim = sim / norm[st, torch.as_tensor(c, device=device), torch.as_tensor(j, device=device)].clamp_min(.01)
    observed, prior = _relative(labels, ids, device)
    # Pack ensembles [S,2,T,kind,slots].
    slots = len(donors)
    pack = torch.full((S, 2, T, 4, slots), float("nan"), device=device)
    order = np.zeros(len(r), dtype=np.int64)
    counter = {}
    for i, key in enumerate(zip(s, c, j, kind, strict=True)):
        order[i] = counter.get(key, 0)
        counter[key] = order[i] + 1
    pack[st, torch.as_tensor(c, device=device), torch.as_tensor(j, device=device),
         torch.as_tensor(kind, device=device), torch.as_tensor(order, device=device)] = rel_sim
    kind_t = torch.as_tensor(kind, device=device)
    planting = out["water_at_planting_mm"][:, 1]
    state_rows, capacity_rows = kind_t == 1, kind_t == 2
    diagnostics = {
        "deficit_below_capacity_31oct_mm": (capacity[soil_index] - start_water)[state_rows],
        "planting_water_state_mm": planting[state_rows], "planting_water_capacity_mm": planting[capacity_rows],
        "state_minus_capacity_relative_yield": rel_sim[state_rows] - rel_sim[capacity_rows],
        "target": torch.as_tensor(target, device=device)[state_rows]}
    return {"pack": pack, "observed": observed, "prior": prior, "kinds": kinds, "ids": ids,
            "diagnostics": diagnostics}


def _soil_water(prepared, soils, crops, years, year_len, observer, values, ids, obs, dates, ends, slot, day,
                wilting_site, depth_soil, S, device):
    obs_t = torch.as_tensor(obs.astype(np.float32), device=device)
    yi = {int(y): i for i, y in enumerate(years)}
    # Starts: first SMAP step whose start date falls in each month 2015-04..2025-09.
    frame = pd.DataFrame({"date": dates, "k": np.arange(len(dates))})
    frame = frame.loc[(frame.date >= "2015-04-01") & (frame.date <= "2025-09-30")]
    starts = frame.groupby([frame.date.dt.year, frame.date.dt.month]).k.min().to_numpy()
    starts = starts[starts + LEADS + 1 < len(dates)]
    rows = []
    for si in range(S):
        for k in starts:
            for ph in range(2):
                for m in range(MEMBERS):
                    rows.append((si, k, ph, m))
    r = np.asarray(rows)
    s, k, ph, m = r.T
    y0 = ends.year.to_numpy()[k]
    begin = ends[k] + pd.Timedelta(days=1)
    w0 = np.array([yi[int(a)] for a in y0])
    w1 = np.minimum(w0 + 1, len(years) - 1)
    cols = y0[:, None] + np.arange(-1, 3)[None]
    traj = _trajectory(_phase_crop(ph[:, None], cols))
    traj["site"] = s
    st, mm = torch.as_tensor(s, device=device), torch.as_tensor(m, device=device)
    replay_index = st * 2 * MEMBERS + torch.as_tensor(ph, device=device) * MEMBERS + mm
    initial = observer.analysis[replay_index, torch.as_tensor(k, device=device)]
    weather_year = torch.as_tensor(np.stack([w0, w1], -1), device=device)
    start = torch.as_tensor((ends[k].day_of_year.to_numpy()), device=device)  # index of the day after the step end
    length0 = year_len[weather_year[:, 0]]
    # Lead bounds: offsets (days from ``begin``) of the starts of steps k+1..k+13.
    following = k[:, None] + 1 + np.arange(LEADS + 1)[None]
    bounds = torch.as_tensor(((dates.to_numpy()[following] - begin.to_numpy()[:, None])
                              / np.timedelta64(1, "D")).astype(np.int64), device=device)
    soil_index = st * MEMBERS + mm
    acc = Accumulator(start, length0, bounds, depth_soil[soil_index], wilting_site[st],
                      year_len[weather_year])
    context = {"initial": {"water": initial}, "soil_index": soil_index, "root_scale": values["root_scale"][mm],
               "kc_scale": values["kc_scale"][mm], "weather_year": weather_year, "start_day": start,
               "observe": acc}
    water_pass(prepared, soils, crops, [], None, traj, chunk=60000, continuation=context)
    model = (acc.sums / acc.counts.clamp_min(1)).view(S, len(starts), 2 * MEMBERS, LEADS).mean(2) / 1000
    analysis = (initial / depth_soil[soil_index] + wilting_site[st] * 1000).view(S, len(starts), 2 * MEMBERS).mean(-1) / 1000
    ks = torch.as_tensor(starts, device=device)
    target_k = ks[:, None] + 1 + torch.arange(LEADS, device=device)[None]  # [starts, leads]
    observed = obs_t[:, target_k]  # [S, starts, leads]
    persistence = obs_t[:, ks]  # [S, starts]
    # Climatology: same within-year step position, other years 2015-2025 (leave target year out).
    position = torch.as_tensor(frame_position(dates), device=device)
    step_year = torch.as_tensor(dates.year.to_numpy(), device=device)
    clim = torch.full_like(observed, float("nan"))
    for a in range(len(starts)):
        for lead in range(LEADS):
            kk = int(target_k[a, lead])
            same = (position == position[kk]) & (step_year != step_year[kk]) & (step_year >= 2015)
            vals = obs_t[:, same]
            clim[:, a, lead] = torch.nanmean(vals, -1)
    return {"model": model, "increment": persistence[..., None] + model - analysis[..., None],
            "persistence": persistence[..., None].expand_as(observed), "climatology": clim,
            "observed": observed, "starts": len(starts)}


def frame_position(dates):
    """Index of each 8-day step within its calendar year."""
    years = dates.year.to_numpy()
    out = np.zeros(len(dates), dtype=np.int64)
    for y in np.unique(years):
        sel = years == y
        out[sel] = np.arange(sel.sum())
    return out


def _yield_scores(parts, device):
    pack = torch.cat([p["pack"] for p in parts])
    observed = torch.cat([p["observed"] for p in parts])
    prior = torch.cat([p["prior"] for p in parts])
    kinds = parts[0]["kinds"]
    target = torch.as_tensor(TARGETS, device=device)
    periods = {"all_2003_2024": target >= 0, "model_only_state_2003_2015": target < ASSIMILATED_FROM,
               "smap_assimilated_state_2016_2024": target >= ASSIMILATED_FROM}
    result = {}
    for c, crop in enumerate(CROPS):
        result[crop] = {}
        for name, period in periods.items():
            y = observed[:, c][:, period]  # [S, T']
            ok = torch.isfinite(y)
            block = {"county_years": int(ok.sum()), "counties": int(ok.any(-1).sum())}
            ens_all = {kind: pack[:, c, :, i][:, period] for i, kind in enumerate(kinds)}
            ens_all["observed_climatology"] = prior[:, c][:, period]
            for kind, ens in ens_all.items():
                e, mask = ens[ok], torch.isfinite(ens[ok])
                obs = y[ok]
                e0 = torch.nan_to_num(e)
                n = mask.sum(-1).clamp_min(1)
                mean = torch.where(mask, e0, 0).sum(-1) / n
                sd = (torch.where(mask, (e0 - mean[:, None]).square(), 0).sum(-1) / (n - 1).clamp_min(1)).sqrt()
                score = crps(e0, mask, obs)
                # Per-county correlation of the ensemble mean with observations (>=8 years).
                full_mean = torch.full_like(y, float("nan"))
                full_mean[ok] = mean
                r = _rowwise_r(full_mean, y)
                block[kind] = {"members_median": float(n.float().median()), "crps": float(score.mean()),
                               "rmse_of_mean": float((mean - obs).square().mean().sqrt()),
                               "mean_spread_sd": float(sd.mean()),
                               "coverage_1sd": float(((obs - mean).abs() <= sd).float().mean()),
                               "coverage_2sd": float(((obs - mean).abs() <= 2 * sd).float().mean()),
                               "pooled_r": float(torch.corrcoef(torch.stack((mean, obs)))[0, 1]),
                               "median_county_r": float(r[torch.isfinite(r)].median()) if bool(torch.isfinite(r).any()) else None,
                               "counties_with_r": int(torch.isfinite(r).sum()),
                               "mean_forecast": float(mean.mean()), "mean_observed": float(obs.mean())}
            sc = pack[:, c, :, 1][:, period][ok]
            fc = pack[:, c, :, 2][:, period][ok]
            block["state_minus_capacity_climate_mean_abs"] = float((torch.nanmean(sc, -1) - torch.nanmean(fc, -1)).abs().mean())
            result[crop][name] = block
    return result


def _pool_innovations(parts):
    """Pool counts, means and sample SDs without retaining every innovation."""
    n = np.array([p["group_updates"] for p in parts], dtype=float)
    out = {"group_updates": int(n.sum()), "assumed_observation_sd_mm": parts[0]["assumed_observation_sd_mm"]}
    for key in ("innovation_mean_mm", "forecast_spread_mm", "analysis_spread_mm"):
        out[key] = float((n * np.array([p[key] for p in parts])).sum() / n.sum())
    expected = np.array([p["expected_innovation_sd_mm"] for p in parts])
    out["expected_innovation_sd_mm"] = float(np.sqrt((n * expected ** 2).sum() / n.sum()))
    mean, sd = np.array([p["innovation_mean_mm"] for p in parts]), np.array([p["innovation_sd_mm"] for p in parts])
    variance = ((n - 1) * sd ** 2 + n * (mean - out["innovation_mean_mm"]) ** 2).sum() / (n.sum() - 1)
    out["innovation_sd_mm"] = float(np.sqrt(variance))
    out["note"] = "Pooled sample SD from batch moments; spreads are update-weighted means."
    return out


def _state_effect(parts, device):
    get = lambda name: torch.cat([p[name] for p in parts])
    q = torch.tensor([.1, .5, .9, .99], device=device)
    out = {}
    for name in ("deficit_below_capacity_31oct_mm", "planting_water_state_mm", "planting_water_capacity_mm"):
        out[name] = dict(zip(("p10", "p50", "p90", "p99"), torch.quantile(get(name), q).tolist(), strict=True))
    diff = get("state_minus_capacity_relative_yield").abs()
    out["abs_relative_yield_difference"] = dict(zip(("p10", "p50", "p90", "p99"), torch.quantile(diff, q).tolist(),
                                                    strict=True))
    out["abs_relative_yield_difference"]["max"] = float(diff.max())
    out["share_planting_water_differs_over_1mm"] = float(
        ((get("planting_water_state_mm") - get("planting_water_capacity_mm")).abs() > 1).float().mean())
    out["trajectories"] = len(diff)
    return out


def _rowwise_r(x, y):
    mask = torch.isfinite(x) & torch.isfinite(y)
    n = mask.sum(-1)
    xx = torch.where(mask, x, 0)
    yy = torch.where(mask, y, 0)
    xx = torch.where(mask, xx - xx.sum(-1, keepdim=True) / n.clamp_min(1)[:, None], 0)
    yy = torch.where(mask, yy - yy.sum(-1, keepdim=True) / n.clamp_min(1)[:, None], 0)
    den = (xx.square().sum(-1) * yy.square().sum(-1)).sqrt()
    return torch.where((n >= 8) & (den > 0), (xx * yy).sum(-1) / den.clamp_min(1e-12), float("nan"))


def _soil_scores(parts):
    get = lambda name: torch.cat([p[name] for p in parts])
    observed = get("observed")
    names = ("model", "increment", "persistence", "climatology")
    preds = {name: get(name) for name in names}
    # Identical (county, start, lead) pairs for every method.
    common = torch.isfinite(observed)
    for pred in preds.values():
        common &= torch.isfinite(pred)
    result = {}
    for name in names:
        pred = preds[name]
        leads = []
        for lead in range(LEADS):
            o, p = observed[..., lead], pred[..., lead]
            ok = common[..., lead]
            err = (p - o)[ok]
            leads.append({"lead_steps": lead + 1, "pairs": int(ok.sum()), "rmse_m3_m3": float(err.square().mean().sqrt()),
                          "bias_m3_m3": float(err.mean())})
        result[name] = leads
    clim = result["climatology"]
    for name in ("model", "increment", "persistence"):
        for row, ref in zip(result[name], clim, strict=True):
            row["mse_skill_vs_climatology"] = 1 - row["rmse_m3_m3"] ** 2 / ref["rmse_m3_m3"] ** 2
    return result


def run(limit=200, batch_size=25):
    started = time.perf_counter()
    device = accelerator()
    labels, ids = selected_labels(limit)
    meta = json.loads(Path("data/processed/eo/eo_county_8day.json").read_text())
    ids = [a for a in ids if a in meta["counties"]]
    parts, timings = [], []
    for i in range(0, len(ids), batch_size):
        out = batch(ids[i:i + batch_size], labels, device)
        parts.append(out)
        timings.append({"counties": len(ids[i:i + batch_size]), "seconds": out["seconds"],
                        "replay_seconds": out["replay_seconds"],
                        "mps_driver_allocated_gb": torch.mps.driver_allocated_memory() / 1e9
                        if device.type == "mps" else None})
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"C batch {i + len(ids[i:i + batch_size])}/{len(ids)} {out['seconds']:.1f}s", flush=True)
    report = {
        "counties": ids, "members": MEMBERS, "targets": [TARGETS[0], TARGETS[-1]],
        "yield": _yield_scores([p["yield"] for p in parts], device),
        "soil_water": _soil_scores([p["soil_water"] for p in parts]),
        "initial_state_effect": _state_effect([p["yield"]["diagnostics"] for p in parts], device),
        "innovations": [p["innovations"] for p in parts],
        "innovations_pooled": _pool_innovations([p["innovations"] for p in parts]),
        "timings": timings, "seconds": time.perf_counter() - started,
        "device": str(device),
        "method": {
            "selection": "planner.validate.selected_labels(200): rainfed corn/soybean series (explicit NON-IRRIGATED, or humid-state proxy), >=25 years, state round-robin.",
            "state": "Continuous 1981+ water replay, 32 stratified members (AWC, root depth, Kc, initial water; state.PARAMETERS), two corn-soybean rotation phases (ASSUMPTION: county fields in that rotation). Grouped EnSRF to SMAP L4 root zone from 2015-03-30 (same operator/error as state.py); model-only before. State = water after 31 October of the year before the target.",
            "forecast": "1 November -> harvest. state_climate / capacity_climate: 43 other weather years 1982-2025 (Nov-Dec of donor-1 + donor year incl. its observed state planting calendar), member (7q+target) mod 32. state_perfect / capacity_perfect: actual weather, 32 members. capacity_* start at field capacity, i.e. without the initial state.",
            "yield_metric": "Season-level FAO-33 relative yield (heat/frost factors included), divided by the replay's mean ratio for that crop over 1983..target-1 (model-only, no NASS). NOT the learned MLP (no forecast covariates), no rotation/pest multiplier.",
            "observed": "NASS county yield / log-linear trend fitted on that county's years before the target only (>=10 years). observed_climatology = prior-year trend-relative values (reference forecast).",
            "soil_water": "Restarts at the post-analysis state at the first SMAP step of each month 2015-04..2025-09; open loop with actual weather; ensemble mean over 2 phases x 32 members vs the next 12 SMAP 8-day values (0-1 m, frozen steps excluded). increment = SMAP at start + modelled change (no fitted bias). Climatology = same within-year step in the other years 2015-2025.",
            "tuning": "None. No parameter was changed after seeing these scores.",
        },
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "hindcast_c.json"
    path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()
    out = run(args.limit, args.batch)
    print(json.dumps(json_safe({"yield": out["yield"], "seconds": out["seconds"]}), indent=1)[:4000])
