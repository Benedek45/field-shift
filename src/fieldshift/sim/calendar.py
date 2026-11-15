"""Vectorised phenology on the accelerator.

A torch port of ``features.stress`` (sowing rule, GDD stages, vernalization,
stage/season stress aggregation) evaluated for many sites and all harvest years
at once. Each harvest year y is a 731-day window from 1 January of y-1 to
31 December of y, so autumn-sown seasons keep their establishment months.
``tests/test_field_sim.py`` checks the port against the pandas reference.
"""

import numpy as np
import pandas as pd
import torch

from fieldshift.features.stress import MEANS, STAGES, SUMS
from fieldshift.features.traits import numeric_traits

VARS = ("tmin", "tmax", "tavg", "prec", "et0", "vpd", "rad", "rsm", "rsm_anomaly", "gwetroot")
WINDOW = 731


class DailyGrid:
    """Sites × days weather with harvest-year windows; float32 on ``device``."""

    def __init__(self, frames, device, values=None):
        dates = pd.DatetimeIndex(frames[0].date)
        for f in ([] if values is not None else frames[1:]):
            if not pd.DatetimeIndex(f.date).equals(dates):
                raise ValueError("All sites must share one daily date axis")
        self.device = device
        self.dates = dates
        self.years = sorted(dates.year.unique())
        rows = []
        for f in ([] if values is not None else frames):
            f = f.copy()
            if "rsm_anomaly" not in f:
                clim = f.groupby(f.date.dt.strftime("%m-%d")).rsm.transform("mean")
                f["rsm_anomaly"] = f.rsm - clim
            rows.append(f[list(VARS)].to_numpy("float32"))
        self.x = values if values is not None else torch.tensor(np.stack(rows), device=device)
        starts = np.array([dates.get_loc(pd.Timestamp(y, 1, 1)) for y in self.years] + [len(dates)])
        self.year_start = starts
        self.year_len = np.diff(starts)
        ny = len(self.years)
        # Days in the calendar year before each harvest year (365 for 1981's virtual 1980).
        self.prev_len = np.array([self.year_len[y - 1] if y else 365 for y in range(ny)])
        offsets = np.arange(WINDOW)
        first = np.array([starts[y] - self.prev_len[y] for y in range(ny)])
        absolute = first[:, None] + offsets[None, :]
        last = starts[1:]
        self.valid_np = (absolute >= 0) & (absolute < last[:, None])
        self.index = torch.tensor(np.clip(absolute, 0, len(dates) - 1), device=device)
        self.valid = torch.tensor(self.valid_np, device=device)
        window_dates = dates[np.clip(absolute, 0, len(dates) - 1).ravel()].to_numpy().reshape(absolute.shape)
        month = pd.DatetimeIndex(window_dates.ravel()).month.to_numpy().reshape(absolute.shape).copy()
        # Days before 1981 do not exist; their month is still computed from the offset.
        for y in range(ny):
            if first[y] < 0:
                fake = pd.date_range(pd.Timestamp(self.years[y] - 1, 1, 1), periods=WINDOW)
                month[y] = fake.month.to_numpy()
        self.month = torch.tensor(month, device=device)
        self.harvest_part = torch.tensor(offsets[None, :] >= self.prev_len[:, None], device=device)
        tavg = self.x[..., VARS.index("tavg")]
        cs = torch.nn.functional.pad(tavg.cumsum(-1), (1, 0))
        rolling = (cs[:, 10:] - cs[:, :-10]) / 10
        self.rolling10 = torch.cat([torch.full_like(tavg[:, :9], float("nan")), rolling], -1)

    def window(self, values):
        """[S, D] -> [S, Y, 731]; invalid days NaN."""
        out = values[:, self.index]
        return torch.where(self.valid, out, torch.full_like(out, float("nan")))

    def var(self, name):
        return self.window(self.x[..., VARS.index(name)])

    def offset(self, y, year, month, day):
        """Window offset of a calendar date for harvest-year index y."""
        base = pd.Timestamp(self.years[y] - 1, 1, 1)
        return (pd.Timestamp(year, month, day) - base).days

    def calendar_days(self, values):
        """[S, Y, 731] -> current-year part [S, Y, 366] and preceding-year part [S, Y, 366]."""
        s, ny = values.shape[0], values.shape[1]
        current = torch.zeros((s, ny, 366) + values.shape[3:], device=values.device, dtype=values.dtype)
        lead = torch.zeros_like(current)
        for y in range(ny):
            p = int(self.prev_len[y])
            current[:, y, :WINDOW - p] = values[:, y, p:]
            lead[:, y, :p] = values[:, y, :p]
        return current, lead


def _month_end(year, month):
    return (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).day


def sowing_offsets(grid, traits):
    """Temperature sowing rule of ``features.stress.sowing_dates``; [S, Y] offsets."""
    t = numeric_traits(traits)
    winter, perennial = bool(t["autumn_sown"]), bool(t.get("perennial", 0))
    reference_start = 244 if winter else (32 if perennial else 60)
    starts, ends = [], []
    for y, harvest_year in enumerate(grid.years):
        year = harvest_year - int(winter)
        start = grid.offset(y, year, 9 if winter else (2 if perennial else 3), 1)
        start += int(t.get("sowing_start_doy", reference_start) - reference_start)
        end = grid.offset(y, year, 12 if winter else 6, 1 if winter else 15)
        end += int(t.get("sowing_end_doy", 335 if winter else 166) - (335 if winter else 166))
        starts.append(start)
        ends.append(end)
    starts = torch.tensor(starts, device=grid.device)[None, :, None]
    ends = torch.tensor(ends, device=grid.device)[None, :, None]
    o = torch.arange(WINDOW, device=grid.device)[None, None, :]
    rolling = grid.window(grid.rolling10)
    crossing = rolling.le(t["sowing_temp_c"]) if winter else rolling.ge(t["sowing_temp_c"])
    hit = crossing & (o >= starts) & (o <= ends) & grid.valid
    first = torch.where(hit, o, WINDOW).amin(-1)
    return torch.where(first < WINDOW, first, ends[..., 0].expand_as(first))


def observed_offsets(grid, record, n_sites=1):
    """State crop-progress dates (by year, else median day of year) as harvest-year offsets."""
    out = []
    for y, year in enumerate(grid.years):
        date = record["by_year"].get(int(year))
        if date is None:
            date = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=record["median_doy"] - 1)
        out.append((pd.Timestamp(date) - pd.Timestamp(year - 1, 1, 1)).days)
    return torch.tensor(out, device=grid.device)[None, :].expand(n_sites, -1)


def gdd_between(grid, traits, first, last):
    """Plain GDD (no vernalization) from offset ``first`` to ``last`` inclusive; [S, Y]."""
    t = numeric_traits(traits)
    tmin, tmax = grid.var("tmin"), grid.var("tmax")
    gdd = (((tmin.clamp(t["base_c"], t["upper_c"]) + tmax.clamp(t["base_c"], t["upper_c"])) / 2)
           - t["base_c"]).clamp_min(0).nan_to_num(0)
    o = torch.arange(WINDOW, device=grid.device)[None, None, :]
    inside = (o >= first[..., None]) & (o <= last[..., None])
    return (gdd * inside).sum(-1)


def stages(grid, traits, sow, scale=None):
    """Daily season arrays as in ``features.stress.daily_stages``.

    ``sow``: [S, Y] window offsets. ``scale``: optional [S, Y] factor applied to
    the four GDD thresholds (variety override). Returns dict of [S, Y, 731].
    """
    t = numeric_traits(traits)
    device = grid.device
    o = torch.arange(WINDOW, device=device)[None, None, :]
    autumn, perennial = bool(t["autumn_sown"]), bool(t.get("perennial", 0))
    ends = []
    for y, harvest_year in enumerate(grid.years):
        month = int(t.get("season_end_month", 8 if autumn else 10))
        ends.append(grid.offset(y, harvest_year, month, _month_end(harvest_year, month)))
    end = torch.tensor(ends, device=device)[None, :, None]
    season = (o >= sow[..., None]) & (o <= end) & grid.valid
    tmin, tmax, tavg = grid.var("tmin"), grid.var("tmax"), grid.var("tavg")
    gdd = (((tmin.clamp(t["base_c"], t["upper_c"]) + tmax.clamp(t["base_c"], t["upper_c"])) / 2)
           - t["base_c"]).clamp_min(0)
    gdd = torch.where(season, gdd, 0)
    if autumn:
        chilling = (tavg >= t["vernalization_min_c"]) & (tavg <= t["vernalization_max_c"]) & season
        vern = chilling.float().cumsum(-1)
        gdd = torch.where(tavg < t["dormancy_c"], 0, gdd)
        unlocked = (vern >= t["vernalization_days"]) & grid.harvest_part[None] & season
        locked = torch.where(unlocked, 0, gdd).cumsum(-1)
        capped = locked.clamp(max=t["pre_vernalization_gdd"])
        increments = capped - torch.nn.functional.pad(capped[..., :-1], (1, 0))
        gdd = torch.where(unlocked, gdd, increments) * season
        vernalized = unlocked.float()
    else:
        vernalized = torch.ones_like(gdd)
    before = gdd.cumsum(-1) - gdd
    if perennial:
        killing = (grid.month[None] >= 8) & (tmin < t["frost_c"]) & season
        stop = torch.where(killing, o, WINDOW).amin(-1, keepdim=True)
        stop = torch.where(stop < WINDOW, stop, end + 1)
        keep = season & (o < stop)
        stage = torch.full_like(gdd, -1, dtype=torch.long)
        maturity = None
    else:
        k = 1 if scale is None else scale[..., None]
        bounds = [t["emergence_gdd"] * k, t["flowering_gdd"] * k,
                  (t["flowering_gdd"] + t["flowering_duration_gdd"]) * k]
        maturity = t["maturity_gdd"] * k
        keep = season & (before < maturity)
        stage = sum((before >= b).long() for b in bounds)
    frost_threshold = torch.where(stage < 2, t["vegetative_frost_c"], t["frost_c"])
    return {"keep": keep, "stage": torch.where(keep, stage, -2), "gdd": gdd, "gdd_before": before,
            "vernalized": vernalized, "maturity": maturity, "perennial": perennial,
            "heat_dd": (tmax - t["heat_critical_c"]).clamp_min(0),
            "frost_days": (tmin < frost_threshold).float(),
            "tmin": tmin, "tmax": tmax, "tavg": tavg}


def _daily_features(grid, s):
    prec, et0, vpd = grid.var("prec"), grid.var("et0"), grid.var("vpd")
    return {
        "gdd": s["gdd"], "heat_dd": s["heat_dd"], "frost_days": s["frost_days"],
        "prec_mm": prec, "et0_mm": et0, "cwb_mm": prec - et0,
        "heat_dd_30": (s["tmax"] - 30).clamp_min(0), "heat_dd_35": (s["tmax"] - 35).clamp_min(0),
        "dry_days": (prec < 1).float(), "water_deficit_mm": (et0 - prec).clamp_min(0),
        "vpd_excess_kpa_days": (vpd - 1.5).clamp_min(0),
        "vpd_kpa": vpd, "radiation_mj_m2": grid.var("rad"),
        "rsm_anomaly": grid.var("rsm_anomaly"), "rsm": grid.var("rsm"),
        "tavg": s["tavg"], "tmin": s["tmin"], "tmax": s["tmax"], "prec": prec, "et0": et0,
        "vpd": vpd, "rad": grid.var("rad"),
    }


def _sum(x, mask):
    count = mask.sum(-1)
    total = torch.where(mask, x.nan_to_num(0), 0).sum(-1)
    return torch.where(count > 0, total, float("nan"))


def _mean(x, mask):
    ok = mask & ~x.isnan()
    count = ok.sum(-1)
    return torch.where(count > 0, torch.where(ok, x, 0).sum(-1) / count.clamp_min(1), float("nan"))


def stress_features(grid, traits, sow):
    """Stage and season features of ``features.stress.compute_stress``; dict of [S, Y]."""
    t = numeric_traits(traits)
    s = stages(grid, traits, sow)
    daily = _daily_features(grid, s)
    keep = s["keep"]
    out = {}
    nan = torch.full(keep.shape[:2], float("nan"), device=grid.device)
    for k, name in enumerate(STAGES):
        mask = keep & (s["stage"] == k)
        count = mask.sum(-1).float()
        for col in SUMS:
            out[f"{name}__{col}"] = _sum(daily[col], mask)
        for col in MEANS:
            out[f"{name}__{col}"] = _mean(daily[col], mask)
        out[f"{name}__length_days"] = torch.where(count > 0, count, float("nan"))
        rsm_count = (mask & ~daily["rsm"].isnan()).sum(-1).float()
        out[f"{name}__rsm_days"] = torch.where(count > 0, rsm_count, float("nan"))
    out["season__length_days"] = keep.sum(-1).float()
    for source, dest in (("prec", "prec_mm"), ("et0", "et0_mm"), ("cwb_mm", "cwb_mm")):
        out[f"season__{dest}"] = _sum(daily[source], keep)
    for source, dest in (("tavg", "tavg_c"), ("tmin", "tmin_c"), ("tmax", "tmax_c"),
                         ("vpd", "vpd_kpa"), ("rad", "radiation_mj_m2")):
        out[f"season__{dest}"] = _mean(daily[source], keep)
    gdd_total = (s["gdd"] * keep).sum(-1)
    if s["perennial"]:
        out["maturity_reached"] = nan
        out["vernalized"] = nan
        out["forage__potential_cut_cycles"] = torch.floor(gdd_total / t.get("cut_interval_gdd", 500))
    else:
        out["maturity_reached"] = (gdd_total >= t["maturity_gdd"]).float()
        out["vernalized"] = torch.where(keep, s["vernalized"], -1).amax(-1)
        out["forage__potential_cut_cycles"] = torch.zeros_like(nan)
    for col in SUMS:
        out[f"season__{col}"] = _sum(daily[col], keep)
    out["season__rsm"] = _mean(daily["rsm"], keep)
    out["season__rsm_anomaly"] = _mean(daily["rsm_anomaly"], keep)
    for col in SUMS + MEANS:
        if s["perennial"]:
            out[f"forage_regrowth__{col}"] = _sum(daily[col], keep) if col in SUMS else _mean(daily[col], keep)
        else:
            out[f"forage_regrowth__{col}"] = nan
    return out
