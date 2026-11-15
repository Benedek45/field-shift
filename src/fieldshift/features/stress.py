"""Daily inputs: Celsius, mm/day, kPa, MJ/m²/day and source-specific RSM.

RSM and its climatology must use the same units: GLDAS kg/m² or POWER wetness
fraction. Perennial forage uses a green-up/frost window and thermal regrowth
opportunities, not fictitious annual flowering or observed harvest dates.

Stages use GDD at the *start* of each day: a threshold-crossing day belongs
to the preceding stage. The first day reaching maturity is included.
"""

import numpy as np
import pandas as pd

from fieldshift.features.traits import numeric_traits

STAGES = ("establishment", "vegetative", "flowering", "grain_fill")
SUMS = ("gdd", "heat_dd", "frost_days", "prec_mm", "et0_mm", "cwb_mm",
        "heat_dd_30", "heat_dd_35", "dry_days", "water_deficit_mm", "vpd_excess_kpa_days")
MEANS = ("vpd_kpa", "radiation_mj_m2", "rsm_anomaly", "rsm")
REQUIRED = ("tmin", "tmax", "tavg", "prec", "et0", "vpd", "rad")


def sowing_dates(daily, traits, harvest_year):
    """First 10-day mean >= spring/green-up threshold, or <= autumn threshold.

    Search March 1–June 15 (spring), September 1–December 1 (autumn).
    Established forage searches for green-up from February 1, never sowing.
    Fall back to the last search day when no threshold crossing exists.
    """
    t = numeric_traits(traits)
    winter = bool(t["autumn_sown"])
    year = harvest_year - int(winter)
    perennial = bool(t.get("perennial", 0))
    reference_start = 244 if winter else (32 if perennial else 60)
    start = pd.Timestamp(year, 9 if winter else (2 if perennial else 3), 1)
    start += pd.Timedelta(days=t.get("sowing_start_doy", reference_start) - reference_start)
    end = pd.Timestamp(year, 12 if winter else 6, 1 if winter else 15)
    end += pd.Timedelta(days=t.get("sowing_end_doy", 335 if winter else 166) - (335 if winter else 166))
    d = daily.sort_values(["adm_id", "date"])
    rolling = d.groupby("adm_id", sort=False).tavg.rolling(
        10, min_periods=10
    ).mean().reset_index(level=0, drop=True).reindex(d.index)
    crossing = rolling.le(t["sowing_temp_c"]) if winter else rolling.ge(t["sowing_temp_c"])
    dates = d.loc[crossing & d.date.between(start, end)].groupby("adm_id").date.min()
    return dates.reindex(d.adm_id.unique()).fillna(end)


def daily_stages(daily, traits, sowing_day, harvest_year=None):
    """Vectorised across all counties; sowing_day is a timestamp or county Series."""
    t = numeric_traits(traits)
    d = daily.copy()
    if d.empty:
        raise ValueError("Daily weather is empty")
    if "adm_id" not in d:
        d["adm_id"] = "field"
    d["date"] = pd.to_datetime(d.date)
    if d.duplicated(["adm_id", "date"]).any():
        raise ValueError("Duplicate county dates")
    if not np.isfinite(d[list(REQUIRED)].to_numpy(dtype=float)).all():
        raise ValueError("Weather contains missing or nonfinite required values")
    if (d.tmin > d.tmax).any():
        raise ValueError("tmin exceeds tmax")
    boundaries = [t["emergence_gdd"], t["flowering_gdd"],
                  t["flowering_gdd"] + t["flowering_duration_gdd"], t["maturity_gdd"]]
    if not t.get("perennial", 0) and not 0 < boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]:
        raise ValueError("GDD thresholds must be strictly increasing")
    if isinstance(sowing_day, pd.Series):
        sow = d.adm_id.map(sowing_day)
    else:
        sow = pd.Series(pd.Timestamp(sowing_day), index=d.index)
    if sow.isna().any():
        raise ValueError("A sowing date is required for every county")
    if harvest_year is None:
        harvest_year = int(sow.dt.year.max()) + int(t["autumn_sown"])
    end = pd.Timestamp(harvest_year, int(t.get("season_end_month", 8 if t["autumn_sown"] else 10)), 31)
    d = d.loc[d.date.ge(sow) & d.date.le(end)].sort_values(["adm_id", "date"])
    if d.empty:
        raise ValueError("No daily weather within sowing-to-harvest window")
    gaps = d.groupby("adm_id").date.diff().dt.days.dropna()
    if gaps.ne(1).any():
        raise ValueError("Daily weather must be contiguous within each season")
    d["gdd"] = (((d.tmin.clip(t["base_c"], t["upper_c"])
                 + d.tmax.clip(t["base_c"], t["upper_c"])) / 2) - t["base_c"]).clip(lower=0)
    if t["autumn_sown"]:
        chilling = d.tavg.between(t["vernalization_min_c"], t["vernalization_max_c"])
        vern = chilling.groupby(d.adm_id).cumsum()
        d.loc[d.tavg.lt(t["dormancy_c"]), "gdd"] = 0.0
        unlocked = vern.ge(t["vernalization_days"]) & d.date.dt.year.eq(harvest_year)
        locked_gdd = d.gdd.where(~unlocked, 0).groupby(d.adm_id).cumsum()
        capped = locked_gdd.clip(upper=t["pre_vernalization_gdd"])
        increments = capped.groupby(d.adm_id).diff().fillna(capped)
        d["gdd"] = np.where(unlocked, d.gdd, increments)
        d["vernalized"] = unlocked.astype(float)
    else:
        d["vernalized"] = 1.0
    cumulative = d.groupby("adm_id").gdd.cumsum()
    d["gdd_before"] = cumulative - d.gdd
    if t.get("perennial", 0):
        killing = d.date.dt.month.ge(8) & d.tmin.lt(t["frost_c"])
        first_frost = d.loc[killing].groupby("adm_id").date.min()
        stop = d.adm_id.map(first_frost).fillna(end)
        d = d.loc[d.date.lt(stop)].copy()
        d["stage"] = -1
    else:
        d = d.loc[d.gdd_before.lt(t["maturity_gdd"])].copy()
        d["stage"] = np.searchsorted(boundaries[:3], d.gdd_before, side="right")
    frost = np.where(d.stage.lt(2), t["vegetative_frost_c"], t["frost_c"])
    d["heat_dd"] = (d.tmax - t["heat_critical_c"]).clip(lower=0)
    d["frost_days"] = (d.tmin.to_numpy() < frost).astype(float)
    d["heat_dd_30"] = (d.tmax - 30).clip(lower=0)
    d["heat_dd_35"] = (d.tmax - 35).clip(lower=0)
    d["dry_days"] = d.prec.lt(1).astype(float)
    d["water_deficit_mm"] = (d.et0 - d.prec).clip(lower=0)
    d["vpd_excess_kpa_days"] = (d.vpd - 1.5).clip(lower=0)
    d["prec_mm"], d["et0_mm"] = d.prec, d.et0
    d["cwb_mm"] = d.prec - d.et0
    d["vpd_kpa"], d["radiation_mj_m2"] = d.vpd, d.rad
    for col in ("rsm", "rsm_anomaly"):
        if col not in d:
            d[col] = np.nan
    return d


def aggregate_stress(d):
    index = pd.Index(d.adm_id.unique(), name="adm_id")
    columns = {}
    for stage, name in enumerate(STAGES):
        group = d.loc[d.stage.eq(stage)].groupby("adm_id")
        summary = pd.concat([
            group[list(SUMS)].sum(min_count=1),
            group[list(MEANS)].mean(),
            group.size().rename("length_days"),
            group.rsm.count().rename("rsm_days"),
        ], axis=1)
        for col in summary:
            columns[f"{name}__{col}"] = summary[col].reindex(index)
    result = pd.DataFrame(columns, index=index)
    extra = {}
    group = d.groupby("adm_id")
    result["season__length_days"] = group.size()
    for source, dest in (("prec", "prec_mm"), ("et0", "et0_mm"), ("cwb_mm", "cwb_mm")):
        result[f"season__{dest}"] = group[source].sum(min_count=1)
    for source, dest in (("tavg", "tavg_c"), ("tmin", "tmin_c"), ("tmax", "tmax_c"),
                         ("vpd", "vpd_kpa"), ("rad", "radiation_mj_m2")):
        result[f"season__{dest}"] = group[source].mean()
    result["maturity_reached"] = (
        np.nan if d.attrs.get("perennial", 0) else
        (group.gdd.sum() >= d.attrs.get("maturity_gdd", np.inf)).astype(float)
    )
    result["vernalized"] = np.nan if d.attrs.get("perennial", 0) else group.vernalized.max()
    result["forage__potential_cut_cycles"] = (
        np.floor(group.gdd.sum() / d.attrs.get("cut_interval_gdd", 500))
        if d.attrs.get("perennial", 0) else 0.
    )
    for col in SUMS:
        result[f"season__{col}"] = group[col].sum(min_count=1)
    result["season__rsm"] = group.rsm.mean()
    result["season__rsm_anomaly"] = group.rsm_anomaly.mean()
    for col in SUMS + MEANS:
        extra[f"forage_regrowth__{col}"] = (
            group[col].sum(min_count=1) if col in SUMS else group[col].mean()
        ) if d.attrs.get("perennial", 0) else np.nan
    extra["season_start_ordinal"] = group.date.min().map(pd.Timestamp.toordinal)
    extra["season_end_ordinal"] = group.date.max().map(pd.Timestamp.toordinal)
    return pd.concat([result, pd.DataFrame(extra, index=index)], axis=1)


def compute_stress(daily, traits, sowing_day, harvest_year=None):
    """Return one county row of named stage/season features.

    Optional rsm_anomaly must be supplied from an external, label-free
    county/calendar-day climatology in the same units as rsm.
    """
    d = daily_stages(daily, traits, sowing_day, harvest_year)
    d.attrs["maturity_gdd"] = numeric_traits(traits)["maturity_gdd"]
    d.attrs["perennial"] = numeric_traits(traits).get("perennial", 0)
    d.attrs["cut_interval_gdd"] = numeric_traits(traits).get("cut_interval_gdd", 500)
    return aggregate_stress(d)


def explanation_stage_names(traits):
    """UI labels for shared physical stage slots; no fictitious forage flowering."""
    t = numeric_traits(traits)
    if t.get("perennial", 0):
        return {"forage_regrowth": "repeated forage regrowth"}
    if t.get("harvest_storage_root", 0):
        return dict(zip(STAGES, ("establishment", "vegetative", "storage-organ initiation", "bulking")))
    return dict(zip(STAGES, STAGES))
