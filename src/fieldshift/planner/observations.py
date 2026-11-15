"""Small, auditable county observation summaries, separate from simulated outcomes."""

from pathlib import Path

import numpy as np
import pandas as pd

from fieldshift.data.eo import load_eo


def drought_summers(weekly):
    """D2 is cumulative: never add D2+D3+D4 (that would double count)."""
    summer = weekly.loc[weekly.map_date.dt.month.isin([6, 7, 8])].copy()
    summer["year"] = summer.map_date.dt.year
    return summer.groupby("year", as_index=False).d2.max().rename(columns={"d2": "max_d2_area_pct"})


def county_observations(adm_id):
    cube, meta = load_eo()
    names = [v["name"] for v in meta["variables"]]
    eo = pd.DataFrame(np.array(cube[meta["counties"].index(adm_id)]), columns=names)
    eo["date"] = pd.to_datetime(meta["step_start_dates"])
    eo["year"], eo["month"] = eo.date.dt.year, eo.date.dt.month
    eo["doy"] = eo.date.dt.dayofyear - 1
    # Only MODIS-era recent greenness; no adjusted pre-2000 AVHRR.
    recent = eo.loc[eo.year.between(2016, 2025)].copy()
    recent["day"] = (recent.date.dt.dayofyear - 1 - ((recent.date.dt.is_leap_year)
                      & recent.date.dt.month.gt(2)).astype(int)) // 8 * 8
    ndvi = recent.groupby("day", as_index=False).ndvi.mean().dropna()
    ndvi["end"] = np.minimum(ndvi.day + 8, 365)
    low = ndvi.loc[ndvi.ndvi.lt(.3)]
    low_days = int((low.end - low.day).sum())
    sm = recent.groupby(["year", "month"], as_index=False).smap_rootzone.mean().dropna()
    weekly_path = Path("data/processed/usdm_county_weekly.parquet")
    drought = pd.DataFrame(columns=["year", "max_d2_area_pct"])
    if weekly_path.exists():
        weekly = pd.read_parquet(weekly_path, filters=[("adm_id", "==", adm_id)],
                                 columns=["adm_id", "map_date", "d2"])
        drought = drought_summers(weekly)
        drought = drought.loc[drought.year.between(2000, 2025)]
    dry_years = drought.loc[drought.max_d2_area_pct.gt(0), "year"].tolist()
    sm["summer"] = np.where(sm.year.isin(dry_years), "D2+ summer", "No D2+ summer")
    grace = eo.loc[eo.grace_observed.eq(1) & eo.grace_tws.notna(), ["date", "grace_tws"]].copy()
    grace["date"] = grace.date.dt.to_period("M").dt.to_timestamp()
    grace = grace.groupby("date", as_index=False).grace_tws.mean()
    static = pd.read_parquet("data/processed/static/county_static.parquet",
                             filters=[("adm_id", "==", adm_id)],
                             columns=["adm_id", "grace_tws_trend_cm_per_year"])
    trend = float(static.grace_tws_trend_cm_per_year.iloc[0]) if len(static) else float("nan")
    share = float(drought.max_d2_area_pct.gt(0).mean()) if len(drought) else float("nan")
    return {"ndvi": ndvi, "low": low, "low_days": low_days, "moisture": sm, "drought": drought,
            "drought_share": share, "grace": grace, "grace_trend": trend,
            "suggest_water": (np.isfinite(trend) and trend < -.5) or (np.isfinite(share) and share >= .3)}
