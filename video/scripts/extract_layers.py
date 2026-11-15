"""Write the real Story County, IA data series that the video draws as charts.

Run from the repository root:  uv run python video/scripts/extract_layers.py
Output: video/assets/data/story_layers.js (sets window.STORY_LAYERS).
Every number on the "data layers" shot comes from this file.
"""

import json
from pathlib import Path

import numpy as np

from fieldshift.data import eo, power, prism

ADM = "US-19-169"  # Story County, IA
OUT = Path("video/assets/data/story_layers.js")


def r(values, digits):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


def main():
    cube, meta = power.load_county_daily()
    daily = power.county_frame(cube, meta, ADM)
    daily["year"] = daily["date"].dt.year
    summer = daily[daily["date"].dt.month.isin([6, 7, 8])]
    jja_power = summer.groupby("year")["prec"].sum()
    jja_tmax = summer.groupby("year")["tmax"].mean()

    pcube, pmeta = prism.load_prism()
    pframe = prism.county_frame(pcube, pmeta, ADM)
    pframe["year"] = pframe["start_date"].dt.year
    psummer = pframe[pframe["start_date"].dt.month.isin([6, 7, 8])]
    jja_prism = psummer.groupby("year")["ppt_sum"].sum()

    ecube, emeta = eo.load_eo()
    series = eo.county_series(ecube, emeta, ADM)
    series["year"] = series["start_date"].dt.year
    series["doy"] = series["start_date"].dt.day_of_year

    modis = series[series["start_date"] >= "2000-02-18"]
    ndvi_years = {}
    for year in (2011, 2012, 2013):
        part = modis[modis["year"] == year]
        ndvi_years[str(year)] = {"doy": part["doy"].tolist(), "ndvi": r(part["ndvi"], 3)}

    smap = series[series["start_date"] >= "2015-04-01"][["start_date", "smap_rootzone"]].dropna()
    smap_monthly = smap.set_index("start_date")["smap_rootzone"].resample("MS").mean().dropna()

    grace = series[series["grace_observed"] > 0][["start_date", "grace_tws"]].dropna()
    grace_monthly = grace.set_index("start_date")["grace_tws"].resample("MS").mean().dropna()

    years = list(range(1981, 2026))
    data = {
        "county": "Story County, IA",
        "adm_id": ADM,
        "power_jja_rain_mm": {"years": years, "values": r(jja_power.reindex(years).values, 1)},
        "power_jja_tmax_c": {"years": years, "values": r(jja_tmax.reindex(years).values, 2)},
        "prism_jja_rain_mm_8day_steps": {
            "years": years, "values": r(jja_prism.reindex(years).values, 1),
            "note": "sum of 8-day steps starting in June-August",
        },
        "modis_ndvi": ndvi_years,
        "smap_rootzone_monthly": {
            "dates": [d.strftime("%Y-%m") for d in smap_monthly.index],
            "values": r(smap_monthly.values, 4),
        },
        "grace_tws_cm_monthly": {
            "dates": [d.strftime("%Y-%m") for d in grace_monthly.index],
            "values": r(grace_monthly.values, 2),
        },
        "sources": {
            "power": "data/processed/weather/power_county_daily.f32 (NASA POWER MERRA-2, cropland-weighted)",
            "prism": "data/processed/weather/prism_county_8day.f32 (PRISM AN daily via Earth Engine)",
            "eo": "data/processed/eo/eo_county_8day.f32 (MODIS MOD13A2 NDVI, SMAP L4 root zone, GRACE mascons)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.STORY_LAYERS = " + json.dumps(data) + ";\n")
    print("wrote", OUT)
    print("JJA rain POWER 1988/2012:", jja_power[1988].round(1), jja_power[2012].round(1),
          "median", jja_power.median().round(1))
    print("JJA rain PRISM 1988/2012:", jja_prism[1988].round(1), jja_prism[2012].round(1))
    print("NDVI Jul-Aug mean 2011/12/13:", [
        round(float(modis[(modis.year == y) & modis.start_date.dt.month.isin([7, 8])]["ndvi"].mean()), 3)
        for y in (2011, 2012, 2013)])
    print("SMAP months", len(smap_monthly), "min/max", smap_monthly.min().round(3), smap_monthly.max().round(3))
    print("GRACE months", len(grace_monthly), grace_monthly.index[0], grace_monthly.index[-1])


if __name__ == "__main__":
    main()
