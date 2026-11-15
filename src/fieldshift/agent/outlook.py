"""Fail-safe, cached Open-Meteo SEAS5 outlook; never changes simulator weather.

Current API documentation and live schema checked 2026-11-14. Raw member
hindcasts are not exposed: use provider model-climatology mean anomalies for
mean-bias adjustment, NOT a fictitious quantile calibration.
"""

import calendar
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("data/processed/engine_v26/outlook")
ENDPOINT = "https://seasonal-api.open-meteo.com/v1/seasonal"
METADATA = "https://seasonal-api.open-meteo.com/data/ecmwf_seas5/static/meta.json"
DOCS = "https://open-meteo.com/en/docs/seasonal-forecast-api"
DAILY = ("precipitation_sum", "temperature_2m_mean", "temperature_2m_min",
         "temperature_2m_max", "et0_fao_evapotranspiration_sum")
MONTHLY = ("precipitation_mean", "precipitation_anomaly", "temperature_2m_mean", "temperature_2m_anomaly")
LOG = logging.getLogger(__name__)
CAVEAT = (
    "All numbers below are mean-bias-adjusted, uncalibrated ensemble guidance, NOT reliable event odds. "
    "Rain is scaled by county/model monthly normal; temperature is offset by their normal difference. "
    "Model normal = Open-Meteo monthly ensemble mean minus its ECMWF hindcast anomaly. "
    "No lead-matched member hindcast distribution is available here, so spread, tails, trends, "
    "wet-day frequency and daily frost bias are NOT corrected. Model-climatology period need not match "
    "the county baseline. Seasonal skill in the central US is low, especially for rain; do not optimize "
    "a rotation on these percentages alone. No county skill score is supplied or validated here."
)


def quantile_map(values, model_history, observed_history):
    """Empirical monotone quantile mapping, for a future matched hindcast archive.

    Endpoint-clamped interpolation; *not* used by the live outlook without such
    an archive. Mapping a complete synthetic model distribution reproduces the
    matching empirical observed distribution.
    """
    model = np.sort(np.asarray(model_history, dtype=float))
    observed = np.sort(np.asarray(observed_history, dtype=float))
    model, observed = model[np.isfinite(model)], observed[np.isfinite(observed)]
    if min(len(model), len(observed)) < 3:
        raise ValueError("Quantile mapping needs at least three finite values in each distribution")
    grid = (np.arange(len(model)) + .5) / len(model)
    unique, index, counts = np.unique(model, return_index=True, return_counts=True)
    rank = np.asarray([grid[i:i + n].mean() for i, n in zip(index, counts, strict=True)])
    percentile = np.interp(np.asarray(values, dtype=float), unique, rank)
    return np.interp(percentile, (np.arange(len(observed)) + .5) / len(observed), observed)


def terciles(values, history):
    values = np.asarray(values, dtype=float)
    history = np.asarray(history, dtype=float)
    values, history = values[np.isfinite(values)], history[np.isfinite(history)]
    if len(values) < 3 or len(history) < 20:
        raise ValueError("Insufficient ensemble or county baseline for terciles")
    lower, upper = np.quantile(history, [1 / 3, 2 / 3])
    # Boundary ties go to near normal, ensuring a partition even with zero rain.
    below, above = float(np.mean(values < lower)), float(np.mean(values > upper))
    return {"below": below, "near": 1 - below - above, "above": above}


def _get(url, params=None):
    """Ordinary verified HTTPS first; install existing DNS fallback only on DNS failure."""
    try:
        response = requests.get(url, params=params, timeout=(5, 20))
    except requests.ConnectionError as exc:
        if not any(t in str(exc).lower() for t in ("name resolution", "getaddrinfo", "nodename", "resolve")):
            raise
        from fieldshift.net import install_dns_fallback
        install_dns_fallback()
        response = requests.get(url, params=params, timeout=(5, 20))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError("Open-Meteo returned no usable forecast object")
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def fetch(adm_id, latitude, longitude, now=None, root=None):
    """Cache raw responses by actual model initialization; check for new runs daily.

    A daily failure cache prevents repeated slow outages from delaying each brief.
    A good cache is accepted for 24 h; stale data are not silently called current.
    """
    now = now or datetime.now(UTC)
    directory = Path(root or ROOT) / adm_id
    latest = directory / "latest.json"
    if latest.exists():
        cached = json.loads(latest.read_text())
        checked = datetime.fromisoformat(cached["checked_at"])
        age = (now - checked).total_seconds()
        if 0 <= age < (600 if cached.get("error") else 86400):
            if cached.get("error"):
                raise ValueError(cached["error"])
            path = directory / cached["file"]
            value = json.loads(path.read_text())
            if value["issue_date"] == cached["issue_date"]:
                return value
    try:
        meta = _get(METADATA)
        stamp = meta["last_run_initialisation_time"]
        issue = datetime.fromtimestamp(stamp, UTC)
        if not now - timedelta(days=45) <= issue <= now:
            raise ValueError("SEAS5 initialization is missing, future-dated or over 45 days old")
        filename = issue.strftime("%Y-%m-%dT%H%M%SZ.json")
        path = directory / filename
        if path.exists():
            result = json.loads(path.read_text())
        else:
            first = issue.date().replace(day=1)
            offset = first.year * 12 + first.month - 1 + 7
            end = datetime(offset // 12, offset % 12 + 1, 1, tzinfo=UTC).date() - timedelta(days=1)
            params = {"latitude": float(latitude), "longitude": float(longitude),
                      "models": "ecmwf_seas5", "daily": ",".join(DAILY), "monthly": ",".join(MONTHLY),
                      "start_date": first.isoformat(), "end_date": end.isoformat(),
                      "forecast_months": 7, "timezone": "GMT"}
            response = _get(ENDPOINT, params)
            # Parsing now avoids marking a bad/empty response as a valid cache.
            member_arrays(response)
            result = {"issue_date": issue.isoformat(), "available_at": meta.get("last_run_availability_time"),
                      "fetched_at": now.isoformat(), "source": ENDPOINT, "params": params,
                      "metadata": meta, "response": response}
            _write(path, result)
        _write(latest, {"checked_at": now.isoformat(), "file": filename, "issue_date": result["issue_date"]})
        return result
    except Exception as exc:
        # Do not persist URLs with coordinates or exception traces in the brief.
        reason = _reason(exc)
        _write(latest, {"checked_at": now.isoformat(), "error": reason})
        raise ValueError(reason) from exc


def _reason(exc):
    if isinstance(exc, requests.Timeout):
        return "Open-Meteo request timed out"
    if isinstance(exc, requests.HTTPError):
        return f"Open-Meteo HTTP {exc.response.status_code}"
    if isinstance(exc, requests.ConnectionError):
        return "Open-Meteo connection/DNS failure"
    message = re.sub(r"https?://\S+", "[remote endpoint]", str(exc))
    return message[:180] or type(exc).__name__


def member_arrays(response):
    """Live JSON: unnumbered control member, then _member01 ... _member50."""
    daily = response["daily"]
    dates = pd.DatetimeIndex(daily["time"])
    if not len(dates) or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Seasonal dates missing or invalid")
    arrays = {}
    for name in DAILY:
        keys = [name] + [f"{name}_member{i:02d}" for i in range(1, 51)]
        if any(k not in daily for k in keys):
            raise ValueError(f"Expected all 51 members for {name}")
        arr = np.asarray([daily[k] for k in keys], dtype=float)
        if arr.shape != (51, len(dates)) or not np.isfinite(arr).any():
            raise ValueError(f"Empty/malformed seasonal variable {name}")
        arrays[name] = arr
    return dates, arrays


def record(farm):
    """County observations (1991–2020), not a forecast-ensemble self-calibration."""
    d = farm.site["daily"]
    return d.loc[d.date.dt.year.between(1991, 2020)].copy()


def _summary_record(d):
    x = d.copy()
    x["month"], x["year"] = x.date.dt.month, x.date.dt.year
    # POWER tavg is the model's daily mean; ET0 remains historical FAO-56.
    return x.groupby(["year", "month"]).agg(rain=("prec", "sum"), temp=("tavg", "mean"),
                                           et0=("et0", "sum"), tmin=("tmin", "mean"))


def summarize(farm, cached, now=None):
    now = now or datetime.now(UTC)
    response = cached["response"]
    dates, arrays = member_arrays(response)
    monthly = response["monthly"]
    times = pd.DatetimeIndex(monthly["time"])
    baseline = record(farm)
    hist = _summary_record(baseline)
    o = farm.record.get("overrides", {})
    factor = 1 + o.get("rain_change_pct", 0) / 100
    delta = o.get("temp_change_c", 0)
    corrected = {k: np.full_like(v, np.nan) for k, v in arrays.items()}
    rows = []
    for i, date in enumerate(times):
        if date.to_period("M") < pd.Timestamp(now.date()).to_period("M"):
            continue
        m = (dates.year == date.year) & (dates.month == date.month)
        days = calendar.monthrange(date.year, date.month)[1]
        # Never scale a partial month to a whole month, nor silently omit null days.
        if m.sum() != days or any(not np.isfinite(a[:, m]).all() for a in arrays.values()):
            continue
        h = hist.xs(date.month, level="month")
        if len(h) != 30:
            raise ValueError("Need all 30 county baseline years for each forecast month")
        rain_mean = float(monthly["precipitation_mean"][i])
        rain_anomaly = float(monthly["precipitation_anomaly"][i])
        temp_mean = float(monthly["temperature_2m_mean"][i])
        temp_anomaly = float(monthly["temperature_2m_anomaly"][i])
        model_rain_normal = rain_mean - rain_anomaly
        model_temp_normal = temp_mean - temp_anomaly
        if not np.isfinite([model_rain_normal, model_temp_normal]).all() or model_rain_normal <= 0:
            raise ValueError("Provider monthly model climatology unavailable; raw forecast is not compared to county normals")
        scale = float(h.rain.mean()) / model_rain_normal
        shift = float(h.temp.mean()) - model_temp_normal
        for key in arrays:
            if key == "precipitation_sum":
                corrected[key][:, m] = arrays[key][:, m] * scale * factor
            elif key.startswith("temperature"):
                corrected[key][:, m] = arrays[key][:, m] + shift + delta
            else:
                # No model ET0 climatology is exposed. Leave it raw and flag the recharge proxy.
                corrected[key][:, m] = arrays[key][:, m]
        if delta:
            # Same hypothetical temperature-override factor used by the simulator;
            # this is not a correction of the forecast model's ET0 bias.
            tbase = corrected["temperature_2m_mean"][:, m] - delta
            corrected["et0_fao_evapotranspiration_sum"][:, m] *= np.maximum(
                0, (tbase + delta + 17.8) / np.maximum(tbase + 17.8, 1))
        rain = corrected["precipitation_sum"][:, m].sum(1)
        temp = corrected["temperature_2m_mean"][:, m].mean(1)
        rh, th = h.rain.to_numpy() * factor, h.temp.to_numpy() + delta
        rows.append({"month": date.strftime("%Y-%m"), "rain": terciles(rain, rh),
                     "temperature": terciles(temp, th), "rain_median_mm": float(np.median(rain)),
                     "rain_normal_mm": float(np.mean(rh)), "temperature_median_c": float(np.median(temp)),
                     "temperature_normal_c": float(np.mean(th)), "members": len(rain),
                     "rain_scale": scale, "temperature_offset_c": shift})
    if not rows:
        raise ValueError("No complete, finite forecast months with usable model-climatology anomalies")
    events = _events(dates, corrected, baseline, factor, delta, now)
    return {"issue_date": cached["issue_date"], "rows": rows, "events": events, "caveat": CAVEAT,
            "method": "provider-hindcast-mean anomaly transfer; no empirical quantile mapping",
            "source": DOCS}


def _events(dates, corrected, baseline, factor, delta, now):
    events = []
    valid = np.isfinite(corrected["precipitation_sum"]).all(axis=0)
    horizon = set(dates[valid].strftime("%Y-%m-%d"))
    year = now.year

    def window(start, end):
        ds = pd.date_range(start, end)
        if ds[0].date() < now.date() or not set(ds.strftime("%Y-%m-%d")) <= horizon:
            return None
        return (dates >= ds[0]) & (dates <= ds[-1])

    # Only report a whole-event chance when every day is within future forecast coverage.
    autumn = window(f"{year}-09-01", f"{year}-11-30")
    if autumn is None:
        events.append("Dry autumn / cover establishment: full future September–November is outside the horizon; no probability.")
    else:
        h = baseline.loc[baseline.date.dt.month.isin([9, 10, 11])].groupby(baseline.date.dt.year).prec.sum()
        rain = corrected["precipitation_sum"][:, autumn].sum(1)
        p = terciles(rain, h.to_numpy() * factor)["below"]
        events.append(f"Dry autumn: {p:.0%} of members put September–November rain below the county lower tercile "
                      "(establishment-rain proxy, not stand survival).")
    spring_year = year if now.month < 5 else year + 1
    spring = window(f"{spring_year}-03-01", f"{spring_year}-04-30")
    if spring is None:
        events.append("Spring recharge: full future March–April is outside the horizon; no probability.")
    else:
        d = baseline.loc[baseline.date.dt.month.isin([3, 4])]
        # Apply farm rain/temperature changes consistently to the historical reference.
        et0 = d.et0.to_numpy()
        if delta:
            et0 = et0 * np.maximum(0, (d.tavg.to_numpy() + delta + 17.8)
                                  / np.maximum(d.tavg.to_numpy() + 17.8, 1))
        balance = pd.Series(d.prec.to_numpy() * factor - et0, index=d.date)
        history = balance.groupby(balance.index.year).sum().to_numpy()
        proxy = (corrected["precipitation_sum"][:, spring]
                 - corrected["et0_fao_evapotranspiration_sum"][:, spring]).sum(1)
        p = terciles(proxy, history)["below"]
        events.append(f"Spring recharge opportunity: {p:.0%} of members put March–April rain minus reference ET below "
                      "the county lower tercile. This is NOT soil-water prediction; forecast ET0 is uncorrected "
                      "(no model ET0 climatology), so this probability is especially uncertain.")
    frost = baseline.assign(year=baseline.date.dt.year, doy=baseline.date.dt.day_of_year)
    last = frost.loc[frost.tmin.add(delta).le(0) & frost.date.dt.month.le(6)].groupby("year").doy.max()
    if len(last) < 20:
        events.append("Late frost: too few baseline frost years for a threshold; no probability.")
    else:
        doy = int(np.median(last))
        start = pd.Timestamp(f"{spring_year}-01-01") + pd.Timedelta(days=doy)
        end = pd.Timestamp(f"{spring_year}-05-31")
        m = window(start, end) if start <= end else None
        if m is None:
            events.append("Late frost: the full late-spring window is outside the horizon; no probability.")
        else:
            p = float(np.mean((corrected["temperature_2m_min"][:, m] <= 0).any(1)))
            events.append(f"Late frost: {p:.0%} of members have Tmin ≤0 °C after baseline median last-spring-frost "
                          f"day {doy} through May 31. Monthly mean-temperature correction is applied to Tmin too; "
                          "daily frost tails are not calibrated, so this is very low-confidence guidance.")
    return events


def _prob(values):
    # Largest-remainder rounding: the displayed triplet also sums to 100.0%.
    raw = np.asarray([values[k] for k in ("below", "near", "above")]) * 1000
    rounded = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - rounded))[:1000 - rounded.sum()]:
        rounded[i] += 1
    return "/".join(f"{v / 10:.1f}" for v in rounded)


def render(report, anonymous=False, year0=2025):
    issue = datetime.fromisoformat(report["issue_date"])
    issue_text = f"year {issue.year-year0:+d}, {issue:%b %d}" if anonymous else f"{issue:%Y-%m-%d %H:%M} UTC"
    lines = [
        (f"Source: Open-Meteo / ECMWF SEAS5, 51 members at ~36 km; initialized {issue_text}. "
         "This outlook uses SEAS5 only (not the separate 46-day EC46 model)."),
        report["caveat"],
        "Below/near/above are member percentages against the county's "
        + ("30-year climate baseline ending at year -5." if anonymous else "1991–2020 terciles.")
        + " Current-month totals include elapsed days of that initialized forecast; not observed month-to-date rain.",
        "| Month | Rain below/near/above % | Rain median / normal mm | Temperature below/near/above % | Temperature median / normal °C |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        date = datetime.strptime(row["month"], "%Y-%m")
        month = f"{date:%b} year {date.year-year0:+d}" if anonymous else row["month"]
        lines.append(f"| {month} | {_prob(row['rain'])} | {row['rain_median_mm']:.1f} / {row['rain_normal_mm']:.1f} "
                     f"| {_prob(row['temperature'])} | {row['temperature_median_c']:.1f} / {row['temperature_normal_c']:.1f} |")
    lines.extend(report["events"])
    lines.extend([
        ("Beyond this horizon the simulator uses historical weather years. This forecast is context for the agent; "
         "it does not replace simulator weather or imply daily weather is predictable months ahead."),
        ("Data attribution: Open-Meteo (https://open-meteo.com/) and ECMWF, CC BY 4.0. "
         "Free API is non-commercial only (<10,000 calls/day, <5,000/hour, <600/minute); commercial deployment needs a subscription."),
    ])
    return "\n\n".join(lines[:3]) + "\n\n" + "\n".join(lines[3:])


def build(farm, now=None, root=None):
    """Never let network, cache, schema or forecast failures break a farm brief."""
    try:
        from fieldshift.planner.data import _static
        raw = _static().loc[farm.record["adm_id"]]
        cached = fetch(farm.record["adm_id"], raw.centroid_lat, raw.centroid_lon, now, root)
        report = summarize(farm, cached, now)
        return render(report, anonymous=farm.record.get("anonymous", False))
    except Exception as exc:  # noqa: BLE001 -- this optional hook must never break the farm brief
        LOG.warning("Seasonal outlook unavailable: %s", _reason(exc))
        return ("No forecast available: " + _reason(exc) + ". "
                "The simulator and agent must use the historical climate record instead; no seasonal signal is assumed.")
