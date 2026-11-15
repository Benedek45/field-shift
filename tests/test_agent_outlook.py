"""Offline tests. Snapshot: Open-Meteo SEAS5, initialized Nov 1, fetched Nov 14
2026 for cropland centroid 42.037883/-93.457442. First two actual daily records,
all 51 members, units and first monthly record; fields unchanged, remainder
omitted for size. Source https://seasonal-api.open-meteo.com/v1/seasonal,
Open-Meteo / ECMWF, CC BY 4.0. No network or accelerator in these tests.
"""

import base64
import json
import zlib
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from fieldshift.agent import outlook

RECORDED = (
    "eNqdmsuO5LgRRf8l1zkJxYMPzdZLf8LYKOTYZZiA64HqbGOMQf+7RVEpsaazIy5nWagrknFPBBkp8ffTP6/lP/87/fz76VZenk8//3LiieNPRD9NdDoff/Dp7+fT+8fzP8p7uV1v5e316cvXl0VPl3QmufCjfz+9PL/8+vyxjFR1dM4XsWS8yOQi53DJlkzW0eY6q6nTRTdd+DxfJksWFhmvMrJkcZXRmaZLsHRpXd10TpdkyfLdOrKDmFedeBbTdF9evMyWju7jscmCKou8uhItmazThrOY5pGuslzNM4cLWxTZ9JjiGkReEkUtWUWhNdZgx5rXWeMSq2ndvAWhZj7xtGZx8MgyrasL52iC5QoiVbBsJhRXEmHR2WnMuloXF2CWdRzu6cl2EK0qpvNkD5dWT7QOZ3qX1yAmVzev49HinTWtTOsOUPPE4i+tJkKtMcs7aSjmmsUWMpFtH3OyXXRL42RP21iwt/NIZREX8+wdQNIq80pW8rFRmMPtG5Q5nLb9ad20reGUtmCzaZ3ytmkvRWYlirb9SZ2aVV1rMTobioY1n+ISqwVC41Y8wQ413S1JZjpp3ha3xGp60vantKSTVbJh2mTUiN2eX96fP663rx/PT1xF19e6/FDNbeX1QPHpYA+1dOK6NkPK+KjSpFKlky3VJtUqzbY0NOlUjwNHGpuUKhqxpWmXxjUpDGnuHEi2dO6k9qjrwX+XzraUOl9tWsRdWI5UOrNsWqT7Ajxf106AtLYMHliKew60QjKkqXOAbGnuUtvxdYZ95anz1V4AU2eWMyp3a7Vri6UrGDsJWbu12gg4dAtwpH1tOWtNcGqvjcM9Xewc4HnfB7x0kWmXemCF9rUmB6xwly52cUtPy5FqZ5ZdWxI6qRNW7JLQLm7pa8umJbkb1TFr7qQ2Le1ry5ESvBWvncaSWbNfWyqwr6p7wSSnYHTbCaO/D2iEj+O1+biH5Swgw/uAzvAhv7Yhlq9lbUNq9779zv9e8KkLoVol+jD1y+cmZGptaDCV0sacqjKZSu1mV1MZmpLrmPY6Y1tnqmPasad9nV7seY/dU87d7KZLrfmob3a82FvvgdBsrQc1pek8SRd7NJW6K4OjDG2dfi61tmP95ePFnnaXxFHmLpds5+c9l/ThZlc+txzrrxbHeaZ9zOCMyWjsLF2GmH6ydjRnU7nVUXCrg486Cs7sfR3ZY+Yuk+3Ye0amUqauNk3nhTqlWR3C8JjScZ9MpXaZbOanhG52O/a4++kpE5p1cux1wVHO3exm1mm/15l+KnVK0yXlbqe1x5ROaXLXjdGa82Qq+73Ojj3uFSeOMnWx2+vMXURm1unccTf9DFsdhR/5ef2tvvyoHwUoPyzz62+fmoq5ptD8cNJDyW3M4I8pTUl1TDGV2mZP5yo3laGNKf464312d8zUxT6bytzG1DqmHdG8Oz8/PIJ3ZfuUsUVkjrk2FVyPNm/21lTMbfZkKuXuvDu73mNfuNtjhjZmrutkUxnvzvP0cOM6lKlb52Qqc+enrZzv6/QY8bRncrYzhKmL3XSJeffTWSdL57w9u+6Z7ClDN7vJnWO3h9hjpq7e7djz7nx2nN8YLTlP9uwyoeuUjVGojMw9RLjLJTMikU5p7jainUu2cqsjn5HE+57sVZwkeMzcrdPc52VGc0k3RpO7hyihuaTccTf3EBX0NFTtzg7TTw3d2WEr4z2TvZNLN0Z+1mm/15l1pBuj5Drfmor52D+fb9PTv65vT8//vb6/3T6ur1/ey8cfrkSk+j1MXfHnCxJprk8J/BS3axX1KcnwU+36RFzalcuEr7B9q5Q1Lnyu9ukyrk8F+Knt+xmfCfD7870LkTHj21fOwGMLbN88QzWe4bnay5JLjEOQtwsb1UF8pj0xlp9ABD/VEkNqVAGfqyVGqr7jQW15kauBcDZt1z5CHPJiywsdygvavn7XBQr+1JYX9Sli9Kn2gmZbIWwh02EG/hAfVTzhT8mxQIaLZL9psjwU4YfCscCBqeJRxQnO9/bK55LidnUBeyjfL2OMRDXvNTzN6EMyHVYozGq7yxLHUrC9K1oiGkmm9troouvxA3uxXXpZjwSFK0u6gwRP3PZq6UL1qQgjbh1h26cZx9USI0vdc+HMlSMzIhxWexN1CVQZwwvUP5UZ7Q3VsrihnVrluBSEL3A7SWpYik8VDlp47d+v6NQmiBL81LZjLIgznLrtPddSwkNHic7HnosXV2tVL9o6p2/n08vb6+3fP7ze+90toe3yT04PrkFdX99ernWoXzj/6Jtpu6r4/f+OZ6Wta710/PT1tdy+HGs7lS9vOa7Xjh9cND69vJzsK8aOgl2FuAp1FcFVRFeRXEV2FbOnqO2oo3A9JddTcj0l11NyPSXXU3I9JddTcj1l11N2PWXXU3Y9ZddTdj1l11N2PWXXU3Y9FddTcT0V11NxPRXXU3E9FddTcT0V11NxPVXXU3U9VddTdT1V11N1PVXXU3U9VddTdT0Nu6cPD8bT375O06/TX07efVpMyKhQUKGiwoAKIypMqDCjwhkUrocgJETJEEqGUDKEkiGUDKFkCCVDKBlCyTBKhlEyjJJhlAyjZBglwygZRskwSoZRMoKSEZSMoGQEJSMoGUHJCEpGUDKCkhGUjKJkFCWjKBlFyShKRlEyipJRlIyiZBQlE2wy5dX5N9YfFLA9KGB3UMDmoIC9QQFbgwJ2BgVsDArYFxSwLShgV1DApqCAPUEBW4ICdgQFbAgK2A8UsB0oYDdQwGaggL1AAVuBAnYCBWwECtgHFLANKGAXUMAmoIA9QAFbgAJ2AAVsAAp4/hfw+C/g6V/Aw7+AZ38Bj/4CnvwFPPgLeO4X8Ngv4KlfwEO/gGd+AY/8Ap74BTzwC3jeF/C4r7dQzX9jx/2nG6iQTkCdgroA6iKoS6Aug7oZ09EE6kAeBPIgkAeBPAjkQSAPAnkQyINAHgzyYJAHgzwY5MEgDwZ5MMiDQR4M8mCQh4A8BOQhIA8BeQjIQ0AeAvIQkIeAPATkoSAPBXkoyENBHgryUJCHgjwU5KEgDwV5/OG49+6Htq8E8MXQETmPyWVMrmPyMCaPY/I0Js9j8nlIfnxLh+90jsjHqNIYVRqjSmNUaYwqjVGlMao0RpXHqPIYVR6jymNUeYwqj1HlMao8RpXHqPIYVRmjKmNUZYyqjFGVMaoyRlXGqMoYVRmjKmNUdYyqjlHVMao6RlXHqOoYVR2jqmNUdYyqjlG936k4bjLCdwa3WxcPrmzslxH/1G2N4+m/nr59+z/PowCX"
)


@pytest.fixture
def recorded():
    return json.loads(zlib.decompress(base64.b64decode(RECORDED)))


def test_recorded_members_and_terciles(recorded):
    dates, arrays = outlook.member_arrays(recorded)
    assert len(dates) == 2 and arrays["precipitation_sum"].shape == (51, 2)
    for a in arrays.values():
        probabilities = outlook.terciles(a.mean(1), np.linspace(-5, 20, 30))
        assert sum(probabilities.values()) == pytest.approx(1)
        assert all(0 <= p <= 1 for p in probabilities.values())
        assert sum(map(float, outlook._prob(probabilities).split("/"))) == pytest.approx(100)
    assert outlook.terciles(np.zeros(51), np.zeros(30)) == {"below": 0, "near": 1, "above": 0}


def test_quantile_map_monotone_and_reproduces_synthetic_record():
    model = np.linspace(0, 10, 101)
    observed = model ** 2 + 3
    mapped = outlook.quantile_map(model, model, observed)
    assert np.allclose(mapped, observed)
    assert np.all(np.diff(mapped) >= 0)
    assert np.all(np.diff(outlook.quantile_map(np.linspace(-2, 12, 300), model, observed)) >= 0)
    assert outlook.quantile_map([-100, 100], model, observed).tolist() == [observed[0], observed[-1]]


def test_failure_never_breaks_brief(monkeypatch):
    farm = SimpleNamespace(record={"adm_id": "US-19-169", "anonymous": False})
    monkeypatch.setattr(outlook, "fetch", lambda *_a, **_k: (_ for _ in ()).throw(requests.Timeout()))
    assert "No forecast available: Open-Meteo request timed out" in outlook.build(farm)


def test_raw_response_without_model_climatology_is_not_used(recorded):
    dates = pd.date_range("1991-01-01", "2020-12-31")
    baseline = pd.DataFrame({"date": dates, "prec": 2., "tavg": 10., "tmin": 5., "et0": 1.})
    farm = SimpleNamespace(record={"overrides": {}}, site={"daily": baseline})
    # Deliberately incomplete recorded month: no unprincipled extrapolation.
    with pytest.raises(ValueError, match="No complete"):
        outlook.summarize(farm, {"response": recorded}, datetime(2026, 11, 1, tzinfo=UTC))


def test_complete_month_bias_transfer_and_anonymous_render(recorded):
    # Synthetic full month derived from the recorded member schema. Set zero
    # anomalies so the mean-adjusted ensemble must match the county mean.
    daily = recorded["daily"]
    for key in daily:
        if key != "time":
            daily[key] = np.resize(daily[key], 30).tolist()
    daily["time"] = pd.date_range("2026-11-01", periods=30).strftime("%Y-%m-%d").tolist()
    _, arrays = outlook.member_arrays(recorded)
    recorded["monthly"]["precipitation_mean"] = [float(arrays["precipitation_sum"].sum(1).mean())]
    recorded["monthly"]["precipitation_anomaly"] = [0.]
    recorded["monthly"]["temperature_2m_mean"] = [float(arrays["temperature_2m_mean"].mean())]
    recorded["monthly"]["temperature_2m_anomaly"] = [0.]
    dates = pd.date_range("1991-01-01", "2020-12-31")
    year = dates.year.to_numpy() - 1991
    baseline = pd.DataFrame({"date": dates, "prec": 1 + year / 30,
                             "tavg": 10 + year / 30, "tmin": 2., "et0": 1.})
    farm = SimpleNamespace(record={"overrides": {}}, site={"daily": baseline})
    result = outlook.summarize(farm, {"response": recorded, "issue_date": "2026-11-01T00:00:00+00:00"},
                               datetime(2026, 11, 1, tzinfo=UTC))
    row = result["rows"][0]
    normal_rain = float(baseline.loc[baseline.date.dt.month.eq(11)].prec.sum() / 30)
    assert row["rain_normal_mm"] == pytest.approx(normal_rain)
    expected = np.median(arrays["precipitation_sum"].sum(1)) * normal_rain / arrays["precipitation_sum"].sum(1).mean()
    assert row["rain_median_mm"] == pytest.approx(expected)
    assert sum(row["rain"].values()) == pytest.approx(1)
    assert sum(row["temperature"].values()) == pytest.approx(1)
    anonymous = outlook.render(result, anonymous=True)
    assert "2026" not in anonymous and "1991" not in anonymous
    assert "year +1" in anonymous and "uncalibrated" in anonymous
    assert "No county skill score" in anonymous


def test_issue_date_cache_offline(recorded, tmp_path, monkeypatch):
    now = datetime(2026, 11, 14, tzinfo=UTC)
    calls = []
    def get(url, params=None):
        calls.append(url)
        return {"last_run_initialisation_time": datetime(2026, 11, 1, tzinfo=UTC).timestamp()} if url == outlook.METADATA else recorded
    monkeypatch.setattr(outlook, "_get", get)
    a = outlook.fetch("US-19-169", 42., -93., now, tmp_path)
    b = outlook.fetch("US-19-169", 42., -93., now, tmp_path)
    assert a == b and len(calls) == 2
    assert (tmp_path / "US-19-169" / "2026-11-01T000000Z.json").exists()
