"""Fold-local hierarchical log-linear county yield trends."""

import numpy as np
import pandas as pd


class YieldTrend:
    """County intercepts and slopes shrunk toward crop-wide slopes.

    Slope shrinkage adds 100 year² to the centred design sum of squares.
    Unseen counties fall back to the crop-wide trend; no test labels are used.
    """

    def fit(self, data):
        d = data[["crop", "adm_id", "year", "yield_t_ha"]].copy()
        if d.empty or d.yield_t_ha.le(0).any():
            raise ValueError("Trend requires positive training yields")
        self.training_years = sorted(d.year.unique().tolist())
        d["x"] = d.year - 2013
        d["y"] = np.log(d.yield_t_ha)
        self.global_ = {}
        for crop, g in d.groupby("crop"):
            x, y = g.x.to_numpy(), g.y.to_numpy()
            slope = float(((x - x.mean()) * (y - y.mean())).sum() / max(((x - x.mean()) ** 2).sum(), 1))
            self.global_[crop] = (float(y.mean() - slope * x.mean()), slope)
        d["residual"] = d.y - [
            self.global_[c][0] + self.global_[c][1] * x for c, x in zip(d.crop, d.x)
        ]
        group = d.groupby(["crop", "adm_id"])
        d["xc"] = d.x - group.x.transform("mean")
        d["xy"] = d.xc * d.residual
        d["xx"] = d.xc ** 2
        stats = d.groupby(["crop", "adm_id"]).agg(
            x=("x", "mean"), residual=("residual", "mean"), xy=("xy", "sum"), xx=("xx", "sum")
        )
        stats["slope"] = stats.xy / (stats.xx + 100)
        stats["intercept"] = stats.residual - stats.slope * stats.x
        self.county_ = stats[["intercept", "slope"]]
        return self

    def predict_log(self, data):
        unknown = set(data.crop) - self.global_.keys()
        if unknown:
            raise ValueError(f"No training baseline for crops: {unknown}")
        x = data.year.to_numpy() - 2013
        base = np.array([self.global_[c][0] + self.global_[c][1] * v for c, v in zip(data.crop, x)])
        county = self.county_.reindex(pd.MultiIndex.from_frame(data[["crop", "adm_id"]])).fillna(0)
        return base + county.intercept.to_numpy() + county.slope.to_numpy() * x

    def transform(self, data):
        return np.log(data.yield_t_ha.to_numpy()) - self.predict_log(data)
