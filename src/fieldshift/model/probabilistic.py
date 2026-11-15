"""Accelerator-resident heteroscedastic ensembles and inner-year calibration."""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fieldshift.model.neural import DeviceAdamW, accelerator_device

SEEDS = (11, 23, 37, 51, 79)
# Thermal-time axis: 112 bins from sowing/green-up to the crop's development
# horizon (maturity, or six forage cut intervals) plus a 16-bin post-horizon tail.
THERMAL_TRAITS = (
    "base_c", "upper_c", "maturity_gdd", "emergence_gdd", "flowering_gdd",
    "flowering_duration_gdd", "perennial", "cut_interval_gdd", "autumn_sown",
    "vernalization_min_c", "vernalization_max_c", "vernalization_days",
    "dormancy_c", "pre_vernalization_gdd",
)
DEVELOPMENT_BINS, TAIL_BINS = 112, 16
FORAGE_HORIZON_CUTS = 6
ARCHITECTURES = {
    "mlp_small": {"legacy": True, "sequence_input": None, "batch": 4096},
    "calendar_cnn": {"legacy": True, "sequence_input": "calendar", "batch": 512},
    "thermal_cnn": {"legacy": True, "sequence_input": "thermal", "batch": 512},
}


def synchronize():
    device = accelerator_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class BatchedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, gain=1., offset=0.):
        super().__init__()
        weights = []
        for seed in SEEDS:
            torch.manual_seed(seed + input_dim * 100 + output_dim)
            w = torch.empty(input_dim, output_dim, device=accelerator_device())
            nn.init.xavier_uniform_(w, gain=gain)
            weights.append(w)
        self.weight = nn.Parameter(torch.stack(weights))
        self.bias = nn.Parameter(torch.full((len(SEEDS), 1, output_dim), offset, device=accelerator_device()))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias


class GroupedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5, stride=3):
        super().__init__()
        weights = []
        for seed in SEEDS:
            torch.manual_seed(seed + in_channels * 100 + out_channels)
            w = torch.empty(out_channels, in_channels, kernel, device=accelerator_device())
            nn.init.kaiming_normal_(w)
            weights.append(w)
        self.weight = nn.Parameter(torch.cat(weights))
        self.bias = nn.Parameter(torch.zeros(len(SEEDS) * out_channels, device=accelerator_device()))
        self.stride, self.kernel = stride, kernel

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, stride=self.stride,
                        padding=self.kernel // 2, groups=len(SEEDS))


class ResidentWeather:
    """One device allocation for all county-days; windows gather on-device."""
    def __init__(self, path, meta, extras=True):
        self.meta = meta
        source = np.memmap(path, mode="r", dtype=meta.get("dtype", "float32"), shape=tuple(meta["shape"]))
        self.values = torch.tensor(source, dtype=torch.float16, device=accelerator_device())
        names = meta["variables"]
        required = ["tmin", "tmax", "tavg", "prec", "rad", "vpd", "et0", "cwb"]
        required.append("rsm" if "rsm" in names else "gwetroot")
        if extras:
            required += [c for c in ("rad_filled", "rh", "wind", "gwettop", "gwetprof") if c in names]
        scales = {
            "tmin": 30, "tmax": 30, "tavg": 30, "prec": 10, "rad": 20,
            "vpd": 2, "et0": 5, "cwb": 10, "rsm": meta["rsm_scale"],
            "gwetroot": meta["rsm_scale"], "rad_filled": 1, "rh": 100,
            "wind": 10, "gwettop": 1, "gwetprof": 1,
        }
        self.input_channels = 2 * len(required)
        # weather + availability, calendar days per bin, 5 stage indicators, forage cycle phase
        self.thermal_channels = 2 * len(required) + 7
        self.columns = torch.tensor([names.index(c) for c in required], device=accelerator_device())
        self.scales = torch.tensor(
            [scales[c] for c in required],
            dtype=torch.float32, device=accelerator_device()
        )
        self.offsets = torch.arange(366, device=accelerator_device())

    def _raw(self, county, start, length):
        days = start[:, None] + self.offsets[None, :]
        valid = (self.offsets[None, :] < length[:, None]) & days.ge(0) & days.lt(self.values.shape[1])
        data = self.values[county[:, None], days.clamp(0, self.values.shape[1] - 1)]
        data = data.index_select(2, self.columns).float()
        if self.meta.get("vpd_to_tmax"):
            low, high = data[..., 0], data[..., 1]
            es_max = .6108 * torch.exp(17.27 * high / (high + 237.3))
            es_min = .6108 * torch.exp(17.27 * low / (low + 237.3))
            data[..., 5] = data[..., 5] + .5 * (es_max - es_min)
        return data, valid

    def thermal_windows(self, county, start, window, jan1, traits):
        """Resample daily weather onto each crop's own cumulative-GDD axis, on-device.

        GDD follows ``features.stress.daily_stages`` (clipped mean temperature,
        winter dormancy and pre-vernalisation cap). Each bin averages the days
        whose GDD-before falls in it; bins crossed within one hot day take that
        day's values. Calendar days per bin (development speed) is a channel.
        """
        n, days = len(county), len(self.offsets)
        raw, inside = self._raw(county, start, window)
        tmin, tmax, tavg = raw[..., 0], raw[..., 1], raw[..., 2]
        ok = inside & torch.isfinite(tmin) & torch.isfinite(tmax)
        t = {k: traits[:, i] for i, k in enumerate(THERMAL_TRAITS)}
        base, upper = t["base_c"][:, None], t["upper_c"][:, None]
        low = torch.minimum(torch.maximum(tmin, base), upper)
        high = torch.minimum(torch.maximum(tmax, base), upper)
        gdd = torch.where(ok, ((low + high) / 2 - base).clamp_min(0), 0.)
        autumn = t["autumn_sown"].gt(.5)
        if bool(autumn.any()):
            mean_t = torch.where(torch.isfinite(tavg), tavg, (tmin + tmax) / 2)
            chill = ok & mean_t.ge(t["vernalization_min_c"][:, None]) & mean_t.le(t["vernalization_max_c"][:, None])
            vernalized = chill.float().cumsum(1).ge(t["vernalization_days"][:, None])
            unlocked = vernalized & self.offsets[None, :].ge(jan1[:, None])
            dormant = torch.where(mean_t.lt(t["dormancy_c"][:, None]), 0., gdd)
            locked = torch.minimum(torch.where(unlocked, 0., dormant).cumsum(1),
                                   t["pre_vernalization_gdd"][:, None])
            increments = torch.diff(locked, dim=1, prepend=torch.zeros_like(locked[:, :1]))
            gdd = torch.where(autumn[:, None], torch.where(unlocked, dormant, increments), gdd)
        cumulative = gdd.cumsum(1)
        before = cumulative - gdd
        perennial = t["perennial"].gt(.5)
        horizon = torch.where(perennial, FORAGE_HORIZON_CUTS * t["cut_interval_gdd"], t["maturity_gdd"])
        width = (horizon / DEVELOPMENT_BINS)[:, None]
        bins = DEVELOPMENT_BINS + TAIL_BINS
        position = (before / width).floor()
        use = ok & position.lt(bins)
        flat = torch.where(use, torch.arange(n, device=gdd.device)[:, None] * bins
                           + position.clamp(0, bins - 1).long(), n * bins).reshape(-1)
        x = raw / self.scales
        v = x.shape[-1]
        finite = torch.isfinite(x) & use[..., None]
        sums = torch.zeros(n * bins + 1, v, device=gdd.device).index_add_(
            0, flat, torch.where(finite, x, 0.).reshape(-1, v))[:-1].view(n, bins, v)
        counts = torch.zeros(n * bins + 1, v, device=gdd.device).index_add_(
            0, flat, finite.float().reshape(-1, v))[:-1].view(n, bins, v)
        day_count = torch.zeros(n * bins + 1, device=gdd.device).index_add_(
            0, flat, use.float().reshape(-1))[:-1].view(n, bins)
        centre = (torch.arange(bins, device=gdd.device)[None, :] + .5) * width
        spanning_day = torch.searchsorted(cumulative.contiguous(), centre.contiguous())
        clamped = spanning_day.clamp(max=days - 1)
        spanning = spanning_day.lt(days) & ok.gather(1, clamped)
        fallback = x.gather(1, clamped[..., None].expand(-1, -1, v))
        fallback_ok = spanning[..., None] & torch.isfinite(fallback)
        available = counts.gt(0) | fallback_ok
        mean = torch.where(counts.gt(0), sums / counts.clamp_min(1), torch.where(fallback_ok, fallback, 0.))
        mean = torch.where(available, mean.clamp(-10, 10), 0.)
        bounds = torch.stack([t["emergence_gdd"], t["flowering_gdd"],
                              t["flowering_gdd"] + t["flowering_duration_gdd"], t["maturity_gdd"]], 1)
        stage = (centre[..., None] >= bounds[:, None, :]).sum(-1)
        annual = (~perennial)[:, None, None].float()
        indicators = F.one_hot(stage, 5).float() * annual
        phase = torch.where(perennial[:, None], torch.frac(centre / t["cut_interval_gdd"][:, None]), 0.)
        return torch.cat([mean, available.float(), (day_count / 5)[..., None], indicators,
                          phase[..., None]], dim=2).transpose(1, 2)

    def windows(self, county, start, length):
        data, valid = self._raw(county, start, length)
        data = data / self.scales
        available = torch.isfinite(data) & valid.unsqueeze(-1)
        data = torch.where(available, data.clamp(-10, 10), 0.)
        # Missing-data flags distinguish dry days from absent POWER/GLDAS values.
        return torch.cat([data, available.float()], dim=2).transpose(1, 2)


class ResidentRows:
    def __init__(self, data, columns, target):
        self.x = torch.tensor(data[columns].to_numpy(dtype=np.float32), device=accelerator_device())
        self.y = torch.tensor(target, dtype=torch.float32, device=accelerator_device())
        self.county = torch.tensor(data.county_index.to_numpy(), dtype=torch.long, device=accelerator_device())
        self.start = torch.tensor(data.start_index.to_numpy(), dtype=torch.long, device=accelerator_device())
        self.length = torch.tensor(data.season_length.to_numpy(), dtype=torch.long, device=accelerator_device())
        counts = data.groupby(["crop", "adm_id", "year"]).year.transform("size").to_numpy()
        self.weight = torch.tensor(1. / counts, dtype=torch.float32, device=accelerator_device())
        trait_columns = [f"trait__{k}" for k in THERMAL_TRAITS]
        if {"year", "season_start_ordinal", "trait__season_end_month"}.issubset(data) and set(trait_columns).issubset(data):
            start = data.season_start_ordinal.to_numpy(dtype=np.int64)
            year = data.year.to_numpy(dtype=np.int64)
            month = data.trait__season_end_month.to_numpy(dtype=np.int64)
            ends = {(y, m): pd.Timestamp(int(y), int(m), 31).toordinal()
                    for y, m in set(zip(year.tolist(), month.tolist()))}
            window_end = np.array([ends[k] for k in zip(year.tolist(), month.tolist())], dtype=np.int64)
            # Annual crops: calendar window through the harvest-year end month, so the
            # post-maturity tail is observable. Forage: green-up to killing frost.
            window = np.where(data.trait__perennial.to_numpy() > .5, data.season_length.to_numpy(),
                              np.minimum(window_end - start + 1, 366))
            first = {y: pd.Timestamp(int(y), 1, 1).toordinal() for y in set(year.tolist())}
            jan1 = np.array([first[y] for y in year.tolist()], dtype=np.int64) - start
            self.window = torch.tensor(window, dtype=torch.long, device=accelerator_device())
            self.jan1 = torch.tensor(jan1, dtype=torch.long, device=accelerator_device())
            self.thermal = torch.tensor(data[trait_columns].to_numpy(dtype=np.float32), device=accelerator_device())

    def indices(self, mask):
        return torch.tensor(np.flatnonzero(mask), dtype=torch.long, device=accelerator_device())


class ResponseEnsemble(nn.Module):
    def __init__(self, n_features, trait_indices, sequence=False, sequence_channels=18,
                 sequence_input="calendar"):
        super().__init__()
        self.sequence = sequence
        self.sequence_input = sequence_input if sequence else None
        self.name = None
        self.n_features = n_features
        self.sequence_channels = sequence_channels
        self.trait_indices_list = trait_indices
        self.register_buffer("trait_indices", torch.tensor(trait_indices, dtype=torch.long, device=accelerator_device()))
        self.register_buffer("center", torch.zeros(n_features, device=accelerator_device()))
        self.register_buffer("scale", torch.ones(n_features, device=accelerator_device()))
        self.register_buffer("calibration_scale", torch.ones((), device=accelerator_device()))
        self.first = BatchedLinear(n_features, 128)
        self.second = BatchedLinear(128, 64)
        if sequence:
            self.convolutions = nn.ModuleList([GroupedConv(sequence_channels, 16), GroupedConv(16, 24), GroupedConv(24, 24)])
            self.film = BatchedLinear(len(trait_indices), 48, gain=.1)
            self.fusion = BatchedLinear(64 + 24 * 4, 64)
        self.mean_head = BatchedLinear(64, 1, gain=.1)
        self.variance_head = BatchedLinear(64, 1, gain=.05, offset=-3.)

    def fit_scaler(self, x):
        self.center.copy_(torch.nan_to_num(torch.nanmean(x, dim=0)))
        scale = torch.nan_to_num(torch.nanmean((x - self.center) ** 2, dim=0).sqrt(), nan=1.)
        self.scale.copy_(torch.where(scale.gt(1e-5), scale, 1.))

    def forward(self, x, weather=None):
        x = torch.nan_to_num((x - self.center) / self.scale).clamp(-8, 8)
        x = x.unsqueeze(0).expand(len(SEEDS), -1, -1)
        hidden = F.silu(self.first(x))
        hidden = F.silu(self.second(hidden))
        if self.sequence:
            # Independent ensemble CNNs as grouped convolutions, one device batch.
            temporal = weather.repeat(1, len(SEEDS), 1)
            for convolution in self.convolutions:
                temporal = F.silu(convolution(temporal))
            temporal = temporal.reshape(len(weather), len(SEEDS), 24, -1).permute(1, 0, 2, 3)
            film = self.film(x.index_select(2, self.trait_indices))
            gamma, beta = film.chunk(2, dim=-1)
            temporal = temporal * (1 + gamma.tanh().unsqueeze(-1)) + beta.unsqueeze(-1)
            # Preserve coarse relative position instead of discarding all timing.
            # MPS adaptive pooling rejects non-divisible lengths (14 -> 4).
            # Four disjoint mean bins implement the same intent on-device.
            length = temporal.shape[-1]
            temporal = torch.stack([
                temporal[..., i * length // 4:(i + 1) * length // 4].mean(dim=-1)
                for i in range(4)
            ], dim=-1).reshape(len(SEEDS), x.shape[1], 24 * 4)
            hidden = F.silu(self.fusion(torch.cat([hidden, temporal], dim=-1)))
        mean = self.mean_head(hidden).squeeze(-1)
        log_variance = self.variance_head(hidden).squeeze(-1).clamp(-8., 1.)
        return mean, log_variance.exp()

    def distribution(self, x, weather=None):
        means, variances = self(x, weather)
        mean = means.mean(dim=0)
        variance = (variances + means.square()).mean(dim=0) - mean.square()
        return mean, variance.clamp_min(1e-8).sqrt() * self.calibration_scale


def _batch_weather(rows, index, weather, net):
    if not net.sequence:
        return None
    if net.sequence_input == "thermal":
        return weather.thermal_windows(rows.county[index], rows.start[index], rows.window[index],
                                       rows.jan1[index], rows.thermal[index])
    return weather.windows(rows.county[index], rows.start[index], rows.length[index])


def build_ensemble(architecture, n_features, trait_indices, weather=None):
    """None keeps the original small MLP (or calendar CNN with sequence=True)."""
    spec = ARCHITECTURES[architecture]
    channels = 0
    if spec.get("sequence_input") == "calendar":
        channels = weather.input_channels
    elif spec.get("sequence_input") == "thermal" or spec.get("sequence"):
        channels = weather.thermal_channels
    net = ResponseEnsemble(n_features, trait_indices, spec["sequence_input"] is not None,
                           channels or 18, spec["sequence_input"] or "calendar")
    net.name = architecture
    return net, spec


def train_response(rows, train_index, trait_indices, epochs=40, weather=None, sequence=False,
                   batch_size=None, architecture=None, amp=None):
    if architecture is None:
        architecture = "calendar_cnn" if sequence else "mlp_small"
    net, spec = build_ensemble(architecture, rows.x.shape[1], trait_indices, weather)
    net.fit_scaler(rows.x[train_index])
    device = accelerator_device()
    optimizer = DeviceAdamW(net.parameters(), lr=spec.get("lr", .0007), weight_decay=.03)
    batch_size = batch_size or spec["batch"]
    # Mixed precision only on CUDA tensor cores; dynamic loss scale stays on-device.
    amp = device.type == "cuda" and not spec.get("legacy") if amp is None else amp
    loss_scale = torch.tensor(2. ** 12 if amp else 1., device=device)
    timing, epoch_losses = [], torch.zeros(epochs, device=device)
    torch.manual_seed(2026)
    net.train()
    for epoch in range(epochs):
        start = time.monotonic()
        order = train_index[torch.randperm(len(train_index), device=device)]
        for index in order.split(batch_size):
            optimizer.zero_grad()
            weather_batch = _batch_weather(rows, index, weather, net)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                mean, variance = net(rows.x[index], weather_batch)
            mean, variance = mean.float(), variance.float()
            error = mean - rows.y[index].unsqueeze(0)
            if epoch < min(10, epochs // 3):
                loss = error.square()  # stable mean warmup; no test-dependent schedule
            else:
                loss = .5 * (variance.log() + error.square() / variance)
            loss = (loss * rows.weight[index].unsqueeze(0)).sum(dim=1) / rows.weight[index].sum()
            epoch_losses[epoch] += loss.mean().detach() * len(index) / len(train_index)
            (loss.sum() * loss_scale).backward()
            if amp:
                grads = [p.grad for p in net.parameters() if p.grad is not None]
                torch._foreach_div_(grads, loss_scale)
                finite = torch.stack([g.isfinite().all() for g in grads]).all()
                for g in grads:
                    g.copy_(torch.where(finite, g, 0.))
                loss_scale = torch.where(finite, (loss_scale * 1.001).clamp(max=2. ** 16), loss_scale / 2)
            # Prevent rare drought observations causing numerical runaway.
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., foreach=False)
            optimizer.step()
        synchronize()
        timing.append(time.monotonic() - start)
    net.eval()
    return net, {"architecture": architecture, "epochs": epochs, "epoch_seconds": timing,
                 "median_epoch_seconds": float(np.median(timing)), "amp": bool(amp),
                 "epoch_train_loss": epoch_losses.tolist(),
                 "parameters": sum(p.numel() for p in net.parameters()),
                 "train_rows": len(train_index), "device": str(device)}


@torch.inference_mode()
def loss_summary(net, rows, index, weather=None):
    """Uncalibrated ensemble Gaussian NLL, MSE and R² (row weights as in training)."""
    predicted = distribution(net, rows, index, weather) if len(index) else None
    if predicted is None:
        return {}
    std = predicted[1] / net.calibration_scale
    y, w = rows.y[index], rows.weight[index]
    w = w / w.sum()
    error = predicted[0] - y
    nll = .5 * (torch.log(2 * torch.pi * std.square()) + error.square() / std.square())
    variance = (w * (y - (w * y).sum()).square()).sum()
    values = torch.stack([(w * nll).sum(), (w * error.square()).sum(),
                          1 - (w * error.square()).sum() / variance])
    nll, mse, r2 = values.tolist()
    return {"nll": nll, "mse": mse, "r2": r2, "rows": len(index)}


@torch.inference_mode()
def distribution(net, rows, index, weather=None, batch_size=4096):
    # Store output on-device and cross to host only once after all batches.
    result = torch.empty((2, len(index)), device=accelerator_device())
    position = 0
    size = 512 if net.sequence else min(batch_size, getattr(net, "spec", {}).get("batch", batch_size))
    for part in index.split(size):
        mean, std = net.distribution(rows.x[part], _batch_weather(rows, part, weather, net))
        result[0, position:position + len(part)] = mean
        result[1, position:position + len(part)] = std
        position += len(part)
    return result


@torch.inference_mode()
def recalibrate(net, rows, calibration_index, weather=None):
    if len(calibration_index) < 100:
        raise ValueError("Too few inner-year calibration observations")
    predicted = distribution(net, rows, calibration_index, weather)
    normalized_error = (rows.y[calibration_index] - predicted[0]).abs() / predicted[1].clamp_min(.005)
    q = torch.quantile(normalized_error, torch.tensor([.682689, .9545], device=accelerator_device()))
    # One SD multiplier, compromise between the 68% and 95% empirical radii.
    scale = (q[0] * q[1] / 2).sqrt().clamp(.25, 4.)
    net.calibration_scale.copy_(scale)
    return {"scale": float(scale.item()), "calibration_rows": len(calibration_index),
            "normalized_abs_error_quantiles": q.tolist(),
            "rule": "sqrt(q68 * q95 / 2), estimated only on inner held-out years"}


def save_response(net, columns, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    torch.save({
        "state": net.state_dict(), "columns": columns, "trait_indices": net.trait_indices_list,
        "sequence": net.sequence, "sequence_channels": net.sequence_channels,
        "architecture": net.name,
        "metadata": metadata, "format": 2,
    }, temporary)
    temporary.replace(path)
    sidecar = path.with_suffix(".json")
    temporary = sidecar.with_name(sidecar.name + ".part")
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(sidecar)
    load_response.cache_clear()


@lru_cache(maxsize=4)
def load_response(path):
    data = torch.load(path, map_location=accelerator_device(), weights_only=True)
    name = data.get("architecture")
    if name and name not in {"mlp_small", "calendar_cnn", "thermal_cnn"}:
        raise ValueError(f"Unsupported architecture {name}; H7/H8 were cancelled before training")
    net = ResponseEnsemble(len(data["columns"]), data["trait_indices"], data["sequence"],
                           data.get("sequence_channels", 18),
                           "thermal" if name == "thermal_cnn" else "calendar")
    net.name = name
    net.load_state_dict(data["state"])
    return net.eval(), data["columns"], data["metadata"]
