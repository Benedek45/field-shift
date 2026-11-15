"""Recursive, mass-conserving field-state ensemble. No future state observations."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from fieldshift.model.neural import accelerator_device
from fieldshift.model.probabilistic import SEEDS, BatchedLinear
from fieldshift.model.recursive_data import DRIVERS, SLOTS, STATE_SCALES


def apply_linear(layer, x):
    shape = x.shape
    return layer(x.reshape(shape[0], -1, shape[-1])).reshape(*shape[:-1], -1)


def water_balance(wetness, precipitation, et0, days, irrigated_share, capacity, logits):
    """Three proxy reservoirs; all transfers/outputs are availability limited.

    Returns new wetness and P/I/ET/runoff/drainage/mass residual in mm/period.
    Root/surface/profile are effective, non-overlapping surrogate reservoirs,
    not an assertion that the overlapping MERRA products measure disjoint masses.
    """
    initial = wetness * capacity
    root, surface, deep = initial.unbind(-1)
    root_cap, surface_cap, deep_cap = capacity.unbind(-1)
    target = .6 + .3 * logits[..., 2].sigmoid()
    irrigation = (target * root_cap - root - .7 * precipitation).clamp_min(0)
    irrigation = torch.minimum(irrigation, 20 * days) * irrigated_share
    incoming = precipitation + irrigation
    runoff = incoming * logits[..., 0].sigmoid()
    surface = surface + incoming - runoff
    duration = days / 8
    capillary = deep * logits[..., 7].sigmoid() * .1 * (1 - wetness[..., 0]) * duration
    deep, root = deep - capillary, root + capillary
    demand = et0 * (.1 + 1.9 * logits[..., 1].sigmoid())
    surface_et = torch.minimum(surface, demand * logits[..., 3].sigmoid())
    surface = surface - surface_et
    infiltration = (surface - surface_cap).clamp_min(0)
    infiltration = infiltration + torch.minimum(surface, surface_cap) * logits[..., 4].sigmoid() * duration
    surface, root = surface - infiltration, root + infiltration
    root_et = torch.minimum(root, (demand - surface_et).clamp_min(0))
    root = root - root_et
    percolation = (root - root_cap).clamp_min(0)
    percolation = percolation + torch.minimum(root, root_cap) * logits[..., 5].sigmoid() * duration
    root, deep = root - percolation, deep + percolation
    drainage = (deep - deep_cap).clamp_min(0)
    drainage = drainage + torch.minimum(deep, deep_cap) * logits[..., 6].sigmoid() * .1 * duration
    deep = deep - drainage
    updated = torch.stack([root, surface, deep], -1)
    et = surface_et + root_et
    residual = updated.sum(-1) - initial.sum(-1) - incoming + et + runoff + drainage
    flux = torch.stack([precipitation, irrigation, et, runoff, drainage, residual], -1)
    return updated / capacity, flux


class RecursiveField(nn.Module):
    """One fixed larger configuration, five seeds; causal driver GRU + state feedback."""

    def __init__(self, static_features, hidden=384, layers=3):
        super().__init__()
        self.static_features, self.hidden, self.layers = static_features, hidden, layers
        device = accelerator_device()
        self.register_buffer("center", torch.zeros(static_features, device=device))
        self.register_buffer("scale", torch.ones(static_features, device=device))
        self.register_buffer("state_scale", torch.tensor(STATE_SCALES, device=device))
        self.register_buffer("yield_calibration", torch.ones((), device=device))
        self.register_buffer("state_error", torch.zeros(92, len(SLOTS), device=device))
        self.embedding = BatchedLinear(len(DRIVERS) + 2 * static_features, 128)
        self.initial_hidden = BatchedLinear(len(SLOTS) + 2 * static_features, layers * hidden, gain=.1)
        grus = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            grus.append(nn.GRU(128, hidden, num_layers=layers, batch_first=True,
                               dropout=.1, device=device))
        self.grus = nn.ModuleList(grus)
        self.flow_first = BatchedLinear(hidden + len(SLOTS) + len(DRIVERS), 128)
        self.flow_second = BatchedLinear(128, 13, gain=.05)
        with torch.no_grad():
            self.flow_second.bias[..., :8].copy_(torch.tensor(
                [-3., 0., 0., -1., -2., -3., -5., -3.], device=device))
            self.flow_second.bias[..., 9] = -3.
        readout_dim = 2 * hidden + 7 * len(SLOTS) + 5 * len(DRIVERS) + 2 * static_features
        self.yield_first = BatchedLinear(readout_dim, 256)
        self.yield_second = BatchedLinear(256, 64)
        self.mean_head = BatchedLinear(64, 1, gain=.1)
        self.variance_head = BatchedLinear(64, 1, gain=.05, offset=-3.)

    def fit_scaler(self, static):
        self.center.copy_(torch.nan_to_num(torch.nanmean(static, 0)))
        sd = torch.nan_to_num(torch.nanmean((static - self.center).square(), 0).sqrt(), nan=1.)
        self.scale.copy_(torch.where(sd > 1e-5, sd, 1.))

    def normalise_static(self, static):
        normal = torch.nan_to_num((static - self.center) / self.scale).clamp(-8, 8)
        return torch.cat([normal, torch.isnan(static).float()], -1)

    def rollout(self, initial, drivers, static, capacity, irrigated_share, observations=None,
                teacher_probability=0., detach_after=None, hidden=None):
        """Free unless explicitly teacher forced; observations never enter the GRU."""
        members, b, steps = len(SEEDS), drivers.shape[0], drivers.shape[1]
        initial_members = initial[None].expand(members, -1, -1) if initial.ndim == 2 else initial
        if static.ndim == 2:
            static = static[:, None].expand(-1, steps, -1)
        z = self.normalise_static(static)
        inputs = torch.cat([drivers, z], -1).unsqueeze(0).expand(members, -1, -1, -1)
        embedded = F.silu(apply_linear(self.embedding, inputs))
        if hidden is None:
            init = torch.cat([initial_members / self.state_scale,
                              z[:, 0][None].expand(members, -1, -1)], -1)
            hidden = self.initial_hidden(init)
            hidden = hidden.reshape(members, b, self.layers, self.hidden).permute(0, 2, 1, 3)
        contexts, final_hidden = [], []
        for member, gru in enumerate(self.grus):
            context, final = gru(embedded[member], hidden[member].to(embedded.dtype).contiguous())
            contexts.append(context.float())
            final_hidden.append(final.float())
        context = torch.stack(contexts)
        state = initial_members
        capacity = capacity[None].expand(members, -1, -1)
        if irrigated_share.ndim == 1:
            irrigated_share = irrigated_share[:, None].expand(-1, steps)
        irrigation = torch.nan_to_num(irrigated_share, nan=0.)[None].expand(members, -1, -1)
        states, fluxes = [state], []
        for step in range(steps):
            if observations is not None and teacher_probability:
                truth = observations[:, step][None].expand(members, -1, -1)
                truth = torch.where(torch.isfinite(truth), truth, state)
                choice = torch.rand(1, b, 1, device=state.device) < teacher_probability
                state = torch.where(choice, truth, state)
            d = drivers[:, step][None].expand(members, -1, -1)
            inputs = torch.cat([context[:, :, step], state / self.state_scale, d], -1)
            logits = self.flow_second(F.silu(self.flow_first(inputs))).float()
            wetness, flux = water_balance(
                state[..., :3], d[..., 3].clamp_min(0) * 40,
                d[..., 6].clamp_min(0) * 30, d[..., 23] * 8,
                irrigation[:, :, step] * d[..., 21] * d[..., 9].gt(0), capacity, logits)
            water_factor = ((wetness[..., 0] - .1) / .6).clamp(0, 1)
            present = d[..., 21]
            ndvi = state[..., 3].clamp(0, 1)
            growth = .15 * logits[..., 8].sigmoid() * water_factor * present * (1 - ndvi)
            senescence = (.02 + .08 * logits[..., 9].sigmoid()) * ndvi
            senescence = senescence * (1 + 2 * (1 - present)
                                       + 2 * (d[..., 10] - .8).clamp_min(0) + 1 - water_factor)
            new_ndvi = (ndvi + growth - senescence).clamp(0, 1)
            delta = new_ndvi - ndvi
            lai = (state[..., 4] + 6 * delta + .2 * logits[..., 10].tanh() * present).clamp(0, 8)
            fpar = (state[..., 5] + .8 * delta + .1 * logits[..., 11].tanh() * present).clamp(0, 1)
            actual_et = flux[..., 2] / (d[..., 23] * 8).clamp_min(1)
            temperature = state[..., 7] + .5 * (10 * logits[..., 12].tanh() - state[..., 7])
            state = torch.cat([wetness, torch.stack(
                [new_ndvi, lai, fpar, actual_et, temperature], -1)], -1)
            states.append(state)
            fluxes.append(flux)
            if detach_after is not None and (step + 1) % detach_after == 0:
                state = state.detach()
        return {"states": torch.stack(states, 2), "fluxes": torch.stack(fluxes, 2),
                "context": context, "hidden": torch.stack(final_hidden), "static": z}

    def yield_distribution(self, rollout, drivers, season):
        """Harvest readout is causal: excludes every post-harvest state/context."""
        state, context = rollout["states"][:, :, 1:], rollout["context"]
        members, b, steps, _ = state.shape
        weights = season / season.sum(-1, keepdim=True).clamp_min(1)
        phase = drivers[..., 16:21] * season[..., None]
        phase = phase / phase.sum(1, keepdim=True).clamp_min(1e-6)
        state_pool = (state * weights[None, :, :, None]).sum(2)
        context_pool = (context * weights[None, :, :, None]).sum(2)
        stage_state = torch.einsum("ebts,btk->ebks", state, phase).flatten(2)
        stage_driver = torch.einsum("btd,btk->bkd", drivers, phase).flatten(1)
        last = torch.where(season > 0, torch.arange(steps, device=season.device), 0).amax(1)
        index = last[None, :, None, None]
        final_state = state.gather(2, index.expand(members, b, 1, len(SLOTS))).squeeze(2)
        final_context = context.gather(2, index.expand(members, b, 1, self.hidden)).squeeze(2)
        static = rollout["static"].gather(1, last[:, None, None].expand(-1, 1, rollout["static"].shape[-1])).squeeze(1)
        extra = torch.cat([stage_driver, static], -1)[None].expand(members, -1, -1)
        inputs = torch.cat([final_context, context_pool, state_pool, final_state,
                            stage_state, extra], -1)
        h = F.silu(self.yield_second(F.silu(self.yield_first(inputs))))
        means = self.mean_head(h).squeeze(-1).float()
        variances = self.variance_head(h).squeeze(-1).float().clamp(-8, 1).exp()
        mean = means.mean(0)
        variance = (variances + means.square()).mean(0) - mean.square()
        return mean, variance.clamp_min(1e-8).sqrt() * self.yield_calibration, means, variances


def save_recursive(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    torch.save({"state": model.state_dict(), "static_features": model.static_features,
                "hidden": model.hidden, "layers": model.layers, "metadata": metadata}, temporary)
    temporary.replace(path)


def load_recursive(path):
    saved = torch.load(path, map_location=accelerator_device(), weights_only=True)
    model = RecursiveField(saved["static_features"], saved["hidden"], saved["layers"])
    model.load_state_dict(saved["state"])
    return model.eval(), saved["metadata"]
