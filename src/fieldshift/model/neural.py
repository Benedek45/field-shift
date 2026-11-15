"""Five independently initialised MLPs, vectorised entirely on Apple MPS."""

from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn


def mps_device():
    if __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for auditable accelerator execution")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS or CUDA is required; neural CPU fallback is forbidden")
    return torch.device("mps")


# CUDA (e.g. Kaggle T4) first, then Apple MPS; never CPU.
accelerator_device = mps_device


class Ensemble(nn.Module):
    def __init__(self, n_features, seeds=(11, 23, 37, 51, 79)):
        super().__init__()
        device = mps_device()
        dimensions = (n_features, 48, 24, 1)
        weights, biases = [], []
        for n_in, n_out in pairwise(dimensions):
            ws = []
            for seed in seeds:
                torch.manual_seed(seed + n_in * 100 + n_out)
                w = torch.empty(n_in, n_out, device=device)
                nn.init.xavier_uniform_(w)
                ws.append(w)
            weights.append(nn.Parameter(torch.stack(ws)))
            biases.append(nn.Parameter(torch.zeros(len(seeds), 1, n_out, device=device)))
        self.weights, self.biases = nn.ParameterList(weights), nn.ParameterList(biases)
        self.register_buffer("center", torch.zeros(n_features, device=device))
        self.register_buffer("scale", torch.ones(n_features, device=device))

    def fit_scaler(self, x):
        self.center.copy_(torch.nan_to_num(torch.nanmean(x, dim=0)))
        variance = torch.nanmean((x - self.center) ** 2, dim=0)
        scale = torch.nan_to_num(variance.sqrt(), nan=1.0)
        self.scale.copy_(torch.where(scale > 1e-5, scale, 1.0))

    def forward(self, x):
        x = torch.nan_to_num((x - self.center) / self.scale, nan=0.0).clamp(-10, 10)
        x = x.unsqueeze(0).expand(len(self.weights[0]), -1, -1)
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, weight) + bias
            if i < len(self.weights) - 1:
                x = torch.relu(x)
        return x.squeeze(-1)


def train_ensemble(x_train, y_train, epochs=50, batch_size=8192):
    device = mps_device()
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    net = Ensemble(x.shape[1])
    net.fit_scaler(x)
    # torch AdamW keeps a CPU step tensor on MPS; use scalar-count AdamW
    # with all moment tensors on-device and no scalar tensor readbacks.
    optimizer = DeviceAdamW(net.parameters())
    torch.manual_seed(2026)
    net.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), device=device)
        for index in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = net(x[index])
            # Sum over ensemble members: same gradient scale as separate fits.
            loss = ((prediction - y[index].unsqueeze(0)) ** 2).mean(dim=1).sum()
            loss.backward()
            optimizer.step()
    torch.mps.synchronize()
    return net.eval()


class DeviceAdamW:
    def __init__(self, parameters, lr=.001, weight_decay=.02):
        self.parameters = list(parameters)
        self.lr, self.weight_decay, self.step_count = lr, weight_decay, 0
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]
        self.parameter_steps = [0] * len(self.parameters)

    def zero_grad(self, set_to_none=True):
        for p in self.parameters:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for i, (p, first, second) in enumerate(zip(self.parameters, self.first, self.second)):
            if p.grad is None:
                continue
            self.parameter_steps[i] += 1
            correction1 = 1 - .9 ** self.parameter_steps[i]
            correction2 = 1 - .999 ** self.parameter_steps[i]
            first.lerp_(p.grad, .1)
            second.mul_(.999).addcmul_(p.grad, p.grad, value=.001)
            p.mul_(1 - self.lr * self.weight_decay)
            denominator = second.sqrt().div_(correction2 ** .5).add_(1e-8)
            p.addcdiv_(first, denominator, value=-self.lr / correction1)


def predict_members(net, x):
    with torch.inference_mode():
        predictions = net(torch.tensor(x, dtype=torch.float32, device=mps_device()))
    # One output transfer after inference, never inside training/inference loops.
    return np.asarray(predictions.tolist(), dtype=float)


def save_ensemble(net, columns, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "columns": columns, "version": 1}, path)


@lru_cache(maxsize=4)
def load_ensemble(path):
    checkpoint = torch.load(path, map_location=mps_device(), weights_only=True)
    net = Ensemble(len(checkpoint["columns"]))
    net.load_state_dict(checkpoint["state"])
    return net.eval(), checkpoint["columns"]
