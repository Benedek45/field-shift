"""Shared device selection for the simulator and learned models."""

import os

import torch


def runtime_device():
    if os.environ.get("FIELDSHIFT_DEVICE", "").strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
