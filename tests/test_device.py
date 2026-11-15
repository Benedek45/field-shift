import numpy as np
import pytest
import torch

from fieldshift.model import neural
from fieldshift.sim.components import accelerator


@pytest.mark.parametrize("cuda,mps,forced,expected", [
    (True, True, None, "cuda"),
    (False, True, None, "mps"),
    (False, False, None, "cpu"),
    (True, True, "cpu", "cpu"),
    (True, True, " CPU ", "cpu"),
])
def test_simulator_and_models_share_device_precedence(monkeypatch, cuda, mps, forced, expected):
    monkeypatch.delenv("FIELDSHIFT_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    if forced is not None:
        monkeypatch.setenv("FIELDSHIFT_DEVICE", forced)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert accelerator().type == neural.accelerator_device().type == expected


def test_cpu_only_machine_trains_without_environment_setting(monkeypatch):
    monkeypatch.delenv("FIELDSHIFT_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU execution must not synchronize a GPU")

    monkeypatch.setattr(torch.mps, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    x = np.ones((8, 2), dtype="float32")
    net = neural.train_ensemble(x, np.ones(8, dtype="float32"), epochs=1, batch_size=4)
    assert all(p.device.type == "cpu" for p in net.parameters())
    assert np.isfinite(neural.predict_members(net, x)).all()
