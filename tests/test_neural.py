import os

import numpy as np
import pytest
import torch

from fieldshift.model.neural import (
    DeviceAdamW,
    accelerator_device,
    load_ensemble,
    predict_members,
    save_ensemble,
    train_ensemble,
)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("FIELDSHIFT_DEVICE") == "cpu" or
         torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="Requires an accelerator or explicit CPU mode",
)


def test_device_adamw_first_step():
    device = torch.empty(0, device=accelerator_device()).device
    p = torch.nn.Parameter(torch.tensor([2., -3.], device=device))
    p.grad = torch.tensor([1., -1.], device=device)
    optimizer = DeviceAdamW([p])
    optimizer.step()
    expected = torch.tensor([2 * (1 - .001 * .02) - .001,
                             -3 * (1 - .001 * .02) + .001], device=device)
    assert torch.allclose(p, expected)
    assert all(t.device == device for t in optimizer.first + optimizer.second)


def test_ensemble_scaler_training_and_serialization_on_selected_device(tmp_path):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(100, 4)).astype("float32")
    x[0, 0] = np.nan
    x[:, 3] = np.nan
    y = .1 * x[:, 1]
    net = train_ensemble(x, y, epochs=2, batch_size=50)
    device = torch.empty(0, device=accelerator_device()).device
    assert all(p.device == device for p in net.parameters())
    assert all(b.device == device for b in net.buffers())
    assert net.center[3].item() == 0
    np.testing.assert_allclose(net.center[0].item(), np.nanmean(x[:, 0]), atol=1e-6)
    prediction = predict_members(net, x)
    assert prediction.shape == (5, 100)
    assert np.isfinite(prediction).all()
    path = tmp_path / "model.pt"
    save_ensemble(net, ["a", "b", "c", "d"], path)
    reloaded, columns = load_ensemble(str(path))
    assert columns == ["a", "b", "c", "d"]
    np.testing.assert_allclose(prediction, predict_members(reloaded, x), atol=1e-6)
