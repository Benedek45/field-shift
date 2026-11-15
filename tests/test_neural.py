import numpy as np
import pytest
import torch

from fieldshift.model.neural import (
    DeviceAdamW,
    load_ensemble,
    predict_members,
    save_ensemble,
    train_ensemble,
)

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS-only model")


def test_device_adamw_first_step():
    p = torch.nn.Parameter(torch.tensor([2., -3.], device="mps"))
    p.grad = torch.tensor([1., -1.], device="mps")
    optimizer = DeviceAdamW([p])
    optimizer.step()
    expected = torch.tensor([2 * (1 - .001 * .02) - .001,
                             -3 * (1 - .001 * .02) + .001], device="mps")
    assert torch.allclose(p, expected)
    assert all(t.device.type == "mps" for t in optimizer.first + optimizer.second)


def test_mps_ensemble_scaler_training_and_serialization(tmp_path):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(100, 4)).astype("float32")
    x[0, 0] = np.nan
    x[:, 3] = np.nan
    y = .1 * x[:, 1]
    net = train_ensemble(x, y, epochs=2, batch_size=50)
    assert all(p.device.type == "mps" for p in net.parameters())
    assert all(b.device.type == "mps" for b in net.buffers())
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
