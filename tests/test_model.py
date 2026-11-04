import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqparam.model.cnn import SeqCNN  # noqa: E402
from seqparam.model.params import (ModelParams, build_model, load_checkpoint,  # noqa: E402
                                   save_checkpoint)


def _gaussian_nll(mean, log_var, y):
    return (0.5 * (torch.exp(-log_var) * (y - mean) ** 2 + log_var)).mean()


def test_forward_shapes_heteroscedastic():
    m = SeqCNN(channels=(8, 8), kernel_sizes=(5, 3), pool_size=4, mlp_hidden=16)
    x = torch.randn(7, 4, 40)
    mean, log_var = m(x)
    assert mean.shape == (7,) and log_var.shape == (7,)


def test_forward_homoscedastic_returns_none_logvar():
    m = SeqCNN(channels=(8,), kernel_sizes=(5,), pool_size=4, mlp_hidden=16,
               heteroscedastic=False)
    mean, log_var = m(torch.randn(3, 4, 40))
    assert mean.shape == (3,) and log_var is None


def test_mismatched_channels_kernels_raises():
    with pytest.raises(ValueError):
        SeqCNN(channels=(8, 8), kernel_sizes=(5,))


def test_predict_returns_positive_std():
    m = SeqCNN(channels=(8,), kernel_sizes=(5,), pool_size=4, mlp_hidden=16)
    mean, std = m.predict(torch.randn(5, 4, 40))
    assert std.shape == (5,) and torch.all(std > 0)


def test_model_learns_a_simple_mapping():
    torch.manual_seed(0)
    # target is a fixed linear readout of the one-hot input -> learnable
    N, L = 512, 24
    X = torch.randint(0, 4, (N, L))
    onehot = torch.zeros(N, 4, L)
    onehot.scatter_(1, X.unsqueeze(1), 1.0)
    w = torch.randn(4, L)
    y = (onehot * w).sum(dim=(1, 2))
    y = (y - y.mean()) / y.std()

    m = SeqCNN(channels=(16, 16), kernel_sizes=(5, 3), pool_size=6, mlp_hidden=32,
               dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    def loss_now():
        mean, log_var = m(onehot)
        return _gaussian_nll(mean, log_var, y)

    init = loss_now().detach().item()
    for _ in range(120):
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        opt.step()
    assert loss_now().detach().item() < init - 0.3    # clearly decreased


def test_param_file_json_roundtrip_is_torch_free(tmp_path):
    p = ModelParams(element_type="rbs", unit="PNI_1_100", seq_len=96,
                    channels=(32, 32), kernel_sizes=(12, 8), sources=["cambray2018"],
                    norm={"per_source": {"cambray2018": {"mean": 3.0, "std": 1.0}}})
    s = p.to_json()
    back = ModelParams.from_json(s)
    assert back.channels == (32, 32) and back.kernel_sizes == (12, 8)
    assert back.element_type == "rbs" and back.seq_len == 96
    assert back.norm_stats().per_source["cambray2018"]["mean"] == 3.0


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path):
    p = ModelParams(element_type="rbs", unit="PNI_1_100", seq_len=40,
                    channels=(8, 8), kernel_sizes=(5, 3), pool_size=4, mlp_hidden=16)
    model = build_model(p)
    x = torch.randn(4, 4, 40)
    before, _ = model.predict(x)

    save_checkpoint(tmp_path / "ckpt", model, p)
    model2, p2 = load_checkpoint(tmp_path / "ckpt")
    after, _ = model2.predict(x)

    assert p2.element_type == "rbs" and p2.seq_len == 40
    assert torch.allclose(before, after, atol=1e-6)
