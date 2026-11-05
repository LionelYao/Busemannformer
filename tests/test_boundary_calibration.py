"""Bounded scale calibration for the centered dictionary attention residual."""

import pytest
import torch

from busemann_transformer import (
    BusemannMHA,
    BusGraphTransformerNC,
    BusTransformer,
    BusTransformerBlock,
    Hyperboloid,
)


def _attention(**kwargs):
    options = dict(
        d_model=8, num_heads=2, dropout=0.0,
        score_type="busemann_log_depth_local_dictionary",
        attn_proj="hyp_linear", linear_boundary_directions=5,
        linear_boundary_fusion="centered_signed",
        linear_boundary_mix_init=0.05,
        linear_boundary_residual_norm="match_local_rms",
    )
    options.update(kwargs)
    return BusemannMHA(**options).double()


def _edges(n, device="cpu"):
    ids = torch.arange(n, device=device)
    edges = torch.stack([
        torch.cat([ids, ids]), torch.cat([ids, (ids + 1) % n]),
    ])
    return edges[:, (edges[0] * n + edges[1]).argsort()]


def test_calibration_matches_centered_local_rms_and_caps_graphwise_gain():
    model = _attention(linear_boundary_residual_max_gain=20.0)
    pattern = torch.tensor([[[-1., 1.], [1., -1.]]], dtype=torch.float64)
    local = (pattern * torch.tensor([2., 3.]).view(2, 1, 1) + 100).requires_grad_()
    global_centered = (pattern * torch.tensor([0.5, 0.01]).view(2, 1, 1)).requires_grad_()
    calibrated = model._calibrate_boundary_residual(
        global_centered, local, torch.tensor(-0.25),
    )
    stats = model.boundary_residual_diagnostics
    expected = {
        "raw_global_centered_rms": [0.5, 0.01],
        "local_centered_rms": [2., 3.],
        "gain": [4., 20.],
        "calibrated_rms": [2., 0.2],
        "effective_rms": [0.5, 0.05],
    }
    for name, values in expected.items():
        assert not stats[name].requires_grad
        torch.testing.assert_close(
            stats[name], torch.tensor(values, dtype=torch.float64),
            rtol=1e-7, atol=1e-9,
        )
    torch.testing.assert_close(
        calibrated, global_centered * torch.tensor([4., 20.]).view(2, 1, 1),
    )
    calibrated.square().mean().backward()
    assert torch.isfinite(global_centered.grad).all()
    assert local.grad is None  # No gradient route through scale calibration.


def test_constant_global_residual_remains_zero_without_nonfinite_gradients():
    model = _attention()
    local = torch.randn(2, 7, 8, dtype=torch.float64)
    constant_global = torch.ones(2, 7, 8, dtype=torch.float64, requires_grad=True)
    centered = constant_global - constant_global.mean(dim=1, keepdim=True)
    calibrated = model._calibrate_boundary_residual(centered, local, model.boundary_mix_scale())
    assert torch.count_nonzero(calibrated) == 0
    calibrated.square().sum().backward()
    assert torch.isfinite(constant_global.grad).all()
    stats = model.boundary_residual_diagnostics
    assert all(torch.isfinite(value).all() for value in stats.values())
    assert torch.count_nonzero(stats["effective_rms"]) == 0
    assert stats["gain"].max() <= 100.0


def test_disabled_calibration_preserves_residual_exactly():
    model = _attention(linear_boundary_residual_norm="none")
    global_centered = torch.randn(2, 7, 8, dtype=torch.float64)
    local = torch.randn_like(global_centered)
    calibrated = model._calibrate_boundary_residual(global_centered, local, model.boundary_mix_scale())
    assert calibrated is global_centered
    torch.testing.assert_close(model.boundary_residual_diagnostics["gain"], torch.ones(2, dtype=torch.float64))


def test_calibrated_attention_has_finite_backward_and_lorentz_output():
    torch.manual_seed(123)
    model = _attention()
    spatial = (0.2 * torch.randn(2, 11, 8, dtype=torch.float64)).requires_grad_()
    x = Hyperboloid.proj(torch.cat([spatial.new_zeros(2, 11, 1), spatial], dim=-1), 1.0)
    output = model(x, attn_edge_index=_edges(11))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(output, output), output.new_full((2, 11, 1), -1.0),
        rtol=1e-10, atol=1e-10,
    )
    output[..., 1:].square().mean().backward()
    assert torch.isfinite(spatial.grad).all()
    for parameter in (model.boundary_mix_logit, model.boundary_direction_raw,
                      model.boundary_gate_logit, model.boundary_gate_bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.boundary_residual_diagnostics["effective_rms"].max() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calibrated_zero_gate_preserves_dropout_shared_gradients_and_rng(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def make(score):
        return BusGraphTransformerNC(
            d_in=5, d_model=8, num_heads=2, num_layers=2, num_classes=3,
            dropout=0.1, learn_curvature=True, score_type=score,
            attn_proj="hyp_linear", use_gnn_branch=False,
            linear_boundary_directions=5, linear_boundary_scales=2,
            linear_boundary_fusion="centered_signed", linear_boundary_mix_init=0.0,
            linear_boundary_preserve_rng=True,
            linear_boundary_residual_norm="match_local_rms",
        ).double().to(device).train()

    torch.manual_seed(317)
    local = make("busemann_log_depth")
    torch.manual_seed(317)
    hybrid = make("busemann_log_depth_local_dictionary")
    for name, parameter in local.state_dict().items():
        torch.testing.assert_close(parameter, hybrid.state_dict()[name], rtol=0, atol=0)
    x = (0.1 * torch.randn(9, 5, dtype=torch.float64, device=device)).requires_grad_()
    y = x.detach().clone().requires_grad_()
    edges = _edges(9, device)
    torch.manual_seed(823)
    out_local = local(x, attn_edge_index=edges)
    next_local = torch.rand(10, device=device)
    cpu_next_local = torch.rand(10)
    torch.manual_seed(823)
    out_hybrid = hybrid(y, attn_edge_index=edges)
    next_hybrid = torch.rand(10, device=device)
    cpu_next_hybrid = torch.rand(10)
    torch.testing.assert_close(out_local, out_hybrid, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(next_local, next_hybrid, rtol=0, atol=0)
    torch.testing.assert_close(cpu_next_local, cpu_next_hybrid, rtol=0, atol=0)
    out_local.square().mean().backward()
    out_hybrid.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-10)
    shared = dict(hybrid.named_parameters())
    for name, parameter in local.named_parameters():
        if parameter.grad is None:
            assert shared[name].grad is None
        else:
            torch.testing.assert_close(parameter.grad, shared[name].grad, rtol=1e-9, atol=1e-10)
    for block in hybrid.blocks:
        assert torch.count_nonzero(block.attn.boundary_direction_raw.grad) == 0
        assert not block.attn.boundary_residual_diagnostics["gain"].requires_grad
        assert torch.count_nonzero(block.attn.boundary_residual_diagnostics["effective_rms"]) == 0


@pytest.mark.parametrize("factory", [
    lambda **kwargs: BusTransformerBlock(d_model=8, num_heads=2, **kwargs).attn,
    lambda **kwargs: BusTransformer(vocab_size=12, d_model=8, num_heads=2,
                                   num_layers=1, num_classes=3, **kwargs).blocks[0].attn,
    lambda **kwargs: BusGraphTransformerNC(d_in=5, d_model=8, num_heads=2,
                                          num_layers=1, num_classes=3, **kwargs).blocks[0].attn,
])
def test_all_model_constructors_forward_calibration_options(factory):
    attention = factory(linear_boundary_residual_norm="match_local_rms",
                        linear_boundary_residual_max_gain=12.0,
                        linear_boundary_residual_eps=2e-6)
    assert attention.linear_boundary_residual_norm == "match_local_rms"
    assert attention.linear_boundary_residual_max_gain == 12.0
    assert attention.linear_boundary_residual_eps == 2e-6


@pytest.mark.parametrize("options", [
    {"linear_boundary_residual_norm": "unknown"},
    {"linear_boundary_residual_max_gain": 0},
    {"linear_boundary_residual_max_gain": float("inf")},
    {"linear_boundary_residual_eps": 0},
    {"linear_boundary_residual_eps": float("nan")},
])
def test_invalid_calibration_options_are_rejected(options):
    with pytest.raises(ValueError, match="linear_boundary_residual"):
        _attention(**options)
