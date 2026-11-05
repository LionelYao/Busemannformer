"""Opt-in fixed dictionary base exponents; synthetic inputs, no dataset training.

The override fixes the dictionary base, not local LD-HSA tau. With G > 1,
learned multiplicative scale offsets still modify the effective exponents.
Dense-reference checks also run on CUDA when available.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from busemann_transformer import (
    BusemannMHA,
    BusGraphTransformerNC,
    BusTransformer,
    BusTransformerBlock,
    Hyperboloid,
)


def _attention(**kwargs):
    options = dict(
        d_model=8, num_heads=2, dropout=0.0, attn_proj="hyp_linear",
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=5,
    )
    options.update(kwargs)
    return BusemannMHA(**options).double()


def _lift(spatial, c=1.0):
    return Hyperboloid.proj(
        torch.cat([spatial.new_zeros(*spatial.shape[:-1], 1), spatial], -1), c,
    )


def _expected_logs(model, q, k, head, base):
    """Explicit mathematical reference, independent of the production helper."""
    directions = F.normalize(model.boundary_direction_raw[head], dim=-1)
    query_direction = F.normalize(q[..., 1:], dim=-1)
    depth = 1 + F.softplus(model.gamma_logit[head]) * q[..., :1].clamp_min(1 + 1e-6).log()
    query_logits = (
        F.softplus(model.boundary_gate_logit[head]) * depth
        * (query_direction @ directions.T) + model.boundary_gate_bias[head]
    )
    direction_log_probability = query_logits.log_softmax(-1)
    busemann = (k[..., :1] - k[..., 1:] @ directions.T).clamp_min(1e-6).log()
    exponent = F.softplus(model.tau_logit[head]) if base is None else q.new_tensor(base)
    if model.linear_boundary_scales == 1:
        return direction_log_probability, -exponent * busemann
    offsets = model.boundary_exponent_offsets[head].clamp(-2, 2)
    scale_logits = (
        -F.softplus(model.boundary_scale_gate_logit[head])
        * (depth.log() - offsets).square() + model.boundary_scale_gate_bias[head]
    )
    query = (direction_log_probability.unsqueeze(-1)
             + scale_logits.log_softmax(-1).unsqueeze(-2)).flatten(-2)
    key = (-busemann.unsqueeze(-1) * (exponent * offsets.exp())).flatten(-2)
    return query, key


def _compare_gradients(first, second, *, exact=False):
    other = dict(second.named_parameters())
    assert dict(first.named_parameters()).keys() == other.keys()
    for name, parameter in first.named_parameters():
        expected = other[name].grad
        if parameter.grad is None:
            assert expected is None, name
        else:
            assert expected is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, expected,
                rtol=0 if exact else 2e-8, atol=0 if exact else 2e-10, msg=name,
            )


@pytest.mark.parametrize("base", [None, 4.0, 7.5])
@pytest.mark.parametrize("scales", [1, 3])
@pytest.mark.parametrize("normalization", ["none", "per_atom"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dense_reference_matches_associative_outputs_and_all_gradients(
    monkeypatch, base, scales, normalization, device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(809)
    actual = _attention(
        linear_boundary_fixed_exponent=base, linear_boundary_scales=scales,
        linear_boundary_key_normalization=normalization,
    ).to(device).eval()
    reference = copy.deepcopy(actual)

    def dense_head(x, c_val, head):
        q, k, value = [projection[head](x, c=c_val)
                       for projection in (reference.W_Q, reference.W_K, reference.W_V)]
        query_log, key_log = _expected_logs(reference, q, k, head, base)
        key = key_log.exp()
        if normalization == "per_atom":
            key = key / key.sum(dim=1, keepdim=True)
        kernel = torch.einsum("bnr,bkr->bnk", query_log.exp(), key)
        kernel = kernel / kernel.sum(-1, keepdim=True)
        aggregate = torch.einsum("bnk,bkd->bnd", kernel, value)
        return reference._normalise_lorentz_aggregate(aggregate, c_val)

    monkeypatch.setattr(reference, "_forward_boundary_dictionary_head", dense_head)
    spatial = (0.15 * torch.randn(2, 7, 8, dtype=torch.float64, device=device)).requires_grad_()
    reference_spatial = spatial.detach().clone().requires_grad_()
    log_c = torch.tensor(-0.2, dtype=torch.float64, device=device, requires_grad=True)
    reference_log_c = log_c.detach().clone().requires_grad_()
    output = actual(_lift(spatial, log_c.exp()), c=log_c.exp())
    expected = reference(_lift(reference_spatial, reference_log_c.exp()), c=reference_log_c.exp())
    torch.testing.assert_close(output, expected, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(output, output),
        output.new_full((2, 7, 1), -1) / log_c.exp(), rtol=2e-9, atol=2e-10,
    )
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(spatial.grad, reference_spatial.grad, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(log_c.grad, reference_log_c.grad, rtol=2e-8, atol=2e-10)
    _compare_gradients(actual, reference)
    assert actual.tau_logit.requires_grad  # Keep the shared parameter/trainability.
    if base is None:
        assert actual.tau_logit.grad is not None and actual.tau_logit.grad.abs().max() > 0
    else:
        assert actual.tau_logit.grad is None  # The global dictionary no longer uses it.
    for name in ("boundary_direction_raw", "boundary_gate_logit", "gamma_logit"):
        gradient = dict(actual.named_parameters())[name].grad
        assert gradient is not None and gradient.abs().max() > 0, name
    if scales > 1:
        gradient = actual.boundary_exponent_offsets.grad
        assert gradient is not None and gradient.abs().max() > 0


@pytest.mark.parametrize("base", [4.0, 7.5])
@pytest.mark.parametrize("scales", [1, 3])
def test_fixed_base_is_independent_of_shared_tau_but_scale_offsets_still_apply(base, scales):
    torch.manual_seed(811)
    model = _attention(linear_boundary_fixed_exponent=base, linear_boundary_scales=scales)
    q = _lift(0.2 * torch.randn(2, 4, 4, dtype=torch.float64))
    k = _lift(0.3 * torch.randn(2, 9, 4, dtype=torch.float64))
    if scales > 1:
        with torch.no_grad():
            model.boundary_exponent_offsets[0].copy_(torch.tensor([-3.0, 0.2, 3.0]))
    first = model._boundary_dictionary_log_features(q, k, 0)
    expected = _expected_logs(model, q, k, 0, base)
    for actual, reference in zip(first, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-14)
    with torch.no_grad():
        model.tau_logit.add_(5.0)
    second = model._boundary_dictionary_log_features(q, k, 0)
    for actual, reference in zip(first, second):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("base", [4.0, 7.5])
def test_fixed_dictionary_does_not_change_local_scores_or_local_tau_gradient(base):
    torch.manual_seed(821)
    model = _attention(score_type="busemann_log_depth_local_dictionary")
    fixed = copy.deepcopy(model)
    fixed.linear_boundary_fixed_exponent = base
    q = _lift(0.2 * torch.randn(1, 4, 4, dtype=torch.float64))
    k = _lift(0.3 * torch.randn(1, 7, 4, dtype=torch.float64))
    expected = model._busemann_log_depth_scores(q, k, 0)
    actual = fixed._busemann_log_depth_scores(q, k, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    query_feature, key_feature = fixed._boundary_dictionary_feature_maps(q, k, 0)
    global_probe = (query_feature.square().sum() + key_feature.square().sum())
    global_gradient, = torch.autograd.grad(global_probe, fixed.tau_logit, allow_unused=True)
    assert global_gradient is None
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(fixed.tau_logit.grad, model.tau_logit.grad, rtol=0, atol=0)
    assert fixed.tau_logit.requires_grad and fixed.tau_logit.grad.abs().max() > 0


@pytest.mark.parametrize("score", [
    "busemann", "busemann_full", "hyp_dist", "busemann_depth_gate",
    "busemann_sym", "lorentz_dot", "busemann_log_depth",
    "busemann_log_depth_sym", "busemann_log_depth_linear",
    "busemann_log_depth_linear_rf", "busemann_add_depth", "hypformer_kernel",
])
def test_override_does_not_affect_other_scores_outputs_gradients_or_rng(score):
    torch.manual_seed(823)
    model = _attention(score_type=score, dropout=0.1).train()
    fixed = copy.deepcopy(model)
    fixed.linear_boundary_fixed_exponent = 7.5
    spatial = (0.1 * torch.randn(1, 7, 8, dtype=torch.float64)).requires_grad_()
    fixed_spatial = spatial.detach().clone().requires_grad_()
    torch.manual_seed(827)
    expected = model(_lift(spatial))
    expected_rng = torch.rand(5)
    torch.manual_seed(827)
    actual = fixed(_lift(fixed_spatial))
    actual_rng = torch.rand(5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(fixed_spatial.grad, spatial.grad, rtol=0, atol=0)
    _compare_gradients(fixed, model, exact=True)


_FACTORIES = [
    lambda **kw: BusemannMHA(d_model=8, num_heads=2, **kw),
    lambda **kw: BusTransformerBlock(d_model=8, num_heads=2, **kw),
    lambda **kw: BusTransformer(vocab_size=12, d_model=8, num_heads=2,
                               num_layers=2, num_classes=3, **kw),
    lambda **kw: BusGraphTransformerNC(d_in=5, d_model=8, num_heads=2,
                                      num_layers=2, num_classes=3, **kw),
]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_constructor_plumbing_default_state_parameters_and_rng_are_unchanged(factory):
    common = dict(score_type="busemann_log_depth_linear_dictionary", linear_boundary_scales=3)
    states = []
    for option in ({}, {"linear_boundary_fixed_exponent": None},
                   {"linear_boundary_fixed_exponent": 4}, {"linear_boundary_fixed_exponent": 7.5}):
        torch.manual_seed(829)
        model = factory(**common, **option)
        states.append((model.state_dict(), torch.rand(5)))
        attentions = ([model] if isinstance(model, BusemannMHA) else
                      [model.attn] if isinstance(model, BusTransformerBlock) else
                      [block.attn for block in model.blocks])
        for attention in attentions:
            assert attention.linear_boundary_fixed_exponent == option.get("linear_boundary_fixed_exponent")
            assert attention.tau_logit.requires_grad
    for state, rng in states[1:]:
        assert state.keys() == states[0][0].keys()
        for name, tensor in state.items():
            torch.testing.assert_close(tensor, states[0][0][name], rtol=0, atol=0)
        torch.testing.assert_close(rng, states[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf"),
                                     -float("inf"), True, "4", [4]])
def test_invalid_fixed_exponents_are_rejected(factory, invalid):
    with pytest.raises(ValueError, match="linear_boundary_fixed_exponent"):
        factory(linear_boundary_fixed_exponent=invalid)


@pytest.mark.parametrize("normalization", ["none", "per_atom"])
def test_omitted_and_explicit_none_have_bitexact_outputs_gradients_and_rng(normalization):
    torch.manual_seed(839)
    model = _attention(dropout=0.1, linear_boundary_scales=3,
                       linear_boundary_key_normalization=normalization).train()
    torch.manual_seed(839)
    explicit = _attention(dropout=0.1, linear_boundary_scales=3,
                          linear_boundary_key_normalization=normalization,
                          linear_boundary_fixed_exponent=None).train()
    spatial = (0.1 * torch.randn(1, 7, 8, dtype=torch.float64)).requires_grad_()
    other_spatial = spatial.detach().clone().requires_grad_()
    torch.manual_seed(853)
    first = model(_lift(spatial))
    first_rng = torch.rand(5)
    torch.manual_seed(853)
    second = explicit(_lift(other_spatial))
    second_rng = torch.rand(5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_rng, second_rng, rtol=0, atol=0)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(spatial.grad, other_spatial.grad, rtol=0, atol=0)
    _compare_gradients(model, explicit, exact=True)


@pytest.mark.parametrize("base", [4.0, 7.5])
def test_fixed_exponent_checkpoint_preserves_gradients_curvature_and_rng(base):
    torch.manual_seed(857)
    model = _attention(dropout=0.1, linear_boundary_scales=3,
                       linear_boundary_fixed_exponent=base,
                       linear_boundary_key_normalization="per_atom").train()
    checked = copy.deepcopy(model)
    checked.linear_boundary_checkpoint_heads = True
    spatial = (0.1 * torch.randn(1, 7, 8, dtype=torch.float64)).requires_grad_()
    other_spatial = spatial.detach().clone().requires_grad_()
    log_c = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    other_log_c = log_c.detach().clone().requires_grad_()

    def evaluate(attention, inputs, curvature):
        torch.manual_seed(859)
        output = attention(_lift(inputs, curvature.exp()), c=curvature.exp())
        output.square().sum().backward()
        return output, torch.rand(5)

    first, first_rng = evaluate(model, spatial, log_c)
    second, second_rng = evaluate(checked, other_spatial, other_log_c)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_rng, second_rng, rtol=0, atol=0)
    torch.testing.assert_close(spatial.grad, other_spatial.grad, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(log_c.grad, other_log_c.grad, rtol=2e-8, atol=2e-10)
    _compare_gradients(model, checked)
    assert model.tau_logit.grad is None and checked.tau_logit.grad is None
