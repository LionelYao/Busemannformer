"""Exact positive dictionary attention with optional per-atom key mass removal.

This is a different kernel, not a stabilization of the default kernel: its
query gates mix key distributions without the default atom-density prior.
Inputs are synthetic tensors; a CUDA-only regression also covers the OGB key
length with a small atom count. No dataset training is involved.
"""

import copy

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
        score_type="busemann_log_depth_linear_dictionary", attn_proj="hyp_linear",
        linear_boundary_directions=5, linear_boundary_key_normalization="per_atom",
    )
    options.update(kwargs)
    return BusemannMHA(**options).double()


def _lift(spatial, c=1.0):
    return Hyperboloid.proj(
        torch.cat([spatial.new_zeros(*spatial.shape[:-1], 1), spatial], dim=-1), c,
    )


def _legacy_maps(model, q, k, head):
    """Pre-option feature implementation, retained as a regression oracle."""
    log_query, log_key = model._boundary_dictionary_log_features(q, k, head)
    key_shift = log_key.amax(dim=1, keepdim=True).detach()
    shifted_query = log_query + key_shift
    query_shift = shifted_query.amax(dim=-1, keepdim=True).detach()
    return (
        (shifted_query - query_shift).clamp(max=0.0).exp(),
        (log_key - key_shift).clamp(max=0.0).exp(),
    )


def _assert_gradients_equal(first, second, *, exact=False):
    other = dict(second.named_parameters())
    assert dict(first.named_parameters()).keys() == other.keys()
    for name, parameter in first.named_parameters():
        gradient = other[name].grad
        if parameter.grad is None:
            assert gradient is None, name
        else:
            assert gradient is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.isfinite(gradient).all(), name
            torch.testing.assert_close(
                parameter.grad, gradient,
                rtol=0 if exact else 2e-8, atol=0 if exact else 2e-10, msg=name,
            )


@pytest.mark.parametrize("scales", [1, 3])
@pytest.mark.parametrize("attn_proj", ["hyp_linear", "htc_qkv"])
def test_dense_normalized_kernel_matches_outputs_and_all_gradients(monkeypatch, scales, attn_proj):
    torch.manual_seed(617)
    model = _attention(linear_boundary_scales=scales, attn_proj=attn_proj).eval()
    reference = copy.deepcopy(model)

    def dense_head(x, c_val, head):
        q = reference.W_Q[head](x, c=c_val)
        k = reference.W_K[head](x, c=c_val)
        value = reference.W_V[head](x, c=c_val)
        log_query, log_key = reference._boundary_dictionary_log_features(q, k, head)
        # Independent dense reference: explicit positive features, partition
        # sums, and materialized N x N kernel, including gradients through Z.
        raw_key = log_key.exp()
        normalized_key = raw_key / raw_key.sum(dim=1, keepdim=True)
        kernel = torch.einsum("bnr,bkr->bnk", log_query.exp(), normalized_key)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        aggregate = torch.einsum("bnk,bkd->bnd", kernel, value)
        return reference._normalise_lorentz_aggregate(aggregate, c_val)

    monkeypatch.setattr(reference, "_forward_boundary_dictionary_head", dense_head)
    first_x = (0.2 * torch.randn(2, 7, 8, dtype=torch.float64)).requires_grad_()
    second_x = first_x.detach().clone().requires_grad_()
    first_log_c = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    second_log_c = first_log_c.detach().clone().requires_grad_()
    first = model(_lift(first_x, first_log_c.exp()), c=first_log_c.exp())
    second = reference(_lift(second_x, second_log_c.exp()), c=second_log_c.exp())
    torch.testing.assert_close(first, second, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(first, first),
        first.new_full((2, 7, 1), -1.0) / first_log_c.exp(),
        rtol=2e-9, atol=2e-10,
    )
    weights = torch.randn_like(first)
    (first * weights).sum().backward()
    (second * weights).sum().backward()
    torch.testing.assert_close(first_x.grad, second_x.grad, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(first_log_c.grad, second_log_c.grad, rtol=2e-8, atol=2e-10)
    assert first_log_c.grad.abs() > 0
    _assert_gradients_equal(model, reference)
    names = ["boundary_direction_raw", "boundary_gate_logit", "boundary_gate_bias",
             "tau_logit", "gamma_logit"]
    if scales > 1:
        names += ["boundary_exponent_offsets", "boundary_scale_gate_logit",
                  "boundary_scale_gate_bias"]
    for name in names:
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert gradient.abs().max() > 0, name


@pytest.mark.parametrize("scales", [1, 3])
def test_feature_mass_and_routing_for_unequal_query_key_lengths(scales):
    torch.manual_seed(619)
    model = _attention(linear_boundary_scales=scales)
    q = _lift(0.2 * torch.randn(2, 4, 4, dtype=torch.float64))
    k = _lift(0.4 * torch.randn(2, 9, 4, dtype=torch.float64))
    contiguous_query, contiguous_key = model._boundary_dictionary_feature_maps(q, k, 0)
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    k = k.transpose(1, 2).contiguous().transpose(1, 2)
    assert not q.is_contiguous() and not k.is_contiguous()
    query, key = model._boundary_dictionary_feature_maps(q, k, 0)
    torch.testing.assert_close(query, contiguous_query)
    torch.testing.assert_close(key, contiguous_key)
    assert query.shape == (2, 4, 5 * scales)
    assert key.shape == (2, 9, 5 * scales)
    assert (query > 0).all() and (key > 0).all()
    torch.testing.assert_close(query.sum(-1), query.new_ones(2, 4))
    torch.testing.assert_close(key.sum(1), key.new_ones(2, 5 * scales))
    kernel = torch.einsum("bqr,bkr->bqk", query, key)
    torch.testing.assert_close(kernel.sum(-1), kernel.new_ones(2, 4))
    effective_routing = query * key.sum(1, keepdim=True)
    effective_routing = effective_routing / effective_routing.sum(-1, keepdim=True)
    torch.testing.assert_close(effective_routing, query)


@pytest.mark.parametrize("scales", [1, 3])
def test_last_axis_layout_matches_native_float64_with_noncontiguous_log_features(monkeypatch, scales):
    torch.manual_seed(743)
    model = _attention(linear_boundary_scales=scales)
    batch, queries, keys, atoms = 3, 7, 11, 5 * scales
    # Give the implementation noncontiguous B,N,R views with unequal query
    # and key counts. Keep separate leaves for the independent native oracle.
    query_logits = torch.randn(batch, atoms, queries, dtype=torch.float64).transpose(1, 2).requires_grad_()
    key_logits = torch.randn(batch, atoms, keys, dtype=torch.float64).transpose(1, 2).requires_grad_()
    reference_query = query_logits.detach().clone().requires_grad_()
    reference_key = key_logits.detach().clone().requires_grad_()
    log_query = query_logits.log_softmax(-1).transpose(1, 2).contiguous().transpose(1, 2)
    assert not log_query.is_contiguous() and not key_logits.is_contiguous()
    monkeypatch.setattr(model, "_boundary_dictionary_log_features", lambda *_: (log_query, key_logits))

    calls = []
    original_softmax = torch.nn.functional.softmax

    def record_softmax(value, dim, **kwargs):
        calls.append((tuple(value.shape), value.is_contiguous(), dim))
        return original_softmax(value, dim=dim, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "softmax", record_softmax)
    dummy = torch.empty(0, dtype=torch.float64)
    query_feature, key_feature = model._boundary_dictionary_feature_maps(dummy, dummy, 0)
    assert calls == [((batch, atoms, keys), True, -1)]
    assert key_feature.shape == (batch, keys, atoms)
    assert not key_feature.is_contiguous()
    native_query = reference_query.log_softmax(-1).exp()
    native_key = torch.softmax(reference_key, dim=1)
    torch.testing.assert_close(query_feature, native_query, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(key_feature, native_key, rtol=1e-13, atol=1e-14)

    # Exercise the same associative downstream consumer with the transposed
    # view, as well as the normalizer's gradient through every key mass.
    values = torch.randn(batch, keys, 6, dtype=torch.float64)
    actual = torch.einsum("bqr,brd->bqd", query_feature,
                          torch.einsum("bkr,bkd->brd", key_feature, values))
    expected = torch.einsum("bqr,brd->bqd", native_query,
                            torch.einsum("bkr,bkd->brd", native_key, values))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (query_logits, key_logits), grad_outputs=probe)
    expected_gradients = torch.autograd.grad(expected, (reference_query, reference_key), grad_outputs=probe)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_ogb_length_key_normalization_matches_float64_oracle(monkeypatch):
    # Real OGB key length, but only eight atoms: bounded allocation and no
    # pairwise tensor. Use the strict normalization-only probe's original
    # tolerances (5e-5 relative / 2e-7 absolute), not native-fp32 bit equality.
    # Native long-axis fp32 accumulation is not an accuracy oracle; the
    # mathematical float64 softmax/Jacobian is the reference here.
    nodes, atoms = 169343, 8
    generator = torch.Generator(device="cuda").manual_seed(719)
    model = _attention(linear_boundary_directions=atoms).float().cuda()
    logits = (4.0 * torch.randn(1, nodes, atoms, generator=generator, device="cuda")).requires_grad_()
    log_query = torch.randn(1, 7, atoms, generator=generator, device="cuda").log_softmax(-1)
    probe = torch.randn(logits.shape, generator=generator, device="cuda")
    monkeypatch.setattr(model, "_boundary_dictionary_log_features", lambda *_: (log_query, logits))
    dummy = torch.empty(0, device="cuda")
    query_feature, key_feature = model._boundary_dictionary_feature_maps(dummy, dummy, 0)
    gradient, = torch.autograd.grad(key_feature, logits, grad_outputs=probe)
    with torch.no_grad():
        reference = logits.double().softmax(dim=1)
        double_probe = probe.double()
        reference_gradient = reference * (
            double_probe - (reference * double_probe).sum(dim=1, keepdim=True)
        )
    assert torch.isfinite(key_feature).all() and torch.isfinite(gradient).all()
    assert (key_feature >= 0).all()
    torch.testing.assert_close(query_feature.sum(-1), query_feature.new_ones(1, 7))
    torch.testing.assert_close(key_feature.sum(1), key_feature.new_ones(1, atoms), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(key_feature, reference.float(), rtol=5e-5, atol=2e-7)
    torch.testing.assert_close(gradient, reference_gradient.float(), rtol=5e-5, atol=2e-7)


def test_atom_log_offsets_cancel_and_shifted_query_trap_is_detected(monkeypatch):
    model = _attention()
    log_query = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], dtype=torch.float64).log()
    base_key = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]], dtype=torch.float64).log()
    dummy = torch.empty(1)
    monkeypatch.setattr(model, "_boundary_dictionary_log_features", lambda *_: (log_query, base_key))
    expected_query, expected_key = model._boundary_dictionary_feature_maps(dummy, dummy, 0)
    shifted_key = base_key + torch.tensor([[[1000., -1000.]]], dtype=torch.float64)
    monkeypatch.setattr(model, "_boundary_dictionary_log_features", lambda *_: (log_query, shifted_key))
    actual_query, actual_key = model._boundary_dictionary_feature_maps(dummy, dummy, 0)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=1e-12, atol=1e-12)
    legacy_query, legacy_key = _legacy_maps(model, dummy, dummy, 0)
    wrong_key = legacy_key / legacy_key.sum(1, keepdim=True)
    wrong = torch.einsum("bqr,bkr->bqk", legacy_query, wrong_key)
    wrong = wrong / wrong.sum(-1, keepdim=True)
    correct = torch.einsum("bqr,bkr->bqk", actual_query, actual_key)
    assert (correct - wrong).abs().max() > 0.5


@pytest.mark.parametrize("scales", [1, 3])
def test_key_permutation_and_complete_key_value_replication_leave_outputs_unchanged(scales):
    torch.manual_seed(631)
    model = _attention(linear_boundary_scales=scales)
    q = _lift(0.2 * torch.randn(2, 4, 4, dtype=torch.float64))
    k = _lift(0.4 * torch.randn(2, 9, 4, dtype=torch.float64))
    value = _lift(0.3 * torch.randn(2, 9, 4, dtype=torch.float64))

    def output(keys, values):
        query_feature, key_feature = model._boundary_dictionary_feature_maps(q, keys, 0)
        prototype = torch.einsum("bkr,bkd->brd", key_feature, values)
        aggregate = torch.einsum("bqr,brd->bqd", query_feature, prototype)
        return model._normalise_lorentz_aggregate(aggregate, 1.0)

    original = output(k, value)
    permutation = torch.randperm(k.shape[1])
    torch.testing.assert_close(original, output(k[:, permutation], value[:, permutation]))
    torch.testing.assert_close(original, output(k.repeat(1, 3, 1), value.repeat(1, 3, 1)))


def test_extreme_finite_keys_have_finite_features_outputs_and_gradients():
    torch.manual_seed(641)
    model = _attention(linear_boundary_scales=3)
    # Lorentz keys span a large depth range; an aligned unit boundary atom
    # exercises the existing Busemann argument floor as well as key softmax.
    with torch.no_grad():
        model.boundary_direction_raw[0, 0].copy_(torch.tensor([1., 0., 0., 0.]))
        model.tau_logit.fill_(100.)
    spatial = torch.tensor([[[0., 0., 0., 0.], [1e7, 0., 0., 0.],
                             [-1e7, 0., 0., 0.], [0., 1e6, 0., 0.]]],
                           dtype=torch.float64, requires_grad=True)
    q = _lift(0.2 * torch.randn(1, 3, 4, dtype=torch.float64))
    query, key = model._boundary_dictionary_feature_maps(q, _lift(spatial), 0)
    assert torch.isfinite(query).all() and torch.isfinite(key).all()
    torch.testing.assert_close(key.sum(1), key.new_ones(1, 15))
    values = _lift(0.3 * torch.randn(1, 4, 4, dtype=torch.float64))
    prototypes = torch.einsum("bkr,bkd->brd", key, values)
    aggregate = torch.einsum("bqr,brd->bqd", query, prototypes)
    output = model._normalise_lorentz_aggregate(aggregate, 1.)
    assert torch.isfinite(output).all()
    output[..., 1:].square().sum().backward()
    assert spatial.grad is not None and torch.isfinite(spatial.grad).all()
    for name in ("boundary_direction_raw", "tau_logit", "boundary_exponent_offsets"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name


def test_default_matches_legacy_outputs_gradients_and_rng_exactly(monkeypatch):
    torch.manual_seed(647)
    default = BusemannMHA(
        d_model=8, num_heads=2, dropout=0.1, attn_proj="hyp_linear",
        score_type="busemann_log_depth_linear_dictionary", linear_boundary_scales=3,
    ).double().train()
    assert default.linear_boundary_key_normalization == "none"
    reference = copy.deepcopy(default)
    monkeypatch.setattr(reference, "_boundary_dictionary_feature_maps",
                        lambda q, k, head: _legacy_maps(reference, q, k, head))
    first_x = (0.2 * torch.randn(1, 7, 8, dtype=torch.float64)).requires_grad_()
    second_x = first_x.detach().clone().requires_grad_()
    torch.manual_seed(653)
    first = default(_lift(first_x))
    first_rng = torch.rand(10)
    torch.manual_seed(653)
    second = reference(_lift(second_x))
    second_rng = torch.rand(10)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_rng, second_rng, rtol=0, atol=0)
    first[..., 1:].square().sum().backward()
    second[..., 1:].square().sum().backward()
    torch.testing.assert_close(first_x.grad, second_x.grad, rtol=0, atol=0)
    _assert_gradients_equal(default, reference, exact=True)


@pytest.mark.parametrize("factory", [
    lambda **kw: BusemannMHA(d_model=8, num_heads=2, **kw),
    lambda **kw: BusTransformerBlock(d_model=8, num_heads=2, **kw),
    lambda **kw: BusTransformer(vocab_size=12, d_model=8, num_heads=2,
                                num_layers=2, num_classes=3, **kw),
    lambda **kw: BusGraphTransformerNC(d_in=5, d_model=8, num_heads=2,
                                       num_layers=2, num_classes=3, **kw),
])
def test_all_constructors_preserve_parameters_rng_defaults_and_forward_option(factory):
    common = dict(score_type="busemann_log_depth_linear_dictionary",
                  linear_boundary_directions=5, linear_boundary_scales=2)
    snapshots = []
    for option in ({}, {"linear_boundary_key_normalization": "none"},
                   {"linear_boundary_key_normalization": "per_atom"}):
        torch.manual_seed(659)
        model = factory(**common, **option)
        snapshots.append((model.state_dict(), torch.rand(10)))
        if isinstance(model, BusemannMHA):
            attentions = [model]
        elif isinstance(model, BusTransformerBlock):
            attentions = [model.attn]
        else:
            attentions = [block.attn for block in model.blocks]
        for attention in attentions:
            assert attention.linear_boundary_key_normalization == option.get(
                "linear_boundary_key_normalization", "none",
            )
    for state, rng in snapshots[1:]:
        assert state.keys() == snapshots[0][0].keys()
        for name, value in state.items():
            torch.testing.assert_close(value, snapshots[0][0][name], rtol=0, atol=0)
        torch.testing.assert_close(rng, snapshots[0][1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="linear_boundary_key_normalization"):
        factory(**common, linear_boundary_key_normalization="unknown")


@pytest.mark.parametrize("keyword", ["bool_mask", "attn_bias"])
def test_key_normalization_does_not_enable_pairwise_masks(keyword):
    model = _attention()
    spatial = 0.1 * torch.randn(1, 7, 8, dtype=torch.float64)
    pairwise = (torch.zeros(1, 7, 7, dtype=torch.bool) if keyword == "bool_mask"
                else torch.zeros(1, 2, 7, 7, dtype=torch.float64))
    with pytest.raises(ValueError, match="supports global attention only"):
        model(_lift(spatial), **{keyword: pairwise})


def _local_edges(n, device):
    ids = torch.arange(n, device=device)
    edges = torch.stack([torch.cat([ids, ids]), torch.cat([ids, (ids + 1) % n])])
    return edges[:, (edges[0] * n + edges[1]).argsort()]


def _graph_model(score, *, checkpoint=False, mix=0.0, device="cpu"):
    return BusGraphTransformerNC(
        d_in=5, d_model=8, num_heads=2, num_layers=2, num_classes=3,
        dropout=0.1, c=0.7, learn_curvature=True, score_type=score,
        attn_proj="hyp_linear", use_gnn_branch=False,
        linear_boundary_directions=5, linear_boundary_scales=2,
        linear_boundary_fusion="centered_signed", linear_boundary_mix_init=mix,
        linear_boundary_preserve_rng=True,
        linear_boundary_residual_norm="match_local_rms",
        linear_boundary_checkpoint_heads=checkpoint,
        linear_boundary_key_normalization="per_atom",
    ).double().to(device).train()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_per_atom_zero_gate_preserves_local_outputs_shared_gradients_and_rng(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(673)
    local = _graph_model("busemann_log_depth", device=device)
    torch.manual_seed(673)
    hybrid = _graph_model("busemann_log_depth_local_dictionary", device=device)
    for name, value in local.state_dict().items():
        torch.testing.assert_close(value, hybrid.state_dict()[name], rtol=0, atol=0)
    first_x = (0.1 * torch.randn(9, 5, dtype=torch.float64, device=device)).requires_grad_()
    second_x = first_x.detach().clone().requires_grad_()
    edges = _local_edges(9, device)
    torch.manual_seed(677)
    first = local(first_x, attn_edge_index=edges)
    first_rng = (torch.rand(10), torch.rand(10, device=device))
    torch.manual_seed(677)
    second = hybrid(second_x, attn_edge_index=edges)
    second_rng = (torch.rand(10), torch.rand(10, device=device))
    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-11)
    for a, b in zip(first_rng, second_rng):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    first.square().mean().backward()
    second.square().mean().backward()
    torch.testing.assert_close(first_x.grad, second_x.grad, rtol=2e-8, atol=2e-10)
    shared = dict(hybrid.named_parameters())
    for name, parameter in local.named_parameters():
        if parameter.grad is None:
            assert shared[name].grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, shared[name].grad,
                                       rtol=2e-8, atol=2e-10, msg=name)


@pytest.mark.parametrize("score", ["busemann_log_depth_linear_dictionary",
                                  "busemann_log_depth_local_dictionary"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_per_atom_checkpoint_preserves_outputs_all_gradients_and_rng(score, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(683)
    normal = _graph_model(score, mix=0.04, device=device)
    checked = copy.deepcopy(normal)
    for block in checked.blocks:
        block.attn.linear_boundary_checkpoint_heads = True
    first_x = (0.1 * torch.randn(9, 5, dtype=torch.float64, device=device)).requires_grad_()
    second_x = first_x.detach().clone().requires_grad_()
    kwargs = ({"attn_edge_index": _local_edges(9, device)}
              if score.endswith("local_dictionary") else {})

    def evaluate(model, features):
        torch.manual_seed(691)
        output = model(features, **kwargs)
        after_forward = (torch.rand(10), torch.rand(10, device=device))
        output.square().mean().backward()
        after_backward = (torch.rand(10), torch.rand(10, device=device))
        return output, after_forward + after_backward

    first, first_rng = evaluate(normal, first_x)
    second, second_rng = evaluate(checked, second_x)
    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(first_x.grad, second_x.grad, rtol=2e-8, atol=2e-10)
    _assert_gradients_equal(normal, checked)
    assert normal.log_c.grad is not None
    assert all(block.log_c_param.grad is not None for block in normal.blocks)
    for a, b in zip(first_rng, second_rng):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
