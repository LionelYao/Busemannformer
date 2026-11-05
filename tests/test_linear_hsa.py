from unittest import mock

import torch

from busemann_transformer import (
    BusemannMHA,
    BusGraphTransformerNC,
    Hyperboloid,
)


def _lorentz_input(
    batch_size: int,
    num_nodes: int,
    d_model: int,
    *,
    c: float = 1.0,
) -> torch.Tensor:
    spatial = 0.25 * torch.randn(batch_size, num_nodes, d_model)
    return Hyperboloid.proj(
        torch.cat(
            [spatial.new_zeros(batch_size, num_nodes, 1), spatial],
            dim=-1,
        ),
        c,
    )


def _make_attention(
    *,
    score_type: str,
    d_model: int = 8,
    num_heads: int = 2,
    linear_landmarks: int = 4,
    linear_rf_features: int = 32,
    linear_rf_seed: int = 17,
    linear_rf_t_max: float = 8.0,
    linear_rf_radial_scale: float | None = None,
    linear_boundary_directions: int = 6,
    linear_boundary_seed: int = 29,
    linear_boundary_scales: int = 1,
    linear_boundary_scale_spread: float = 0.35,
    linear_boundary_fusion: str = "additive",
    linear_boundary_mix_init: float = 0.1,
) -> BusemannMHA:
    return BusemannMHA(
        d_model=d_model,
        num_heads=num_heads,
        c=1.0,
        dropout=0.0,
        score_type=score_type,
        attn_proj="hyp_linear",
        linear_landmarks=linear_landmarks,
        linear_rf_features=linear_rf_features,
        linear_rf_seed=linear_rf_seed,
        linear_rf_t_max=linear_rf_t_max,
        linear_rf_radial_scale=linear_rf_radial_scale,
        linear_boundary_directions=linear_boundary_directions,
        linear_boundary_seed=linear_boundary_seed,
        linear_boundary_scales=linear_boundary_scales,
        linear_boundary_scale_spread=linear_boundary_scale_spread,
        linear_boundary_fusion=linear_boundary_fusion,
        linear_boundary_mix_init=linear_boundary_mix_init,
    )


def test_linear_hsa_is_exact_when_every_key_is_a_landmark():
    torch.manual_seed(3)
    x = _lorentz_input(2, 7, 8)
    dense = _make_attention(
        score_type="busemann_log_depth",
        linear_landmarks=7,
    )
    linear = _make_attention(
        score_type="busemann_log_depth_linear",
        linear_landmarks=7,
    )
    linear.load_state_dict(dense.state_dict(), strict=True)

    dense.eval()
    linear.eval()
    torch.testing.assert_close(
        linear(x),
        dense(x),
        rtol=2e-5,
        atol=2e-6,
    )


def test_linear_hsa_uses_only_requested_number_of_score_columns():
    torch.manual_seed(5)
    x = _lorentz_input(1, 19, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear",
        linear_landmarks=4,
    )

    with mock.patch.object(
        linear,
        "_busemann_log_depth_scores",
        wraps=linear._busemann_log_depth_scores,
    ) as score_spy:
        y = linear(x)

    assert y.shape == (1, 19, 9)
    assert len(score_spy.call_args_list) == linear.num_heads
    for call in score_spy.call_args_list:
        query, key_landmarks, _ = call.args
        assert query.shape[1] == 19
        assert key_landmarks.shape[1] == 4


def test_linear_hsa_forward_backward_is_finite_and_on_manifold():
    torch.manual_seed(7)
    x = _lorentz_input(2, 23, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear",
        linear_landmarks=5,
    )

    y = linear(x)
    assert torch.isfinite(y).all()
    expected_norm = y.new_full((*y.shape[:-1], 1), -1.0)
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(y, y),
        expected_norm,
        rtol=2e-4,
        atol=2e-4,
    )

    y[..., 1:].square().mean().backward()
    for name, parameter in linear.named_parameters():
        # log_tau_dot is retained for checkpoint compatibility but LD-HSA uses
        # tau_logit instead.
        if parameter.requires_grad and name != "log_tau_dot":
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_linear_hsa_is_permutation_equivariant_without_feature_ties():
    torch.manual_seed(11)
    x = _lorentz_input(1, 17, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear",
        linear_landmarks=5,
    )
    linear.eval()

    permutation = torch.randperm(x.shape[1])
    inverse = torch.argsort(permutation)
    expected = linear(x)
    actual = linear(x[:, permutation])[:, inverse]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_linear_hsa_rejects_arbitrary_pairwise_inputs():
    torch.manual_seed(13)
    x = _lorentz_input(1, 6, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear",
        linear_landmarks=3,
    )
    pairwise_inputs = {
        "bool_mask": torch.zeros(1, 6, 6, dtype=torch.bool),
        "attn_bias": torch.zeros(1, 2, 6, 6),
    }

    for input_name, pairwise in pairwise_inputs.items():
        try:
            linear(x, **{input_name: pairwise})
        except ValueError as error:
            assert "supports global attention only" in str(error)
        else:
            raise AssertionError(f"{input_name} should be rejected")


def test_random_feature_contraction_matches_materialized_feature_kernel():
    torch.manual_seed(19)
    x = _lorentz_input(2, 11, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_features=24,
    )
    linear.eval()

    q = linear.W_Q[0](x)
    k = linear.W_K[0](x)
    value = linear.W_V[0](x)
    phi_query, phi_key = linear._linear_rf_feature_maps(
        q, k, 0, 1.0
    )

    feature_kernel = torch.einsum(
        "bnr,bmr->bnm", phi_query, phi_key
    )
    explicit = torch.einsum(
        "bnm,bmd->bnd", feature_kernel, value
    )
    key_value = torch.einsum(
        "bmr,bmd->brd", phi_key, value
    )
    associative = torch.einsum(
        "bnr,brd->bnd", phi_query, key_value
    )

    assert (feature_kernel >= 0).all()
    torch.testing.assert_close(
        associative,
        explicit,
        rtol=2e-5,
        atol=2e-6,
    )


def test_random_feature_hsa_forward_backward_is_finite_and_on_manifold():
    torch.manual_seed(23)
    x = _lorentz_input(2, 29, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_features=32,
    )

    y = linear(x)
    assert y.shape == (2, 29, 9)
    assert torch.isfinite(y).all()
    expected_norm = y.new_full((*y.shape[:-1], 1), -1.0)
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(y, y),
        expected_norm,
        rtol=2e-4,
        atol=2e-4,
    )

    y[..., 1:].square().mean().backward()
    for name, parameter in linear.named_parameters():
        # log_tau_dot is retained for checkpoint compatibility; LD-HSA uses
        # tau_logit instead.
        if parameter.requires_grad and name != "log_tau_dot":
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_random_features_are_seeded_buffers_and_round_trip():
    first = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_seed=31,
    )
    second = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_seed=31,
    )
    different = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_seed=32,
    )

    for buffer_name in ("linear_rf_t", "linear_rf_omega", "linear_rf_zeta"):
        torch.testing.assert_close(
            getattr(first, buffer_name),
            getattr(second, buffer_name),
        )
    assert not torch.equal(first.linear_rf_omega, different.linear_rf_omega)

    restored = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_seed=999,
    )
    restored.load_state_dict(first.state_dict())
    torch.testing.assert_close(restored.linear_rf_t, first.linear_rf_t)
    torch.testing.assert_close(restored.linear_rf_omega, first.linear_rf_omega)
    torch.testing.assert_close(restored.linear_rf_zeta, first.linear_rf_zeta)


def test_random_feature_radial_scale_auto_default_and_override():
    automatic = _make_attention(
        score_type="busemann_log_depth_linear_rf",
    )
    overridden = _make_attention(
        score_type="busemann_log_depth_linear_rf",
        linear_rf_radial_scale=3.5,
    )

    assert automatic.linear_rf_radial_scale == 4.0
    assert overridden.linear_rf_radial_scale == 3.5


def test_random_feature_hsa_rejects_arbitrary_pairwise_inputs():
    x = _lorentz_input(1, 6, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_rf",
    )

    for input_name, pairwise in {
        "bool_mask": torch.zeros(1, 6, 6, dtype=torch.bool),
        "attn_bias": torch.zeros(1, 2, 6, 6),
    }.items():
        try:
            linear(x, **{input_name: pairwise})
        except ValueError as error:
            assert "supports global attention only" in str(error)
        else:
            raise AssertionError(f"{input_name} should be rejected")


def test_boundary_dictionary_contraction_matches_materialized_kernel():
    torch.manual_seed(37)
    x = _lorentz_input(2, 13, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=7,
    )
    linear.eval()

    q = linear.W_Q[0](x)
    k = linear.W_K[0](x)
    value = linear.W_V[0](x)
    phi_query, phi_key = linear._boundary_dictionary_feature_maps(
        q, k, 0
    )

    feature_kernel = torch.einsum(
        "bnr,bmr->bnm", phi_query, phi_key
    )
    explicit = torch.einsum(
        "bnm,bmd->bnd", feature_kernel, value
    )
    key_value = torch.einsum(
        "bmr,bmd->brd", phi_key, value
    )
    associative = torch.einsum(
        "bnr,brd->bnd", phi_query, key_value
    )

    assert (feature_kernel >= 0).all()
    torch.testing.assert_close(
        associative,
        explicit,
        rtol=2e-5,
        atol=2e-6,
    )


def test_multiscale_boundary_dictionary_contraction_matches_materialized_kernel():
    torch.manual_seed(39)
    x = _lorentz_input(2, 13, 8)
    directions, scales = 5, 3
    linear = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=directions,
        linear_boundary_scales=scales,
    )
    linear.eval()

    q = linear.W_Q[0](x)
    k = linear.W_K[0](x)
    value = linear.W_V[0](x)
    phi_query, phi_key = linear._boundary_dictionary_feature_maps(
        q, k, 0
    )
    assert phi_query.shape[-1] == directions * scales
    assert phi_key.shape[-1] == directions * scales

    feature_kernel = torch.einsum(
        "bnr,bmr->bnm", phi_query, phi_key
    )
    explicit = torch.einsum(
        "bnm,bmd->bnd", feature_kernel, value
    )
    key_value = torch.einsum(
        "bmr,bmd->brd", phi_key, value
    )
    associative = torch.einsum(
        "bnr,brd->bnd", phi_query, key_value
    )
    assert (feature_kernel >= 0).all()
    torch.testing.assert_close(
        associative, explicit, rtol=2e-5, atol=2e-6
    )

    associative[..., 1:].square().mean().backward()
    for parameter in (
        linear.boundary_direction_raw,
        linear.boundary_exponent_offsets,
        linear.boundary_scale_gate_logit,
        linear.boundary_scale_gate_bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_boundary_dictionary_stabilization_preserves_normalized_kernel():
    torch.manual_seed(41)
    x = _lorentz_input(1, 17, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=8,
    )
    linear.eval()

    q = linear.W_Q[0](x)
    k = linear.W_K[0](x)
    log_query, log_key = linear._boundary_dictionary_log_features(
        q, k, 0
    )
    exact_log_kernel = torch.logsumexp(
        log_query.unsqueeze(2) + log_key.unsqueeze(1), dim=-1
    )
    exact_attention = exact_log_kernel.softmax(dim=-1)

    phi_query, phi_key = linear._boundary_dictionary_feature_maps(
        q, k, 0
    )
    stable_kernel = torch.einsum(
        "bnr,bmr->bnm", phi_query, phi_key
    )
    stable_attention = stable_kernel / stable_kernel.sum(
        dim=-1, keepdim=True
    )
    torch.testing.assert_close(
        stable_attention,
        exact_attention,
        rtol=3e-5,
        atol=3e-6,
    )


def test_boundary_dictionary_spherical_initialization_and_seed():
    first = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=5,
        linear_boundary_seed=43,
    )
    second = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=5,
        linear_boundary_seed=43,
    )
    different = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=5,
        linear_boundary_seed=44,
    )
    torch.testing.assert_close(
        first.boundary_direction_raw,
        second.boundary_direction_raw,
    )
    assert not torch.equal(
        first.boundary_direction_raw,
        different.boundary_direction_raw,
    )

    generator = torch.Generator().manual_seed(47)
    samples = torch.randn(
        first.num_heads,
        40,
        first.d_head,
        generator=generator,
    )
    before = first.boundary_direction_raw.detach().clone()
    first.initialize_boundary_dictionary(samples, iterations=5)
    assert not torch.equal(before, first.boundary_direction_raw)
    torch.testing.assert_close(
        first.boundary_directions().norm(dim=-1),
        torch.ones(first.num_heads, 5),
        rtol=1e-6,
        atol=1e-6,
    )


def test_boundary_dictionary_forward_backward_is_finite_and_on_manifold():
    torch.manual_seed(53)
    x = _lorentz_input(2, 31, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
        linear_boundary_directions=9,
    )

    y = linear(x)
    assert y.shape == (2, 31, 9)
    assert torch.isfinite(y).all()
    expected_norm = y.new_full((*y.shape[:-1], 1), -1.0)
    torch.testing.assert_close(
        Hyperboloid.minkowski_dot(y, y),
        expected_norm,
        rtol=2e-4,
        atol=2e-4,
    )

    y[..., 1:].square().mean().backward()
    for name, parameter in linear.named_parameters():
        if parameter.requires_grad and name != "log_tau_dot":
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_boundary_dictionary_rejects_arbitrary_pairwise_inputs():
    x = _lorentz_input(1, 6, 8)
    linear = _make_attention(
        score_type="busemann_log_depth_linear_dictionary",
    )

    for input_name, pairwise in {
        "bool_mask": torch.zeros(1, 6, 6, dtype=torch.bool),
        "attn_bias": torch.zeros(1, 2, 6, 6),
    }.items():
        try:
            linear(x, **{input_name: pairwise})
        except ValueError as error:
            assert "supports global attention only" in str(error)
        else:
            raise AssertionError(f"{input_name} should be rejected")


def test_full_model_spherical_initialization_is_finite_and_data_dependent():
    torch.manual_seed(59)
    model = BusGraphTransformerNC(
        d_in=12,
        d_model=8,
        num_heads=2,
        num_layers=2,
        num_classes=3,
        dropout=0.0,
        score_type="busemann_log_depth_linear_dictionary",
        attn_proj="hyp_linear",
        use_gnn_branch=False,
        linear_boundary_directions=5,
    )
    features = torch.randn(23, 12)
    before = [
        block.attn.boundary_direction_raw.detach().clone()
        for block in model.blocks
    ]
    model.initialize_boundary_dictionaries(features, iterations=4)

    for old, block in zip(before, model.blocks):
        assert not torch.equal(old, block.attn.boundary_direction_raw)
        torch.testing.assert_close(
            block.attn.boundary_directions().norm(dim=-1),
            torch.ones(2, 5),
            rtol=1e-6,
            atol=1e-6,
        )
    output = model(features)
    assert output.shape == (23, 3)
    assert torch.isfinite(output).all()
