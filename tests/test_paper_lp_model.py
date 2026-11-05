"""Independent native-dense versus sparse LP operator checks on tiny inputs."""
import copy
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from busemann_transformer import BusemannMHA, Hyperboloid
from paper_lp_model import ARMS, SCORES, build_model, sparse_score_control


def fixture(device="cpu", dtype=torch.float64):
    torch.manual_seed(118)
    spatial = .2*torch.randn(2, 7, 8, dtype=dtype, device=device)
    point = Hyperboloid.proj(torch.cat((spatial[..., :1]*0, spatial), -1), .7)
    allowed = torch.eye(7, dtype=torch.bool, device=device)
    allowed[0, 1] = allowed[1, 0] = allowed[2, 4] = allowed[4, 5] = True
    support = allowed.nonzero().T.contiguous()
    return point, support, (~allowed).unsqueeze(0).expand(2, -1, -1)


@pytest.mark.parametrize("score", list(SCORES.values()))
def test_sparse_matches_native_dense_outputs_all_gradients_and_curvature(score):
    point, support, mask = fixture()
    dense = BusemannMHA(8, 2, c=.7, dropout=0., score_type=score, attn_proj="hyp_linear").double()
    sparse = copy.deepcopy(dense)
    sparse._forward_sparse_hsa = MethodType(sparse_score_control, sparse)
    dx, sx = point.clone().requires_grad_(), point.clone().requires_grad_()
    dc, sc = torch.tensor(.7, dtype=torch.float64, requires_grad=True), torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    a, b = dense(dx, bool_mask=mask, c=dc), sparse(sx, attn_edge_index=support, c=sc)
    torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-11)
    (a.square().mean()+.2*a[..., 0].mean()).backward()
    (b.square().mean()+.2*b[..., 0].mean()).backward()
    torch.testing.assert_close(dx.grad, sx.grad, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(dc.grad, sc.grad, rtol=2e-9, atol=2e-10)
    for (name, p), (other, q) in zip(dense.named_parameters(), sparse.named_parameters()):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-9, atol=2e-10)


def test_all_arms_share_common_initial_weights_and_decoder_state():
    states = [build_model(arm, 42, 5, "cpu", width=8, heads=2, layers=1)[0].state_dict() for arm in ARMS]
    common = set.intersection(*(set(state) for state in states))
    for name in common:
        assert all(torch.equal(states[0][name], state[name]) for state in states)


@pytest.mark.parametrize("arm", ARMS)
def test_only_used_score_parameters_are_optimized_and_unused_state_is_retained(arm):
    model, optimizer = build_model(arm, 42, 5, "cpu", width=8, heads=4, layers=2, dropout=0.)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    state = model.state_dict()
    for index, block in enumerate(model.base.blocks):
        parameter = block.attn.log_tau_dot
        used = arm.startswith("dot_")
        assert parameter.requires_grad is used
        assert (id(parameter) in optimized) is used
        assert torch.equal(state[f"base.blocks.{index}.attn.log_tau_dot"], parameter.detach())
        if not used:
            assert block.attn.tau_logit.requires_grad and id(block.attn.tau_logit) in optimized
            assert block.attn.gamma_logit.requires_grad and id(block.attn.gamma_logit) in optimized


@pytest.mark.parametrize("power", (1, 2))
def test_ld_stored_and_effective_parameter_differences_are_distinct(power):
    counts = {}
    for score in SCORES:
        model, _ = build_model(f"{score}_p{power}", 42, 5, "cpu", width=8, heads=4, layers=2)
        counts[score] = (sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in model.parameters() if p.requires_grad))
    for score in ("directed", "bidirectional"):
        # Four heads in two layers: LD stores two new parameters per head,
        # but replaces one active dot temperature with those two parameters.
        assert counts[score][0] - counts["dot"][0] == 16
        assert counts[score][1] - counts["dot"][1] == 8


@pytest.mark.parametrize("arm", ARMS)
def test_complete_lp_model_trains_without_classifier_head(arm):
    torch.manual_seed(72)
    features = torch.randn(7, 5)
    _, support, _ = fixture(dtype=torch.float32)
    model, optimizer = build_model(arm, 42, 5, "cpu", width=8, heads=2, layers=1, dropout=0.)
    edges = torch.tensor([[0, 1], [0, 2], [2, 4], [3, 5]])
    before = model.base.input_proj.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        logits = model.edge_logits(model.embed(features, support), edges)
        loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0., 1., 0.]))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    assert not torch.equal(before, model.base.input_proj.weight)
    assert all(not p.requires_grad and p.grad is None for p in model.base.head.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA H200",
    reason="Historical H200-specific check; CPU parity tests run on every platform",
)
def test_paper_lp_native_cuda_dense_parity_and_training():
    assert torch.cuda.get_device_name(0) == "NVIDIA H200"
    torch.use_deterministic_algorithms(True)
    for score in SCORES.values():
        point, support, mask = fixture("cuda", torch.float32)
        dense = BusemannMHA(8, 2, c=.7, dropout=0., score_type=score, attn_proj="hyp_linear").cuda()
        sparse = copy.deepcopy(dense)
        sparse._forward_sparse_hsa = MethodType(sparse_score_control, sparse)
        a, b = dense(point, bool_mask=mask), sparse(point, attn_edge_index=support)
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(dense.parameters(), sparse.parameters()):
            if p.grad is None:
                assert q.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=3e-4)
    for arm in ARMS:
        model, optimizer = build_model(arm, 42, 5, "cuda", width=8, heads=2, layers=1)
        features = torch.randn(7, 5, device="cuda")
        edges = torch.tensor([[0, 1], [0, 2], [2, 4], [3, 5]], device="cuda")
        before = model.base.input_proj.weight.detach().clone()
        logits = model.edge_logits(model.embed(features, support), edges)
        loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0., 1., 0.], device="cuda"))
        loss.backward(); optimizer.step()
        assert torch.isfinite(loss) and not torch.equal(before, model.base.input_proj.weight)
