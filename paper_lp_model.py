"""Opt-in sparse score controls for the clean paper LP study.

The archived implementation stays unchanged. This adapter evaluates its three
existing dense scores on exactly the same sparse support, with native Lorentz
value aggregation. It is not global linear attention or a new attention score.
"""
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

from busemann_transformer import BusGraphTransformerNC, Hyperboloid

SCORES = {"directed": "busemann_log_depth", "bidirectional": "busemann_log_depth_sym",
          "dot": "lorentz_dot"}
ARMS = tuple(f"{name}_p{power}" for name in SCORES for power in (1, 2))


def sparse_score_control(self, x, attn_edge_index, edge_attn_bias, c_val):
    if self.score_type not in SCORES.values():
        raise ValueError("Unsupported paper LP score")
    if edge_attn_bias is not None:
        raise ValueError("The clean LP protocol has no attention bias")
    batch, nodes, _ = x.shape
    rows, columns, counts = self._prepare_sparse_edges(attn_edge_index, nodes, x.device)
    lengths = counts.unsqueeze(0).expand(batch, -1)
    outputs = []
    for head in range(self.num_heads):
        q = self.W_Q[head](x, c=c_val)
        k = self.W_K[head](x, c=c_val)
        value = self.W_V[head](x, c=c_val)
        query, key = q[:, rows], k[:, columns]
        if self.score_type == "lorentz_dot":
            score = self.log_tau_dot[head].exp() * (query[..., 1:] * key[..., 1:]).sum(-1)
        else:
            tau, gamma = F.softplus(self.tau_logit[head]), F.softplus(self.gamma_logit[head])
            direction = F.normalize(query[..., 1:], dim=-1)
            depth = (key[..., 0] - (direction * key[..., 1:]).sum(-1)).clamp(min=1e-6).log()
            factor = 1 + gamma * query[..., 0].clamp(min=1+1e-6).log()
            score = -tau * factor * depth
            if self.score_type == "busemann_log_depth_sym":
                key_direction = F.normalize(key[..., 1:], dim=-1)
                reverse_depth = (query[..., 0] - (key_direction * query[..., 1:]).sum(-1)).clamp(min=1e-6).log()
                reverse_factor = 1 + gamma * key[..., 0].clamp(min=1+1e-6).log()
                score = .5 * (score - tau * reverse_factor * reverse_depth)
        weights = self.attn_drop(self._sparse_segment_softmax(score, rows, counts))
        aggregate = torch.segment_reduce(weights[..., None] * value[:, columns], "sum", lengths=lengths, axis=1)
        norm = (-Hyperboloid.minkowski_dot(aggregate, aggregate, keepdim=True)).clamp(min=1e-6).sqrt()
        outputs.append(aggregate / (c_val**.5 * norm))
    spatial = torch.cat([output[..., 1:] for output in outputs], -1)
    point = Hyperboloid.proj(torch.cat((spatial.new_zeros(*spatial.shape[:-1], 1), spatial), -1), c_val)
    return self.out_proj(point, c=c_val)


def install_sparse_controls(base):
    for block in base.blocks:
        if block.attn is None or block.attn.score_type not in SCORES.values():
            raise ValueError("Every paper LP block must use a supported attention score")
        block.attn._forward_sparse_hsa = MethodType(sparse_score_control, block.attn)
    return base


class PaperLPModel(nn.Module):
    def __init__(self, base, power):
        super().__init__()
        if type(power) is not int or power not in (1, 2):
            raise ValueError("Decoder power must be 1 or 2")
        self.base, self.power = base, power
        self.r = nn.Parameter(torch.tensor(2.0))
        self.t = nn.Parameter(torch.tensor(1.0))

    def embed(self, features, support):
        base = self.base
        spatial = base.drop(base.input_proj(features))
        point = Hyperboloid.proj(torch.cat((spatial.new_zeros(len(spatial), 1), spatial), -1), base.log_c.exp())
        point = point.unsqueeze(0)
        for block in base.blocks:
            point = block(point, attn_edge_index=support)
        curvature = base.blocks[-1].curvature() if len(base.blocks) else base.log_c.exp()
        return base.final_norm(point.squeeze(0), c=curvature)

    def edge_logits(self, embeddings, edges):
        curvature = self.base.blocks[-1].curvature() if len(self.base.blocks) else self.base.log_c.exp()
        squared = Hyperboloid.sqdist(embeddings[edges[:, 0]], embeddings[edges[:, 1]], curvature)
        distance = squared if self.power == 2 else squared.clamp(min=1e-12).sqrt()
        return (self.r - distance) / self.t.clamp(min=1e-3)


def build_model(arm, seed, features, device, *, width=64, heads=4, layers=2, dropout=.1):
    if arm not in ARMS:
        raise ValueError("Unknown clean LP arm")
    score, power = arm.rsplit("_p", 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base = BusGraphTransformerNC(d_in=features, d_model=width, num_heads=heads, num_layers=layers,
        num_classes=2, c=1.0, dropout=dropout, learn_curvature=True, score_type=SCORES[score],
        use_attn_bias=False, attn_proj="hyp_linear", head_type="bmlr", use_attention=True,
        use_gnn_branch=False)
    model = PaperLPModel(install_sparse_controls(base), int(power)).to(device)
    # The NC-only head is retained for state/initialization provenance, but has
    # no role in LP and is explicitly frozen/excluded from the optimizer.
    for parameter in model.base.head.parameters():
        parameter.requires_grad_(False)
    # LD uses tau_logit/gamma_logit, not the native dot-score temperature.
    # Retain that tensor for paired state provenance, but do not count it as
    # active or include it in the optimizer when its score branch is unused.
    if score != "dot":
        for block in model.base.blocks:
            block.attn.log_tau_dot.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)
    return model, optimizer
