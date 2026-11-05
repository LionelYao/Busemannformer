"""
Busemannformer — current graph / sequence hyperbolic Transformer implementation.

Route-A style stack: pre-LN, HypLinear FFN, Graphormer bias, Lorentz GNN, chunked
Busemann MHA; attention Q/K may use HTC (full (d+1)-dim input) with HypLinear V
by default (``attn_proj``), optional full HypLinear or full HTC QKV.

Conventions adopted from:

  • HGCN       (HazyResearch/hgcn)
      - Hyperboloid manifold class with proj / expmap0 / logmap0 / mobius_matvec
      - HypLinear: spatial affine map followed by hyperboloid reconstruction
      - Configurable curvature c (K = −c)

  • Hypformer  (Graph-and-Geometric-Learning/hyperbolic-transformer)
      - HypLayerNorm: LayerNorm on spatial part, reproject to H^d_c
      - HypActivation: activation on spatial part, reproject
      - HypCLS: distance-based classifier head (class prototypes on manifold)
      - Pre-LayerNorm (norm-first) residual blocks

  • Graphormer (Microsoft/Graphormer)
      - GraphAttnBias: precompute (B, H, N, N) structural bias from SPD + adjacency
      - attn_weights += attn_bias  before softmax
      - Graph token prepended at position 0 (virtual "super-node")
      - Pre-LayerNorm architecture throughout

Architecture (K = −c, curvature c > 0):
  H^d_c = {x ∈ R^{d+1} : ⟨x,x⟩_L = −1/c, x₀ > 0}
  ⟨x,y⟩_L = −x₀y₀ + ⟨xₛ,yₛ⟩

  Classes:
    Hyperboloid          – manifold utilities
    HypLinear            – FHNN/HyboNet-style spatial affine layer
    HTC                  – Hypformer Eq. 9 (optional Q/K or QKV in BusemannMHA)
    HypLayerNorm         – Hypformer-style norm
    HypActivation        – activation on spatial part
    GraphAttnBias        – Graphormer structural bias module
    BusemannMHA          – horospherical multi-head attention + bias injection
    BusTransformerBlock  – pre-LN block (Graphormer style)
    BMLR                 – legacy signed-scale Busemann-logit classifier
    HypCLS               – distance-based classifier head (Hypformer)
    BusTransformer       – full sequence model with graph token
    BusGraphTransformerNC – node-classification wrapper (no graph token)
    EucGraphTransformerNC – Euclidean baseline (pre-LN, Graphormer style)
"""

import math
import os
import random
from numbers import Real
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _validate_boundary_scale_routing(mode):
    if not isinstance(mode, str) or mode not in ("adaptive", "static"):
        raise ValueError(
            "linear_boundary_scale_routing must be 'adaptive' or "
            f"'static', got {mode!r}"
        )


# ============================================================================
# 1.  Hyperboloid Manifold  (K = −c, curvature c > 0)
#     Convention: ⟨x,x⟩_L = −1/c,  origin o = (1/√c, 0,…,0)
#     API mirrors HazyResearch/hgcn manifolds/hyperboloid.py
# ============================================================================

class Hyperboloid:
    """Static manifold utilities for the Lorentz / Hyperboloid model.

    All methods are @staticmethod so they can be called as
    ``Hyperboloid.proj(x, c)`` without instantiation.  This mirrors
    the HGCN Manifold base-class pattern.
    """

    # ------------------------------------------------------------------
    # Bilinear form
    # ------------------------------------------------------------------

    @staticmethod
    def minkowski_dot(x: torch.Tensor, y: torch.Tensor,
                      keepdim: bool = True) -> torch.Tensor:
        """Minkowski inner product ⟨x,y⟩_L = −x₀y₀ + ⟨xₛ,yₛ⟩.
        x, y : (..., d+1)  →  (..., 1)  or  (...)  depending on keepdim.
        """
        res = (
            -x[..., :1] * y[..., :1]
            + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        )
        return res if keepdim else res.squeeze(-1)

    # ------------------------------------------------------------------
    # Projection onto the manifold
    # ------------------------------------------------------------------

    @staticmethod
    def proj(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
        """Project x onto H^d_c by recomputing the time coordinate.
        Given spatial part xₛ, set x₀ = sqrt(1/c + ‖xₛ‖²).
        """
        xs = x[..., 1:]
        x0 = (1.0 / c + (xs * xs).sum(dim=-1, keepdim=True)).clamp(min=eps).sqrt()
        return torch.cat([x0, xs], dim=-1)

    @staticmethod
    def proj_tan0(u: torch.Tensor) -> torch.Tensor:
        """Project onto tangent space at origin (zero the time component)."""
        return torch.cat([u.new_zeros(*u.shape[:-1], 1), u[..., 1:]], dim=-1)

    # ------------------------------------------------------------------
    # Exponential / logarithmic maps at the origin
    # ------------------------------------------------------------------

    @staticmethod
    def expmap0(u: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
        """Exp map at origin o = (1/√c, 0,…,0).
        Input u is a pure spatial vector (time component = 0 or ignored).
        Exp_o(u) = cosh(√c‖uₛ‖)·o + sinh(√c‖uₛ‖)/(√c‖uₛ‖)·ũ
        where ũ = (0, uₛ) is the full-dimensional tangent vector.
        """
        sqrt_c = math.sqrt(c)
        u_s    = u[..., 1:]                                     # spatial part
        norm_u = u_s.norm(dim=-1, keepdim=True).clamp(min=eps)
        theta  = sqrt_c * norm_u
        x0     = torch.cosh(theta) / sqrt_c
        xs     = torch.sinh(theta) / (sqrt_c * norm_u) * u_s
        return torch.cat([x0, xs], dim=-1)

    @staticmethod
    def logmap0(x: torch.Tensor, c: float, eps: float = 1e-6) -> torch.Tensor:
        """Log map at origin.  Returns tangent vector (0, uₛ).
        Log_o(x) = d(o,x) / ‖xₛ‖ · (0, xₛ)
        where d(o,x) = (1/√c)·arcosh(√c·x₀).
        """
        # Keep the historical Python-float path bit-for-bit unchanged, while
        # supporting the learnable scalar tensor curvatures used by the model.
        # ``Tensor.sqrt`` also preserves gradients with respect to curvature.
        sqrt_c = c.sqrt() if isinstance(c, torch.Tensor) else math.sqrt(c)
        x0_safe = (sqrt_c * x[..., :1]).clamp(min=1.0 + eps)
        dist    = torch.acosh(x0_safe) / sqrt_c                 # geodesic dist from o
        xs      = x[..., 1:]
        norm_xs = xs.norm(dim=-1, keepdim=True).clamp(min=eps)
        u_s     = (dist / norm_xs) * xs
        return torch.cat([x.new_zeros(*x.shape[:-1], 1), u_s], dim=-1)

    # ------------------------------------------------------------------
    # Möbius matrix-vector multiplication  (HGCN key primitive)
    # ------------------------------------------------------------------

    @staticmethod
    def mobius_matvec(M: torch.Tensor, x: torch.Tensor,
                      c: float) -> torch.Tensor:
        """Linear map in H^d_c via the tangent space at origin.
        Equivalent to:  expmap0( proj_tan0( M · logmap0(x) ) , c )

        M : (out_features, in_features)
        x : (..., in_features + 1)
        Returns: (..., out_features + 1)
        """
        u   = Hyperboloid.logmap0(x, c)          # (..., in+1), u₀ ≈ 0
        u_s = u[..., 1:]                          # (..., in)
        y_s = u_s @ M.T                           # (..., out)
        y   = torch.cat([y_s.new_zeros(*y_s.shape[:-1], 1), y_s], dim=-1)
        return Hyperboloid.expmap0(y, c)

    # ------------------------------------------------------------------
    # Geodesic distances
    # ------------------------------------------------------------------

    @staticmethod
    def sqdist(x: torch.Tensor, y: torch.Tensor,
               c: float, eps: float = 1e-5) -> torch.Tensor:
        """Squared geodesic distance (1/c)·arcosh²(−c·⟨x,y⟩_L).
        d(x,y) = (1/√c)·arcosh(−c·⟨x,y⟩_L)  →  d²(x,y) = (1/c)·arcosh²(…)
        x, y : (..., d+1)  →  (...)"""
        prod = Hyperboloid.minkowski_dot(x, y, keepdim=False)
        arg  = (-c * prod).clamp(min=1.0 + eps)
        return (1.0 / c) * torch.acosh(arg) ** 2

    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor,
             c: float, eps: float = 1e-5) -> torch.Tensor:
        """Geodesic distance (1/√c)·arcosh(−c·⟨x,y⟩_L)."""
        prod = Hyperboloid.minkowski_dot(x, y, keepdim=False)
        arg  = (-c * prod).clamp(min=1.0 + eps)
        return (1.0 / math.sqrt(c)) * torch.acosh(arg)

    # ------------------------------------------------------------------
    # Parallel transport from origin  (used in HGCN HypAgg)
    # ------------------------------------------------------------------

    @staticmethod
    def ptransp0(x: torch.Tensor, u: torch.Tensor,
                 c: float, eps: float = 1e-5) -> torch.Tensor:
        """Parallel transport of tangent u from origin to x.
        PT_{o→x}(u) = u + c·⟨x,u⟩_L / (1 − c·⟨o,x⟩_L) · (o + x)
        """
        sqrt_c = math.sqrt(c)
        o      = torch.zeros_like(x)
        o[..., 0] = 1.0 / sqrt_c                          # origin broadcast

        xu     = Hyperboloid.minkowski_dot(x, u)           # (..., 1)
        ox     = Hyperboloid.minkowski_dot(o, x)           # (..., 1)
        denom  = (1.0 - c * ox).clamp(min=eps)
        return u + c * (xu / denom) * (o + x)


# ============================================================================
# 2.  HypLinear  (FHNN/HyboNet spatial affine map + reconstruction)
# ============================================================================

class HypLinear(nn.Module):
    """Fully hyperbolic linear layer (FHNN eq. 3, Chen et al. 2022).

    y = [ sqrt( ‖W xₛ + b‖² + 1/c ),   W xₛ + b ]

    Operates directly on the spatial part xₛ in ambient R^{out}.  No
    logmap / expmap / arcosh anywhere — the output time coordinate is
    computed from the hyperboloid constraint x₀ = sqrt(1/c + ‖xₛ‖²).

    Unlike the HGCN logmap0→W→expmap0 pipeline, this layer does not map through
    the tangent space: it uses one spatial affine map and reconstructs the time
    coordinate from the manifold constraint.

    in_features / out_features are *spatial* dimensions (not d+1).
    """

    def __init__(self, in_features: int, out_features: int,
                 c: float = 1.0, dropout: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.c            = c
        self.dropout_p    = dropout

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        nn.init.xavier_uniform_(self.weight, gain=1.0 / math.sqrt(2))

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        """x : (..., in+1)  →  (..., out+1).
        c : optional curvature override (float or scalar tensor); uses self.c if None.
        """
        c_val = c if c is not None else self.c
        w  = F.dropout(self.weight, p=self.dropout_p, training=self.training)
        xs = x[..., 1:] @ w.T                                     # (..., out)
        if self.bias is not None:
            xs = xs + self.bias
        x0 = (1.0 / c_val + (xs * xs).sum(dim=-1, keepdim=True)
              ).clamp(min=1e-6).sqrt()                             # (..., 1)
        return torch.cat([x0, xs], dim=-1)                        # (..., out+1)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"c={self.c}, dropout={self.dropout_p}, bias={self.bias is not None}")


# ============================================================================
# 2b.  HTC  (Hyperbolic Transformation with Curvatures, Hypformer Eq. 9)
#
#     Same implementation as v3; used in v4 only inside BusemannMHA when
#     attn_proj is "htc_qk" or "htc_qkv".
# ============================================================================

class HTC(nn.Module):
    """Hyperbolic Transformation with Curvatures (Hypformer, KDD 2024).

    f(x) = W x + b  where x ∈ R^{d+1} is the FULL Lorentz vector.
    HTC(x) = [ √( (c_in/c_out)·‖f(x)‖² + 1/c_out ),  √(c_in/c_out)·f(x) ]

    in_features, out_features refer to the *spatial* dimension; the matrix
    stored is (out_features, in_features+1) to absorb the time coordinate.
    """

    def __init__(self, in_features: int, out_features: int,
                 c: float = 1.0, dropout: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.c            = c
        self.dropout_p    = dropout

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features + 1))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        nn.init.xavier_uniform_(self.weight, gain=1.0 / math.sqrt(2))

    def forward(self,
                x:     torch.Tensor,
                c:     Optional[float] = None,
                c_in:  Optional[float] = None,
                ) -> torch.Tensor:
        """x : (..., in+1)  →  (..., out+1)"""
        c_out = c    if c    is not None else self.c
        c_in_ = c_in if c_in is not None else c_out

        w  = F.dropout(self.weight, p=self.dropout_p, training=self.training)
        fx = x @ w.T
        if self.bias is not None:
            fx = fx + self.bias

        if isinstance(c_in_, torch.Tensor) or isinstance(c_out, torch.Tensor):
            scale = (c_in_ / c_out) ** 0.5
        else:
            scale = math.sqrt(c_in_ / c_out)

        fx_sq  = (fx * fx).sum(dim=-1, keepdim=True)
        t_sq   = (c_in_ / c_out) * fx_sq + 1.0 / c_out
        t_part = t_sq.clamp(min=1e-6).sqrt()
        s_part = scale * fx

        return torch.cat([t_part, s_part], dim=-1)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"c={self.c}, dropout={self.dropout_p}, "
                f"bias={self.bias is not None}")


# ============================================================================
# 3.  HypLayerNorm  (spatial LayerNorm — no arcosh)
# ============================================================================

class HypLayerNorm(nn.Module):
    """Spatial LayerNorm: normalise xₛ directly, recompute x₀ from the
    manifold constraint x₀ = sqrt(1/c + ‖xₛ‖²).

    Replaces the old logmap0→LN→expmap0 pipeline, which called arcosh(√c·x₀)
    inside logmap0.  At depth r the gradient of arcosh is 1/sinh(r) → 0,
    effectively blocking updates to x₀ for deep nodes.

    This version has no arcosh anywhere in the forward or backward pass.
    The output is guaranteed to lie on H^d_c by construction.

    Pipeline:
        xₛ' = LN(xₛ)                     # normalise spatial part in-place
        x₀' = sqrt(1/c + ‖xₛ'‖²)        # recompute time coord from constraint
    """

    def __init__(self, d_model: int, c: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.c  = c
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        """x : (..., d+1)  →  (..., d+1).
        c : optional curvature override (float or scalar tensor); uses self.c if None.
        """
        c_val = c if c is not None else self.c
        xs = self.ln(x[..., 1:])                                    # (..., d)
        x0 = (1.0 / c_val + (xs * xs).sum(dim=-1, keepdim=True)
              ).clamp(min=1e-6).sqrt()                              # (..., 1)
        return torch.cat([x0, xs], dim=-1)                          # (..., d+1)


# ============================================================================
# 4.  HypActivation  (Hypformer pattern: activate spatial, reproject)
# ============================================================================

class HypActivation(nn.Module):
    """Apply a Euclidean activation to the spatial part, then reproject."""

    def __init__(self, act=F.gelu, c: float = 1.0):
        super().__init__()
        self.act = act
        self.c   = c

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        """c : optional curvature override; uses self.c if None."""
        c_val = c if c is not None else self.c
        spatial = self.act(x[..., 1:])
        return Hyperboloid.proj(
            torch.cat([x.new_zeros(*x.shape[:-1], 1), spatial], dim=-1),
            c_val,
        )


# ============================================================================
# 5.  LorentzGNN  (one-hop neighborhood aggregation on the hyperboloid)
#
#     For each node i, aggregate its neighbors using the Lorentz centroid:
#         µᵢ = Σⱼ wᵢⱼ xⱼ / (√c · ‖Σⱼ wᵢⱼ xⱼ‖_L)
#     where wᵢⱼ is a row-normalized adjacency weight (with self-loop).
#
#     This mirrors the GNN branch in Hypformer:  "spatial" message-passing
#     that runs in parallel with the Transformer attention branch, then is
#     fused via a shared residual connection.
# ============================================================================

class LorentzGNN(nn.Module):
    """Single-layer Lorentz GNN: centroid aggregation + HypLinear projection.

    The forward pass requires pre-computed row-normalised adjacency weights
    (including self-loops) so it can be shared across blocks without
    recomputing the sparsity pattern.
    """

    def __init__(self, d_model: int, c: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.c       = c
        self.proj    = HypLinear(d_model, d_model, c=c, dropout=dropout)

    def forward(self,
                x:           torch.Tensor,    # (B, N, d+1) on H^d_c
                adj_weights: torch.Tensor,    # (B, N, N)  row-normalised, with self-loops
                c=None,                       # optional curvature override
                ) -> torch.Tensor:
        """Returns (B, N, d+1) on H^d_c."""
        c_val = c if c is not None else self.c

        # Weighted sum of neighbour Lorentz vectors
        agg = torch.einsum("bnm,bmd->bnd", adj_weights, x)   # (B, N, d+1)

        # Re-project to manifold via Lorentz centroid normalisation
        # µ = agg / (√c · ‖agg‖_L)
        lor_sq   = (-Hyperboloid.minkowski_dot(agg, agg, keepdim=True)
                    ).clamp(min=1e-6)                          # (B, N, 1)
        agg      = agg / (c_val ** 0.5 * lor_sq.sqrt())       # (B, N, d+1)

        return self.proj(agg, c=c_val)                         # (B, N, d+1)


# ============================================================================
# 6.  GraphAttnBias  (Graphormer §3: structural encoding as additive bias)
#
#     Graphormer adds per-head bias to raw attention scores before softmax:
#         attn_weights += attn_bias    (shape: B, H, N, N)
#     This module precomputes that bias from:
#       (a) shortest-path distance (SPD) bucket embeddings  [Graphormer §3.1]
#       (b) direct-adjacency indicator embedding
# ============================================================================

class GraphAttnBias(nn.Module):
    """Precompute (B, H, N, N) attention bias from graph structure.

    Usage (Graphormer pattern):
        bias = GraphAttnBias(num_heads)(spd_matrix, adj_matrix)
        raw_scores += bias       # before softmax, per head
    """

    def __init__(self, num_heads: int, max_spd: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.max_spd   = max_spd
        # SPD buckets: 0 = self-loop, 1…max_spd = distances, max_spd+1 = unreachable
        self.spd_embed  = nn.Embedding(max_spd + 2, num_heads)
        self.edge_embed = nn.Embedding(2, num_heads)    # 0 = no edge, 1 = edge

    def forward(self,
                spd_matrix: torch.Tensor,    # (B, N, N) int64, SPD values
                adj_matrix: torch.Tensor,    # (B, N, N) bool, direct edges
                ) -> torch.Tensor:
        """Returns attn_bias : (B, H, N, N)."""
        spd_clamp  = spd_matrix.clamp(0, self.max_spd + 1)       # (B, N, N)
        spd_bias   = self.spd_embed(spd_clamp)                    # (B, N, N, H)
        edge_bias  = self.edge_embed(adj_matrix.long())           # (B, N, N, H)
        return (spd_bias + edge_bias).permute(0, 3, 1, 2)         # (B, H, N, N)


# ============================================================================
# 6.  BusemannMHA  (Hybrid HSA + Graphormer-style bias injection)
#
#     Hybrid attention score per head h:
#
#       align_ij = ⟨vᵢ, kⱼ,p⟩                    directional alignment in Poincaré repr
#       s_ij     = gate(qᵢ) · [ τ_dot · align_ij
#                               + αₕ  · B̃(align_ij) ]
#
#     where:
#       qᵢ,p = qᵢₛ / qᵢ₀  ∈ Bᵈ           Poincaré-ball coords of query (‖·‖ < 1)
#       kⱼ,p = kⱼₛ / kⱼ₀  ∈ Bᵈ           Poincaré-ball coords of key
#       vᵢ   = qᵢ,p / ‖qᵢ,p‖             ideal-boundary direction (unit vector)
#       gate(qᵢ) = ‖qᵢ,p‖ = ‖qᵢₛ‖/qᵢ₀   Poincaré radius ∈ [0,1)
#                                           = 0 near origin (direction ill-defined)
#                                           → 1 for deep nodes (direction stable)
#       align_ij = ⟨vᵢ, kⱼ,p⟩            ∈ (-1,1)  bounded, well-scaled
#
#       B̃(a) = −log(1 − a)               centered Busemann via Poincaré depth
#                                          (strips depth-saturation offset log(k₀))
#       τ_dot  per-head temperature, init 1/√d_head
#       αₕ     per-head mixing weight, init 0 (starts as pure Poincaré-dot attention)
#
#     Why Poincaré dot instead of Lorentz inner product ⟨q,k⟩_L:
#       Lorentz: ⟨q,k⟩_L = −q₀k₀ + ⟨qₛ,kₛ⟩ ~ −cosh²(r) → −2.2M at depth 8.
#       All scores O(−2.2M/√d), softmax → silently uniform, no gradient.
#       Poincaré dot: ⟨qₛ/q₀, kₛ/k₀⟩ ∈ (−1,1), always well-scaled.
#
#     The score factors as gate(q) · f(align), so gate appears once and
#     both the linear-Poincaré and log-Busemann terms share the same
#     directional alignment computation.
#
#     Value aggregation uses the Lorentz centroid (FHNN eq. 4):
#       µ = Σaᵢⱼvⱼ / (√c · ‖Σaᵢⱼvⱼ‖_L)
# ============================================================================

class BusemannMHA(nn.Module):
    """Hybrid Horospherical Multi-Head Attention with Graphormer structural bias.

    attn_proj (v4): how Q/K/V are mapped from d_model Lorentz vectors to per-head space
      "hyp_linear" — all HypLinear (same as v2)
      "htc_qk"     — HTC for Q and K, HypLinear for V (Route A default)
      "htc_qkv"    — all HTC (matches v3 attention maps; FFN still HypLinear in v4)

    score_type controls the attention score:
      "busemann"       (default) — hybrid Poincaré-dot + gated centered Busemann
      "busemann_full"            — original proposal: −τ·B_{v_i}(k_j)
                                   B_{v}(x) = log(x₀ − ⟨xₛ,v⟩),  v=normalize(qₛ)
                                   Directed: scores how deep k is in the horoball of q
                                   Encodes BOTH depth (via k₀) and alignment (via ⟨kₛ,v⟩)
      "hyp_dist"                 — negative squared hyperbolic distance (ablation baseline)
                                   cosh(d·√c) = (1−⟨q_p,k_p⟩) / (√(1−‖q_p‖²)·√(1−‖k_p‖²))
      "hypformer_kernel"         — Hypformer's focused feature-map kernel,
                                   materialised pairwise so the same k-hop mask,
                                   value aggregation, and backbone remain fixed
      "busemann_log_depth_sym"   — bidirectional log-depth score, averaging
                                   query-to-key and key-to-query LD terms
      "busemann_log_depth_linear" — positive landmark approximation to LD-HSA.
                                   Uses M key coresets and O(NM) attention,
                                   which is linear in N for fixed M.
      "busemann_log_depth_local_landmark" — exact sparse local LD-HSA in
                                   parallel with global landmark LD-HSA.
      "busemann_log_depth_local_dictionary" — exact sparse local LD-HSA in
                                   parallel with exact-linear global boundary-
                                   dictionary HSA. Costs O(E + NM).
      "busemann_log_depth_linear_rf" — positive random-feature approximation
                                   of the LD-HSA kernel. Uses R features and
                                   associative O(NR) kernel attention.
      "busemann_log_depth_linear_dictionary" — exact finite-rank Busemann
                                   attention over M learned ideal-boundary
                                   directions, with associative O(NM)
                                   attention.
    """

    def __init__(self, d_model: int, num_heads: int,
                 c: float = 1.0, dropout: float = 0.0,
                 score_type: str = "busemann",
                 attn_proj: str = "htc_qk",
                 linear_landmarks: int = 64,
                 linear_rf_features: int = 128,
                 linear_rf_seed: int = 0,
                 linear_rf_t_max: float = 12.0,
                 linear_rf_radial_scale: Optional[float] = None,
                 linear_boundary_directions: int = 64,
                 linear_boundary_seed: int = 0,
                 linear_boundary_gate_init: Optional[float] = None,
                 linear_boundary_mix_init: float = 0.1,
                 linear_boundary_scales: int = 1,
                 linear_boundary_scale_spread: float = 0.35,
                 linear_boundary_fusion: str = "additive",
                 linear_boundary_preserve_rng: bool = False,
                 ld_tau_init: Optional[float] = None,
                 ld_gamma_init: Optional[float] = None,
                 linear_boundary_residual_norm: str = "none",
                 linear_boundary_residual_max_gain: float = 100.0,
                 linear_boundary_residual_eps: float = 1e-6,
                 linear_boundary_checkpoint_heads: bool = False,
                 linear_boundary_key_normalization: str = "none",
                 linear_boundary_fixed_exponent: Optional[float] = None,
                 linear_boundary_scale_routing: str = "adaptive"):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        _valid = ("busemann", "busemann_full", "hyp_dist",
                  "busemann_depth_gate", "busemann_sym", "lorentz_dot",
                  "busemann_log_depth", "busemann_log_depth_sym",
                  "busemann_log_depth_local_landmark",
                  "busemann_log_depth_local_dictionary",
                  "busemann_log_depth_linear",
                  "busemann_log_depth_linear_rf",
                  "busemann_log_depth_linear_dictionary",
                  "busemann_add_depth",
                  "hypformer_kernel")
        assert score_type in _valid, \
            f"score_type must be one of {_valid}, got {score_type!r}"
        _ap = ("hyp_linear", "htc_qk", "htc_qkv")
        assert attn_proj in _ap, f"attn_proj must be one of {_ap}, got {attn_proj!r}"
        if linear_landmarks <= 0:
            raise ValueError(
                f"linear_landmarks must be positive, got {linear_landmarks}"
            )
        if linear_rf_features <= 0:
            raise ValueError(
                f"linear_rf_features must be positive, got {linear_rf_features}"
            )
        if linear_rf_t_max <= 0:
            raise ValueError(
                f"linear_rf_t_max must be positive, got {linear_rf_t_max}"
            )
        if linear_rf_radial_scale is not None and linear_rf_radial_scale <= 0:
            raise ValueError(
                "linear_rf_radial_scale must be positive when provided, "
                f"got {linear_rf_radial_scale}"
            )
        if linear_boundary_directions <= 0:
            raise ValueError(
                "linear_boundary_directions must be positive, got "
                f"{linear_boundary_directions}"
            )
        if (
            linear_boundary_gate_init is not None
            and linear_boundary_gate_init <= 0
        ):
            raise ValueError(
                "linear_boundary_gate_init must be positive when provided, "
                f"got {linear_boundary_gate_init}"
            )
        if (
            not math.isfinite(linear_boundary_mix_init)
            or (
                linear_boundary_fusion == "additive"
                and linear_boundary_mix_init <= 0
            )
        ):
            raise ValueError(
                "linear_boundary_mix_init must be finite, and positive for "
                "additive fusion; got "
                f"{linear_boundary_mix_init}"
            )
        if linear_boundary_scales <= 0:
            raise ValueError(
                "linear_boundary_scales must be positive, got "
                f"{linear_boundary_scales}"
            )
        if (
            not math.isfinite(linear_boundary_scale_spread)
            or linear_boundary_scale_spread < 0
        ):
            raise ValueError(
                "linear_boundary_scale_spread must be finite and "
                f"non-negative, got {linear_boundary_scale_spread}"
            )
        if linear_boundary_fusion not in ("additive", "centered_signed"):
            raise ValueError(
                "linear_boundary_fusion must be 'additive' or "
                f"'centered_signed', got {linear_boundary_fusion!r}"
            )
        if linear_boundary_residual_norm not in ("none", "match_local_rms"):
            raise ValueError(
                "linear_boundary_residual_norm must be 'none' or "
                f"'match_local_rms', got {linear_boundary_residual_norm!r}"
            )
        if linear_boundary_key_normalization not in ("none", "per_atom"):
            raise ValueError(
                "linear_boundary_key_normalization must be 'none' or "
                f"'per_atom', got {linear_boundary_key_normalization!r}"
            )
        _validate_boundary_scale_routing(linear_boundary_scale_routing)
        if linear_boundary_fixed_exponent is not None and (
            isinstance(linear_boundary_fixed_exponent, bool)
            or not isinstance(linear_boundary_fixed_exponent, Real)
            or not math.isfinite(linear_boundary_fixed_exponent)
            or linear_boundary_fixed_exponent <= 0
        ):
            raise ValueError(
                "linear_boundary_fixed_exponent must be None or a finite "
                f"positive scalar, got {linear_boundary_fixed_exponent!r}"
            )
        if (
            not math.isfinite(linear_boundary_residual_max_gain)
            or linear_boundary_residual_max_gain <= 0
        ):
            raise ValueError("linear_boundary_residual_max_gain must be finite and positive")
        if (
            not math.isfinite(linear_boundary_residual_eps)
            or linear_boundary_residual_eps <= 0
        ):
            raise ValueError("linear_boundary_residual_eps must be finite and positive")
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_head     = d_model // num_heads
        self.c          = c
        self.score_type = score_type
        self.attn_proj  = attn_proj
        self.linear_landmarks = int(linear_landmarks)
        self.linear_rf_features = int(linear_rf_features)
        self.linear_rf_seed = int(linear_rf_seed)
        self.linear_rf_t_max = float(linear_rf_t_max)
        self.linear_rf_radial_scale = (
            float(linear_rf_radial_scale)
            if linear_rf_radial_scale is not None
            else 2.0 * math.sqrt(self.d_head)
        )
        self.linear_boundary_directions = int(linear_boundary_directions)
        self.linear_boundary_seed = int(linear_boundary_seed)
        self.linear_boundary_scales = int(linear_boundary_scales)
        self.linear_boundary_scale_spread = float(
            linear_boundary_scale_spread
        )
        self.linear_boundary_fusion = str(linear_boundary_fusion)
        self.linear_boundary_preserve_rng = bool(linear_boundary_preserve_rng)
        self.linear_boundary_residual_norm = str(linear_boundary_residual_norm)
        self.linear_boundary_residual_max_gain = float(linear_boundary_residual_max_gain)
        self.linear_boundary_residual_eps = float(linear_boundary_residual_eps)
        self.linear_boundary_checkpoint_heads = bool(linear_boundary_checkpoint_heads)
        self.linear_boundary_key_normalization = str(linear_boundary_key_normalization)
        self.linear_boundary_scale_routing = linear_boundary_scale_routing
        self.linear_boundary_fixed_exponent = (
            None if linear_boundary_fixed_exponent is None
            else float(linear_boundary_fixed_exponent)
        )
        # Latest per-graph forward diagnostics, not parameters/checkpoint state.
        # Keep these detached on-device to avoid synchronization in training.
        self.boundary_residual_diagnostics: dict[str, torch.Tensor] = {}
        # Only populated by multiscale no-grad evaluation; no node arrays.
        self.boundary_scale_query_diagnostics: dict[int, dict] = {}

        def _make_qk():
            if attn_proj in ("htc_qk", "htc_qkv"):
                return nn.ModuleList(
                    [HTC(d_model, self.d_head, c=c, dropout=dropout)
                     for _ in range(num_heads)])
            return nn.ModuleList(
                [HypLinear(d_model, self.d_head, c=c, dropout=dropout)
                 for _ in range(num_heads)])

        def _make_v():
            if attn_proj == "htc_qkv":
                return nn.ModuleList(
                    [HTC(d_model, self.d_head, c=c, dropout=dropout)
                     for _ in range(num_heads)])
            return nn.ModuleList(
                [HypLinear(d_model, self.d_head, c=c, dropout=dropout)
                 for _ in range(num_heads)])

        self.W_Q = _make_qk()
        self.W_K = _make_qk()
        self.W_V = _make_v()

        # Shared temperature parameter (used by both score types)
        self.log_tau_dot = nn.Parameter(
            torch.full((num_heads,), -0.5 * math.log(self.d_head)))

        if score_type == "busemann":
            # Centered-Busemann per-head mixing weight α, init = 0
            # (model starts as pure Poincaré-dot; α grows during training)
            self.alpha = nn.Parameter(torch.zeros(num_heads))

        if score_type in (
            "busemann_log_depth",
            "busemann_log_depth_sym",
            "busemann_log_depth_local_landmark",
            "busemann_log_depth_local_dictionary",
            "busemann_log_depth_linear",
            "busemann_log_depth_linear_rf",
            "busemann_log_depth_linear_dictionary",
        ):
            # s_ij = −τ_h · (1 + γ_h · log q_{i,0}) · B_{v_i}(k_j)
            # τ_h = softplus(τ̂_h), init τ_h ≈ √d_h
            # γ_h = softplus(g_h),  init γ_h ≈ 0.049  (near-zero start)
            target_tau = math.sqrt(self.d_head)
            if ld_tau_init is None:
                # Preserve the submitted model's exact default initialization.
                tau_init = math.log(math.expm1(target_tau))
            else:
                tau_init = self._positive_to_softplus_logit(
                    ld_tau_init, "ld_tau_init"
                )
            if ld_gamma_init is None:
                # Preserve the submitted model's exact default initialization.
                gamma_init = -3.0
            else:
                gamma_init = self._positive_to_softplus_logit(
                    ld_gamma_init, "ld_gamma_init"
                )
            self.tau_logit   = nn.Parameter(torch.full((num_heads,), tau_init))
            self.gamma_logit = nn.Parameter(torch.full((num_heads,), gamma_init))

        if score_type in (
            "busemann_log_depth_linear_dictionary",
            "busemann_log_depth_local_dictionary",
        ):
            # The visual boundary at the Lorentz origin is S^(d_head-1).
            # A local generator keeps the dictionary seed from changing the
            # paired Q/K/V initialization used by controlled experiments.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.linear_boundary_seed)
            initial_directions = torch.randn(
                num_heads,
                self.linear_boundary_directions,
                self.d_head,
                generator=generator,
            )
            initial_directions = F.normalize(
                initial_directions, dim=-1
            )
            self.boundary_direction_raw = nn.Parameter(initial_directions)
            target_gate = (
                math.sqrt(self.d_head)
                if linear_boundary_gate_init is None
                else float(linear_boundary_gate_init)
            )
            gate_init = self._positive_to_softplus_logit(
                target_gate, "linear_boundary_gate_init"
            )
            self.boundary_gate_logit = nn.Parameter(
                torch.full((num_heads,), gate_init)
            )
            # These are log mixture priors in the vMF interpretation.
            self.boundary_gate_bias = nn.Parameter(torch.zeros(
                num_heads, self.linear_boundary_directions
            ))
            if self.linear_boundary_scales > 1:
                # Adaptive queries route by direction and Busemann sharpness
                # using M*G channels (rank <= M*G). Static scale routing keeps
                # these parameters/channels but collapses to rank <= M in
                # evaluation/dropout-off; feature dropout can break that form.
                offsets = torch.linspace(
                    -self.linear_boundary_scale_spread,
                    self.linear_boundary_scale_spread,
                    self.linear_boundary_scales,
                )
                self.boundary_exponent_offsets = nn.Parameter(
                    offsets.unsqueeze(0).expand(num_heads, -1).clone()
                )
                self.boundary_scale_gate_logit = nn.Parameter(
                    torch.full((num_heads,), gate_init)
                )
                self.boundary_scale_gate_bias = nn.Parameter(torch.zeros(
                    num_heads, self.linear_boundary_scales
                ))

        if score_type == "busemann_log_depth_local_landmark":
            # The local branch starts dominant. softplus(log(expm1(0.1)))=0.1.
            self.landmark_mix_logit = nn.Parameter(
                torch.tensor(math.log(math.expm1(0.1)))
            )

        if score_type == "busemann_log_depth_local_dictionary":
            if self.linear_boundary_fusion == "additive":
                # Backward-compatible positive additive fusion.
                mix_parameter = self._positive_to_softplus_logit(
                    linear_boundary_mix_init,
                    "linear_boundary_mix_init",
                )
            else:
                # A zero value is an exact local-HSA initialization. Unlike a
                # positive softplus gate, this signed residual can reject or
                # subtract a harmful global component.
                mix_parameter = float(linear_boundary_mix_init)
            self.boundary_mix_logit = nn.Parameter(
                torch.tensor(mix_parameter)
            )

        if score_type == "busemann_log_depth_linear_rf":
            # Fixed positive random features. A local Python RNG avoids
            # perturbing torch's initialization RNG and paired experiment seeds.
            proposal_shape = math.sqrt(self.d_head)
            rng = random.Random(self.linear_rf_seed)
            t_samples = []
            omega_samples = []
            zeta_samples = []

            def _sample_truncated_gamma() -> float:
                # Rejection sampling gives Gamma(a0,1) conditioned on
                # T <= T_max. The missing CDF normalization is a global
                # positive factor, so it cancels in the Lorentz centroid.
                for _attempt in range(100_000):
                    sample = rng.gammavariate(proposal_shape, 1.0)
                    if sample <= self.linear_rf_t_max:
                        return max(sample, 1e-6)
                raise ValueError(
                    "linear_rf_t_max is too small for stable truncated-Gamma "
                    f"sampling (shape={proposal_shape:.3g}, "
                    f"t_max={self.linear_rf_t_max:.3g})"
                )

            for _head in range(num_heads):
                t_samples.append([
                    _sample_truncated_gamma()
                    for _ in range(self.linear_rf_features)
                ])
                omega_samples.append([
                    [rng.gauss(0.0, 1.0) for _ in range(self.d_head)]
                    for _ in range(self.linear_rf_features)
                ])
                zeta_samples.append([
                    rng.gauss(0.0, 1.0)
                    for _ in range(self.linear_rf_features)
                ])
            self.register_buffer(
                "linear_rf_t",
                torch.tensor(t_samples, dtype=torch.float32),
            )
            self.register_buffer(
                "linear_rf_omega",
                torch.tensor(omega_samples, dtype=torch.float32),
            )
            self.register_buffer(
                "linear_rf_zeta",
                torch.tensor(zeta_samples, dtype=torch.float32),
            )
            self.linear_rf_proposal_shape = proposal_shape

        if score_type == "busemann_add_depth":
            # s_ij = −τ_h · B_{v_i}(k_j) + β_h · log q_{i,0}
            # β_h init = 0: model starts as pure busemann_full, β adapts
            self.beta = nn.Parameter(torch.zeros(num_heads))

        self.attn_drop = nn.Dropout(dropout)

        # Output projection: spatial concat d_model → d_model, back to manifold
        self.out_proj  = HypLinear(num_heads * self.d_head, d_model, c=c)

    @staticmethod
    def _positive_to_softplus_logit(value: float, name: str) -> float:
        """Return a stable inverse-softplus for a positive actual value."""
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
        # log(exp(value) - 1), written to avoid overflow for large values and
        # cancellation for small values such as 1/16.
        return value + math.log(-math.expm1(-value))

    @staticmethod
    def _hypformer_feature_map(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
        """Focused positive feature map from Hypformer's released attention.

        Taking ``log(phi(q) @ phi(k))`` before softmax yields exactly the
        normalized pairwise kernel weights while still allowing the controlled
        k-hop mask used by the other score variants.
        """
        phi = F.relu(x) + 1e-6
        norm_phi = phi.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        powered = phi.pow(power)
        norm_powered = powered.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return (norm_phi / norm_powered) * powered

    def _busemann_log_depth_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head: int,
    ) -> torch.Tensor:
        """Pairwise LD-HSA scores for arbitrary query/key sequence lengths."""
        tau_h = F.softplus(self.tau_logit[head])
        gamma_h = F.softplus(self.gamma_logit[head])
        v_dir = F.normalize(q[..., 1:], dim=-1)
        inner = torch.einsum("bnd,bmd->bnm", v_dir, k[..., 1:])
        k0 = k[..., 0].unsqueeze(1)
        buse_arg = (k0 - inner).clamp(min=1e-6)
        q0 = q[..., 0:1].clamp(min=1.0 + 1e-6)
        depth_fac = 1.0 + gamma_h * torch.log(q0)
        return -tau_h * depth_fac * torch.log(buse_arg)

    def boundary_directions(self) -> torch.Tensor:
        """Return unit ideal-boundary directions for every attention head."""
        if not hasattr(self, "boundary_direction_raw"):
            raise RuntimeError(
                "boundary directions exist only for "
                "score_type='busemann_log_depth_linear_dictionary'"
            )
        return F.normalize(self.boundary_direction_raw, dim=-1)

    def boundary_mix_scale(self) -> torch.Tensor:
        """Return the actual local/global fusion coefficient."""
        if not hasattr(self, "boundary_mix_logit"):
            raise RuntimeError(
                "boundary mix exists only for local dictionary HSA"
            )
        if self.linear_boundary_fusion == "centered_signed":
            return self.boundary_mix_logit
        return F.softplus(self.boundary_mix_logit)

    def _calibrate_boundary_residual(
        self,
        global_centered: torch.Tensor,
        local_spatial: torch.Tensor,
        mix_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Optionally match a centered global residual to local node variation.

        The graphwise gain uses RMS over nodes and spatial channels, is bounded,
        and is detached: this is a scale calibration, not a new route for
        gradients through the local branch. It adds O(Nd) work and no parameters.
        A zero signed mix still gives exact local outputs and shared gradients.
        ``effective_rms`` includes the signed mix magnitude; ``calibrated_rms``
        measures the returned residual before mixing.
        """
        with torch.no_grad():
            compute_dtype = (
                torch.float32
                if global_centered.dtype in (torch.float16, torch.bfloat16)
                else global_centered.dtype
            )
            global_work = global_centered.detach().to(compute_dtype)
            local_work = local_spatial.detach().to(compute_dtype)
            local_centered = local_work - local_work.mean(dim=1, keepdim=True)
            global_rms = global_work.square().mean(dim=(1, 2), keepdim=True).sqrt()
            local_rms = local_centered.square().mean(dim=(1, 2), keepdim=True).sqrt()
            if self.linear_boundary_residual_norm == "match_local_rms":
                gain = (
                    local_rms
                    / global_rms.clamp_min(self.linear_boundary_residual_eps)
                ).clamp(max=self.linear_boundary_residual_max_gain)
            else:
                gain = torch.ones_like(global_rms)
            calibrated_rms = global_rms * gain
            effective_rms = calibrated_rms * mix_scale.detach().abs().to(compute_dtype)
            self.boundary_residual_diagnostics = {
                "raw_global_centered_rms": global_rms.reshape(-1),
                "local_centered_rms": local_rms.reshape(-1),
                "gain": gain.reshape(-1),
                "calibrated_rms": calibrated_rms.reshape(-1),
                "effective_rms": effective_rms.reshape(-1),
            }
        if self.linear_boundary_residual_norm == "none":
            return global_centered
        return global_centered * gain.to(global_centered.dtype)

    @torch.no_grad()
    def initialize_boundary_dictionary(
        self,
        query_directions: torch.Tensor,
        iterations: int = 20,
    ) -> None:
        """Initialize each head's dictionary by deterministic spherical k-means.

        ``query_directions`` has shape ``(H, S, d_head)``. The initializer uses
        farthest-point seeding under cosine distance, followed by hard vMF
        maximum-likelihood updates. It changes only the boundary atoms; gate
        concentrations and mixture priors remain trainable parameters.
        """
        if not hasattr(self, "boundary_direction_raw"):
            raise RuntimeError(
                "spherical initialization requires boundary-dictionary HSA"
            )
        expected_rank = 3
        if query_directions.ndim != expected_rank:
            raise ValueError(
                "query_directions must have shape (heads, samples, d_head), "
                f"got {tuple(query_directions.shape)}"
            )
        heads, samples, dimension = query_directions.shape
        expected = (self.num_heads, self.d_head)
        if (heads, dimension) != expected:
            raise ValueError(
                "query_directions head/dimension mismatch: expected "
                f"({expected[0]}, samples, {expected[1]}), got "
                f"{tuple(query_directions.shape)}"
            )
        if samples < self.linear_boundary_directions:
            raise ValueError(
                "spherical initialization needs at least as many samples as "
                "directions: got "
                f"{samples} < {self.linear_boundary_directions}"
            )
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}")

        work = F.normalize(
            query_directions.to(
                device=self.boundary_direction_raw.device,
                dtype=self.boundary_direction_raw.dtype,
            ),
            dim=-1,
        )
        centers_per_head = []
        for head in range(self.num_heads):
            points = work[head]
            # Start at the point closest to the spherical mean, then cover the
            # remaining support by maximizing distance to the selected set.
            mean_direction = F.normalize(
                points.mean(dim=0, keepdim=True), dim=-1
            ).squeeze(0)
            current = int((points @ mean_direction).argmax().item())
            selected = []
            closest_similarity = points.new_full(
                (samples,), float("-inf")
            )
            for _ in range(self.linear_boundary_directions):
                center = points[current]
                selected.append(center)
                similarity = points @ center
                closest_similarity = torch.maximum(
                    closest_similarity, similarity
                )
                current = int(closest_similarity.argmin().item())
            centers = torch.stack(selected, dim=0)

            for _ in range(iterations):
                assignment = (points @ centers.T).argmax(dim=-1)
                updated = torch.zeros_like(centers)
                updated.scatter_add_(
                    0,
                    assignment.unsqueeze(-1).expand(-1, self.d_head),
                    points,
                )
                counts = torch.bincount(
                    assignment,
                    minlength=self.linear_boundary_directions,
                )
                nonempty = counts > 0
                updated[nonempty] = F.normalize(
                    updated[nonempty], dim=-1
                )
                updated[~nonempty] = centers[~nonempty]
                centers = updated
            centers_per_head.append(centers)

        self.boundary_direction_raw.copy_(
            torch.stack(centers_per_head, dim=0)
        )

    def _boundary_dictionary_log_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log features for exact finite-rank boundary-dictionary HSA.

        With unit atoms xi_m and optional exponent anchors g, this implements

          phi_m(q) = softmax_m(kappa(q) <u(q), xi_m> + b_m),
          rho_g(q) = softmax_g(-eta (log d(q) - delta_g)^2 + c_g),
          psi_mg(k) = exp(-a_h exp(delta_g) B_{xi_m}(k)),
          K_MG(q,k) = sum_mg phi_m(q) rho_g(q) psi_mg(k).

        The LD-HSA gamma parameter modulates routing concentration rather than
        a query-dependent key exponent. With multiple exponent anchors it also
        routes queries toward the anchor closest to the desired depth factor.
        The key map remains independent of the query, which is required for an
        exact associative contraction. One scale exactly recovers the original
        dictionary kernel and parameterization.

        By default a_h = softplus(tau_logit[h]). The opt-in fixed exponent
        replaces only this dictionary base exponent, leaving the local LD-HSA
        temperature trainable and unchanged. With multiple scales, effective
        exponents remain fixed_base * exp(delta_g); the offsets still learn.
        The override is configuration, not a parameter or checkpoint buffer.
        """
        compute_dtype = (
            torch.float32
            if q.dtype in (torch.float16, torch.bfloat16)
            else q.dtype
        )
        q_work = q.to(dtype=compute_dtype)
        k_work = k.to(dtype=compute_dtype)
        directions = self.boundary_directions()[head].to(
            device=q.device, dtype=compute_dtype
        )

        query_direction = F.normalize(q_work[..., 1:], dim=-1)
        q0 = q_work[..., 0:1].clamp(min=1.0 + 1e-6)
        gamma_h = F.softplus(self.gamma_logit[head]).to(
            dtype=compute_dtype
        )
        gate_concentration = F.softplus(
            self.boundary_gate_logit[head]
        ).to(dtype=compute_dtype)
        depth_fac = 1.0 + gamma_h * torch.log(q0)
        gate_logits = (
            gate_concentration
            * depth_fac
            * torch.einsum(
                "bnd,md->bnm", query_direction, directions
            )
            + self.boundary_gate_bias[head].to(dtype=compute_dtype)
        )
        log_direction_query = F.log_softmax(gate_logits, dim=-1)

        inner = torch.einsum(
            "bnd,md->bnm", k_work[..., 1:], directions
        )
        buse_arg = (k_work[..., 0:1] - inner).clamp(min=1e-6)
        if self.linear_boundary_fixed_exponent is None:
            exponent = F.softplus(self.tau_logit[head]).to(
                dtype=compute_dtype
            )
        else:
            exponent = k_work.new_tensor(self.linear_boundary_fixed_exponent)
        log_buse_arg = torch.log(buse_arg)
        if self.linear_boundary_scales == 1:
            return log_direction_query, -exponent * log_buse_arg

        exponent_offsets = self.boundary_exponent_offsets[head].to(
            dtype=compute_dtype
        ).clamp(min=-2.0, max=2.0)
        scale_concentration = F.softplus(
            self.boundary_scale_gate_logit[head]
        ).to(dtype=compute_dtype)
        desired_log_offset = torch.log(depth_fac.clamp(min=1e-6))
        if self.linear_boundary_scale_routing == "static":
            # Matched control: keep learned scale atoms/gates, but remove only
            # the query-depth-dependent target. Direction routing stays intact.
            # Parameter/compute channels match adaptive; evaluation/dropout-off
            # rank does not: static scale weights can be absorbed into M keys.
            desired_log_offset = torch.zeros_like(desired_log_offset)
        scale_logits = (
            -scale_concentration
            * (
                desired_log_offset
                - exponent_offsets.view(1, 1, -1)
            ).square()
            + self.boundary_scale_gate_bias[head].to(dtype=compute_dtype)
        )
        log_scale_query = F.log_softmax(scale_logits, dim=-1)
        if not self.training and not torch.is_grad_enabled():
            self._record_boundary_scale_query_diagnostics(
                log_scale_query, desired_log_offset, head,
            )

        log_query = (
            log_direction_query.unsqueeze(-1)
            + log_scale_query.unsqueeze(-2)
        ).flatten(start_dim=-2)
        anchor_exponents = exponent * exponent_offsets.exp()
        log_key = (
            -log_buse_arg.unsqueeze(-1)
            * anchor_exponents.view(1, 1, 1, -1)
        ).flatten(start_dim=-2)
        return log_query, log_key

    def _record_boundary_scale_query_diagnostics(self, log_scale_query, desired_log_offset, head):
        """Detached, pre-key-mass/pre-dropout scale utilization in no-grad eval.

        Never called by training or gradient/checkpoint recomputation. Each
        head stores only O(G) summaries over all B*N original graph queries.
        """
        if self.linear_boundary_scales <= 1 or self.training or torch.is_grad_enabled():
            return
        self.boundary_scale_query_diagnostics.pop(head, None)
        query_count = log_scale_query.shape[0] * log_scale_query.shape[1]
        if query_count == 0:
            return
        probabilities = log_scale_query.detach().exp()
        probability_std, probability_mean = torch.std_mean(probabilities, dim=(0, 1), correction=0)
        desired_std, desired_mean = torch.std_mean(desired_log_offset.detach(), correction=0)
        self.boundary_scale_query_diagnostics[head] = {
            "query_count": query_count,
            "scale_probability_mean": probability_mean.detach(),
            "scale_probability_std_population": probability_std.detach(),
            "mean_entropy": torch.special.entr(probabilities).sum(-1).mean().detach(),
            "desired_log_offset_mean": desired_mean.detach(),
            "desired_log_offset_std_population": desired_std.detach(),
        }

    def _boundary_dictionary_feature_maps(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stable positive features for the exact dictionary kernel.

        A key-dependent scale for each feature is transferred to its query
        feature. A final query-row scale prevents overflow and cancels exactly
        in normalized attention and Lorentz-centroid aggregation.

        Opt-in ``per_atom`` instead normalizes each atom over this graph's
        keys. This defines a different positive, finite-rank kernel: query
        gates mix atom-conditional key distributions without the original
        per-atom key-mass (density) prior. Gradients flow through the key
        normalizer. The default kernel and its stabilization are unchanged.
        """
        log_query, log_key = self._boundary_dictionary_log_features(
            q, k, head
        )
        if self.linear_boundary_key_normalization == "per_atom":
            # Normalize unshifted features: the default's transferred key max
            # must not remain in the query gates of this different kernel.
            # log_query already normalizes direction and optional scale gates.
            # Make the long key axis contiguous for CUDA's last-axis softmax.
            # This is the same per-atom normalization, including its gradient;
            # the returned view is consumed directly by the associative sum.
            phi_key = F.softmax(
                log_key.transpose(1, 2).contiguous(), dim=-1,
            ).transpose(1, 2)
            return log_query.exp(), phi_key
        key_shift = log_key.amax(dim=1, keepdim=True).detach()
        shifted_query = log_query + key_shift
        query_shift = shifted_query.amax(dim=-1, keepdim=True).detach()
        phi_query = torch.exp(
            (shifted_query - query_shift).clamp(max=0.0)
        )
        phi_key = torch.exp((log_key - key_shift).clamp(max=0.0))
        return phi_query, phi_key

    @staticmethod
    def _farthest_point_indices(
        features: torch.Tensor,
        num_landmarks: int,
    ) -> torch.Tensor:
        """Deterministic farthest-point coreset indices.

        The first landmark is farthest from the origin. Subsequent landmarks
        maximize distance to the selected set. Selection is performed on
        detached features; gradients still flow through the pooled prototypes.
        """
        B, N, _ = features.shape
        M = min(int(num_landmarks), N)
        work = features.detach()
        batch = torch.arange(B, device=features.device)
        selected = torch.empty(B, M, dtype=torch.long, device=features.device)
        chosen = torch.zeros(B, N, dtype=torch.bool, device=features.device)

        current = work.square().sum(dim=-1).argmax(dim=-1)
        min_dist = torch.full(
            (B, N),
            float("inf"),
            dtype=features.dtype,
            device=features.device,
        )
        for landmark in range(M):
            selected[:, landmark] = current
            chosen[batch, current] = True
            center = work[batch, current].unsqueeze(1)
            distance = (work - center).square().sum(dim=-1)
            min_dist = torch.minimum(min_dist, distance)
            min_dist = min_dist.masked_fill(chosen, -1.0)
            current = min_dist.argmax(dim=-1)
        return selected

    @staticmethod
    def _gather_nodes(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return x.gather(1, gather_index)

    def _linear_landmark_coreset(
        self,
        k: torch.Tensor,
        value: torch.Tensor,
        c_val,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress keys/values into a positive horospherical coreset.

        Hard assignments P contain no negative entries. For cluster r, the
        approximation replaces every key score by the score of its pooled
        Lorentz prototype and retains the exact cluster multiplicity:

          sum_j K(q,k_j)V_j ~= sum_r K(q,l_r) count_r mean_r(V).
        """
        B, N, _ = k.shape
        M = min(self.linear_landmarks, N)
        if M == N:
            counts = k.new_ones(B, N)
            return k, value, counts

        sqrt_c = torch.as_tensor(c_val, dtype=k.dtype, device=k.device).sqrt()
        klein = k[..., 1:] / k[..., 0:1].clamp(min=1e-6)
        radial = torch.log(
            (sqrt_c * k[..., 0:1]).clamp(min=1.0 + 1e-6)
        )
        cluster_features = torch.cat([klein, radial], dim=-1)
        landmark_indices = self._farthest_point_indices(cluster_features, M)
        centers = self._gather_nodes(cluster_features.detach(), landmark_indices)

        # O(NM), but never O(N^2) when M is fixed.
        distances = torch.cdist(cluster_features.detach(), centers)
        assignment = distances.argmin(dim=-1)
        scatter_index_k = assignment.unsqueeze(-1).expand(-1, -1, k.shape[-1])
        scatter_index_v = assignment.unsqueeze(-1).expand(
            -1, -1, value.shape[-1]
        )

        counts = k.new_zeros(B, M)
        counts.scatter_add_(1, assignment, k.new_ones(B, N))

        key_sums = k.new_zeros(B, M, k.shape[-1])
        key_sums.scatter_add_(1, scatter_index_k, k)
        value_sums = value.new_zeros(B, M, value.shape[-1])
        value_sums.scatter_add_(1, scatter_index_v, value)

        denominator = counts.clamp(min=1.0).unsqueeze(-1)
        key_mean = key_sums / denominator
        value_mean = value_sums / denominator

        # Empty clusters are possible only under exact feature ties. Their
        # multiplicity is zero, but finite fallback prototypes avoid NaNs.
        sampled_k = self._gather_nodes(k, landmark_indices)
        sampled_v = self._gather_nodes(value, landmark_indices)
        nonempty = (counts > 0).unsqueeze(-1)
        key_mean = torch.where(nonempty, key_mean, sampled_k)
        value_mean = torch.where(nonempty, value_mean, sampled_v)

        key_landmarks = Hyperboloid.proj(
            torch.cat(
                [key_mean.new_zeros(B, M, 1), key_mean[..., 1:]],
                dim=-1,
            ),
            c_val,
        )
        return key_landmarks, value_mean, counts

    def _linear_rf_feature_maps(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head: int,
        c_val,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive features for the query-dependent LD-HSA kernel.

        For ``a(q)=tau*(1+gamma*log(q0))`` and ``u=normalize(q_s)``,

          exp(score(q,k)) = (k0 - <u,k_s>)**(-a(q)).

        Up to a positive query-only curvature factor, write ``z=k_s/k0`` and
        ``ell=log(sqrt(c)*k0)``. Then the kernel is

          exp(-a*ell) * (1 - <u,z>)**(-a).

        A scalar Gaussian feature linearizes the radial exponential. A
        Gamma-Laplace feature followed by a Gaussian positive feature
        linearizes the inverse angular factor. The resulting query and key
        maps are positive and support associative linear attention even though
        ``a`` varies by query.

        ``T`` uses rejection-sampled truncated Gamma draws for finite-variance
        numerical behavior. This estimates a truncated Laplace integral (up to
        a global factor that cancels), so it is a controlled biased
        approximation rather than the untruncated exactly unbiased estimator.
        ``linear_rf_radial_scale`` only balances the two Gaussian radial
        features; it does not change their expected kernel.
        """
        compute_dtype = (
            torch.float32
            if q.dtype in (torch.float16, torch.bfloat16)
            else q.dtype
        )
        q_work = q.to(dtype=compute_dtype)
        k_work = k.to(dtype=compute_dtype)

        tau_h = F.softplus(self.tau_logit[head]).to(dtype=compute_dtype)
        gamma_h = F.softplus(self.gamma_logit[head]).to(dtype=compute_dtype)
        q0 = q_work[..., 0:1].clamp(min=1.0 + 1e-6)
        exponent = tau_h * (1.0 + gamma_h * torch.log(q0))
        exponent = exponent.clamp(min=1e-3, max=32.0)

        direction = F.normalize(q_work[..., 1:], dim=-1)
        direction_sq = direction.square().sum(dim=-1, keepdim=True)

        sqrt_c = torch.as_tensor(
            c_val, dtype=compute_dtype, device=q.device
        ).sqrt()
        k0 = k_work[..., 0:1].clamp(min=1e-6)
        klein = k_work[..., 1:] / k0
        klein_sq = klein.square().sum(dim=-1, keepdim=True)
        log_radial = torch.log(
            (sqrt_c * k0).clamp(min=1.0 + 1e-6)
        )

        t = self.linear_rf_t[head].to(
            device=q.device, dtype=compute_dtype
        ).clamp(min=1e-6)
        omega = self.linear_rf_omega[head].to(
            device=q.device, dtype=compute_dtype
        )
        zeta = self.linear_rf_zeta[head].to(
            device=q.device, dtype=compute_dtype
        )
        sqrt_t_omega = t.sqrt().unsqueeze(-1) * omega

        angular_q = torch.einsum(
            "bnd,rd->bnr", direction, sqrt_t_omega
        )
        angular_k = torch.einsum(
            "bnd,rd->bnr", klein, sqrt_t_omega
        )

        proposal_shape = exponent.new_tensor(self.linear_rf_proposal_shape)
        log_importance = (
            torch.lgamma(proposal_shape)
            - torch.lgamma(exponent)
            + (exponent - proposal_shape) * t.log().view(1, 1, -1)
        )
        scaled_exponent = exponent / self.linear_rf_radial_scale
        log_query = (
            log_importance
            + zeta.view(1, 1, -1) * scaled_exponent
            - 0.5 * scaled_exponent.square()
            + angular_q
            - 0.5 * direction_sq * t.view(1, 1, -1)
        )
        scaled_radial = self.linear_rf_radial_scale * log_radial
        log_key = (
            -zeta.view(1, 1, -1) * scaled_radial
            - 0.5 * scaled_radial.square()
            + angular_k
            - 0.5 * klein_sq * t.view(1, 1, -1)
        )

        # Per-feature key shifts are transferred to the query map exactly.
        # The final per-query shift is a positive row scalar and therefore
        # cancels in normalized attention and in the Lorentz centroid.
        key_shift = log_key.amax(dim=1, keepdim=True).detach()
        shifted_query = log_query + key_shift
        query_shift = shifted_query.amax(dim=-1, keepdim=True).detach()
        # Both maxima are non-positive after shifting, so exponentiation
        # cannot overflow. Allowing very negative entries to underflow to
        # zero preserves the exact shift cancellation; flooring them would
        # add a data-dependent background kernel.
        log_query_stable = (shifted_query - query_shift).clamp(max=0.0)
        log_key_stable = (log_key - key_shift).clamp(max=0.0)

        scale = math.sqrt(self.linear_rf_features)
        phi_query = torch.exp(log_query_stable) / scale
        phi_key = torch.exp(log_key_stable) / scale
        return phi_query, phi_key

    @staticmethod
    def _normalise_lorentz_aggregate(
        aggregate: torch.Tensor,
        c_val,
    ) -> torch.Tensor:
        """Normalize a positive weighted ambient sum onto the hyperboloid."""
        c_tensor = torch.as_tensor(
            c_val, dtype=aggregate.dtype, device=aggregate.device
        )
        lorentz_sq = -Hyperboloid.minkowski_dot(
            aggregate, aggregate, keepdim=True
        )
        valid = (
            torch.isfinite(aggregate).all(dim=-1, keepdim=True)
            & (aggregate[..., 0:1] > 0)
            & (lorentz_sq > 1e-12)
        )
        normalized = aggregate / (
            c_tensor.sqrt() * lorentz_sq.clamp(min=1e-12).sqrt()
        )
        origin = torch.cat(
            [
                aggregate.new_ones(*aggregate.shape[:-1], 1)
                / c_tensor.sqrt(),
                aggregate.new_zeros(*aggregate.shape[:-1],
                                    aggregate.shape[-1] - 1),
            ],
            dim=-1,
        )
        return torch.where(valid, normalized, origin)

    def _forward_linear_rf_hsa(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        bool_mask: Optional[torch.Tensor],
        c_val,
    ) -> torch.Tensor:
        """Positive random-feature LD-HSA with O(NR) time and memory."""
        if attn_bias is not None or bool_mask is not None:
            raise ValueError(
                "busemann_log_depth_linear_rf supports global attention only; "
                "disable pairwise bool_mask/attn_bias (e.g. set khop=0 and "
                "Graphormer bias off)"
            )

        head_outs = []
        for h in range(self.num_heads):
            q = self.W_Q[h](x, c=c_val)
            k = self.W_K[h](x, c=c_val)
            value = self.W_V[h](x, c=c_val)
            phi_query, phi_key = self._linear_rf_feature_maps(
                q, k, h, c_val
            )

            # Feature dropout is used instead of inaccessible pairwise-weight
            # dropout. It preserves positivity and the O(NR) contraction.
            phi_query = self.attn_drop(phi_query)
            value_work = value.to(dtype=phi_key.dtype)
            key_value = torch.einsum(
                "bnr,bnd->brd", phi_key, value_work
            )
            aggregate = torch.einsum(
                "bnr,brd->bnd", phi_query, key_value
            )
            aggregate = self._normalise_lorentz_aggregate(
                aggregate, c_val
            )
            head_outs.append(aggregate.to(dtype=x.dtype))

        spatial_cat = torch.cat(
            [head_out[..., 1:] for head_out in head_outs], dim=-1
        )
        dummy = Hyperboloid.proj(
            torch.cat(
                [
                    spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1),
                    spatial_cat,
                ],
                dim=-1,
            ),
            c_val,
        )
        return self.out_proj(dummy, c=c_val)

    def _forward_boundary_dictionary_head(
        self,
        x: torch.Tensor,
        c_val,
        head: int,
    ) -> torch.Tensor:
        """Pure single-head dictionary computation, also used for recomputation.

        Keep projections and feature dropout inside the checkpoint boundary.
        The explicit head and curvature arguments avoid late-bound loop state
        and preserve learned-curvature gradients. Training/recomputation never
        updates diagnostics: non-reentrant checkpointing may stop early. The
        feature helper records scale summaries only during no-grad evaluation.
        """
        q = self.W_Q[head](x, c=c_val)
        k = self.W_K[head](x, c=c_val)
        value = self.W_V[head](x, c=c_val)
        phi_query, phi_key = self._boundary_dictionary_feature_maps(q, k, head)

        # Feature dropout retains the associative contraction. Pairwise
        # attention dropout is unavailable without materializing N x N.
        phi_query = self.attn_drop(phi_query)
        value_work = value.to(dtype=phi_key.dtype)
        key_value = torch.einsum("bnm,bnd->bmd", phi_key, value_work)
        aggregate = torch.einsum("bnm,bmd->bnd", phi_query, key_value)
        aggregate = self._normalise_lorentz_aggregate(aggregate, c_val)
        return aggregate.to(dtype=x.dtype)

    def _forward_boundary_dictionary_hsa(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        bool_mask: Optional[torch.Tensor],
        c_val,
    ) -> torch.Tensor:
        """Exact O(NM) attention over learned ideal-boundary directions."""
        if attn_bias is not None or bool_mask is not None:
            raise ValueError(
                "busemann_log_depth_linear_dictionary supports global "
                "attention only; disable pairwise bool_mask/attn_bias "
                "(e.g. set khop=0 and Graphormer bias off)"
            )

        if self.linear_boundary_scales > 1 and not self.training and not torch.is_grad_enabled():
            self.boundary_scale_query_diagnostics.clear()
        head_outs = []
        for head in range(self.num_heads):
            if (
                self.linear_boundary_checkpoint_heads
                and self.training
                and torch.is_grad_enabled()
            ):
                head_output = checkpoint(
                    self._forward_boundary_dictionary_head,
                    x,
                    c_val,
                    head,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                head_output = self._forward_boundary_dictionary_head(x, c_val, head)
            head_outs.append(head_output)

        spatial_cat = torch.cat(
            [head_out[..., 1:] for head_out in head_outs], dim=-1
        )
        dummy = Hyperboloid.proj(
            torch.cat(
                [
                    spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1),
                    spatial_cat,
                ],
                dim=-1,
            ),
            c_val,
        )
        return self.out_proj(dummy, c=c_val)

    def _forward_linear_hsa(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        bool_mask: Optional[torch.Tensor],
        c_val,
    ) -> torch.Tensor:
        """Positive landmark LD-HSA with O(NM) attention.

        Arbitrary pairwise masks and Graphormer biases do not factor through a
        global key coreset. Refusing them prevents an accidental dense fallback
        from being reported as linear attention.
        """
        if attn_bias is not None or bool_mask is not None:
            raise ValueError(
                "busemann_log_depth_linear supports global attention only; "
                "disable pairwise bool_mask/attn_bias (e.g. set khop=0 and "
                "Graphormer bias off)"
            )

        head_outs = []
        for h in range(self.num_heads):
            q = self.W_Q[h](x, c=c_val)
            k = self.W_K[h](x, c=c_val)
            value = self.W_V[h](x, c=c_val)
            key_landmarks, value_landmarks, counts = (
                self._linear_landmark_coreset(k, value, c_val)
            )

            scores = self._busemann_log_depth_scores(
                q, key_landmarks, h
            )
            log_counts = torch.where(
                counts > 0,
                counts.clamp(min=1.0).log(),
                counts.new_full(counts.shape, float("-inf")),
            )
            scores = scores + log_counts.unsqueeze(1)
            attn = self.attn_drop(F.softmax(scores, dim=-1))

            agg = torch.einsum(
                "bqm,bmd->bqd", attn, value_landmarks
            )
            lor_norm = (
                -Hyperboloid.minkowski_dot(agg, agg, keepdim=True)
            ).clamp(min=1e-6).sqrt()
            agg = agg / (torch.as_tensor(
                c_val, dtype=agg.dtype, device=agg.device
            ).sqrt() * lor_norm)
            head_outs.append(agg)

        spatial_cat = torch.cat(
            [head_out[..., 1:] for head_out in head_outs], dim=-1
        )
        dummy = Hyperboloid.proj(
            torch.cat(
                [spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1),
                 spatial_cat],
                dim=-1,
            ),
            c_val,
        )
        return self.out_proj(dummy, c=c_val)

    def _prepare_sparse_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and cache a query-major, coalesced attention edge list.

        ``edge_index[0]`` is the query (dense score row) and
        ``edge_index[1]`` is the key (dense score column). Edges must be sorted
        lexicographically by ``(query, key)``, contain no duplicates, and give
        every query at least one key. The benchmark CSR builder guarantees
        these properties and includes self-loops.
        """
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "attn_edge_index must have shape (2, E), got "
                f"{tuple(edge_index.shape)}"
            )
        if edge_index.dtype != torch.long:
            raise TypeError(
                "attn_edge_index must use torch.long indices, got "
                f"{edge_index.dtype}"
            )
        if edge_index.device != device:
            raise ValueError(
                "attn_edge_index and attention inputs must share a device"
            )
        if edge_index.shape[1] == 0:
            raise ValueError("attn_edge_index must contain at least one edge")

        cache_key = (
            edge_index.data_ptr(),
            edge_index._version,
            edge_index.shape[1],
            num_nodes,
            edge_index.device,
        )
        if getattr(self, "_sparse_edge_cache_key", None) == cache_key:
            return (
                self._sparse_query_index,
                self._sparse_key_index,
                self._sparse_query_lengths,
            )

        query_index = edge_index[0]
        key_index = edge_index[1]
        encoded = query_index * num_nodes + key_index
        valid_bounds = (
            (query_index >= 0).all()
            & (query_index < num_nodes).all()
            & (key_index >= 0).all()
            & (key_index < num_nodes).all()
        )
        if not bool(valid_bounds):
            raise ValueError("attn_edge_index contains an out-of-range node")
        if encoded.numel() > 1 and not bool((encoded[1:] > encoded[:-1]).all()):
            raise ValueError(
                "attn_edge_index must be query-major sorted and coalesced "
                "(no duplicate pairs)"
            )

        lengths = torch.bincount(query_index, minlength=num_nodes)
        if not bool((lengths > 0).all()):
            raise ValueError(
                "every query must have at least one sparse attention edge; "
                "include self-loops for isolated nodes"
            )

        self._sparse_edge_cache_key = cache_key
        self._sparse_query_index = query_index
        self._sparse_key_index = key_index
        self._sparse_query_lengths = lengths
        return query_index, key_index, lengths

    @staticmethod
    def _sparse_segment_softmax(
        scores: torch.Tensor,
        query_index: torch.Tensor,
        query_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Stable softmax over query-major edge segments.

        ``scores`` has shape ``(B, E)`` and the returned tensor has the same
        shape. ``torch.segment_reduce`` avoids atomic scatter reductions and is
        compatible with deterministic CUDA training when supported by PyTorch.
        """
        B, _ = scores.shape
        lengths = query_lengths.unsqueeze(0).expand(B, -1)
        # The max is only a numerical shift. Detaching it preserves the exact
        # softmax derivative while avoiding an unnecessary max-reduction
        # backward (important for deterministic accelerator execution).
        row_max = torch.segment_reduce(
            scores,
            "max",
            lengths=lengths,
            axis=1,
        ).detach()
        exponentiated = torch.exp(scores - row_max[:, query_index])
        denominator = torch.segment_reduce(
            exponentiated,
            "sum",
            lengths=lengths,
            axis=1,
        )
        tiny = torch.finfo(scores.dtype).tiny
        return exponentiated / denominator[:, query_index].clamp(min=tiny)

    def _forward_sparse_hsa(
        self,
        x: torch.Tensor,
        attn_edge_index: torch.Tensor,
        edge_attn_bias: Optional[torch.Tensor],
        c_val,
    ) -> torch.Tensor:
        """Exact LD-HSA on a sparse set of allowed query-key pairs.

        This is not a kernel approximation: with the same edge support and
        dropout disabled, it is numerically equivalent to dense LD-HSA plus a
        boolean mask. Complexity is ``O(B E d)`` rather than ``O(B N^2 d)``.
        """
        if self.score_type not in (
            "busemann_log_depth",
            "busemann_log_depth_local_landmark",
            "busemann_log_depth_local_dictionary",
        ):
            raise ValueError(
                "exact sparse HSA currently supports score_type="
                "'busemann_log_depth', "
                "'busemann_log_depth_local_landmark', and "
                "'busemann_log_depth_local_dictionary' only"
            )

        B, N, _ = x.shape
        query_index, key_index, query_lengths = self._prepare_sparse_edges(
            attn_edge_index, N, x.device
        )
        E = query_index.numel()
        if edge_attn_bias is not None:
            if edge_attn_bias.device != x.device:
                raise ValueError(
                    "edge_attn_bias and attention inputs must share a device"
                )
            if not edge_attn_bias.is_floating_point():
                raise TypeError("edge_attn_bias must be floating point")
            if not bool(torch.isfinite(edge_attn_bias).all()):
                raise ValueError(
                    "edge_attn_bias must be finite; remove blocked pairs from "
                    "attn_edge_index instead of assigning -inf"
                )
            if edge_attn_bias.ndim == 2:
                edge_attn_bias = edge_attn_bias.unsqueeze(0).expand(B, -1, -1)
            elif (
                edge_attn_bias.ndim == 3
                and edge_attn_bias.shape[0] == 1
                and B != 1
            ):
                edge_attn_bias = edge_attn_bias.expand(B, -1, -1)
            expected = (B, self.num_heads, E)
            if edge_attn_bias.shape != expected:
                raise ValueError(
                    "edge_attn_bias must have shape (H, E), (1, H, E), "
                    "or (B, H, E), got "
                    f"{tuple(edge_attn_bias.shape)} instead of {expected}"
                )

        query_chunk_size = int(
            os.environ.get("BUSFORMER_SPARSE_QUERY_CHUNK", "0")
        )
        if query_chunk_size < 0:
            raise ValueError(
                "BUSFORMER_SPARSE_QUERY_CHUNK must be non-negative"
            )
        use_query_chunks = 0 < query_chunk_size < N
        edge_offsets = None
        if use_query_chunks:
            # Copy only N+1 CSR offsets to the host. Edge pairs remain on the
            # accelerator; Python uses these offsets solely to form complete
            # query segments, so a row's softmax is never split across chunks.
            edge_offsets = torch.cat(
                [
                    query_lengths.new_zeros(1),
                    query_lengths.cumsum(dim=0),
                ]
            ).cpu().tolist()

        head_outs = []
        for h in range(self.num_heads):
            q = self.W_Q[h](x, c=c_val)
            k = self.W_K[h](x, c=c_val)
            value = self.W_V[h](x, c=c_val)

            tau_h = F.softplus(self.tau_logit[h])
            gamma_h = F.softplus(self.gamma_logit[h])
            head_index = h

            def aggregate_query_range(
                q_all: torch.Tensor,
                k_all: torch.Tensor,
                value_all: torch.Tensor,
                tau: torch.Tensor,
                gamma: torch.Tensor,
                *,
                query_start: int,
                query_end: int,
                edge_start: int,
                edge_end: int,
            ) -> torch.Tensor:
                edge_queries = query_index[edge_start:edge_end]
                edge_keys = key_index[edge_start:edge_end]
                local_queries = edge_queries - query_start
                local_lengths = query_lengths[query_start:query_end]

                query = q_all[:, edge_queries]
                key = k_all[:, edge_keys]
                direction = F.normalize(query[..., 1:], dim=-1)
                inner = (direction * key[..., 1:]).sum(dim=-1)
                busemann_argument = (
                    key[..., 0] - inner
                ).clamp(min=1e-6)
                query_time = query[..., 0].clamp(min=1.0 + 1e-6)
                depth_factor = 1.0 + gamma * torch.log(query_time)
                scores = (
                    -tau
                    * depth_factor
                    * torch.log(busemann_argument)
                )
                if edge_attn_bias is not None:
                    scores = (
                        scores
                        + edge_attn_bias[
                            :, head_index, edge_start:edge_end
                        ]
                    )

                attention = self._sparse_segment_softmax(
                    scores, local_queries, local_lengths
                )
                attention = self.attn_drop(attention)
                weighted_values = (
                    attention.unsqueeze(-1) * value_all[:, edge_keys]
                )
                return torch.segment_reduce(
                    weighted_values,
                    "sum",
                    lengths=local_lengths.unsqueeze(0).expand(B, -1),
                    axis=1,
                )

            if use_query_chunks:
                aggregates = []
                for query_start in range(0, N, query_chunk_size):
                    query_end = min(query_start + query_chunk_size, N)
                    edge_start = edge_offsets[query_start]
                    edge_end = edge_offsets[query_end]

                    def chunk_fn(
                        q_all,
                        k_all,
                        value_all,
                        tau,
                        gamma,
                        _query_start=query_start,
                        _query_end=query_end,
                        _edge_start=edge_start,
                        _edge_end=edge_end,
                    ):
                        return aggregate_query_range(
                            q_all,
                            k_all,
                            value_all,
                            tau,
                            gamma,
                            query_start=_query_start,
                            query_end=_query_end,
                            edge_start=_edge_start,
                            edge_end=_edge_end,
                        )

                    if self.training and torch.is_grad_enabled():
                        aggregate = checkpoint(
                            chunk_fn,
                            q,
                            k,
                            value,
                            tau_h,
                            gamma_h,
                            use_reentrant=False,
                        )
                    else:
                        aggregate = chunk_fn(
                            q, k, value, tau_h, gamma_h
                        )
                    lorentz_norm = (
                        -Hyperboloid.minkowski_dot(
                            aggregate, aggregate, keepdim=True
                        )
                    ).clamp(min=1e-6).sqrt()
                    aggregates.append(
                        aggregate / (c_val ** 0.5 * lorentz_norm)
                    )
                head_outs.append(torch.cat(aggregates, dim=1))
            else:
                aggregate = aggregate_query_range(
                    q,
                    k,
                    value,
                    tau_h,
                    gamma_h,
                    query_start=0,
                    query_end=N,
                    edge_start=0,
                    edge_end=E,
                )
                # Keep this normalization byte-for-byte aligned with the
                # dense masked path so sparse support changes only the
                # reduction layout, not the model's numerical fallback.
                lorentz_norm = (
                    -Hyperboloid.minkowski_dot(
                        aggregate, aggregate, keepdim=True
                    )
                ).clamp(min=1e-6).sqrt()
                head_outs.append(
                    aggregate / (c_val ** 0.5 * lorentz_norm)
                )

        spatial_cat = torch.cat(
            [head_out[..., 1:] for head_out in head_outs], dim=-1
        )
        dummy = Hyperboloid.proj(
            torch.cat(
                [
                    spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1),
                    spatial_cat,
                ],
                dim=-1,
            ),
            c_val,
        )
        return self.out_proj(dummy, c=c_val)

    def forward(self,
                x:         torch.Tensor,                     # (B, N, d+1)
                attn_bias: Optional[torch.Tensor] = None,    # (B, H, N, N)
                bool_mask: Optional[torch.Tensor] = None,    # (B, N, N) True=block
                attn_edge_index: Optional[torch.Tensor] = None,  # (2,E), row=query
                edge_attn_bias: Optional[torch.Tensor] = None,   # (B,H,E)
                c=None,                                      # optional curvature override
                ) -> torch.Tensor:
        c_val = c if c is not None else self.c
        B, N, _ = x.shape
        if attn_edge_index is not None:
            if attn_bias is not None or bool_mask is not None:
                raise ValueError(
                    "sparse HSA uses attn_edge_index as the complete support; "
                    "do not also pass dense attn_bias or bool_mask"
                )
            local_output = self._forward_sparse_hsa(
                x,
                attn_edge_index=attn_edge_index,
                edge_attn_bias=edge_attn_bias,
                c_val=c_val,
            )
            if self.score_type not in (
                "busemann_log_depth_local_landmark",
                "busemann_log_depth_local_dictionary",
            ):
                return local_output
            if self.score_type == "busemann_log_depth_local_landmark":
                global_output = self._forward_linear_hsa(
                    x,
                    attn_bias=None,
                    bool_mask=None,
                    c_val=c_val,
                )
                global_scale = F.softplus(self.landmark_mix_logit)
            else:
                # Paired experiments must not let additional branch dropout
                # shift the RNG stream used by subsequent shared layers.
                # Keep legacy stochastic behavior unless explicitly enabled.
                with torch.random.fork_rng(
                    devices=[x.device] if x.is_cuda else [],
                    enabled=self.linear_boundary_preserve_rng and self.training,
                ):
                    global_output = self._forward_boundary_dictionary_hsa(
                        x,
                        attn_bias=None,
                        bool_mask=None,
                        c_val=c_val,
                    )
                global_scale = self.boundary_mix_scale()
                global_spatial = global_output[..., 1:]
                if self.linear_boundary_fusion == "centered_signed":
                    # Remove graph-wide common mode: the linear branch then
                    # contributes only query-dependent global information.
                    # This keeps the operation O(NMG) and makes a zero gate an
                    # exact local-HSA initialization.
                    global_spatial = (
                        global_spatial
                        - global_spatial.mean(dim=1, keepdim=True)
                    )
                    global_spatial = self._calibrate_boundary_residual(
                        global_spatial, local_output[..., 1:], global_scale
                    )
            if self.score_type == "busemann_log_depth_local_landmark":
                global_spatial = global_output[..., 1:]
            fused_spatial = (
                local_output[..., 1:]
                + global_scale * global_spatial
            )
            return Hyperboloid.proj(
                torch.cat(
                    [
                        fused_spatial.new_zeros(
                            *fused_spatial.shape[:-1], 1
                        ),
                        fused_spatial,
                    ],
                    dim=-1,
                ),
                c_val,
            )
        if edge_attn_bias is not None:
            raise ValueError(
                "edge_attn_bias requires attn_edge_index"
            )
        if self.score_type == "busemann_log_depth_local_dictionary":
            raise ValueError(
                "busemann_log_depth_local_dictionary requires sparse "
                "attn_edge_index support"
            )
        if self.score_type == "busemann_log_depth_linear_rf":
            return self._forward_linear_rf_hsa(
                x, attn_bias=attn_bias, bool_mask=bool_mask, c_val=c_val
            )
        if self.score_type == "busemann_log_depth_linear_dictionary":
            return self._forward_boundary_dictionary_hsa(
                x, attn_bias=attn_bias, bool_mask=bool_mask, c_val=c_val
            )
        if self.score_type == "busemann_log_depth_linear":
            return self._forward_linear_hsa(
                x, attn_bias=attn_bias, bool_mask=bool_mask, c_val=c_val
            )
        if N > 6144:
            return self._forward_chunked_queries(x, attn_bias, bool_mask, c_val)
        head_outs = []

        for h in range(self.num_heads):
            tau_dot = self.log_tau_dot[h].exp()              # scalar

            q = self.W_Q[h](x, c=c_val)    # (B, N, d_head+1)
            k = self.W_K[h](x, c=c_val)    # (B, N, d_head+1)
            v = self.W_V[h](x, c=c_val)    # (B, N, d_head+1)

            # Poincaré-ball coordinates: q_p = qₛ/q₀ ∈ open unit ball (‖·‖ < 1)
            # Derived from q₀ = sqrt(1/c + ‖qₛ‖²)  →  q₀ = 1/(√c·√(1−‖q_p‖²))
            q_p = q[..., 1:] / q[..., 0:1].clamp(min=1e-6)  # (B, N, d_head)
            k_p = k[..., 1:] / k[..., 0:1].clamp(min=1e-6)  # (B, N, d_head)

            if self.score_type == "busemann":
                # --- Hybrid Busemann score ---
                # gate: Poincaré radius of query ∈ [0,1)
                #   → 0 near origin (direction ill-defined)
                #   → 1 for deep nodes (direction stable)
                gate  = q_p.norm(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
                v_dir = F.normalize(q_p, dim=-1)             # (B, N, d_head)

                # Directional alignment ∈ (-1, 1)
                align = torch.einsum("bnd,bmd->bnm", v_dir, k_p)  # (B, N, N)

                # Term1: Poincaré dot  =  gate · τ · align
                # Term2: centered Busemann  =  gate · α · B̃(align)
                #   B̃(a) = −log(1−a)  strips the depth-offset log(x₀) from B_v(x)
                b_cen  = -torch.log((1.0 - align).clamp(min=1e-6))
                alpha_h = self.alpha[h]
                scores = gate * (tau_dot * align + alpha_h * b_cen)

            elif self.score_type == "busemann_full":
                # --- Original proposal score: −τ · B_{v_i}(k_j) ---
                # v_i = normalize(q_{i,s})  (ideal-boundary direction of query)
                # B_{v}(k) = log(k₀ − ⟨kₛ, v⟩)
                # score = −τ · B_{v_i}(k_j) = τ · log(1 / (k_{j,0} − ⟨k_{j,s}, v_i⟩))
                #
                # Key insight: k_{j,0} − ⟨k_{j,s}, v_i⟩
                #   = small  when k_j is deep AND aligned with v_i  →  high score
                #   = k_{j,0} ≈ 1/√c   when k_j is near origin     →  low score
                # Encodes both depth (k₀) and directional alignment (⟨kₛ,v⟩),
                # unlike the centered version which only captures alignment.
                v_dir    = F.normalize(q[..., 1:], dim=-1)          # (B, N, d_head)
                inner    = torch.einsum("bnd,bmd->bnm",
                                        v_dir, k[..., 1:])           # (B, N_q, N_k)
                # k₀ for each key, broadcast over queries: (B, 1, N_k)
                k0       = k[..., 0].unsqueeze(1)
                buse_arg = (k0 - inner).clamp(min=1e-6)              # (B, N_q, N_k) > 0
                scores   = -tau_dot * torch.log(buse_arg)            # (B, N_q, N_k)

            elif self.score_type == "busemann_depth_gate":
                # --- Query-depth-gated Busemann score ---
                # query_direction = normalize(q_spatial)           ∈ S^{d_h-1}
                # query_depth     = arcosh(√c · q₀)  (geodesic dist from origin)
                # query_gate      = tanh(query_depth)              ∈ [0, 1)
                # score           = gate_i · (−τ · B_{v_i}(k_j))
                #
                # Motivation: when a query node is deep (large q₀),
                # it has a well-defined tree branch → the Busemann direction
                # carries strong signal and the gate is ≈1.
                # Near-root queries (small q₀) have an ambiguous direction →
                # gate ≈ 0 suppresses the Busemann term, falling back to
                # the bias term only.  This is the asymmetry that
                # hyp_dist has over busemann_full.
                v_dir = F.normalize(q[..., 1:], dim=-1)           # (B, N, d_head)

                # Geodesic distance from origin: d(o, q) = arcosh(√c · q₀)
                # (For c=1: d = arcosh(q₀);  q₀ ≥ 1/√c always.)
                q_depth = torch.acosh(
                    (c_val ** 0.5 * q[..., 0]).clamp(min=1.0 + 1e-5)
                )                                                   # (B, N)
                gate = torch.tanh(q_depth).unsqueeze(-1)           # (B, N, 1)

                inner    = torch.einsum("bnd,bmd->bnm", v_dir, k[..., 1:])
                k0       = k[..., 0].unsqueeze(1)                  # (B, 1, N)
                buse_arg = (k0 - inner).clamp(min=1e-6)            # (B, N, N)
                scores   = gate * (-tau_dot * torch.log(buse_arg)) # (B, N, N)

            elif self.score_type == "busemann_sym":
                # --- Symmetric Busemann score ---
                # s_ij = -τ/2 · (B_{v_i}(k_j) + B_{v_j}(q_i))
                #
                # B_{v_i}(k_j) = log(k_{j,0} − ⟨k_{j,s}, v_i⟩)
                #   → key j's horospherical depth in query i's direction
                # B_{v_j}(q_i) = log(q_{i,0} − ⟨q_{i,s}, v_j⟩)
                #   → query i's horospherical depth in key j's direction
                #
                # Together the log-sum encodes both q-depth and k-depth,
                # mirroring how hyp_dist encodes both via
                #   log(1−‖q_p‖²) and log(1−‖k_p‖²) in the denominator.
                v_q = F.normalize(q[..., 1:], dim=-1)   # (B, N, d_head)
                v_k = F.normalize(k[..., 1:], dim=-1)   # (B, N, d_head)

                # B_{v_i}(k_j): key depth from query direction
                inner_qk = torch.einsum("bnd,bmd->bnm", v_q, k[..., 1:])
                k0       = k[..., 0].unsqueeze(1)                      # (B, 1, N_k)
                buse_kj  = torch.log((k0 - inner_qk).clamp(min=1e-6))  # (B, N_q, N_k)

                # B_{v_j}(q_i): query depth from key direction
                inner_kq = torch.einsum("bmd,bnd->bnm", v_k, q[..., 1:])
                q0       = q[..., 0].unsqueeze(2)                      # (B, N_q, 1)
                buse_qi  = torch.log((q0 - inner_kq).clamp(min=1e-6))  # (B, N_q, N_k)

                scores   = -tau_dot * (buse_kj + buse_qi) * 0.5        # (B, N_q, N_k)

            elif self.score_type == "lorentz_dot":
                # --- Lorentz spatial dot product ---
                scores = tau_dot * torch.einsum(
                    "bnd,bmd->bnm", q[..., 1:], k[..., 1:])

            elif self.score_type == "hypformer_kernel":
                # Hypformer focused linear-attention compatibility.  We
                # materialise the positive kernel and take its logarithm so
                # softmax produces kernel-normalized weights under the exact
                # same mask and Lorentz value aggregation as every other row.
                phi_q = self._hypformer_feature_map(q[..., 1:])
                phi_k = self._hypformer_feature_map(k[..., 1:])
                kernel = torch.einsum("bnd,bmd->bnm", phi_q, phi_k)
                scores = torch.log(kernel.clamp(min=1e-12))

            elif self.score_type == "busemann_log_depth":
                # --- HSA with multiplicative log-depth factor (v5) ---
                # s_ij = −τ_h · (1 + γ_h · log q_{i,0}) · B_{v_i}(k_j)
                # At unit curvature, the origin has q₀=1 and the factor is 1.
                # With learned curvature, clamped log q₀ is a monotone radial
                # proxy; larger values amplify the score without changing key order.
                tau_h   = F.softplus(self.tau_logit[h])
                gamma_h = F.softplus(self.gamma_logit[h])
                v_dir   = F.normalize(q[..., 1:], dim=-1)       # (B, N, d_head)
                inner   = torch.einsum("bnd,bmd->bnm",
                                       v_dir, k[..., 1:])        # (B, N_q, N_k)
                k0      = k[..., 0].unsqueeze(1)                 # (B, 1, N_k)
                buse_arg = (k0 - inner).clamp(min=1e-6)
                B_v     = torch.log(buse_arg)                    # (B, N_q, N_k)
                q0      = q[..., 0:1].clamp(min=1.0 + 1e-6)     # (B, N_q, 1)
                depth_fac = 1.0 + gamma_h * torch.log(q0)       # (B, N_q, 1)
                scores  = -tau_h * depth_fac * B_v               # (B, N_q, N_k)

            elif self.score_type == "busemann_log_depth_sym":
                # Bidirectional counterpart of the LD score. Query and key
                # projections need not be tied, so "bidirectional" is more
                # precise than claiming the resulting matrix is symmetric.
                tau_h   = F.softplus(self.tau_logit[h])
                gamma_h = F.softplus(self.gamma_logit[h])
                v_q = F.normalize(q[..., 1:], dim=-1)
                v_k = F.normalize(k[..., 1:], dim=-1)

                inner_qk = torch.einsum("bnd,bmd->bnm", v_q, k[..., 1:])
                k0 = k[..., 0].unsqueeze(1)
                buse_qk = torch.log((k0 - inner_qk).clamp(min=1e-6))
                q_depth_fac = 1.0 + gamma_h * torch.log(
                    q[..., 0:1].clamp(min=1.0 + 1e-6)
                )
                directed_qk = -tau_h * q_depth_fac * buse_qk

                inner_kq = torch.einsum("bmd,bnd->bnm", v_k, q[..., 1:])
                q0 = q[..., 0].unsqueeze(2)
                buse_kq = torch.log((q0 - inner_kq).clamp(min=1e-6))
                k_depth_fac = 1.0 + gamma_h * torch.log(
                    k[..., 0].unsqueeze(1).clamp(min=1.0 + 1e-6)
                )
                directed_kq = -tau_h * k_depth_fac * buse_kq
                scores = 0.5 * (directed_qk + directed_kq)

            elif self.score_type == "busemann_add_depth":
                # --- HSA with additive query-depth bias (v6) ---
                # s_ij = −τ · B_{v_i}(k_j) + β_h · log q_{i,0}
                # β init=0: identical to busemann_full at start; β adapts freely.
                # Additive (not multiplicative): log q_0 enters linearly as a bias,
                # without cross-coupling to the directional B_v term.
                v_dir   = F.normalize(q[..., 1:], dim=-1)
                inner   = torch.einsum("bnd,bmd->bnm",
                                       v_dir, k[..., 1:])
                k0      = k[..., 0].unsqueeze(1)
                buse_arg = (k0 - inner).clamp(min=1e-6)
                q0      = q[..., 0:1].clamp(min=1.0 + 1e-6)
                scores  = (-tau_dot * torch.log(buse_arg)
                           + self.beta[h] * torch.log(q0))       # (B, N_q, N_k)

            else:  # "hyp_dist"
                # --- Negative squared hyperbolic distance (ablation baseline) ---
                # Exact formula for d(q,k) from Poincaré-ball coords:
                #   cosh(d·√c) = (1−⟨q_p,k_p⟩) / (√(1−‖q_p‖²)·√(1−‖k_p‖²))
                # Gradient-stable for ‖q_p‖,‖k_p‖ < 1.
                q_sq   = (q_p * q_p).sum(-1)                  # (B, N)
                k_sq   = (k_p * k_p).sum(-1)                  # (B, N)
                qk     = torch.einsum("bnd,bmd->bnm", q_p, k_p)  # (B, N, N)
                num    = (1.0 - qk)                            # (B, N, N)
                den    = ((1.0 - q_sq).unsqueeze(2) *
                          (1.0 - k_sq).unsqueeze(1)
                          ).clamp(min=1e-6).sqrt()             # (B, N, N)
                cosh_d = (num / den).clamp(min=1.0 + 1e-5)    # (B, N, N)
                dist_sq = torch.acosh(cosh_d) ** 2             # ∝ d(q,k)² · c
                scores  = -tau_dot * dist_sq                   # closer → higher

            # --- Graphormer pattern: add structural bias before softmax ---
            if attn_bias is not None:
                scores = scores + attn_bias[:, h]            # (B, N, N)

            # Neighbourhood mask (True = block → fill with −∞)
            if bool_mask is not None:
                scores = scores.masked_fill(bool_mask, float("-inf"))

            attn = F.softmax(scores, dim=-1)                 # (B, N, N)
            attn = self.attn_drop(attn)

            # Lorentz centroid (FHNN eq. 4): µ = Σaᵢⱼvⱼ / (√c · ‖Σaᵢⱼvⱼ‖_L)
            # Efficient proxy for the intrinsic Fréchet mean. Unlike
            # proj(weighted_sum), this normalises the full ambient vector
            # (preserving relative x₀ weight); proj only recomputes x₀ from xₛ.
            agg = torch.einsum("bqk,bkd->bqd", attn, v)     # (B, N, d_head+1)
            lor_norm = (
                -Hyperboloid.minkowski_dot(agg, agg, keepdim=True)
            ).clamp(min=1e-6).sqrt()                          # (B, N, 1)
            agg = agg / (c_val ** 0.5 * lor_norm)             # project to H^d_head
            head_outs.append(agg)

        # Concatenate spatial parts; reproject to get a valid Lorentz point,
        # then out_proj back to d_model.
        # NOTE: head_outs[0][..., :1] is the time coord of a d_head-dim point,
        # not d_model-dim, so it must NOT be reused — recompute x₀ via proj.
        spatial_cat = torch.cat(
            [h_out[..., 1:] for h_out in head_outs], dim=-1)   # (B, N, d_model)
        dummy = Hyperboloid.proj(
            torch.cat([spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1),
                       spatial_cat], dim=-1),
            c_val,
                )                                                        # (B, N, d_model+1)
        return self.out_proj(dummy, c=c_val)                     # (B, N, d_model+1)

    def _forward_chunked_queries(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        bool_mask: Optional[torch.Tensor],
        c_val,
        chunk: int = 256,
    ) -> torch.Tensor:
        """Memory-friendly path for very large N: never materialise full (N×N) score
        matrix; process query rows in chunks. Used automatically when N > 6144."""
        B, N, _ = x.shape
        head_out_parts_per_head = []

        for h in range(self.num_heads):
            tau_dot = self.log_tau_dot[h].exp()

            q = self.W_Q[h](x, c=c_val)
            k = self.W_K[h](x, c=c_val)
            v = self.W_V[h](x, c=c_val)
            q_p = q[..., 1:] / q[..., 0:1].clamp(min=1e-6)
            k_p = k[..., 1:] / k[..., 0:1].clamp(min=1e-6)

            q_sq = (q_p * q_p).sum(-1) if self.score_type == "hyp_dist" else None
            k_sq = (k_p * k_p).sum(-1) if self.score_type == "hyp_dist" else None

            rows_out = []
            for s in range(0, N, chunk):
                e = min(s + chunk, N)

                if self.score_type == "busemann":
                    gate = q_p[:, s:e].norm(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
                    v_dir = F.normalize(q_p[:, s:e], dim=-1)
                    align = torch.einsum("bnd,bmd->bnm", v_dir, k_p)
                    b_cen = -torch.log((1.0 - align).clamp(min=1e-6))
                    alpha_h = self.alpha[h]
                    scores = gate * (tau_dot * align + alpha_h * b_cen)
                elif self.score_type == "busemann_full":
                    v_dir = F.normalize(q[:, s:e, 1:], dim=-1)
                    inner = torch.einsum("bnd,bmd->bnm", v_dir, k[..., 1:])
                    k0 = k[..., 0].unsqueeze(1)
                    buse_arg = (k0 - inner).clamp(min=1e-6)
                    scores = -tau_dot * torch.log(buse_arg)
                elif self.score_type == "busemann_depth_gate":
                    v_dir = F.normalize(q[:, s:e, 1:], dim=-1)
                    q_depth = torch.acosh(
                        (c_val ** 0.5 * q[:, s:e, 0]).clamp(min=1.0 + 1e-5)
                    )
                    gate = torch.tanh(q_depth).unsqueeze(-1)
                    inner = torch.einsum("bnd,bmd->bnm", v_dir, k[..., 1:])
                    k0 = k[..., 0].unsqueeze(1)
                    buse_arg = (k0 - inner).clamp(min=1e-6)
                    scores = gate * (-tau_dot * torch.log(buse_arg))
                elif self.score_type == "busemann_sym":
                    v_q = F.normalize(q[:, s:e, 1:], dim=-1)
                    v_k = F.normalize(k[..., 1:], dim=-1)
                    inner_qk = torch.einsum("bnd,bmd->bnm", v_q, k[..., 1:])
                    k0 = k[..., 0].unsqueeze(1)
                    buse_kj = torch.log((k0 - inner_qk).clamp(min=1e-6))
                    inner_kq = torch.einsum("bmd,bnd->bmn", v_k, q[:, s:e, 1:])
                    q0_chunk = q[:, s:e, 0:1]
                    buse_qi = torch.log(
                        (q0_chunk - inner_kq.transpose(1, 2)).clamp(min=1e-6)
                    )
                    scores = -tau_dot * (buse_kj + buse_qi) * 0.5
                elif self.score_type == "lorentz_dot":
                    q_s = q[:, s:e, 1:]
                    scores = tau_dot * torch.einsum(
                        "bnd,bmd->bnm", q_s, k[..., 1:])
                elif self.score_type == "hypformer_kernel":
                    phi_q = self._hypformer_feature_map(q[:, s:e, 1:])
                    phi_k = self._hypformer_feature_map(k[..., 1:])
                    kernel = torch.einsum("bnd,bmd->bnm", phi_q, phi_k)
                    scores = torch.log(kernel.clamp(min=1e-12))
                elif self.score_type == "busemann_log_depth":
                    tau_h   = F.softplus(self.tau_logit[h])
                    gamma_h = F.softplus(self.gamma_logit[h])
                    v_dir   = F.normalize(q[:, s:e, 1:], dim=-1)
                    inner   = torch.einsum("bnd,bmd->bnm",
                                           v_dir, k[..., 1:])
                    k0      = k[..., 0].unsqueeze(1)
                    buse_arg = (k0 - inner).clamp(min=1e-6)
                    B_v     = torch.log(buse_arg)
                    q0      = q[:, s:e, 0:1].clamp(min=1.0 + 1e-6)
                    depth_fac = 1.0 + gamma_h * torch.log(q0)
                    scores  = -tau_h * depth_fac * B_v
                elif self.score_type == "busemann_log_depth_sym":
                    tau_h   = F.softplus(self.tau_logit[h])
                    gamma_h = F.softplus(self.gamma_logit[h])
                    v_q = F.normalize(q[:, s:e, 1:], dim=-1)
                    v_k = F.normalize(k[..., 1:], dim=-1)

                    inner_qk = torch.einsum(
                        "bnd,bmd->bnm", v_q, k[..., 1:]
                    )
                    k0 = k[..., 0].unsqueeze(1)
                    buse_qk = torch.log(
                        (k0 - inner_qk).clamp(min=1e-6)
                    )
                    q_depth_fac = 1.0 + gamma_h * torch.log(
                        q[:, s:e, 0:1].clamp(min=1.0 + 1e-6)
                    )
                    directed_qk = -tau_h * q_depth_fac * buse_qk

                    inner_kq = torch.einsum(
                        "bmd,bnd->bmn", v_k, q[:, s:e, 1:]
                    )
                    q0 = q[:, s:e, 0:1]
                    buse_kq = torch.log(
                        (q0 - inner_kq.transpose(1, 2)).clamp(min=1e-6)
                    )
                    k_depth_fac = 1.0 + gamma_h * torch.log(
                        k[..., 0].unsqueeze(1).clamp(min=1.0 + 1e-6)
                    )
                    directed_kq = -tau_h * k_depth_fac * buse_kq
                    scores = 0.5 * (directed_qk + directed_kq)
                elif self.score_type == "busemann_add_depth":
                    v_dir   = F.normalize(q[:, s:e, 1:], dim=-1)
                    inner   = torch.einsum("bnd,bmd->bnm",
                                           v_dir, k[..., 1:])
                    k0      = k[..., 0].unsqueeze(1)
                    buse_arg = (k0 - inner).clamp(min=1e-6)
                    q0      = q[:, s:e, 0:1].clamp(min=1.0 + 1e-6)
                    scores  = (-tau_dot * torch.log(buse_arg)
                               + self.beta[h] * torch.log(q0))
                else:  # hyp_dist
                    qp_sub = q_p[:, s:e, :]
                    qk = torch.einsum("bnd,bmd->bnm", qp_sub, k_p)
                    num = 1.0 - qk
                    den = (
                        (1.0 - q_sq[:, s:e]).unsqueeze(2)
                        * (1.0 - k_sq).unsqueeze(1)
                    ).clamp(min=1e-6).sqrt()
                    cosh_d = (num / den).clamp(min=1.0 + 1e-5)
                    dist_sq = torch.acosh(cosh_d) ** 2
                    scores = -tau_dot * dist_sq

                if attn_bias is not None:
                    scores = scores + attn_bias[:, h, s:e, :]
                if bool_mask is not None:
                    scores = scores.masked_fill(bool_mask[:, s:e, :], float("-inf"))

                attn = F.softmax(scores, dim=-1)
                attn = self.attn_drop(attn)
                agg = torch.einsum("bqk,bkd->bqd", attn, v)
                lor_norm = (
                    -Hyperboloid.minkowski_dot(agg, agg, keepdim=True)
                ).clamp(min=1e-6).sqrt()
                agg = agg / (c_val ** 0.5 * lor_norm)
                rows_out.append(agg)

            head_out = torch.cat(rows_out, dim=1)
            head_out_parts_per_head.append(head_out)

        spatial_cat = torch.cat([hout[..., 1:] for hout in head_out_parts_per_head], dim=-1)
        dummy = Hyperboloid.proj(
            torch.cat(
                [spatial_cat.new_zeros(*spatial_cat.shape[:-1], 1), spatial_cat],
                dim=-1,
            ),
            c_val,
        )
        return self.out_proj(dummy, c=c_val)

#
#     Graphormer uses pre-LayerNorm (norm_first=True in PyTorch):
#         x = x + Attn( LN(x) , attn_bias )
#         x = x + FFN ( LN(x) )
#     Here "+" is the ambient additive residual: proj(x + sublayer(x)).
# ============================================================================

class BusTransformerBlock(nn.Module):
    """Pre-LN Busemann Transformer block with per-layer learnable curvature
    and optional attention / Lorentz GNN branches.

    Sub-layers:
        x_norm = HypNorm₁(x)
        x ← proj( x [+ BusemannMHA(x_norm, attn_bias)] [+ LorentzGNN(x_norm, adj)] )
        x ← proj( x + FFN( HypNorm₂(x) ) )

    Each block holds its own log_c parameter (initialised to log(c_init)),
    which is exponentiated at forward time and threaded through all sub-layers.
    This mirrors Hypformer's per-layer learnable curvatures.

    If adj_weights is not supplied to forward(), the GNN branch is skipped
    (backward-compatible with pure-transformer usage).
    """

    def __init__(self, d_model: int, num_heads: int,
                 c: float = 1.0,
                 ffn_hidden: Optional[int] = None,
                 dropout: float = 0.0,
                 score_type: str = "busemann",
                 attn_proj: str = "htc_qk",
                 linear_landmarks: int = 64,
                 linear_rf_features: int = 128,
                 linear_rf_seed: int = 0,
                 linear_rf_t_max: float = 12.0,
                 linear_rf_radial_scale: Optional[float] = None,
                 linear_boundary_directions: int = 64,
                 linear_boundary_seed: int = 0,
                 linear_boundary_gate_init: Optional[float] = None,
                 linear_boundary_mix_init: float = 0.1,
                 linear_boundary_scales: int = 1,
                 linear_boundary_scale_spread: float = 0.35,
                 linear_boundary_fusion: str = "additive",
                 linear_boundary_preserve_rng: bool = False,
                 learn_curvature: bool = True,
                 use_attention: bool = True,
                 use_gnn_branch: bool = True,
                 ld_tau_init: Optional[float] = None,
                 ld_gamma_init: Optional[float] = None,
                 residual_gate_mode: str = "none",
                 attn_gate_init: float = 1.0,
                 gnn_gate_init: float = 1.0,
                 ffn_gate_init: float = 1.0,
                 linear_boundary_residual_norm: str = "none",
                 linear_boundary_residual_max_gain: float = 100.0,
                 linear_boundary_residual_eps: float = 1e-6,
                 linear_boundary_checkpoint_heads: bool = False,
                 linear_boundary_key_normalization: str = "none",
                 linear_boundary_fixed_exponent: Optional[float] = None,
                 linear_boundary_scale_routing: str = "adaptive"):
        super().__init__()
        _validate_boundary_scale_routing(linear_boundary_scale_routing)
        if residual_gate_mode not in ("none", "sigmoid"):
            raise ValueError(
                "residual_gate_mode must be 'none' or 'sigmoid', "
                f"got {residual_gate_mode!r}"
            )
        self.c   = c          # kept for backward compat / checkpointing
        self.use_attention = use_attention
        self.use_gnn_branch = use_gnn_branch
        self.residual_gate_mode = residual_gate_mode
        ffn_dim  = ffn_hidden or 4 * d_model

        # Learnable per-layer curvature (log-parameterised → always > 0)
        self.log_c_param = nn.Parameter(
            torch.tensor(math.log(c)),
            requires_grad=learn_curvature,
        )

        # Pre-LayerNorm (Graphormer / norm_first pattern)
        self.norm1 = HypLayerNorm(d_model, c=c)
        self.norm2 = HypLayerNorm(d_model, c=c)

        # Construct in a fixed order even when a branch is disabled so common
        # modules receive identical initial weights under paired seeds.
        attn = BusemannMHA(
            d_model,
            num_heads,
            c=c,
            dropout=dropout,
            score_type=score_type,
            attn_proj=attn_proj,
            linear_landmarks=linear_landmarks,
            linear_rf_features=linear_rf_features,
            linear_rf_seed=linear_rf_seed,
            linear_rf_t_max=linear_rf_t_max,
            linear_rf_radial_scale=linear_rf_radial_scale,
            linear_boundary_directions=linear_boundary_directions,
            linear_boundary_seed=linear_boundary_seed,
            linear_boundary_gate_init=linear_boundary_gate_init,
            linear_boundary_mix_init=linear_boundary_mix_init,
            linear_boundary_scales=linear_boundary_scales,
            linear_boundary_scale_spread=linear_boundary_scale_spread,
            linear_boundary_fusion=linear_boundary_fusion,
            linear_boundary_preserve_rng=linear_boundary_preserve_rng,
            linear_boundary_residual_norm=linear_boundary_residual_norm,
            linear_boundary_residual_max_gain=linear_boundary_residual_max_gain,
            linear_boundary_residual_eps=linear_boundary_residual_eps,
            linear_boundary_checkpoint_heads=linear_boundary_checkpoint_heads,
            linear_boundary_key_normalization=linear_boundary_key_normalization,
            linear_boundary_fixed_exponent=linear_boundary_fixed_exponent,
            linear_boundary_scale_routing=linear_boundary_scale_routing,
            ld_tau_init=ld_tau_init,
            ld_gamma_init=ld_gamma_init,
        )
        self.attn = attn if use_attention else None

        # Lorentz GNN branch (skipped when adj_weights is None)
        gnn = LorentzGNN(d_model, c=c, dropout=dropout)
        self.gnn = gnn if use_gnn_branch else None

        # FFN: HypLinear → HypActivation → HypLinear  (Hypformer pattern)
        self.ffn_l1  = HypLinear(d_model, ffn_dim,  c=c, dropout=dropout)
        self.ffn_act = HypActivation(act=F.gelu, c=c)
        self.ffn_l2  = HypLinear(ffn_dim,  d_model, c=c, dropout=dropout)

        self.drop    = nn.Dropout(dropout)
        if residual_gate_mode == "sigmoid":
            self.attn_gate_logit = nn.Parameter(torch.tensor(
                self._actual_to_sigmoid_logit(
                    attn_gate_init, "attn_gate_init"
                )
            ))
            self.gnn_gate_logit = nn.Parameter(torch.tensor(
                self._actual_to_sigmoid_logit(
                    gnn_gate_init, "gnn_gate_init"
                )
            ))
            self.ffn_gate_logit = nn.Parameter(torch.tensor(
                self._actual_to_sigmoid_logit(
                    ffn_gate_init, "ffn_gate_init"
                )
            ))

    def curvature(self) -> torch.Tensor:
        return self.log_c_param.exp()

    @staticmethod
    def _actual_to_sigmoid_logit(value: float, name: str) -> float:
        """Map an actual residual scale to its learnable sigmoid logit.

        A requested value of one is represented by a finite value just inside
        the open sigmoid range. This makes the documented identity-like
        default usable while retaining a nonzero gradient.
        """
        value = float(value)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be in (0, 1], got {value}"
            )
        if value == 1.0:
            value = 1.0 - 1e-6
        return math.log(value) - math.log1p(-value)

    def _gate_residual_output(
        self,
        y: torch.Tensor,
        gate_name: str,
        c,
    ) -> torch.Tensor:
        """Scale a sublayer output without leaving the hyperboloid.

        Ambient scalar multiplication does not preserve the Lorentz constraint.
        We therefore scale only the spatial coordinates and reconstruct the
        time coordinate before the ambient residual addition.
        """
        if self.residual_gate_mode == "none":
            return y
        gate = torch.sigmoid(getattr(self, f"{gate_name}_gate_logit"))
        spatial = gate * y[..., 1:]
        return Hyperboloid.proj(
            torch.cat(
                [spatial.new_zeros(*spatial.shape[:-1], 1), spatial],
                dim=-1,
            ),
            c,
        )

    def _res(self, x: torch.Tensor, y: torch.Tensor, c) -> torch.Tensor:
        """Ambient additive residual: add Lorentz vectors, reproject to H^d_c.
        Equivalent to xₛ += yₛ then recompute x₀ = sqrt(1/c + ‖xₛ‖²).
        Gives identity gradient through the skip path (no λ attenuation).
        c can be a Python float or a scalar tensor.
        """
        return Hyperboloid.proj(x + y, c)

    def forward(self,
                x:           torch.Tensor,
                attn_bias:   Optional[torch.Tensor] = None,
                bool_mask:   Optional[torch.Tensor] = None,
                attn_edge_index: Optional[torch.Tensor] = None,
                edge_attn_bias: Optional[torch.Tensor] = None,
                adj_weights: Optional[torch.Tensor] = None,  # (B,N,N) row-norm
                ) -> torch.Tensor:
        # Dynamic curvature for this block (scalar tensor, gradient flows through)
        c_dyn = self.curvature()

        # Pre-norm shared by attention and GNN
        x_norm = self.norm1(x, c=c_dyn)

        # Attention sublayer (disabled for backbone-only / GNN-only controls).
        if self.attn is not None:
            attn_out = self.attn(
                x_norm,
                attn_bias=attn_bias,
                bool_mask=bool_mask,
                attn_edge_index=attn_edge_index,
                edge_attn_bias=edge_attn_bias,
                c=c_dyn,
            )
            attn_out = self._gate_residual_output(
                self.drop(attn_out), "attn", c_dyn
            )
            x = self._res(x, attn_out, c_dyn)

        # GNN sublayer (parallel with attention, independent residual)
        if self.gnn is not None and adj_weights is not None:
            gnn_out = self.gnn(x_norm, adj_weights, c=c_dyn)
            gnn_out = self._gate_residual_output(
                self.drop(gnn_out), "gnn", c_dyn
            )
            x = self._res(x, gnn_out, c_dyn)

        # FFN sublayer
        x_norm2  = self.norm2(x, c=c_dyn)
        h = self.ffn_l1(x_norm2, c=c_dyn)
        h = self.ffn_act(h,       c=c_dyn)
        h = self.ffn_l2(h,        c=c_dyn)
        h = self._gate_residual_output(self.drop(h), "ffn", c_dyn)
        x = self._res(x, h, c_dyn)
        return x


# ============================================================================
# 8.  BMLR-style Busemann-logit head (legacy signed-scale parameterization)
#     P(y=c|x) ∝ exp(−αc · B_{vc}(x) + bc)
# ============================================================================

class BMLR(nn.Module):
    """Busemann-logit classifier used by the reported experiments.

    Unlike HBNN's exact BMLR, ``alpha`` is optimized directly rather than
    exponentiated, so it is not constrained to remain positive. The class name
    is retained for compatibility with existing scripts and pinned runs.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(num_classes, d_model))
        self.alpha = nn.Parameter(torch.ones(num_classes))
        self.bias  = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., d+1)  →  logits : (..., num_classes)."""
        v      = F.normalize(self.v_raw, dim=-1)          # (C, d)
        inner  = x[..., 1:] @ v.T                         # (..., C)
        arg    = (x[..., :1] - inner).clamp(min=1e-6)     # (..., C)
        buse   = torch.log(arg)
        return -self.alpha * buse + self.bias


# ============================================================================
# 9.  HypCLS  (Hypformer-style distance-based classifier)
#     logit_c(x) = −d²(x, wc)  where wc ∈ H^d_c are learnable prototypes
# ============================================================================

class HypCLS(nn.Module):
    """Distance-based classifier with manifold class prototypes (Hypformer)."""

    def __init__(self, d_model: int, num_classes: int, c: float = 1.0):
        super().__init__()
        self.c       = c
        # Spatial parts of prototype points (projected to manifold in forward)
        self.proto_s = nn.Parameter(0.1 * torch.randn(num_classes, d_model))

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        """x : (..., d+1)  →  logits : (..., num_classes)."""
        c_val = c if c is not None else self.c
        # Project prototypes to manifold
        proto = Hyperboloid.proj(
            torch.cat([self.proto_s.new_zeros(self.proto_s.shape[0], 1),
                       self.proto_s], dim=-1),
            c_val,
        )                                                  # (C, d+1)

        # Pairwise squared distances via Minkowski inner product
        # prod[..., c] = ⟨x, proto_c⟩_L
        x0_p0 = (-x[..., :1]) * proto[:, 0]               # (..., C)
        xs_ps = x[..., 1:] @ proto[:, 1:].T               # (..., C)
        prod  = x0_p0 + xs_ps                              # (..., C)  = minkowski_dot

        arg      = (-c_val * prod).clamp(min=1.0 + 1e-5)
        sqdist   = (2.0 / c_val) * torch.acosh(arg) ** 2  # (..., C)
        return -sqdist


# ============================================================================
# 10.  BusTransformer  (full sequence model with Graphormer graph token)
# ============================================================================

class BusTransformer(nn.Module):
    """Busemann Transformer for sequence classification.

    Graphormer additions:
      • A virtual graph token is prepended at position 0
        – its representation is used for classification (CLS-style)
      • GraphAttnBias structural encoding injected into every layer

    Hypformer additions:
      • HypLayerNorm (final norm before head)
      • Choice of BMLR or HypCLS head

    HGCN additions:
      • Embeddings lifted to manifold via Hyperboloid.proj
      • All internal layers use HypLinear / BusemannMHA
    """

    def __init__(
        self,
        vocab_size:        int,
        d_model:           int,
        num_heads:         int,
        num_layers:        int,
        num_classes:       int,
        max_seq_len:       int   = 512,
        c:                 float = 1.0,
        ffn_hidden:        Optional[int]  = None,
        dropout:           float = 0.1,
        use_graph_token:   bool  = True,
        use_attn_bias:     bool  = True,
        max_spd:           int   = 20,
        head_type:         str   = "bmlr",   # "bmlr" | "hypcls"
        attn_proj:         str   = "htc_qk",
        score_type:        str   = "busemann",
        linear_landmarks:  int   = 64,
        linear_rf_features: int  = 128,
        linear_rf_seed:     int  = 0,
        linear_rf_t_max:    float = 12.0,
        linear_rf_radial_scale: Optional[float] = None,
        linear_boundary_directions: int = 64,
        linear_boundary_seed: int = 0,
        linear_boundary_gate_init: Optional[float] = None,
        linear_boundary_mix_init: float = 0.1,
        linear_boundary_scales: int = 1,
        linear_boundary_scale_spread: float = 0.35,
        linear_boundary_fusion: str = "additive",
        linear_boundary_preserve_rng: bool = False,
        linear_boundary_residual_norm: str = "none",
        linear_boundary_residual_max_gain: float = 100.0,
        linear_boundary_residual_eps: float = 1e-6,
        linear_boundary_checkpoint_heads: bool = False,
        linear_boundary_key_normalization: str = "none",
        linear_boundary_fixed_exponent: Optional[float] = None,
        linear_boundary_scale_routing: str = "adaptive",
    ):
        super().__init__()
        _validate_boundary_scale_routing(linear_boundary_scale_routing)
        self.c               = c
        self.use_graph_token = use_graph_token
        self.use_attn_bias   = use_attn_bias

        # Euclidean embeddings, lifted to manifold after summing
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len + 1, d_model)  # +1 for graph token

        if use_graph_token:
            # Learnable Euclidean vector; lifted in forward (Graphormer §3)
            self.graph_token_emb = nn.Parameter(0.02 * torch.randn(1, 1, d_model))

        if use_attn_bias:
            self.attn_bias_mod = GraphAttnBias(num_heads, max_spd=max_spd)

        self.blocks = nn.ModuleList([
            BusTransformerBlock(
                d_model,
                num_heads,
                c=c,
                ffn_hidden=ffn_hidden,
                dropout=dropout,
                score_type=score_type,
                attn_proj=attn_proj,
                linear_landmarks=linear_landmarks,
                linear_rf_features=linear_rf_features,
                linear_rf_seed=linear_rf_seed + layer_index,
                linear_rf_t_max=linear_rf_t_max,
                linear_rf_radial_scale=linear_rf_radial_scale,
                linear_boundary_directions=linear_boundary_directions,
                linear_boundary_seed=linear_boundary_seed + layer_index,
                linear_boundary_gate_init=linear_boundary_gate_init,
                linear_boundary_mix_init=linear_boundary_mix_init,
                linear_boundary_scales=linear_boundary_scales,
                linear_boundary_scale_spread=linear_boundary_scale_spread,
                linear_boundary_fusion=linear_boundary_fusion,
                linear_boundary_preserve_rng=linear_boundary_preserve_rng,
                linear_boundary_residual_norm=linear_boundary_residual_norm,
                linear_boundary_residual_max_gain=linear_boundary_residual_max_gain,
                linear_boundary_residual_eps=linear_boundary_residual_eps,
                linear_boundary_checkpoint_heads=linear_boundary_checkpoint_heads,
                linear_boundary_key_normalization=linear_boundary_key_normalization,
                linear_boundary_fixed_exponent=linear_boundary_fixed_exponent,
                linear_boundary_scale_routing=linear_boundary_scale_routing,
            )
            for layer_index in range(num_layers)
        ])

        self.final_norm = HypLayerNorm(d_model, c=c)
        self.drop       = nn.Dropout(dropout)

        if head_type == "bmlr":
            self.head = BMLR(d_model, num_classes)
        elif head_type == "hypcls":
            self.head = HypCLS(d_model, num_classes, c=c)
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

    def _lift(self, e: torch.Tensor) -> torch.Tensor:
        """Lift Euclidean d-dim vector to H^d_c."""
        return Hyperboloid.proj(
            torch.cat([e.new_zeros(*e.shape[:-1], 1), e], dim=-1),
            self.c,
        )

    def forward(
        self,
        input_ids:  torch.Tensor,                    # (B, N)
        bool_mask:  Optional[torch.Tensor] = None,   # (B, N, N) bool True=block
        spd_matrix: Optional[torch.Tensor] = None,   # (B, N, N) int64
        adj_matrix: Optional[torch.Tensor] = None,   # (B, N, N) bool
    ) -> torch.Tensor:
        B, N = input_ids.shape
        dev  = input_ids.device

        pos = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)
        e   = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))  # (B,N,d)
        x   = self._lift(e)                                               # (B,N,d+1)

        # Prepend graph token (Graphormer)
        if self.use_graph_token:
            gt  = self.graph_token_emb.expand(B, -1, -1)    # (B,1,d)
            gt  = self._lift(gt)                              # (B,1,d+1)
            x   = torch.cat([gt, x], dim=1)                  # (B,N+1,d+1)
            N_eff = N + 1
        else:
            N_eff = N

        # Build structural attention bias
        attn_bias = None
        if self.use_attn_bias and spd_matrix is not None:
            if self.use_graph_token:
                # Pad for the graph-token row/column with zeros
                s_pad = torch.zeros(B, N_eff, N_eff, dtype=torch.long,  device=dev)
                a_pad = torch.zeros(B, N_eff, N_eff, dtype=torch.bool,  device=dev)
                s_pad[:, 1:, 1:] = spd_matrix
                a_pad[:, 1:, 1:] = adj_matrix if adj_matrix is not None else False
            else:
                s_pad = spd_matrix
                a_pad = adj_matrix if adj_matrix is not None else \
                        torch.zeros_like(spd_matrix, dtype=torch.bool)
            attn_bias = self.attn_bias_mod(s_pad, a_pad)    # (B, H, N_eff, N_eff)

        # Pad bool_mask for graph token
        if bool_mask is not None and self.use_graph_token:
            top  = torch.zeros(B, 1,  N_eff, dtype=torch.bool, device=dev)
            left = torch.zeros(B, N,  1,     dtype=torch.bool, device=dev)
            bool_mask = torch.cat(
                [top, torch.cat([left, bool_mask], dim=2)], dim=1)       # (B,N+1,N+1)

        for block in self.blocks:
            x = block(x, attn_bias=attn_bias, bool_mask=bool_mask)

        c_final = self.blocks[-1].curvature() if len(self.blocks) else self.c
        x = self.final_norm(x, c=c_final)

        # Classification: graph token (position 0) or mean pool
        if self.use_graph_token:
            x_cls = x[:, 0]                     # (B, d+1)
        else:
            x_cls = Hyperboloid.proj(
                torch.cat([x[..., :1].mean(1), x[..., 1:].mean(1)], dim=-1),
                c_final,
            )

        return self.head(x_cls)                 # (B, num_classes)


# ============================================================================
# 11.  BusGraphTransformerNC  (node classification – no graph token)
# ============================================================================

class BusGraphTransformerNC(nn.Module):
    """Busemann Graph Transformer for node classification.

    All N nodes are treated as a "sequence" of Lorentz points.
    2-hop neighbourhood masking restricts attention to local structure.

    Optionally injects Graphormer-style shortest-path-distance + edge embeddings
    as additive bias to attention logits (requires ``spd_matrix`` in forward /
    LP ``embed``, same convention as ``BusTransformer``).
    """

    def __init__(self, d_in: int, d_model: int, num_heads: int,
                 num_layers: int, num_classes: int,
                 c: float = 1.0, dropout: float = 0.1,
                 learn_curvature: bool = False,
                 score_type: str = "busemann",
                 use_attn_bias: bool = False,
                 max_spd: int = 20,
                 attn_proj: str = "htc_qk",
                 head_type: str = "bmlr",
                 final_readout_curvature: str = "learned",
                 use_attention: bool = True,
                 use_gnn_branch: bool = True,
                 linear_landmarks: int = 64,
                 linear_rf_features: int = 128,
                 linear_rf_seed: int = 0,
                 linear_rf_t_max: float = 12.0,
                 linear_rf_radial_scale: Optional[float] = None,
                 linear_boundary_directions: int = 64,
                 linear_boundary_seed: int = 0,
                 linear_boundary_gate_init: Optional[float] = None,
                 linear_boundary_mix_init: float = 0.1,
                 linear_boundary_scales: int = 1,
                 linear_boundary_scale_spread: float = 0.35,
                 linear_boundary_fusion: str = "additive",
                 linear_boundary_preserve_rng: bool = False,
                 num_virtual_nodes: int = 0,
                 sparse_num_edge_types: int = 0,
                 ld_tau_init: Optional[float] = None,
                 ld_gamma_init: Optional[float] = None,
                 residual_gate_mode: str = "none",
                 attn_gate_init: float = 1.0,
                 gnn_gate_init: float = 1.0,
                 ffn_gate_init: float = 1.0,
                 linear_boundary_residual_norm: str = "none",
                 linear_boundary_residual_max_gain: float = 100.0,
                 linear_boundary_residual_eps: float = 1e-6,
                 linear_boundary_checkpoint_heads: bool = False,
                 linear_boundary_key_normalization: str = "none",
                 linear_boundary_fixed_exponent: Optional[float] = None,
                 linear_boundary_scale_routing: str = "adaptive"):
        super().__init__()
        _validate_boundary_scale_routing(linear_boundary_scale_routing)
        if final_readout_curvature not in ("learned", "unit"):
            raise ValueError(
                "final_readout_curvature must be 'learned' or 'unit', "
                f"got {final_readout_curvature!r}"
            )
        if num_virtual_nodes < 0:
            raise ValueError("num_virtual_nodes must be non-negative")
        if sparse_num_edge_types < 0:
            raise ValueError("sparse_num_edge_types must be non-negative")
        self.c          = c
        self.use_attention = use_attention
        self.use_gnn_branch = use_gnn_branch
        self.final_readout_curvature = final_readout_curvature
        self.num_virtual_nodes = num_virtual_nodes
        self.sparse_num_edge_types = sparse_num_edge_types
        # Global log_c for input lift (and final norm when no blocks are present).
        self.log_c      = nn.Parameter(torch.tensor(math.log(c)),
                                       requires_grad=learn_curvature)
        self.input_proj = nn.Linear(d_in, d_model)
        self.use_attn_bias = use_attn_bias
        self.attn_bias_mod = (
            GraphAttnBias(num_heads, max_spd=max_spd)
            if use_attn_bias and use_attention else None
        )
        # Each block owns its curvature; ``learn_curvature`` freezes/unfreezes it.
        self.blocks = nn.ModuleList([
            BusTransformerBlock(
                d_model,
                num_heads,
                c=c,
                dropout=dropout,
                score_type=score_type,
                attn_proj=attn_proj,
                linear_landmarks=linear_landmarks,
                linear_rf_features=linear_rf_features,
                linear_rf_seed=linear_rf_seed + layer_index,
                linear_rf_t_max=linear_rf_t_max,
                linear_rf_radial_scale=linear_rf_radial_scale,
                linear_boundary_directions=linear_boundary_directions,
                linear_boundary_seed=linear_boundary_seed + layer_index,
                linear_boundary_gate_init=linear_boundary_gate_init,
                linear_boundary_mix_init=linear_boundary_mix_init,
                linear_boundary_scales=linear_boundary_scales,
                linear_boundary_scale_spread=linear_boundary_scale_spread,
                linear_boundary_fusion=linear_boundary_fusion,
                linear_boundary_preserve_rng=linear_boundary_preserve_rng,
                linear_boundary_residual_norm=linear_boundary_residual_norm,
                linear_boundary_residual_max_gain=linear_boundary_residual_max_gain,
                linear_boundary_residual_eps=linear_boundary_residual_eps,
                linear_boundary_checkpoint_heads=linear_boundary_checkpoint_heads,
                linear_boundary_key_normalization=linear_boundary_key_normalization,
                linear_boundary_fixed_exponent=linear_boundary_fixed_exponent,
                linear_boundary_scale_routing=linear_boundary_scale_routing,
                learn_curvature=learn_curvature,
                use_attention=use_attention,
                use_gnn_branch=use_gnn_branch,
                ld_tau_init=ld_tau_init,
                ld_gamma_init=ld_gamma_init,
                residual_gate_mode=residual_gate_mode,
                attn_gate_init=attn_gate_init,
                gnn_gate_init=gnn_gate_init,
                ffn_gate_init=ffn_gate_init,
            )
            for layer_index in range(num_layers)
        ])
        self.final_norm = HypLayerNorm(d_model, c=c)
        if head_type == "bmlr":
            self.head = BMLR(d_model, num_classes)
        elif head_type == "hypcls":
            self.head = HypCLS(d_model, num_classes, c=c)
        elif head_type == "tangent_linear":
            self.head = nn.Linear(d_model, num_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")
        self.head_type  = head_type
        self.drop       = nn.Dropout(dropout)
        self.virtual_node_emb = (
            nn.Parameter(0.02 * torch.randn(num_virtual_nodes, d_model))
            if num_virtual_nodes > 0 else None
        )
        self.sparse_edge_type_bias = (
            nn.Parameter(torch.zeros(
                num_layers, num_heads, sparse_num_edge_types
            ))
            if sparse_num_edge_types > 0 else None
        )

    @torch.no_grad()
    def initialize_boundary_dictionaries(
        self,
        x_feat: torch.Tensor,
        iterations: int = 20,
    ) -> None:
        """Spherical-k-means initialization on each layer's query directions.

        This data-dependent initializer is intentionally explicit rather than
        triggered by ``forward``. It supports the global boundary-dictionary
        configuration used by the linear-attention experiment. Each layer is
        initialized from its actual pre-normalized hidden representation, then
        evaluated once so the next layer sees the corresponding hidden state.
        """
        if self.num_virtual_nodes > 0:
            raise ValueError(
                "boundary-dictionary initialization does not support virtual "
                "nodes"
            )
        if not self.blocks:
            return
        for block in self.blocks:
            if (
                block.attn is None
                or block.attn.score_type
                != "busemann_log_depth_linear_dictionary"
            ):
                raise ValueError(
                    "all initialized blocks must use global "
                    "boundary-dictionary attention"
                )

        was_training = self.training
        self.eval()
        try:
            num_nodes = x_feat.shape[0]
            input_curvature = self.log_c.exp()
            spatial = self.input_proj(x_feat)
            x = Hyperboloid.proj(
                torch.cat(
                    [spatial.new_zeros(num_nodes, 1), spatial], dim=-1
                ),
                input_curvature,
            ).unsqueeze(0)

            for block in self.blocks:
                curvature = block.curvature()
                x_norm = block.norm1(x, c=curvature)
                query_directions = []
                for head in range(block.attn.num_heads):
                    query = block.attn.W_Q[head](x_norm, c=curvature)
                    query_directions.append(
                        F.normalize(query[..., 1:], dim=-1).reshape(
                            -1, block.attn.d_head
                        )
                    )
                block.attn.initialize_boundary_dictionary(
                    torch.stack(query_directions, dim=0),
                    iterations=iterations,
                )
                x = block(x)
        finally:
            self.train(was_training)

    def _prep_attn_bias(
        self,
        spd_matrix: Optional[torch.Tensor],   # (1, N, N) int64
        adj_matrix: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Returns (1, H, N, N) additive bias or None."""
        if self.attn_bias_mod is None or spd_matrix is None or adj_matrix is None:
            return None
        # adjacency as 0/1 long for embedding table
        A = adj_matrix
        if A.dim() == 2:
            A = A.unsqueeze(0)
        Ad = A.long().clamp(min=0, max=1)
        return self.attn_bias_mod(spd_matrix, Ad.bool())

    @staticmethod
    def _build_adj_weights(adj_matrix: torch.Tensor, N: int,
                           device: torch.device) -> torch.Tensor:
        """Build row-normalised adjacency weights with self-loops.

        adj_matrix : (N, N) or (1, N, N) bool/float
        Returns    : (1, N, N) float, rows summing to 1
        """
        A = adj_matrix.float()
        if A.dim() == 2:
            A = A.unsqueeze(0)                        # (1, N, N)
        # Add self-loops
        I   = torch.eye(N, device=device, dtype=A.dtype).unsqueeze(0)
        A   = (A + I).clamp(max=1.0)
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return A / deg                                # (1, N, N)

    def forward(self,
                x_feat:      torch.Tensor,                     # (N, d_in)
                bool_mask:   Optional[torch.Tensor] = None,    # (1, N, N) bool
                spd_matrix:  Optional[torch.Tensor] = None,    # (1, N, N) int64 SPD
                adj_matrix: Optional[torch.Tensor] = None,     # (N, N) or (1, N, N) bool
                attn_edge_index: Optional[torch.Tensor] = None,  # (2,E), row=query
                edge_attn_bias: Optional[torch.Tensor] = None,   # (H,E) or (1,H,E)
                attn_edge_type: Optional[torch.Tensor] = None,   # (E,) integer type
                ) -> torch.Tensor:
        N   = x_feat.shape[0]
        dev = x_feat.device

        # Lift features to hyperboloid using global curvature
        c     = self.log_c.exp()
        xs    = self.drop(self.input_proj(x_feat))            # (N, d_model)
        dummy = torch.cat([xs.new_zeros(N, 1), xs], dim=-1)
        x     = Hyperboloid.proj(dummy, c).unsqueeze(0)       # (1, N, d_model+1)

        if self.num_virtual_nodes > 0:
            if attn_edge_index is None:
                raise ValueError(
                    "virtual nodes require a sparse attn_edge_index"
                )
            if bool_mask is not None or spd_matrix is not None:
                raise ValueError(
                    "virtual nodes do not support dense masks or SPD bias"
                )
            virtual_spatial = self.virtual_node_emb
            virtual_dummy = torch.cat(
                [
                    virtual_spatial.new_zeros(
                        self.num_virtual_nodes, 1
                    ),
                    virtual_spatial,
                ],
                dim=-1,
            )
            virtual = Hyperboloid.proj(
                virtual_dummy, c
            ).unsqueeze(0)
            x = torch.cat([x, virtual], dim=1)

        if attn_edge_type is not None:
            if attn_edge_index is None:
                raise ValueError(
                    "attn_edge_type requires attn_edge_index"
                )
            if self.sparse_edge_type_bias is None:
                raise ValueError(
                    "attn_edge_type requires sparse_num_edge_types > 0"
                )
            if (
                attn_edge_type.ndim != 1
                or attn_edge_type.shape[0] != attn_edge_index.shape[1]
            ):
                raise ValueError(
                    "attn_edge_type must have shape (E,)"
                )
            if attn_edge_type.dtype != torch.long:
                raise TypeError("attn_edge_type must use torch.long indices")
            if attn_edge_type.device != dev:
                raise ValueError(
                    "attn_edge_type and model inputs must share a device"
                )
            if not bool(
                (
                    (attn_edge_type >= 0)
                    & (attn_edge_type < self.sparse_num_edge_types)
                ).all()
            ):
                raise ValueError("attn_edge_type contains an invalid type")

        if (
            attn_edge_index is not None
            and self.attn_bias_mod is not None
            and spd_matrix is not None
        ):
            raise ValueError(
                "exact sparse HSA does not materialize Graphormer pairwise "
                "bias; pass an edge_attn_bias defined only on attn_edge_index "
                "without spd_matrix, or disable use_attn_bias"
            )
        attn_bias = self._prep_attn_bias(spd_matrix, adj_matrix)

        # Pre-compute row-normalised adjacency once for all blocks
        adj_weights = (self._build_adj_weights(adj_matrix, N, dev)
                       if adj_matrix is not None else None)   # (1, N, N) or None
        if adj_weights is not None and self.num_virtual_nodes > 0:
            N_eff = N + self.num_virtual_nodes
            padded = adj_weights.new_zeros(
                adj_weights.shape[0], N_eff, N_eff
            )
            padded[:, :N, :N] = adj_weights
            virtual_index = torch.arange(N, N_eff, device=dev)
            padded[:, virtual_index, virtual_index] = 1.0
            adj_weights = padded

        for layer_index, block in enumerate(self.blocks):
            layer_edge_bias = edge_attn_bias
            if attn_edge_type is not None:
                type_bias = self.sparse_edge_type_bias[
                    layer_index, :, attn_edge_type
                ]
                layer_edge_bias = (
                    type_bias
                    if layer_edge_bias is None
                    else layer_edge_bias + type_bias
                )
            x = block(
                x,
                attn_bias=attn_bias,
                bool_mask=bool_mask,
                attn_edge_index=attn_edge_index,
                edge_attn_bias=layer_edge_bias,
                adj_weights=adj_weights,
            )

        c_final = self.blocks[-1].curvature() if len(self.blocks) else c
        if self.final_readout_curvature == "unit":
            # Legacy NC readout: canonicalise the representation on H_{-1}
            # before applying the unit-curvature BMLR head.
            c_readout = self.c
            x = self.final_norm(x)
        else:
            c_readout = c_final
            x = self.final_norm(x, c=c_readout)
        x = x.squeeze(0)[:N]
        if self.head_type == "hypcls":
            return self.head(x, c=c_readout)                   # (N, num_classes)
        if self.head_type == "tangent_linear":
            tangent_spatial = Hyperboloid.logmap0(
                x, c_readout
            )[..., 1:]
            return self.head(tangent_spatial)                  # (N, num_classes)
        return self.head(x)                                    # (N, num_classes)


# ============================================================================
# 12.  EucGraphTransformerNC  (Euclidean baseline with pre-LN, Graphormer style)
# ============================================================================

class EucGraphTransformerNC(nn.Module):
    """Standard Euclidean Graph Transformer baseline.
    Uses pre-LayerNorm residuals (norm_first=True) matching Graphormer."""

    def __init__(self, d_in: int, d_model: int, num_heads: int,
                 num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True,
            norm_first=True,              # pre-LayerNorm (Graphormer style)
        )
        self.encoder    = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x_feat: torch.Tensor,
                float_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        float_mask : (N, N) float,  0.0 = attend, −∞ = block
        """
        xs = self.drop(self.input_proj(x_feat))
        x  = self.encoder(xs.unsqueeze(0), mask=float_mask).squeeze(0)
        return self.classifier(x)


# ============================================================================
# Backward-compatibility aliases
# ============================================================================

# v1 names → v2 equivalents
# Legacy API alias. HypLinear is not the Busemann FC layer introduced by HBNN.
BFC                       = HypLinear
BusemannMultiHeadAttention = BusemannMHA        # old name alias
project_to_lorentz = lambda x, eps=1e-5: Hyperboloid.proj(x, c=1.0)


# ============================================================================
# Smoke test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("Busemannformer (busemann_transformer.py) – smoke test")
    print("=" * 60)

    # --- Sequence model (graph token + structural bias) ---
    B, N, d, H, L, C = 2, 16, 32, 4, 2, 5
    model = BusTransformer(
        vocab_size=100, d_model=d, num_heads=H, num_layers=L,
        num_classes=C, max_seq_len=N, c=1.0, dropout=0.0,
        use_graph_token=True, use_attn_bias=False,
    )
    ids    = torch.randint(0, 100, (B, N))
    logits = model(ids)
    print(f"BusTransformer  logits shape : {logits.shape}")   # (2, 5)
    assert logits.shape == (B, C), f"Expected ({B},{C}), got {logits.shape}"

    # --- Node-classification model (no GNN) ---
    nc     = BusGraphTransformerNC(d_in=9, d_model=d, num_heads=H,
                                   num_layers=L, num_classes=2, c=1.0)
    x_feat = torch.randn(100, 9)
    mask   = torch.zeros(1, 100, 100, dtype=torch.bool)
    out    = nc(x_feat, bool_mask=mask)
    print(f"BusGraphTransNC logits shape : {out.shape}")       # (100, 2)
    assert out.shape == (100, 2)

    # --- Node-classification model WITH Lorentz GNN + per-layer curvature ---
    nc_gnn  = BusGraphTransformerNC(d_in=9, d_model=d, num_heads=H,
                                    num_layers=L, num_classes=2, c=1.0,
                                    learn_curvature=True)
    adj_rnd = (torch.rand(100, 100) > 0.8).bool()              # random sparse graph
    out_gnn = nc_gnn(x_feat, bool_mask=mask, adj_matrix=adj_rnd)
    print(f"BusGraphTransNC+GNN logits  : {out_gnn.shape}")    # (100, 2)
    assert out_gnn.shape == (100, 2)

    # Verify each block learns a distinct curvature after one backward pass
    loss = out_gnn.sum()
    loss.backward()
    cs = [blk.log_c_param.exp().item() for blk in nc_gnn.blocks]
    print(f"Per-layer curvatures (init)  : {[f'{c:.4f}' for c in cs]}")
    grads_ok = all(blk.log_c_param.grad is not None for blk in nc_gnn.blocks)
    assert grads_ok, "log_c_param gradient missing in some block"
    print(f"log_c_param gradients exist  : {grads_ok}")

    # --- HypCLS head ---
    seq_cls = BusTransformer(
        vocab_size=100, d_model=d, num_heads=H, num_layers=L,
        num_classes=C, max_seq_len=N, c=1.0, dropout=0.0,
        use_graph_token=True, use_attn_bias=False, head_type="hypcls",
    )
    logits2 = seq_cls(ids)
    print(f"HypCLS head     logits shape : {logits2.shape}")   # (2, 5)
    assert logits2.shape == (B, C)

    # --- Verify Lorentz constraint (‖x‖_L ≈ −1/c) ---
    with torch.no_grad():
        e   = model.token_emb(ids) + model.pos_emb(
            torch.arange(N).unsqueeze(0).expand(B, -1))
        x_h = Hyperboloid.proj(
            torch.cat([e.new_zeros(*e.shape[:-1], 1), e], dim=-1), c=1.0)
        norms = Hyperboloid.minkowski_dot(x_h, x_h, keepdim=False)
        print(f"Lorentz norms   (should ≈ -1) : {norms[0, :3].tolist()}")

    print("\nAll smoke tests passed.")
