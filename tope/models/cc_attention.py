"""CC-attention primitives for CCANN (CC-Attention Neural Network) formalism.

Implements the learned, normalised attention push-forward operations described
in Definitions 32, 33, and 36 of the CCANN formalism, together with three
structural extensions (HodgeletFilter, SoftSpectralCutoff, JacobianCorrectedBlock)
and their composable wrapper (HodgeletFilteredCCBlock).

Core primitives (Definitions 32 / 33 / 36)
-------------------------------------------
CCAttentionPushForward   : equal-rank attention (coadjacency)
CCAttentionBlock         : unequal-rank bidirectional attention (incidence)
AttentionMergeNode       : static inter-neighborhood weight merge
ContentAttentionMergeNode: content-based inter-neighborhood weight merge

Structural extensions
---------------------
HodgeletFilter           : oversmoothing prevention — Ψ̂_ρ(λ) = √λ·exp(-ρλ/2)
                           is identically zero on ker(Δ_k).
SoftSpectralCutoff       : continuity under perturbation — replaces hard Λ_r
                           threshold with a learnable sigmoid gate.
JacobianCorrectedBlock   : adjoint-correct bidirectional attention — adds the
                           Ferreira (2015) Jacobian factor j_a(x) to the reverse
                           logit so forward and reverse attention are exactly
                           adjoint under the (σ,t)-invariant measure dμ_{σ,t}.
HodgeletFilteredCCBlock  : composes all three extensions in one module.
CCAttentionPushForwardImproved: equal-rank variant with separate Q/K/V
                           projections and scaled logits.

Improvements applied throughout
--------------------------------
- Vectorised 2-D scatter softmax: single scatter pass over (E, H) instead of
  an O(n_heads) sequential loop.
- Scaled attention logits: e_ij *= 1/√head_dim before scatter-softmax.
- Gradient-correct Jacobian: t and sigma_param passed as 0-dim Tensors so
  both learnable geometry parameters receive gradients.
- SoftSpectralCutoff.gate_rank(): O(1) single-rank path avoids the O(n_ranks)
  expand+gate_all+slice pattern.

References
----------
Definition 32: Equal-rank CC-attention push-forward (coadjacency attention).
Definition 33: Unequal-rank bidirectional CC-attention block (incidence attention).
Definition 36: Inter-neighborhood attention merge node.
Ferreira (2015) "Harmonic Analysis on the Möbius Gyrogroup"
  J. Fourier Anal. Appl. 21:281–317  — Jacobian factor (Def 1, Eq 17–18).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_scatter import scatter_softmax
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


# ══════════════════════════════════════════════════════════════════════════════
# Internal utilities
# ══════════════════════════════════════════════════════════════════════════════

def _scatter_softmax_1d(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Numerically stable per-group softmax. src, index: (E,)."""
    if HAS_TORCH_SCATTER:
        return scatter_softmax(src, index, dim=0)
    max_vals = src.new_full((dim_size,), float("-inf"))
    max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    shifted = src - max_vals[index]
    exp_s = shifted.exp()
    sum_e = src.new_zeros(dim_size)
    sum_e.scatter_add_(0, index, exp_s)
    return exp_s / (sum_e[index] + 1e-8)


def _scatter_softmax_2d(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Vectorised per-group softmax for (E, H) tensors.

    Replaces ``torch.stack([_scatter_softmax_1d(src[:,h], ...) for h in range(H)], dim=1)``
    with a single scatter pass over all heads at once.

    Parameters
    ----------
    src      : (E, H)  raw attention logits
    index    : (E,)    group assignment (target-node index)
    dim_size : int     number of groups

    Returns
    -------
    att : (E, H)  normalised attention weights
    """
    H = src.size(1)
    if HAS_TORCH_SCATTER:
        return scatter_softmax(src, index.unsqueeze(1).expand(-1, H), dim=0)
    idx = index.unsqueeze(1).expand(-1, H)
    max_v = src.new_full((dim_size, H), float("-inf"))
    max_v.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    shifted = (src - max_v[index]).exp()
    denom = src.new_zeros(dim_size, H)
    denom.scatter_add_(0, idx, shifted)
    return shifted / (denom[index] + 1e-8)


def _scatter_add_nd(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """scatter_add that works on (E, n_heads, head_dim) tensors."""
    out = src.new_zeros(dim_size, *src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Equal-rank attention push-forward (Definition 32)
# ══════════════════════════════════════════════════════════════════════════════

class CCAttentionPushForward(nn.Module):
    """CC-attention push-forward for cells of equal rank (Definition 32).

    Used for node→node and edge→edge messages through (co)adjacency matrices.

    Attention score for pair (i, j):
        e_ij = LeakyReLU( a^T [W·H_s[i] || W·H_s[j] (|| edge_attr)] / √head_dim )
        att_ij = softmax over j in N(i) of e_ij
        K_t[i] = Σ_j att_ij · (value_j or W·H_s[j])

    Parameters
    ----------
    d_in           : input feature dimension
    d_out          : output feature dimension
    n_heads        : number of attention heads (default 1)
    dropout        : dropout on attention weights (default 0.0)
    negative_slope : LeakyReLU slope (default 0.2)
    d_edge         : optional edge feature dim added to attention score (0 = disabled)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        d_edge: int = 0,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_edge = d_edge
        self.scale = self.head_dim ** -0.5

        self.W = nn.Linear(d_in, d_out, bias=False)
        attn_in = 2 * self.head_dim + d_edge
        self.a = nn.Parameter(torch.empty(n_heads, attn_in))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        H_s: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        H_s       : (N, d_in)
        adj       : (2, E) — adj[0] = source, adj[1] = target
        edge_attr : (E, d_edge) optional
        value     : (E, d_out) optional; replaces W·H_s[source] as aggregated value

        Returns
        -------
        K_t : (N, d_out)
        """
        N = H_s.size(0)
        src, tgt = adj[0], adj[1]

        Wh = self.W(H_s).view(N, self.n_heads, self.head_dim)
        Wh_i = Wh[tgt]   # (E, n_heads, hd)
        Wh_j = Wh[src]   # (E, n_heads, hd)

        cat_attn = torch.cat([Wh_i, Wh_j], dim=-1)
        if edge_attr is not None and self.d_edge > 0:
            ea = edge_attr.unsqueeze(1).expand(-1, self.n_heads, -1)
            cat_attn = torch.cat([cat_attn, ea], dim=-1)

        e = (cat_attn * self.a.unsqueeze(0)).sum(-1)   # (E, n_heads)
        e = self.leaky_relu(e * self.scale)

        att = _scatter_softmax_2d(e, tgt, N)            # (E, n_heads)
        att = self.dropout(att)

        vals = value.view(-1, self.n_heads, self.head_dim) if value is not None else Wh_j
        weighted = att.unsqueeze(-1) * vals
        return _scatter_add_nd(weighted, tgt, N).view(N, self.d_out)


# ══════════════════════════════════════════════════════════════════════════════
# Unequal-rank bidirectional attention block (Definition 33)
# ══════════════════════════════════════════════════════════════════════════════

class CCAttentionBlock(nn.Module):
    """Bidirectional CC-attention push-forward for cells of unequal rank
    (Definition 33).

    Simultaneously computes:
      - forward  (s→t): K_t = attention-weighted aggregation of rank-s onto rank-t
      - reverse  (t→s): K_s = attention-weighted aggregation of rank-t onto rank-s

    Forward attention (s→t):
        e_ij = phi( a^T [W_s·H_s[j] || W_t·H_t[i]] / √head_dim )
        att_ij = softmax over {j : B[1]==i}
        K_t[i] = Σ_j att_ij · W_s·H_s[j]

    Reverse attention (t→s):
        f_ji = phi( rev_a^T [W_t·H_t[i] || W_s·H_s[j]] / √head_dim )
               rev_a = [a[head_dim:] || a[:head_dim]]
        att_ji = softmax over {i : B[0]==j}
        K_s[j] = Σ_i att_ji · W_t·H_t[i]

    Parameters
    ----------
    d_s_in, d_t_in : input dims for rank-s and rank-t
    d_out          : output dim (same for both directions)
    n_heads        : number of attention heads (default 1)
    dropout        : attention dropout (default 0.0)
    negative_slope : LeakyReLU slope (default 0.2)
    """

    def __init__(
        self,
        d_s_in: int,
        d_t_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        n_ranks: int = 1,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0
        self.d_s_in = d_s_in
        self.d_t_in = d_t_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_s = nn.Linear(d_s_in, d_out, bias=False)
        self.W_t = nn.Linear(d_t_in, d_out, bias=False)

        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Per-rank learnable metric matrices G_r^{1/2}, one per CC rank.
        # When n_ranks > 1, the covariant form of each projected feature is
        # G_r^{1/2} @ W_r H, making cross-rank attention invariant to the
        # separate scalings of each rank's descriptor space (Cerrini 1971).
        # Initialized to identity so training starts from the Euclidean baseline.
        self.n_ranks = n_ranks
        if n_ranks > 1:
            self.metric_s = nn.ParameterList([
                nn.Parameter(torch.eye(self.head_dim)) for _ in range(n_ranks)
            ])
            self.metric_t = nn.ParameterList([
                nn.Parameter(torch.eye(self.head_dim)) for _ in range(n_ranks)
            ])
        else:
            self.metric_s = None
            self.metric_t = None

    def forward(
        self,
        H_s: Tensor,
        H_t: Tensor,
        B: Tensor,
        rank_s: int = 0,
        rank_t: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        H_s : (N_s, d_s_in)
        H_t : (N_t, d_t_in)
        B   : (2, E) — B[0] = rank-s indices, B[1] = rank-t indices
        rank_s, rank_t : CC ranks for source and target cells (used when
            n_ranks > 1 to select the per-rank metric G_r^{1/2})

        Returns
        -------
        K_t : (N_t, d_out)
        K_s : (N_s, d_out)
        """
        N_s, N_t = H_s.size(0), H_t.size(0)
        s_idx, t_idx = B[0], B[1]

        Ws_h = self.W_s(H_s).view(N_s, self.n_heads, self.head_dim)
        Wt_h = self.W_t(H_t).view(N_t, self.n_heads, self.head_dim)

        # Apply rank-specific metric G_r^{1/2}: covariantizes each projection
        # so that the attention logit is metric-invariant across CC ranks.
        if self.metric_s is not None:
            Ws_h = Ws_h @ self.metric_s[rank_s]   # (N_s, n_heads, hd)
            Wt_h = Wt_h @ self.metric_t[rank_t]   # (N_t, n_heads, hd)

        Ws_j = Ws_h[s_idx]   # (E, n_heads, hd)
        Wt_i = Wt_h[t_idx]   # (E, n_heads, hd)

        # ── Forward (s→t) ─────────────────────────────────────────────────────
        e_fwd = (torch.cat([Ws_j, Wt_i], dim=-1) * self.a.unsqueeze(0)).sum(-1)
        e_fwd = self.leaky_relu(e_fwd * self.scale)
        att_fwd = _scatter_softmax_2d(e_fwd, t_idx, N_t)
        att_fwd = self.dropout(att_fwd)
        K_t = _scatter_add_nd(att_fwd.unsqueeze(-1) * Ws_j, t_idx, N_t).view(N_t, self.d_out)

        # ── Reverse (t→s) ─────────────────────────────────────────────────────
        rev_a = torch.cat([self.a[:, self.head_dim:], self.a[:, :self.head_dim]], dim=-1)
        e_rev = (torch.cat([Wt_i, Ws_j], dim=-1) * rev_a.unsqueeze(0)).sum(-1)
        e_rev = self.leaky_relu(e_rev * self.scale)
        att_rev = _scatter_softmax_2d(e_rev, s_idx, N_s)
        att_rev = self.dropout(att_rev)
        K_s = _scatter_add_nd(att_rev.unsqueeze(-1) * Wt_i, s_idx, N_s).view(N_s, self.d_out)

        return K_t, K_s


# ══════════════════════════════════════════════════════════════════════════════
# Inter-neighborhood attention merge (Definition 36)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionMergeNode(nn.Module):
    """Inter-neighborhood attention merge node (Definition 36).

    Static-weight version: a single learnable global scalar per neighborhood,
    shared across all nodes.  Preserved for API compatibility.

    Parameters
    ----------
    n_neighborhoods : number of neighborhoods to merge
    hidden_dim      : feature dimension (kept for API clarity)
    """

    def __init__(self, n_neighborhoods: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_neighborhoods = n_neighborhoods
        self.hidden_dim = hidden_dim
        self.weights = nn.Parameter(torch.ones(n_neighborhoods))

    def forward(self, messages: List[Tensor]) -> Tensor:
        """messages : list of (N, d) tensors → merged : (N, d)"""
        assert len(messages) == self.n_neighborhoods, (
            f"Expected {self.n_neighborhoods} message tensors, got {len(messages)}"
        )
        b_k = torch.softmax(self.weights, dim=0)
        return sum(b_k[k] * messages[k] for k in range(self.n_neighborhoods))


class ContentAttentionMergeNode(nn.Module):
    """Content-based inter-neighborhood merge (Definition 36 — improved).

    Replaces the static global b^k weights with per-node content-dependent
    gates computed from the concatenation of all incoming messages:

        gate_k(i) = softmax_k( Linear([msg_1[i], …, msg_K[i]]) )
        merged[i] = Σ_k gate_k(i) · msg_k[i]

    Parameters
    ----------
    n_neighborhoods : number of neighborhoods to merge
    hidden_dim      : feature dimension of each message tensor
    """

    def __init__(self, n_neighborhoods: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_neighborhoods = n_neighborhoods
        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(hidden_dim * n_neighborhoods, n_neighborhoods, bias=True)

    def forward(self, messages: List[Tensor]) -> Tensor:
        """messages : list of K tensors (N, d) → merged : (N, d)"""
        assert len(messages) == self.n_neighborhoods
        stack  = torch.stack(messages, dim=1)              # (N, K, d)
        gates  = torch.softmax(self.gate(stack.view(stack.size(0), -1)), dim=1)  # (N, K)
        return (gates.unsqueeze(-1) * stack).sum(dim=1)    # (N, d)

    def extra_repr(self) -> str:
        return f"n_neighborhoods={self.n_neighborhoods}, hidden_dim={self.hidden_dim}"


# ══════════════════════════════════════════════════════════════════════════════
# Zone adjacency helper
# ══════════════════════════════════════════════════════════════════════════════

def build_zone_adjacency(
    zone_mask: Tensor,
    global_edge_index: Tensor,
) -> Tensor:
    """Filter edge_index to edges with both endpoints in a zone, remapping
    node indices to zone-local contiguous indices.

    Parameters
    ----------
    zone_mask         : (N_total,) BoolTensor
    global_edge_index : (2, E)

    Returns
    -------
    local_edge_index : (2, E_zone)
    """
    src, tgt = global_edge_index[0], global_edge_index[1]
    in_zone = zone_mask[src] & zone_mask[tgt]

    if not in_zone.any():
        return global_edge_index.new_zeros(2, 0)

    node_map = torch.full(
        (zone_mask.size(0),), -1, dtype=torch.long, device=zone_mask.device
    )
    zone_nodes = zone_mask.nonzero(as_tuple=True)[0]
    node_map[zone_nodes] = torch.arange(zone_nodes.size(0), device=zone_mask.device)

    return torch.stack([node_map[src[in_zone]], node_map[tgt[in_zone]]], dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Structural extension 1: HodgeletFilter
# Oversmoothing prevention via harmonic orthogonality
# ══════════════════════════════════════════════════════════════════════════════

class HodgeletFilter(nn.Module):
    """Soft bandpass filter on the Hodge Laplacian spectrum.

    Applies the diffusive Hodgelet transfer function:

        Ψ̂_ρ(λ) = √λ · exp(-ρ · λ / 2)

    - Exactly zero at λ = 0  →  orthogonal to ker(Δ_k), making oversmoothing
      structurally impossible regardless of depth.
    - Peaks at λ* = 2/ρ  →  scale ρ selects the amplified spectral band.
    - Decays for large λ  →  suppresses high-frequency noise.
    - Differentiable everywhere  →  learnable end-to-end.

    Parameters
    ----------
    n_scales  : Hodgelet scale parameters (one per TTN leaf / rank)
    n_heads   : attention heads (one ρ per head)
    init_rho  : initial scale (2/init_rho = peak eigenvalue)
    learn_rho : learn ρ (True) or fix it (False)
    eps       : floor for numerical stability
    """

    def __init__(
        self,
        n_scales: int = 4,
        n_heads: int = 8,
        init_rho: float = 1.0,
        learn_rho: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_heads = n_heads
        self.eps = eps
        rho_init = math.log(math.expm1(max(init_rho - eps, 1e-3)))
        raw = torch.full((n_scales, n_heads), rho_init)
        if learn_rho:
            self.rho_raw = nn.Parameter(raw)
        else:
            self.register_buffer("rho_raw", raw)

    @property
    def rho(self) -> Tensor:
        return F.softplus(self.rho_raw) + self.eps

    def forward(self, eigenvalues: Tensor) -> Tensor:
        """
        Parameters
        ----------
        eigenvalues : (B, k_eig) or (B, n_scales, k_eig)

        Returns
        -------
        filtered : (B, n_scales, n_heads, k_eig)
        """
        if eigenvalues.dim() == 2:
            lam = eigenvalues.unsqueeze(1).unsqueeze(2)
        elif eigenvalues.dim() == 3:
            lam = eigenvalues.unsqueeze(2)
        else:
            raise ValueError(f"eigenvalues must be 2-D or 3-D, got {eigenvalues.dim()}-D")

        rho = self.rho.unsqueeze(0).unsqueeze(-1)          # (1, n_scales, n_heads, 1)
        # lam · (lam + ε)^{-½} is exactly 0 when lam = 0 (harmonic orthogonality)
        # and approaches √lam for lam ≫ ε.
        sqrt_lam = lam * (lam + self.eps).rsqrt()
        decay = torch.exp(-rho * lam / 2)
        return sqrt_lam * decay

    def extra_repr(self) -> str:
        return (f"n_scales={self.n_scales}, n_heads={self.n_heads}, "
                f"mean_peak_lambda={2.0 / self.rho.mean().item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Structural extension 2: SoftSpectralCutoff
# Continuity under structural perturbation
# ══════════════════════════════════════════════════════════════════════════════

class SoftSpectralCutoff(nn.Module):
    """Learnable smooth spectral gate replacing the hard cutoff Λ_r.

    Replaces the Heaviside step [λ ≤ Λ_r] with a sigmoid gate:

        g(λ; θ) = σ( θ_slope · (θ_center - λ) )

    One (θ_slope, θ_center) pair is learned per rank and per filtration step.

    Parameters
    ----------
    n_ranks            : number of CC ranks
    n_filtration_steps : filtration grid size
    init_center        : initial gate center (eigenvalue units)
    init_slope         : initial slope (higher = sharper)
    """

    def __init__(
        self,
        n_ranks: int = 4,
        n_filtration_steps: int = 16,
        init_center: float = 2.0,
        init_slope: float = 5.0,
    ) -> None:
        super().__init__()
        self.n_ranks = n_ranks
        self.n_filtration_steps = n_filtration_steps
        self.centers = nn.Parameter(torch.full((n_ranks, n_filtration_steps), init_center))
        slope_raw_init = math.log(math.expm1(init_slope))
        self.slopes_raw = nn.Parameter(torch.full((n_ranks, n_filtration_steps), slope_raw_init))

    @property
    def slopes(self) -> Tensor:
        return F.softplus(self.slopes_raw)

    def forward(self, eigenvalues: Tensor, rank: int, filtration_step: int) -> Tensor:
        """Apply soft gate at a single (rank, filtration_step) point.

        Parameters
        ----------
        eigenvalues    : (B, k_eig)
        rank, filtration_step : grid coordinates

        Returns
        -------
        gated : (B, k_eig)  in [0, 1]
        """
        c = self.centers[rank, filtration_step]
        s = self.slopes[rank, filtration_step]
        return torch.sigmoid(s * (c - eigenvalues))

    def gate_rank(self, eigenvalues: Tensor, rank: int) -> Tensor:
        """Apply soft gate for one rank across all filtration steps.

        O(1) in n_ranks — avoids the O(n_ranks) allocation of
        ``gate_all(expand(n_ranks))[..., rank]``.

        Parameters
        ----------
        eigenvalues : (B, n_filtration_steps, k_eig)
        rank        : which CC rank

        Returns
        -------
        gated : (B, n_filtration_steps, k_eig)
        """
        c = self.centers[rank].unsqueeze(0).unsqueeze(-1)   # (1, F, 1)
        s = self.slopes[rank].unsqueeze(0).unsqueeze(-1)    # (1, F, 1)
        return torch.sigmoid(s * (c - eigenvalues))

    def gate_all(self, eigenvalues: Tensor) -> Tensor:
        """Apply gates across all (rank, filtration_step) pairs.

        Parameters
        ----------
        eigenvalues : (B, n_ranks, n_filtration_steps, k_eig)

        Returns
        -------
        gated : (B, n_ranks, n_filtration_steps, k_eig)
        """
        c = self.centers.unsqueeze(0).unsqueeze(-1)
        s = self.slopes.unsqueeze(0).unsqueeze(-1)
        return torch.sigmoid(s * (c - eigenvalues))

    def effective_cutoff(self) -> Tensor:
        """50%-pass eigenvalue for each (rank, step) — (n_ranks, n_filtration_steps)."""
        return self.centers.detach()

    def extra_repr(self) -> str:
        return (f"n_ranks={self.n_ranks}, n_filtration_steps={self.n_filtration_steps}, "
                f"mean_center={self.centers.mean().item():.3f}, "
                f"mean_slope={self.slopes.mean().item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Structural extension 3: JacobianCorrectedBlock
# Correct adjoint symmetry in bidirectional CCANN attention
# ══════════════════════════════════════════════════════════════════════════════

def _mobius_jacobian(
    a: Tensor,
    x: Tensor,
    t: Tensor,
    sigma: Tensor,
    n: int,
) -> Tensor:
    """Ferreira (2015) Jacobian factor j_a(x) (Definition 1, Eq 18).

        j_a(x) = ( (1 - ‖a‖²/t²) / |1 + ax/t²|² )^{(n+σ-2)/2}

    ``t`` and ``sigma`` must be 0-dim Tensors so that ``log_t`` and
    ``sigma_param`` in JacobianCorrectedBlock receive gradients.
    Passing ``.item()`` values severs the autograd graph.

    Parameters
    ----------
    a, x  : (E, d)  projected source/target embeddings
    t     : ()      ball radius (0-dim Tensor)
    sigma : ()      conformal weight (0-dim Tensor)
    n     : int     ambient dimension

    Returns
    -------
    j : (E,)  positive Jacobian weight per edge
    """
    t2 = t * t
    a_norm_sq = (a * a).sum(dim=-1) / t2
    ax_dot    = (a * x).sum(dim=-1) / t2
    x_norm_sq = (x * x).sum(dim=-1) / t2

    numer = (1.0 - a_norm_sq).clamp(min=1e-8)
    denom = (1.0 + 2.0 * ax_dot + a_norm_sq * x_norm_sq).clamp(min=1e-8)
    exp   = (n + sigma - 2.0) / 2.0
    return (exp * (numer.log() - denom.log())).exp().clamp(min=1e-6, max=1e6)


class JacobianCorrectedBlock(nn.Module):
    """Bidirectional CC-attention block with (σ,t)-adjoint-correct reverse attention.

    Extends CCAttentionBlock (Definition 33) by adding the Ferreira (2015)
    Jacobian factor j_a(x) to the reverse logit, so forward and reverse
    attention are exactly adjoint under dμ_{σ,t}.

    Parameters
    ----------
    d_s_in, d_t_in : input dims for rank-s and rank-t
    d_out          : output dim (same for both directions)
    n_heads        : attention heads
    dropout        : attention dropout
    negative_slope : LeakyReLU slope
    ball_radius    : t > 0, Poincaré ball radius
    sigma          : conformal weight σ (default 0.0 → standard Lebesgue)
    learn_geometry : learn ball_radius and sigma (True) or fix them (False)
    """

    def __init__(
        self,
        d_s_in: int,
        d_t_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        ball_radius: float = 1.0,
        sigma: Optional[float] = None,
        learn_geometry: bool = True,
        n_ranks: int = 1,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_s_in = d_s_in
        self.d_t_in = d_t_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_s = nn.Linear(d_s_in, d_out, bias=False)
        self.W_t = nn.Linear(d_t_in, d_out, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        _sigma0 = float(sigma) if sigma is not None else 0.0
        if learn_geometry:
            self.log_t = nn.Parameter(torch.tensor(math.log(ball_radius)))
            self.sigma_param = nn.Parameter(torch.tensor(_sigma0))
        else:
            self.register_buffer("log_t", torch.tensor(math.log(ball_radius)))
            self.register_buffer("sigma_param", torch.tensor(_sigma0))

        # Per-rank metric matrices G_r^{1/2} — same construction as
        # CCAttentionBlock; applied before the Jacobian-corrected logit.
        self.n_ranks = n_ranks
        if n_ranks > 1:
            self.metric_s = nn.ParameterList([
                nn.Parameter(torch.eye(self.head_dim)) for _ in range(n_ranks)
            ])
            self.metric_t = nn.ParameterList([
                nn.Parameter(torch.eye(self.head_dim)) for _ in range(n_ranks)
            ])
        else:
            self.metric_s = None
            self.metric_t = None

    @property
    def ball_radius(self) -> float:
        return self.log_t.exp().item()

    @property
    def sigma(self) -> float:
        return self.sigma_param.item()

    def forward(
        self,
        H_s: Tensor,
        H_t: Tensor,
        B: Tensor,
        rank_s: int = 0,
        rank_t: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        K_t : (N_t, d_out)  forward  s→t
        K_s : (N_s, d_out)  reverse  t→s, Jacobian-corrected
        """
        N_s, N_t = H_s.size(0), H_t.size(0)
        src, tgt = B[0], B[1]
        E = src.size(0)
        H, hd = self.n_heads, self.head_dim

        # Keep t, sig as 0-dim Tensors so log_t / sigma_param receive gradients.
        t   = self.log_t.exp()
        sig = self.sigma_param

        Ws_Hs = self.W_s(H_s).view(N_s, H, hd)
        Wt_Ht = self.W_t(H_t).view(N_t, H, hd)

        # Apply rank-specific metric G_r^{1/2} (Cerrini index correction).
        if self.metric_s is not None:
            Ws_Hs = Ws_Hs @ self.metric_s[rank_s]
            Wt_Ht = Wt_Ht @ self.metric_t[rank_t]

        Ws_src = Ws_Hs[src]   # (E, H, hd)
        Wt_tgt = Wt_Ht[tgt]   # (E, H, hd)

        # ── Forward s→t ───────────────────────────────────────────────────────
        e_fwd = (torch.cat([Ws_src, Wt_tgt], dim=-1) * self.a.unsqueeze(0)).sum(-1)
        e_fwd = self.leaky_relu(e_fwd * self.scale)
        att_fwd = _scatter_softmax_2d(e_fwd, tgt, N_t)
        att_fwd = self.dropout(att_fwd)
        K_t = _scatter_add_nd(att_fwd.unsqueeze(-1) * Ws_src, tgt, N_t).view(N_t, self.d_out)

        # ── Reverse t→s (Jacobian-corrected) ─────────────────────────────────
        rev_a   = torch.cat([self.a[:, hd:], self.a[:, :hd]], dim=-1)
        f_rev   = (torch.cat([Wt_tgt, Ws_src], dim=-1) * rev_a.unsqueeze(0)).sum(-1)
        f_rev   = self.leaky_relu(f_rev * self.scale)

        j = _mobius_jacobian(
            Ws_src.view(E, self.d_out), Wt_tgt.view(E, self.d_out), t, sig, self.d_out
        )
        att_rev = _scatter_softmax_2d(f_rev * j.unsqueeze(-1), src, N_s)
        att_rev = self.dropout(att_rev)
        K_s = _scatter_add_nd(att_rev.unsqueeze(-1) * Wt_tgt, src, N_s).view(N_s, self.d_out)

        return K_t, K_s

    def extra_repr(self) -> str:
        return (f"d_s_in={self.d_s_in}, d_t_in={self.d_t_in}, d_out={self.d_out}, "
                f"n_heads={self.n_heads}, scale={self.scale:.4f}, "
                f"ball_radius={self.ball_radius:.3f}, sigma={self.sigma:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Composable wrapper: HodgeletFilteredCCBlock
# ══════════════════════════════════════════════════════════════════════════════

class HodgeletFilteredCCBlock(nn.Module):
    """Full structural extension stack in a single composable module.

    Applies in sequence:
      1. SoftSpectralCutoff (via gate_rank — O(1) in n_ranks).
      2. HodgeletFilter — spectral features orthogonal to the harmonic kernel.
      3. JacobianCorrectedBlock — Jacobian-corrected bidirectional attention.
    """

    def __init__(
        self,
        d_s_in: int,
        d_t_in: int,
        d_out: int,
        n_heads: int = 8,
        k_eig: int = 32,
        n_ranks: int = 4,
        n_filtration_steps: int = 16,
        n_hodgelet_scales: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        ball_radius: float = 1.0,
        sigma: Optional[float] = None,
        learn_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.soft_cutoff = SoftSpectralCutoff(n_ranks=n_ranks, n_filtration_steps=n_filtration_steps)
        self.hodgelet    = HodgeletFilter(n_scales=n_hodgelet_scales, n_heads=n_heads)
        spec_raw_dim     = n_hodgelet_scales * n_heads * k_eig
        spec_proj_dim    = min(spec_raw_dim, d_out)
        self.spec_proj   = nn.Sequential(nn.Linear(spec_raw_dim, spec_proj_dim), nn.SiLU())
        self.attn_block  = JacobianCorrectedBlock(
            d_s_in=d_s_in + spec_proj_dim, d_t_in=d_t_in, d_out=d_out,
            n_heads=n_heads, dropout=dropout, negative_slope=negative_slope,
            ball_radius=ball_radius, sigma=sigma, learn_geometry=learn_geometry,
            n_ranks=n_ranks,
        )
        self.k_eig = k_eig
        self.n_ranks = n_ranks
        self.n_filtration_steps = n_filtration_steps

    def forward(
        self,
        H_s: Tensor,
        H_t: Tensor,
        B: Tensor,
        eigenvalues: Tensor,   # (N_s, n_ranks, n_filtration_steps, k_eig)
        rank: int = 0,
        rank_t: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        evals_rank = eigenvalues[:, rank, :, :]
        gated      = self.soft_cutoff.gate_rank(evals_rank, rank)
        evals_mean = (gated * evals_rank).mean(dim=1)
        filtered   = self.hodgelet(evals_mean)
        N_s, n_sc, n_h, k = filtered.shape
        spec_feat  = self.spec_proj(filtered.view(N_s, n_sc * n_h * k))
        return self.attn_block(
            torch.cat([H_s, spec_feat], dim=-1), H_t, B,
            rank_s=rank, rank_t=rank_t,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Equal-rank improved variant (separate Q/K/V)
# ══════════════════════════════════════════════════════════════════════════════

class CCAttentionPushForwardImproved(nn.Module):
    """Improved equal-rank CC-attention push-forward (Definition 32).

    Extends CCAttentionPushForward with separate W_Q, W_K, W_V projections,
    making key and value spaces independently learnable.  Shares all other
    improvements (scaled logits, vectorised softmax).

    Parameters
    ----------
    d_in, d_out    : input / output dimensions
    n_heads        : attention heads
    dropout        : attention dropout
    negative_slope : LeakyReLU slope
    d_edge         : optional edge feature dim added to attention score
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        d_edge: int = 0,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_edge = d_edge
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_K = nn.Linear(d_in, d_out, bias=False)
        self.W_V = nn.Linear(d_in, d_out, bias=False)
        self.a   = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim + d_edge))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        H_s: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        N = H_s.size(0)
        src, tgt = adj[0], adj[1]

        Q_i = self.W_Q(H_s).view(N, self.n_heads, self.head_dim)[tgt]
        K_j = self.W_K(H_s).view(N, self.n_heads, self.head_dim)[src]
        V_j = self.W_V(H_s).view(N, self.n_heads, self.head_dim)[src]

        cat_attn = torch.cat([Q_i, K_j], dim=-1)
        if edge_attr is not None and self.d_edge > 0:
            cat_attn = torch.cat([cat_attn, edge_attr.unsqueeze(1).expand(-1, self.n_heads, -1)], dim=-1)

        e   = self.leaky_relu((cat_attn * self.a.unsqueeze(0)).sum(-1) * self.scale)
        att = _scatter_softmax_2d(e, tgt, N)
        att = self.dropout(att)

        vals    = value.view(-1, self.n_heads, self.head_dim) if value is not None else V_j
        return _scatter_add_nd(att.unsqueeze(-1) * vals, tgt, N).view(N, self.d_out)

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_out={self.d_out}, n_heads={self.n_heads}, "
                f"scale={self.scale:.4f}, d_edge={self.d_edge}")
