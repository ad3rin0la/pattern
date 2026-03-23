"""
tope_residual.py
================
Deep Residual Learning for ToPE
================================
Implements three complementary residual mechanisms derived from the ToPE
architecture analysis:
  1. IntraRankResidualBlock
       Wraps the β update function in Definition 34 (Hajij et al.) so that
       cochain updates are additive corrections rather than full replacements.
       Drop-in replacement for the update step inside CCAttentionBlock.
  2. InterRankResidualMergeNode
       Residual-augmented merge node for rank-lifting via incidence matrices
       B_{r,r+1}. The lower-rank cochain is projected into the higher-rank
       space and added to the merge-node output, so rank lifts are incremental
       refinements rather than independent transformations.
       This is the allostery-critical residual: Zone-3 (>20 Å) mutations at
       rank 4 (interface) propagate back as corrections to rank 2 (residue)
       and rank 1 (bond) via the bidirectional variant.
  3. SpectralResidualTTN
       Wraps MultiParameterTTN so each spectral band (filtration level Λ_r)
       contributes an incremental correction to a running representation.
       Concretely: rep(Λ_r) = rep(Λ_{r-1}) + ΔF(band_r).
       Designed to absorb Phase-5B Pulay gradients as a fourth additive
       correction on top of the VOIP + geometry bands.
Design principles
-----------------
* All skip connections use LayerNorm-then-add (Pre-LN ResNet style) which is
  more stable than Post-LN for deep stacks.
* Projection shortcuts (for dimension mismatches) are 1×1 linear layers
  initialised near-identity where possible (kaiming_uniform with small gain).
* The inter-rank bidirectional variant reuses the reversed attention vector
  already present in CCAttentionBlock — no new parameters for the skip.
* SpectralResidualTTN is backward-compatible: if no Pulay band is provided,
  it falls back to the two-band (spatial + VOIP) original.
Dependencies
------------
  torch, torch.nn
  tope.models.cc_attention  (CCAttentionBlock, AttentionMergeNode)
  tope.topology.ttn_persistent_homology  (MultiParameterTTN, TTNPHConfig)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_projection(d_in: int, d_out: int) -> nn.Module:
    """1×1 linear projection for dimension-mismatched skip connections.

    Initialised with small gain so the shortcut starts near-zero and
    the main path dominates early in training (mirrors ResNet v2 behaviour).
    """
    proj = nn.Linear(d_in, d_out, bias=False)
    nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5), nonlinearity="linear")
    proj.weight.data.mul_(0.1)  # near-zero init
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intra-rank residual block
# ─────────────────────────────────────────────────────────────────────────────

class IntraRankResidualBlock(nn.Module):
    """Residual wrapper for the β update function (Definition 34, eq. 17).

    Replaces
        h^(l+1)_x = β(h^(l)_x, m_x)
    with
        h^(l+1)_x = h^(l)_x + F(LN(h^(l)_x), m_x)

    where F is a two-layer MLP with a gate on the aggregated message m_x,
    and LN is Pre-LayerNorm for training stability.

    This is the straightforward ResNet inductive bias applied to per-cell
    cochain feature updates.  It allows arbitrarily deep stacking of CC
    message-passing layers without degradation.

    Parameters
    ----------
    d_cell : int
        Feature dimension of the cell cochain (d_h).
    d_msg  : int
        Feature dimension of the aggregated message m_x.
        If d_msg != d_cell a learned projection aligns them before the add.
    hidden_mul : float
        Hidden-layer width multiplier relative to d_cell.  Default 2.
    dropout : float
        Dropout applied inside F.
    """

    def __init__(
        self,
        d_cell: int,
        d_msg: int,
        hidden_mul: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_cell = d_cell
        self.d_msg = d_msg
        d_h = int(d_cell * hidden_mul)

        # Pre-LN on cell features
        self.norm_cell = nn.LayerNorm(d_cell)
        # Pre-LN on message
        self.norm_msg = nn.LayerNorm(d_msg)

        # Project message into cell space if dimensions differ
        if d_msg != d_cell:
            self.msg_proj = nn.Linear(d_msg, d_cell, bias=False)
        else:
            self.msg_proj = nn.Identity()

        # F: two-layer MLP operating on [normed_cell || projected_msg]
        self.F = nn.Sequential(
            nn.Linear(d_cell + d_cell, d_h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_cell),
        )

        # Gating: learned scalar per cell dim controlling message influence
        self.gate = nn.Sequential(
            nn.Linear(d_cell, d_cell),
            nn.Sigmoid(),
        )

        # Output projection if input/output dims differ (never needed here,
        # but kept for symmetry with InterRankResidualMergeNode)
        self.out_proj: nn.Module = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init last linear of F → identity shortcut at initialisation
        nn.init.zeros_(self.F[-1].weight)
        nn.init.zeros_(self.F[-1].bias)

    def forward(self, h: Tensor, m: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h : (N, d_cell)   current cell features at layer l
        m : (N, d_msg)    aggregated inter-neighborhood message m_x

        Returns
        -------
        h_new : (N, d_cell)
        """
        # Normalise inputs
        h_n = self.norm_cell(h)
        m_p = self.msg_proj(self.norm_msg(m))

        # Compute residual correction
        combined = torch.cat([h_n, m_p], dim=-1)   # (N, 2*d_cell)
        delta = self.F(combined)                    # (N, d_cell)
        gate  = self.gate(h_n)                      # (N, d_cell)
        delta = gate * delta                        # gated correction

        return h + delta   # skip connection


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inter-rank residual merge node
# ─────────────────────────────────────────────────────────────────────────────

class InterRankResidualMergeNode(nn.Module):
    """Residual-augmented rank-lifting merge node (forward pass).

    For a rank-lift from C^r → C^{r+1} via incidence matrix B_{r,r+1}:
        H^{r+1}_new = MergeNode(H^r, H^{r+1})  +  P_{r→r+1}(H^r_pooled)

    where P_{r→r+1} is a learned projection and H^r_pooled is the
    incidence-weighted mean of rank-r features in each rank-(r+1) cell.

    Physical motivation
    -------------------
    Allostery is a perturbation on top of local structure.  A mutation at
    rank 4 (interface, >20 Å) *changes* rank-2 (residue) features, not
    replaces them.  The skip connection enforces this interpretation:
    the MergeNode learns the *correction* ΔH^{r+1}, while the shortcut
    carries the baseline rank-r information forward.

    Bidirectional variant (use_bidirectional=True)
    ----------------------------------------------
    Also computes the reverse skip: the aggregated rank-(r+1) signal is
    projected back and added to H^r as a downward correction.  This is the
    mechanism by which interface (rank-4) changes propagate to bond (rank-1)
    features — the critical path for allosteric mutant detection.

    Parameters
    ----------
    d_r    : int   feature dim of source (lower) rank
    d_r1   : int   feature dim of target (higher) rank
    d_out  : int   output feature dim (both ranks in bidirectional mode)
    n_heads: int   attention heads passed to the inner merge attention
    use_bidirectional : bool   if True, also compute reverse skip
    dropout: float
    """

    def __init__(
        self,
        d_r: int,
        d_r1: int,
        d_out: int,
        n_heads: int = 4,
        use_bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_r   = d_r
        self.d_r1  = d_r1
        self.d_out = d_out
        self.use_bidirectional = use_bidirectional

        # ── Core merge attention (forward: r → r+1) ──────────────────────────
        # Inlined lightweight attention to avoid circular imports with
        # cc_attention.py (CCAttentionBlock / AttentionMergeNode).
        assert d_out % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_out // n_heads

        self.W_r  = nn.Linear(d_r,  d_out, bias=False)
        self.W_r1 = nn.Linear(d_r1, d_out, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ── Skip projections ──────────────────────────────────────────────────
        # Forward skip: pool rank-r features per rank-(r+1) cell, project up
        self.skip_fwd = _make_projection(d_r, d_out)
        self.norm_fwd = nn.LayerNorm(d_out)

        # Reverse skip: pool rank-(r+1) features per rank-r cell, project down
        if use_bidirectional:
            self.skip_rev = _make_projection(d_r1, d_out)
            self.norm_rev = nn.LayerNorm(d_out)
            # Projection from d_r to d_out for the reverse residual base
            if d_r != d_out:
                self.base_proj_r = _make_projection(d_r, d_out)
            else:
                self.base_proj_r = nn.Identity()

        # Project d_r1 to d_out for the forward residual base
        if d_r1 != d_out:
            self.base_proj_r1 = _make_projection(d_r1, d_out)
        else:
            self.base_proj_r1 = nn.Identity()

    # ── Internal attention-weighted aggregation ───────────────────────────────

    def _agg(
        self,
        src: Tensor,          # (N_src, d_out)   projected source
        tgt: Tensor,          # (N_tgt, d_out)   projected target
        s_idx: Tensor,        # (E,) source indices
        t_idx: Tensor,        # (E,) target indices
        N_tgt: int,
        forward: bool = True,
    ) -> Tensor:
        """Single-direction attention-weighted scatter aggregation."""
        src_j = src.view(-1, self.n_heads, self.head_dim)[s_idx]  # (E,H,hd)
        tgt_i = tgt.view(-1, self.n_heads, self.head_dim)[t_idx]  # (E,H,hd)

        if forward:
            e = (torch.cat([src_j, tgt_i], dim=-1) * self.attn_vec.unsqueeze(0)).sum(-1)
        else:
            rev = torch.cat([self.attn_vec[:, self.head_dim:],
                             self.attn_vec[:, :self.head_dim]], dim=-1)
            e = (torch.cat([tgt_i, src_j], dim=-1) * rev.unsqueeze(0)).sum(-1)

        e   = self.leaky(e)
        att = self._softmax_scatter(e, t_idx if forward else s_idx,
                                    N_tgt)           # (E, H)
        att = self.drop(att)

        vals    = src_j if forward else tgt_i        # (E, H, hd)
        weighted = att.unsqueeze(-1) * vals           # (E, H, hd)

        N = N_tgt
        idx_scatter = (t_idx if forward else s_idx)
        out = torch.zeros(N, self.n_heads, self.head_dim,
                          device=src.device, dtype=src.dtype)
        expand_idx = idx_scatter.unsqueeze(1).unsqueeze(2).expand_as(weighted)
        out.scatter_add_(0, expand_idx, weighted)
        return out.view(N, self.d_out)

    @staticmethod
    def _softmax_scatter(e: Tensor, idx: Tensor, N: int) -> Tensor:
        """Per-target-node softmax over edges."""
        e_max = torch.zeros(N, e.size(1), device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, idx.unsqueeze(1).expand_as(e), e, reduce="amax",
                              include_self=True)
        e_exp = (e - e_max[idx]).exp()
        denom = torch.zeros_like(e_max)
        denom.scatter_add_(0, idx.unsqueeze(1).expand_as(e_exp), e_exp)
        denom = denom.clamp(min=1e-8)
        return e_exp / denom[idx]

    # ── Pooled skip signal ────────────────────────────────────────────────────

    @staticmethod
    def _pool(feat: Tensor, gather_idx: Tensor, scatter_idx: Tensor, N_out: int) -> Tensor:
        """
        For each cell in the target (scatter_idx), average the features of
        all source cells connected to it (gather_idx selects into feat).
        """
        src_feat = feat[gather_idx]                  # (E, d)
        d = feat.size(1)
        out   = torch.zeros(N_out, d, device=feat.device, dtype=feat.dtype)
        count = torch.zeros(N_out, 1, device=feat.device, dtype=feat.dtype)
        out.scatter_add_(0, scatter_idx.unsqueeze(1).expand(-1, d), src_feat)
        count.scatter_add_(0, scatter_idx.unsqueeze(1),
                           torch.ones(scatter_idx.size(0), 1, device=feat.device))
        return out / count.clamp(min=1.0)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        H_r:  Tensor,   # (N_r,  d_r)   rank-r  cochain
        H_r1: Tensor,   # (N_r1, d_r1)  rank-r+1 cochain
        B: Tensor,      # (2, E) incidence: B[0]=rank-r idx, B[1]=rank-(r+1) idx
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns
        -------
        H_r1_new : (N_r1, d_out)   updated rank-(r+1) cochain
        H_r_new  : (N_r,  d_out) | None   updated rank-r cochain (bidirectional only)
        """
        N_r  = H_r.size(0)
        N_r1 = H_r1.size(0)
        s_idx = B[0]   # rank-r  indices
        t_idx = B[1]   # rank-r1 indices

        # Project both cochains to d_out
        Wr_h  = self.W_r(H_r)    # (N_r,  d_out)
        Wr1_h = self.W_r1(H_r1)  # (N_r1, d_out)

        # ── Forward: r → r+1 (main merge-node attention) ─────────────────────
        merge_fwd = self._agg(Wr_h, Wr1_h, s_idx, t_idx, N_r1, forward=True)

        # Forward skip: pool rank-r features per rank-(r+1) cell
        pooled_r = self._pool(H_r, s_idx, t_idx, N_r1)   # (N_r1, d_r)
        skip_r   = self.skip_fwd(pooled_r)                # (N_r1, d_out)

        # Residual add with Pre-LN on the skip signal
        H_r1_new = self.base_proj_r1(H_r1) + self.norm_fwd(merge_fwd + skip_r)

        # ── Reverse: r+1 → r (bidirectional skip) ────────────────────────────
        H_r_new = None
        if self.use_bidirectional:
            merge_rev = self._agg(Wr_h, Wr1_h, s_idx, t_idx, N_r, forward=False)

            # Reverse skip: pool rank-(r+1) features per rank-r cell
            pooled_r1  = self._pool(H_r1, t_idx, s_idx, N_r)  # (N_r, d_r1)
            skip_r1    = self.skip_rev(pooled_r1)              # (N_r, d_out)

            H_r_new = self.base_proj_r(H_r) + self.norm_rev(merge_rev + skip_r1)

        return H_r1_new, H_r_new


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spectral residual TTN
# ─────────────────────────────────────────────────────────────────────────────

class SpectralResidualTTN(nn.Module):
    """Spectral residual stack over TTN filtration bands.

    Instead of computing a monolithic TTN output, each spectral band
    contributes an incremental correction to a running representation:
        rep(Λ_0)    = BandEncoder_0(spatial_band)
        rep(Λ_1)    = rep(Λ_0)  + BandEncoder_1(voip_band)
        rep(Λ_2)    = rep(Λ_1)  + BandEncoder_2(pulay_band)   [Phase 5B]

    This wavelet-like decomposition ensures:
    - Coarse features (low Λ, spatial) are established first
    - Electronic features are additive corrections on top
    - Pulay forces (Phase 5B) are a fourth additive correction
    - The filtration parameter Λ_r is continuous and differentiable through
      the residual path

    Each BandEncoder is a two-layer MLP with Pre-LN residual gating.
    The final representation is the cumulative sum across bands.

    Parameters
    ----------
    d_spatial    : int   output dim of spatial TTN leaf (k_eigenvalues * n_zones)
    d_voip       : int   output dim of VOIP TTN leaf
    d_out        : int   desired output dimension of the fused representation
    d_pulay      : int | None   dim of Pulay gradient features (Phase 5B).
                   If None, the Pulay band is skipped.
    hidden_mul   : float   hidden width multiplier for BandEncoders
    dropout      : float
    """

    def __init__(
        self,
        d_spatial:  int,
        d_voip:     int,
        d_out:      int,
        d_pulay:    Optional[int] = None,
        hidden_mul: float = 2.0,
        dropout:    float = 0.0,
    ) -> None:
        super().__init__()
        self.d_out   = d_out
        self.d_pulay = d_pulay

        # Band 0: spatial (base representation)
        self.band0 = _BandEncoder(d_spatial, d_out, hidden_mul, dropout)

        # Band 1: VOIP correction on top of spatial
        self.band1 = _BandEncoder(d_voip, d_out, hidden_mul, dropout)
        self.gate1 = _ResidualGate(d_out)

        # Band 2 (optional): Pulay gradient correction (Phase 5B)
        if d_pulay is not None:
            self.band2 = _BandEncoder(d_pulay, d_out, hidden_mul, dropout)
            self.gate2 = _ResidualGate(d_out)
        else:
            self.band2 = None
            self.gate2 = None

        # Final LayerNorm over the accumulated representation
        self.norm_out = nn.LayerNorm(d_out)

    def forward(
        self,
        spatial_feats: Tensor,              # (B, d_spatial)
        voip_feats:    Tensor,              # (B, d_voip)
        pulay_feats:   Optional[Tensor] = None,  # (B, d_pulay) or None
    ) -> Tensor:
        """
        Parameters
        ----------
        spatial_feats  : (batch, d_spatial)  output of TTN spatial branch
        voip_feats     : (batch, d_voip)     output of TTN VOIP branch
        pulay_feats    : (batch, d_pulay)    Pulay gradient features (Phase 5B)

        Returns
        -------
        rep : (batch, d_out)   accumulated spectral representation
        """
        # Band 0: coarse spatial base
        rep = self.band0(spatial_feats)              # (B, d_out)

        # Band 1: electronic VOIP correction
        delta1 = self.band1(voip_feats)              # (B, d_out)
        rep    = rep + self.gate1(rep, delta1)       # gated residual add

        # Band 2: Pulay gradient correction (Phase 5B)
        if self.band2 is not None and pulay_feats is not None:
            delta2 = self.band2(pulay_feats)         # (B, d_out)
            rep    = rep + self.gate2(rep, delta2)

        return self.norm_out(rep)

    def get_band_contributions(
        self,
        spatial_feats: Tensor,
        voip_feats:    Tensor,
        pulay_feats:   Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return per-band representation snapshots for ablation/interpretability."""
        rep0 = self.band0(spatial_feats)
        delta1 = self.band1(voip_feats)
        rep1   = rep0 + self.gate1(rep0, delta1)
        result = {
            "spatial_base":          rep0,
            "after_voip_correction": rep1,
        }
        if self.band2 is not None and pulay_feats is not None:
            delta2 = self.band2(pulay_feats)
            rep2   = rep1 + self.gate2(rep1, delta2)
            result["after_pulay_correction"] = rep2
        return result


# ── Sub-modules used by SpectralResidualTTN ───────────────────────────────────

class _BandEncoder(nn.Module):
    """Two-layer MLP that encodes a single spectral band into d_out space."""

    def __init__(self, d_in: int, d_out: int, hidden_mul: float, dropout: float) -> None:
        super().__init__()
        d_h = int(d_out * hidden_mul)
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_out),
        )
        # Zero-init output so band starts as zero correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _ResidualGate(nn.Module):
    """Learned gate scalar that controls how much of a delta is added.

    g = sigmoid(W_rep · rep + W_delta · delta)
    out = g * delta
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.W_rep   = nn.Linear(d, d, bias=False)
        self.W_delta = nn.Linear(d, d, bias=False)
        self.bias    = nn.Parameter(torch.zeros(d))

    def forward(self, rep: Tensor, delta: Tensor) -> Tensor:
        g = torch.sigmoid(self.W_rep(rep) + self.W_delta(delta) + self.bias)
        return g * delta


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full residual ToPE encoder stack
# ─────────────────────────────────────────────────────────────────────────────

class ToPERankStack(nn.Module):
    """Stacks N_layers of IntraRankResidualBlock per rank.

    Wraps the per-rank update loop so that:
      - Each rank has its own stack of intra-rank residual blocks
      - Between rank stacks, InterRankResidualMergeNode lifts features upward
        and (optionally) sends corrections downward

    Rank order:  0 (atoms) → 1 (bonds) → 2 (residues) → 3 (subunits) → 4 (interfaces)

    Parameters
    ----------
    rank_dims    : List[int]   feature dims per rank [d_0, d_1, d_2, d_3, d_4]
    d_out        : int         unified output dim after inter-rank lifts
    n_layers     : int         number of intra-rank residual layers per rank
    n_heads      : int         attention heads in inter-rank merge nodes
    bidirectional: bool        whether inter-rank merge nodes are bidirectional
    dropout      : float
    """

    def __init__(
        self,
        rank_dims:     List[int],
        d_out:         int,
        n_layers:      int = 3,
        n_heads:       int = 4,
        bidirectional: bool = True,
        dropout:       float = 0.0,
    ) -> None:
        super().__init__()
        assert len(rank_dims) == 5, "ToPE has 5 ranks (0-4)"
        self.rank_dims = rank_dims
        self.d_out     = d_out
        self.n_ranks   = len(rank_dims)

        # ── Intra-rank residual stacks (one per rank) ────────────────────────
        self.intra_stacks = nn.ModuleList()
        for d in rank_dims:
            stack = nn.ModuleList([
                IntraRankResidualBlock(d_cell=d, d_msg=d, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.intra_stacks.append(stack)

        # ── Inter-rank residual merge nodes ──────────────────────────────────
        # One per consecutive rank pair: (0,1), (1,2), (2,3), (3,4)
        self.inter_merges = nn.ModuleList([
            InterRankResidualMergeNode(
                d_r=rank_dims[r],
                d_r1=rank_dims[r + 1],
                d_out=d_out,
                n_heads=n_heads,
                use_bidirectional=bidirectional,
                dropout=dropout,
            )
            for r in range(self.n_ranks - 1)
        ])

        # Input projections: bring each rank to d_out before inter-rank merge
        self.in_projs = nn.ModuleList([
            nn.Linear(d, d_out) if d != d_out else nn.Identity()
            for d in rank_dims
        ])

    def forward(
        self,
        cochains:     List[Tensor],   # [H_0, H_1, H_2, H_3, H_4]
        incidences:   List[Tensor],   # [B_01, B_12, B_23, B_34]   (2,E) each
        adj_matrices: Optional[List[Tensor]] = None,  # intra-rank adj per rank
    ) -> List[Tensor]:
        """
        Parameters
        ----------
        cochains    : one (N_r, d_r) tensor per rank
        incidences  : one (2, E) sparse incidence index tensor per rank pair
        adj_matrices: one (2, E) same-rank adjacency per rank (for intra-rank
                      message passing); if None, intra-rank blocks run with
                      self-loop messages (m = h, no neighbourhood aggregation)

        Returns
        -------
        List[Tensor]  updated cochains, each (N_r, d_out)
        """
        H = list(cochains)  # mutable copy

        # ── 1. Intra-rank residual updates ────────────────────────────────────
        for r, stack in enumerate(self.intra_stacks):
            h = H[r]
            for block in stack:
                # If no adj provided, message = h itself (self-loop baseline)
                m = h if adj_matrices is None else self._gather_msg(
                    h, adj_matrices[r]
                )
                h = block(h, m)
            H[r] = h

        # ── 2. Inter-rank residual lifts ─────────────────────────────────────
        # Project all ranks to d_out first
        H_proj = [proj(h) for proj, h in zip(self.in_projs, H)]

        # Lift bottom-up, collecting downward corrections if bidirectional
        down_corrections: Dict[int, Tensor] = {}
        for r, merge in enumerate(self.inter_merges):
            H_r_in  = H[r]
            H_r1_in = H[r + 1]
            B       = incidences[r]

            H_r1_new, H_r_correction = merge(H_r_in, H_r1_in, B)
            H_proj[r + 1] = H_r1_new

            if H_r_correction is not None:
                # Accumulate downward correction; applied after full upward pass
                if r in down_corrections:
                    down_corrections[r] = down_corrections[r] + H_r_correction
                else:
                    down_corrections[r] = H_r_correction

        # Apply accumulated downward corrections
        for r, corr in down_corrections.items():
            H_proj[r] = H_proj[r] + corr

        return H_proj

    @staticmethod
    def _gather_msg(h: Tensor, adj: Tensor) -> Tensor:
        """Simple mean aggregation over adjacency for intra-rank message."""
        src, tgt = adj[0], adj[1]
        N, d = h.size()
        msg   = torch.zeros(N, d, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        msg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, d), h[src])
        count.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=h.device))
        return msg / count.clamp(min=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Test 1: IntraRankResidualBlock ────────────────────────────────────────
    print("=== 1. IntraRankResidualBlock ===")
    block = IntraRankResidualBlock(d_cell=64, d_msg=64).to(device)
    h = torch.randn(128, 64, device=device)
    m = torch.randn(128, 64, device=device)
    h_new = block(h, m)
    print(f"  Input shape : {h.shape}")
    print(f"  Output shape: {h_new.shape}")
    assert h_new.shape == h.shape
    print(f"  Max residual at init: {(h_new - h).abs().max().item():.6f}  (should be ~0)")

    # ── Test 2: InterRankResidualMergeNode ────────────────────────────────────
    print("\n=== 2. InterRankResidualMergeNode ===")
    merge = InterRankResidualMergeNode(
        d_r=64, d_r1=128, d_out=64, n_heads=4, use_bidirectional=True
    ).to(device)
    N_r, N_r1 = 500, 200
    H_r  = torch.randn(N_r,  64,  device=device)
    H_r1 = torch.randn(N_r1, 128, device=device)
    E = 600
    s_idx = torch.randint(0, N_r,  (E,), device=device)
    t_idx = torch.randint(0, N_r1, (E,), device=device)
    B = torch.stack([s_idx, t_idx])
    H_r1_new, H_r_new = merge(H_r, H_r1, B)
    print(f"  H_r  input : {H_r.shape}")
    print(f"  H_r1 input : {H_r1.shape}")
    print(f"  H_r1 output: {H_r1_new.shape}")
    print(f"  H_r  output: {H_r_new.shape}  (bidirectional correction)")

    # ── Test 3: SpectralResidualTTN ───────────────────────────────────────────
    print("\n=== 3. SpectralResidualTTN ===")
    ttn_res = SpectralResidualTTN(
        d_spatial=48,
        d_voip=48,
        d_out=128,
        d_pulay=32,
    ).to(device)
    B_size = 8
    spatial = torch.randn(B_size, 48, device=device)
    voip    = torch.randn(B_size, 48, device=device)
    pulay   = torch.randn(B_size, 32, device=device)
    rep = ttn_res(spatial, voip, pulay)
    print(f"  Spatial input: {spatial.shape}")
    print(f"  VOIP    input: {voip.shape}")
    print(f"  Pulay   input: {pulay.shape}")
    print(f"  Fused  output: {rep.shape}")
    bands = ttn_res.get_band_contributions(spatial, voip, pulay)
    for name, feat in bands.items():
        print(f"  Band '{name}': {feat.shape}")

    # ── Test 4: ToPERankStack ─────────────────────────────────────────────────
    print("\n=== 4. ToPERankStack (full pipeline) ===")
    stack = ToPERankStack(
        rank_dims=[64, 64, 64, 64, 64],
        d_out=64,
        n_layers=2,
        n_heads=4,
        bidirectional=True,
        dropout=0.1,
    ).to(device)
    sizes = [500, 520, 100, 4, 2]
    cochains = [torch.randn(n, 64, device=device) for n in sizes]
    incidences = []
    for r in range(4):
        n_r, n_r1 = sizes[r], sizes[r + 1]
        E = n_r1 * 3
        incidences.append(torch.stack([
            torch.randint(0, n_r,  (E,), device=device),
            torch.randint(0, n_r1, (E,), device=device),
        ]))
    out = stack(cochains, incidences)
    for r, o in enumerate(out):
        print(f"  Rank {r} output: {o.shape}")
    loss = sum(o.sum() for o in out)
    loss.backward()
    print(f"\n  Backward pass: OK")
    print(f"  Param count  : {sum(p.numel() for p in stack.parameters()):,}")
    print("\nAll tests passed.")
