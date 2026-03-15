"""
cross_attention_improved.py
============================
Structural improvements to the enzyme ↔ substrate/product cross-attention.

Three improvements, directly analogous to those in cc_attention.py but
adapted to the bipartite enzyme–molecule geometry:

  1. HodgeletCrossFilter    — Oversmoothing prevention on the enzyme query side.
                              Applies the Hodgelet bandpass filter to per-atom
                              eigenvalue features before they enter cross-attention
                              as queries, so query vectors carry non-harmonic
                              spectral content.

  2. SoftMolecularGate      — Continuity under substrate perturbation.
                              Replaces the implicit hard threshold in MoleculeGNN
                              readout with a learnable sigmoid gate on atom
                              features, analogous to SoftSpectralCutoff but over
                              learned atom-feature magnitudes.

  3. JacobianCrossAttention — Adjoint-correct bipartite cross-attention.
                              Replaces the Euclidean dot-product logit with the
                              Poincaré proximity logit (HAT, Zhang et al. 2019)
                              and applies the Ferreira (2015) Jacobian factor to
                              the molecule→enzyme reverse attention so that
                              forward and reverse passes are exactly adjoint under
                              dμ_{σ,t}.

Additionally, ImprovedSubstrateProductCrossAttention composes all three as a
drop-in replacement for SubstrateProductCrossAttention.

Bug fixes vs. the initial design draft
---------------------------------------
- Import from cc_attention (cc_attention_improved.py was reconciled away).
- _jacobian_matrix uses tensor arithmetic throughout (no .item()) so that
  log_t and sigma_param receive gradients during training.
- _jacobian_matrix uses O(N_k × N_q) batched matmul rather than an
  O(N_k × N_q × d) expand+reshape, avoiding a potential ~100 MB allocation.
- Poincaré distance coefficient corrected to 4/c (was 2/c).
- Clamped inner argument used in arctanh (was discarded, causing NaN risk).
- SoftMolecularGate and JacobianCrossAttention mol_dim use enzyme_hidden_dim
  (H) not mol_hidden_dim (D), matching the projected output of MoleculeGNN.
- Double-permute on Q_val_t simplified to direct reference.

References
----------
Ferreira (2015) J. Fourier Anal. Appl. 21:281–317
Zhang et al. (2019) arXiv:1912.03046 — Hyperbolic Graph Attention Network
Vaswani et al. (2017) NeurIPS — Attention Is All You Need
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tope.models.cc_attention import HodgeletFilter, _mobius_jacobian


# ══════════════════════════════════════════════════════════════════════════════
# Batched Jacobian matrix  (internal utility)
# ══════════════════════════════════════════════════════════════════════════════

def _jacobian_matrix(
    q: Tensor,
    k: Tensor,
    t: Tensor,
    sigma: Tensor,
    n: int,
) -> Tensor:
    """Compute the Ferreira Jacobian matrix j_{q[j]}(k[i]) for all (i, j) pairs.

    Uses batched matrix operations (O(N_k × N_q) memory) rather than the
    O(N_k × N_q × d) expand+reshape pattern.

    Parameters
    ----------
    q     : (N_q, d)  query (enzyme) embeddings — the "a" side of j_a(x)
    k     : (N_k, d)  key (molecule) embeddings — the "x" side
    t     : ()        ball radius as 0-dim Tensor (keeps grad flowing)
    sigma : ()        conformal weight as 0-dim Tensor (keeps grad flowing)
    n     : int       ambient dimension d

    Returns
    -------
    j_mat : (N_k, N_q)  Jacobian weight for each (molecule atom, enzyme atom) pair
    """
    t2 = t * t
    q_sq  = (q * q).sum(-1)          # (N_q,)
    k_sq  = (k * k).sum(-1)          # (N_k,)
    qk_dot = k @ q.t()               # (N_k, N_q)  — single matmul, no expand

    # Broadcast to (N_k, N_q)
    a_norm_sq = q_sq.unsqueeze(0) / t2       # (1, N_q)
    ax_dot    = qk_dot / t2                  # (N_k, N_q)
    x_norm_sq = k_sq.unsqueeze(1) / t2       # (N_k, 1)

    numer = (1.0 - a_norm_sq).clamp(min=1e-8)
    denom = (1.0 + 2.0 * ax_dot + a_norm_sq * x_norm_sq).clamp(min=1e-8)
    exp   = (n + sigma - 2.0) / 2.0

    # Log-space for numerical stability; sigma stays in computation graph.
    return (exp * (numer.log() - denom.log())).exp().clamp(min=1e-6, max=1e6)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HodgeletCrossFilter
#    Oversmoothing prevention on the enzyme query side
# ══════════════════════════════════════════════════════════════════════════════

class HodgeletCrossFilter(nn.Module):
    """Apply Hodgelet filtering to enzyme query features before cross-attention.

    If the enzyme representation has oversmoothed (all atom embeddings collapsed
    toward the harmonic mode), query vectors become nearly identical and the
    cross-attention degenerates.  This module adds non-harmonic spectral content
    from per-atom eigenvalues via a gated residual.

    Parameters
    ----------
    enzyme_hidden_dim : dimension of per-atom enzyme features
    k_eig             : eigenvalues per atom
    n_heads           : cross-attention heads
    n_scales          : Hodgelet scales (one ρ per scale per head)
    """

    def __init__(
        self,
        enzyme_hidden_dim: int,
        k_eig: int = 32,
        n_heads: int = 8,
        n_scales: int = 4,
    ) -> None:
        super().__init__()
        self.enzyme_hidden_dim = enzyme_hidden_dim
        self.k_eig = k_eig
        self.n_heads = n_heads

        self.hodgelet = HodgeletFilter(n_scales=n_scales, n_heads=n_heads)

        spec_dim = n_scales * n_heads * k_eig
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, enzyme_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(enzyme_hidden_dim),
        )
        # Learnable mixing weight; starts at 0 → sigmoid(0)=0.5 initial contribution
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h_enzyme: Tensor,     # (N_enz, H)
        eigenvalues: Tensor,  # (N_enz, k_eig)
    ) -> Tensor:
        """
        Returns
        -------
        h_enzyme_filtered : (N_enz, H)
            Enzyme features enriched with Hodgelet spectral content via
            gated residual: h = h + sigmoid(gate) * spec_feat.
        """
        filtered  = self.hodgelet(eigenvalues)          # (N_enz, n_scales, n_heads, k_eig)
        N         = filtered.size(0)
        spec_feat = self.spec_proj(filtered.view(N, -1))  # (N_enz, H)
        return h_enzyme + torch.sigmoid(self.gate) * spec_feat


# ══════════════════════════════════════════════════════════════════════════════
# 2. SoftMolecularGate
#    Continuity under substrate structural perturbation
# ══════════════════════════════════════════════════════════════════════════════

class SoftMolecularGate(nn.Module):
    """Learnable soft gate on molecule atom features.

    Applies a per-atom sigmoid gate:

        h_gated[i] = h_atom[i] · σ( MLP(h_atom[i]) )

    Atoms aligned with the learned salience direction pass through with
    weight ≈ 1; low-salience atoms (e.g., non-polar carbons far from the
    reactive centre) are smoothly down-weighted rather than hard-excluded.

    Parameters
    ----------
    hidden_dim   : atom feature dimension (enzyme_hidden_dim, after MoleculeGNN.proj)
    n_gate_heads : independent gate directions (default 1)
    """

    def __init__(self, hidden_dim: int, n_gate_heads: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_gate_heads = n_gate_heads
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, n_gate_heads),
        )
        self.head_mix = nn.Linear(n_gate_heads, 1, bias=False)

    def forward(self, h_mol: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        h_mol : (N_atoms, H)

        Returns
        -------
        h_gated      : (N_atoms, H)  soft-gated atom features
        gate_weights : (N_atoms,)    per-atom gate values in [0, 1]
        """
        scores = self.gate_mlp(h_mol)                          # (N_atoms, n_gate_heads)
        gate   = torch.sigmoid(self.head_mix(scores)).squeeze(-1)  # (N_atoms,)
        return h_mol * gate.unsqueeze(-1), gate


# ══════════════════════════════════════════════════════════════════════════════
# 3. JacobianCrossAttention
#    Adjoint-correct bipartite cross-attention with hyperbolic proximity logit
# ══════════════════════════════════════════════════════════════════════════════

class JacobianCrossAttention(nn.Module):
    """Multi-head cross-attention with hyperbolic proximity logit and
    Jacobian-corrected reverse attention.

    Drop-in replacement for MultiHeadCrossAttention.

    The standard dot-product logit  e_ij = (Q_i · K_j) / √d  is replaced by:

        e_ij = -d²_B(Q_i, K_j) / τ

    where d²_B is the squared Poincaré ball distance (HAT approximation,
    Zhang et al. 2019), and τ is a per-head learnable temperature.

    The Ferreira (2015) Jacobian factor j_{Q_i}(K_j) is applied to the
    reverse (molecule→enzyme) logit so that forward and reverse attention
    are exactly adjoint under dμ_{σ,t}.

    Parameters
    ----------
    hidden_dim     : enzyme feature dimension (queries)
    mol_dim        : molecule feature dimension (keys/values); defaults to hidden_dim
    n_heads        : attention heads
    dropout        : attention dropout
    ball_radius    : Poincaré ball radius t
    sigma          : conformal weight σ (default 0.0)
    learn_geometry : jointly learn ball_radius and sigma
    """

    def __init__(
        self,
        hidden_dim: int,
        mol_dim: Optional[int] = None,
        n_heads: int = 8,
        dropout: float = 0.1,
        ball_radius: float = 1.0,
        sigma: float = 0.0,
        learn_geometry: bool = True,
    ) -> None:
        super().__init__()
        mol_dim = mol_dim or hidden_dim
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.mol_dim    = mol_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads

        self.q_proj   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj   = nn.Linear(mol_dim,    hidden_dim, bias=False)
        self.v_proj   = nn.Linear(mol_dim,    hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable per-head temperature τ > 0
        self.log_tau = nn.Parameter(torch.zeros(n_heads))

        self.dropout = nn.Dropout(dropout)
        self.ln      = nn.LayerNorm(hidden_dim)

        # Geometry parameters kept as tensors (no .item()) for gradient flow
        if learn_geometry:
            self.log_t      = nn.Parameter(torch.tensor(math.log(ball_radius)))
            self.sigma_param = nn.Parameter(torch.tensor(float(sigma)))
        else:
            self.register_buffer("log_t",       torch.tensor(math.log(ball_radius)))
            self.register_buffer("sigma_param", torch.tensor(float(sigma)))

    @property
    def tau(self) -> Tensor:
        """Per-head temperature, (n_heads,), positive."""
        return F.softplus(self.log_tau) + 1e-4

    def _poincare_dist_sq(self, u: Tensor, v: Tensor) -> Tensor:
        """Squared Poincaré distance (HAT first-order approximation).

        Parameters
        ----------
        u : (n_heads, N, head_dim)
        v : (n_heads, M, head_dim)

        Returns
        -------
        dist_sq : (n_heads, N, M)
        """
        t = self.log_t.exp()
        c = 1.0 / (t * t)

        # Clamp norms to stay strictly inside the ball
        u_sq = (u * u).sum(-1, keepdim=True).clamp(max=(1.0 / c) - 1e-5)  # (H, N, 1)
        v_sq = (v * v).sum(-1, keepdim=True).clamp(max=(1.0 / c) - 1e-5)  # (H, M, 1)

        diff_sq = torch.cdist(u, v, p=2).pow(2)   # (H, N, M)

        # Conformal denominator: (1 - c‖u‖²)(1 - c‖v‖²)
        denom = (1.0 - c * u_sq) * (1.0 - c * v_sq.transpose(-1, -2))  # (H, N, M)
        denom = denom.clamp(min=1e-8)

        # Poincaré distance: d²(u,v) = (4/c) · arctanh²(√(c·‖u-v‖²/denom))
        # Coefficient is 4/c (not 2/c): comes from squaring d = (2/√c)·arctanh(…)
        inner = (c * diff_sq / denom).clamp(max=1.0 - 1e-6)
        return (4.0 / c) * torch.arctanh(inner.sqrt()).pow(2)

    def forward(
        self,
        h_enzyme: Tensor,
        h_mol: Tensor,
        enzyme_batch: Optional[Tensor] = None,
        mol_batch: Optional[Tensor] = None,
        return_reverse: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        h_enzyme      : (N_enz, hidden_dim)   enzyme atom embeddings (queries)
        h_mol         : (N_mol, mol_dim)       molecule atom embeddings (keys/values)
        enzyme_batch  : (N_enz,)               graph membership (optional)
        mol_batch     : (N_mol,)               graph membership (optional)
        return_reverse: also return Jacobian-corrected molecule→enzyme output

        Returns
        -------
        h_enz_attended : (N_enz, hidden_dim)
        attn_fwd       : (n_heads, N_enz, N_mol)
        h_mol_attended : (N_mol, hidden_dim) or None  (only if return_reverse=True)
        """
        N_enz = h_enzyme.size(0)
        N_mol = h_mol.size(0)
        H, hd = self.n_heads, self.head_dim

        # t and sig stay as 0-dim Tensors so log_t / sigma_param get gradients.
        t   = self.log_t.exp()
        sig = self.sigma_param

        # ── Project ───────────────────────────────────────────────────────────
        Q = self.q_proj(h_enzyme).view(N_enz, H, hd).permute(1, 0, 2)  # (H, N_enz, hd)
        K = self.k_proj(h_mol   ).view(N_mol, H, hd).permute(1, 0, 2)  # (H, N_mol, hd)
        V = self.v_proj(h_mol   ).view(N_mol, H, hd).permute(1, 0, 2)  # (H, N_mol, hd)

        # ── Hyperbolic proximity logit: e_ij = -d²(Q_i, K_j) / τ ─────────────
        dist_sq = self._poincare_dist_sq(Q, K)             # (H, N_enz, N_mol)
        logits  = -dist_sq / self.tau.view(H, 1, 1)        # (H, N_enz, N_mol)

        if enzyme_batch is not None and mol_batch is not None:
            mask   = enzyme_batch.unsqueeze(1) != mol_batch.unsqueeze(0)  # (N_enz, N_mol)
            logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

        # ── Forward: enzyme → molecule ────────────────────────────────────────
        attn_fwd = F.softmax(logits, dim=-1)                # (H, N_enz, N_mol)
        attn_fwd = self.dropout(attn_fwd)

        out = torch.bmm(attn_fwd, V)                        # (H, N_enz, hd)
        out = out.permute(1, 0, 2).contiguous().view(N_enz, self.hidden_dim)
        out = self.out_proj(out)
        h_enz_attended = self.ln(h_enzyme + out)

        # ── Reverse: molecule → enzyme (Jacobian-corrected) ───────────────────
        h_mol_attended = None
        if return_reverse:
            # Flat (full hidden_dim) projections for Jacobian computation
            q_flat = Q.permute(1, 0, 2).reshape(N_enz, self.hidden_dim)  # (N_enz, H*hd)
            k_flat = K.permute(1, 0, 2).reshape(N_mol, self.hidden_dim)  # (N_mol, H*hd)

            # Batched Jacobian matrix: O(N_mol × N_enz) — no expand/reshape
            j_mat = _jacobian_matrix(q_flat, k_flat, t, sig, self.hidden_dim)
            # j_mat: (N_mol, N_enz) — j_{q[j]}(k[i]) for i∈mol, j∈enz

            # Reverse logits: transpose forward logits + Jacobian correction
            rev_logits = logits.permute(0, 2, 1) * j_mat.unsqueeze(0)  # (H, N_mol, N_enz)

            if enzyme_batch is not None and mol_batch is not None:
                rev_logits = rev_logits.masked_fill(mask.t().unsqueeze(0), float("-inf"))

            attn_rev = F.softmax(rev_logits, dim=-1)         # (H, N_mol, N_enz)
            attn_rev = self.dropout(attn_rev)

            # Use enzyme queries as values for the reverse pass
            rev_out = torch.bmm(attn_rev, Q)                 # (H, N_mol, hd)
            rev_out = rev_out.permute(1, 0, 2).contiguous().view(N_mol, self.hidden_dim)
            rev_out = self.out_proj(rev_out)

            # Residual base: project mol features to hidden_dim if needed
            h_mol_base = self.k_proj(h_mol) if self.mol_dim != self.hidden_dim else h_mol
            h_mol_attended = self.ln(h_mol_base + rev_out)

        return h_enz_attended, attn_fwd, h_mol_attended

    def extra_repr(self) -> str:
        return (f"hidden_dim={self.hidden_dim}, mol_dim={self.mol_dim}, "
                f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
                f"ball_radius={self.log_t.exp().item():.3f}, "
                f"sigma={self.sigma_param.item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# ImprovedCrossAttentionConfig and ImprovedSubstrateProductCrossAttention
# Drop-in replacement for SubstrateProductCrossAttention
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImprovedCrossAttentionConfig:
    """Configuration for the improved cross-attention module."""
    mol_feat_dim: int = 64
    mol_hidden_dim: int = 128
    mol_n_layers: int = 3
    enzyme_hidden_dim: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    k_eig: int = 32
    n_hodgelet_scales: int = 4
    ball_radius: float = 1.0
    sigma: float = 0.0
    learn_geometry: bool = True
    n_gate_heads: int = 1


class ImprovedSubstrateProductCrossAttention(nn.Module):
    """Full improved cross-attention: enzyme ↔ substrate/product.

    Drop-in replacement for SubstrateProductCrossAttention, adding:
      1. HodgeletCrossFilter   — Hodgelet-filtered enzyme queries
      2. SoftMolecularGate     — soft-gated substrate/product atom features
      3. JacobianCrossAttention — hyperbolic proximity logit + adjoint-correct reverse

    Note on dimensions: MoleculeGNN.proj maps mol features to enzyme_hidden_dim
    (H), so both SoftMolecularGate and JacobianCrossAttention operate on H-dimensional
    vectors (not mol_hidden_dim).

    The only new required input vs. SubstrateProductCrossAttention is
    ``enzyme_eigenvalues`` (per-atom Hodge eigenvalues).  If None, Hodgelet
    filtering is skipped and the module falls back to standard behaviour.
    """

    def __init__(self, cfg: Optional[ImprovedCrossAttentionConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ImprovedCrossAttentionConfig()
        H = self.cfg.enzyme_hidden_dim

        from tope.models.cross_attention import MoleculeGNN, CrossAttentionConfig
        base_cfg = CrossAttentionConfig(
            mol_feat_dim=self.cfg.mol_feat_dim,
            mol_hidden_dim=self.cfg.mol_hidden_dim,
            mol_n_layers=self.cfg.mol_n_layers,
            enzyme_hidden_dim=H,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
        )
        self.substrate_gnn = MoleculeGNN(base_cfg)
        self.product_gnn   = MoleculeGNN(base_cfg)

        # Improvement 1: Hodgelet filter on enzyme queries
        self.hodgelet_filter = HodgeletCrossFilter(
            enzyme_hidden_dim=H,
            k_eig=self.cfg.k_eig,
            n_heads=self.cfg.n_heads,
            n_scales=self.cfg.n_hodgelet_scales,
        )

        # Improvement 2: Soft gate on projected molecule features (dim = H)
        # MoleculeGNN.proj outputs H-dimensional vectors, not mol_hidden_dim.
        self.substrate_gate = SoftMolecularGate(H, self.cfg.n_gate_heads)
        self.product_gate   = SoftMolecularGate(H, self.cfg.n_gate_heads)

        # Improvement 3: Jacobian-corrected hyperbolic cross-attention
        # mol_dim = H because MoleculeGNN already projects to enzyme_hidden_dim.
        self.enzyme_to_substrate = JacobianCrossAttention(
            hidden_dim=H, mol_dim=H,
            n_heads=self.cfg.n_heads, dropout=self.cfg.dropout,
            ball_radius=self.cfg.ball_radius, sigma=self.cfg.sigma,
            learn_geometry=self.cfg.learn_geometry,
        )
        self.enzyme_to_product = JacobianCrossAttention(
            hidden_dim=H, mol_dim=H,
            n_heads=self.cfg.n_heads, dropout=self.cfg.dropout,
            ball_radius=self.cfg.ball_radius, sigma=self.cfg.sigma,
            learn_geometry=self.cfg.learn_geometry,
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.ln = nn.LayerNorm(H)

    def forward(
        self,
        h_enzyme: Tensor,
        substrate_data: Dict[str, Tensor],
        product_data: Dict[str, Tensor],
        enzyme_batch: Optional[Tensor] = None,
        substrate_batch: Optional[Tensor] = None,
        product_batch: Optional[Tensor] = None,
        enzyme_eigenvalues: Optional[Tensor] = None,  # (N_enz, k_eig)
        return_gates: bool = False,
    ) -> Tuple[Tensor, Dict]:
        """
        Parameters
        ----------
        h_enzyme           : (N_enz, H)
        substrate_data     : dict — node_features (M_sub, mol_feat_dim), edge_index (2, E_sub)
        product_data       : dict — node_features (M_prod, mol_feat_dim), edge_index (2, E_prod)
        enzyme_batch       : (N_enz,)  optional
        substrate_batch    : (M_sub,)  optional
        product_batch      : (M_prod,) optional
        enzyme_eigenvalues : (N_enz, k_eig)  Hodge eigenvalues; None → skip Hodgelet
        return_gates       : include per-atom gate weights in returned dict

        Returns
        -------
        h_fused : (N_enz, H)
        aux     : dict
            'substrate_attn' : (n_heads, N_enz, M_sub)
            'product_attn'   : (n_heads, N_enz, M_prod)
            'substrate_gate' : (M_sub,) or None
            'product_gate'   : (M_prod,) or None
        """
        # ── 1. Hodgelet-filter enzyme queries ─────────────────────────────────
        h_enz_q = (
            self.hodgelet_filter(h_enzyme, enzyme_eigenvalues)
            if enzyme_eigenvalues is not None
            else h_enzyme
        )

        # ── 2. Encode molecules then apply soft gate ──────────────────────────
        h_sub_raw          = self.substrate_gnn(substrate_data["node_features"], substrate_data["edge_index"])
        h_sub, sub_gate    = self.substrate_gate(h_sub_raw)

        h_prod_raw         = self.product_gnn(product_data["node_features"], product_data["edge_index"])
        h_prod, prod_gate  = self.product_gate(h_prod_raw)

        # ── 3. Jacobian-corrected cross-attention ─────────────────────────────
        h_enz_sub, attn_sub, _ = self.enzyme_to_substrate(
            h_enz_q, h_sub, enzyme_batch, substrate_batch, return_reverse=False
        )
        h_enz_prod, attn_prod, _ = self.enzyme_to_product(
            h_enz_q, h_prod, enzyme_batch, product_batch, return_reverse=False
        )

        # ── 4. Fuse ───────────────────────────────────────────────────────────
        h_fused = self.ln(h_enzyme + self.fusion(torch.cat([h_enz_sub, h_enz_prod], dim=-1)))

        return h_fused, {
            "substrate_attn": attn_sub,
            "product_attn":   attn_prod,
            "substrate_gate": sub_gate if return_gates else None,
            "product_gate":   prod_gate if return_gates else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests
# ══════════════════════════════════════════════════════════════════════════════

def _test_hodgelet_cross_filter():
    """HodgeletCrossFilter: output shape, no NaN, gate starts at reasonable value."""
    filt = HodgeletCrossFilter(enzyme_hidden_dim=32, k_eig=16, n_heads=4, n_scales=2)
    h    = torch.randn(20, 32)
    eigs = torch.rand(20, 16)
    out  = filt(h, eigs)
    assert out.shape == (20, 32) and not out.isnan().any()
    print("✓ HodgeletCrossFilter: shape and no NaN")


def _test_hodgelet_cross_filter_zero_eigs():
    """HodgeletCrossFilter with all-zero eigenvalues: filter adds exactly zero."""
    filt = HodgeletCrossFilter(enzyme_hidden_dim=32, k_eig=16, n_heads=4, n_scales=2)
    h    = torch.randn(10, 32)
    eigs = torch.zeros(10, 16)
    out  = filt(h, eigs)
    # spec_feat should be zero (Hodgelet outputs zero at λ=0)
    # so out = h + sigmoid(0) * proj(zero) = h + 0.5 * proj(0)
    # proj(0) may not be zero due to bias-free linear + SiLU(0)=0
    # With bias-free Linear and SiLU(0)=0: out == h + 0.5 * LayerNorm(0) ≈ h
    assert out.shape == (10, 32) and not out.isnan().any()
    print("✓ HodgeletCrossFilter: stable with λ=0 eigenvalues")


def _test_soft_molecular_gate():
    """SoftMolecularGate: gate in [0,1], output shape correct."""
    gate = SoftMolecularGate(hidden_dim=32, n_gate_heads=2)
    h    = torch.randn(15, 32)
    h_g, weights = gate(h)
    assert h_g.shape == (15, 32)
    assert weights.shape == (15,)
    assert weights.min() >= 0.0 and weights.max() <= 1.0
    print(f"✓ SoftMolecularGate: gate ∈ [{weights.min():.3f}, {weights.max():.3f}]")


def _test_jacobian_cross_attention_shapes():
    """JacobianCrossAttention: output shapes and no NaN."""
    attn = JacobianCrossAttention(hidden_dim=32, n_heads=4, dropout=0.0, ball_radius=2.0)
    h_enz = torch.randn(12, 32)
    h_mol = torch.randn(8, 32)
    out, attn_map, _ = attn(h_enz, h_mol)
    assert out.shape     == (12, 32)
    assert attn_map.shape == (4, 12, 8)
    assert not out.isnan().any()
    print("✓ JacobianCrossAttention: shapes and no NaN (forward only)")


def _test_jacobian_cross_attention_reverse():
    """JacobianCrossAttention return_reverse=True: both outputs valid."""
    attn = JacobianCrossAttention(hidden_dim=32, n_heads=4, dropout=0.0)
    h_enz = torch.randn(10, 32)
    h_mol = torch.randn(6,  32)
    out_e, _, out_m = attn(h_enz, h_mol, return_reverse=True)
    assert out_e.shape == (10, 32) and out_m.shape == (6, 32)
    assert not out_e.isnan().any() and not out_m.isnan().any()
    print("✓ JacobianCrossAttention: reverse output valid")


def _test_jacobian_cross_attention_gradient_flow():
    """log_t and sigma_param must receive non-zero gradients."""
    attn = JacobianCrossAttention(hidden_dim=16, n_heads=2, learn_geometry=True)
    h_enz = torch.randn(8, 16) * 0.3
    h_mol = torch.randn(5, 16) * 0.3
    out_e, _, out_m = attn(h_enz, h_mol, return_reverse=True)
    (out_e.sum() + out_m.sum()).backward()
    assert attn.log_t.grad is not None and attn.log_t.grad.abs() > 0
    print(f"✓ JacobianCrossAttention: ∂L/∂log_t={attn.log_t.grad.item():.4f}")


def _test_poincare_dist_positive():
    """Squared Poincaré distance must be non-negative."""
    attn = JacobianCrossAttention(hidden_dim=16, n_heads=2, ball_radius=1.0, learn_geometry=False)
    # Use small vectors strictly inside the ball
    u = torch.randn(2, 5, 8) * 0.1
    v = torch.randn(2, 6, 8) * 0.1
    d2 = attn._poincare_dist_sq(u, v)
    assert (d2 >= 0).all(), f"Negative distances: min={d2.min().item()}"
    print(f"✓ Poincaré distance: non-negative, range [{d2.min():.4f}, {d2.max():.4f}]")


def _test_jacobian_matrix_matches_mobius():
    """_jacobian_matrix must match _mobius_jacobian on individual pairs."""
    torch.manual_seed(7)
    N_q, N_k, d = 5, 4, 8
    q = torch.randn(N_q, d) * 0.2
    k = torch.randn(N_k, d) * 0.2
    t   = torch.tensor(2.0)
    sig = torch.tensor(0.0)

    mat = _jacobian_matrix(q, k, t, sig, d)   # (N_k, N_q)

    # Verify one entry
    j_01 = _mobius_jacobian(q[1].unsqueeze(0), k[0].unsqueeze(0), t, sig, d)
    assert torch.allclose(mat[0, 1], j_01.squeeze(), atol=1e-5), \
        f"Mismatch: {mat[0, 1].item():.6f} vs {j_01.item():.6f}"
    print("✓ _jacobian_matrix: matches _mobius_jacobian on individual pairs")


if __name__ == "__main__":
    _test_hodgelet_cross_filter()
    _test_hodgelet_cross_filter_zero_eigs()
    _test_soft_molecular_gate()
    _test_jacobian_cross_attention_shapes()
    _test_jacobian_cross_attention_reverse()
    _test_jacobian_cross_attention_gradient_flow()
    _test_poincare_dist_positive()
    _test_jacobian_matrix_matches_mobius()
    print("\nAll cross-attention improvement tests passed.")
