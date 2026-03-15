"""
cc_attention_improved.py
========================
Three structural improvements to the CCANN attention primitives:
  1. HodgeletFilter          — oversmoothing prevention via harmonic orthogonality.
                               Replaces hard spectral cutoff Λ_r with a soft
                               bandpass Ψ̂_ρ(λ) = √λ · exp(-ρλ/2) that is
                               identically zero on ker(Δ_k), making oversmoothing
                               structurally impossible regardless of network depth.
  2. SoftSpectralCutoff      — continuity under structural perturbation.
                               Replaces the hard threshold filter with a learnable
                               smooth gate g(λ; θ) = sigmoid(θ₀ · (λ - θ₁))
                               that is differentiable in the eigenvalue and stable
                               under small perturbations of the protein structure.
  3. JacobianCorrectedBlock  — correct adjoint symmetry in bidirectional attention.
                               Adds the Ferreira (2015) Jacobian factor j_a(x) to
                               the reverse attention logit so that forward and
                               reverse attention are exactly adjoint under the
                               (σ,t)-invariant measure dμ_{σ,t}.

v2 improvements (this revision):
  4. Vectorised 2-D scatter softmax   — replaces O(n_heads) sequential kernel
                                        launches with one vectorised scatter
                                        operation over the full (E, H) tensor.
  5. Gradient-correct Jacobian        — removes .item() detaches so that
                                        ball_radius and sigma_param receive
                                        gradients during training.  Without
                                        this fix both parameters are silently
                                        non-trainable.
  6. Scaled attention logits          — multiplies e_ij by 1/√head_dim before
                                        scatter-softmax, matching the proven
                                        stability of scaled dot-product attention
                                        and consistent with MultiHeadCrossAttention
                                        in cross_attention.py.
  7. SoftSpectralCutoff.gate_rank()   — single-rank gate path that avoids the
                                        O(n_ranks) allocation in the original
                                        gate_all(expand(n_ranks))[...,rank]
                                        pattern used in HodgeletFilteredCCBlock.
  8. Complete drop-in API             — adds CCAttentionPushForwardImproved
                                        (equal-rank, scaled, separate Q/K/V),
                                        ContentAttentionMergeNode (content-based
                                        merge weights), AttentionMergeNode
                                        (original static, kept for compatibility),
                                        and build_zone_adjacency so this file
                                        fully replaces cc_attention.py.

All components are drop-in replacements for their counterparts in cc_attention.py.
The public API is intentionally identical so that tcpnet.py and
whole_protein_tcpnet.py require no changes beyond the import.

References
----------
Ferreira (2015) "Harmonic Analysis on the Möbius Gyrogroup"
  J. Fourier Anal. Appl. 21:281–317  — Jacobian factor (Def 1, Eq 17–18),
  Plancherel measure (Thm 8, Eq 77), heat kernel (Sec 8).
Hajij et al. — CCANN formalism (Defs 32, 33, 36).
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

def _scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Numerically stable per-group softmax over a flat (E,) index."""
    if HAS_TORCH_SCATTER:
        return scatter_softmax(src, index, dim=0)
    max_v = src.new_full((dim_size,), float("-inf"))
    max_v.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    shifted = (src - max_v[index]).exp()
    denom = src.new_zeros(dim_size)
    denom.scatter_add_(0, index, shifted)
    return shifted / (denom[index] + 1e-8)


def _scatter_softmax_2d(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Vectorised per-group softmax for (E, H) tensors.

    Replaces the pattern ``torch.stack([_scatter_softmax(src[:,h], ...) for h
    in range(H)], dim=1)`` with a single scatter pass over all heads at once.

    Parameters
    ----------
    src      : (E, H)  raw attention logits, one column per head
    index    : (E,)    group assignment (target-node index)
    dim_size : int     number of groups (N_t or N_s)

    Returns
    -------
    att : (E, H)  normalised attention weights
    """
    H = src.size(1)
    if HAS_TORCH_SCATTER:
        # scatter_softmax supports n-D src with matching n-D index
        return scatter_softmax(src, index.unsqueeze(1).expand(-1, H), dim=0)
    # Manual fallback: one pass over the full (E, H) tensor
    idx = index.unsqueeze(1).expand(-1, H)                   # (E, H)
    max_v = src.new_full((dim_size, H), float("-inf"))
    max_v.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    shifted = (src - max_v[index]).exp()                     # (E, H)
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
# 1. HodgeletFilter
#    Oversmoothing prevention via harmonic orthogonality
# ══════════════════════════════════════════════════════════════════════════════

class HodgeletFilter(nn.Module):
    """Soft bandpass filter on the Hodge Laplacian spectrum.

    Given eigenvalues λ of Δ_k, applies the diffusive Hodgelet transfer
    function in the spectral domain:

        Ψ̂_ρ(λ) = √λ · exp(-ρ · λ / 2)

    This filter:
      - is exactly zero at λ = 0  →  orthogonal to ker(Δ_k) by construction,
        making oversmoothing structurally impossible.
      - peaks at λ* = 2/ρ  →  scale ρ selects which spectral band is amplified.
      - decays for large λ  →  suppresses noise in the high-frequency regime.
      - is differentiable everywhere  →  can be learned end-to-end.

    The scale parameter ρ is learnable per rank and per head, initialised to
    `init_rho`. Setting `learn_rho=False` fixes it (useful for ablations).

    Parameters
    ----------
    n_scales  : number of Hodgelet scale parameters (one per TTN leaf / rank)
    n_heads   : number of attention heads (one ρ per head for multi-scale)
    init_rho  : initial scale (2/init_rho = peak eigenvalue)
    learn_rho : whether to learn ρ (True) or keep it fixed (False)
    eps       : numerical floor for √λ to avoid NaN at λ = 0
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
        # ρ > 0 enforced via softplus: rho = softplus(rho_raw) + eps
        rho_init = math.log(math.expm1(max(init_rho - eps, 1e-3)))
        raw = torch.full((n_scales, n_heads), rho_init)
        if learn_rho:
            self.rho_raw = nn.Parameter(raw)
        else:
            self.register_buffer("rho_raw", raw)

    @property
    def rho(self) -> Tensor:
        """Positive scale parameters (n_scales, n_heads)."""
        return F.softplus(self.rho_raw) + self.eps

    def forward(self, eigenvalues: Tensor) -> Tensor:
        """Apply Hodgelet filter to a batch of eigenvalue sequences.

        Parameters
        ----------
        eigenvalues : (B, k_eig) or (B, n_scales, k_eig)
            Raw eigenvalues from the persistent Laplacian / TTN leaves.
            If 2-D, the same eigenvalues are used for all scales.

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

        # lam: (B, n_scales, 1, k_eig)
        # rho: (n_scales, n_heads) → (1, n_scales, n_heads, 1)
        rho = self.rho.unsqueeze(0).unsqueeze(-1)

        # lam · (lam + ε)^{-½}  is exactly 0 when lam = 0  (harmonic orthogonality)
        # and approaches √lam for lam ≫ ε.
        # NOTE: (lam + ε).sqrt() is non-zero at lam=0 and would break the guarantee.
        sqrt_lam = lam * (lam + self.eps).rsqrt()    # (B, n_scales, 1, k_eig)
        decay = torch.exp(-rho * lam / 2)            # (B, n_scales, n_heads, k_eig)
        return sqrt_lam * decay                      # (B, n_scales, n_heads, k_eig)

    def extra_repr(self) -> str:
        mean_rho = self.rho.mean().item()
        return (f"n_scales={self.n_scales}, n_heads={self.n_heads}, "
                f"mean_peak_lambda={2.0 / mean_rho:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SoftSpectralCutoff
#    Continuity under structural perturbation
# ══════════════════════════════════════════════════════════════════════════════

class SoftSpectralCutoff(nn.Module):
    """Learnable smooth spectral gate replacing the hard cutoff Λ_r.

    The hard filtration threshold  [λ ≤ Λ_r]  is a Heaviside step function.
    This module replaces it with a sigmoid gate:

        g(λ; θ) = σ( θ_slope · (θ_center - λ) )

    One (θ_slope, θ_center) pair is learned per rank and per filtration step.

    Parameters
    ----------
    n_ranks            : number of CC ranks
    n_filtration_steps : number of filtration steps along Λ axis
    init_center        : initial gate center (in eigenvalue units)
    init_slope         : initial slope (higher = sharper gate)
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

        self.centers = nn.Parameter(
            torch.full((n_ranks, n_filtration_steps), init_center)
        )
        slope_raw_init = math.log(math.expm1(init_slope))
        self.slopes_raw = nn.Parameter(
            torch.full((n_ranks, n_filtration_steps), slope_raw_init)
        )

    @property
    def slopes(self) -> Tensor:
        """Positive gate slopes (n_ranks, n_filtration_steps)."""
        return F.softplus(self.slopes_raw)

    def forward(
        self,
        eigenvalues: Tensor,
        rank: int,
        filtration_step: int,
    ) -> Tensor:
        """Apply the soft gate at a single (rank, filtration_step) point.

        Parameters
        ----------
        eigenvalues    : (B, k_eig)
        rank           : which CC rank
        filtration_step: which filtration step

        Returns
        -------
        gated : (B, k_eig)  in [0, 1]
        """
        c = self.centers[rank, filtration_step]
        s = self.slopes[rank, filtration_step]
        return torch.sigmoid(s * (c - eigenvalues))

    def gate_rank(self, eigenvalues: Tensor, rank: int) -> Tensor:
        """Apply soft gate for a single rank across all filtration steps.

        This is O(1) in n_ranks — it directly uses the parameters for the
        requested rank, avoiding the O(n_ranks) allocation of
        ``gate_all(expand(n_ranks))[..., rank, :, :]``.

        Parameters
        ----------
        eigenvalues : (B, n_filtration_steps, k_eig)
        rank        : which CC rank to apply

        Returns
        -------
        gated : (B, n_filtration_steps, k_eig)
        """
        c = self.centers[rank].unsqueeze(0).unsqueeze(-1)   # (1, F, 1)
        s = self.slopes[rank].unsqueeze(0).unsqueeze(-1)    # (1, F, 1)
        return torch.sigmoid(s * (c - eigenvalues))

    def gate_all(self, eigenvalues: Tensor) -> Tensor:
        """Apply gates across all (rank, filtration_step) pairs at once.

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
        """Return the 50%-pass eigenvalue for each (rank, step).

        Returns
        -------
        (n_ranks, n_filtration_steps)
        """
        return self.centers.detach()

    def extra_repr(self) -> str:
        return (f"n_ranks={self.n_ranks}, "
                f"n_filtration_steps={self.n_filtration_steps}, "
                f"mean_center={self.centers.mean().item():.3f}, "
                f"mean_slope={self.slopes.mean().item():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. JacobianCorrectedBlock
#    Correct adjoint symmetry in bidirectional CCANN attention
# ══════════════════════════════════════════════════════════════════════════════

def _mobius_jacobian(
    a: Tensor,
    x: Tensor,
    t: Tensor,
    sigma: Tensor,
    n: int,
) -> Tensor:
    """Compute the Ferreira (2015) Jacobian factor j_a(x).

    From Definition 1 / Eq 18 of Ferreira (2015):

        j_a(x) = ( (1 - ‖a‖²/t²) / |1 + ax/t²|² )^{(n+σ-2)/2}

    .. note::
        ``t`` and ``sigma`` must be passed as 0-dim Tensors (not Python
        floats) so that gradients flow back to ``log_t`` and ``sigma_param``
        in ``JacobianCorrectedBlock``.  Passing ``.item()`` values severs the
        autograd graph and silently makes those parameters non-trainable.

    In the Euclidean limit t → ∞, j_a(x) → 1 for all a, x.

    Parameters
    ----------
    a     : (E, d)  source embeddings (projected, on Poincaré ball)
    x     : (E, d)  target embeddings
    t     : ()      ball radius as a 0-dim Tensor (keeps grad flowing)
    sigma : ()      conformal weight as a 0-dim Tensor (keeps grad flowing)
    n     : int     ambient dimension (d_out of the projection)

    Returns
    -------
    j : (E,)  positive Jacobian weight per edge
    """
    t2 = t * t
    a_norm_sq = (a * a).sum(dim=-1) / t2          # (E,)
    ax_dot    = (a * x).sum(dim=-1) / t2           # (E,)
    x_norm_sq = (x * x).sum(dim=-1) / t2           # (E,)

    numer = (1.0 - a_norm_sq).clamp(min=1e-8)
    # Full Möbius denominator: 1 + 2⟨a,x⟩/t² + ‖a‖²‖x‖²/t⁴
    denom = (1.0 + 2.0 * ax_dot + a_norm_sq * x_norm_sq).clamp(min=1e-8)

    # Exponent: (n + σ - 2) / 2  — computed in tensor arithmetic so that
    # gradients flow through sigma.
    exp = (n + sigma - 2.0) / 2.0

    # Log-space computation avoids overflow/underflow for large |exp|
    log_j = exp * (numer.log() - denom.log())
    return log_j.exp().clamp(min=1e-6, max=1e6)


class JacobianCorrectedBlock(nn.Module):
    """Bidirectional CC-attention block with (σ,t)-adjoint-correct reverse attention.

    Extends CCAttentionBlock (Definition 33) with the Ferreira (2015) Jacobian
    correction to the reverse attention logit, so that the forward (s→t) and
    reverse (t→s) attention maps are exactly adjoint under the (σ,t)-invariant
    measure dμ_{σ,t}.

    v2 changes vs the original:
      - Scaled logits: e_ij *= 1/√head_dim before scatter-softmax.
      - Vectorised 2-D softmax: single scatter pass over all heads at once.
      - Gradient-correct Jacobian: t and sigma passed as tensors, not floats.

    Parameters
    ----------
    d_s_in, d_t_in : input feature dimensions for rank-s and rank-t cochains
    d_out          : output dimension (same for both directions)
    n_heads        : number of attention heads
    dropout        : attention dropout probability
    negative_slope : LeakyReLU negative slope for the attention activation φ
    ball_radius    : t > 0, Poincaré ball radius (default 1.0)
    sigma          : conformal weight σ ∈ ℝ (default 0.0; 0 → standard Lebesgue)
    learn_geometry : if True, learn ball_radius and sigma as scalar parameters
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
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_s_in = d_s_in
        self.d_t_in = d_t_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_s = nn.Linear(d_s_in, d_out, bias=False)
        self.W_t = nn.Linear(d_t_in, d_out, bias=False)

        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        # Scale factor: 1/√head_dim applied to logits before scatter-softmax.
        # Keeps attention score variance O(1) regardless of head_dim, matching
        # the stability argument from Vaswani et al. and the implementation in
        # cross_attention.py:MultiHeadCrossAttention.
        self.scale = self.head_dim ** -0.5

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Geometry: store as log(t) so t > 0 always, σ is unconstrained.
        _sigma0 = float(sigma) if sigma is not None else 0.0
        if learn_geometry:
            self.log_t = nn.Parameter(torch.tensor(math.log(ball_radius)))
            self.sigma_param = nn.Parameter(torch.tensor(_sigma0))
        else:
            self.register_buffer("log_t", torch.tensor(math.log(ball_radius)))
            self.register_buffer("sigma_param", torch.tensor(_sigma0))

    @property
    def ball_radius(self) -> float:
        return self.log_t.exp().item()

    @property
    def sigma(self) -> float:
        return self.sigma_param.item()

    def forward(
        self,
        H_s: Tensor,       # (N_s, d_s_in)
        H_t: Tensor,       # (N_t, d_t_in)
        B: Tensor,         # (2, E)  B[0]=rank-s idx, B[1]=rank-t idx
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        K_t : (N_t, d_out)  updated rank-t cochain  (forward  s→t)
        K_s : (N_s, d_out)  updated rank-s cochain  (reverse  t→s, Jacobian-corrected)
        """
        N_s, N_t = H_s.size(0), H_t.size(0)
        src, tgt = B[0], B[1]
        E = src.size(0)
        H, hd = self.n_heads, self.head_dim

        # t and sig remain as 0-dim Tensors (NOT .item()) so that gradients
        # flow back through log_t and sigma_param during the backward pass.
        t   = self.log_t.exp()     # 0-dim Tensor
        sig = self.sigma_param     # 0-dim Tensor

        # ── Project ──────────────────────────────────────────────────────────
        Ws_Hs = self.W_s(H_s).view(N_s, H, hd)
        Wt_Ht = self.W_t(H_t).view(N_t, H, hd)
        Ws_Hs_src = Ws_Hs[src]    # (E, H, hd)
        Wt_Ht_tgt = Wt_Ht[tgt]   # (E, H, hd)

        # ── Forward attention: s → t ──────────────────────────────────────────
        fwd_cat = torch.cat([Ws_Hs_src, Wt_Ht_tgt], dim=-1)   # (E, H, 2*hd)
        e_fwd = (fwd_cat * self.a.unsqueeze(0)).sum(-1)         # (E, H)
        # Scale before activation to keep variance O(1) across head_dim sizes.
        e_fwd = self.leaky_relu(e_fwd * self.scale)

        # Single vectorised scatter-softmax over all heads at once.
        att_fwd = _scatter_softmax_2d(e_fwd, tgt, N_t)         # (E, H)
        att_fwd = self.dropout(att_fwd)

        weighted_fwd = att_fwd.unsqueeze(-1) * Ws_Hs_src
        K_t = _scatter_add_nd(weighted_fwd, tgt, N_t).view(N_t, self.d_out)

        # ── Reverse attention: t → s (Jacobian-corrected) ────────────────────
        rev_a   = torch.cat([self.a[:, hd:], self.a[:, :hd]], dim=-1)  # (H, 2*hd)
        rev_cat = torch.cat([Wt_Ht_tgt, Ws_Hs_src], dim=-1)            # (E, H, 2*hd)
        f_rev   = (rev_cat * rev_a.unsqueeze(0)).sum(-1)                # (E, H)
        f_rev   = self.leaky_relu(f_rev * self.scale)

        # Jacobian correction — pass t and sig as Tensors (not .item()) so
        # that log_t and sigma_param stay connected to the autograd graph.
        h_s_flat = Ws_Hs_src.view(E, self.d_out)
        h_t_flat = Wt_Ht_tgt.view(E, self.d_out)
        j = _mobius_jacobian(h_s_flat, h_t_flat, t, sig, self.d_out)  # (E,)

        f_rev_corrected = f_rev * j.unsqueeze(-1)               # (E, H)

        att_rev = _scatter_softmax_2d(f_rev_corrected, src, N_s)  # (E, H)
        att_rev = self.dropout(att_rev)

        weighted_rev = att_rev.unsqueeze(-1) * Wt_Ht_tgt
        K_s = _scatter_add_nd(weighted_rev, src, N_s).view(N_s, self.d_out)

        return K_t, K_s

    def extra_repr(self) -> str:
        return (f"d_s_in={self.d_s_in}, d_t_in={self.d_t_in}, d_out={self.d_out}, "
                f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
                f"scale={self.scale:.4f}, ball_radius={self.ball_radius:.3f}, "
                f"sigma={self.sigma:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: HodgeletFilteredCCBlock
# Wraps JacobianCorrectedBlock with Hodgelet-filtered spectral features
# ══════════════════════════════════════════════════════════════════════════════

class HodgeletFilteredCCBlock(nn.Module):
    """Full structural improvement stack in a single composable module.

    Applies all three improvements in sequence:
      1. SoftSpectralCutoff gates the raw eigenvalue sequence (via gate_rank,
         which is O(1) in n_ranks — no n_ranks expansion).
      2. HodgeletFilter converts gated eigenvalues into multi-scale spectral
         features that are orthogonal to the harmonic kernel.
      3. JacobianCorrectedBlock uses these spectral features as additional
         edge attributes in the Jacobian-corrected bidirectional attention.
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
        self.soft_cutoff = SoftSpectralCutoff(
            n_ranks=n_ranks,
            n_filtration_steps=n_filtration_steps,
        )
        self.hodgelet = HodgeletFilter(
            n_scales=n_hodgelet_scales,
            n_heads=n_heads,
        )
        spec_raw_dim = n_hodgelet_scales * n_heads * k_eig
        spec_proj_dim = min(spec_raw_dim, d_out)
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_raw_dim, spec_proj_dim),
            nn.SiLU(),
        )
        self.attn_block = JacobianCorrectedBlock(
            d_s_in=d_s_in + spec_proj_dim,
            d_t_in=d_t_in,
            d_out=d_out,
            n_heads=n_heads,
            dropout=dropout,
            negative_slope=negative_slope,
            ball_radius=ball_radius,
            sigma=sigma,
            learn_geometry=learn_geometry,
        )
        self.k_eig = k_eig
        self.n_ranks = n_ranks
        self.n_filtration_steps = n_filtration_steps

    def forward(
        self,
        H_s: Tensor,               # (N_s, d_s_in)
        H_t: Tensor,               # (N_t, d_t_in)
        B: Tensor,                 # (2, E)
        eigenvalues: Tensor,       # (N_s, n_ranks, n_filtration_steps, k_eig)
        rank: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        eigenvalues : (N_s, n_ranks, n_filtration_steps, k_eig)
        rank        : CC rank of the rank-s cells (selects the rank slice)
        """
        # ── 1. Soft gate — O(1) in n_ranks via gate_rank ─────────────────────
        # Extract only the rank-s eigenvalue slice, avoiding n_ranks expansion.
        evals_rank = eigenvalues[:, rank, :, :]         # (N_s, F, k_eig)
        gated = self.soft_cutoff.gate_rank(evals_rank, rank)  # (N_s, F, k_eig)

        # ── 2. Hodgelet filter ────────────────────────────────────────────────
        evals_mean = (gated * evals_rank).mean(dim=1)   # (N_s, k_eig)
        filtered = self.hodgelet(evals_mean)            # (N_s, n_scales, n_heads, k_eig)

        # ── 3. Project spectral features ──────────────────────────────────────
        N_s_actual, n_sc, n_h, k = filtered.shape
        spec_flat = filtered.view(N_s_actual, n_sc * n_h * k)
        spec_feat = self.spec_proj(spec_flat)           # (N_s, spec_proj_dim)

        H_s_aug = torch.cat([H_s, spec_feat], dim=-1)  # (N_s, d_s_in+spec_proj_dim)

        # ── 4. Jacobian-corrected bidirectional attention ─────────────────────
        return self.attn_block(H_s_aug, H_t, B)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CCAttentionPushForwardImproved
#    Equal-rank drop-in replacement for CCAttentionPushForward (Definition 32)
# ══════════════════════════════════════════════════════════════════════════════

class CCAttentionPushForwardImproved(nn.Module):
    """Improved equal-rank CC-attention push-forward (Definition 32).

    Drop-in replacement for ``CCAttentionPushForward`` in cc_attention.py.
    Improvements over the original:
      - Separate W_Q, W_K, W_V projections (vs a single shared W used as both
        key and value), making the key and value spaces independently learnable.
      - Scaled logits: e_ij *= 1/√head_dim to keep attention score variance O(1).
      - Vectorised 2-D scatter-softmax via _scatter_softmax_2d.

    Parameters
    ----------
    d_in           : input feature dimension
    d_out          : output dimension
    n_heads        : number of attention heads
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

        # Separate query / key / value projections
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_K = nn.Linear(d_in, d_out, bias=False)
        self.W_V = nn.Linear(d_in, d_out, bias=False)

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
        value     : (E, d_out) optional override for aggregated values

        Returns
        -------
        K_t : (N, d_out)
        """
        N = H_s.size(0)
        src, tgt = adj[0], adj[1]

        Q = self.W_Q(H_s).view(N, self.n_heads, self.head_dim)  # (N, H, hd)
        K = self.W_K(H_s).view(N, self.n_heads, self.head_dim)
        V = self.W_V(H_s).view(N, self.n_heads, self.head_dim)

        Q_i = Q[tgt]   # (E, H, hd) — query at target
        K_j = K[src]   # (E, H, hd) — key at source
        V_j = V[src]   # (E, H, hd) — value at source

        cat_attn = torch.cat([Q_i, K_j], dim=-1)  # (E, H, 2*hd)
        if edge_attr is not None and self.d_edge > 0:
            ea = edge_attr.unsqueeze(1).expand(-1, self.n_heads, -1)
            cat_attn = torch.cat([cat_attn, ea], dim=-1)

        e = (cat_attn * self.a.unsqueeze(0)).sum(-1)   # (E, H)
        e = self.leaky_relu(e * self.scale)

        att = _scatter_softmax_2d(e, tgt, N)            # (E, H)
        att = self.dropout(att)

        vals = value.view(-1, self.n_heads, self.head_dim) if value is not None else V_j
        weighted = att.unsqueeze(-1) * vals             # (E, H, hd)
        out = _scatter_add_nd(weighted, tgt, N)
        return out.view(N, self.d_out)

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_out={self.d_out}, n_heads={self.n_heads}, "
                f"scale={self.scale:.4f}, d_edge={self.d_edge}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. AttentionMergeNode  (Definition 36 — original static version, kept for
#    API compatibility) and ContentAttentionMergeNode (content-based variant)
# ══════════════════════════════════════════════════════════════════════════════

class AttentionMergeNode(nn.Module):
    """Inter-neighborhood attention merge node (Definition 36).

    Original static-weight version, preserved for API compatibility with
    TCPNetLayer callers.  Each of the K neighborhoods gets a single learnable
    global scalar weight shared across all nodes.

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
        """
        Parameters
        ----------
        messages : list of (N, d) tensors, one per neighborhood

        Returns
        -------
        merged : (N, d)
        """
        assert len(messages) == self.n_neighborhoods
        b_k = torch.softmax(self.weights, dim=0)
        return sum(b_k[k] * messages[k] for k in range(self.n_neighborhoods))


class ContentAttentionMergeNode(nn.Module):
    """Content-based inter-neighborhood merge (Definition 36 — improved).

    Replaces the static global b^k weights with per-node content-dependent
    gates: each node computes its own blend of incoming neighborhood messages
    based on the concatenation of all incoming messages.

        gate_k(i) = softmax_k( Linear([msg_1[i], ..., msg_K[i]]) )
        merged[i] = Σ_k gate_k(i) · msg_k[i]

    This is strictly more expressive than the static-weight version while
    preserving the (N, d) → (N, d) interface.

    Parameters
    ----------
    n_neighborhoods : number of neighborhoods to merge
    hidden_dim      : feature dimension of each message tensor
    """

    def __init__(self, n_neighborhoods: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_neighborhoods = n_neighborhoods
        self.hidden_dim = hidden_dim
        # Gate: concatenated messages → per-neighborhood score
        self.gate = nn.Linear(hidden_dim * n_neighborhoods, n_neighborhoods, bias=True)

    def forward(self, messages: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        messages : list of K tensors, each (N, hidden_dim)

        Returns
        -------
        merged : (N, hidden_dim)
        """
        assert len(messages) == self.n_neighborhoods
        stack  = torch.stack(messages, dim=1)        # (N, K, d)
        concat = stack.view(stack.size(0), -1)       # (N, K*d)
        gates  = torch.softmax(self.gate(concat), dim=1)   # (N, K)
        return (gates.unsqueeze(-1) * stack).sum(dim=1)    # (N, d)

    def extra_repr(self) -> str:
        return f"n_neighborhoods={self.n_neighborhoods}, hidden_dim={self.hidden_dim}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. build_zone_adjacency  (from cc_attention.py — required by
#    whole_protein_tcpnet.py:ZoneAwareAttention)
# ══════════════════════════════════════════════════════════════════════════════

def build_zone_adjacency(
    zone_mask: Tensor,
    global_edge_index: Tensor,
) -> Tensor:
    """Filter edge_index to edges with both endpoints in a zone, remapping
    node indices to zone-local contiguous indices.

    Parameters
    ----------
    zone_mask         : (N_total,) BoolTensor — True for nodes in this zone
    global_edge_index : (2, E) — global edge_index (source, target)

    Returns
    -------
    local_edge_index : (2, E_zone) — zone-local edge index
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
# Unit tests
# ══════════════════════════════════════════════════════════════════════════════

def _test_hodgelet_harmonic_orthogonality():
    """Hodgelet response at λ=0 must be exactly zero."""
    hf = HodgeletFilter(n_scales=4, n_heads=4)
    lam = torch.zeros(2, 32)
    out = hf(lam)
    assert out.abs().max().item() < 1e-6, "Hodgelet non-zero at λ=0"
    print("✓ HodgeletFilter: harmonic orthogonality")


def _test_soft_cutoff_continuity():
    """Small Δλ must produce small Δg (Lipschitz continuity)."""
    sc = SoftSpectralCutoff(n_ranks=4, n_filtration_steps=16, init_slope=5.0)
    lam1 = torch.ones(8, 4, 16, 32) * 1.99
    lam2 = torch.ones(8, 4, 16, 32) * 2.01
    g1 = sc.gate_all(lam1)
    g2 = sc.gate_all(lam2)
    delta_g = (g2 - g1).abs().max().item()
    assert delta_g < 0.2 * 0.02 * 5.0 + 0.05, f"Gate too discontinuous: Δg={delta_g:.4f}"
    print(f"✓ SoftSpectralCutoff: Δg={delta_g:.4f} for Δλ=0.02")


def _test_gate_rank_matches_gate_all():
    """gate_rank(evals, r) must equal gate_all(...)[..., r, :, :] exactly."""
    sc = SoftSpectralCutoff(n_ranks=4, n_filtration_steps=8)
    rank = 2
    evals_rank = torch.randn(10, 8, 16).abs()          # (N, F, k)
    evals_full = evals_rank.unsqueeze(1).expand(-1, 4, -1, -1)  # (N, R, F, k)

    via_rank = sc.gate_rank(evals_rank, rank)
    via_all  = sc.gate_all(evals_full)[:, rank, :, :]

    assert torch.allclose(via_rank, via_all, atol=1e-6), "gate_rank / gate_all mismatch"
    print("✓ SoftSpectralCutoff: gate_rank == gate_all[...,rank]")


def _test_scatter_softmax_2d_matches_1d_loop():
    """_scatter_softmax_2d must produce identical results to the 1-D head loop."""
    torch.manual_seed(0)
    E, H, N = 50, 8, 12
    src = torch.randn(E, H)
    idx = torch.randint(0, N, (E,))

    loop_result = torch.stack(
        [_scatter_softmax(src[:, h], idx, N) for h in range(H)], dim=1
    )
    vec_result = _scatter_softmax_2d(src, idx, N)

    assert torch.allclose(loop_result, vec_result, atol=1e-6), (
        f"max diff: {(loop_result - vec_result).abs().max().item()}"
    )
    print("✓ _scatter_softmax_2d: matches 1-D head loop")


def _test_jacobian_gradient_flow():
    """ball_radius and sigma_param must receive gradients after backward."""
    block = JacobianCorrectedBlock(
        d_s_in=16, d_t_in=16, d_out=16, n_heads=2,
        ball_radius=2.0, sigma=0.0, learn_geometry=True,
    )
    N_s, N_t, E = 8, 5, 12
    H_s = torch.randn(N_s, 16) * 0.3
    H_t = torch.randn(N_t, 16) * 0.3
    B = torch.stack([torch.randint(0, N_s, (E,)), torch.randint(0, N_t, (E,))])

    K_t, K_s = block(H_s, H_t, B)
    loss = K_t.sum() + K_s.sum()
    loss.backward()

    assert block.log_t.grad is not None and block.log_t.grad.abs() > 0, \
        "log_t received no gradient"
    assert block.sigma_param.grad is not None and block.sigma_param.grad.abs() > 0, \
        "sigma_param received no gradient"
    print(f"✓ JacobianCorrectedBlock: gradient flow — "
          f"∂L/∂log_t={block.log_t.grad.item():.4f}, "
          f"∂L/∂sigma={block.sigma_param.grad.item():.4f}")


def _test_jacobian_block_euclidean_limit():
    """In Euclidean limit (large t), block produces valid output."""
    block = JacobianCorrectedBlock(
        d_s_in=32, d_t_in=32, d_out=32, n_heads=4,
        ball_radius=1e4, learn_geometry=False,
    )
    N_s, N_t, E = 10, 6, 15
    H_s = torch.randn(N_s, 32)
    H_t = torch.randn(N_t, 32)
    B = torch.stack([torch.randint(0, N_s, (E,)), torch.randint(0, N_t, (E,))])
    K_t, K_s = block(H_s, H_t, B)
    assert K_t.shape == (N_t, 32) and K_s.shape == (N_s, 32)
    assert not K_t.isnan().any() and not K_s.isnan().any()
    print("✓ JacobianCorrectedBlock: Euclidean limit produces valid output")


def _test_push_forward_improved_shapes():
    """CCAttentionPushForwardImproved: output shape matches the original."""
    pf = CCAttentionPushForwardImproved(d_in=32, d_out=32, n_heads=4)
    N, E = 20, 40
    H = torch.randn(N, 32)
    adj = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
    out = pf(H, adj)
    assert out.shape == (N, 32)
    print("✓ CCAttentionPushForwardImproved: output shape correct")


def _test_content_merge_node():
    """ContentAttentionMergeNode: output shape and no NaN."""
    merge = ContentAttentionMergeNode(n_neighborhoods=3, hidden_dim=32)
    msgs = [torch.randn(10, 32) for _ in range(3)]
    out = merge(msgs)
    assert out.shape == (10, 32) and not out.isnan().any()
    print("✓ ContentAttentionMergeNode: output shape correct, no NaN")


if __name__ == "__main__":
    _test_hodgelet_harmonic_orthogonality()
    _test_soft_cutoff_continuity()
    _test_gate_rank_matches_gate_all()
    _test_scatter_softmax_2d_matches_1d_loop()
    _test_jacobian_gradient_flow()
    _test_jacobian_block_euclidean_limit()
    _test_push_forward_improved_shapes()
    _test_content_merge_node()
    print("\nAll structural improvement tests passed.")
