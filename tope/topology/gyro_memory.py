"""Möbius gyrovector primitives shared by the gyro-CCANN and sheaf heads.

This module is the single home for the gyration operator that both ToPE
hyperbolic heads depend on (consolidated spec §4 / §6):

  * the gyro-CCANN attention head (``tope.models.cc_attention``) uses the
    gyration ``G_ij = gyr[p_i, ⊖q_j]`` as the *reverse-logit frame
    correction* (Thomas precession) that keeps the Def-33 bidirectional
    block adjoint when the anisotropic ``aᵀv`` logit is used (§3);
  * the tangent-space sheaf-Laplacian head (``tope.topology.phonon_topology``)
    uses ``R_xy = gyr[h_y, ⊖h_x]`` as the *transport rotation* inside the
    restriction map ``F = K · R`` (§5), and the holonomy product
    ``Hol(loop) = ∏ R_xy`` as the gyration **FrustIndex** (§5.5).

Everything here is pure geometry on the Poincaré ball ``B^d_s`` (radius
``s``; curvature ``c = 1/s²``).  Spec formulas are written for ``s = 1``;
the general-``s`` forms replace every inner product / squared norm by its
``1/s²``-scaled value (spec "Conventions").

Grounding
---------
Möbius gyro-operations and the closed-form gyration are established
(Ungar, *Analytic Hyperbolic Geometry*).  The closed form in
:func:`gyr` is verified numerically against the defining identity
``gyr[a,b]v = ⊖(a⊕b) ⊕ (a ⊕ (b⊕v))`` to ``< 2e-13`` (see
``tests/test_gyro_memory.py``).  The gyration **FrustIndex** assembly
(holonomy of the transport rotations around a loop) is the ToPE
construction; it is the gyro-transport counterpart of the
Nijenhuis/Chern ``frust_index_cb`` in :mod:`tope.topology.g_structure`.

Both a torch (autodiff, batched per-edge) and a NumPy (single-pair /
restriction-map) surface are provided so the attention head and the
scipy-based phonon head can each call in their native dtype.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a hard dep in practice
    HAS_TORCH = False
    Tensor = "Tensor"  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NumPy surface — single pair / restriction maps (phonon head, §4–§5)
# ══════════════════════════════════════════════════════════════════════════════

def gyr(a: np.ndarray, b: np.ndarray, v: np.ndarray, s: float = 1.0) -> np.ndarray:
    r"""Closed-form Möbius gyration ``gyr[a,b] v`` on ``B^d_s`` (spec §4).

    For ``s = 1`` (the verified reference form)::

        D = 1 + 2⟨a,b⟩ + ‖a‖²‖b‖²
        gyr[a,b]v = v + (2/D){ [(1+2⟨a,b⟩)⟨b,v⟩ − ‖b‖²⟨a,v⟩] a
                               − [⟨a,v⟩ + ‖a‖²⟨b,v⟩] b }

    For general ``s`` every inner product / squared norm is scaled by
    ``1/s²`` ("Conventions"):  ``⟨·,·⟩ → ⟨·,·⟩/s²``,
    ``‖·‖² → ‖·‖²/s²``, and ``D = 1 + 2⟨a,b⟩/s² + ‖a‖²‖b‖²/s⁴``.

    The result is an orthogonal map (a rotation in the ``a``–``b`` plane,
    Thomas precession): ``det = +1``, fixes ``span{a,b}^⊥``.

    Parameters
    ----------
    a, b, v : (..., d)  points in the ball (batched over leading axes).
    s : float           ball radius (default 1.0).

    Returns
    -------
    (..., d) array — ``gyr[a,b] v``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    s2 = s * s

    ab = np.sum(a * b, axis=-1, keepdims=True) / s2
    av = np.sum(a * v, axis=-1, keepdims=True) / s2
    bv = np.sum(b * v, axis=-1, keepdims=True) / s2
    aa = np.sum(a * a, axis=-1, keepdims=True) / s2
    bb = np.sum(b * b, axis=-1, keepdims=True) / s2

    D = 1.0 + 2.0 * ab + aa * bb
    ca = (2.0 / D) * ((1.0 + 2.0 * ab) * bv - bb * av)
    cb = (2.0 / D) * (-av - aa * bv)
    return v + ca * a + cb * b


def exp_map_origin(u: np.ndarray, s: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Origin chart inverse ``exp_o u = s·tanh(‖u‖/s)·u/‖u‖`` (NumPy)."""
    u = np.asarray(u, dtype=float)
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return s * np.tanh(norm / s) * u / norm


def log_map_origin(x: np.ndarray, s: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Origin chart ``log_o x = s·artanh(‖x‖/s)·x/‖x‖`` (NumPy)."""
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    scaled = np.minimum(norm / s, 1.0 - 1e-9)
    return s * np.arctanh(scaled) * x / norm


def conformal_gamma(x: np.ndarray, s: float = 1.0) -> np.ndarray:
    r"""Conformal factor ``γ_x = (1 − ‖x‖²/s²)^{-1/2}`` (NumPy)."""
    x = np.asarray(x, dtype=float)
    norm_sq = np.sum(x * x, axis=-1) / (s * s)
    return 1.0 / np.sqrt(np.clip(1.0 - norm_sq, 1e-12, None))


def transport_rotation(h_x: np.ndarray, h_y: np.ndarray, s: float = 1.0) -> np.ndarray:
    r"""Gyro-transport rotation ``R_xy = gyr[h_y, ⊖h_x]`` (spec §4 / §5.2).

    This is the rotation carried by the sheaf restriction map
    ``F_{x◁y} = K_xy · R_xy``.  Dropping it is the silent metric-flattening
    the spec warns about at the ``phonon_topology`` restriction-map site.

    Returns the *matrix* form ``R ∈ O(d)`` (so it can be composed for
    holonomy), obtained by applying the gyration to each basis vector.
    """
    h_x = np.asarray(h_x, dtype=float)
    h_y = np.asarray(h_y, dtype=float)
    d = h_x.shape[-1]
    # gyr[h_y, -h_x] applied column-wise to I_d gives the rotation matrix.
    eye = np.eye(d)
    # Broadcast: a=h_y, b=-h_x fixed; v ranges over basis columns.
    cols = [gyr(h_y, -h_x, eye[:, k], s=s) for k in range(d)]
    return np.stack(cols, axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Gyration FrustIndex — holonomy of transport rotations (spec §5.5)
# ══════════════════════════════════════════════════════════════════════════════

def loop_holonomy(
    features: np.ndarray,
    loop: Sequence[int],
    s: float = 1.0,
) -> np.ndarray:
    r"""Holonomy of the gyro-transport connection around a closed loop.

        Hol(loop) = ∏_{(x→y) in loop} R_xy,   R_xy = gyr[h_y, ⊖h_x]

    A flat sheaf has ``Hol = I`` for every loop; ``Hol ≠ I`` is the sheaf
    curvature that the spec identifies with the holonomy **FrustIndex**
    (§5.5) — the obstruction to the full Hodge tower, and the carrier of
    allostery.

    Parameters
    ----------
    features : (N, d)  hyperbolic feature field ``{h_x}``.
    loop : sequence of cell indices ``[i0, i1, ..., i0]`` (the closing edge
        ``i_last → i0`` is appended automatically if the loop is open).
    s : float ball radius.

    Returns
    -------
    (d, d) holonomy matrix.
    """
    features = np.asarray(features, dtype=float)
    d = features.shape[1]
    idx = list(loop)
    if idx[0] != idx[-1]:
        idx = idx + [idx[0]]

    Hol = np.eye(d)
    # Compose in path order: first transported edge is applied first (rightmost).
    for x, y in zip(idx[:-1], idx[1:]):
        R = transport_rotation(features[x], features[y], s=s)
        Hol = R @ Hol
    return Hol


def frust_index_gyro(
    features: np.ndarray,
    loops: Sequence[Sequence[int]],
    s: float = 1.0,
) -> Dict[str, object]:
    r"""Gyration (holonomy) FrustIndex over a set of loops (spec §5.5).

    For each loop the holonomy deviation from the identity is measured by
    the geodesic angle of the rotation,

        frust(loop) = ‖log R_Hol‖_F / √2 = |rotation angle|   (in the plane),

    computed stably from the orthogonal matrix via
    ``θ = arccos((tr Hol − (d−2)) / 2)`` for the dominant 2-plane, with a
    Frobenius fallback ``‖Hol − I‖_F`` for higher-rank holonomy.

    This is the gyro-transport counterpart of
    :func:`tope.topology.g_structure.frust_index_cb` (which uses the
    Nijenhuis/Chern obstruction).  They are complementary readings of the
    same frustration; the spec keeps the frustration as a *feature*, not an
    obstruction to remove.

    Returns
    -------
    dict with keys:
        'holonomies'  : list[(d,d)]  per-loop holonomy matrices.
        'frust'       : (L,)         per-loop frustration magnitudes.
        'frust_total' : float        mean frustration (drop-in scalar).
        'is_flat'     : bool         True iff all loops are (numerically) flat.
    """
    holonomies: List[np.ndarray] = []
    frust = np.zeros(len(loops), dtype=float)
    for li, loop in enumerate(loops):
        Hol = loop_holonomy(features, loop, s=s)
        holonomies.append(Hol)
        frust[li] = float(np.linalg.norm(Hol - np.eye(Hol.shape[0]), ord="fro"))

    return {
        "holonomies": holonomies,
        "frust": frust,
        "frust_total": float(frust.mean()) if len(loops) else 0.0,
        "is_flat": bool(np.all(frust < 1e-8)) if len(loops) else True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Torch surface — batched per-edge (gyro-CCANN head, §3)
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    def gyr_torch(a: Tensor, b: Tensor, v: Tensor, s: float = 1.0) -> Tensor:
        r"""Batched autodiff gyration ``gyr[a,b] v`` (spec §4), torch form.

        Identical algebra to :func:`gyr`; inner products are taken over the
        last axis so ``a, b, v`` may carry arbitrary leading (edge / head)
        batch dimensions.  Used for the reverse-logit frame correction
        ``G_ij = gyr[p_i, ⊖q_j]`` in the anisotropic Def-33 block.
        """
        s2 = s * s
        ab = (a * b).sum(dim=-1, keepdim=True) / s2
        av = (a * v).sum(dim=-1, keepdim=True) / s2
        bv = (b * v).sum(dim=-1, keepdim=True) / s2
        aa = (a * a).sum(dim=-1, keepdim=True) / s2
        bb = (b * b).sum(dim=-1, keepdim=True) / s2

        D = (1.0 + 2.0 * ab + aa * bb).clamp(min=1e-12)
        ca = (2.0 / D) * ((1.0 + 2.0 * ab) * bv - bb * av)
        cb = (2.0 / D) * (-av - aa * bv)
        return v + ca * a + cb * b

    def mobius_add(a: Tensor, b: Tensor, s: float = 1.0) -> Tensor:
        """Möbius addition ``a ⊕ b`` on ``B^d_s`` (last-axis batched)."""
        s2 = s * s
        ab = (a * b).sum(dim=-1, keepdim=True) / s2
        aa = (a * a).sum(dim=-1, keepdim=True) / s2
        bb = (b * b).sum(dim=-1, keepdim=True) / s2
        num = (1.0 + 2.0 * ab + bb) * a + (1.0 - aa) * b
        den = (1.0 + 2.0 * ab + aa * bb).clamp(min=1e-12)
        return num / den

    def log_o(x: Tensor, s: float = 1.0, eps: float = 1e-9) -> Tensor:
        """Origin chart ``log_o x = s·artanh(‖x‖/s)·x/‖x‖`` (spec §2)."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        scaled = (norm / s).clamp(max=1.0 - 1e-6)
        return s * torch.atanh(scaled) * x / norm

    def exp_o(u: Tensor, s: float = 1.0, eps: float = 1e-9) -> Tensor:
        """Origin chart inverse ``exp_o u = s·tanh(‖u‖/s)·u/‖u‖`` (spec §2)."""
        norm = u.norm(dim=-1, keepdim=True).clamp(min=eps)
        return s * torch.tanh(norm / s) * u / norm

    def gyrodistance(a: Tensor, b: Tensor, s: float = 1.0, eps: float = 1e-9) -> Tensor:
        r"""Gyrodistance ``d_G(a,b) = 2s·artanh(‖(⊖a)⊕b‖/s)`` (spec §2)."""
        diff = mobius_add(-a, b, s=s)
        n = (diff.norm(dim=-1) / s).clamp(max=1.0 - 1e-6, min=0.0)
        return 2.0 * s * torch.atanh(n)

    def gyrobary(weights: Tensor, points: Tensor, s: float = 1.0) -> Tensor:
        r"""Einstein gyrobarycenter (spec §2), batched.

            gyrobary(w; v) = Σ_j w_j γ_{v_j} v_j / Σ_j w_j γ_{v_j},
            γ_v = (1 − ‖v‖²/s²)^{-1/2}.

        Stays in the ball; → Euclidean weighted mean as ``s → ∞``.

        Parameters
        ----------
        weights : (..., M)     non-negative weights over the M points.
        points  : (..., M, d)  ball points to average.
        """
        s2 = s * s
        norm_sq = (points * points).sum(dim=-1) / s2          # (..., M)
        gamma = (1.0 - norm_sq).clamp(min=1e-9).rsqrt()        # (..., M)
        w = (weights * gamma).unsqueeze(-1)                    # (..., M, 1)
        num = (w * points).sum(dim=-2)                         # (..., d)
        den = w.sum(dim=-2).clamp(min=1e-12)                   # (..., 1)
        return num / den

    def mobius_matmul(W: Tensor, x: Tensor, s: float = 1.0) -> Tensor:
        r"""Möbius matrix multiplication ``W ⊗_M x = exp_o(W log_o x)`` (§2).

        ``W`` is ``(d_out, d_in)``; ``x`` is ``(..., d_in)``.
        """
        u = log_o(x, s=s)                       # (..., d_in)
        Wu = u @ W.transpose(-1, -2)            # (..., d_out)
        return exp_o(Wu, s=s)
