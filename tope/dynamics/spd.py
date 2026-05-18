"""SPD-manifold primitives for the dynamics layer.

Three small toolkits, designed to be composed by ``nma`` and
``free_energy`` so the rank-1 ANM degeneracy and the hand-tuned
unfolded-baseline both go away by construction.

* ``LogEuclideanContact`` — parametrise each contact tensor as
  ``Γ = expm(L)`` with ``L ∈ Sym(3)``. Gradients (when wired into a
  torch model) flow freely through the matrix exponential while the
  output is always SPD. Replaces the scalar-spring ``(γ/d²) r⊗r``
  parametrisation that produces zero transverse stiffness for
  collinear chains.

* ``gaussian_chain_covariance`` — Σ_unfold for an ideal Gaussian chain
  on ``n`` residues. SPD on the ``3(n−1)``-dimensional non-translation
  subspace; built from the path-graph Laplacian.

* ``relative_entropy_spd`` / ``log_det_ratio_from_eigvals`` — intrinsic
  entropy difference between two Gaussian states living on the same
  SPD manifold. Replaces ``S_folded`` (Schlitter, harmonic) minus
  ``S_unfolded`` (per-residue Flory constant) with a single log-det
  ratio whose sign is determined by Loewner order.

Units convention
----------------
Everything in this module assumes a *single consistent* energy/length
scale set by the caller. ``segment_var_A2`` is in Å²; force constants
fed via ``LogEuclideanContact`` should produce Hessian eigenvalues in
the same energy/length² unit as the folded Hessian, so that log-det
ratios come out dimensionless. The scaffold does not enforce this —
it's the caller's responsibility (and the place where a future
supervised head needs to be calibrated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import expm, logm


# ── Log-Euclidean contact parametrisation ────────────────────────────────────

@dataclass
class LogEuclideanContact:
    """Per-bond SPD contact tensor parametrised on Sym(3).

    Stores the symmetric matrix log ``L`` and returns ``Γ = expm(L) ∈
    SPD(3)`` on demand. ``L`` defaults to the symmetric log of γ·I,
    which reduces to the scalar-spring case for backward compatibility.

    For learned-parameter use, treat ``L`` as the free variable (vector
    space, autograd-friendly) and read ``.tensor`` to get the SPD
    operator that lands in the Hessian.
    """

    L: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    @classmethod
    def isotropic(cls, gamma: float) -> "LogEuclideanContact":
        """Convenience: isotropic contact with stiffness γ on all axes."""
        return cls(L=np.log(max(gamma, 1e-30)) * np.eye(3))

    @classmethod
    def from_spd(cls, gamma_spd: np.ndarray) -> "LogEuclideanContact":
        """Recover the log-Euclidean parameter from an SPD matrix."""
        return cls(L=np.asarray(logm(gamma_spd), dtype=np.float64).real)

    @property
    def tensor(self) -> np.ndarray:
        """The SPD contact operator ``Γ = expm(L)``."""
        L_sym = 0.5 * (self.L + self.L.T)
        return np.asarray(expm(L_sym), dtype=np.float64).real


def isotropic_contact_tensors(n_pairs: int, gamma: float = 1.0) -> np.ndarray:
    """Return ``(n_pairs, 3, 3)`` array of γ·I — the scalar-ANM limit."""
    block = gamma * np.eye(3)
    return np.broadcast_to(block, (n_pairs, 3, 3)).copy()


# ── Gaussian-chain unfolded covariance ────────────────────────────────────────

def path_laplacian(n: int) -> np.ndarray:
    """Combinatorial Laplacian of the path graph on ``n`` nodes."""
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A[1:, :-1], 1.0)
    np.fill_diagonal(A[:-1, 1:], 1.0)
    D = np.diag(A.sum(axis=1))
    return D - A


def gaussian_chain_covariance(
    n_residues: int,
    segment_var_A2: float = 13.0,
    exclude_translation: bool = True,
) -> np.ndarray:
    """Σ_unfold for an ideal Gaussian chain on n residues.

    Each residue–residue bond has displacement variance ``σ² = segment_var_A2``
    per spatial coordinate (~13 Å² is the textbook ideal-chain value).
    The chain's effective Hessian is the path-graph Laplacian rescaled
    by ``1/σ²``; the covariance is its Moore-Penrose pseudoinverse
    tensored with ``I_3``, optionally restricted to the non-translation
    subspace.

    Returns
    -------
    Σ : (3n × 3n) ndarray, SPSD with 3 translation null directions
        unless ``exclude_translation`` is True, in which case the
        translation block is projected out and the matrix is SPD on
        rank ``3(n − 1)``.
    """
    n = int(n_residues)
    if n < 2:
        raise ValueError("Gaussian chain needs at least 2 residues")

    L = path_laplacian(n)
    H_chain = (1.0 / segment_var_A2) * L
    sigma_1d = np.linalg.pinv(H_chain)        # (n, n)
    sigma = np.kron(sigma_1d, np.eye(3))      # (3n, 3n), 3 zero modes

    if not exclude_translation:
        return sigma

    # Project out the 3 translational modes (constant vector ⊗ I₃).
    ones = np.ones(n) / np.sqrt(n)
    P_trans = np.kron(np.outer(ones, ones), np.eye(3))
    proj = np.eye(3 * n) - P_trans
    return proj @ sigma @ proj


# ── Intrinsic entropy / log-det machinery ────────────────────────────────────

def relative_entropy_spd(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """KL divergence of N(0, Σ₁) ‖ N(0, Σ₂) on a common subspace.

    ``KL = ½ [tr(Σ₂⁻¹ Σ₁) − d + log(det Σ₂ / det Σ₁)]``

    Both inputs are assumed SPD on the same support; eigenvalues below
    ``floor`` are clipped (the caller is responsible for projecting
    onto a common non-rigid subspace beforehand).
    """
    sigma1 = 0.5 * (sigma1 + sigma1.T)
    sigma2 = 0.5 * (sigma2 + sigma2.T)
    d = sigma1.shape[0]

    eigs1 = np.maximum(np.linalg.eigvalsh(sigma1), floor)
    eigs2 = np.maximum(np.linalg.eigvalsh(sigma2), floor)
    sigma2_inv = np.linalg.pinv(sigma2)
    tr_term = float(np.trace(sigma2_inv @ sigma1))
    log_det = float(np.log(eigs2).sum() - np.log(eigs1).sum())
    return 0.5 * (tr_term - d + log_det)


def log_det_ratio_from_eigvals(
    eigs_num: np.ndarray,
    eigs_den: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """``log(Π eigs_num / Π eigs_den)`` with a numerical floor.

    Useful when the eigenvalue spectra of two Hessians are already in
    hand and you don't need the full SPD matrices. Handles spectra of
    unequal length by truncating to the shorter — paired *largest to
    smallest* so the comparison is between corresponding "stiff" modes.
    """
    a = np.sort(np.maximum(np.asarray(eigs_num, dtype=np.float64), floor))[::-1]
    b = np.sort(np.maximum(np.asarray(eigs_den, dtype=np.float64), floor))[::-1]
    k = min(a.shape[0], b.shape[0])
    return float(np.log(a[:k]).sum() - np.log(b[:k]).sum())


def gaussian_entropy_change_from_hessians(
    H_fold_eigs: np.ndarray,
    H_unfold_eigs: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """Intrinsic ΔS_unfold ∝ ½ log(det Σ_unfold / det Σ_fold).

    Since ``Σ = H⁺``, the log-det ratio in covariances becomes a log-det
    ratio of Hessian eigenvalues with sign flipped:

        ½ log(det Σ_unf / det Σ_fold)
            = ½ log(Π λ(H_fold) / Π λ(H_unfold))

    Returned in units of ``log`` (multiply by ``k_B`` to get entropy in
    energy/T units).
    """
    return 0.5 * log_det_ratio_from_eigvals(H_fold_eigs, H_unfold_eigs, floor=floor)
