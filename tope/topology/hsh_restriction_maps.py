"""
HSH Restriction Maps  —  Phase 2 upgrade for ToPE sheaf restriction maps.

Replaces the rank-1 outer-product restriction map

        R_ij = (G δ) (G δ)^T / (δ^T G δ)

with a learnable expansion in the 36-element basis of Sym(R^8) obtained from
the SO(8) Hyperspherical Harmonics of degrees λ=0 and λ=2 (Avery 1994).

Counts (Avery §2):
    N(d=8, λ=0) = 1     isotropic   (identity component)
    N(d=8, λ=2) = 35    traceless symmetric   (orbital anisotropy)
                  → 1 + 35 = 36 = dim Sym(R^8)

The λ=1 antisymmetric subspace so(8) (dim 28) is excluded — symmetric
descriptor differences yield self-adjoint restriction maps.

References
----------
Avery, J. (1994). "Hyperspherical Harmonics: Some Properties and Applications."
Cerrini, S. (1971). Acta Cryst. A27, 130.
Hirshfeld, F. L. (1976). Acta Cryst. A32, 239.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # numpy-only inference path remains usable
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────────
# 1. Counting and Gegenbauer recursion (Avery §2-3)
# ──────────────────────────────────────────────────────────────────────────────

def n_hsh(d: int, lam: int) -> int:
    """Dimension of the space H_λ^d of degree-λ harmonics on S^{d-1}.

    Avery (1994) eq. for N(d,λ):
        N(d, λ) = C(d+λ-1, λ) - C(d+λ-3, λ-2)
    with the convention that the second term vanishes for λ < 2.
    """
    if lam < 0:
        return 0
    first = math.comb(d + lam - 1, lam)
    second = math.comb(d + lam - 3, lam - 2) if lam >= 2 else 0
    return first - second


def gegenbauer(n: int, alpha: float, t: float) -> float:
    """Gegenbauer C_n^α(t) via three-term recursion (Avery eq. 1)."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * alpha * t
    c0, c1 = 1.0, 2.0 * alpha * t
    for k in range(1, n):
        c2 = (2.0 * (k + alpha) * t * c1 - (k + 2.0 * alpha - 1.0) * c0) / (k + 1)
        c0, c1 = c1, c2
    return c1


def gegenbauer_vec(n: int, alpha: float, t: np.ndarray) -> np.ndarray:
    """Vectorised Gegenbauer evaluation over an array of arguments."""
    t = np.asarray(t, dtype=float)
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return 2.0 * alpha * t
    c0 = np.ones_like(t)
    c1 = 2.0 * alpha * t
    for k in range(1, n):
        c2 = (2.0 * (k + alpha) * t * c1 - (k + 2.0 * alpha - 1.0) * c0) / (k + 1)
        c0, c1 = c1, c2
    return c1


def surface_area_sphere(d: int) -> float:
    """Ω_d = surface area of S^{d-1} in R^d:   2 π^{d/2} / Γ(d/2)."""
    return 2.0 * math.pi ** (d / 2.0) / math.gamma(d / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Basis construction for Sym(R^d), λ ∈ {0, 2}
# ──────────────────────────────────────────────────────────────────────────────

def compute_hsh_basis_matrices(
    d: int = 8,
    lambda_max: int = 2,
) -> Tuple[np.ndarray, List[str]]:
    """Construct the Frobenius-orthonormal HSH basis of Sym(R^d).

    Returns
    -------
    basis : (n_basis, d, d) ndarray
        Stack of basis matrices. For ``d=8, lambda_max=2`` shape is (36, 8, 8).
    labels : list[str]
        Human-readable label per basis element (e.g. ``"T(0,0)"``,
        ``"T(2,off,2,4)"``, ``"T(2,diag,3)"``).

    Notes
    -----
    Only λ ∈ {0, 2} are populated; ``lambda_max`` is accepted for forward
    compatibility (λ=1 antisymmetric and λ≥3 higher-order are out of scope).
    For ``lambda_max=0`` only the identity element is returned (rank-0 / pure
    isotropic basis), which is what the isotropic-only fallback uses.
    """
    if lambda_max not in (0, 2):
        raise ValueError(
            f"lambda_max={lambda_max} not implemented; supported: 0, 2"
        )

    basis: List[np.ndarray] = []
    labels: List[str] = []

    # λ = 0  —  one element, normalised identity
    T00 = np.eye(d) / math.sqrt(d)
    basis.append(T00)
    labels.append("T(0,0)")

    if lambda_max >= 2:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        # λ = 2 off-diagonal:  C(d,2) elements
        for i in range(d):
            for j in range(i + 1, d):
                T = np.zeros((d, d))
                T[i, j] = inv_sqrt2
                T[j, i] = inv_sqrt2
                basis.append(T)
                labels.append(f"T(2,off,{i},{j})")

        # λ = 2 diagonal traceless:  d-1 elements.
        # Gell-Mann-style generalised diagonal generators, which (unlike the
        # naive (e_k e_k^T − e_{k+1} e_{k+1}^T)/√2 choice) are mutually
        # Frobenius-orthogonal across different k:
        #   λ_k = (1/√(k(k+1))) · diag(1,…,1, −k, 0,…,0)
        # with k ones followed by −k at position k and zeros after.
        # Properties:  tr(λ_k) = 0, tr(λ_k λ_l) = δ_{kl}.
        for k in range(1, d):
            T = np.zeros((d, d))
            norm = math.sqrt(k * (k + 1))
            for i in range(k):
                T[i, i] = 1.0 / norm
            T[k, k] = -float(k) / norm
            basis.append(T)
            labels.append(f"T(2,diag,{k})")

    return np.stack(basis, axis=0), labels


def hsh_coefficients(
    delta_hat: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Expansion coefficients c_k = tr(T_k^T (δ̂⊗δ̂)) = δ̂^T T_k δ̂."""
    return np.einsum("kij,i,j->k", basis, delta_hat, delta_hat)


def parseval_check(delta_hat: np.ndarray, basis: np.ndarray) -> float:
    """||δ̂⊗δ̂||_F^2 = Σ_k c_k^2 should equal 1 for unit δ̂.

    Returns the sum Σ c_k^2.  Test code should assert closeness to 1.
    """
    c = hsh_coefficients(delta_hat, basis)
    return float(np.sum(c * c))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Addition theorem (Avery §4)
# ──────────────────────────────────────────────────────────────────────────────

def zonal_hsh_kernel(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    d: int,
    lam: int,
) -> float:
    """Zonal kernel  Σ_μ Y_{λμ}(x̂) Y_{λμ}(ŷ) = N(d,λ)/Ω_d · C_λ^{(d-2)/2}(x̂·ŷ).

    This is the right-hand side of the HSH addition theorem (Avery eq. for the
    generalised addition formula).  Used by :func:`verify_addition_theorem`
    to confirm that the discrete λ=2 basis reproduces the continuous kernel.
    """
    alpha = (d - 2) / 2.0
    cos_theta = float(np.dot(x_hat, y_hat))
    return (n_hsh(d, lam) / surface_area_sphere(d)) * gegenbauer(lam, alpha, cos_theta)


def verify_addition_theorem(
    d: int = 8,
    lam: int = 2,
    n_samples: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo check of the addition theorem at degree λ.

    The discrete basis projection
        K_disc(x̂, ŷ) = Σ_k_in_λ  (x̂^T T_k x̂)(ŷ^T T_k ŷ)
    equals  ||proj_λ(x̂⊗x̂ : ŷ⊗ŷ)|| = the Frobenius inner product of the
    λ-projections of the two rank-1 outer products.  For our particular basis
    of Sym(R^d) the λ=2 subspace projection of u u^T is
        u u^T  −  (u^T u / d) I.
    So  Σ_k c_k(x) c_k(y) = tr[(x x^T − I/d)(y y^T − I/d)] for unit vectors,
                       = (x·y)^2 − 1/d.
    This is, up to a normalisation constant, what the Gegenbauer kernel gives.
    We compare ratios so that absolute normalisation cancels.

    Returns
    -------
    mae : float
        Mean absolute error between the discrete projection and the
        Gegenbauer kernel after matching their normalisations.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    basis, _ = compute_hsh_basis_matrices(d=d, lambda_max=2)

    # Indices of λ=2 basis elements (all except the very first, which is λ=0).
    if lam == 0:
        sel = slice(0, 1)
    elif lam == 2:
        sel = slice(1, basis.shape[0])
    else:
        raise ValueError(f"lam={lam} not in supported set {{0, 2}}")
    B_lam = basis[sel]

    # Sample unit vectors on S^{d-1}
    X = rng.standard_normal((n_samples, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n_samples, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    # Discrete projection coefficients
    cx = np.einsum("kij,ni,nj->nk", B_lam, X, X)
    cy = np.einsum("kij,ni,nj->nk", B_lam, Y, Y)
    K_disc = np.einsum("nk,nk->n", cx, cy)

    # Closed-form continuous kernel (up to the Ω_d / N(d,λ) prefactor which
    # only rescales — we match by the leading constant via least-squares).
    alpha = (d - 2) / 2.0
    cos_theta = np.einsum("ni,ni->n", X, Y)
    K_cont = gegenbauer_vec(lam, alpha, cos_theta)

    # Match scale: K_disc ≈ s · K_cont.  Use ratio of means of squares to
    # avoid singularities at K_cont = 0 (which occurs at the Gegenbauer roots).
    s = float(np.dot(K_disc, K_cont) / max(np.dot(K_cont, K_cont), 1e-30))
    mae = float(np.mean(np.abs(K_disc - s * K_cont)))
    return mae


# ──────────────────────────────────────────────────────────────────────────────
# 4. Numpy inference path
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HSHConfig:
    """Configuration for the HSH restriction map."""
    d: int = 8
    lambda_max: int = 2
    init_mode: str = "rank1"            # "rank1" | "isotropic" | "random"
    sparsity_lambda: float = 0.0        # L1 on λ=2 weights (training-side)


class HSHSheafLaplacian:
    """Pure-numpy HSH restriction-map builder for inference.

    Drop-in replacement for the rank-1 outer-product block inside
    :class:`tope.topology.phonon_topology.SheafENM`.  Holds 36 scalar weights
    that are normally synced from a trained :class:`HSHRestrictionMap`
    (PyTorch) but can also be used standalone with the
    backward-compatible ``init_mode="rank1"`` default.

    Initialisation modes
    --------------------
    ``"rank1"`` (default)
        All 36 weights = 1.  By basis completeness this reproduces the
        current Cerrini outer-product map  R_ij = δ̂_G ⊗ δ̂_G  exactly
        (modulo the final ‖·‖_F normalisation, which is a no-op since
        ‖δ̂_G ⊗ δ̂_G‖_F = 1 for unit δ̂_G).  Use this for zero-diff
        integration tests.

    ``"isotropic"``
        Only w_0=1, rest = 0.  Map is proportional to I — does *not*
        reproduce the current rank-1 behaviour, but is the natural
        "no orbital preference" starting point if you intend to learn
        the anisotropic weights from scratch.

    ``"random"``
        Small Gaussian (σ=0.01) around w_0=1.

    Parameters
    ----------
    d : int
        Stalk dimension (8 for Ioffe descriptors).
    descriptor_metric : (d, d) ndarray, optional
        Cerrini metric G applied to δ before normalising.  Defaults to I.
    config : HSHConfig
    """

    def __init__(
        self,
        d: int = 8,
        descriptor_metric: Optional[np.ndarray] = None,
        config: Optional[HSHConfig] = None,
    ):
        self.config = config or HSHConfig(d=d)
        if self.config.d != d:
            self.config = HSHConfig(
                d=d,
                lambda_max=self.config.lambda_max,
                init_mode=self.config.init_mode,
                sparsity_lambda=self.config.sparsity_lambda,
            )
        self.d = d
        self.descriptor_metric = (
            descriptor_metric if descriptor_metric is not None else np.eye(d)
        )
        self.basis, self.labels = compute_hsh_basis_matrices(
            d=d, lambda_max=self.config.lambda_max
        )
        self.n_basis = self.basis.shape[0]
        self.weights = self._init_weights(self.config.init_mode)

    def _init_weights(self, mode: str) -> np.ndarray:
        if mode == "rank1":
            return np.ones(self.n_basis)
        if mode == "isotropic":
            w = np.zeros(self.n_basis)
            w[0] = 1.0
            return w
        if mode == "random":
            w = np.zeros(self.n_basis)
            w[0] = 1.0
            w[1:] = 0.01 * np.random.default_rng(0).standard_normal(self.n_basis - 1)
            return w
        raise ValueError(f"Unknown init_mode={mode!r}")

    def restriction_map(
        self,
        s_i: np.ndarray,
        s_j: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Build R_ij from two sheaf sections via the metric-aware HSH expansion.

        Steps (per spec §4.4):
            δ      = s_i − s_j
            G_δ    = G δ                 (covariant descriptor difference)
            inner  = δ^T G δ
            δ̂_G   = G_δ / √inner         (metric-normalised direction)
            δ̂     = δ̂_G / ‖δ̂_G‖         (Euclidean unit vector — the spec's
                                            second normalisation; required so
                                            that the basis expansion lives on
                                            S^{d-1} in the Euclidean sense)
            c_k    = δ̂^T T_k δ̂
            F      = Σ_k (w_k · c_k) · T_k
            R_ij   = F / ‖F‖_F          (if normalize)
        """
        delta = s_i - s_j
        G = self.descriptor_metric
        G_delta = G @ delta
        inner = float(delta @ G_delta) + 1e-10
        delta_hat_G = G_delta / math.sqrt(inner)
        norm_G = float(np.linalg.norm(delta_hat_G)) + 1e-10
        delta_hat = delta_hat_G / norm_G

        c = np.einsum("kij,i,j->k", self.basis, delta_hat, delta_hat)
        F = np.einsum("k,kij->ij", c * self.weights, self.basis)
        if normalize:
            F = F / (np.linalg.norm(F) + 1e-10)
        return F

    def build_restriction_maps(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
        edge_iter: Iterable[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Build the dict {(i,j) → R_ij} for the given edge list.

        ``edge_iter`` is supplied externally (the Hodge Laplacian sparsity
        pattern is owned by :class:`SheafENM`).  This keeps the class free of
        any direct coupling to the rest of the topology stack.
        """
        R: Dict[Tuple[int, int], np.ndarray] = {}
        for (i, j) in edge_iter:
            if i == j:
                continue
            if i in sheaf_sections and j in sheaf_sections:
                R[(i, j)] = self.restriction_map(
                    sheaf_sections[i], sheaf_sections[j]
                )
        return R

    # ── Introspection helpers (for monitoring during training) ────────────

    def isotropy_fraction(self) -> float:
        """w_0^2 / Σ w_k^2 — fraction of weight on the λ=0 component."""
        total = float(np.sum(self.weights ** 2)) + 1e-30
        return float(self.weights[0] ** 2 / total)

    def top_anisotropic_terms(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k λ=2 basis elements by |weight|."""
        w_aniso = self.weights[1:]
        labels_aniso = self.labels[1:]
        order = np.argsort(-np.abs(w_aniso))[:k]
        return [(labels_aniso[i], float(w_aniso[i])) for i in order]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Learnable PyTorch module
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_TORCH:

    class HSHRestrictionMap(nn.Module):
        """Learnable HSH restriction map shared across all edges at a given rank.

        forward(s_i, s_j) → (d, d) tensor
        batch_forward(sections, edges) → dict[(i,j) → (d,d) tensor]

        The 36 weights are global per face-relation rank — the physics says
        the orbital coupling structure is an intrinsic property of the enzyme
        class, not of each individual bond.
        """

        def __init__(
            self,
            d: int = 8,
            descriptor_metric: Optional[np.ndarray] = None,
            config: Optional[HSHConfig] = None,
        ):
            super().__init__()
            self.config = config or HSHConfig(d=d)
            self.d = d
            basis_np, labels = compute_hsh_basis_matrices(
                d=d, lambda_max=self.config.lambda_max
            )
            self.labels = labels
            # Non-trainable basis tensor
            self.register_buffer(
                "basis", torch.from_numpy(basis_np.astype(np.float32))
            )
            G = descriptor_metric if descriptor_metric is not None else np.eye(d)
            self.register_buffer(
                "G", torch.from_numpy(G.astype(np.float32))
            )
            # Trainable expansion weights
            init = HSHSheafLaplacian(d=d, config=self.config)._init_weights(
                self.config.init_mode
            )
            self.weights = nn.Parameter(
                torch.from_numpy(init.astype(np.float32))
            )

        def _direction(self, s_i: "torch.Tensor", s_j: "torch.Tensor") -> "torch.Tensor":
            delta = s_i - s_j
            G_delta = self.G @ delta
            inner = torch.dot(delta, G_delta) + 1e-10
            delta_hat_G = G_delta / torch.sqrt(inner)
            return delta_hat_G / (torch.linalg.norm(delta_hat_G) + 1e-10)

        def forward(
            self,
            s_i: "torch.Tensor",
            s_j: "torch.Tensor",
            normalize: bool = True,
        ) -> "torch.Tensor":
            delta_hat = self._direction(s_i, s_j)
            c = torch.einsum("kij,i,j->k", self.basis, delta_hat, delta_hat)
            F = torch.einsum("k,kij->ij", c * self.weights, self.basis)
            if normalize:
                F = F / (torch.linalg.norm(F) + 1e-10)
            return F

        def batch_forward(
            self,
            sections: Dict[int, "torch.Tensor"],
            edges: Iterable[Tuple[int, int]],
        ) -> Dict[Tuple[int, int], "torch.Tensor"]:
            out: Dict[Tuple[int, int], "torch.Tensor"] = {}
            for (i, j) in edges:
                if i == j or i not in sections or j not in sections:
                    continue
                out[(i, j)] = self.forward(sections[i], sections[j])
            return out

        def sync_to_numpy(self, target: HSHSheafLaplacian) -> None:
            """Copy trained weights into a numpy ``HSHSheafLaplacian`` for inference."""
            target.weights = self.weights.detach().cpu().numpy().astype(float)

        def sparsity_penalty(self) -> "torch.Tensor":
            """L1 penalty on λ=2 weights (optional interpretability regulariser)."""
            lam = self.config.sparsity_lambda
            if lam == 0.0:
                return torch.zeros((), device=self.weights.device)
            return lam * torch.norm(self.weights[1:], p=1)

else:  # pragma: no cover  — torch import failed
    HSHRestrictionMap = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# 6. Self-contained validation suite (spec §5)
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(d: int = 8, verbose: bool = True) -> bool:
    """Run all six validation tests; return True iff all pass."""
    results: Dict[str, Tuple[bool, str]] = {}

    # Test 1 — basis shape
    basis, labels = compute_hsh_basis_matrices(d=d, lambda_max=2)
    expected = 1 + (d * (d - 1) // 2) + (d - 1)
    ok = basis.shape == (expected, d, d) and len(labels) == expected
    results["basis_shape"] = (ok, f"shape={basis.shape}")

    # Test 2 — Frobenius orthonormality
    gram = np.einsum("kij,lij->kl", basis, basis)
    off_diag_max = float(np.max(np.abs(gram - np.eye(expected))))
    ok = off_diag_max < 1e-12
    results["frobenius_orthonormality"] = (ok, f"max|G-I|={off_diag_max:.2e}")

    # Test 3 — Parseval completeness
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((100, d))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    max_err = 0.0
    for v in vecs:
        max_err = max(max_err, abs(parseval_check(v, basis) - 1.0))
    ok = max_err < 1e-12
    results["parseval"] = (ok, f"max|Σc²-1|={max_err:.2e}")

    # Test 4 — isotropic limit (lambda_max=0): F ∝ I for any δ
    basis_iso, _ = compute_hsh_basis_matrices(d=d, lambda_max=0)
    iso_max_err = 0.0
    for v in vecs[:10]:
        c = hsh_coefficients(v, basis_iso)
        F = np.einsum("k,kij->ij", c, basis_iso)
        # Should be proportional to I — off-diagonals zero
        off = F - np.diag(np.diag(F))
        diag_var = float(np.var(np.diag(F)))
        iso_max_err = max(iso_max_err, float(np.max(np.abs(off))), diag_var)
    ok = iso_max_err < 1e-12
    results["isotropic_limit"] = (ok, f"max(off, var(diag))={iso_max_err:.2e}")

    # Test 5 — Gegenbauer C_2^3 at t = 1, 0, -1
    vals = {t: gegenbauer(2, 3.0, t) for t in (1.0, 0.0, -1.0)}
    ok = (
        abs(vals[1.0] - 21.0) < 1e-12
        and abs(vals[0.0] - (-3.0)) < 1e-12
        and abs(vals[-1.0] - 21.0) < 1e-12
    )
    results["gegenbauer_values"] = (ok, f"{vals}")

    # Test 6 — addition theorem Monte Carlo
    mae_lam0 = verify_addition_theorem(d=d, lam=0, n_samples=2000)
    mae_lam2 = verify_addition_theorem(d=d, lam=2, n_samples=5000)
    ok = mae_lam0 < 1e-10 and mae_lam2 < 1e-10
    results["addition_theorem"] = (
        ok, f"MAE λ=0: {mae_lam0:.2e}, λ=2: {mae_lam2:.2e}"
    )

    all_ok = all(v[0] for v in results.values())
    if verbose:
        for name, (passed, msg) in results.items():
            marker = "✓" if passed else "✗"
            print(f"  {marker} {name:30s}  {msg}")
        print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


__all__ = [
    "n_hsh",
    "gegenbauer",
    "gegenbauer_vec",
    "surface_area_sphere",
    "compute_hsh_basis_matrices",
    "hsh_coefficients",
    "parseval_check",
    "zonal_hsh_kernel",
    "verify_addition_theorem",
    "HSHConfig",
    "HSHSheafLaplacian",
    "HSHRestrictionMap",
    "run_validation",
]
