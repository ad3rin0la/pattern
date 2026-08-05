"""
TLS Crystallographic Compression for ToPE
==========================================

Implements the Translation-Libration-Screw (TLS) model for compressing
atomic ADP (Anisotropic Displacement Parameter) fields.

The TLS model represents the full atomic ADP field of a rigid subunit using
exactly 20 parameters. For atom i at position r_i relative to the TLS origin:

    U_i = T + L×(r_i⊗r_i) + S×r_i + r_i^T×S^T

where T (6 params, SPD), L (6 params, SPD), S (8 params after trace constraint).

Compression consequences:
- Subunit ADP field (N_atoms × 6 floats) → 20 floats: ~150× for N_atoms=500
- Tucker rank r_0 ≤ 6 (ADP tensor components, not ~20 orbital types)
- Hirshfeld rigid-bond constraint: 1 DOF eliminated per intra-subunit bond

Cerrini (1971) TTN leaf structure:
- Leaf 0: Isotropic B-factor (l=0) — 1 param
- Leaf 1: Translation tensor T (l=0 + l=2) — 6 params
- Leaf 2: Libration tensor L (l=0 + l=2) — 6 params
- Leaf 3: Screw correlation S (l=1 + l=2 mixed) — 8 params

References
----------
Schomaker & Trueblood 1968 — TLS model definition.
Cerrini 1971 — Tensor analysis of harmonic vibrations (Acta Cryst A27).
Hirshfeld 1976 — Rigid-bond test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

warnings.warn(
    "tope.compression.tls_compression (Phase 5B/5G — TLS crystallographic "
    "compression) is deferred.  Correct TLS → Tucker rank reduction requires "
    "geometry-dependent VOIP tensors from Phase 5A, which has not yet been "
    "validated.  Do not integrate this module into the training pipeline until "
    "Phase 5A ablations pass.",
    FutureWarning,
    stacklevel=2,
)


# ── TLS physics helpers ───────────────────────────────────────────────────────

def _tls_predict_adp(
    T: np.ndarray,   # (3,3) SPD translation tensor
    L: np.ndarray,   # (3,3) SPD libration tensor
    S: np.ndarray,   # (3,3) screw correlation (traceless)
    r: np.ndarray,   # (3,) atom position relative to TLS origin
) -> np.ndarray:
    """Predict ADP tensor U_i from TLS parameters.

    U_i = T + ε_kij * L_jm * r_m + ε_kij * S_jk  (Schomaker & Trueblood 1968)
    In matrix form: U_i = T + L×(r⊗r) + sym(S×r)

    Parameters
    ----------
    T, L : (3,3) SPD matrices
    S    : (3,3) matrix (tr(S) = 0 constraint)
    r    : (3,) atom position

    Returns
    -------
    U_i : (3,3) predicted ADP tensor (symmetric)
    """
    # Libration contribution: L_kl * r_k * r_l - 0.5 * tr(L) * I + ...
    # Simplified Schomaker-Trueblood formula:
    r_cross = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0],
    ])
    # U = T - R^T L R + sym(R^T S)  where R = r_cross
    libration_term = -r_cross.T @ L @ r_cross
    screw_term = (r_cross.T @ S + S.T @ r_cross) * 0.5
    U = T + libration_term + screw_term
    return (U + U.T) * 0.5  # symmetrize


def _fit_tls_least_squares(
    coords: np.ndarray,      # (N, 3)
    adp_tensors: np.ndarray, # (N, 3, 3)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit TLS parameters via least-squares.

    Builds the normal equations for the linear TLS model:
        vec(U_i) = A_i * [vec(T); vec(L); vec(S)]

    Parameters
    ----------
    coords      : (N, 3) atom coordinates (relative to TLS origin = mean)
    adp_tensors : (N, 3, 3) observed ADP tensors

    Returns
    -------
    T, L, S : fitted TLS matrices
    """
    N = len(coords)
    if N == 0:
        return np.eye(3) * 0.01, np.eye(3) * 0.01, np.zeros((3, 3))

    # Build design matrix: each atom contributes 6 equations (symmetric U_i)
    # Parameters: T (6), L (6), S (8 = 9 - 1 trace constraint) = 20 total
    n_params = 20
    A_rows = []
    b_rows = []

    for i in range(N):
        r = coords[i]
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0],
        ])
        # For each symmetric element (k,l) with k <= l
        for k in range(3):
            for l in range(k, 3):
                row = np.zeros(n_params)
                # T contribution: T_kl = delta_{kl} T_kk + ... (symmetric)
                T_idx = k * 3 + l - (k * (k + 1) // 2)  # map to upper triangular
                row[T_idx] = 1.0  # T_{kl}

                # L contribution: -(R^T L R)_{kl}
                # (R^T L R)_{kl} = sum_mn R^T_{km} L_{mn} R_{nl}
                # = sum_mn (-R_{mk}) L_{mn} R_{nl}
                for m in range(3):
                    for n in range(3):
                        L_idx = 6 + m * 3 + n - (m * (m + 1) // 2)
                        if m <= n:
                            row[L_idx] += -r_cross.T[k, m] * r_cross[n, l]

                # S contribution: sym(R^T S)_{kl} = 0.5*(R^T S + S^T R)_{kl}
                for m in range(3):
                    for n in range(3):
                        # S has 8 params (trace-free): S_{mn} for all (m,n) except
                        # constrained by tr(S) = 0 → S_{22} = -(S_{00} + S_{11})
                        if m == 2 and n == 2:
                            continue  # constrained
                        s_idx = 12 + m * 3 + n
                        if s_idx < 20:
                            row[s_idx] += 0.5 * (r_cross.T[k, m] + r_cross.T[l, m]) * (n == l) * 0.5

                U_obs = adp_tensors[i, k, l]
                A_rows.append(row)
                b_rows.append(U_obs)

    if not A_rows:
        return np.eye(3) * 0.01, np.eye(3) * 0.01, np.zeros((3, 3))

    A = np.array(A_rows)   # (N*6, 20)
    b = np.array(b_rows)   # (N*6,)

    # Least-squares: min ||A x - b||^2
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Unpack
    T_vec = x[:6]
    L_vec = x[6:12]
    S_vec = x[12:20]

    # Build symmetric T, L
    T = np.zeros((3, 3))
    L = np.zeros((3, 3))
    S = np.zeros((3, 3))

    idx = 0
    for k in range(3):
        for l in range(k, 3):
            T[k, l] = T[l, k] = T_vec[idx]
            L[k, l] = L[l, k] = L_vec[idx]
            idx += 1

    # S (8 free params + 1 trace constraint)
    s_idx = 0
    for m in range(3):
        for n in range(3):
            if m == 2 and n == 2:
                S[2, 2] = -(S[0, 0] + S[1, 1])
            elif s_idx < 8:
                S[m, n] = S_vec[s_idx]
                s_idx += 1

    # Ensure T, L are SPD (clamp negative eigenvalues)
    for M in [T, L]:
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-6)
        M[:] = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return T, L, S


# ── TLSGroup ──────────────────────────────────────────────────────────────────

class TLSGroup:
    """Stores T, L, S matrices for one rigid subunit.

    Storage budget: 20 floats × 4 bytes = 80 bytes per subunit.

    Parameters
    ----------
    T : (3,3) SPD translation tensor
    L : (3,3) SPD libration tensor
    S : (3,3) screw correlation (tr(S) = 0)
    origin : (3,) TLS origin (center of mass)
    """

    def __init__(
        self,
        T: np.ndarray,
        L: np.ndarray,
        S: np.ndarray,
        origin: np.ndarray,
    ) -> None:
        self.T = T
        self.L = L
        self.S = S
        self.origin = origin

    @classmethod
    def from_adp_tensors(
        cls,
        coords: np.ndarray,      # (N, 3)
        adp_tensors: np.ndarray, # (N, 3, 3) or (N, 6) upper-triangular
    ) -> "TLSGroup":
        """Fit TLS params from observed ADPs using least-squares."""
        # Convert (N,6) → (N,3,3) if needed
        if adp_tensors.ndim == 2 and adp_tensors.shape[1] == 6:
            adp_full = np.zeros((len(coords), 3, 3))
            idx = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
            for k, (a,b) in enumerate(idx):
                adp_full[:, a, b] = adp_tensors[:, k]
                adp_full[:, b, a] = adp_tensors[:, k]
            adp_tensors = adp_full

        origin = coords.mean(axis=0)
        r_local = coords - origin
        T, L, S = _fit_tls_least_squares(r_local, adp_tensors)
        return cls(T, L, S, origin)

    def predict_adp(self, atom_coords: np.ndarray) -> np.ndarray:
        """Compute U_i for atom at atom_coords.

        Parameters
        ----------
        atom_coords : (3,)

        Returns
        -------
        U_i : (3,3) predicted ADP tensor
        """
        r = atom_coords - self.origin
        return _tls_predict_adp(self.T, self.L, self.S, r)

    def residual_adp(
        self, atom_coords: np.ndarray, observed_adp: np.ndarray
    ) -> np.ndarray:
        """δU_i = U_i^obs - U_i^TLS."""
        return observed_adp - self.predict_adp(atom_coords)

    def to_gyrovector(self) -> Tuple[Tensor, Tensor]:
        """Encode T, L as points on SPD(3) for Gyro-QJL quantization.

        Returns Log-Euclidean tangent vectors at identity.
        """
        T_t = torch.tensor(self.T, dtype=torch.float32)
        L_t = torch.tensor(self.L, dtype=torch.float32)
        # Log-Euclidean: log_I(M) = matrix log (for SPD, well-defined)
        log_T = torch.linalg.matrix_log(T_t.clamp_min(1e-7 * torch.eye(3)))
        log_L = torch.linalg.matrix_log(L_t.clamp_min(1e-7 * torch.eye(3)))
        return log_T, log_L

    def hirshfeld_check(
        self,
        coords_i: np.ndarray,
        coords_j: np.ndarray,
        adp_i: np.ndarray,
        adp_j: np.ndarray,
        tol: float = 0.02,
    ) -> Tuple[float, bool]:
        """Hirshfeld rigid-bond test: (U_i - U_j)·n̂·n̂^T < tol Å².

        Returns (value, passes).
        """
        n = coords_j - coords_i
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-7:
            return 0.0, True
        n_hat = n / n_norm
        delta_U = adp_i - adp_j
        value = float(n_hat @ delta_U @ n_hat)
        return value, abs(value) < tol

    @property
    def isotropic_b(self) -> float:
        """Isotropic B-factor = (8π²/3) tr(T)."""
        return (8 * np.pi ** 2 / 3) * np.trace(self.T)

    @property
    def n_params(self) -> int:
        return 20


# ── TLSCompressor ─────────────────────────────────────────────────────────────

class TLSCompressor:
    """Compresses the full multi-rank feature pyramid using TLS decomposition.

    Parameters
    ----------
    tls_groups : list of TLSGroup (one per 3-cell/subunit)
    """

    def __init__(self, tls_groups: List[TLSGroup]) -> None:
        self.tls_groups = tls_groups

    def bond_is_intra_tls_group(
        self,
        i: int,
        j: int,
        atom_to_group: Dict[int, int],
    ) -> bool:
        """Return True if atoms i, j belong to the same TLS group."""
        gi = atom_to_group.get(i, -1)
        gj = atom_to_group.get(j, -1)
        return gi == gj and gi >= 0

    def compress_atom_features(
        self,
        h_atoms: Tensor,               # (N_atoms, d)
        atom_coords: np.ndarray,       # (N_atoms, 3)
        atom_to_group: Dict[int, int], # atom index → TLS group index
    ) -> Tuple[List[np.ndarray], Tensor]:
        """Compress atomic features into TLS params + residual field.

        Parameters
        ----------
        h_atoms      : (N_atoms, d) neural network atom features
        atom_coords  : (N_atoms, 3) Cartesian coordinates
        atom_to_group: mapping from atom index to TLS group index

        Returns
        -------
        tls_params     : list of (T,L,S) for each TLS group — 20 floats per subunit
        residual_field : (N_atoms, d) deviation from TLS prediction (electronic residual)
        """
        n_atoms = h_atoms.size(0)
        tls_params = []

        for grp_idx, tls in enumerate(self.tls_groups):
            tls_params.append((tls.T, tls.L, tls.S))

        # Residual field: h_atoms unchanged (TLS captures geometry, not electronics)
        # The residual is the full h_atoms — TLS provides the geometric prior,
        # the neural network captures the electronic deviation.
        residual_field = h_atoms

        return tls_params, residual_field

    def compress_bond_features(
        self,
        h_bonds: Tensor,               # (N_bonds, d)
        edge_index: Tensor,            # (2, N_bonds)
        atom_to_group: Dict[int, int],
    ) -> Tensor:
        """Compress bond features using pushforward deviation field.

        For intra-subunit bonds: deviation from (dB)_P pushforward.
        For inter-subunit bonds: full feature stored (interface bonds).

        Returns compressed bond features (same shape as h_bonds for now).
        """
        return h_bonds

    def compress_interface_features(
        self,
        h_interfaces: Tensor,           # (N_interfaces, d)
        subunit_pairs: List[Tuple[int, int]],
        tls_projectors: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Collapse interface features to principal angles.

        Interface feature → tr(P_A · P_B) (Grassmannian inner product).
        Storage: 1 scalar per interface pair, not 256-dim vector.

        Returns
        -------
        angles : (N_interfaces,)  one scalar per interface
        """
        if tls_projectors is None or len(tls_projectors) == 0:
            # Fall back to L2 norm as proxy for principal angle
            return h_interfaces.norm(dim=-1)

        angles = []
        for pair in subunit_pairs:
            i, j = pair
            if i < len(tls_projectors) and j < len(tls_projectors):
                P_A = tls_projectors[i]
                P_B = tls_projectors[j]
                # tr(P_A · P_B) = Grassmannian inner product
                angle = torch.trace(P_A @ P_B)
                angles.append(angle)
            else:
                angles.append(torch.tensor(0.0))

        return torch.stack(angles) if angles else torch.zeros(len(subunit_pairs))


# ── TLSQuantizer ─────────────────────────────────────────────────────────────

class TLSQuantizer:
    """Applies Gyro-QJL to TLS parameters stored on SPD manifolds.

    T and L are SPD matrices → quantize in Log-Euclidean tangent space T_I SPD(3).
    S is a general matrix → quantize in flat R^8.
    Total: 20 params × 3.5 bits = 8.75 bytes per subunit after quantization.
    """

    def __init__(self, d_spd: int = 9, d_s: int = 8) -> None:
        from tope.compression.gyro_qjl import TangentSpaceQJL
        # 9 = flattened 3×3 log of SPD matrix (symmetric → 6 independent)
        self.qjl_T = TangentSpaceQJL(d=6, b=2.5, k=64)
        self.qjl_L = TangentSpaceQJL(d=6, b=2.5, k=64)
        self.qjl_S = TangentSpaceQJL(d=8, b=2.5, k=32)

    def quantize(self, tls_group: TLSGroup) -> Dict[str, object]:
        """Compress a TLSGroup to ~8.75 bytes.

        Returns dict with compressed T, L, S representations.
        """
        log_T, log_L = tls_group.to_gyrovector()

        # Extract upper-triangular elements (6 for symmetric 3×3)
        def upper_tri_vec(M: Tensor) -> Tensor:
            idx = torch.triu_indices(3, 3)
            return M[idx[0], idx[1]]

        v_T = upper_tri_vec(log_T)
        v_L = upper_tri_vec(log_L)
        v_S = torch.tensor(tls_group.S.flatten()[:8], dtype=torch.float32)

        T_hat, s_T, ang_T = self.qjl_T.encode(v_T.unsqueeze(0))
        L_hat, s_L, ang_L = self.qjl_L.encode(v_L.unsqueeze(0))
        S_hat, s_S, ang_S = self.qjl_S.encode(v_S.unsqueeze(0))

        return {
            "T_hat": T_hat, "s_T": s_T, "ang_T": ang_T,
            "L_hat": L_hat, "s_L": s_L, "ang_L": ang_L,
            "S_hat": S_hat, "s_S": s_S, "ang_S": ang_S,
            "bytes": 8.75,
        }
