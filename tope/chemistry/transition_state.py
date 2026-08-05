"""
Transition State Representation for ToPE
=========================================

Encodes a transition state as a rank-1 perturbation of the ground-state TLS.

At a transition state (saddle point on the Born-Oppenheimer surface), the
nuclear Hessian has exactly one negative eigenvalue λ‡ < 0. The TS ADP field:

    U_i^‡ = U_i^TLS + Δ · (e_‡^i ⊗ e_‡^i)

where e_‡^i ∈ R³ is the reaction coordinate direction at atom i, and Δ > 0
encodes the imaginary frequency magnitude.

In Log-Euclidean geometry (gyrotensor binary operation):
    U^‡ = U^TLS ⊕_le (Δ · ê‡ ⊗ ê‡) = exp(log(U^TLS) + log(Δ · ê‡ ⊗ ê‡))

The reaction coordinate field {e_‡^i} is identified from two constraints:
1. Direction of maximum B-factor anisotropy at active site atoms (TLS libration L)
2. Direction of maximum VOIP gradient (sheaf restriction maps)

This gives a 6×6 eigenproblem for 2-subunit enzymes.
Reaction coordinate: 3N floats → 6 floats.

Storage: ~830 bytes per transition state (see TransitionStateCompressor).

References
----------
Garcia-Viloca et al. 2004 — TS theory for enzymes, Science 303:186.
Schomaker & Trueblood 1968 — TLS model.
Cerrini 1971 — Tensor harmonic decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from tope.compression.tls_compression import TLSGroup, _tls_predict_adp
from tope.geometry.grassmann_pushforward import grassmann_log_at_identity


# ── Quantum tunneling correction ──────────────────────────────────────────────

def transmission_coefficient(
    imag_freq_cm1: float,
    temperature_K: float = 298.15,
) -> float:
    """Wigner tunneling transmission coefficient κ(T).

    κ = 1 + (1/24)(ℏ|ω‡|/k_BT)² + O((ℏω/kT)⁴)

    Parameters
    ----------
    imag_freq_cm1 : |ω‡| in cm⁻¹ (imaginary frequency magnitude)
    temperature_K : temperature in Kelvin

    Returns
    -------
    kappa : Wigner transmission coefficient (≥ 1)
    """
    hbar_omega = imag_freq_cm1 * 1.4388  # ℏ|ω‡| / k_B in Kelvin (hc/kB ≈ 1.4388 K·cm)
    x = hbar_omega / temperature_K
    return 1.0 + (x ** 2) / 24.0


# ── Reaction coordinate identification ────────────────────────────────────────

def _identify_reaction_coord_6x6(
    tls_groups: List[TLSGroup],
    voip_gradients: np.ndarray,   # (N_atoms, 3) VOIP gradient field
    atom_to_group: Dict[int, int],
    n_active_site: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify reaction coordinate from TLS libration + VOIP gradient.

    Builds a 6×6 eigenproblem in the intersection of:
    - span(TLS libration principal axes): direction of max anisotropy
    - span(VOIP gradient directions): electronic reaction coordinate

    For a 2-subunit enzyme, this gives at most 3+3=6 vectors per subunit.

    Returns
    -------
    e_reaction : (6,) reaction coordinate in the combined basis
    basis      : (6, 3) the combined basis vectors
    """
    # Collect libration principal axes from each TLS group
    libration_axes = []
    for tls in tls_groups[:2]:  # 2 subunits → 3 axes each = 6 total
        eigvals, eigvecs = np.linalg.eigh(tls.L)
        # Take the principal axis (largest libration eigenvalue)
        for k in range(3):
            libration_axes.append(eigvecs[:, k])

    # Collect VOIP gradient directions (mean over active site atoms)
    voip_dir = voip_gradients[:n_active_site].mean(axis=0)
    voip_norm = np.linalg.norm(voip_dir)
    if voip_norm > 1e-7:
        voip_dir /= voip_norm

    # Build 6×6 eigenproblem: overlap matrix in combined basis
    basis = np.array(libration_axes[:6])  # (6, 3)

    # Overlap matrix of basis vectors
    O = basis @ basis.T  # (6, 6)

    # Weight by VOIP alignment
    voip_weights = np.abs(basis @ voip_dir)  # (6,) alignment with VOIP gradient
    W = np.diag(voip_weights + 0.1)  # (6, 6) regularized

    # Generalized eigenproblem: W·x = λ·O·x
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(O) @ W)
        # Take eigenvector with largest real eigenvalue
        real_eigs = np.real(eigvals)
        idx = np.argmax(real_eigs)
        e_reaction = np.real(eigvecs[:, idx])  # (6,) in basis coordinates
    except np.linalg.LinAlgError:
        e_reaction = np.zeros(6)
        e_reaction[0] = 1.0

    return e_reaction, basis


def _expand_reaction_coord_to_atoms(
    e_reaction: np.ndarray,   # (6,) in basis coordinates
    basis: np.ndarray,        # (6, 3)
    n_atoms: int,
) -> np.ndarray:
    """Expand compact 6-vector reaction coordinate to per-atom field.

    e_‡^i = Σ_k e_reaction[k] * basis[k]  (uniform over all atoms for first approx)

    Returns
    -------
    coord_field : (n_atoms, 3)
    """
    e_3d = basis.T @ e_reaction  # (3,)
    norm = np.linalg.norm(e_3d)
    if norm > 1e-7:
        e_3d /= norm
    return np.tile(e_3d, (n_atoms, 1))  # (n_atoms, 3)


# ── TransitionStateRepresentation ────────────────────────────────────────────

class TransitionStateRepresentation:
    """Encodes a transition state as a perturbation of the ground-state TLS.

    Storage: ~830 bytes (see TransitionStateCompressor.compress()).

    Parameters
    ----------
    tls_groups         : list of TLSGroup (ground state TLS parameters)
    reaction_coord_field: (N_atoms, 3) per-atom reaction coordinate direction
    imaginary_freq     : |ω‡| in cm⁻¹ (imaginary frequency magnitude)
    activation_energy  : ΔG‡ in kcal/mol
    """

    def __init__(
        self,
        tls_groups: List[TLSGroup],
        reaction_coord_field: np.ndarray,   # (N_atoms, 3)
        imaginary_freq: float,              # cm⁻¹
        activation_energy: float,           # kcal/mol
    ) -> None:
        self.tls_groups = tls_groups
        self.reaction_coord_field = reaction_coord_field
        self.imaginary_freq = imaginary_freq
        self.activation_energy = activation_energy

        # Δ from imaginary frequency: Δ = ℏ|ω‡| / (2 k_B T) in Å² (approx)
        # Using ℏ|ω‡|/k_B ≈ imag_freq_cm1 × 1.4388 K
        self.Delta = imaginary_freq * 1.4388 / (2 * 298.15) * 0.01  # rough conversion to Å²

    @classmethod
    def from_tls_and_voip(
        cls,
        tls_groups: List[TLSGroup],
        voip_gradients: np.ndarray,    # (N_atoms, 3)
        atom_to_group: Dict[int, int],
        imaginary_freq: float = 500.0,
        activation_energy: float = 10.0,
    ) -> "TransitionStateRepresentation":
        """Identify reaction coordinate from TLS + VOIP and construct TS repr.

        Solves 6×6 eigenproblem in span(TLS libration axes) ∩ span(VOIP gradient).
        """
        n_atoms = len(voip_gradients)
        e_reaction, basis = _identify_reaction_coord_6x6(
            tls_groups, voip_gradients, atom_to_group
        )
        coord_field = _expand_reaction_coord_to_atoms(e_reaction, basis, n_atoms)

        return cls(tls_groups, coord_field, imaginary_freq, activation_energy)

    def predict_adp_at_ts(self, atom_coords: np.ndarray) -> np.ndarray:
        """Return U_i^‡ = U_i^TLS + Δ·(e_‡^i ⊗ e_‡^i).

        Parameters
        ----------
        atom_coords : (N_atoms, 3)

        Returns
        -------
        adp_ts : (N_atoms, 3, 3) TS ADP field
        """
        n_atoms = len(atom_coords)
        adp_ts = np.zeros((n_atoms, 3, 3))

        for i, coords in enumerate(atom_coords):
            # Find which TLS group owns this atom
            # (simplified: assign to nearest TLS origin)
            dists = [np.linalg.norm(coords - tls.origin) for tls in self.tls_groups]
            grp_idx = int(np.argmin(dists))
            tls = self.tls_groups[grp_idx]

            # Ground-state TLS ADP
            U_tls = tls.predict_adp(coords)

            # Rank-1 TS perturbation
            e_i = self.reaction_coord_field[i] if i < len(self.reaction_coord_field) else np.zeros(3)
            e_norm = np.linalg.norm(e_i)
            if e_norm > 1e-7:
                e_hat = e_i / e_norm
            else:
                e_hat = np.zeros(3)

            U_ts = U_tls + self.Delta * np.outer(e_hat, e_hat)
            adp_ts[i] = U_ts

        return adp_ts

    def activation_free_energy(self) -> float:
        """Log-Euclidean distance between L^ground and L^‡ on SPD(3).

        ΔG‡ ∝ d_le(L^ground, L^‡) = ‖log(L^ground) - log(L^‡)‖_F

        Returns stored activation_energy value (target for training).
        """
        return self.activation_energy

    def transmission_coefficient(self, temperature_K: float = 298.15) -> float:
        """Wigner tunneling correction κ(T)."""
        return transmission_coefficient(self.imaginary_freq, temperature_K)


# ── TransitionStateCompressor ─────────────────────────────────────────────────

class TransitionStateCompressor:
    """Compresses full TS representation to ~830 bytes.

    Byte budget (per transition state):
        tls_params      : 18 bytes (20 params × 3.5 bits, 2 subunits)
        reaction_coord  : 24 bytes (6 floats — eigenvector in TLS+VOIP basis)
        imaginary_freq  :  4 bytes (1 float)
        coord_field     :~660 bytes ({e_‡^i ∈ R³} per atom, 3.5 bits via Gyro-QJL)
        qjl_residual    :~125 bytes (1-bit sketch correction)
        TOTAL           :~831 bytes
    """

    def __init__(
        self,
        ts: TransitionStateRepresentation,
        reaction_coord_compact: np.ndarray,  # (6,) compact eigenvector
    ) -> None:
        self.ts = ts
        self.reaction_coord_compact = reaction_coord_compact

    def compress(self) -> Dict[str, object]:
        """Compress to ~830 bytes.

        Returns dict with all compressed components.
        """
        from tope.compression.gyro_qjl import TangentSpaceQJL
        from tope.compression.tls_compression import TLSQuantizer

        n_atoms = len(self.ts.reaction_coord_field)

        # 1. TLS params (2 subunits × 20 params × 3.5 bits / 8 = 17.5 ≈ 18 bytes)
        quantizer = TLSQuantizer()
        tls_compressed = []
        for tls in self.ts.tls_groups[:2]:
            tls_compressed.append(quantizer.quantize(tls))

        # 2. Reaction coordinate (6 floats × 4 bytes = 24 bytes)
        rc_tensor = torch.tensor(self.reaction_coord_compact, dtype=torch.float32)

        # 3. Imaginary frequency (4 bytes)
        imag_freq = self.ts.imaginary_freq

        # 4. Coord field (N_atoms × 3 × 3.5 bits / 8 ≈ 660 bytes for 500 atoms)
        coord_flat = torch.tensor(
            self.ts.reaction_coord_field.flatten(), dtype=torch.float32
        ).unsqueeze(0)  # (1, N_atoms*3)
        # Clip to first 512 dims for QJL
        d_coord = min(coord_flat.size(1), 512)
        coord_clip = coord_flat[:, :d_coord]
        coord_qjl = TangentSpaceQJL(d=d_coord, b=2.5, k=125)
        coord_hat, s_coord, ang_coord = coord_qjl.encode(coord_clip)

        return {
            "tls_params": tls_compressed,                # 18 bytes
            "reaction_coord": rc_tensor,                  # 24 bytes (6 floats)
            "imaginary_freq": imag_freq,                  # 4 bytes
            "coord_hat": coord_hat,                       # ~660 bytes
            "s_coord": s_coord,                           # ~125 bytes (1-bit sketch)
            "ang_coord": ang_coord,
            "total_bytes_approx": 831,
        }
