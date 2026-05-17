"""Anisotropic-ENM normal-mode analysis.

Given residue Cα coordinates and a contact cutoff (or an explicit
contact list), build the 3N×3N anisotropic-network Hessian, diagonalise
it, and expose:

* per-mode propagation:   ``r(t) = r0 + A · v_k · cos(ω_k t)``
* harmonic ensemble:      coords ~ N(r0, Σ_k (k_B T / ω_k²) v_k v_kᵀ)
* thermal MSF:            ⟨Δr_i²⟩  ∝  B-factor proxy

This is a *static* operator analysis — the modes are computed once from
the X-ray geometry and propagated analytically. Use it as the cheap
fallback when you don't want to integrate Newton's equations.

Conventions
-----------
Units are deliberately abstract: force constants in (energy / length²),
masses in any consistent unit, temperatures in K (with ``k_B`` taken
from ``scipy.constants``). Callers can rescale to physical units as
needed — for relative B-factor profiles the units cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.constants import Boltzmann as _KB_SI

# k_B in kcal/(mol·K) — matches the rest of tope.engineering.
KB_KCAL = 1.987204e-3


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class NMAConfig:
    """Knobs for `NormalModeAnalysis`."""

    contact_cutoff: float = 12.0      # Å — typical ANM cutoff
    force_constant: float = 1.0       # uniform spring γ
    n_modes: int = 20                 # how many non-trivial modes to keep
    n_trivial_modes: int = 6          # 3 translations + 3 rotations
    mass_weighted: bool = True
    temperature_K: float = 298.15
    eigenvalue_floor: float = 1e-8    # numerical floor for "zero" modes


# ── Hessian construction ──────────────────────────────────────────────────────

def build_anm_hessian(
    coords: np.ndarray,
    cutoff: float = 12.0,
    force_constant: float = 1.0,
    contacts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the 3N×3N anisotropic-network Hessian.

    For each contact pair (i, j) within ``cutoff`` (or each pair listed
    in ``contacts``), the off-diagonal block is

        H_ij = −(γ / |r_ij|²) · (r_ij ⊗ r_ij)

    and the diagonal block H_ii is the negative sum of its row's
    off-diagonal blocks (so that rigid translations are zero modes).

    Parameters
    ----------
    coords : (N, 3) ndarray
    cutoff : Å, ignored if `contacts` is given
    force_constant : uniform γ
    contacts : (E, 2) ndarray of contact pairs, optional
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    H = np.zeros((3 * n, 3 * n), dtype=np.float64)

    if contacts is None:
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = (diff ** 2).sum(axis=-1)
        np.fill_diagonal(d2, np.inf)
        pairs = np.argwhere((d2 <= cutoff ** 2))
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    else:
        pairs = np.asarray(contacts, dtype=int)

    for i, j in pairs:
        r_ij = coords[j] - coords[i]
        dist2 = float(r_ij @ r_ij)
        if dist2 < 1e-12:
            continue
        block = (force_constant / dist2) * np.outer(r_ij, r_ij)
        ii, jj = 3 * i, 3 * j
        H[ii:ii + 3, jj:jj + 3] -= block
        H[jj:jj + 3, ii:ii + 3] -= block
        H[ii:ii + 3, ii:ii + 3] += block
        H[jj:jj + 3, jj:jj + 3] += block

    return H


# ── Core analysis class ───────────────────────────────────────────────────────

class NormalModeAnalysis:
    """Diagonalise an ANM Hessian and expose harmonic dynamics.

    Modes are mass-weighted when ``cfg.mass_weighted`` is True and a
    ``masses`` array is supplied — that gives physically meaningful
    frequencies (eigenvalues are ω²). Without mass weighting the
    spectrum is in arbitrary units but mode shapes are unaffected.
    """

    def __init__(
        self,
        coords: np.ndarray,
        cfg: Optional[NMAConfig] = None,
        masses: Optional[np.ndarray] = None,
        contacts: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg or NMAConfig()
        self.coords = np.asarray(coords, dtype=np.float64).copy()
        self.n = self.coords.shape[0]

        if masses is None:
            masses = np.ones(self.n, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (self.n,):
            raise ValueError(f"masses shape {masses.shape} != ({self.n},)")
        self.masses = masses

        self.H = build_anm_hessian(
            self.coords,
            cutoff=self.cfg.contact_cutoff,
            force_constant=self.cfg.force_constant,
            contacts=contacts,
        )

        if self.cfg.mass_weighted:
            m3 = np.repeat(self.masses, 3)
            inv_sqrt_m = 1.0 / np.sqrt(m3)
            self._operator = (inv_sqrt_m[:, None] * self.H) * inv_sqrt_m[None, :]
            self._mw_scale = inv_sqrt_m
        else:
            self._operator = self.H
            self._mw_scale = np.ones(3 * self.n)

        # Symmetrise to clean up numerical asymmetry, then diagonalise.
        op = 0.5 * (self._operator + self._operator.T)
        eigvals, eigvecs = np.linalg.eigh(op)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs  # columns are modes in (mass-weighted) space

    # ── derived quantities ──────────────────────────────────────────────

    @property
    def frequencies(self) -> np.ndarray:
        """Mode angular frequencies ω_k = sqrt(max(λ_k, 0))."""
        return np.sqrt(np.maximum(self.eigenvalues, 0.0))

    def nontrivial_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, mode_vectors) with rigid modes stripped.

        Mode vectors are in Cartesian-coordinate space (un-mass-weighted),
        shape (n_modes, 3N). Limited to ``cfg.n_modes`` modes after the
        first ``cfg.n_trivial_modes`` zero-frequency rigid motions.
        """
        start = self.cfg.n_trivial_modes
        end = start + self.cfg.n_modes
        sel = slice(start, end)
        omega = self.frequencies[sel]
        v = self.eigenvectors[:, sel]
        if self.cfg.mass_weighted:
            v = v * self._mw_scale[:, None]
        return omega, v.T  # (n_modes,), (n_modes, 3N)

    # ── dynamics ────────────────────────────────────────────────────────

    def propagate_mode(
        self,
        mode_idx: int,
        amplitude: float,
        n_steps: int = 50,
        period_fraction: float = 1.0,
    ) -> np.ndarray:
        """Analytic harmonic propagation of a single mode.

        Returns a (n_steps, N, 3) trajectory where the chosen mode
        oscillates as ``cos(ω t)`` and other coords are held fixed.
        Useful for visualising mode shapes — not Newtonian dynamics.
        """
        omega, modes = self.nontrivial_modes()
        if mode_idx >= modes.shape[0]:
            raise IndexError(f"mode_idx {mode_idx} >= {modes.shape[0]}")
        v = modes[mode_idx].reshape(self.n, 3)
        w = omega[mode_idx]
        # Avoid period blow-up for near-zero frequencies — use phase only.
        # endpoint=False so step n/2 lands exactly on phase π.
        phases = np.linspace(
            0.0, 2.0 * np.pi * period_fraction, n_steps, endpoint=False,
        )
        traj = np.empty((n_steps, self.n, 3))
        for t, phi in enumerate(phases):
            traj[t] = self.coords + amplitude * np.cos(phi) * v
        return traj

    def sample_harmonic_ensemble(
        self,
        n_samples: int = 100,
        temperature_K: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw ``n_samples`` coordinate sets from the harmonic ensemble.

        For each non-trivial mode k, sample amplitude
        ``a_k ~ N(0, k_B T / ω_k²)`` and reconstruct coords as
        ``r = r0 + Σ_k a_k v_k``. Returns shape (n_samples, N, 3).
        """
        return sample_harmonic_ensemble(self, n_samples, temperature_K, rng)

    def thermal_msf(
        self,
        temperature_K: Optional[float] = None,
    ) -> np.ndarray:
        """Mean-square fluctuation per residue: (N,) B-factor proxy.

        ``<Δr_i²> = k_B T · Σ_k (|v_ik|² / ω_k²)`` summed over non-trivial
        modes. Returned in (length²) units matching ``coords``.
        """
        return thermal_msf(self, temperature_K)


# ── Module-level helpers (functional access) ──────────────────────────────────

def propagate_mode(
    nma: NormalModeAnalysis,
    mode_idx: int,
    amplitude: float,
    n_steps: int = 50,
) -> np.ndarray:
    return nma.propagate_mode(mode_idx, amplitude, n_steps)


def sample_harmonic_ensemble(
    nma: NormalModeAnalysis,
    n_samples: int = 100,
    temperature_K: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    T = temperature_K if temperature_K is not None else nma.cfg.temperature_K
    rng = rng or np.random.default_rng()
    omega, modes = nma.nontrivial_modes()
    floor = nma.cfg.eigenvalue_floor
    omega2 = np.maximum(omega ** 2, floor)
    variances = KB_KCAL * T / omega2  # per-mode amplitude variance

    out = np.empty((n_samples, nma.n, 3))
    for s in range(n_samples):
        a = rng.normal(0.0, np.sqrt(variances))  # (n_modes,)
        delta = (a[:, None] * modes).sum(axis=0).reshape(nma.n, 3)
        out[s] = nma.coords + delta
    return out


def thermal_msf(
    nma: NormalModeAnalysis,
    temperature_K: Optional[float] = None,
) -> np.ndarray:
    T = temperature_K if temperature_K is not None else nma.cfg.temperature_K
    omega, modes = nma.nontrivial_modes()
    floor = nma.cfg.eigenvalue_floor
    omega2 = np.maximum(omega ** 2, floor)
    # |v_ik|² for residue i, mode k.
    v_per_res = modes.reshape(modes.shape[0], nma.n, 3)
    v_norm2 = (v_per_res ** 2).sum(axis=-1)  # (n_modes, N)
    msf = KB_KCAL * T * (v_norm2 / omega2[:, None]).sum(axis=0)
    return msf
