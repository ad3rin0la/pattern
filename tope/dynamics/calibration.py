"""Bridge-agnostic γ calibration for the ANM Hessian.

Fits a single global scalar ``c`` (units: (kcal/mol)/Å²) such that

::

    B_pred,i(c) = (8π²/3) · (k_B·T / c) · Σ_m |v_m,i|² / λ̃_m

matches experimental B-factors, with ``λ̃_m`` the eigenvalues of the
``c=1`` Hessian (scale-free spectrum). Only the intercept of the linear
relation between log(B_exp) and log(Σ_m |v|²/λ̃_m) is fit — the slope is
forced to 1 by construction.

This module is **bridge-agnostic**: the caller supplies a
``stiffness_fn(coords) -> {(i, j): float | np.ndarray}`` that produces
per-contact stiffness (scalar for isotropic-ANM, 3×3 SPD for tensor-
ANM). The calibration doesn't care which bridge from SheafENM was
picked — it just calibrates ``c`` against whatever stiffness model the
caller supplies. Baseline-ANM (uniform stiffness) is included as the
comparator that decides whether a fancier bridge is adding signal.

Key acceptance metrics produced by ``fit_gamma_calibration``:

* ``c``               — fitted global scalar
* ``per_protein_c``   — distribution of per-protein optimal ``c``'s
* ``cv``              — std/mean of per-protein ``c`` (the spec's pass
                        condition: < 20% to call this calibration good)
* ``pearson_r``       — log-space Pearson correlation of predicted vs
                        experimental B-factors
* ``baseline_pearson_r`` — same metric under uniform-γ ANM, the floor
                          a useful stiffness bridge must clear
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tope.data.bfactor import BFactorRecord
from tope.dynamics.nma import build_anm_hessian

logger = logging.getLogger(__name__)

# Boltzmann constant in kcal/(mol·K) — consistent with the rest of the
# tope.engineering / tope.dynamics modules.
KB_KCAL = 1.987204e-3

# B-factor prefactor: B = (8π²/3) · ⟨Δr²⟩ in Å² when ⟨Δr²⟩ is in Å².
B_PREFACTOR = (8.0 * math.pi ** 2) / 3.0


# Type alias for the bridge callable supplied by the caller.
# Inputs: (coords, contact_pairs). Output: dict mapping (i, j) to a
# scalar (isotropic spring) or a (3, 3) SPD matrix (tensor-ANM).
StiffnessFn = Callable[
    [np.ndarray, np.ndarray],
    Dict[Tuple[int, int], Union[float, np.ndarray]],
]


# ── Default bridges (baselines / placeholders) ───────────────────────────────

def uniform_stiffness(
    coords: np.ndarray,
    pairs: np.ndarray,
    gamma: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """Classical isotropic ANM: γ_ij = γ for every contact. The
    comparator any sheaf-derived bridge must beat."""
    return {(int(i), int(j)): float(gamma) for i, j in pairs}


def make_uniform_stiffness_fn(gamma: float = 1.0) -> StiffnessFn:
    def _fn(coords: np.ndarray, pairs: np.ndarray):
        return uniform_stiffness(coords, pairs, gamma=gamma)
    return _fn


# ── Per-protein scale-free spectrum ──────────────────────────────────────────

@dataclass
class ProteinSpectrum:
    """Cached per-protein NMA output used in the calibration loop.

    Computed once per (protein, stiffness_fn) pair — the scale-free
    spectrum doesn't depend on ``c``.
    """

    pdb_id: str
    eigenvalues: np.ndarray        # (3N − 6,) non-rigid eigenvalues of H/c
    eigenvectors: np.ndarray       # (3N, 3N − 6) corresponding eigenvectors
    log_S: np.ndarray              # (N,) log Σ_m |v_m,i|² / λ̃_m per residue


def _contact_pairs(coords: np.ndarray, cutoff: float) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    pairs = np.argwhere(d2 <= cutoff ** 2)
    return pairs[pairs[:, 0] < pairs[:, 1]]


def _hessian_from_stiffness(
    coords: np.ndarray,
    pairs: np.ndarray,
    stiffness: Dict[Tuple[int, int], Union[float, np.ndarray]],
) -> np.ndarray:
    """Build the 3N×3N Hessian from a per-contact stiffness dict.

    Dispatches on whether the dict values are scalars (passes through
    the scalar-spring path of build_anm_hessian via varying γ) or 3×3
    matrices (tensor-ANM path).
    """
    n = coords.shape[0]
    if not pairs.size:
        return np.zeros((3 * n, 3 * n), dtype=np.float64)

    sample = next(iter(stiffness.values()))
    if np.ndim(sample) == 0:
        H = np.zeros((3 * n, 3 * n), dtype=np.float64)
        for i, j in pairs:
            g = float(stiffness.get((int(i), int(j)), 0.0))
            if g == 0.0:
                continue
            r_ij = coords[j] - coords[i]
            d2 = float(r_ij @ r_ij)
            if d2 < 1e-12:
                continue
            block = (g / d2) * np.outer(r_ij, r_ij)
            ii, jj = 3 * i, 3 * j
            H[ii:ii + 3, jj:jj + 3] -= block
            H[jj:jj + 3, ii:ii + 3] -= block
            H[ii:ii + 3, ii:ii + 3] += block
            H[jj:jj + 3, jj:jj + 3] += block
        return H

    def _tensor(i: int, j: int) -> np.ndarray:
        return np.asarray(stiffness[(int(i), int(j))], dtype=np.float64)

    return build_anm_hessian(
        coords, cutoff=np.inf, contacts=pairs, contact_tensors=_tensor,
    )


def compute_spectrum(
    record: BFactorRecord,
    stiffness_fn: StiffnessFn,
    cutoff_A: float = 12.0,
    n_trivial_modes: int = 6,
    eigenvalue_floor: float = 1e-8,
) -> ProteinSpectrum:
    """Diagonalise the scale-free Hessian and compute per-residue log S_i."""
    pairs = _contact_pairs(record.coords, cutoff_A)
    stiffness = stiffness_fn(record.coords, pairs)
    H = _hessian_from_stiffness(record.coords, pairs, stiffness)

    H = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals[n_trivial_modes:], eigenvalue_floor)
    eigvecs = eigvecs[:, n_trivial_modes:]

    n = record.n_residues
    v_per_res = eigvecs.reshape(n, 3, -1)
    v_norm2 = (v_per_res ** 2).sum(axis=1)  # (N, n_modes)
    S = (v_norm2 / eigvals[None, :]).sum(axis=1)  # (N,)
    log_S = np.log(np.maximum(S, eigenvalue_floor))

    return ProteinSpectrum(
        pdb_id=record.pdb_id,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        log_S=log_S,
    )


# ── The calibration fit ──────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Output of ``fit_gamma_calibration``."""

    c: float                                  # global fitted scalar
    log_c: float
    per_protein_c: Dict[str, float] = field(default_factory=dict)
    cv: float = 0.0                           # std/mean of per_protein_c
    pearson_r_log: float = 0.0                # log-space correlation
    mse_log: float = 0.0
    n_records: int = 0
    n_residues_total: int = 0
    T_K: float = 298.15

    # Optional baseline-ANM comparator (uniform γ=1) on the same data.
    baseline_c: Optional[float] = None
    baseline_pearson_r_log: Optional[float] = None
    baseline_mse_log: Optional[float] = None

    def passes_spec(
        self,
        max_cv: float = 0.20,
        min_pearson_r: float = 0.30,
    ) -> Tuple[bool, str]:
        """Phase 6E.4 acceptance check."""
        if self.cv > max_cv:
            return False, f"CV {self.cv:.3f} > {max_cv} (global-scalar assumption failing)"
        if self.pearson_r_log < min_pearson_r:
            return False, f"r={self.pearson_r_log:.3f} < {min_pearson_r} (Γᵢⱼ shape wrong)"
        if (self.baseline_pearson_r_log is not None
                and self.pearson_r_log < self.baseline_pearson_r_log - 1e-3):
            return False, (
                f"r={self.pearson_r_log:.3f} below baseline uniform-ANM "
                f"r={self.baseline_pearson_r_log:.3f} — bridge adds noise"
            )
        return True, "ok"


def _fit_intercept_log_c(
    log_S_concat: np.ndarray,
    log_B_concat: np.ndarray,
    T_K: float,
) -> float:
    """Closed-form intercept-only fit.

    log(B_exp) = log(8π² k_B T / 3) − log(c) + log(S)
    ⇒ log(c) = mean(log(8π² k_B T / 3) + log(S) − log(B_exp))
    """
    const_term = math.log(B_PREFACTOR * KB_KCAL * T_K)
    log_c = float(np.mean(const_term + log_S_concat - log_B_concat))
    return log_c


def fit_gamma_calibration(
    records: Sequence[BFactorRecord],
    stiffness_fn: StiffnessFn,
    T_K: float = 298.15,
    cutoff_A: float = 12.0,
    eigenvalue_floor: float = 1e-8,
    compute_baseline: bool = True,
) -> CalibrationResult:
    """Fit the global scalar ``c`` against B-factor data.

    Parameters
    ----------
    records : list of BFactorRecord (the calibration split, typically)
    stiffness_fn : bridge callable mapping coords → per-contact stiffness
    T_K : crystal refinement temperature (default 298.15 — override per-
        protein at the caller if your dataset records mixed temperatures)
    cutoff_A : contact cutoff for the ANM graph
    compute_baseline : if True, also fit uniform-γ ANM and stash the
        comparator metrics on the result

    Returns
    -------
    CalibrationResult
    """
    if not records:
        return CalibrationResult(c=1.0, log_c=0.0, T_K=T_K)

    per_protein_log_c: List[float] = []
    per_protein_ids: List[str] = []
    log_S_all: List[np.ndarray] = []
    log_B_all: List[np.ndarray] = []

    for rec in records:
        spec = compute_spectrum(
            rec, stiffness_fn,
            cutoff_A=cutoff_A,
            eigenvalue_floor=eigenvalue_floor,
        )
        log_B = rec.log_b()
        log_S_all.append(spec.log_S)
        log_B_all.append(log_B)
        per_protein_log_c.append(_fit_intercept_log_c(spec.log_S, log_B, T_K))
        per_protein_ids.append(rec.pdb_id)

    log_S_concat = np.concatenate(log_S_all)
    log_B_concat = np.concatenate(log_B_all)
    global_log_c = _fit_intercept_log_c(log_S_concat, log_B_concat, T_K)

    per_protein_c = {
        pid: math.exp(lc) for pid, lc in zip(per_protein_ids, per_protein_log_c)
    }
    c_arr = np.array(list(per_protein_c.values()))
    cv = float(c_arr.std() / c_arr.mean()) if c_arr.mean() != 0 else float("inf")

    # Log-space Pearson r between predicted and experimental B-factors
    # using the global c. Predicted log B = log(prefactor·T) − log(c) + log S.
    pred_log_b = math.log(B_PREFACTOR * KB_KCAL * T_K) - global_log_c + log_S_concat
    pearson_r = _pearson(pred_log_b, log_B_concat)
    mse_log = float(np.mean((pred_log_b - log_B_concat) ** 2))

    result = CalibrationResult(
        c=math.exp(global_log_c),
        log_c=global_log_c,
        per_protein_c=per_protein_c,
        cv=cv,
        pearson_r_log=pearson_r,
        mse_log=mse_log,
        n_records=len(records),
        n_residues_total=int(log_B_concat.shape[0]),
        T_K=T_K,
    )

    if compute_baseline:
        baseline = fit_gamma_calibration(
            records,
            make_uniform_stiffness_fn(gamma=1.0),
            T_K=T_K,
            cutoff_A=cutoff_A,
            eigenvalue_floor=eigenvalue_floor,
            compute_baseline=False,
        )
        result.baseline_c = baseline.c
        result.baseline_pearson_r_log = baseline.pearson_r_log
        result.baseline_mse_log = baseline.mse_log

    return result


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = math.sqrt(float((a_c ** 2).sum()) * float((b_c ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((a_c * b_c).sum() / denom)


# ── Held-out drift and per-fold CV ───────────────────────────────────────────

@dataclass
class HeldOutResult:
    cal: CalibrationResult
    val: CalibrationResult
    drift_fraction: float   # |c_cal − c_val| / c_cal


def held_out_drift(
    calibration_records: Sequence[BFactorRecord],
    validation_records: Sequence[BFactorRecord],
    stiffness_fn: StiffnessFn,
    T_K: float = 298.15,
    cutoff_A: float = 12.0,
) -> HeldOutResult:
    """Fit ``c`` on the calibration split and again on the held-out
    validation split. The drift fraction is the spec's cross-validation
    stability check."""
    cal = fit_gamma_calibration(calibration_records, stiffness_fn, T_K=T_K,
                                cutoff_A=cutoff_A, compute_baseline=False)
    val = fit_gamma_calibration(validation_records, stiffness_fn, T_K=T_K,
                                cutoff_A=cutoff_A, compute_baseline=False)
    drift = abs(cal.c - val.c) / max(cal.c, 1e-30)
    return HeldOutResult(cal=cal, val=val, drift_fraction=float(drift))


# ── Numerical probe: distribution of stiffness values ────────────────────────

@dataclass
class StiffnessProbe:
    """Quick statistics from running ``stiffness_fn`` over the dataset.

    Phase 6E.7.1 asks whether per-contact stiffness varies smoothly or
    spans orders of magnitude. The latter would mean a global ``c`` fit
    is overdetermined; this probe answers the question before any fit.
    """

    n_contacts: int
    min_value: float
    max_value: float
    median_value: float
    log10_range: float          # log10(max / min); >3 ⇒ global-scalar fit fragile
    eigenvalue_log10_range: Optional[float] = None  # only for tensor-ANM


def probe_stiffness_distribution(
    records: Sequence[BFactorRecord],
    stiffness_fn: StiffnessFn,
    cutoff_A: float = 12.0,
) -> StiffnessProbe:
    """Aggregate stiffness statistics across the dataset.

    For scalar-stiffness bridges, returns the spread of γᵢⱼ. For tensor-
    stiffness bridges, additionally reports the spread of the largest
    eigenvalue across all contact tensors (a proxy for the spec's
    "do the d=8 eigenvalues span orders of magnitude" question).
    """
    scalars: List[float] = []
    eigvals: List[float] = []
    for rec in records:
        pairs = _contact_pairs(rec.coords, cutoff_A)
        s = stiffness_fn(rec.coords, pairs)
        for v in s.values():
            if np.ndim(v) == 0:
                scalars.append(float(v))
            else:
                e = np.linalg.eigvalsh(np.asarray(v))
                eigvals.append(float(e.max()))

    values = np.asarray(scalars or eigvals, dtype=np.float64)
    if values.size == 0:
        return StiffnessProbe(0, 0.0, 0.0, 0.0, 0.0)

    pos = values[values > 0]
    if pos.size == 0:
        return StiffnessProbe(values.size, float(values.min()),
                              float(values.max()), float(np.median(values)), 0.0)
    log10_range = math.log10(float(pos.max()) / max(float(pos.min()), 1e-30))

    eig_range = None
    if eigvals:
        e_arr = np.asarray(eigvals, dtype=np.float64)
        e_pos = e_arr[e_arr > 0]
        if e_pos.size:
            eig_range = math.log10(float(e_pos.max()) / max(float(e_pos.min()), 1e-30))

    return StiffnessProbe(
        n_contacts=values.size,
        min_value=float(values.min()),
        max_value=float(values.max()),
        median_value=float(np.median(values)),
        log10_range=log10_range,
        eigenvalue_log10_range=eig_range,
    )
