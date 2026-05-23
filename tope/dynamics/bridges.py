"""Three candidate SheafENM-to-ANM bridge functions.

Each bridge maps the 8×8 sheaf restriction maps produced by
``SheafENM.build_restriction_maps`` to per-contact stiffness in the
form expected by ``tope.dynamics.calibration.StiffnessFn``. They are
intentionally short and structurally similar so the data — not
aesthetics — decides which performs best on the calibration set.

Notation
--------
Each contact (i, j) has a sheaf-section difference ``δ = s_i − s_j``
in ℝ⁸ and a metric ``G ∈ SPD(8)``. The sheaf restriction is

    R_ij = (Gδ)(Gδ)ᵀ / (δᵀ G δ)        # rank-1 SPSD on ℝ⁸

The three bridges differ in how they collapse this 8×8 chemistry-space
object to a 3×3 mechanical stiffness on real space.

* **A0a — scalar isotropic.** ``γ_ij = c · Tr(R_ij)`` feeds the scalar-
  spring path. Hessian block becomes ``(γ_ij / d²)(r ⊗ r)``. Per-contact
  stiffness magnitude tracks chemistry, geometry comes from the bond.
  Same rank-1 along the bond as classical ANM ⇒ collinear-chain
  degeneracy returns.

* **A0b — scalar × bond geometry.** Same scalar ``Tr(R_ij)`` as A0a,
  but built into a 3×3 outer product along the bond direction:
  ``Γ_ij = Tr(R_ij) · (r̂ ⊗ r̂)``. Mathematically equivalent to A0a after
  the (1/d²) factor — included as an explicit comparator so the data
  shows whether the equivalence holds numerically.

* **A0c — diagonal sub-block.** ``Γ_ij = R_ij[indices, indices] + ε·I``,
  picking 3 of the 8 Ioffe descriptors to identify with spatial axes.
  Structurally different: stiffness is in descriptor space, *not* along
  the bond. Rank-1 + regulariser. Preserves transverse stiffness in
  collinear chains at the cost of physical interpretability.

Comparison harness
------------------
``compare_bridges(records, bridges)`` runs probe + calibration on each
and returns a single comparison table. ``select_best_bridge`` picks the
one that passes ``passes_spec`` with the best Pearson r over baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tope.data.bfactor import BFactorRecord
from tope.dynamics.calibration import (
    CalibrationResult,
    StiffnessFn,
    StiffnessProbe,
    fit_gamma_calibration,
    probe_stiffness_distribution,
)


# ── Bridge builders ───────────────────────────────────────────────────────────

def _trace_R(delta: np.ndarray, G: np.ndarray) -> float:
    """Tr(R_ij) for R_ij = (Gδ)(Gδ)ᵀ / (δᵀGδ); a positive scalar."""
    Gd = G @ delta
    denom = float(delta @ Gd) + 1e-12
    return float(Gd @ Gd) / denom


def trace_isotropic_bridge(
    sheaf_sections: Dict[int, np.ndarray],
    descriptor_metric: np.ndarray,
) -> StiffnessFn:
    """A0a: γ_ij = Tr(R_ij), scalar per contact."""
    G = np.asarray(descriptor_metric, dtype=np.float64)

    def fn(coords: np.ndarray, pairs: np.ndarray):
        out: Dict[Tuple[int, int], float] = {}
        for i, j in pairs:
            i, j = int(i), int(j)
            delta = sheaf_sections[i] - sheaf_sections[j]
            out[(i, j)] = _trace_R(delta, G)
        return out
    return fn


def trace_bond_directed_bridge(
    sheaf_sections: Dict[int, np.ndarray],
    descriptor_metric: np.ndarray,
) -> StiffnessFn:
    """A0b: Γ_ij = Tr(R_ij) · (r̂ ⊗ r̂), 3×3 rank-1 along bond."""
    G = np.asarray(descriptor_metric, dtype=np.float64)

    def fn(coords: np.ndarray, pairs: np.ndarray):
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for i, j in pairs:
            i, j = int(i), int(j)
            delta = sheaf_sections[i] - sheaf_sections[j]
            r = coords[j] - coords[i]
            r_hat = r / (np.linalg.norm(r) + 1e-12)
            out[(i, j)] = _trace_R(delta, G) * np.outer(r_hat, r_hat)
        return out
    return fn


def diagonal_spd_bridge(
    sheaf_sections: Dict[int, np.ndarray],
    descriptor_metric: np.ndarray,
    indices: Tuple[int, int, int] = (0, 1, 2),
    regularizer: float = 1e-3,
) -> StiffnessFn:
    """A0c: Γ_ij = R_ij[indices, indices] + ε·I, rank-3 by regularisation."""
    G = np.asarray(descriptor_metric, dtype=np.float64)
    idx = np.array(indices, dtype=int)
    eps_I = regularizer * np.eye(3)

    def fn(coords: np.ndarray, pairs: np.ndarray):
        out: Dict[Tuple[int, int], np.ndarray] = {}
        for i, j in pairs:
            i, j = int(i), int(j)
            delta = sheaf_sections[i] - sheaf_sections[j]
            Gd = G @ delta
            denom = float(delta @ Gd) + 1e-12
            R_sub = np.outer(Gd[idx], Gd[idx]) / denom
            out[(i, j)] = R_sub + eps_I
        return out
    return fn


# ── Comparison harness ───────────────────────────────────────────────────────

@dataclass
class BridgeReport:
    """Side-by-side metrics for one bridge over a record set."""

    name: str
    probe: StiffnessProbe
    result: CalibrationResult
    passes: bool
    reason: str


def compare_bridges(
    records: Sequence[BFactorRecord],
    bridges: Dict[str, StiffnessFn],
    T_K: float = 298.15,
    cutoff_A: float = 12.0,
) -> List[BridgeReport]:
    """Probe + calibrate each candidate bridge on the same record set."""
    reports: List[BridgeReport] = []
    for name, fn in bridges.items():
        probe = probe_stiffness_distribution(records, fn, cutoff_A=cutoff_A)
        result = fit_gamma_calibration(
            records, fn, T_K=T_K, cutoff_A=cutoff_A, compute_baseline=True,
        )
        ok, reason = result.passes_spec()
        reports.append(BridgeReport(name, probe, result, ok, reason))
    return reports


def select_best_bridge(
    reports: Sequence[BridgeReport],
) -> Optional[BridgeReport]:
    """Pick the bridge that passes_spec with the best log-space Pearson r
    relative to the baseline. Returns None if none pass."""
    passing = [r for r in reports if r.passes]
    if not passing:
        return None
    def _score(r: BridgeReport) -> float:
        # Bigger is better: how much does the bridge beat baseline uniform-ANM?
        if r.result.baseline_pearson_r_log is None:
            return r.result.pearson_r_log
        return r.result.pearson_r_log - r.result.baseline_pearson_r_log
    return max(passing, key=_score)


def format_comparison_table(reports: Sequence[BridgeReport]) -> str:
    """Plain-text summary table for printing in a script."""
    headers = (
        "bridge", "log10_range", "c", "cv",
        "r_log", "r_baseline", "passes", "reason",
    )
    rows = []
    for r in reports:
        rows.append((
            r.name,
            f"{r.probe.log10_range:.2f}",
            f"{r.result.c:.3e}",
            f"{r.result.cv:.3f}",
            f"{r.result.pearson_r_log:.3f}",
            (f"{r.result.baseline_pearson_r_log:.3f}"
             if r.result.baseline_pearson_r_log is not None else "n/a"),
            "yes" if r.passes else "no",
            r.reason,
        ))
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
              for i, h in enumerate(headers)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)
