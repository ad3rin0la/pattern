"""Discrete gauge connections for orbital frames and matrix-product states.

The functions here remove representation-only phase, permutation and virtual
bond changes before electronic residuals are formed.  They do not claim that a
smoothly aligned state is dynamically accurate; singular-value and holonomy
diagnostics are exposed so callers can escalate to higher-capacity or reference
calculations when local parallel transport becomes unreliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Optional, Sequence

import torch
from torch import Tensor


def _adjoint(matrix: Tensor) -> Tensor:
    return matrix.transpose(-1, -2).conj()


def _validate_square(matrix: Tensor, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")


def orbital_overlap(previous: Tensor, current: Tensor, metric: Optional[Tensor] = None) -> Tensor:
    """Return ``<previous|current>`` for column-oriented orbital frames.

    ``metric`` may contain the atomic-orbital overlap matrix for a nonorthogonal
    basis.  Both frames must use the same underlying basis ordering.
    """
    if previous.ndim != 2 or current.shape != previous.shape:
        raise ValueError("orbital frames must have matching (basis, orbitals) shapes")
    if not torch.isfinite(previous).all() or not torch.isfinite(current).all():
        raise ValueError("orbital frames must be finite")
    if metric is None:
        return _adjoint(previous) @ current
    if metric.shape != (previous.shape[0], previous.shape[0]):
        raise ValueError("metric must have shape (basis, basis)")
    if not torch.isfinite(metric).all():
        raise ValueError("metric must be finite")
    return _adjoint(previous) @ metric @ current


def procrustes_transport(overlap: Tensor) -> tuple[Tensor, Tensor]:
    """Return the right-acting unitary transport and overlap singular values.

    If ``overlap = U diag(s) V^H``, the returned ``W = V U^H`` maximizes
    ``Re Tr(overlap W)``.  A current frame is aligned as ``current @ W``.
    """
    _validate_square(overlap, "overlap")
    u, singular_values, vh = torch.linalg.svd(overlap, full_matrices=False)
    transport = _adjoint(vh) @ _adjoint(u)
    return transport, singular_values


@dataclass(frozen=True)
class OrbitalAlignment:
    aligned: Tensor
    transport: Tensor
    overlap: Tensor
    singular_values: Tensor
    residual_norm: Tensor

    @property
    def sigma_min(self) -> Tensor:
        return self.singular_values.min()


def align_orbitals(
    previous: Tensor,
    current: Tensor,
    metric: Optional[Tensor] = None,
) -> OrbitalAlignment:
    """Apply discrete parallel transport to phases, swaps and subspace rotations."""
    overlap = orbital_overlap(previous, current, metric)
    transport, singular_values = procrustes_transport(overlap)
    aligned = current @ transport
    residual = orbital_overlap(previous, aligned, metric)
    identity = torch.eye(
        residual.shape[0], dtype=residual.dtype, device=residual.device
    )
    return OrbitalAlignment(
        aligned=aligned,
        transport=transport,
        overlap=overlap,
        singular_values=singular_values,
        residual_norm=torch.linalg.matrix_norm(residual - identity),
    )


def align_global_phase(previous: Tensor, current: Tensor, epsilon: float = 1e-12) -> Tensor:
    """Align the global U(1) phase of two matching state vectors/tensors."""
    if previous.shape != current.shape:
        raise ValueError("states must have matching shapes")
    overlap = torch.vdot(previous.reshape(-1), current.reshape(-1))
    if overlap.abs() <= epsilon:
        return current
    return current * overlap.conj() / overlap.abs()


def _validate_mps(tensors: Sequence[Tensor], name: str) -> None:
    if not tensors:
        raise ValueError(f"{name} MPS cannot be empty")
    for index, tensor in enumerate(tensors):
        if tensor.ndim != 3:
            raise ValueError(f"{name} tensor {index} must have shape (left, physical, right)")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} tensor {index} must be finite")
        if index and tensors[index - 1].shape[2] != tensor.shape[0]:
            raise ValueError(f"{name} MPS has incompatible bond {index - 1}")
    if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
        raise ValueError(f"{name} MPS must have open boundary dimensions of one")


def left_canonicalize_mps(tensors: Sequence[Tensor]) -> tuple[Tensor, ...]:
    """QR sweep that fixes GL(chi) freedom to an isometric left gauge."""
    _validate_mps(tensors, "input")
    result = [tensor.clone() for tensor in tensors]
    for index in range(len(result) - 1):
        left, physical, right = result[index].shape
        if left * physical < right:
            raise ValueError(
                f"bond {index} cannot be left-canonical with right dimension {right} "
                f"greater than left*physical={left * physical}"
            )
        q, r = torch.linalg.qr(result[index].reshape(left * physical, right), mode="reduced")
        result[index] = q.reshape(left, physical, q.shape[1])
        result[index + 1] = torch.einsum("ab,bpc->apc", r, result[index + 1])
    return tuple(result)


def mps_overlap(previous: Sequence[Tensor], current: Sequence[Tensor]) -> Tensor:
    """Contract the scalar many-body overlap without expanding the full state."""
    _validate_mps(previous, "previous")
    _validate_mps(current, "current")
    if len(previous) != len(current):
        raise ValueError("MPS states must have the same number of sites")
    dtype = torch.promote_types(previous[0].dtype, current[0].dtype)
    environment = torch.ones(
        (1, 1), dtype=dtype, device=previous[0].device
    )
    for index, (left, right) in enumerate(zip(previous, current)):
        if left.shape[1] != right.shape[1]:
            raise ValueError(f"physical dimension differs at site {index}")
        environment = torch.einsum(
            "ab,apc,bpd->cd", environment, left.conj(), right
        )
    return environment.squeeze()


@dataclass(frozen=True)
class MPSAlignment:
    reference: tuple[Tensor, ...]
    aligned: tuple[Tensor, ...]
    bond_transports: tuple[Tensor, ...]
    bond_singular_values: tuple[Tensor, ...]
    state_overlap: Tensor


def align_mps(
    previous: Sequence[Tensor],
    current: Sequence[Tensor],
    *,
    canonicalize: bool = True,
    phase_epsilon: float = 1e-12,
) -> MPSAlignment:
    """Canonicalize and parallel-transport all MPS virtual bonds.

    Each right-bond rotation is compensated on the next tensor, so the physical
    current wavefunction is unchanged except for the final global phase choice.
    """
    _validate_mps(previous, "previous")
    _validate_mps(current, "current")
    if len(previous) != len(current):
        raise ValueError("MPS states must have the same number of sites")
    if [tensor.shape for tensor in previous] != [tensor.shape for tensor in current]:
        raise ValueError("parallel bond transport requires matching MPS shapes")

    reference = list(left_canonicalize_mps(previous) if canonicalize else previous)
    aligned = [tensor.clone() for tensor in (
        left_canonicalize_mps(current) if canonicalize else current
    )]
    transports, spectra = [], []
    for index in range(len(aligned) - 1):
        right = aligned[index].shape[2]
        reference_matrix = reference[index].reshape(-1, right)
        current_matrix = aligned[index].reshape(-1, right)
        overlap = _adjoint(reference_matrix) @ current_matrix
        transport, singular_values = procrustes_transport(overlap)
        aligned[index] = (current_matrix @ transport).reshape_as(aligned[index])
        aligned[index + 1] = torch.einsum(
            "ab,bpc->apc", _adjoint(transport), aligned[index + 1]
        )
        transports.append(transport)
        spectra.append(singular_values)

    state_overlap = mps_overlap(reference, aligned)
    if state_overlap.abs() > phase_epsilon:
        phase = state_overlap.conj() / state_overlap.abs()
        aligned[-1] = aligned[-1] * phase
        state_overlap = mps_overlap(reference, aligned)
    return MPSAlignment(
        reference=tuple(reference),
        aligned=tuple(aligned),
        bond_transports=tuple(transports),
        bond_singular_values=tuple(spectra),
        state_overlap=state_overlap,
    )


def wilson_loop(transports: Sequence[Tensor]) -> Tensor:
    """Ordered discrete holonomy around a closed loop of equal-size frames."""
    if not transports:
        raise ValueError("a Wilson loop requires at least one transport")
    size = transports[0].shape[0]
    result = torch.eye(size, dtype=transports[0].dtype, device=transports[0].device)
    for transport in transports:
        _validate_square(transport, "transport")
        if transport.shape != result.shape:
            raise ValueError("all loop transports must have the same shape")
        result = result @ transport
    return result


def holonomy_deviation(transports: Sequence[Tensor], area: float = 1.0) -> Tensor:
    """Gauge-invariant small-loop curvature proxy ``||H-I||_F / area``."""
    if not math.isfinite(area) or area <= 0:
        raise ValueError("plaquette area must be finite and positive")
    holonomy = wilson_loop(transports)
    identity = torch.eye(
        holonomy.shape[0], dtype=holonomy.dtype, device=holonomy.device
    )
    return torch.linalg.matrix_norm(holonomy - identity) / area


@dataclass(frozen=True)
class GaugeDiagnostics:
    sigma_min: float
    subspace_leakage: float
    curvature: Optional[float] = None
    predictive_error: Optional[float] = None
    truncation_weight: Optional[float] = None

    def __post_init__(self) -> None:
        required = {
            "sigma_min": self.sigma_min,
            "subspace_leakage": self.subspace_leakage,
        }
        optional = {
            "curvature": self.curvature,
            "predictive_error": self.predictive_error,
            "truncation_weight": self.truncation_weight,
        }
        if any(not math.isfinite(value) or value < 0 for value in required.values()):
            raise ValueError("gauge diagnostics must be finite and non-negative")
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in optional.values()
        ):
            raise ValueError("optional gauge diagnostics must be finite and non-negative")

    @classmethod
    def from_singular_values(cls, singular_values: Tensor, **kwargs) -> "GaugeDiagnostics":
        if singular_values.ndim != 1 or not singular_values.numel():
            raise ValueError("singular_values must be a nonempty vector")
        if not torch.isfinite(singular_values).all():
            raise ValueError("singular_values must be finite")
        leakage = (1.0 - singular_values.square()).clamp_min(0).sum()
        return cls(float(singular_values.min()), float(leakage), **kwargs)


class GaugeTier(IntEnum):
    STABLE = 0
    REALIGN = 1
    REFINE = 2
    REFERENCE = 3


@dataclass(frozen=True)
class GaugeTriggerConfig:
    mild_sigma: float = 0.95
    moderate_sigma: float = 0.80
    severe_sigma: float = 0.50
    curvature_check_sigma: float = 0.95
    moderate_curvature: float = 0.10
    severe_curvature: float = 0.50
    moderate_error: float = 0.05
    severe_error: float = 0.20
    truncation_threshold: float = 1e-5
    bond_growth_factor: float = 1.5

    def __post_init__(self) -> None:
        if not 0 <= self.severe_sigma <= self.moderate_sigma <= self.mild_sigma <= 1:
            raise ValueError("sigma thresholds must satisfy severe <= moderate <= mild <= 1")
        if not 0 <= self.curvature_check_sigma <= 1:
            raise ValueError("curvature_check_sigma must lie in [0, 1]")
        if not 0 <= self.moderate_curvature <= self.severe_curvature:
            raise ValueError("curvature thresholds are inconsistent")
        if not 0 <= self.moderate_error <= self.severe_error:
            raise ValueError("error thresholds are inconsistent")
        if self.truncation_threshold < 0 or self.bond_growth_factor <= 1:
            raise ValueError("invalid truncation threshold or bond growth factor")


@dataclass(frozen=True)
class GaugeAction:
    tier: GaugeTier
    realign: bool
    local_refinement: bool
    increase_bond_dimension: bool
    reference_calculation: bool
    reduce_timestep: bool
    update_replay: bool
    reason: str


class GaugeTriggerPolicy:
    """Map gauge diagnostics to conservative online-learning interventions."""

    def __init__(self, config: Optional[GaugeTriggerConfig] = None):
        self.config = config or GaugeTriggerConfig()

    def should_check_curvature(self, sigma_min: float, predictive_error: float = 0.0) -> bool:
        """Cheap first-stage gate for the more expensive plaquette diagnostic."""
        return (
            sigma_min <= self.config.curvature_check_sigma
            or predictive_error >= self.config.moderate_error
        )

    def decide(self, diagnostics: GaugeDiagnostics) -> GaugeAction:
        cfg = self.config
        curvature = diagnostics.curvature or 0.0
        error = diagnostics.predictive_error or 0.0
        truncation = diagnostics.truncation_weight or 0.0
        severe = (
            diagnostics.sigma_min <= cfg.severe_sigma
            or curvature >= cfg.severe_curvature
            or error >= cfg.severe_error
        )
        moderate = (
            diagnostics.sigma_min <= cfg.moderate_sigma
            or curvature >= cfg.moderate_curvature
            or error >= cfg.moderate_error
        )
        many_body_evidence = (
            error >= cfg.moderate_error or truncation >= cfg.truncation_threshold
        )
        if severe:
            return GaugeAction(
                GaugeTier.REFERENCE, True, True, many_body_evidence,
                True, True, True, "gauge_breakdown",
            )
        if moderate:
            return GaugeAction(
                GaugeTier.REFINE, True, True, many_body_evidence,
                False, False, False, "subspace_or_curvature_warning",
            )
        if diagnostics.sigma_min <= cfg.mild_sigma:
            return GaugeAction(
                GaugeTier.REALIGN, True, True, False,
                False, False, False, "mild_overlap_loss",
            )
        return GaugeAction(
            GaugeTier.STABLE, False, False, False,
            False, False, False, "stable",
        )

    def proposed_bond_dimension(self, current: int, maximum: int, action: GaugeAction) -> int:
        if current <= 0 or maximum < current:
            raise ValueError("bond dimensions must satisfy 0 < current <= maximum")
        if not action.increase_bond_dimension:
            return current
        return min(maximum, math.ceil(self.config.bond_growth_factor * current))
