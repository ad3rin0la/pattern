"""Spectral heat diffusion on molecular chain complexes.

This module generalizes circle/torus harmonic diffusion to signed molecular
boundary operators.  Pooling memberships (including learned domain assignments
``Q``) are deliberately not converted into boundaries: a Hodge complex must
satisfy the chain condition ``B_k @ B_{k+1} = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from tope.quantum.electronic_complex import MultiresolutionElectronicFingerprint


@dataclass
class HodgeDiffusionConfig:
    """Numerical controls for molecular Hodge spectra and heat diffusion."""

    max_modes: Optional[int] = None
    chain_tolerance: float = 1e-5
    eigenvalue_tolerance: float = 1e-7
    validate_chain: bool = True

    def __post_init__(self) -> None:
        if self.max_modes is not None and self.max_modes < 1:
            raise ValueError("max_modes must be positive or None")
        if self.chain_tolerance < 0 or self.eigenvalue_tolerance < 0:
            raise ValueError("Hodge tolerances must be non-negative")


@dataclass
class HodgeSpectrum:
    """Eigenbasis of one molecular cochain rank."""

    rank: int
    laplacian: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


@dataclass
class HodgeDiffusionResult:
    """Modal coefficients and reconstructions over one or more diffusion times."""

    spectrum: HodgeSpectrum
    times: torch.Tensor
    coefficients: torch.Tensor
    diffused_coefficients: torch.Tensor
    reconstruction: torch.Tensor


@dataclass
class HodgeDecomposition:
    """Orthogonal exact, coexact, and harmonic parts of a k-cochain."""

    exact: torch.Tensor
    coexact: torch.Tensor
    harmonic: torch.Tensor


def validate_boundary_operators(
    boundaries: Sequence[torch.Tensor], tolerance: float = 1e-5
) -> None:
    """Validate dimensions and the boundary-of-a-boundary identity."""

    if not boundaries:
        raise ValueError("at least one boundary operator is required")
    if tolerance < 0:
        raise ValueError("chain tolerance must be non-negative")
    reference = boundaries[0]
    for index, boundary in enumerate(boundaries):
        if boundary.dim() != 2:
            raise ValueError(f"B_{index + 1} must be a matrix")
        if not boundary.is_floating_point():
            raise TypeError("boundary operators must use floating values")
        if boundary.layout != torch.strided:
            raise TypeError("boundary operators must currently be dense tensors")
        if boundary.device != reference.device or boundary.dtype != reference.dtype:
            raise ValueError("boundary operators must share one device and dtype")
        if not torch.isfinite(boundary).all():
            raise ValueError("boundary operators must contain only finite values")
        if index:
            previous = boundaries[index - 1]
            if previous.size(1) != boundary.size(0):
                raise ValueError(
                    f"B_{index} and B_{index + 1} have incompatible cell counts"
                )
            residue = previous @ boundary
            if residue.numel() and residue.abs().max().item() > tolerance:
                raise ValueError(
                    f"chain condition failed: B_{index} @ B_{index + 1} != 0; "
                    "unsigned memberships are not Hodge boundary operators"
                )


def hodge_laplacian(
    boundaries: Sequence[torch.Tensor], rank: int, validate: bool = True,
    chain_tolerance: float = 1e-5,
) -> torch.Tensor:
    """Construct ``Delta_k = B_k^T B_k + B_{k+1} B_{k+1}^T``."""

    if not boundaries:
        raise ValueError("at least one boundary operator is required")
    if validate:
        validate_boundary_operators(boundaries, chain_tolerance)
    if rank < 0 or rank > len(boundaries):
        raise ValueError(f"rank must lie in [0, {len(boundaries)}]")

    if rank == 0:
        n_cells = boundaries[0].size(0)
    else:
        n_cells = boundaries[rank - 1].size(1)
    reference = boundaries[min(rank, len(boundaries) - 1)]
    laplacian = torch.zeros(
        n_cells, n_cells, dtype=reference.dtype, device=reference.device
    )
    if rank > 0:
        lower = boundaries[rank - 1]
        laplacian = laplacian + lower.transpose(-2, -1) @ lower
    if rank < len(boundaries):
        upper = boundaries[rank]
        laplacian = laplacian + upper @ upper.transpose(-2, -1)
    return 0.5 * (laplacian + laplacian.transpose(-2, -1))


def hodge_decomposition(
    signal: torch.Tensor,
    boundaries: Sequence[torch.Tensor],
    rank: int,
    *,
    chain_tolerance: float = 1e-5,
    rcond: Optional[float] = None,
) -> HodgeDecomposition:
    """Split a k-cochain into exact, coexact, and harmonic components.

    Exact cochains lie in ``im(B_k^T)`` and coexact cochains in
    ``im(B_{k+1})``.  The chain identity makes these subspaces orthogonal; the
    residual is the harmonic component in the joint kernel of the lower and
    upper Hodge terms.
    """

    validate_boundary_operators(boundaries, chain_tolerance)
    if rcond is not None and rcond < 0:
        raise ValueError("rcond must be non-negative or None")
    if rank < 0 or rank > len(boundaries):
        raise ValueError(f"rank must lie in [0, {len(boundaries)}]")
    n_cells = boundaries[0].size(0) if rank == 0 else boundaries[rank - 1].size(1)
    squeeze = signal.dim() == 1
    working = signal[:, None] if squeeze else signal
    if working.dim() != 2 or working.size(0) != n_cells:
        raise ValueError("signal must have shape (rank_cells, channels)")

    def project_onto_columns(image: torch.Tensor) -> torch.Tensor:
        if image.size(1) == 0:
            return torch.zeros_like(working)
        image = image.to(dtype=working.dtype, device=working.device)
        if rcond is None:
            return image @ (torch.linalg.pinv(image) @ working)
        return image @ (torch.linalg.pinv(image, rtol=rcond) @ working)

    exact = torch.zeros_like(working)
    if rank > 0:
        exact = project_onto_columns(boundaries[rank - 1].transpose(0, 1))
    coexact = torch.zeros_like(working)
    if rank < len(boundaries):
        coexact = project_onto_columns(boundaries[rank])
    harmonic = working - exact - coexact
    if squeeze:
        exact, coexact, harmonic = exact[:, 0], coexact[:, 0], harmonic[:, 0]
    return HodgeDecomposition(exact, coexact, harmonic)


class HodgeHeatDiffusion(nn.Module):
    """Project cochains onto Hodge modes and apply ``exp(-t Delta_k)``."""

    def __init__(self, config: Optional[HodgeDiffusionConfig] = None):
        super().__init__()
        self.config = config or HodgeDiffusionConfig()

    def spectra(self, boundaries: Sequence[torch.Tensor]) -> List[HodgeSpectrum]:
        if not boundaries:
            raise ValueError("at least one boundary operator is required")
        if self.config.validate_chain:
            validate_boundary_operators(boundaries, self.config.chain_tolerance)
        spectra = []
        for rank in range(len(boundaries) + 1):
            laplacian = hodge_laplacian(boundaries, rank, validate=False)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
            eigenvalues = eigenvalues.clamp_min(0.0)
            eigenvalues = torch.where(
                eigenvalues < self.config.eigenvalue_tolerance,
                torch.zeros_like(eigenvalues), eigenvalues,
            )
            if self.config.max_modes is not None:
                keep = min(self.config.max_modes, eigenvalues.numel())
                eigenvalues, eigenvectors = eigenvalues[:keep], eigenvectors[:, :keep]
            spectra.append(HodgeSpectrum(rank, laplacian, eigenvalues, eigenvectors))
        return spectra

    def diffuse(
        self, signal: torch.Tensor, spectrum: HodgeSpectrum,
        times: Union[torch.Tensor, float],
    ) -> HodgeDiffusionResult:
        """Diffuse an ``(n_k, channels)`` cochain at each requested time."""

        if spectrum.eigenvectors.dim() != 2 or spectrum.eigenvalues.dim() != 1:
            raise ValueError("spectrum must contain a matrix basis and vector eigenvalues")
        if spectrum.eigenvectors.size(1) != spectrum.eigenvalues.numel():
            raise ValueError("spectrum eigenvectors and eigenvalues do not align")
        if signal.dim() == 1:
            signal = signal[:, None]
        if signal.dim() != 2 or signal.size(0) != spectrum.eigenvectors.size(0):
            raise ValueError("signal must have shape (rank_cells, channels)")
        if not torch.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")
        real_dtype = signal.real.dtype
        time_tensor = torch.as_tensor(times, dtype=real_dtype, device=signal.device).reshape(-1)
        if not torch.isfinite(time_tensor).all():
            raise ValueError("diffusion times must be finite")
        if (time_tensor < 0).any():
            raise ValueError("diffusion times must be non-negative")
        modes = spectrum.eigenvectors.to(dtype=signal.dtype, device=signal.device)
        eigenvalues = spectrum.eigenvalues.to(dtype=real_dtype, device=signal.device)
        coefficients = modes.transpose(0, 1) @ signal
        decay = torch.exp(-time_tensor[:, None] * eigenvalues[None, :])
        diffused = decay[:, :, None] * coefficients[None, :, :]
        reconstruction = torch.einsum("nm,tmc->tnc", modes, diffused)
        return HodgeDiffusionResult(
            spectrum, time_tensor, coefficients, diffused, reconstruction
        )

    def forward(
        self, signals: Sequence[torch.Tensor], boundaries: Sequence[torch.Tensor],
        times: Union[torch.Tensor, float],
    ) -> List[HodgeDiffusionResult]:
        spectra = self.spectra(boundaries)
        if len(signals) != len(spectra):
            raise ValueError("signals must provide one cochain for every Hodge rank")
        return [self.diffuse(signal, spectrum, times)
                for signal, spectrum in zip(signals, spectra)]


class ElectronicHodgeDiffusion(nn.Module):
    """Apply Hodge heat diffusion to fields stored on an electronic complex."""

    ALLOWED_FIELDS = {
        "embedding", "scalar", "vector", "quadrupole", "charge", "dipole",
        "charge_quadrupole", "coherence", "observables", "spectral_density",
        "spectral_dipole", "spectral_quadrupole",
    }
    VECTOR_FIELDS = {"vector", "dipole", "spectral_dipole"}
    TENSOR_FIELDS = {"quadrupole", "charge_quadrupole", "spectral_quadrupole"}

    def __init__(self, config: Optional[HodgeDiffusionConfig] = None):
        super().__init__()
        self.heat = HodgeHeatDiffusion(config)

    def forward(
        self, fingerprint: MultiresolutionElectronicFingerprint,
        boundaries: Sequence[torch.Tensor], times: Union[torch.Tensor, float],
        field: str = "embedding",
        frame: str = "global",
    ) -> List[HodgeDiffusionResult]:
        if field not in self.ALLOWED_FIELDS:
            raise ValueError(f"unsupported electronic field: {field}")
        if len(fingerprint.cochains) != len(boundaries) + 1:
            raise ValueError("electronic ranks must align with signed boundary operators")
        if frame not in {"global", "local"}:
            raise ValueError("frame must be global or local")
        signals = []
        for cochain in fingerprint.cochains:
            signal = getattr(cochain, field)
            if frame == "global" and field in self.VECTOR_FIELDS:
                if signal.dim() == 2:
                    signal = torch.einsum("cij,cj->ci", cochain.frames, signal)
                else:
                    signal = torch.einsum("cij,csj->csi", cochain.frames, signal)
            elif frame == "global" and field in self.TENSOR_FIELDS:
                if signal.dim() == 3:
                    signal = torch.einsum(
                        "cia,cab,cjb->cij", cochain.frames, signal, cochain.frames
                    )
                else:
                    signal = torch.einsum(
                        "cia,csab,cjb->csij", cochain.frames, signal, cochain.frames
                    )
            if signal.dim() == 1:
                signal = signal[:, None]
            elif signal.dim() > 2:
                signal = signal.flatten(start_dim=1)
            signals.append(signal)
        return self.heat(signals, boundaries, times)


def torus_heat_decay(
    coefficients: torch.Tensor, wavevectors: torch.Tensor,
    times: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Diffuse Fourier modes on a coupled torsion torus.

    A mode with integer wavevector ``k`` receives ``exp(-t ||k||^2)``.  The
    independent-circle case is recovered when each wavevector has one nonzero
    component; coupled terms such as ``cos(theta_i-theta_j)`` use ``(1, -1)``.
    """

    if coefficients.dim() < 1 or wavevectors.dim() != 2 or coefficients.size(0) != wavevectors.size(0):
        raise ValueError("coefficients and wavevectors must share their mode axis")
    if not torch.isfinite(coefficients).all() or not torch.isfinite(wavevectors).all():
        raise ValueError("coefficients and wavevectors must be finite")
    if not torch.equal(wavevectors, wavevectors.round()):
        raise ValueError("torus wavevectors must be integer-valued")
    real_dtype = coefficients.real.dtype
    time_tensor = torch.as_tensor(times, dtype=real_dtype, device=coefficients.device).reshape(-1)
    if not torch.isfinite(time_tensor).all():
        raise ValueError("diffusion times must be finite")
    if (time_tensor < 0).any():
        raise ValueError("diffusion times must be non-negative")
    eigenvalues = wavevectors.to(
        dtype=real_dtype, device=coefficients.device
    ).square().sum(dim=-1)
    decay = torch.exp(-time_tensor[:, None] * eigenvalues[None, :])
    return decay[(...,) + (None,) * (coefficients.dim() - 1)] * coefficients.unsqueeze(0)


@dataclass
class CoupledTorsionState:
    """Sparse torus features for torsion cells selected by molecular topology."""

    raw_features: torch.Tensor
    cell_embeddings: torch.Tensor
    torsions_per_cell: torch.Tensor


class CoupledTorsionHarmonics(nn.Module):
    """Encode only torsion groups supplied as higher-order molecular cells.

    Each cell contains independent harmonics plus pairwise difference and sum
    modes.  Difference modes explicitly expose correlated terms such as
    ``cos(theta_i - theta_j)`` without constructing the complete ``T^n``.
    """

    def __init__(self, max_frequency: int = 3, hidden_dim: int = 64):
        super().__init__()
        if max_frequency < 1:
            raise ValueError("max_frequency must be positive")
        self.max_frequency = max_frequency
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(6 * max_frequency, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, angles: torch.Tensor, torsion_cell_incidence: torch.Tensor,
        n_cells: Optional[int] = None,
    ) -> CoupledTorsionState:
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        if angles.dim() != 2:
            raise ValueError("angles must have shape (torsions,) or (batch, torsions)")
        if not angles.is_floating_point():
            raise TypeError("angles must use a floating dtype")
        if not torch.isfinite(angles).all():
            raise ValueError("angles must contain only finite values")
        if torsion_cell_incidence.dim() != 2 or torsion_cell_incidence.size(0) != 2:
            raise ValueError("torsion_cell_incidence must have shape (2, memberships)")
        if torsion_cell_incidence.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise TypeError("torsion_cell_incidence must contain integer indices")
        torsion_ids, cell_ids = torsion_cell_incidence.long()
        if torsion_ids.numel() and (
            torsion_ids.min() < 0 or torsion_ids.max() >= angles.size(1)
        ):
            raise ValueError("torsion incidence contains an invalid torsion index")
        if cell_ids.numel() and cell_ids.min() < 0:
            raise ValueError("torsion incidence contains a negative cell index")
        if torsion_ids.numel():
            pairs = torch.stack([torsion_ids, cell_ids], dim=-1)
            if torch.unique(pairs, dim=0).size(0) != pairs.size(0):
                raise ValueError("torsion incidence contains duplicate memberships")
        inferred = int(cell_ids.max().item()) + 1 if cell_ids.numel() else 0
        n_cells = inferred if n_cells is None else n_cells
        if n_cells < 0:
            raise ValueError("n_cells must be non-negative")
        if n_cells < inferred:
            raise ValueError("n_cells is smaller than the incidence requires")

        batch, _, frequency = angles.size(0), angles.size(1), self.max_frequency
        features, counts = [], []
        harmonics = torch.arange(
            1, frequency + 1, dtype=angles.dtype, device=angles.device
        )
        for cell in range(n_cells):
            members = torch.sort(torsion_ids[cell_ids == cell]).values
            values = angles[:, members]
            counts.append(members.numel())
            if members.numel():
                phase = values[..., None] * harmonics
                independent = torch.cat(
                    [phase.cos().mean(1), phase.sin().mean(1)], dim=-1
                )
            else:
                independent = angles.new_zeros(batch, 2 * frequency)
            if members.numel() >= 2:
                pairs = torch.triu_indices(
                    members.numel(), members.numel(), offset=1, device=angles.device
                )
                first, second = values[:, pairs[0]], values[:, pairs[1]]
                difference = (first - second)[..., None] * harmonics
                summation = (first + second)[..., None] * harmonics
                coupled = torch.cat([
                    difference.cos().mean(1), difference.sin().mean(1),
                    summation.cos().mean(1), summation.sin().mean(1),
                ], dim=-1)
            else:
                coupled = angles.new_zeros(batch, 4 * frequency)
            features.append(torch.cat([independent, coupled], dim=-1))
        raw = torch.stack(features, dim=1) if features else angles.new_zeros(
            batch, 0, 6 * frequency
        )
        count_tensor = torch.tensor(counts, dtype=torch.long, device=angles.device)
        return CoupledTorsionState(raw, self.encoder(raw), count_tensor)
