"""Geometry-aware multiresolution electronic cochains.

Electronic spectra live on cells of a combinatorial complex instead of being
collapsed into one molecule-wide vector. Rank lifting is sparse in the number
of incidences and preserves scalar (l=0), vector (l=1), and quadrupolar (l=2)
channels in cell-local coordinate frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from tope import _core


@dataclass
class ElectronicComplexConfig:
    spectrum_dim: int = 32
    hidden_dim: int = 128
    max_ranks: int = 6
    dropout: float = 0.1
    frame_epsilon: float = 1e-7
    attention_distance_scale: float = 1.0
    rank_observable_dim: int = 8


@dataclass
class ElectronicCochain:
    """Electronic state attached to all cells of one complex rank."""

    scalar: torch.Tensor                 # (C, S), l=0 spectral channel
    vector: torch.Tensor                 # (C, S, 3), l=1 in local cell frame
    quadrupole: torch.Tensor             # (C, S, 3, 3), l=2 in local frame
    embedding: torch.Tensor              # (C, H), learned rank representation
    centers: torch.Tensor                # (C, 3)
    frames: torch.Tensor                 # (C, 3, 3), local axes as columns
    charge: torch.Tensor                 # (C,)
    dipole: torch.Tensor                 # (C, 3), local-frame charge dipole
    charge_quadrupole: torch.Tensor       # (C, 3, 3), local-frame tensor
    spectral_density: torch.Tensor       # (C, S), extensive spectral moment
    spectral_dipole: torch.Tensor        # (C, S, 3), local-frame moment
    spectral_quadrupole: torch.Tensor    # (C, S, 3, 3), local-frame moment
    coherence: torch.Tensor              # (C,), directional coherence [0,1]
    observables: torch.Tensor             # (C, O), rank-specific physics
    rank_name: str = ""


@dataclass
class MultiresolutionElectronicFingerprint:
    """Structured electronic fingerprint E(X)={F_c : c in X}."""

    cochains: List[ElectronicCochain]
    incidences: List[torch.Tensor] = field(default_factory=list)
    incidence_weights: List[torch.Tensor] = field(default_factory=list)

    def as_dict(self) -> Dict[str, ElectronicCochain]:
        return {cochain.rank_name or f"rank_{i}": cochain
                for i, cochain in enumerate(self.cochains)}


def _identity_frames(n: int, reference: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, device=reference.device, dtype=reference.dtype).expand(n, -1, -1).clone()


def _symmetric_traceless(tensor: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (tensor + tensor.transpose(-1, -2))
    trace = symmetric.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    eye = torch.eye(3, device=tensor.device, dtype=tensor.dtype)
    return symmetric - trace[..., None, None] * eye


def _segment_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
    """Differentiable segmented sum backed by ATen on CPU and accelerators."""
    output = values.new_zeros((n_segments,) + values.shape[1:])
    return output.index_add(0, segment_ids, values)


def _segment_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
    epsilon: float,
) -> torch.Tensor:
    """Softmax independently over every incidence segment."""
    maxima = logits.new_full((n_segments,), float("-inf"))
    maxima.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    unnormalized = (logits - maxima[segment_ids]).exp()
    normalizer = _segment_sum(unnormalized, segment_ids, n_segments)
    return unnormalized / normalizer[segment_ids].clamp_min(epsilon)


def _validate_incidence(
    incidence: torch.Tensor,
    n_child: int,
    device: torch.device,
) -> None:
    if incidence.ndim != 2 or incidence.size(0) != 2:
        raise ValueError("incidence must have shape (2, n_incidence)")
    if incidence.dtype != torch.long:
        raise TypeError("incidence must use torch.long indices")
    if incidence.device != device:
        raise ValueError("incidence and child tensors must be on the same device")
    if incidence.numel() == 0:
        return
    child, parent = incidence
    if int(child.min().item()) < 0 or int(child.max().item()) >= n_child:
        raise IndexError("incidence contains an out-of-range child index")
    if int(parent.min().item()) < 0:
        raise IndexError("incidence contains a negative parent index")


def local_cell_frames(
    child_coords: torch.Tensor,
    incidence: torch.Tensor,
    n_parent: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted centers and PCA frames without a Python parent loop."""
    if child_coords.ndim != 2 or child_coords.size(-1) != 3:
        raise ValueError("child_coords must have shape (n_child, 3)")
    _validate_incidence(incidence, child_coords.size(0), child_coords.device)
    child, parent = incidence
    if n_parent is None:
        n_parent = int(parent.max().item()) + 1 if parent.numel() else 0
    if n_parent < 0 or (parent.numel() and int(parent.max().item()) >= n_parent):
        raise ValueError("n_parent must cover every parent index")
    weights = weights if weights is not None else torch.ones(
        child.numel(), device=child_coords.device, dtype=child_coords.dtype
    )
    if weights.shape != (child.numel(),):
        raise ValueError("weights must have one value per incidence")
    if weights.device != child_coords.device:
        raise ValueError("weights and child_coords must be on the same device")
    if n_parent == 0:
        return child_coords.new_zeros((0, 3)), _identity_frames(0, child_coords)

    positive_weights = weights.to(child_coords.dtype).clamp_min(0)
    weight_sum = _segment_sum(positive_weights, parent, n_parent)
    normalized = positive_weights / weight_sum[parent].clamp_min(epsilon)
    edge_coords = child_coords[child]
    centers = _segment_sum(normalized[:, None] * edge_coords, parent, n_parent)
    relative = edge_coords - centers[parent]
    covariance = _segment_sum(
        normalized[:, None, None]
        * torch.einsum("ei,ej->eij", relative, relative),
        parent,
        n_parent,
    )
    counts = _segment_sum(
        torch.ones_like(parent, dtype=child_coords.dtype), parent, n_parent
    )
    has_frame = (counts >= 2) & (weight_sum > epsilon)
    identity = _identity_frames(n_parent, child_coords)
    safe_covariance = torch.where(
        has_frame[:, None, None], covariance, identity
    )
    _, eigenvectors = torch.linalg.eigh(safe_covariance)
    raw_frame = eigenvectors.flip(-1)

    # Resolve signs against the first farthest member in incidence order.
    distance2 = relative.square().sum(-1)
    farthest_distance = distance2.new_full((n_parent,), float("-inf"))
    farthest_distance.scatter_reduce_(
        0, parent, distance2, reduce="amax", include_self=True
    )
    edge_order = torch.arange(parent.numel(), device=parent.device)
    sentinel = torch.full_like(edge_order, parent.numel())
    farthest_order = torch.where(
        distance2 == farthest_distance[parent], edge_order, sentinel
    )
    anchor_index = torch.full(
        (n_parent,), parent.numel(), device=parent.device, dtype=torch.long
    )
    anchor_index.scatter_reduce_(
        0, parent, farthest_order, reduce="amin", include_self=True
    )
    safe_anchor = anchor_index.clamp_max(max(parent.numel() - 1, 0))
    anchors = relative[safe_anchor] if parent.numel() else centers
    signs = torch.where(
        torch.einsum("pi,pi->p", raw_frame[:, :, 0], anchors) < 0,
        raw_frame.new_tensor(-1.0),
        raw_frame.new_tensor(1.0),
    )
    signs1 = torch.where(
        torch.einsum("pi,pi->p", raw_frame[:, :, 1], anchors) < 0,
        raw_frame.new_tensor(-1.0),
        raw_frame.new_tensor(1.0),
    )
    axis0 = raw_frame[:, :, 0] * signs[:, None]
    axis1 = raw_frame[:, :, 1] * signs1[:, None]
    axis2 = torch.linalg.cross(axis0, axis1, dim=-1)
    axis2 = axis2 / axis2.norm(dim=-1, keepdim=True).clamp_min(epsilon)
    frames = torch.stack([axis0, axis1, axis2], dim=-1)
    frames = torch.where(has_frame[:, None, None], frames, identity)
    return centers, frames


def dense_membership_to_incidence(
    membership: torch.Tensor,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a soft child×parent membership matrix into sparse incidence."""
    child, parent = torch.where(membership > threshold)
    incidence = torch.stack([child, parent], dim=0)
    return incidence, membership[child, parent]


class GeometryConditionedElectronicPool(nn.Module):
    """Lift one electronic cochain rank through geometrically weighted incidence."""

    def __init__(self, cfg: ElectronicComplexConfig):
        super().__init__()
        self.cfg = cfg
        h, s = cfg.hidden_dim, cfg.spectrum_dim
        # child embedding + distance + local direction(3) + charge +
        # coordination + orbital alignment
        self.weight_network = nn.Sequential(
            nn.Linear(h + 7, h), nn.SiLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )
        self.rank_update = nn.Sequential(
            nn.Linear(2 * h + 8 + cfg.rank_observable_dim, h),
            nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, h), nn.LayerNorm(h),
        )
        self.spectral_encoder = nn.Sequential(nn.Linear(s, h), nn.SiLU(), nn.Linear(h, h))

    def forward(
        self,
        child_state: ElectronicCochain,
        incidence: torch.Tensor,
        incidence_weights: Optional[torch.Tensor] = None,
        coordination: Optional[torch.Tensor] = None,
        orbital_alignment: Optional[torch.Tensor] = None,
        cell_observables: Optional[torch.Tensor] = None,
        rank_name: str = "",
    ) -> ElectronicCochain:
        child, parent = incidence
        n_parent = int(parent.max().item()) + 1 if parent.numel() else 0
        dtype, device = child_state.scalar.dtype, child_state.scalar.device
        membership = incidence_weights if incidence_weights is not None else torch.ones(
            child.numel(), dtype=dtype, device=device
        )
        coordination = coordination if coordination is not None else torch.zeros(
            child_state.scalar.size(0), dtype=dtype, device=device
        )
        orbital_alignment = orbital_alignment if orbital_alignment is not None else torch.zeros(
            child.numel(), dtype=dtype, device=device
        )
        centers, frames = local_cell_frames(
            child_state.centers, incidence, n_parent, membership, self.cfg.frame_epsilon
        )
        rel = child_state.centers[child] - centers[parent]
        distance = rel.norm(dim=-1, keepdim=True)
        rel_local = torch.einsum("eij,ei->ej", frames[parent], rel)
        direction = rel_local / distance.clamp_min(self.cfg.frame_epsilon)
        geometry = torch.cat([
            distance, direction, child_state.charge[child, None],
            coordination[child, None], orbital_alignment[:, None],
        ], dim=-1)
        logits = self.weight_network(torch.cat([child_state.embedding[child], geometry], -1)).squeeze(-1)
        logits = logits + membership.clamp_min(self.cfg.frame_epsilon).log()
        # Extensive moments use a partition of unity across overlapping parent
        # cells, preventing rings/bonds that share atoms from double-counting
        # total charge or spectral density at coarser ranks.
        child_mass = membership.new_zeros(child_state.scalar.size(0))
        child_mass.scatter_add_(0, child, membership)
        physical_membership = membership / child_mass[child].clamp_min(
            self.cfg.frame_epsilon
        )

        observables = child_state.scalar.new_zeros(
            (n_parent, self.cfg.rank_observable_dim)
        ) if cell_observables is None else cell_observables
        if observables.shape != (n_parent, self.cfg.rank_observable_dim):
            raise ValueError(
                "cell_observables must have shape "
                f"({n_parent}, {self.cfg.rank_observable_dim})"
            )

        child_vector_global = torch.einsum(
            "eij,esj->esi", child_state.frames[child], child_state.vector[child]
        )
        child_vector_local = torch.einsum(
            "eij,esi->esj", frames[parent], child_vector_global
        )
        child_quadrupole_global = torch.einsum(
            "eia,esab,ejb->esij",
            child_state.frames[child], child_state.quadrupole[child], child_state.frames[child],
        )
        child_quadrupole_local = torch.einsum(
            "eia,esij,ejb->esab",
            frames[parent], child_quadrupole_global, frames[parent],
        )
        child_dipole_global = torch.einsum(
            "eij,ej->ei", child_state.frames[child], child_state.dipole[child]
        )
        child_dipole_local = torch.einsum(
            "eij,ei->ej", frames[parent], child_dipole_global
        )
        child_charge_quadrupole_global = torch.einsum(
            "eia,eab,ejb->eij",
            child_state.frames[child], child_state.charge_quadrupole[child], child_state.frames[child],
        )
        child_charge_quadrupole_local = torch.einsum(
            "eia,eij,ejb->eab",
            frames[parent], child_charge_quadrupole_global, frames[parent],
        )
        child_spectral_dipole_global = torch.einsum(
            "eij,esj->esi", child_state.frames[child], child_state.spectral_dipole[child]
        )
        child_spectral_dipole_local = torch.einsum(
            "eij,esi->esj", frames[parent], child_spectral_dipole_global
        )
        child_spectral_quadrupole_global = torch.einsum(
            "eia,esab,ejb->esij",
            child_state.frames[child], child_state.spectral_quadrupole[child], child_state.frames[child],
        )
        child_spectral_quadrupole_local = torch.einsum(
            "eia,esij,ejb->esab",
            frames[parent], child_spectral_quadrupole_global, frames[parent],
        )
        alpha = _segment_softmax(logits, parent, n_parent, self.cfg.frame_epsilon)
        scalar = _segment_sum(
            alpha[:, None] * child_state.scalar[child], parent, n_parent
        )
        vector = _segment_sum(
            alpha[:, None, None] * child_vector_local, parent, n_parent
        )
        quadrupole = _symmetric_traceless(_segment_sum(
            alpha[:, None, None, None] * child_quadrupole_local,
            parent,
            n_parent,
        ))
        pooled_embedding = _segment_sum(
            alpha[:, None] * child_state.embedding[child], parent, n_parent
        )

        edge_charge = child_state.charge[child]
        charge = _segment_sum(edge_charge * physical_membership, parent, n_parent)
        translated_dipole = child_dipole_local + edge_charge[:, None] * rel_local
        dipole = _segment_sum(
            physical_membership[:, None] * translated_dipole, parent, n_parent
        )
        eye = torch.eye(3, device=device, dtype=dtype)
        radius2 = rel_local.square().sum(-1)
        raw_q2 = (
            3.0 * torch.einsum("ei,ej->eij", rel_local, rel_local)
            - radius2[:, None, None] * eye
        )
        dipole_translation = 3.0 * (
            torch.einsum("ei,ej->eij", rel_local, child_dipole_local)
            + torch.einsum("ei,ej->eij", child_dipole_local, rel_local)
        ) - 2.0 * (rel_local * child_dipole_local).sum(-1)[:, None, None] * eye
        translated_quadrupole = (
            child_charge_quadrupole_local
            + edge_charge[:, None, None] * raw_q2
            + dipole_translation
        )
        charge_quadrupole = _segment_sum(
            physical_membership[:, None, None] * translated_quadrupole,
            parent,
            n_parent,
        )

        edge_density = child_state.spectral_density[child]
        spectral_density = _segment_sum(
            physical_membership[:, None] * edge_density, parent, n_parent
        )
        spectral_dipole = _segment_sum(
            physical_membership[:, None, None]
            * (child_spectral_dipole_local + edge_density[:, :, None] * rel_local[:, None, :]),
            parent,
            n_parent,
        )
        spectral_translation = 3.0 * (
            torch.einsum("esi,ej->esij", child_spectral_dipole_local, rel_local)
            + torch.einsum("ei,esj->esij", rel_local, child_spectral_dipole_local)
        ) - 2.0 * torch.einsum(
            "esi,ei->es", child_spectral_dipole_local, rel_local
        )[..., None, None] * eye
        spectral_quadrupole = _segment_sum(
            physical_membership[:, None, None, None]
            * (
                child_spectral_quadrupole_local
                + edge_density[:, :, None, None] * raw_q2[:, None]
                + spectral_translation
            ),
            parent,
            n_parent,
        )

        numerator = vector.norm(dim=-1).mean(dim=-1)
        denominator = _segment_sum(
            alpha[:, None] * child_vector_local.norm(dim=-1), parent, n_parent
        ).mean(dim=-1).clamp_min(self.cfg.frame_epsilon)
        coherence = (numerator / denominator).clamp(0, 1)
        moment_summary = torch.cat([
            charge[:, None],
            dipole,
            charge_quadrupole.diagonal(dim1=-2, dim2=-1),
            coherence[:, None],
        ], dim=-1)
        embedding = self.rank_update(torch.cat([
            pooled_embedding,
            self.spectral_encoder(spectral_density),
            moment_summary,
            observables,
        ], dim=-1))
        parent_counts = _segment_sum(torch.ones_like(logits), parent, n_parent)
        embedding = torch.where(
            parent_counts[:, None] > 0, embedding, torch.zeros_like(embedding)
        )

        return ElectronicCochain(
            scalar=scalar, vector=vector, quadrupole=quadrupole,
            embedding=embedding, centers=centers, frames=frames,
            charge=charge, dipole=dipole,
            charge_quadrupole=charge_quadrupole,
            spectral_density=spectral_density,
            spectral_dipole=spectral_dipole,
            spectral_quadrupole=spectral_quadrupole,
            coherence=coherence, observables=observables, rank_name=rank_name,
        )


class ElectronicComplexEncoder(nn.Module):
    """Construct the structured electronic fingerprint over all supplied ranks."""

    DEFAULT_RANK_NAMES = ("atom", "bond", "motif", "region", "domain", "molecule")

    def __init__(self, cfg: Optional[ElectronicComplexConfig] = None):
        super().__init__()
        self.cfg = cfg or ElectronicComplexConfig()
        h, s = self.cfg.hidden_dim, self.cfg.spectrum_dim
        self.atom_encoder = nn.Sequential(
            nn.Linear(s + 2, h), nn.SiLU(), nn.Dropout(self.cfg.dropout), nn.Linear(h, h)
        )
        self.pools = nn.ModuleList([
            GeometryConditionedElectronicPool(self.cfg)
            for _ in range(self.cfg.max_ranks - 1)
        ])

    def atom_cochain(
        self,
        coords: torch.Tensor,
        scalar_spectrum: torch.Tensor,
        charge: torch.Tensor,
        spin: Optional[torch.Tensor] = None,
        vector_spectrum: Optional[torch.Tensor] = None,
        quadrupole_spectrum: Optional[torch.Tensor] = None,
    ) -> ElectronicCochain:
        n, s = scalar_spectrum.shape
        if s != self.cfg.spectrum_dim:
            raise ValueError(f"expected spectrum_dim={self.cfg.spectrum_dim}, got {s}")
        spin = spin if spin is not None else torch.zeros_like(charge)
        vector_spectrum = vector_spectrum if vector_spectrum is not None else scalar_spectrum.new_zeros((n, s, 3))
        quadrupole_spectrum = quadrupole_spectrum if quadrupole_spectrum is not None else scalar_spectrum.new_zeros((n, s, 3, 3))
        quadrupole_spectrum = _symmetric_traceless(quadrupole_spectrum)
        return ElectronicCochain(
            scalar=scalar_spectrum,
            vector=vector_spectrum,
            quadrupole=quadrupole_spectrum,
            embedding=self.atom_encoder(torch.cat([scalar_spectrum, charge[:, None], spin[:, None]], -1)),
            centers=coords,
            frames=_identity_frames(n, coords),
            charge=charge,
            dipole=coords.new_zeros((n, 3)),
            charge_quadrupole=coords.new_zeros((n, 3, 3)),
            spectral_density=scalar_spectrum,
            spectral_dipole=vector_spectrum,
            spectral_quadrupole=quadrupole_spectrum,
            coherence=charge.new_ones(n),
            observables=scalar_spectrum.new_zeros(
                (n, self.cfg.rank_observable_dim)
            ),
            rank_name="atom",
        )

    def forward(
        self,
        coords: torch.Tensor,
        scalar_spectrum: torch.Tensor,
        charge: torch.Tensor,
        incidences: Sequence[torch.Tensor],
        incidence_weights: Optional[Sequence[Optional[torch.Tensor]]] = None,
        spin: Optional[torch.Tensor] = None,
        vector_spectrum: Optional[torch.Tensor] = None,
        quadrupole_spectrum: Optional[torch.Tensor] = None,
        coordination: Optional[torch.Tensor] = None,
        orbital_alignments: Optional[Sequence[Optional[torch.Tensor]]] = None,
        rank_observables: Optional[Sequence[Optional[torch.Tensor]]] = None,
        rank_names: Optional[Sequence[str]] = None,
    ) -> MultiresolutionElectronicFingerprint:
        if len(incidences) > len(self.pools):
            raise ValueError(f"at most {len(self.pools)} rank lifts are configured")
        for label, values in (
            ("incidence_weights", incidence_weights),
            ("orbital_alignments", orbital_alignments),
            ("rank_observables", rank_observables),
        ):
            if values is not None and len(values) != len(incidences):
                raise ValueError(f"{label} must align 1:1 with incidences")
        weights = list(incidence_weights or [None] * len(incidences))
        alignments = list(orbital_alignments or [None] * len(incidences))
        observables = list(rank_observables or [None] * len(incidences))
        names = list(rank_names or self.DEFAULT_RANK_NAMES[:len(incidences) + 1])
        if len(names) != len(incidences) + 1:
            raise ValueError("rank_names must include the atom rank and every parent rank")
        state = self.atom_cochain(
            coords, scalar_spectrum, charge, spin, vector_spectrum, quadrupole_spectrum
        )
        state.rank_name = names[0]
        cochains = [state]
        saved_weights = []
        for rank, incidence in enumerate(incidences):
            w = weights[rank]
            state = self.pools[rank](
                state, incidence, w,
                coordination=coordination if rank == 0 else None,
                orbital_alignment=alignments[rank],
                cell_observables=observables[rank],
                rank_name=names[rank + 1],
            )
            cochains.append(state)
            saved_weights.append(
                w if w is not None else torch.ones(
                    incidence.size(1), device=coords.device, dtype=coords.dtype
                )
            )
        return MultiresolutionElectronicFingerprint(cochains, list(incidences), saved_weights)

    def append_soft_rank(
        self,
        fingerprint: MultiresolutionElectronicFingerprint,
        membership: torch.Tensor,
        rank_name: str = "learned_domain",
        source_rank: int = -1,
    ) -> MultiresolutionElectronicFingerprint:
        """Append learned domains using Q as a soft residue→domain incidence."""
        incidence, weights = dense_membership_to_incidence(membership)
        pool_idx = min(len(fingerprint.cochains) - 1, len(self.pools) - 1)
        parent = self.pools[pool_idx](
            fingerprint.cochains[source_rank], incidence, weights, rank_name=rank_name
        )
        return MultiresolutionElectronicFingerprint(
            fingerprint.cochains + [parent],
            fingerprint.incidences + [incidence],
            fingerprint.incidence_weights + [weights],
        )


class AdaptiveElectronicReadout(nn.Module):
    """Choose both relevant cells and electronic resolution for each query."""

    def __init__(self, cfg: Optional[ElectronicComplexConfig] = None):
        super().__init__()
        self.cfg = cfg or ElectronicComplexConfig()
        h = self.cfg.hidden_dim
        self.rank_gate = nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(
        self,
        query_embedding: torch.Tensor,
        query_coords: torch.Tensor,
        fingerprint: MultiresolutionElectronicFingerprint,
        candidate_indices: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        rank_contexts, rank_scores, cell_weights = [], [], []
        scale = self.cfg.attention_distance_scale
        if candidate_indices is not None and len(candidate_indices) != len(fingerprint.cochains):
            raise ValueError("candidate_indices must align with fingerprint ranks")
        for rank, cochain in enumerate(fingerprint.cochains):
            if candidate_indices is None:
                distance = torch.cdist(query_coords, cochain.centers)
                logits = query_embedding @ cochain.embedding.transpose(0, 1)
                logits = logits / (self.cfg.hidden_dim ** 0.5) - scale * distance
                selected_embeddings = cochain.embedding.unsqueeze(0).expand(
                    query_embedding.size(0), -1, -1
                )
                valid = torch.ones_like(logits, dtype=torch.bool)
            else:
                indices = candidate_indices[rank]
                valid = indices >= 0
                safe = indices.clamp_min(0)
                selected_embeddings = cochain.embedding[safe]
                selected_centers = cochain.centers[safe]
                distance = (query_coords[:, None, :] - selected_centers).norm(dim=-1)
                logits = torch.einsum("qh,qkh->qk", query_embedding, selected_embeddings)
                logits = logits / (self.cfg.hidden_dim ** 0.5) - scale * distance
                logits = logits.masked_fill(~valid, float("-inf"))
            weights = torch.softmax(logits, dim=-1)
            context = torch.einsum("qk,qkh->qh", weights, selected_embeddings)
            nearest = distance.masked_fill(~valid, float("inf")).min(dim=-1).values[:, None]
            score = self.rank_gate(torch.cat([query_embedding, context, nearest], -1))
            rank_contexts.append(context)
            rank_scores.append(score)
            cell_weights.append(weights)
        rank_weights = torch.softmax(torch.cat(rank_scores, dim=-1), dim=-1)
        stacked = torch.stack(rank_contexts, dim=1)
        context = (rank_weights.unsqueeze(-1) * stacked).sum(dim=1)
        return context, rank_weights, cell_weights


def hierarchical_candidate_indices(
    query_coords: torch.Tensor,
    fingerprint: MultiresolutionElectronicFingerprint,
    top_coarse: int = 4,
    max_children: int = 64,
) -> List[torch.Tensor]:
    """Select coarse cells globally, then descend only through their children.

    Returned padded indices make adaptive reads proportional to the retained
    hierarchy width instead of all atom-query pairs.
    """
    n_ranks = len(fingerprint.cochains)
    if n_ranks == 0:
        return []
    if top_coarse < 1 or max_children < 1:
        raise ValueError("top_coarse and max_children must both be positive")
    if query_coords.ndim != 2 or query_coords.size(-1) != 3:
        raise ValueError("query_coords must have shape (n_query, 3)")
    if len(fingerprint.incidences) != n_ranks - 1:
        raise ValueError("fingerprint incidences must connect every adjacent rank")
    coarse = fingerprint.cochains[-1]
    if coarse.centers.size(0) == 0:
        raise ValueError("electronic fingerprint ranks cannot be empty")
    centers = [cochain.centers for cochain in fingerprint.cochains]
    for rank, center in enumerate(centers):
        if center.device != query_coords.device:
            raise ValueError("queries and all cochain centers must share a device")
        if center.dtype != query_coords.dtype:
            raise ValueError("queries and all cochain centers must share a dtype")
        if center.size(0) == 0:
            raise ValueError(f"electronic fingerprint rank {rank} is empty")
    for rank, incidence in enumerate(fingerprint.incidences):
        _validate_incidence(incidence, centers[rank].size(0), query_coords.device)
        parent = incidence[1]
        if parent.numel() and int(parent.max().item()) >= centers[rank + 1].size(0):
            raise IndexError("incidence contains an out-of-range parent index")
    return _core.hierarchical_candidates(
        query_coords,
        centers,
        fingerprint.incidences,
        top_coarse,
        max_children,
    )


class GeometryElectronicFeedback(nn.Module):
    """Energy/force interface for coupling geometry and electronic cochains."""

    def __init__(self, encoder: ElectronicComplexEncoder):
        super().__init__()
        self.encoder = encoder
        self.energy_head = nn.Sequential(
            nn.Linear(encoder.cfg.hidden_dim, encoder.cfg.hidden_dim), nn.SiLU(),
            nn.Linear(encoder.cfg.hidden_dim, 1),
        )
        self.rank_energy = nn.Parameter(torch.zeros(encoder.cfg.max_ranks))

    def energy_and_forces(self, **encoder_inputs):
        coords = encoder_inputs["coords"]
        if not coords.requires_grad:
            coords = coords.detach().requires_grad_(True)
            encoder_inputs = dict(encoder_inputs, coords=coords)
        fingerprint = self.encoder(**encoder_inputs)
        rank_weights = torch.softmax(self.rank_energy[:len(fingerprint.cochains)], dim=0)
        energy = sum(
            rank_weights[i] * self.energy_head(cochain.embedding).sum()
            for i, cochain in enumerate(fingerprint.cochains)
        )
        forces = -torch.autograd.grad(energy, coords, create_graph=self.training)[0]
        return energy, forces, fingerprint
