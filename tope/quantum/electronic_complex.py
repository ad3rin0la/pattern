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


def local_cell_frames(
    child_coords: torch.Tensor,
    incidence: torch.Tensor,
    n_parent: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted cell centers and deterministic PCA coordinate frames."""
    child, parent = incidence
    if n_parent is None:
        n_parent = int(parent.max().item()) + 1 if parent.numel() else 0
    weights = weights if weights is not None else torch.ones(
        child.numel(), device=child_coords.device, dtype=child_coords.dtype
    )
    centers = child_coords.new_zeros((n_parent, 3))
    frames = _identity_frames(n_parent, child_coords)
    for parent_id in range(n_parent):
        mask = parent == parent_id
        if not mask.any():
            continue
        coords = child_coords[child[mask]]
        w = weights[mask].clamp_min(0)
        w = w / w.sum().clamp_min(epsilon)
        center = (w.unsqueeze(-1) * coords).sum(0)
        centers[parent_id] = center
        if coords.size(0) < 2:
            continue
        rel = coords - center
        covariance = torch.einsum("n,ni,nj->ij", w, rel, rel)
        _, eigenvectors = torch.linalg.eigh(covariance)
        frame = eigenvectors.flip(-1)
        # Resolve eigenvector signs against the farthest member, then enforce
        # a right-handed frame. Degenerate cells retain a stable fallback.
        anchor = rel[rel.square().sum(-1).argmax()]
        sign0 = torch.where(
            torch.dot(frame[:, 0], anchor) < 0,
            frame.new_tensor(-1.0), frame.new_tensor(1.0),
        )
        sign1 = torch.where(
            torch.dot(frame[:, 1], anchor) < 0,
            frame.new_tensor(-1.0), frame.new_tensor(1.0),
        )
        axis0 = frame[:, 0] * sign0
        axis1 = frame[:, 1] * sign1
        axis2 = torch.linalg.cross(axis0, axis1)
        axis2 = axis2 / axis2.norm().clamp_min(epsilon)
        frame = torch.stack([axis0, axis1, axis2], dim=-1)
        frames[parent_id] = frame
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

        scalar = child_state.scalar.new_zeros((n_parent, self.cfg.spectrum_dim))
        vector = child_state.vector.new_zeros((n_parent, self.cfg.spectrum_dim, 3))
        quadrupole = child_state.quadrupole.new_zeros((n_parent, self.cfg.spectrum_dim, 3, 3))
        embedding = child_state.embedding.new_zeros((n_parent, self.cfg.hidden_dim))
        charge = child_state.charge.new_zeros(n_parent)
        dipole = child_state.centers.new_zeros((n_parent, 3))
        charge_quadrupole = child_state.centers.new_zeros((n_parent, 3, 3))
        spectral_density = scalar.clone()
        spectral_dipole = vector.clone()
        spectral_quadrupole = quadrupole.clone()
        coherence = child_state.charge.new_zeros(n_parent)
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
        for parent_id in range(n_parent):
            mask = parent == parent_id
            if not mask.any():
                continue
            alpha = torch.softmax(logits[mask], dim=0)
            physical_w = physical_membership[mask]
            child_ids = child[mask]
            scalar[parent_id] = (alpha[:, None] * child_state.scalar[child_ids]).sum(0)
            vector[parent_id] = (alpha[:, None, None] * child_vector_local[mask]).sum(0)
            quadrupole[parent_id] = _symmetric_traceless(
                (alpha[:, None, None, None] * child_quadrupole_local[mask]).sum(0)
            )
            pooled_embedding = (alpha[:, None] * child_state.embedding[child_ids]).sum(0)

            rel_p = rel_local[mask]
            child_charge = child_state.charge[child_ids]
            q = child_charge * physical_w
            charge[parent_id] = q.sum()
            translated_dipole = child_dipole_local[mask] + child_charge[:, None] * rel_p
            dipole[parent_id] = (physical_w[:, None] * translated_dipole).sum(0)
            r2 = rel_p.square().sum(-1)
            eye = torch.eye(3, device=device, dtype=dtype)
            raw_q2 = 3.0 * torch.einsum("ni,nj->nij", rel_p, rel_p) - r2[:, None, None] * eye
            mu = child_dipole_local[mask]
            dipole_translation = 3.0 * (
                torch.einsum("ni,nj->nij", rel_p, mu)
                + torch.einsum("ni,nj->nij", mu, rel_p)
            ) - 2.0 * (rel_p * mu).sum(-1)[:, None, None] * eye
            translated_quadrupole = (
                child_charge_quadrupole_local[mask]
                + child_charge[:, None, None] * raw_q2
                + dipole_translation
            )
            charge_quadrupole[parent_id] = (
                physical_w[:, None, None] * translated_quadrupole
            ).sum(0)

            child_density = child_state.spectral_density[child_ids] * physical_w[:, None]
            spectral_density[parent_id] = child_density.sum(0)
            spectral_dipole[parent_id] = (
                physical_w[:, None, None] * (
                    child_spectral_dipole_local[mask]
                    + child_state.spectral_density[child_ids, :, None] * rel_p[:, None, :]
                )
            ).sum(0)
            spectral_mu = child_spectral_dipole_local[mask]
            spectral_translation = 3.0 * (
                torch.einsum("nsi,nj->nsij", spectral_mu, rel_p)
                + torch.einsum("ni,nsj->nsij", rel_p, spectral_mu)
            ) - 2.0 * torch.einsum("nsi,ni->ns", spectral_mu, rel_p)[..., None, None] * eye
            spectral_quadrupole[parent_id] = (
                physical_w[:, None, None, None] * (
                    child_spectral_quadrupole_local[mask]
                    + child_state.spectral_density[child_ids, :, None, None]
                    * raw_q2[:, None]
                    + spectral_translation
                )
            ).sum(0)

            numerator = vector[parent_id].norm(dim=-1).mean()
            denominator = (
                alpha[:, None] * child_vector_local[mask].norm(dim=-1)
            ).sum(0).mean().clamp_min(self.cfg.frame_epsilon)
            coherence[parent_id] = (numerator / denominator).clamp(0, 1)
            moment_summary = torch.cat([
                charge[parent_id, None], dipole[parent_id],
                charge_quadrupole[parent_id].diagonal(), coherence[parent_id, None],
            ])
            embedding[parent_id] = self.rank_update(torch.cat([
                pooled_embedding,
                self.spectral_encoder(spectral_density[parent_id]),
                moment_summary, observables[parent_id],
            ]))

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
    n_queries = query_coords.size(0)
    candidates: List[Optional[torch.Tensor]] = [None] * n_ranks
    coarse = fingerprint.cochains[-1]
    if coarse.centers.size(0) == 0:
        raise ValueError("electronic fingerprint ranks cannot be empty")
    coarse_k = min(top_coarse, coarse.centers.size(0))
    candidates[-1] = torch.cdist(query_coords, coarse.centers).topk(
        coarse_k, largest=False
    ).indices

    for rank in range(n_ranks - 2, -1, -1):
        child, parent = fingerprint.incidences[rank]
        selected_per_query: List[torch.Tensor] = []
        max_width = 1
        for query_id in range(n_queries):
            selected_parents = candidates[rank + 1][query_id]
            selected_parents = selected_parents[selected_parents >= 0]
            mask = torch.isin(parent, selected_parents)
            child_ids = child[mask].unique()
            if child_ids.numel() == 0:
                child_ids = torch.arange(
                    fingerprint.cochains[rank].centers.size(0), device=query_coords.device
                )
            distance = (
                fingerprint.cochains[rank].centers[child_ids]
                - query_coords[query_id]
            ).norm(dim=-1)
            keep = min(max_children, child_ids.numel())
            child_ids = child_ids[distance.topk(keep, largest=False).indices]
            selected_per_query.append(child_ids)
            max_width = max(max_width, child_ids.numel())
        padded = torch.full(
            (n_queries, max_width), -1, dtype=torch.long, device=query_coords.device
        )
        for query_id, child_ids in enumerate(selected_per_query):
            padded[query_id, :child_ids.numel()] = child_ids
        candidates[rank] = padded
    if any(candidate is None for candidate in candidates):
        raise RuntimeError("failed to construct candidates for every electronic rank")
    return [candidate for candidate in candidates if candidate is not None]


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
