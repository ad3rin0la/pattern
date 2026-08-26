"""Self-supervised latent protein-domain discovery.

The module learns a soft residue-to-domain incidence matrix without domain,
family, or EC labels. Curated CATH/Pfam/ECOD annotations are deliberately not
accepted by the training API; they belong in evaluation code only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DomainDiscoveryConfig:
    hidden_dim: int = 256
    sequence_feat_dim: int = 0
    max_domains: int = 8
    min_domains: float = 1.5
    max_domain_fraction: float = 0.8
    temperature: float = 0.7
    dropout: float = 0.1
    cut_weight: float = 1.0
    continuity_weight: float = 0.1
    reconstruction_weight: float = 0.5
    contact_weight: float = 0.25
    consistency_weight: float = 0.5
    agreement_weight: float = 0.25
    complexity_weight: float = 0.1
    max_contact_pairs: int = 4096
    mask_probability: float = 0.15
    perturbation_noise: float = 0.02


def soft_domain_pool(
    h_residues: torch.Tensor,
    assignments: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool residue embeddings into ``(B, K, H)`` soft domain cells."""
    if batch is None:
        batch = torch.zeros(h_residues.size(0), dtype=torch.long, device=h_residues.device)
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    k = assignments.size(1)
    h = h_residues.size(1)
    pooled = h_residues.new_zeros((n_graphs, k, h))
    mass = assignments.new_zeros((n_graphs, k))
    for graph_id in range(n_graphs):
        mask = batch == graph_id
        q_g = assignments[mask]
        h_g = h_residues[mask]
        pooled[graph_id] = q_g.transpose(0, 1) @ h_g
        mass[graph_id] = q_g.sum(dim=0)
    pooled = pooled / mass.clamp_min(1e-8).unsqueeze(-1)
    return pooled, mass


def permutation_match(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Permute ``other`` domain columns to maximize overlap with ``reference``.

    Matching is discrete and intentionally detached. SciPy's Hungarian solver
    is used when available; a deterministic greedy fallback keeps the module
    dependency-light.
    """
    similarity = (reference.detach().transpose(0, 1) @ other.detach()).cpu()
    k = similarity.size(0)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment((-similarity.numpy()))
        order = [0] * k
        for row, col in zip(rows.tolist(), cols.tolist()):
            order[row] = col
    except ImportError:  # pragma: no cover - SciPy is an optional dependency
        available = set(range(k))
        order = []
        for row in range(k):
            col = max(available, key=lambda c: float(similarity[row, c]))
            order.append(col)
            available.remove(col)
    return other[:, torch.tensor(order, device=other.device)]


class DomainDiscoveryHead(nn.Module):
    """Predict soft latent-domain assignments and domain embeddings."""

    def __init__(self, cfg: Optional[DomainDiscoveryConfig] = None):
        super().__init__()
        self.cfg = cfg or DomainDiscoveryConfig()
        h, k = self.cfg.hidden_dim, self.cfg.max_domains
        self.structure_assignment = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(h, k),
        )
        self.sequence_assignment = None
        if self.cfg.sequence_feat_dim > 0:
            self.sequence_assignment = nn.Sequential(
                nn.LayerNorm(self.cfg.sequence_feat_dim),
                nn.Linear(self.cfg.sequence_feat_dim, h), nn.SiLU(), nn.Linear(h, k),
            )
        self.slot_presence = nn.Parameter(torch.zeros(k))
        self.reconstruction = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h),
        )

    def _assign(self, logits: torch.Tensor) -> torch.Tensor:
        gate = F.logsigmoid(self.slot_presence).unsqueeze(0)
        return torch.softmax((logits + gate) / self.cfg.temperature, dim=-1)

    def forward(
        self,
        h_residues: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        sequence_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q_structure = self._assign(self.structure_assignment(h_residues))
        q_sequence = None
        if sequence_features is not None:
            if self.sequence_assignment is None:
                raise ValueError("sequence_feat_dim must be configured for sequence evidence")
            q_sequence = self._assign(self.sequence_assignment(sequence_features))
            if batch is None:
                q_sequence = permutation_match(q_structure, q_sequence)
            else:
                aligned = torch.empty_like(q_sequence)
                for graph_id in batch.unique().tolist():
                    mask = batch == graph_id
                    aligned[mask] = permutation_match(q_structure[mask], q_sequence[mask])
                q_sequence = aligned
            assignments = 0.5 * (q_structure + q_sequence)
            assignments = assignments / assignments.sum(dim=-1, keepdim=True)
        else:
            assignments = q_structure
        domain_embeddings, domain_mass = soft_domain_pool(
            h_residues, assignments, batch
        )
        return {
            "assignments": assignments,
            "structure_assignments": q_structure,
            "sequence_assignments": q_sequence,
            "domain_embeddings": domain_embeddings,
            "domain_mass": domain_mass,
            "slot_presence": torch.sigmoid(self.slot_presence),
        }

    def losses(
        self,
        outputs: Dict[str, torch.Tensor],
        h_residues: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        chain_index: Optional[torch.Tensor] = None,
        sequence_index: Optional[torch.Tensor] = None,
        masked_residues: Optional[torch.Tensor] = None,
        perturbed_assignments: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute label-free structural, reconstruction, and stability losses."""
        q = outputs["assignments"]
        if batch is None:
            batch = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        row, col = edge_index

        # Normalized soft graph cut: tr(Q^T L Q) / tr(Q^T D Q).
        if row.numel():
            cut_num = 0.5 * (q[row] - q[col]).square().sum()
            degree = torch.zeros(q.size(0), device=q.device)
            degree.scatter_add_(0, row, torch.ones_like(row, dtype=q.dtype))
            cut_den = (degree.unsqueeze(-1) * q.square()).sum().clamp_min(1e-8)
            cut = cut_num / cut_den
        else:
            cut = q.sum() * 0.0

        # Soft sequence continuity, respecting graph and chain boundaries.
        if sequence_index is None:
            sequence_index = torch.arange(q.size(0), device=q.device)
        seq_min = sequence_index.min()
        stride = int((sequence_index.max() - seq_min).item()) + 2
        order = torch.argsort(batch * stride + (sequence_index - seq_min))
        left, right = order[:-1], order[1:]
        adjacent = (batch[left] == batch[right]) & (
            sequence_index[right] == sequence_index[left] + 1
        )
        if chain_index is not None:
            adjacent = adjacent & (chain_index[left] == chain_index[right])
        continuity = (
            (q[right[adjacent]] - q[left[adjacent]]).abs().mean()
            if adjacent.any() else q.sum() * 0.0
        )

        # Domain bottleneck reconstruction; callers can restrict it to masked residues.
        domains = outputs["domain_embeddings"]
        reconstructed = torch.empty_like(h_residues)
        for graph_id in range(domains.size(0)):
            mask = batch == graph_id
            reconstructed[mask] = q[mask] @ domains[graph_id]
        reconstructed = self.reconstruction(reconstructed)
        target_mask = masked_residues if masked_residues is not None else torch.ones(
            q.size(0), dtype=torch.bool, device=q.device
        )
        reconstruction = F.mse_loss(
            reconstructed[target_mask], h_residues.detach()[target_mask]
        ) if target_mask.any() else q.sum() * 0.0

        # Reconstruct residue contacts from same-domain probability.
        contact_terms = []
        for graph_id in batch.unique().tolist():
            nodes = torch.where(batch == graph_id)[0]
            if nodes.numel() < 2:
                continue
            in_graph = (batch[row] == graph_id) & (batch[col] == graph_id)
            pos_u, pos_v = row[in_graph], col[in_graph]
            if pos_u.numel() > self.cfg.max_contact_pairs // 2:
                pos_u = pos_u[: self.cfg.max_contact_pairs // 2]
                pos_v = pos_v[: self.cfg.max_contact_pairs // 2]
            n_negative = max(1, pos_u.numel())
            neg_u = nodes[torch.randint(nodes.numel(), (n_negative * 3,), device=q.device)]
            neg_v = nodes[torch.randint(nodes.numel(), (n_negative * 3,), device=q.device)]
            edge_codes = row[in_graph] * q.size(0) + col[in_graph]
            neg_codes = neg_u * q.size(0) + neg_v
            keep = (neg_u != neg_v) & ~torch.isin(neg_codes, edge_codes)
            neg_u, neg_v = neg_u[keep][:n_negative], neg_v[keep][:n_negative]
            probs, labels = [], []
            if pos_u.numel():
                probs.append((q[pos_u] * q[pos_v]).sum(dim=-1))
                labels.append(torch.ones(pos_u.numel(), device=q.device))
            if neg_u.numel():
                probs.append((q[neg_u] * q[neg_v]).sum(dim=-1))
                labels.append(torch.zeros(neg_u.numel(), device=q.device))
            if probs:
                contact_terms.append(F.binary_cross_entropy(
                    torch.cat(probs).clamp(1e-6, 1 - 1e-6), torch.cat(labels)
                ))
        contact = torch.stack(contact_terms).mean() if contact_terms else q.sum() * 0.0

        consistency_terms = []
        if perturbed_assignments is not None:
            for graph_id in batch.unique().tolist():
                mask = batch == graph_id
                matched = permutation_match(q[mask], perturbed_assignments[mask])
                consistency_terms.append(F.mse_loss(q[mask], matched))
        consistency = (
            torch.stack(consistency_terms).mean() if consistency_terms else q.sum() * 0.0
        )

        q_sequence = outputs.get("sequence_assignments")
        if q_sequence is not None:
            agreement_terms = []
            q_structure = outputs["structure_assignments"]
            for graph_id in batch.unique().tolist():
                mask = batch == graph_id
                matched = permutation_match(q_structure[mask], q_sequence[mask])
                agreement_terms.append(F.mse_loss(q_structure[mask], matched))
            agreement = torch.stack(agreement_terms).mean()
        else:
            agreement = q.sum() * 0.0

        # Prevent single-domain collapse while allowing unused slots.
        _, mass = soft_domain_pool(h_residues, q, batch)
        fractions = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        effective_domains = 1.0 / fractions.square().sum(dim=-1).clamp_min(1e-8)
        collapse = F.relu(self.cfg.min_domains - effective_domains).square().mean()
        oversized = F.relu(fractions.max(dim=-1).values - self.cfg.max_domain_fraction).square().mean()
        assignment_entropy = -(q * q.clamp_min(1e-8).log()).sum(dim=-1).mean()
        unnecessary_slots = outputs["slot_presence"].mean()
        complexity = collapse + oversized + 0.01 * assignment_entropy + 0.01 * unnecessary_slots

        total = (
            self.cfg.cut_weight * cut
            + self.cfg.continuity_weight * continuity
            + self.cfg.reconstruction_weight * reconstruction
            + self.cfg.contact_weight * contact
            + self.cfg.consistency_weight * consistency
            + self.cfg.agreement_weight * agreement
            + self.cfg.complexity_weight * complexity
        )
        return {
            "total": total,
            "cut": cut,
            "continuity": continuity,
            "reconstruction": reconstruction,
            "contact": contact,
            "consistency": consistency,
            "agreement": agreement,
            "complexity": complexity,
        }


class DomainSubstrateAttention(nn.Module):
    """Attend from a substrate-conditioned enzyme query to learned domains."""

    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        substrate_context: torch.Tensor,
        domain_embeddings: torch.Tensor,
        domain_mass: torch.Tensor,
        masked_domains: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = domain_mass <= 1e-6
        if masked_domains is not None:
            key_padding_mask = key_padding_mask | masked_domains.bool()
        attended, weights = self.attention(
            substrate_context.unsqueeze(1), domain_embeddings, domain_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.norm(substrate_context + attended.squeeze(1)), weights.squeeze(1)
