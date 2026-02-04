"""
Whole-Protein TCPNet — Hierarchical Message Passing
=====================================================

Extends the active-site TCPNet to operate over the entire protein via a
three-zone multi-scale graph.  Information propagates from atomic detail
at the active site through residue-level shells out to distant structural
context, and back — capturing allosteric effects and long-range
electrostatics.

Architecture
------------
1. Zone-specific linear embeddings project heterogeneous node features
   into a common hidden dimension.
2. Deep message-passing layers (10 by default) enable information to
   travel > 100 Å across the protein.
3. Zone-aware attention pooling weights the contribution of each residue
   zone to the global enzyme embedding.

The encoder supports **curriculum depth scheduling**: early training uses
fewer message-passing layers (local context), gradually increasing to the
full depth (whole-protein context).

References
----------
Wang et al., Topotein, arXiv 2509.03885 (2025).
SchNet, Schütt et al., NeurIPS 2017.
DimeNet++, Gasteiger et al., ICLR 2021.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tope_model.tcpnet import GaussianRBF, CosineCutoff


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class WholeProteinConfig:
    """Hyper-parameters for the whole-protein TCPNet."""

    zone1_feat_dim: int = 128      # Zone 1 input feature dim
    zone2_feat_dim: int = 64       # Zone 2 input feature dim
    zone3_feat_dim: int = 32       # Zone 3 input feature dim
    edge_feat_dim: int = 4         # Edge feature dim (from graph builder)
    hidden_dim: int = 256          # Shared hidden dim
    n_message_layers: int = 10     # Depth — increased from 6 for long range
    n_rbf: int = 20                # Gaussian RBF centres for distance encoding
    distance_cutoff: float = 15.0  # Å — RBF / envelope cutoff
    dropout: float = 0.1
    max_degree: int = 3            # Unused until e3nn upgrade; kept for compat


# ── Equivariant message function ─────────────────────────────────────────────

class EquivariantMessageFunction(nn.Module):
    """SE(3)-aware message using radial basis distance encoding.

    Uses distance-gated messages with learned Gaussian RBF features.
    Full tensor-product SE(3) equivariance is available via the
    ``EnzymeTCPNet`` path; this module provides a lighter alternative
    suitable for the residue-level whole-protein graph.
    """

    def __init__(self, hidden_dim: int, n_rbf: int = 20, cutoff: float = 15.0):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_src     : (E, H)  source node features
        h_tgt     : (E, H)  target node features
        edge_dist : (E, 1)  pairwise distances

        Returns
        -------
        messages : (E, H)
        """
        rbf_feat = self.rbf(edge_dist.squeeze(-1))           # (E, n_rbf)
        envelope = self.cutoff_fn(edge_dist.squeeze(-1))      # (E,)

        msg_in = torch.cat([h_src, h_tgt, rbf_feat], dim=-1)
        messages = self.message_mlp(msg_in) * envelope.unsqueeze(-1)
        return messages


# ── Single message-passing layer ──────────────────────────────────────────────

class ProteinMessagePassingLayer(nn.Module):
    """One residue-to-residue message-passing layer."""

    def __init__(self, hidden_dim: int, n_rbf: int = 20, cutoff: float = 15.0,
                 dropout: float = 0.1):
        super().__init__()
        self.message_fn = EquivariantMessageFunction(hidden_dim, n_rbf, cutoff)

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        edge_vec = coords[row] - coords[col]
        edge_dist = edge_vec.norm(dim=-1, keepdim=True)

        messages = self.message_fn(h_nodes[row], h_nodes[col], edge_dist)

        # Aggregate → nodes (scatter-add to target)
        node_agg = torch.zeros_like(h_nodes)
        node_agg.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)

        h_nodes_new = self.node_update(torch.cat([h_nodes, node_agg], dim=-1))
        h_edges_new = self.edge_update(
            torch.cat([h_edges, h_nodes[row], h_nodes[col]], dim=-1)
        )

        return h_nodes_new, h_edges_new


# ── Zone-aware attention pooling ──────────────────────────────────────────────

class ZoneAwareAttention(nn.Module):
    """Global attention pooling with learned zone biases.

    Active-site residues (Zone 1) receive a higher prior; the model
    learns to up-weight distant residues when they carry allosteric signal.
    """

    def __init__(self, hidden_dim: int, n_zones: int = 3):
        super().__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learnable zone priors — Zone 1 starts with higher bias
        self.zone_bias = nn.Parameter(torch.tensor([2.0, 0.5, 0.0]))

    def forward(
        self,
        h_nodes: torch.Tensor,
        zone_assignments: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_nodes          : (N, H)
        zone_assignments : (N,)  values in {0, 1, 2}
        batch            : (N,)  graph membership (optional)

        Returns
        -------
        pooled : (B, H)
        """
        attn_logits = self.attention_mlp(h_nodes).squeeze(-1)  # (N,)
        zone_bias = self.zone_bias[zone_assignments]            # (N,)
        attn_logits = attn_logits + zone_bias

        if batch is None:
            # Single graph
            weights = torch.softmax(attn_logits, dim=0)
            return (weights.unsqueeze(-1) * h_nodes).sum(dim=0, keepdim=True)

        # Per-graph softmax
        weights = torch.zeros_like(attn_logits)
        for gid in batch.unique():
            mask = batch == gid
            weights[mask] = torch.softmax(attn_logits[mask], dim=0)

        weighted = weights.unsqueeze(-1) * h_nodes
        n_graphs = int(batch.max().item()) + 1
        H = h_nodes.size(-1)
        out = torch.zeros(n_graphs, H, device=h_nodes.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
        return out


# ── Full whole-protein TCPNet ─────────────────────────────────────────────────

class WholeProteinTCPNet(nn.Module):
    """Hierarchical message passing over the whole protein.

    Information flows:
        atoms (Z1) → near residues (Z2) → distant residues (Z3) → back

    With 10 layers and 12 Å edges, effective receptive field ≈ 120 Å,
    covering the full protein.
    """

    def __init__(self, cfg: Optional[WholeProteinConfig] = None):
        super().__init__()
        self.cfg = cfg or WholeProteinConfig()
        H = self.cfg.hidden_dim

        # Zone-specific input projections
        self.zone1_embed = nn.Sequential(
            nn.Linear(self.cfg.zone1_feat_dim, H), nn.SiLU(), nn.Linear(H, H)
        )
        self.zone2_embed = nn.Sequential(
            nn.Linear(self.cfg.zone2_feat_dim, H), nn.SiLU(), nn.Linear(H, H)
        )
        self.zone3_embed = nn.Sequential(
            nn.Linear(self.cfg.zone3_feat_dim, H), nn.SiLU(), nn.Linear(H, H)
        )

        # Edge feature embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(self.cfg.edge_feat_dim, H), nn.SiLU(), nn.Linear(H, H)
        )

        # Deep message-passing stack
        self.message_layers = nn.ModuleList([
            ProteinMessagePassingLayer(
                H, self.cfg.n_rbf, self.cfg.distance_cutoff, self.cfg.dropout
            )
            for _ in range(self.cfg.n_message_layers)
        ])

        # Layer-norm after residual connections
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(H) for _ in range(self.cfg.n_message_layers)
        ])
        self.edge_norms = nn.ModuleList([
            nn.LayerNorm(H) for _ in range(self.cfg.n_message_layers)
        ])

        # Zone-aware readout
        self.zone_pool = ZoneAwareAttention(H)

        # Curriculum: how many layers to use
        self._active_layers = self.cfg.n_message_layers

    @property
    def active_layers(self) -> int:
        return self._active_layers

    def set_active_message_layers(self, n: int) -> None:
        """For curriculum learning: gradually increase depth."""
        self._active_layers = min(n, len(self.message_layers))

    def _embed_heterogeneous_nodes(
        self,
        node_features: torch.Tensor,
        zone_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Apply zone-specific embeddings to heterogeneous nodes."""
        H = self.cfg.hidden_dim
        h = torch.zeros(len(node_features), H, device=node_features.device)

        z1 = zone_assignments == 0
        z2 = zone_assignments == 1
        z3 = zone_assignments == 2

        if z1.any():
            h[z1] = self.zone1_embed(node_features[z1, : self.cfg.zone1_feat_dim])
        if z2.any():
            h[z2] = self.zone2_embed(node_features[z2, : self.cfg.zone2_feat_dim])
        if z3.any():
            h[z3] = self.zone3_embed(node_features[z3, : self.cfg.zone3_feat_dim])

        return h

    def forward(
        self,
        protein_graph: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        protein_graph : dict with keys
            node_features    : (N, max_feat_dim)
            node_coords      : (N, 3)
            edge_index       : (2, E)
            edge_features    : (E, edge_feat_dim)
            zone_assignments : (N,)  values in {0, 1, 2}
            batch            : (N,)  optional graph membership

        Returns
        -------
        enzyme_embedding   : (B, H)
        residue_embeddings : (N, H)
        """
        h_nodes = self._embed_heterogeneous_nodes(
            protein_graph["node_features"],
            protein_graph["zone_assignments"],
        )
        h_edges = self.edge_embed(protein_graph["edge_features"])
        coords = protein_graph["node_coords"]
        edge_index = protein_graph["edge_index"]
        batch = protein_graph.get("batch")

        for i in range(self._active_layers):
            h_nodes_new, h_edges_new = self.message_layers[i](
                h_nodes, h_edges, edge_index, coords
            )
            # Residual + layer norm
            h_nodes = self.node_norms[i](h_nodes + h_nodes_new)
            h_edges = self.edge_norms[i](h_edges + h_edges_new)

        enzyme_embedding = self.zone_pool(
            h_nodes, protein_graph["zone_assignments"], batch
        )

        return enzyme_embedding, h_nodes
