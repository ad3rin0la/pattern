"""
TCPNet — Topology-Complete Perceptron Network for Enzyme Active Sites.

Adapts Topotein's TCPNet architecture (arXiv 2509.03885) to operate on
Enzyme Combinatorial Complexes (Enzyme-PCC) built from active-site
extractions.  Messages pass simultaneously across three cell dimensions:

    0-cells (atoms)  ↔  1-cells (bonds)  ↔  2-cells (residue clusters)

SE(3) equivariance is enforced through spherical-harmonic edge features
and equivariant tensor products (via ``e3nn``).  The implementation is
self-contained so the model can be trained even when only ``torch`` and
``torch_geometric`` are installed — the ``e3nn`` path is used when
available, otherwise a scalar-only fallback is provided.

References
----------
Wang et al., Topotein, arXiv 2509.03885 (2025).
Geiger & Smidt, e3nn, arXiv 2207.09453 (2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional e3nn for full SE(3) equivariance.
try:
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet as e3FC
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False

# Optional torch_geometric message-passing primitives.
try:
    from torch_geometric.nn import MessagePassing as PyGMessagePassing
    from torch_geometric.utils import softmax as pyg_softmax
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TCPNetConfig:
    """Hyper-parameters for the TCPNet encoder."""

    node_feat_dim: int = 128       # Input per-atom feature dimension (Phase 2)
    edge_feat_dim: int = 32        # Input per-bond feature dimension
    hidden_dim: int = 256          # Internal representation width
    n_layers: int = 6              # Number of TCP message-passing layers
    max_degree: int = 3            # Max spherical-harmonic degree for SE(3)
    n_heads: int = 8               # Multi-head attention heads
    dropout: float = 0.1
    use_e3nn: bool = HAS_E3NN      # Fall back to scalar if e3nn absent
    residual: bool = True
    layer_norm: bool = True


# ── Radial basis functions ────────────────────────────────────────────────────

class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for distance encoding."""

    def __init__(self, n_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("offsets", offsets)
        self.width = (offsets[1] - offsets[0]).item() if n_rbf > 1 else 1.0

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E,) → (E, n_rbf)"""
        return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offsets) / self.width) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cutoff envelope for distance-dependent interactions."""

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * (dist <= self.cutoff).float()


# ── Scalar message-passing (fallback when e3nn absent) ────────────────────────

class ScalarEdgeMessage(nn.Module):
    """Distance-gated message from nodes to edges (scalar-only path)."""

    def __init__(self, hidden_dim: int, n_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index                              # (E,) each
        edge_vec = pos[row] - pos[col]                     # (E, 3)
        dist = edge_vec.norm(dim=-1)                       # (E,)
        rbf = self.rbf(dist)                               # (E, n_rbf)
        envelope = self.cutoff_fn(dist).unsqueeze(-1)      # (E, 1)
        x_j = h_nodes[col]                                 # (E, H)
        msg = self.mlp(torch.cat([x_j, rbf], dim=-1))     # (E, H)
        return msg * envelope                              # (E, H)


class ScalarNodeMessage(nn.Module):
    """Aggregate edge messages back to nodes (scalar-only path)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_edges: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        row, _ = edge_index
        # Scatter-add edge features to target nodes
        out = torch.zeros(n_nodes, h_edges.size(-1), device=h_edges.device)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(h_edges), self.mlp(h_edges))
        return out


# ── SE(3)-equivariant message passing (when e3nn is available) ────────────────

class EquivariantEdgeMessage(nn.Module):
    """SE(3)-equivariant messages using spherical harmonics + tensor products."""

    def __init__(self, hidden_dim: int, max_degree: int = 3, n_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        if not HAS_E3NN:
            raise ImportError("e3nn required for SE(3)-equivariant messages")

        self.hidden_dim = hidden_dim
        self.max_degree = max_degree
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Irreps for spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_degree)

        # Scalar input → mixed irreps via radial network
        self.irreps_in = o3.Irreps(f"{hidden_dim}x0e")
        # Output keeps scalar channel dominant for downstream heads
        n_vec = hidden_dim // 4
        n_l2 = hidden_dim // 8
        self.irreps_out = o3.Irreps(f"{hidden_dim}x0e + {n_vec}x1o + {n_l2}x2e")

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )
        self.weight_net = nn.Sequential(
            nn.Linear(n_rbf, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        dist = edge_vec.norm(dim=-1)
        direction = edge_vec / (dist.unsqueeze(-1) + 1e-8)

        rbf = self.rbf(dist)
        envelope = self.cutoff_fn(dist).unsqueeze(-1)
        sh = o3.spherical_harmonics(self.irreps_sh, direction, normalize=True)

        # Radial-dependent weights for the tensor product
        weights = self.weight_net(rbf)

        x_j = h_nodes[col]
        msg = self.tp(x_j, sh, weights) * envelope  # (E, irreps_out_dim)

        # Keep only scalar part for compatibility with downstream layers
        scalar_dim = self.hidden_dim
        return msg[:, :scalar_dim]


# ── Face (2-cell) message passing ─────────────────────────────────────────────

class FaceMessagePassing(nn.Module):
    """Messages from 2-cells (triangular residue clusters) to 1-cells (edges).

    Each 2-cell is a triangle formed by three atoms.  Its feature is the
    mean of the three edge features bounding it, transformed through an MLP.
    That transformed feature is then scattered back to each constituent edge.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_edges: torch.Tensor,
        face_to_edge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_edges : (E, H)
            Current edge features.
        face_to_edge : (3, F)
            Each column lists the three edge indices forming a face.

        Returns
        -------
        (E, H)  Messages aggregated from faces back onto edges.
        """
        if face_to_edge is None or face_to_edge.numel() == 0:
            return torch.zeros_like(h_edges)

        n_faces = face_to_edge.size(1)
        # Gather the three edge features per face → (F, 3, H)
        face_feats = h_edges[face_to_edge.T]  # (F, 3, H)
        # Pool within each face
        face_repr = face_feats.mean(dim=1)     # (F, H)
        face_msg = self.mlp(face_repr)         # (F, H)

        # Scatter face messages back to constituent edges
        out = torch.zeros_like(h_edges)
        for k in range(3):
            edge_ids = face_to_edge[k]          # (F,)
            out.scatter_add_(
                0,
                edge_ids.unsqueeze(-1).expand(-1, h_edges.size(-1)),
                face_msg,
            )
        return out


# ── Single TCPNet layer ──────────────────────────────────────────────────────

class TCPNetLayer(nn.Module):
    """One layer of topology-complete message passing.

    Simultaneously exchanges information between 0-cells (atoms),
    1-cells (bonds), and 2-cells (residue clusters).
    """

    def __init__(self, cfg: TCPNetConfig):
        super().__init__()
        H = cfg.hidden_dim

        # 0-cell → 1-cell (atom features aggregated onto edges)
        if cfg.use_e3nn and HAS_E3NN:
            self.node_to_edge = EquivariantEdgeMessage(H, cfg.max_degree)
        else:
            self.node_to_edge = ScalarEdgeMessage(H)

        # 1-cell → 0-cell
        self.edge_to_node = ScalarNodeMessage(H)

        # 2-cell → 1-cell
        self.face_to_edge = FaceMessagePassing(H)

        # Self-interaction MLPs
        self.node_self = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_self = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))

        self.residual = cfg.residual
        self.ln_node = nn.LayerNorm(H) if cfg.layer_norm else nn.Identity()
        self.ln_edge = nn.LayerNorm(H) if cfg.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        face_to_edge: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Messages
        msg_n2e = self.node_to_edge(h_nodes, edge_index, pos)
        msg_e2n = self.edge_to_node(h_edges, edge_index, h_nodes.size(0))
        msg_f2e = self.face_to_edge(h_edges, face_to_edge)

        # Update nodes
        h_nodes_new = msg_e2n + self.node_self(h_nodes)
        if self.residual:
            h_nodes_new = h_nodes + self.dropout(h_nodes_new)
        h_nodes_new = self.ln_node(h_nodes_new)

        # Update edges
        h_edges_new = msg_n2e + msg_f2e + self.edge_self(h_edges)
        if self.residual:
            h_edges_new = h_edges + self.dropout(h_edges_new)
        h_edges_new = self.ln_edge(h_edges_new)

        return h_nodes_new, h_edges_new


# ── Global attention pooling ──────────────────────────────────────────────────

class GlobalAttentionPooling(nn.Module):
    """Attention-weighted global pooling for graph-level readout."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(
        self,
        h_nodes: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_nodes : (N_total, H)
        batch   : (N_total,)  graph membership indices.  If None, single graph.

        Returns
        -------
        (B, H)
        """
        gate_logits = self.gate(h_nodes).squeeze(-1)  # (N,)

        if batch is None:
            weights = torch.softmax(gate_logits, dim=0)
            return (h_nodes * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        # Per-graph softmax
        weights = torch.zeros_like(gate_logits)
        for gid in batch.unique():
            mask = batch == gid
            weights[mask] = torch.softmax(gate_logits[mask], dim=0)

        weighted = h_nodes * weights.unsqueeze(-1)

        n_graphs = int(batch.max().item()) + 1
        H = h_nodes.size(-1)
        out = torch.zeros(n_graphs, H, device=h_nodes.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted), weighted)
        return out


# ── Full EnzymeTCPNet encoder ─────────────────────────────────────────────────

class EnzymeTCPNet(nn.Module):
    """Topology-Complete Perceptron Network for enzyme active sites.

    Takes an Enzyme-PCC (atoms, bonds, faces, coordinates, Phase 2 spectral
    features) and returns a fixed-size enzyme-level embedding suitable for
    the downstream multi-task heads.
    """

    def __init__(self, cfg: Optional[TCPNetConfig] = None):
        super().__init__()
        self.cfg = cfg or TCPNetConfig()
        H = self.cfg.hidden_dim

        # Initial projections from Phase 2 feature dimensions
        self.node_embed = nn.Sequential(
            nn.Linear(self.cfg.node_feat_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(self.cfg.edge_feat_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # TCPNet message-passing stack
        self.layers = nn.ModuleList([
            TCPNetLayer(self.cfg) for _ in range(self.cfg.n_layers)
        ])

        # Readout
        self.pool = GlobalAttentionPooling(H)

    def forward(self, enzyme_pcc: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        enzyme_pcc : dict with keys
            node_features : (N, node_feat_dim)
            edge_index    : (2, E)
            edge_features : (E, edge_feat_dim)
            pos           : (N, 3)
            face_to_edge  : (3, F) or None
            batch         : (N,)   graph membership (optional, for batching)

        Returns
        -------
        enzyme_embedding : (B, hidden_dim)
            Graph-level active-site representation.
        h_nodes : (N, hidden_dim)
            Per-atom representations (for cross-attention & attribution).
        """
        h_nodes = self.node_embed(enzyme_pcc["node_features"])
        h_edges = self.edge_embed(enzyme_pcc["edge_features"])
        edge_index = enzyme_pcc["edge_index"]
        pos = enzyme_pcc["pos"]
        face_to_edge = enzyme_pcc.get("face_to_edge")
        batch = enzyme_pcc.get("batch")

        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, edge_index, pos, face_to_edge)

        enzyme_embedding = self.pool(h_nodes, batch)
        return enzyme_embedding, h_nodes
