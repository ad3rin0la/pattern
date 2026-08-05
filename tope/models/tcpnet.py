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

from tope.models.cc_attention import CCAttentionBlock, AttentionMergeNode

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
    face_feat_dim: int = 0         # Input per-residue feature dim; 0 = synthesize from edges
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


# ── Incidence matrix builders ─────────────────────────────────────────────────

def _build_B_01(edge_index: torch.Tensor) -> torch.Tensor:
    """Build B_01 incidence matrix (atoms→bonds) from edge_index.

    Each bond e has two incident atoms: src[e] and dst[e].
    Returns (2, 2E) where B_01[0] = atom indices, B_01[1] = bond indices.
    """
    E = edge_index.size(1)
    src, dst = edge_index
    bond_idx = torch.arange(E, device=edge_index.device)
    return torch.stack(
        [torch.cat([src, dst]), torch.cat([bond_idx, bond_idx])],
        dim=0,
    )


def _build_B_12(face_to_edge: torch.Tensor) -> torch.Tensor:
    """Build B_12 incidence matrix (bonds→residues) from face_to_edge.

    face_to_edge: (3, F) — each column holds 3 edge indices of a face.
    Returns (2, 3F) where B_12[0] = bond indices, B_12[1] = residue indices.
    """
    F_count = face_to_edge.size(1)
    face_idx = torch.arange(F_count, device=face_to_edge.device)
    bond_idx = face_to_edge.reshape(-1)              # (3F,)
    res_idx = face_idx.unsqueeze(0).expand(3, -1).reshape(-1)  # (3F,)
    return torch.stack([bond_idx, res_idx], dim=0)


def _build_B_12_from_atom_residue(
    edge_index: torch.Tensor,
    atom_residue: torch.Tensor,
) -> torch.Tensor:
    """Build B_12 (bonds→residues) from the curated atom→residue incidence.

    This is the data-driven boundary operator: rather than re-deriving the
    2-cells from a synthetic face triangulation (``_build_B_12``), each bond
    inherits its incident residues directly from the residues its two endpoint
    atoms belong to (``ActiveSite.atom_residue_incidence()``).  A bond ``e =
    (a_i, a_j)`` is incident to ``residue(a_i)`` and ``residue(a_j)``; when both
    endpoints share a residue (an intra-residue bond) the single membership is
    emitted once.

    Parameters
    ----------
    edge_index   : (2, E)  atom adjacency (the rank-1 bonds).
    atom_residue : (N,)    residue index per atom (-1 if the atom's parent
                           residue is absent — such endpoints are skipped).

    Returns
    -------
    B_12 : (2, E_12)  B_12[0] = bond indices, B_12[1] = residue indices, with
           duplicate (bond, residue) pairs removed.
    """
    E = edge_index.size(1)
    if E == 0:
        return edge_index.new_zeros(2, 0)
    src, dst = edge_index
    bond_idx = torch.arange(E, device=edge_index.device)
    res_src = atom_residue[src]
    res_dst = atom_residue[dst]

    bonds = torch.cat([bond_idx, bond_idx])
    residues = torch.cat([res_src, res_dst])
    # Drop endpoints whose parent residue is absent.
    valid = residues >= 0
    bonds, residues = bonds[valid], residues[valid]
    if bonds.numel() == 0:
        return edge_index.new_zeros(2, 0)

    # Deduplicate (bond, residue) pairs (intra-residue bonds map both endpoints
    # to the same residue).
    pairs = torch.stack([bonds, residues], dim=0)
    pairs = torch.unique(pairs, dim=1)
    return pairs


# ── Single TCPNet layer ──────────────────────────────────────────────────────

class TCPNetLayer(nn.Module):
    """One layer of topology-complete message passing using CC-attention.

    Simultaneously exchanges information between 0-cells (atoms),
    1-cells (bonds), and 2-cells (residue clusters) via learned,
    per-neighborhood normalized attention (CCANN formalism).

    Replaces isotropic scatter_add aggregation with:
      - attn_01 : CCAttentionBlock for atoms ↔ bonds (B_01 incidence)
      - attn_12 : CCAttentionBlock for bonds ↔ residues (B_12 incidence)
      - edge_merge : AttentionMergeNode over two bond neighborhoods
    """

    def __init__(self, cfg: TCPNetConfig):
        super().__init__()
        H = cfg.hidden_dim

        # CC-attention blocks (replace ScalarEdgeMessage, ScalarNodeMessage,
        # FaceMessagePassing, and EquivariantEdgeMessage)
        self.attn_01 = CCAttentionBlock(d_s_in=H, d_t_in=H, d_out=H)  # atoms ↔ bonds
        self.attn_12 = CCAttentionBlock(d_s_in=H, d_t_in=H, d_out=H)  # bonds ↔ residues

        # Merge bond messages from two neighborhoods: atoms (B_01) + residues (B_12)
        self.edge_merge = AttentionMergeNode(n_neighborhoods=2, hidden_dim=H)

        # Self-interaction MLPs
        self.node_self = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_self = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))
        self.face_self = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))

        self.residual = cfg.residual
        self.ln_node = nn.LayerNorm(H) if cfg.layer_norm else nn.Identity()
        self.ln_edge = nn.LayerNorm(H) if cfg.layer_norm else nn.Identity()
        self.ln_face = nn.LayerNorm(H) if cfg.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h_nodes: torch.Tensor,   # (N_atoms, H)
        h_edges: torch.Tensor,   # (N_bonds, H)
        h_faces: torch.Tensor,   # (N_residues, H)
        B_01: torch.Tensor,      # (2, E_01) — B[0]=atom, B[1]=bond
        B_12: torch.Tensor,      # (2, E_12) — B[0]=bond, B[1]=residue
        pos: torch.Tensor,       # (N_atoms, 3)  kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_nodes_new : (N_atoms, H)
        h_edges_new : (N_bonds, H)
        h_faces_new : (N_residues, H)
        """
        # attn_01: H_s=atoms, H_t=bonds, B=B_01
        #   K_t = bonds attended by atoms  = msg_n2e
        #   K_s = atoms attended by bonds  = msg_e2n
        msg_n2e, msg_e2n = self.attn_01(h_nodes, h_edges, B_01)

        # attn_12: H_s=bonds, H_t=residues, B=B_12
        #   K_t = residues attended by bonds = face update
        #   K_s = bonds attended by residues  = msg_f2e
        K_faces_update, msg_f2e = self.attn_12(h_edges, h_faces, B_12)

        # ── Node update: one neighborhood (bonds via B_01 reverse) ───────────
        h_nodes_new = msg_e2n + self.node_self(h_nodes)
        if self.residual:
            h_nodes_new = h_nodes + self.dropout(h_nodes_new)
        h_nodes_new = self.ln_node(h_nodes_new)

        # ── Edge update: two neighborhoods (atoms + residues) ─────────────────
        h_edges_new = self.edge_merge([msg_n2e, msg_f2e]) + self.edge_self(h_edges)
        if self.residual:
            h_edges_new = h_edges + self.dropout(h_edges_new)
        h_edges_new = self.ln_edge(h_edges_new)

        # ── Face update: one neighborhood (bonds via B_12 forward) ────────────
        h_faces_new = K_faces_update + self.face_self(h_faces)
        if self.residual:
            h_faces_new = h_faces + self.dropout(h_faces_new)
        h_faces_new = self.ln_face(h_faces_new)

        return h_nodes_new, h_edges_new, h_faces_new


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
        active_site_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_nodes          : (N_total, H)
        batch            : (N_total,)  graph membership indices.  If None, single graph.
        active_site_mask : (N_total,) bool or float — optional per-atom bias that
                           up-weights active-site atoms without excluding allosteric
                           ones.  True / 1.0 atoms receive a +1 logit bonus.

        Returns
        -------
        (B, H)
        """
        gate_logits = self.gate(h_nodes).squeeze(-1)  # (N,)

        if active_site_mask is not None:
            # Additive bias: up-weights catalytic atoms while keeping allosteric
            # atoms in the pool (they contribute with lower but non-zero weight)
            bias = active_site_mask.float()
            gate_logits = gate_logits + bias

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
        # Face (2-cell / residue) embedding.
        # If face_feat_dim > 0, embed from explicit face features;
        # otherwise project from H (synthesized as mean of incident edge features).
        face_in = self.cfg.face_feat_dim if self.cfg.face_feat_dim > 0 else H
        self.face_embed = nn.Linear(face_in, H)

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
            edge_index    : (2, E)        — kept for backward compat (= B_01 atom adjacency)
            edge_features : (E, edge_feat_dim)
            pos           : (N, 3)
            B_01          : (2, E_01)     — atoms→bonds incidence (optional; derived if absent)
            B_12          : (2, E_12)     — bonds→residues incidence (optional; derived if absent)
            atom_residue  : (N,) or None  — curated atom→residue membership
                            (ActiveSite.atom_residue_incidence()).  When present
                            it drives B_12 directly (preferred over face_to_edge),
                            so the curated complex defines the rank-2 boundary
                            operator instead of the model re-deriving it.
            face_features : (F, face_feat_dim) or None
            face_to_edge  : (3, F) or None   — synthetic B_12 / h_faces fallback
            batch         : (N,)              — graph membership (optional, for batching)

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

        # ── Build / retrieve B_01 ─────────────────────────────────────────────
        if "B_01" in enzyme_pcc:
            B_01 = enzyme_pcc["B_01"]
        else:
            B_01 = _build_B_01(edge_index)

        # ── Build / retrieve B_12 ─────────────────────────────────────────────
        # Preference order: explicit B_12 > curated atom→residue incidence >
        # synthetic face triangulation.  The atom_residue path lets the curated
        # complex drive the rank-2 boundary operator directly, instead of the
        # model re-deriving residues from a face_to_edge proxy.
        atom_residue = enzyme_pcc.get("atom_residue")
        if "B_12" in enzyme_pcc:
            B_12 = enzyme_pcc["B_12"]
        elif atom_residue is not None and atom_residue.numel() > 0:
            B_12 = _build_B_12_from_atom_residue(edge_index, atom_residue)
        elif face_to_edge is not None and face_to_edge.numel() > 0:
            B_12 = _build_B_12(face_to_edge)
        else:
            # No 2-cells: empty incidence matrix (1 dummy residue)
            B_12 = edge_index.new_zeros(2, 0)

        # ── Initialise h_faces ────────────────────────────────────────────────
        if "face_features" in enzyme_pcc and enzyme_pcc["face_features"] is not None:
            h_faces = self.face_embed(enzyme_pcc["face_features"])
        elif B_12.size(1) > 0:
            # Synthesise: mean of incident edge features per residue
            n_faces = int(B_12[1].max().item()) + 1
            bond_idx = B_12[0]  # (E_12,)
            res_idx = B_12[1]   # (E_12,)
            H = h_edges.size(-1)
            face_feat = torch.zeros(n_faces, H, device=h_edges.device, dtype=h_edges.dtype)
            count = torch.zeros(n_faces, 1, device=h_edges.device, dtype=h_edges.dtype)
            face_feat.scatter_add_(0, res_idx.unsqueeze(-1).expand(-1, H), h_edges[bond_idx])
            count.scatter_add_(0, res_idx.unsqueeze(-1), torch.ones(B_12.size(1), 1, device=h_edges.device))
            face_feat = face_feat / count.clamp(min=1)
            h_faces = self.face_embed(face_feat)
        else:
            # No faces at all — single dummy residue
            h_faces = self.face_embed(
                torch.zeros(1, self.face_embed.in_features, device=h_nodes.device)
            )

        # Optional active-site mask (whole-protein migration, Phase 2)
        # Passed as a per-atom boolean/float tensor from WholeProteinPCC.is_active_site
        active_site_mask = enzyme_pcc.get("is_active_site")

        for layer in self.layers:
            h_nodes, h_edges, h_faces = layer(h_nodes, h_edges, h_faces, B_01, B_12, pos)

        enzyme_embedding = self.pool(h_nodes, batch, active_site_mask=active_site_mask)
        return enzyme_embedding, h_nodes
