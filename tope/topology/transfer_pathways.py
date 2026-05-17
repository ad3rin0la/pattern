"""
Transfer Pathway Prediction for ToPE
=====================================

Predicts electron and proton transfer pathways from the topological encoding,
integrating vibrational topology from Chalopin's phonon-assisted transfer theory.

Chalopin identifies two types of charge transfer pathways in [FeFe] hydrogenase:
    - Electron transfer: Through FeS clusters (FS4A→FS4B→FS4C or via FS2)
    - Proton transfer: Through H-bond network (R286→E282→S319→E279→C299)

Both pathways correspond to chains of residues at thermal hotspots connected
by contacts at ~3.85 Å spacing. This is a natural prediction target for ToPE.

The transfer pathway head predicts:
    1. Per-edge pathway classification (electron/proton/substrate)
    2. Per-residue pathway membership probability
    3. Full pathway enumeration via graph traversal

Training data sources:
    - M-CSA catalytic residue annotations
    - Literature-curated transfer pathways (e.g., PETN databases)
    - Computed tunneling pathways from QM/MM studies

Validation via Chalopin's criterion: predicted pathway residues should
coincide with thermal hotspot positions in the localization landscape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_mean, scatter_add
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

class PathwayType(IntEnum):
    """Types of transfer pathways in enzymes."""
    NONE = 0
    ELECTRON_TRANSFER = 1
    PROTON_TRANSFER = 2
    SUBSTRATE_CHANNEL = 3
    WATER_CHANNEL = 4
    COFACTOR_BRIDGE = 5


@dataclass
class TransferPathwayConfig:
    """Configuration for transfer pathway prediction."""

    hidden_dim: int = 256
    n_pathway_types: int = 6  # Including NONE

    # Edge classification
    edge_mlp_layers: int = 2
    edge_dropout: float = 0.1

    # Node classification
    node_mlp_layers: int = 2
    node_dropout: float = 0.1

    # Pathway enumeration
    max_pathway_length: int = 20
    min_pathway_confidence: float = 0.5

    # Localization landscape features
    use_localization_landscape: bool = True
    use_stiffness_gradient: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Transfer Pathway Prediction Head
# ══════════════════════════════════════════════════════════════════════════════

class TransferPathwayHead(nn.Module):
    """
    Predict electron and proton transfer pathways from the topological encoding.

    For each pair of residues, predict the probability that they
    participate in a functional transfer pathway.

    The head uses:
        1. Node embeddings from TCPNet
        2. Localization landscape values (u_h)
        3. Pairwise distances
        4. Stiffness gradient (|u_h_i - u_h_j|)

    Training data sources:
        - M-CSA catalytic residue annotations
        - Literature-curated transfer pathways (e.g., PETN databases)
        - Computed tunneling pathways from QM/MM studies

    Validation via Chalopin's criterion: predicted pathway residues
    should coincide with thermal hotspot positions in the
    localization landscape.
    """

    def __init__(self, config: Optional[TransferPathwayConfig] = None):
        super().__init__()
        self.config = config or TransferPathwayConfig()

        hidden_dim = self.config.hidden_dim
        n_types = self.config.n_pathway_types

        # Input dimension: 2*hidden_dim (src+dst) + topological features
        topo_features = 4 if self.config.use_localization_landscape else 1
        input_dim = hidden_dim * 2 + topo_features

        # Edge classifier: does edge (i,j) belong to pathway type t?
        edge_layers = []
        for i in range(self.config.edge_mlp_layers):
            in_features = input_dim if i == 0 else hidden_dim
            edge_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.edge_dropout),
            ])
        edge_layers.append(nn.Linear(hidden_dim, n_types))
        self.edge_classifier = nn.Sequential(*edge_layers)

        # Node classifier: is residue i part of pathway type t?
        node_layers = []
        node_input_dim = hidden_dim + 2  # + u_h + vibrational_bin
        for i in range(self.config.node_mlp_layers):
            in_features = node_input_dim if i == 0 else hidden_dim
            node_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.node_dropout),
            ])
        node_layers.append(nn.Linear(hidden_dim, n_types))
        self.node_classifier = nn.Sequential(*node_layers)

        # Pathway confidence head
        self.pathway_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        u_h: Optional[torch.Tensor] = None,
        vibrational_bins: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict transfer pathways.

        Args:
            node_embeddings: (N, hidden_dim) from TCPNet
            edge_index: (2, E) connectivity
            distances: (E,) pairwise distances
            u_h: (N,) localization landscape values
            vibrational_bins: (N,) vibrational filtration bin assignments

        Returns:
            Dict with:
                - edge_logits: (E, n_pathway_types) per-edge classification
                - node_logits: (N, n_pathway_types) per-node classification
                - edge_probs: (E, n_pathway_types) softmax probabilities
                - node_probs: (N, n_pathway_types) softmax probabilities
        """
        src, dst = edge_index
        N = node_embeddings.size(0)
        E = edge_index.size(1)
        device = node_embeddings.device

        # Default u_h if not provided
        if u_h is None:
            u_h = torch.ones(N, device=device)

        # Default vibrational bins
        if vibrational_bins is None:
            vibrational_bins = torch.zeros(N, device=device)

        # === Edge classification ===
        # Concatenate node features with topological descriptors
        edge_features = [
            node_embeddings[src],
            node_embeddings[dst],
            distances.unsqueeze(-1),
        ]

        if self.config.use_localization_landscape:
            edge_features.extend([
                u_h[src].unsqueeze(-1),
                u_h[dst].unsqueeze(-1),
            ])

        if self.config.use_stiffness_gradient:
            stiffness_grad = (u_h[src] - u_h[dst]).abs().unsqueeze(-1)
            edge_features.append(stiffness_grad)

        edge_features = torch.cat(edge_features, dim=-1)
        edge_logits = self.edge_classifier(edge_features)

        # === Node classification ===
        node_features = torch.cat([
            node_embeddings,
            u_h.unsqueeze(-1),
            vibrational_bins.unsqueeze(-1).float(),
        ], dim=-1)
        node_logits = self.node_classifier(node_features)

        # Probabilities
        edge_probs = F.softmax(edge_logits, dim=-1)
        node_probs = F.softmax(node_logits, dim=-1)

        return {
            'edge_logits': edge_logits,
            'node_logits': node_logits,
            'edge_probs': edge_probs,
            'node_probs': node_probs,
        }

    def predict_pathways(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        u_h: Optional[torch.Tensor] = None,
        pathway_type: PathwayType = PathwayType.ELECTRON_TRANSFER,
    ) -> List[List[int]]:
        """
        Enumerate complete pathways of a given type.

        Uses graph traversal on high-confidence edges to find
        connected paths through the enzyme.

        Args:
            node_embeddings: (N, hidden_dim) from TCPNet
            edge_index: (2, E) connectivity
            distances: (E,) pairwise distances
            u_h: (N,) localization landscape values
            pathway_type: Type of pathway to enumerate

        Returns:
            List of pathways, each a list of residue indices
        """
        with torch.no_grad():
            outputs = self.forward(node_embeddings, edge_index, distances, u_h)
            edge_probs = outputs['edge_probs']

        # Get edges with high probability for this pathway type
        type_idx = int(pathway_type)
        type_probs = edge_probs[:, type_idx].cpu().numpy()
        edge_index_np = edge_index.cpu().numpy()

        # Build adjacency list for high-confidence edges
        min_conf = self.config.min_pathway_confidence
        adj = {}
        for i, (src, dst) in enumerate(edge_index_np.T):
            if type_probs[i] >= min_conf:
                adj.setdefault(src, []).append(dst)
                adj.setdefault(dst, []).append(src)

        # Find connected components (pathways)
        visited = set()
        pathways = []

        def dfs(node: int, path: List[int]) -> None:
            if len(path) >= self.config.max_pathway_length:
                return
            visited.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)

        for node in adj:
            if node not in visited:
                path = []
                dfs(node, path)
                if len(path) > 1:  # Pathway must have at least 2 residues
                    pathways.append(path)

        return pathways


# ══════════════════════════════════════════════════════════════════════════════
# Transfer Pathway Loss
# ══════════════════════════════════════════════════════════════════════════════

class TransferPathwayLoss(nn.Module):
    """
    Loss function for transfer pathway prediction.

    Combines:
        1. Edge classification cross-entropy
        2. Node classification cross-entropy
        3. Pathway consistency regularization
        4. Localization landscape correlation
    """

    def __init__(
        self,
        edge_weight: float = 1.0,
        node_weight: float = 1.0,
        consistency_weight: float = 0.1,
        landscape_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.consistency_weight = consistency_weight
        self.landscape_weight = landscape_weight

        self.edge_ce = nn.CrossEntropyLoss(weight=class_weights)
        self.node_ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        edge_labels: torch.Tensor,
        node_labels: torch.Tensor,
        edge_index: torch.Tensor,
        u_h: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pathway prediction loss.

        Args:
            predictions: Output from TransferPathwayHead
            edge_labels: (E,) ground truth edge pathway types
            node_labels: (N,) ground truth node pathway types
            edge_index: (2, E) connectivity
            u_h: (N,) localization landscape for correlation loss

        Returns:
            Dict with total loss and individual components
        """
        edge_logits = predictions['edge_logits']
        node_logits = predictions['node_logits']
        edge_probs = predictions['edge_probs']
        node_probs = predictions['node_probs']

        # Edge classification loss
        edge_loss = self.edge_ce(edge_logits, edge_labels)

        # Node classification loss
        node_loss = self.node_ce(node_logits, node_labels)

        # Consistency: if edge (i,j) is in pathway, both nodes should be too
        src, dst = edge_index
        edge_pathway_probs = 1.0 - edge_probs[:, 0]  # P(edge in any pathway)
        node_pathway_probs = 1.0 - node_probs[:, 0]  # P(node in any pathway)

        # Both endpoints should have high pathway probability
        src_probs = node_pathway_probs[src]
        dst_probs = node_pathway_probs[dst]
        min_node_probs = torch.min(src_probs, dst_probs)

        # If edge has high prob, nodes should too
        consistency_loss = F.mse_loss(
            edge_pathway_probs,
            torch.max(edge_pathway_probs, min_node_probs)
        )

        # Landscape correlation: pathway nodes should have high u_h
        landscape_loss = torch.tensor(0.0, device=edge_logits.device)
        if u_h is not None and self.landscape_weight > 0:
            # Pathway probability should correlate with u_h
            u_h_normalized = (u_h - u_h.mean()) / (u_h.std() + 1e-8)
            pathway_probs_normalized = (node_pathway_probs - node_pathway_probs.mean()) / \
                                       (node_pathway_probs.std() + 1e-8)
            correlation = (u_h_normalized * pathway_probs_normalized).mean()
            # Loss: encourage positive correlation
            landscape_loss = 1.0 - correlation

        # Total loss
        total_loss = (
            self.edge_weight * edge_loss +
            self.node_weight * node_loss +
            self.consistency_weight * consistency_loss +
            self.landscape_weight * landscape_loss
        )

        return {
            'total': total_loss,
            'edge': edge_loss,
            'node': node_loss,
            'consistency': consistency_loss,
            'landscape': landscape_loss,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Pathway Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_pathway_metrics(
    predictions: Dict[str, torch.Tensor],
    edge_labels: torch.Tensor,
    node_labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute pathway prediction metrics.

    Returns:
        Dict with accuracy, precision, recall, F1 for edges and nodes
    """
    edge_probs = predictions['edge_probs']
    node_probs = predictions['node_probs']

    edge_preds = edge_probs.argmax(dim=-1)
    node_preds = node_probs.argmax(dim=-1)

    # Edge metrics
    edge_correct = (edge_preds == edge_labels).float().mean().item()

    # Per-class edge metrics (excluding NONE class)
    edge_metrics = {}
    for pathway_type in PathwayType:
        if pathway_type == PathwayType.NONE:
            continue

        type_idx = int(pathway_type)
        type_mask = edge_labels == type_idx
        type_pred_mask = edge_preds == type_idx

        if type_mask.sum() > 0:
            tp = (type_mask & type_pred_mask).sum().float()
            fp = (~type_mask & type_pred_mask).sum().float()
            fn = (type_mask & ~type_pred_mask).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            edge_metrics[f'edge_{pathway_type.name.lower()}_precision'] = precision.item()
            edge_metrics[f'edge_{pathway_type.name.lower()}_recall'] = recall.item()
            edge_metrics[f'edge_{pathway_type.name.lower()}_f1'] = f1.item()

    # Node metrics
    node_correct = (node_preds == node_labels).float().mean().item()

    # Pathway presence detection (any pathway vs none)
    edge_has_pathway = (edge_labels != PathwayType.NONE)
    edge_pred_pathway = (edge_preds != PathwayType.NONE)
    pathway_detection_acc = (edge_has_pathway == edge_pred_pathway).float().mean().item()

    return {
        'edge_accuracy': edge_correct,
        'node_accuracy': node_correct,
        'pathway_detection_accuracy': pathway_detection_acc,
        **edge_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pathway Visualization
# ══════════════════════════════════════════════════════════════════════════════

def visualize_pathways(
    predictions: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    pathway_type: PathwayType = PathwayType.ELECTRON_TRANSFER,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Prepare pathway visualization data.

    Returns coordinates and colors for pathway edges and nodes.
    """
    edge_probs = predictions['edge_probs']
    node_probs = predictions['node_probs']

    type_idx = int(pathway_type)

    # Get high-confidence pathway edges
    edge_confidence = edge_probs[:, type_idx].cpu().numpy()
    pathway_edges = edge_confidence >= min_confidence

    # Get pathway nodes
    node_confidence = node_probs[:, type_idx].cpu().numpy()
    pathway_nodes = node_confidence >= min_confidence

    edge_index_np = edge_index.cpu().numpy()
    coords_np = coords.cpu().numpy()

    # Extract pathway edge coordinates
    pathway_edge_coords = []
    for i, is_pathway in enumerate(pathway_edges):
        if is_pathway:
            src, dst = edge_index_np[:, i]
            pathway_edge_coords.append({
                'start': coords_np[src].tolist(),
                'end': coords_np[dst].tolist(),
                'confidence': float(edge_confidence[i]),
            })

    # Extract pathway node coordinates
    pathway_node_coords = []
    for i, is_pathway in enumerate(pathway_nodes):
        if is_pathway:
            pathway_node_coords.append({
                'position': coords_np[i].tolist(),
                'confidence': float(node_confidence[i]),
            })

    return {
        'pathway_type': pathway_type.name,
        'edges': pathway_edge_coords,
        'nodes': pathway_node_coords,
        'n_pathway_edges': len(pathway_edge_coords),
        'n_pathway_nodes': len(pathway_node_coords),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Integration with ToPE Model
# ══════════════════════════════════════════════════════════════════════════════

class TransferPathwayModule(nn.Module):
    """
    Complete transfer pathway prediction module for ToPE.

    Integrates:
        - TransferPathwayHead for prediction
        - Localization landscape processing
        - Multi-type pathway enumeration
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        config: Optional[TransferPathwayConfig] = None,
    ):
        super().__init__()
        config = config or TransferPathwayConfig(hidden_dim=hidden_dim)
        self.config = config

        self.pathway_head = TransferPathwayHead(config)

        # Localization landscape projection
        self.u_h_projection = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        u_h: Optional[torch.Tensor] = None,
        vibrational_bins: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional landscape enhancement."""
        # Optionally enhance u_h through projection
        if u_h is not None:
            u_h_enhanced = self.u_h_projection(u_h.unsqueeze(-1)).squeeze(-1)
            u_h = u_h + u_h_enhanced

        return self.pathway_head(
            node_embeddings, edge_index, distances, u_h, vibrational_bins
        )

    def predict_all_pathways(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        u_h: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[List[int]]]:
        """
        Predict pathways for all types.

        Returns dict mapping pathway type name to list of pathways.
        """
        pathways = {}
        for pathway_type in PathwayType:
            if pathway_type == PathwayType.NONE:
                continue

            type_pathways = self.pathway_head.predict_pathways(
                node_embeddings, edge_index, distances, u_h, pathway_type
            )
            pathways[pathway_type.name.lower()] = type_pathways

        return pathways


# ── Compatibility shims ──────────────────────────────────────────────────────

class TransferPathwayVisualizer:
    """Thin wrapper around :func:`visualize_pathways` exposing a class API."""

    def __init__(self, **defaults):
        self.defaults = defaults

    def __call__(self, *args, **kwargs):
        merged = {**self.defaults, **kwargs}
        return visualize_pathways(*args, **merged)


def extract_transfer_pathway_graph(
    node_embeddings,
    edge_index,
    distances,
    u_h,
    pathway_type: "PathwayType",
    config: "TransferPathwayConfig | None" = None,
):
    """Run the pathway head and return predicted pathways for a given type.

    Convenience entry point that constructs a :class:`TransferPathwayHead`
    on demand and dispatches to :meth:`predict_pathways`.
    """
    head = TransferPathwayHead(config=config or TransferPathwayConfig())
    return head.predict_pathways(
        node_embeddings, edge_index, distances, u_h, pathway_type
    )
