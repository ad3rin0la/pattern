"""
ToPE Model — Full Topological Pattern Recognition Model
=========================================================

Two model variants:

**ToPEModel** (original):
    Active-site encoder (EnzymeTCPNet) + substrate/product cross-attention
    + multi-task heads.  Operates on 8 Å active-site Enzyme-PCC.

**CompleteToPEModel** (updated):
    Whole-protein encoder (WholeProteinTCPNet) + enhanced task heads
    (including residue-level kinetics attention and distant mutation
    effect prediction).  Operates on the full multi-scale protein graph.

Usage
-----
    from tope_model import ToPEModel, ToPEConfig
    from tope_model.tope_model import CompleteToPEModel, CompleteToPEConfig

    # Active-site only
    model = ToPEModel(ToPEConfig())

    # Whole-protein
    model = CompleteToPEModel(CompleteToPEConfig())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tope_model.tcpnet import EnzymeTCPNet, TCPNetConfig, GlobalAttentionPooling
from tope_model.cross_attention import (
    SubstrateProductCrossAttention,
    CrossAttentionConfig,
)
from tope_model.task_heads import (
    MultiTaskHeads,
    TaskHeadsConfig,
    WholeProteinTaskHeads,
    DistantMutationEffectPredictor,
)
from tope_model.whole_protein_tcpnet import WholeProteinTCPNet, WholeProteinConfig


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ToPEConfig:
    """Master configuration for the ToPE model.

    Aggregates sub-configs for each component and provides
    convenience defaults.
    """

    # TCPNet encoder
    node_feat_dim: int = 128
    edge_feat_dim: int = 32
    hidden_dim: int = 256
    n_tcpnet_layers: int = 6
    max_degree: int = 3
    n_heads: int = 8
    dropout: float = 0.1
    use_e3nn: bool = True

    # Cross-attention
    mol_feat_dim: int = 64
    mol_hidden_dim: int = 128
    mol_n_layers: int = 3

    # Task heads
    ec_levels: List[int] = field(default_factory=lambda: [7, 70, 250, 800])
    selectivity_dim: int = 1
    kinetics_targets: int = 3
    head_hidden_dim: int = 512

    # Whether to include cross-attention (requires substrate/product data)
    use_cross_attention: bool = True

    def to_tcpnet_config(self) -> TCPNetConfig:
        return TCPNetConfig(
            node_feat_dim=self.node_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_tcpnet_layers,
            max_degree=self.max_degree,
            n_heads=self.n_heads,
            dropout=self.dropout,
            use_e3nn=self.use_e3nn,
        )

    def to_cross_attention_config(self) -> CrossAttentionConfig:
        return CrossAttentionConfig(
            mol_feat_dim=self.mol_feat_dim,
            mol_hidden_dim=self.mol_hidden_dim,
            mol_n_layers=self.mol_n_layers,
            enzyme_hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )

    def to_task_heads_config(self) -> TaskHeadsConfig:
        return TaskHeadsConfig(
            hidden_dim=self.hidden_dim,
            ec_levels=self.ec_levels,
            selectivity_dim=self.selectivity_dim,
            kinetics_targets=self.kinetics_targets,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
        )


# ── Full ToPE Model ──────────────────────────────────────────────────────────

class ToPEModel(nn.Module):
    """End-to-end Topological Pattern Recognition model for enzyme catalysis.

    Architecture:
        Enzyme-PCC → EnzymeTCPNet → per-atom embeddings
        ↓                          ↓ (pool)
        ↓                     enzyme embedding
        ↓                          ↓
        Substrate/Product → CrossAttention → fused embeddings
                                   ↓
                            MultiTaskHeads → EC / selectivity / kinetics
    """

    def __init__(self, cfg: Optional[ToPEConfig] = None):
        super().__init__()
        self.cfg = cfg or ToPEConfig()

        # 1. Enzyme encoder (TCPNet over Enzyme-PCC)
        self.encoder = EnzymeTCPNet(self.cfg.to_tcpnet_config())

        # 2. Substrate–product cross-attention (optional)
        if self.cfg.use_cross_attention:
            self.cross_attention = SubstrateProductCrossAttention(
                self.cfg.to_cross_attention_config()
            )
        else:
            self.cross_attention = None

        # Pooling for cross-attended node features
        self.cross_pool = GlobalAttentionPooling(self.cfg.hidden_dim)

        # 3. Multi-task prediction heads
        self.heads = MultiTaskHeads(self.cfg.to_task_heads_config())

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        batch : dict with keys
            enzyme_pcc : dict
                node_features : (N, node_feat_dim)
                edge_index    : (2, E)
                edge_features : (E, edge_feat_dim)
                pos           : (N, 3)
                face_to_edge  : (3, F) or None
                batch         : (N,) graph membership (optional)
            substrate : dict (optional)
                node_features : (M_sub, mol_feat_dim)
                edge_index    : (2, E_sub)
                batch         : (M_sub,) optional
            product : dict (optional)
                node_features : (M_prod, mol_feat_dim)
                edge_index    : (2, E_prod)
                batch         : (M_prod,) optional

        Returns
        -------
        outputs : dict
            ec_logits    : list of 4 tensors [(B, C0), ..., (B, C3)]
            selectivity  : (B, 1) or None
            kinetics     : (B, 3) or None
            enzyme_embedding : (B, H)
            h_nodes      : (N, H)
            attn_maps    : dict or None
        """
        # ─── 1. Encode enzyme active site ───
        enzyme_pcc = batch["enzyme_pcc"]
        enzyme_embedding, h_nodes = self.encoder(enzyme_pcc)
        # enzyme_embedding: (B, H), h_nodes: (N, H)

        enzyme_batch = enzyme_pcc.get("batch")

        # ─── 2. Cross-attention with substrate/product ───
        h_sub_attended = None
        h_prod_attended = None
        attn_maps = None

        has_substrate = "substrate" in batch and batch["substrate"] is not None
        has_product = "product" in batch and batch["product"] is not None

        if self.cross_attention is not None and has_substrate and has_product:
            substrate_data = batch["substrate"]
            product_data = batch["product"]

            h_fused, attn_maps = self.cross_attention(
                h_nodes,
                substrate_data,
                product_data,
                enzyme_batch=enzyme_batch,
                substrate_batch=substrate_data.get("batch"),
                product_batch=product_data.get("batch"),
            )

            # Pool fused node features to graph level
            h_fused_pooled = self.cross_pool(h_fused, enzyme_batch)

            # For the heads, we need separate substrate-attended and
            # product-attended pooled features.  We approximate by using
            # the fused pooled as "substrate-attended" and the base
            # enzyme embedding as a stand-in — but more precisely we can
            # pool from the attention outputs.  For now, use the fused
            # representation for both roles.
            h_sub_attended = h_fused_pooled
            h_prod_attended = enzyme_embedding  # base enzyme as comparison

        # ─── 3. Multi-task prediction heads ───
        head_outputs = self.heads(
            enzyme_embedding,
            h_sub_attended,
            h_prod_attended,
        )

        outputs = {
            **head_outputs,
            "enzyme_embedding": enzyme_embedding,
            "h_nodes": h_nodes,
            "attn_maps": attn_maps,
        }

        return outputs

    def get_enzyme_embedding(
        self,
        enzyme_pcc: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Extract enzyme embedding only (no substrate/product needed)."""
        embedding, _ = self.encoder(enzyme_pcc)
        return embedding

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        counts = {}
        counts["encoder"] = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        if self.cross_attention is not None:
            counts["cross_attention"] = sum(
                p.numel()
                for p in self.cross_attention.parameters()
                if p.requires_grad
            )
        counts["heads"] = sum(
            p.numel() for p in self.heads.parameters() if p.requires_grad
        )
        counts["total"] = sum(counts.values())
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# Complete ToPE Model — Whole-Protein Context
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompleteToPEConfig:
    """Configuration for the whole-protein ToPE model."""

    # Whole-protein encoder
    zone1_feat_dim: int = 128
    zone2_feat_dim: int = 64
    zone3_feat_dim: int = 32
    edge_feat_dim: int = 4
    hidden_dim: int = 256
    n_message_layers: int = 10
    n_rbf: int = 20
    distance_cutoff: float = 15.0
    dropout: float = 0.1

    # Task heads
    ec_levels: List[int] = field(default_factory=lambda: [7, 70, 250, 800])
    selectivity_dim: int = 1
    head_hidden_dim: int = 512
    n_heads: int = 8

    # Cross-attention (optional — for substrate/product if available)
    mol_feat_dim: int = 64
    mol_hidden_dim: int = 128
    mol_n_layers: int = 3
    use_cross_attention: bool = True

    def to_whole_protein_config(self) -> WholeProteinConfig:
        return WholeProteinConfig(
            zone1_feat_dim=self.zone1_feat_dim,
            zone2_feat_dim=self.zone2_feat_dim,
            zone3_feat_dim=self.zone3_feat_dim,
            edge_feat_dim=self.edge_feat_dim,
            hidden_dim=self.hidden_dim,
            n_message_layers=self.n_message_layers,
            n_rbf=self.n_rbf,
            distance_cutoff=self.distance_cutoff,
            dropout=self.dropout,
        )

    def to_task_heads_config(self) -> TaskHeadsConfig:
        return TaskHeadsConfig(
            hidden_dim=self.hidden_dim,
            ec_levels=self.ec_levels,
            selectivity_dim=self.selectivity_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout=self.dropout,
        )

    def to_cross_attention_config(self) -> CrossAttentionConfig:
        return CrossAttentionConfig(
            mol_feat_dim=self.mol_feat_dim,
            mol_hidden_dim=self.mol_hidden_dim,
            mol_n_layers=self.mol_n_layers,
            enzyme_hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )


class CompleteToPEModel(nn.Module):
    """Whole-protein ToPE model with multi-scale message passing.

    Architecture::

        Multi-scale protein graph
            → WholeProteinTCPNet (10-layer deep MP)
            → zone-aware pooling → enzyme embedding
            → (optional) substrate/product cross-attention
            → WholeProteinTaskHeads
                ├── EC classification (hierarchical)
                ├── Selectivity (substrate vs product)
                ├── Enhanced kinetics (residue-level attention)
                └── Mutation effect prediction

    Captures both local active-site chemistry and long-range allosteric
    effects through deep message passing across the full protein.
    """

    def __init__(self, cfg: Optional[CompleteToPEConfig] = None):
        super().__init__()
        self.cfg = cfg or CompleteToPEConfig()
        H = self.cfg.hidden_dim

        # Whole-protein encoder
        self.encoder = WholeProteinTCPNet(self.cfg.to_whole_protein_config())

        # Optional cross-attention for substrate/product
        if self.cfg.use_cross_attention:
            self.cross_attention = SubstrateProductCrossAttention(
                self.cfg.to_cross_attention_config()
            )
        else:
            self.cross_attention = None

        self.cross_pool = GlobalAttentionPooling(H)

        # Task heads (whole-protein version)
        self.heads = WholeProteinTaskHeads(self.cfg.to_task_heads_config())

        # Mutation effect head (uses per-residue embeddings directly)
        self.mutation_head = DistantMutationEffectPredictor(
            H, dropout=self.cfg.dropout
        )

        # Kinetic head reference for external access
        self.kinetic_head = self.heads.kinetics_head

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        batch : dict with keys
            protein_graph : dict from MultiScaleProteinGraph.build()
                node_features, node_coords, edge_index, edge_features,
                zone_assignments, catalytic_mask, ...
            substrate : dict (optional)
            product   : dict (optional)
            targets   : dict (optional, for loss computation)

        Returns
        -------
        predictions : dict
        """
        pg = batch["protein_graph"]

        # ─── 1. Encode whole protein ───
        enzyme_embedding, h_residues = self.encoder(pg)
        # enzyme_embedding: (B, H), h_residues: (N, H)

        batch_idx = pg.get("batch")
        zone_assigns = pg["zone_assignments"]

        # Reshape h_residues for heads: (B, N_res, H)
        # For single-graph case, add batch dim
        if h_residues.dim() == 2 and (batch_idx is None or batch_idx.max() == 0):
            h_res_3d = h_residues.unsqueeze(0)
            zone_3d = zone_assigns.unsqueeze(0)
        else:
            # Multi-graph batching: group by batch index
            h_res_3d, zone_3d = _batch_residues(h_residues, zone_assigns, batch_idx)

        # ─── 2. Optional cross-attention ───
        h_sub_attended = None
        h_prod_attended = None

        has_sub = "substrate" in batch and batch["substrate"] is not None
        has_prod = "product" in batch and batch["product"] is not None

        if self.cross_attention is not None and has_sub and has_prod:
            h_fused, _ = self.cross_attention(
                h_residues,
                batch["substrate"],
                batch["product"],
                enzyme_batch=batch_idx,
                substrate_batch=batch["substrate"].get("batch"),
                product_batch=batch["product"].get("batch"),
            )
            h_sub_attended = self.cross_pool(h_fused, batch_idx)
            h_prod_attended = enzyme_embedding

        # ─── 3. Task heads ───
        outputs = self.heads(
            enzyme_embedding,
            h_residues=h_res_3d,
            zone_assignments=zone_3d,
            h_sub_attended=h_sub_attended,
            h_prod_attended=h_prod_attended,
        )

        outputs["enzyme_embedding"] = enzyme_embedding
        outputs["h_residues"] = h_residues

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        counts = {}
        counts["encoder"] = sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        if self.cross_attention is not None:
            counts["cross_attention"] = sum(
                p.numel()
                for p in self.cross_attention.parameters()
                if p.requires_grad
            )
        counts["heads"] = sum(
            p.numel() for p in self.heads.parameters() if p.requires_grad
        )
        counts["mutation_head"] = sum(
            p.numel() for p in self.mutation_head.parameters() if p.requires_grad
        )
        counts["total"] = sum(counts.values())
        return counts


def _batch_residues(
    h_residues: torch.Tensor,
    zone_assignments: torch.Tensor,
    batch_idx: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape flat (N_total, H) residue embeddings into (B, N_max, H).

    Pads shorter graphs with zeros.
    """
    if batch_idx is None:
        return h_residues.unsqueeze(0), zone_assignments.unsqueeze(0)

    B = int(batch_idx.max().item()) + 1
    H = h_residues.size(-1)

    # Count nodes per graph
    counts = torch.bincount(batch_idx, minlength=B)
    N_max = int(counts.max().item())

    h_out = torch.zeros(B, N_max, H, device=h_residues.device)
    z_out = torch.zeros(B, N_max, dtype=torch.long, device=h_residues.device)

    for gid in range(B):
        mask = batch_idx == gid
        n = int(mask.sum().item())
        h_out[gid, :n] = h_residues[mask]
        z_out[gid, :n] = zone_assignments[mask]

    return h_out, z_out
