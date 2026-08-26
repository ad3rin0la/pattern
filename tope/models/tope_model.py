"""
ToPE Model
==========

**Training model (Phase 2+):**

    CompleteToPEModel — whole-protein encoder (WholeProteinTCPNet) with the
    full multi-scale protein graph, is_active_site attention bias, and
    enhanced task heads.  This is the *only* model used in the training
    pipeline from Phase 2 onward.

**Baseline / ablation reference (deprecated):**

    ToPEModel — 8 Å active-site crop variant.  Retained *only* as an
    ablation baseline for Phase 2 F-score comparisons.  It raises a
    DeprecationWarning on instantiation and must not appear in any training
    script or production inference path.

Usage
-----
    from tope.models import CompleteToPEModel, CompleteToPEConfig

    model = CompleteToPEModel(CompleteToPEConfig())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tope.models.tcpnet import EnzymeTCPNet, TCPNetConfig, GlobalAttentionPooling
from tope.models.cross_attention import (
    SubstrateProductCrossAttention,
    CrossAttentionConfig,
)
from tope.models.task_heads import (
    MultiTaskHeads,
    TaskHeadsConfig,
    WholeProteinTaskHeads,
    DistantMutationEffectPredictor,
)
from tope.models.whole_protein_tcpnet import WholeProteinTCPNet, WholeProteinConfig
from tope.models.domain_discovery import (
    DomainDiscoveryConfig,
    DomainDiscoveryHead,
    DomainSubstrateAttention,
)
from tope.quantum.electronic_complex import (
    ElectronicComplexConfig,
    ElectronicComplexEncoder,
)


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
        warnings.warn(
            "ToPEModel (8Å active-site crop) is deprecated. "
            "Use CompleteToPEModel, which operates on the full protein via "
            "WholeProteinPCC and is the sole training model as of Phase 2. "
            "ToPEModel is retained only as a baseline for ablations.",
            DeprecationWarning,
            stacklevel=2,
        )
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

            _h_fused, attn_maps = self.cross_attention(
                h_nodes,
                substrate_data,
                product_data,
                enzyme_batch=enzyme_batch,
                substrate_batch=substrate_data.get("batch"),
                product_batch=product_data.get("batch"),
            )

            h_sub_attended = self.cross_pool(
                attn_maps["h_substrate_attended"], enzyme_batch
            )
            h_prod_attended = self.cross_pool(
                attn_maps["h_product_attended"], enzyme_batch
            )

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

    # Label-free latent domain discovery
    use_domain_discovery: bool = True
    max_latent_domains: int = 8
    sequence_feat_dim: int = 21

    # Optional multiresolution electronic cochain stack
    use_electronic_complex: bool = False
    electronic_spectrum_dim: int = 32
    electronic_hidden_dim: int = 128
    electronic_max_ranks: int = 6

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

    def to_domain_discovery_config(self) -> DomainDiscoveryConfig:
        return DomainDiscoveryConfig(
            hidden_dim=self.hidden_dim,
            sequence_feat_dim=self.sequence_feat_dim,
            max_domains=self.max_latent_domains,
            dropout=self.dropout,
        )

    def to_electronic_complex_config(self) -> ElectronicComplexConfig:
        return ElectronicComplexConfig(
            spectrum_dim=self.electronic_spectrum_dim,
            hidden_dim=self.electronic_hidden_dim,
            max_ranks=self.electronic_max_ranks,
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

        if self.cfg.use_domain_discovery:
            self.domain_discovery = DomainDiscoveryHead(
                self.cfg.to_domain_discovery_config()
            )
            self.domain_substrate_attention = DomainSubstrateAttention(
                H, self.cfg.n_heads, self.cfg.dropout
            )
            self.domain_residue_norm = nn.LayerNorm(H)
        else:
            self.domain_discovery = None
            self.domain_substrate_attention = None
            self.domain_residue_norm = None

        if self.cfg.use_electronic_complex:
            self.electronic_complex = ElectronicComplexEncoder(
                self.cfg.to_electronic_complex_config()
            )
            self.electronic_domain_projection = nn.Linear(
                self.cfg.electronic_hidden_dim, H
            )
        else:
            self.electronic_complex = None
            self.electronic_domain_projection = None

        # Task heads (whole-protein version)
        self.heads = WholeProteinTaskHeads(self.cfg.to_task_heads_config())

        # Mutation effect head (uses per-residue embeddings directly)
        self.mutation_head = DistantMutationEffectPredictor(
            H, dropout=self.cfg.dropout
        )

        # Kinetic head reference for external access
        self.kinetic_head = self.heads.kinetics_head

    def forward(
        self,
        batch: Dict[str, Any],
        masked_domains: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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

        domain_outputs = None
        domain_losses = None
        electronic_fingerprint = None
        h_residues_domain = h_residues
        if self.domain_discovery is not None:
            domain_outputs = self.domain_discovery(
                h_residues,
                batch=batch_idx,
                sequence_features=pg.get("sequence_features"),
            )
            if batch.get("compute_domain_losses", self.training):
                masked_residues = pg.get("masked_residues")
                loss_domain_outputs = domain_outputs
                if masked_residues is None and self.training:
                    masked_residues = torch.rand(
                        h_residues.size(0), device=h_residues.device
                    ) < self.domain_discovery.cfg.mask_probability
                    if not masked_residues.any() and masked_residues.numel():
                        masked_residues[0] = True
                if masked_residues is not None and masked_residues.any():
                    masked_h = h_residues.clone()
                    masked_h[masked_residues] = 0.0
                    masked_sequence = pg.get("sequence_features")
                    if masked_sequence is not None:
                        masked_sequence = masked_sequence.clone()
                        masked_sequence[masked_residues] = 0.0
                    loss_domain_outputs = self.domain_discovery(
                        masked_h,
                        batch=batch_idx,
                        sequence_features=masked_sequence,
                    )
                perturbed_assignments = batch.get("perturbed_domain_assignments")
                if perturbed_assignments is None and self.training:
                    noisy_h = h_residues + self.domain_discovery.cfg.perturbation_noise * torch.randn_like(
                        h_residues
                    )
                    perturbed_assignments = self.domain_discovery(
                        noisy_h,
                        batch=batch_idx,
                        sequence_features=pg.get("sequence_features"),
                    )["assignments"]
                domain_losses = self.domain_discovery.losses(
                    loss_domain_outputs,
                    h_residues,
                    pg["edge_index"],
                    batch=batch_idx,
                    chain_index=pg.get("chain_index"),
                    sequence_index=pg.get("sequence_index"),
                    masked_residues=masked_residues,
                    perturbed_assignments=perturbed_assignments,
                )
            # Soft domain→residue restriction map closes the learned
            # residue↔domain cross-rank path before substrate attention.
            if batch_idx is None:
                domain_batch = torch.zeros(
                    h_residues.size(0), dtype=torch.long, device=h_residues.device
                )
            else:
                domain_batch = batch_idx
            domain_broadcast = torch.empty_like(h_residues)
            for graph_id in range(domain_outputs["domain_embeddings"].size(0)):
                mask = domain_batch == graph_id
                domain_broadcast[mask] = (
                    domain_outputs["assignments"][mask]
                    @ domain_outputs["domain_embeddings"][graph_id]
                )
            h_residues_domain = self.domain_residue_norm(
                h_residues + domain_broadcast
            )

        if self.electronic_complex is not None and batch.get("electronic_complex") is not None:
            electronic_inputs = dict(batch["electronic_complex"])
            residue_rank = int(electronic_inputs.pop("residue_rank", -1))
            electronic_fingerprint = self.electronic_complex(**electronic_inputs)
            if domain_outputs is not None:
                residue_cells = electronic_fingerprint.cochains[residue_rank].scalar.size(0)
                if residue_cells != domain_outputs["assignments"].size(0):
                    raise ValueError(
                        "electronic residue rank must align with latent-domain residues"
                    )
                electronic_fingerprint = self.electronic_complex.append_soft_rank(
                    electronic_fingerprint,
                    domain_outputs["assignments"],
                    rank_name="learned_domain",
                    source_rank=residue_rank,
                )

        # Reshape h_residues for heads: (B, N_res, H)
        # For single-graph case, add batch dim
        if h_residues.dim() == 2 and (batch_idx is None or batch_idx.max() == 0):
            h_res_3d = h_residues_domain.unsqueeze(0)
            zone_3d = zone_assigns.unsqueeze(0)
        else:
            # Multi-graph batching: group by batch index
            h_res_3d, zone_3d = _batch_residues(
                h_residues_domain, zone_assigns, batch_idx
            )

        # ─── 2. Optional cross-attention ───
        h_sub_attended = None
        h_prod_attended = None

        has_sub = "substrate" in batch and batch["substrate"] is not None
        has_prod = "product" in batch and batch["product"] is not None

        if self.cross_attention is not None and has_sub and has_prod:
            _h_fused, attended = self.cross_attention(
                h_residues_domain,
                batch["substrate"],
                batch["product"],
                enzyme_batch=batch_idx,
                substrate_batch=batch["substrate"].get("batch"),
                product_batch=batch["product"].get("batch"),
            )
            h_sub_attended = self.cross_pool(
                attended["h_substrate_attended"], batch_idx
            )
            h_prod_attended = self.cross_pool(
                attended["h_product_attended"], batch_idx
            )
            if domain_outputs is not None:
                substrate_domain_embeddings = []
                assignments = domain_outputs["assignments"]
                if batch_idx is None:
                    domain_batch = torch.zeros(
                        h_residues.size(0), dtype=torch.long, device=h_residues.device
                    )
                else:
                    domain_batch = batch_idx
                n_graphs = domain_outputs["domain_embeddings"].size(0)
                for graph_id in range(n_graphs):
                    mask = domain_batch == graph_id
                    q_g = assignments[mask]
                    h_g = attended["h_substrate_attended"][mask]
                    pooled = q_g.transpose(0, 1) @ h_g
                    pooled = pooled / q_g.sum(dim=0).clamp_min(1e-8).unsqueeze(-1)
                    substrate_domain_embeddings.append(pooled)
                substrate_domain_embeddings = torch.stack(substrate_domain_embeddings)
                if electronic_fingerprint is not None:
                    if substrate_domain_embeddings.size(0) != 1:
                        raise ValueError(
                            "batched electronic fingerprints require one hierarchy per graph"
                        )
                    electronic_domains = electronic_fingerprint.cochains[-1].embedding
                    if electronic_domains.size(0) != substrate_domain_embeddings.size(1):
                        raise ValueError("electronic and geometric latent-domain counts differ")
                    substrate_domain_embeddings = substrate_domain_embeddings + (
                        self.electronic_domain_projection(electronic_domains).unsqueeze(0)
                    )
                h_sub_attended, domain_attention = self.domain_substrate_attention(
                    h_sub_attended,
                    substrate_domain_embeddings,
                    domain_outputs["domain_mass"],
                    masked_domains,
                )
            else:
                domain_attention = None
        else:
            domain_attention = None

        # ─── 3. Task heads ───
        outputs = self.heads(
            enzyme_embedding,
            h_residues=h_res_3d,
            zone_assignments=zone_3d,
            h_sub_attended=h_sub_attended,
            h_prod_attended=h_prod_attended,
        )

        outputs["enzyme_embedding"] = enzyme_embedding
        outputs["h_residues"] = h_residues_domain
        outputs["h_residues_pre_domain"] = h_residues
        outputs["latent_domains"] = domain_outputs
        outputs["domain_losses"] = domain_losses
        outputs["domain_substrate_attention"] = domain_attention
        outputs["electronic_fingerprint"] = electronic_fingerprint

        return outputs

    @torch.no_grad()
    def counterfactual_domain_specificity(
        self, batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Return Δ_k(s)=ŷ(e,s)-ŷ(e without latent domain k,s)."""
        was_training = self.training
        self.eval()
        baseline = self(batch)
        if baseline.get("log_efficiency") is None or baseline.get("latent_domains") is None:
            raise ValueError("counterfactual specificity requires molecules and domain discovery")
        base_y = baseline["log_efficiency"]
        mass = baseline["latent_domains"]["domain_mass"]
        deltas = []
        for domain_idx in range(mass.size(1)):
            mask = torch.zeros_like(mass, dtype=torch.bool)
            mask[:, domain_idx] = True
            masked_y = self(batch, masked_domains=mask)["log_efficiency"]
            deltas.append(base_y - masked_y)
        self.train(was_training)
        return torch.cat(deltas, dim=-1)

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
        if self.domain_discovery is not None:
            counts["domain_discovery"] = sum(
                p.numel() for p in self.domain_discovery.parameters() if p.requires_grad
            ) + sum(
                p.numel() for p in self.domain_substrate_attention.parameters()
                if p.requires_grad
            ) + sum(
                p.numel() for p in self.domain_residue_norm.parameters()
                if p.requires_grad
            )
        if self.electronic_complex is not None:
            counts["electronic_complex"] = sum(
                p.numel() for p in self.electronic_complex.parameters()
                if p.requires_grad
            ) + sum(
                p.numel() for p in self.electronic_domain_projection.parameters()
                if p.requires_grad
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
