"""
Multi-Task Prediction Heads
============================

Downstream heads for the ToPE model, including both the original
active-site-only heads and the updated whole-protein heads:

1. **EC Classification** — Hierarchical 4-level EC number prediction
   (multi-label with level-consistency regularisation).

2. **Selectivity Regression** — Predicts selectivity scores (e.g.
   enantiomeric excess %ee) from the difference between substrate-
   attended and product-attended enzyme representations.

3. **Kinetics Regression** — Predicts log kcat, log Km, and log(kcat/Km)
   from the substrate-aware enzyme embedding.

4. **Enhanced Kinetics Head** — Whole-protein kinetics prediction with
   residue-level attention over all zones (captures allosteric effects).

5. **Distant Mutation Effect Predictor** — Predicts ΔΔG for mutations,
   including those >15 Å from the active site.

All heads accept graph-level pooled embeddings (B, H).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TaskHeadsConfig:
    """Hyper-parameters for the multi-task prediction heads."""

    hidden_dim: int = 256          # Input embedding dimension (from TCPNet)
    ec_levels: List[int] = field(default_factory=lambda: [7, 70, 250, 800])
    # Number of classes at each EC level (approximate; actual depends on dataset)
    # Level 0: 7 main classes, Level 1: ~70, Level 2: ~250, Level 3: ~800

    selectivity_dim: int = 1       # Output dimension for selectivity
    kinetics_targets: int = 3      # log_kcat, log_Km, log(kcat/Km)
    head_hidden_dim: int = 512     # Hidden dimension within heads
    dropout: float = 0.1


# ── EC Classification Head ───────────────────────────────────────────────────

class ECClassificationHead(nn.Module):
    """Hierarchical EC number classifier.

    Predicts EC number at four levels of increasing specificity.
    Each level's prediction is conditioned on the previous level's
    representation to encourage hierarchical consistency.
    """

    def __init__(self, cfg: TaskHeadsConfig):
        super().__init__()
        H = cfg.hidden_dim
        H_head = cfg.head_hidden_dim

        self.level_encoders = nn.ModuleList()
        self.level_classifiers = nn.ModuleList()

        in_dim = H
        for n_classes in cfg.ec_levels:
            encoder = nn.Sequential(
                nn.Linear(in_dim, H_head),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(H_head, H_head),
            )
            classifier = nn.Linear(H_head, n_classes)
            self.level_encoders.append(encoder)
            self.level_classifiers.append(classifier)
            # Next level receives the encoded representation concatenated
            # with the original embedding
            in_dim = H_head + H

    def forward(
        self,
        h_enzyme: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        h_enzyme : (B, H)
            Graph-level enzyme embedding.

        Returns
        -------
        logits : list of 4 tensors
            [(B, n_classes_level_0), ..., (B, n_classes_level_3)]
        """
        logits = []
        h = h_enzyme
        for encoder, classifier in zip(self.level_encoders, self.level_classifiers):
            encoded = encoder(h)
            logits.append(classifier(encoded))
            # Condition next level on current + original
            h = torch.cat([encoded, h_enzyme], dim=-1)
        return logits


# ── Selectivity Regression Head ──────────────────────────────────────────────

class SelectivityHead(nn.Module):
    """Predicts selectivity from the difference between substrate-attended
    and product-attended enzyme representations.

    The intuition: selectivity depends on how differently the enzyme
    recognises the substrate versus the product.
    """

    def __init__(self, cfg: TaskHeadsConfig):
        super().__init__()
        H = cfg.hidden_dim

        # Operates on concatenation of [enzyme_emb, |sub_attn - prod_attn|, sub_attn * prod_attn]
        self.mlp = nn.Sequential(
            nn.Linear(3 * H, cfg.head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim // 2, cfg.selectivity_dim),
        )

    def forward(
        self,
        h_enzyme: torch.Tensor,
        h_sub_attended: torch.Tensor,
        h_prod_attended: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_enzyme        : (B, H)  Graph-level enzyme embedding
        h_sub_attended  : (B, H)  Substrate-attended enzyme embedding (pooled)
        h_prod_attended : (B, H)  Product-attended enzyme embedding (pooled)

        Returns
        -------
        selectivity : (B, selectivity_dim)
        """
        diff = torch.abs(h_sub_attended - h_prod_attended)
        prod = h_sub_attended * h_prod_attended
        combined = torch.cat([h_enzyme, diff, prod], dim=-1)
        return self.mlp(combined)


# ── Kinetics Regression Head ─────────────────────────────────────────────────

class KineticsHead(nn.Module):
    """Predicts kinetic parameters: log kcat, log Km, log(kcat/Km).

    Uses the substrate-aware enzyme embedding.  Each kinetic parameter
    gets its own final linear layer to allow independent scaling.
    """

    def __init__(self, cfg: TaskHeadsConfig):
        super().__init__()
        H = cfg.hidden_dim
        H_head = cfg.head_hidden_dim

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(2 * H, H_head),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(H_head, H_head // 2),
            nn.SiLU(),
        )

        # Per-target output layers
        self.target_heads = nn.ModuleList([
            nn.Linear(H_head // 2, 1)
            for _ in range(cfg.kinetics_targets)
        ])

    def forward(
        self,
        h_enzyme: torch.Tensor,
        h_sub_attended: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_enzyme       : (B, H)  Graph-level enzyme embedding
        h_sub_attended : (B, H)  Substrate-attended enzyme embedding (pooled)

        Returns
        -------
        kinetics : (B, kinetics_targets)
            Columns: [log_kcat, log_Km, log(kcat/Km)]
        """
        combined = torch.cat([h_enzyme, h_sub_attended], dim=-1)
        shared_repr = self.shared(combined)
        preds = [head(shared_repr) for head in self.target_heads]
        return torch.cat(preds, dim=-1)


# ── Combined multi-task head ─────────────────────────────────────────────────

class MultiTaskHeads(nn.Module):
    """Combines all three task heads into a single module."""

    def __init__(self, cfg: Optional[TaskHeadsConfig] = None):
        super().__init__()
        self.cfg = cfg or TaskHeadsConfig()

        self.ec_head = ECClassificationHead(self.cfg)
        self.selectivity_head = SelectivityHead(self.cfg)
        self.kinetics_head = KineticsHead(self.cfg)

    def forward(
        self,
        h_enzyme: torch.Tensor,
        h_sub_attended: Optional[torch.Tensor] = None,
        h_prod_attended: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        h_enzyme        : (B, H)
        h_sub_attended  : (B, H) or None (skips selectivity + kinetics)
        h_prod_attended : (B, H) or None (skips selectivity)

        Returns
        -------
        outputs : dict with keys
            'ec_logits' : list of 4 tensors  [(B, C0), ..., (B, C3)]
            'selectivity' : (B, 1) or None
            'kinetics' : (B, 3) or None
        """
        outputs: Dict[str, object] = {}

        # EC classification always runs
        outputs["ec_logits"] = self.ec_head(h_enzyme)

        # Selectivity requires both substrate and product attention
        if h_sub_attended is not None and h_prod_attended is not None:
            outputs["selectivity"] = self.selectivity_head(
                h_enzyme, h_sub_attended, h_prod_attended
            )
        else:
            outputs["selectivity"] = None

        # Kinetics requires substrate attention
        if h_sub_attended is not None:
            outputs["kinetics"] = self.kinetics_head(h_enzyme, h_sub_attended)
        else:
            outputs["kinetics"] = None

        return outputs


# ── Enhanced Kinetics Head (whole-protein context) ───────────────────────────

class EnhancedKineticEfficiencyHead(nn.Module):
    """Predicts kcat/Km using both global and residue-level information.

    Captures allosteric effects via learned attention over distant residues.
    Unlike the basic ``KineticsHead``, this operates on per-residue
    embeddings from the whole-protein encoder rather than a single pooled
    vector.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        H = hidden_dim

        # Global pathway (from pooled enzyme embedding)
        self.global_kcat = nn.Sequential(
            nn.Linear(H, H // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(H // 2, 1)
        )
        self.global_km = nn.Sequential(
            nn.Linear(H, H // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(H // 2, 1)
        )

        # Residue-level attention (which residues influence kinetics?)
        self.residue_attention = nn.Sequential(
            nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 1)
        )

        # Combined predictors
        self.kcat_combiner = nn.Linear(H + 1, 1)
        self.km_combiner = nn.Linear(H + 1, 1)

    def forward(
        self,
        enzyme_embedding: torch.Tensor,
        h_residues: torch.Tensor,
        zone_assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        enzyme_embedding : (B, H)         global enzyme representation
        h_residues       : (B, N_res, H)  per-residue embeddings
        zone_assignments : (B, N_res)     zone labels

        Returns
        -------
        log_kcat     : (B, 1)
        log_km       : (B, 1)
        log_efficiency : (B, 1)  log(kcat/Km)
        attn_weights : (B, N_res)
        """
        log_kcat_global = self.global_kcat(enzyme_embedding)   # (B, 1)
        log_km_global = self.global_km(enzyme_embedding)       # (B, 1)

        # Residue-level attention
        attn_logits = self.residue_attention(h_residues).squeeze(-1)  # (B, N)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Weighted residue features
        residue_contribution = (attn_weights.unsqueeze(-1) * h_residues).sum(dim=1)  # (B, H)

        # Combine global + residue-level
        log_kcat = self.kcat_combiner(
            torch.cat([residue_contribution, log_kcat_global], dim=-1)
        )
        log_km = self.km_combiner(
            torch.cat([residue_contribution, log_km_global], dim=-1)
        )

        log_efficiency = log_kcat - log_km

        return log_kcat, log_km, log_efficiency, attn_weights


# ── Distant Mutation Effect Predictor ────────────────────────────────────────

class DistantMutationEffectPredictor(nn.Module):
    """Predict ΔΔG (change in activation energy) for mutations.

    Key validation target: the model should capture effects of mutations
    distant from the active site (>15 Å), demonstrating that the
    whole-protein message passing learns allosteric communication
    pathways.
    """

    def __init__(self, hidden_dim: int = 256, n_attn_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        H = hidden_dim

        # Mutation encoder (wild-type vs mutant difference)
        self.mutation_encoder = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
        )

        # Context encoder (which surrounding residues mediate the effect?)
        self.context_attention = nn.MultiheadAttention(
            H, num_heads=n_attn_heads, batch_first=True, dropout=dropout
        )
        self.context_proj = nn.Linear(H, H)

        # Effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(H // 2 + H, H // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, 1),  # ΔΔG in kcal/mol
        )

    def forward(
        self,
        h_residues_wt: torch.Tensor,
        h_residues_mut: torch.Tensor,
        mutation_site_idx: torch.Tensor,
        zone_assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h_residues_wt   : (B, N_res, H)  wild-type residue embeddings
        h_residues_mut  : (B, N_res, H)  mutant residue embeddings
        mutation_site_idx : (B,)          index of mutated residue
        zone_assignments  : (B, N_res)   zone labels

        Returns
        -------
        delta_delta_G : (B, 1)
        attribution   : (B, N_res)  which residues mediate the effect
        """
        B = h_residues_wt.size(0)
        device = h_residues_wt.device

        # Extract mutation site embeddings
        batch_idx = torch.arange(B, device=device)
        h_wt_site = h_residues_wt[batch_idx, mutation_site_idx]   # (B, H)
        h_mut_site = h_residues_mut[batch_idx, mutation_site_idx]  # (B, H)

        # Encode mutation difference
        mutation_encoding = self.mutation_encoder(
            torch.cat([h_wt_site, h_mut_site], dim=-1)
        )  # (B, H//2)

        # Cross-attend: mutation site queries the mutant residue context
        # Project mutation encoding to H for attention
        query = self.context_proj(
            F.pad(mutation_encoding, (0, h_residues_mut.size(-1) - mutation_encoding.size(-1)))
        ).unsqueeze(1)  # (B, 1, H)

        context, attn_weights = self.context_attention(
            query=query,
            key=h_residues_mut,
            value=h_residues_mut,
        )  # context: (B, 1, H), attn: (B, 1, N)
        context = context.squeeze(1)  # (B, H)

        # Predict ΔΔG
        combined = torch.cat([mutation_encoding, context], dim=-1)
        delta_delta_G = self.effect_predictor(combined)

        attribution = attn_weights.squeeze(1)  # (B, N_res)

        return delta_delta_G, attribution


# ── Whole-protein multi-task heads ───────────────────────────────────────────

class WholeProteinTaskHeads(nn.Module):
    """Combined task heads for the whole-protein ToPE model.

    Includes all original heads plus the enhanced kinetics head and
    the distant mutation effect predictor.
    """

    def __init__(self, cfg: Optional[TaskHeadsConfig] = None):
        super().__init__()
        self.cfg = cfg or TaskHeadsConfig()
        H = self.cfg.hidden_dim

        self.ec_head = ECClassificationHead(self.cfg)
        self.selectivity_head = SelectivityHead(self.cfg)
        self.kinetics_head = EnhancedKineticEfficiencyHead(H, self.cfg.dropout)
        self.mutation_head = DistantMutationEffectPredictor(H, dropout=self.cfg.dropout)

        # Residue importance scoring (auxiliary task)
        self.residue_importance = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.SiLU(),
            nn.Linear(H // 2, 1),
        )

    def forward(
        self,
        enzyme_embedding: torch.Tensor,
        h_residues: Optional[torch.Tensor] = None,
        zone_assignments: Optional[torch.Tensor] = None,
        h_sub_attended: Optional[torch.Tensor] = None,
        h_prod_attended: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        enzyme_embedding : (B, H)
        h_residues       : (B, N, H) or None — per-residue embeddings
        zone_assignments : (B, N) or None
        h_sub_attended   : (B, H) or None
        h_prod_attended  : (B, H) or None

        Returns
        -------
        outputs : dict with keys
            'ec_logits'   : list of 4 tensors
            'selectivity' : (B, 1) or None
            'log_kcat'    : (B, 1) or None
            'log_km'      : (B, 1) or None
            'log_efficiency' : (B, 1) or None
            'kinetic_attention' : (B, N) or None
            'residue_importance_scores' : (B, N) or None
        """
        outputs: Dict[str, Any] = {}

        # EC classification always runs
        outputs["ec_logits"] = self.ec_head(enzyme_embedding)

        # Selectivity
        if h_sub_attended is not None and h_prod_attended is not None:
            outputs["selectivity"] = self.selectivity_head(
                enzyme_embedding, h_sub_attended, h_prod_attended
            )
        else:
            outputs["selectivity"] = None

        # Enhanced kinetics (requires per-residue embeddings)
        if h_residues is not None and zone_assignments is not None:
            log_kcat, log_km, log_eff, kin_attn = self.kinetics_head(
                enzyme_embedding, h_residues, zone_assignments
            )
            outputs["log_kcat"] = log_kcat
            outputs["log_km"] = log_km
            outputs["log_efficiency"] = log_eff
            outputs["kinetic_attention"] = kin_attn

            # Residue importance scores (auxiliary)
            outputs["residue_importance_scores"] = (
                self.residue_importance(h_residues).squeeze(-1)
            )
        else:
            outputs["log_kcat"] = None
            outputs["log_km"] = None
            outputs["log_efficiency"] = None
            outputs["kinetic_attention"] = None
            outputs["residue_importance_scores"] = None

        return outputs
