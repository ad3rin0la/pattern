"""
Multi-Task Loss with Learned Uncertainty Weighting
====================================================

Implements the homoscedastic uncertainty weighting strategy from
Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics".

Each task has a learnable log-variance parameter ``log_σ²`` that
automatically balances the relative contribution of:

    L_total = Σ_t  (1 / 2σ²_t) · L_t  +  log σ_t

This avoids manual tuning of loss weights while maintaining gradient
scale parity across tasks.

Task losses
-----------
- EC classification : cross-entropy at each hierarchy level
- Selectivity       : Huber loss (robust to outliers in %ee)
- Kinetics          : MSE on log-scale values

Auxiliary losses
----------------
- EC hierarchy consistency penalty
- Mutation effect ΔΔG prediction (MSE)
- Residue importance ranking (binary cross-entropy)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECHierarchyConsistencyLoss(nn.Module):
    """Penalises predictions that violate EC hierarchy constraints.

    If a sample is predicted as EC 1.2.3.4, it should also be classified
    under EC 1, EC 1.2, and EC 1.2.3.  This module adds a soft penalty
    when child-level probabilities are high but parent-level probabilities
    for the correct parent are low.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        ec_logits: List[torch.Tensor],
        ec_targets: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        ec_logits  : list of 4 tensors [(B, C0), ..., (B, C3)]
        ec_targets : list of 4 tensors [(B,), ..., (B,)] with class indices

        Returns
        -------
        Scalar consistency penalty.
        """
        penalty = torch.tensor(0.0, device=ec_logits[0].device)

        for level in range(1, len(ec_logits)):
            # Probability of predicted class at current level
            child_probs = F.softmax(ec_logits[level], dim=-1)
            child_confidence = child_probs.max(dim=-1).values  # (B,)

            # Probability of correct parent at parent level
            parent_probs = F.softmax(ec_logits[level - 1], dim=-1)
            parent_target = ec_targets[level - 1]  # (B,)
            parent_correct_prob = parent_probs.gather(
                1, parent_target.unsqueeze(-1)
            ).squeeze(-1)  # (B,)

            # Penalise high child confidence with low parent probability
            violation = F.relu(child_confidence - parent_correct_prob)
            penalty = penalty + violation.mean()

        return self.weight * penalty


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting.

    Implements Kendall et al. (2018) homoscedastic uncertainty:

        L_total = Σ_t  (1 / 2σ²_t) · L_t  +  log σ_t

    Parameters are learnable log-variance terms for each task.
    """

    def __init__(
        self,
        n_ec_levels: int = 4,
        hierarchy_weight: float = 0.1,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        # Learnable log-variance for each task group
        # Initialise to 0 → σ² = 1 → equal weighting initially
        self.log_var_ec = nn.Parameter(torch.zeros(1))
        self.log_var_selectivity = nn.Parameter(torch.zeros(1))
        self.log_var_kinetics = nn.Parameter(torch.zeros(1))

        self.n_ec_levels = n_ec_levels
        self.hierarchy_loss = ECHierarchyConsistencyLoss(hierarchy_weight)
        self.label_smoothing = label_smoothing

    def _ec_loss(
        self,
        ec_logits: List[torch.Tensor],
        ec_targets: List[torch.Tensor],
        level_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Cross-entropy loss across all EC hierarchy levels."""
        if level_weights is None:
            # Weight finer levels slightly less (harder to classify)
            level_weights = [1.0, 0.8, 0.6, 0.4]

        total = torch.tensor(0.0, device=ec_logits[0].device)
        for logits, targets, w in zip(ec_logits, ec_targets, level_weights):
            total = total + w * F.cross_entropy(
                logits, targets, label_smoothing=self.label_smoothing
            )

        # Hierarchy consistency
        total = total + self.hierarchy_loss(ec_logits, ec_targets)
        return total

    def _selectivity_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss for selectivity regression (robust to outliers)."""
        return F.huber_loss(pred.squeeze(-1), target, delta=10.0)

    def _kinetics_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on log-scale kinetic parameters, masked for missing data.

        Parameters
        ----------
        pred   : (B, 3)  [log_kcat, log_Km, log(kcat/Km)]
        target : (B, 3)
        mask   : (B, 3)  1.0 where target is available, 0.0 otherwise
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        sq_err = (pred - target) ** 2
        masked = sq_err * mask
        return masked.sum() / mask.sum().clamp(min=1)

    def forward(
        self,
        predictions: Dict[str, object],
        targets: Dict[str, torch.Tensor],
        active_tasks: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        predictions : dict from MultiTaskHeads.forward()
            'ec_logits'    : list of 4 tensors
            'selectivity'  : (B, 1) or None
            'kinetics'     : (B, 3) or None
        targets : dict
            'ec_levels'    : list of 4 tensors [(B,), ...]
            'selectivity'  : (B,) or None
            'kinetics'     : (B, 3) or None
            'kinetics_mask': (B, 3) or None
        active_tasks : dict (for curriculum learning)
            'ec': True/False, 'selectivity': True/False, 'kinetics': True/False
            If None, all available tasks are active.

        Returns
        -------
        loss_dict : dict
            'total'       : scalar total loss
            'ec'          : scalar EC loss
            'selectivity' : scalar selectivity loss (0 if inactive)
            'kinetics'    : scalar kinetics loss (0 if inactive)
            'weights'     : dict of current uncertainty weights {task: 1/2σ²}
        """
        device = predictions["ec_logits"][0].device
        if active_tasks is None:
            active_tasks = {"ec": True, "selectivity": True, "kinetics": True}

        loss_dict: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device)

        # ─── EC classification ───
        if active_tasks.get("ec", True):
            l_ec = self._ec_loss(predictions["ec_logits"], targets["ec_levels"])
            precision_ec = torch.exp(-self.log_var_ec)
            weighted_ec = 0.5 * precision_ec * l_ec + 0.5 * self.log_var_ec
            total = total + weighted_ec.squeeze()
            loss_dict["ec"] = l_ec.detach()
        else:
            loss_dict["ec"] = torch.tensor(0.0, device=device)

        # ─── Selectivity ───
        if (
            active_tasks.get("selectivity", False)
            and predictions.get("selectivity") is not None
            and targets.get("selectivity") is not None
        ):
            l_sel = self._selectivity_loss(
                predictions["selectivity"], targets["selectivity"]
            )
            precision_sel = torch.exp(-self.log_var_selectivity)
            weighted_sel = 0.5 * precision_sel * l_sel + 0.5 * self.log_var_selectivity
            total = total + weighted_sel.squeeze()
            loss_dict["selectivity"] = l_sel.detach()
        else:
            loss_dict["selectivity"] = torch.tensor(0.0, device=device)

        # ─── Kinetics ───
        if (
            active_tasks.get("kinetics", False)
            and predictions.get("kinetics") is not None
            and targets.get("kinetics") is not None
        ):
            kinetics_mask = targets.get(
                "kinetics_mask",
                torch.ones_like(targets["kinetics"]),
            )
            l_kin = self._kinetics_loss(
                predictions["kinetics"], targets["kinetics"], kinetics_mask
            )
            precision_kin = torch.exp(-self.log_var_kinetics)
            weighted_kin = 0.5 * precision_kin * l_kin + 0.5 * self.log_var_kinetics
            total = total + weighted_kin.squeeze()
            loss_dict["kinetics"] = l_kin.detach()
        else:
            loss_dict["kinetics"] = torch.tensor(0.0, device=device)

        loss_dict["total"] = total

        # Current effective weights for logging
        loss_dict["weights"] = {
            "ec": (0.5 * torch.exp(-self.log_var_ec)).item(),
            "selectivity": (0.5 * torch.exp(-self.log_var_selectivity)).item(),
            "kinetics": (0.5 * torch.exp(-self.log_var_kinetics)).item(),
        }

        return loss_dict


class EnhancedMultiTaskLoss(nn.Module):
    """Extended multi-task loss for the whole-protein ToPE model.

    Adds two auxiliary tasks on top of the original three:
    - Mutation effect ΔΔG prediction (MSE)
    - Residue importance ranking (BCE with catalytic mask)

    Uncertainty weighting (Kendall et al. 2018) is extended to cover
    all five task groups.
    """

    def __init__(
        self,
        n_ec_levels: int = 4,
        hierarchy_weight: float = 0.1,
        label_smoothing: float = 0.05,
    ):
        super().__init__()

        # Learnable log-variance per task group
        self.log_var_ec = nn.Parameter(torch.zeros(1))
        self.log_var_selectivity = nn.Parameter(torch.zeros(1))
        self.log_var_kcat = nn.Parameter(torch.zeros(1))
        self.log_var_km = nn.Parameter(torch.zeros(1))
        self.log_var_mutation = nn.Parameter(torch.zeros(1))
        self.log_var_residue_imp = nn.Parameter(torch.zeros(1))

        self.hierarchy_loss = ECHierarchyConsistencyLoss(hierarchy_weight)
        self.label_smoothing = label_smoothing

    def _ec_loss(
        self,
        ec_logits: List[torch.Tensor],
        ec_targets: List[torch.Tensor],
    ) -> torch.Tensor:
        level_weights = [1.0, 0.8, 0.6, 0.4]
        total = torch.tensor(0.0, device=ec_logits[0].device)
        for logits, targets, w in zip(ec_logits, ec_targets, level_weights):
            total = total + w * F.cross_entropy(
                logits, targets, label_smoothing=self.label_smoothing
            )
        total = total + self.hierarchy_loss(ec_logits, ec_targets)
        return total

    def _weighted(
        self, loss: torch.Tensor, log_var: nn.Parameter
    ) -> torch.Tensor:
        precision = torch.exp(-log_var)
        return (0.5 * precision * loss + 0.5 * log_var).squeeze()

    def forward(
        self,
        predictions: Dict[str, object],
        targets: Dict[str, torch.Tensor],
        active_tasks: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        predictions : dict from WholeProteinTaskHeads.forward()
        targets : dict with keys matching task outputs
        active_tasks : dict controlling which tasks contribute to the loss

        Returns
        -------
        loss_dict with 'total' and per-task losses
        """
        device = predictions["ec_logits"][0].device
        if active_tasks is None:
            active_tasks = {
                "ec": True, "selectivity": True, "kcat": True,
                "km": True, "mutation": True, "residue_importance": True,
            }

        loss_dict: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device)

        # EC classification
        if active_tasks.get("ec", True) and "ec_levels" in targets:
            l_ec = self._ec_loss(predictions["ec_logits"], targets["ec_levels"])
            total = total + self._weighted(l_ec, self.log_var_ec)
            loss_dict["ec"] = l_ec.detach()
        else:
            loss_dict["ec"] = torch.tensor(0.0, device=device)

        # Selectivity
        if (
            active_tasks.get("selectivity", False)
            and predictions.get("selectivity") is not None
            and targets.get("selectivity") is not None
        ):
            sel_mask = ~torch.isnan(targets["selectivity"])
            if sel_mask.sum() > 0:
                l_sel = F.huber_loss(
                    predictions["selectivity"].squeeze(-1)[sel_mask],
                    targets["selectivity"][sel_mask],
                    delta=10.0,
                )
                total = total + self._weighted(l_sel, self.log_var_selectivity)
                loss_dict["selectivity"] = l_sel.detach()
            else:
                loss_dict["selectivity"] = torch.tensor(0.0, device=device)
        else:
            loss_dict["selectivity"] = torch.tensor(0.0, device=device)

        # kcat
        if (
            active_tasks.get("kcat", False)
            and predictions.get("log_kcat") is not None
            and targets.get("log_kcat") is not None
        ):
            kcat_mask = ~torch.isnan(targets["log_kcat"])
            if kcat_mask.sum() > 0:
                l_kcat = F.huber_loss(
                    predictions["log_kcat"].squeeze(-1)[kcat_mask],
                    targets["log_kcat"][kcat_mask],
                    delta=1.0,
                )
                total = total + self._weighted(l_kcat, self.log_var_kcat)
                loss_dict["kcat"] = l_kcat.detach()
            else:
                loss_dict["kcat"] = torch.tensor(0.0, device=device)
        else:
            loss_dict["kcat"] = torch.tensor(0.0, device=device)

        # Km
        if (
            active_tasks.get("km", False)
            and predictions.get("log_km") is not None
            and targets.get("log_km") is not None
        ):
            km_mask = ~torch.isnan(targets["log_km"])
            if km_mask.sum() > 0:
                l_km = F.huber_loss(
                    predictions["log_km"].squeeze(-1)[km_mask],
                    targets["log_km"][km_mask],
                    delta=1.0,
                )
                total = total + self._weighted(l_km, self.log_var_km)
                loss_dict["km"] = l_km.detach()
            else:
                loss_dict["km"] = torch.tensor(0.0, device=device)
        else:
            loss_dict["km"] = torch.tensor(0.0, device=device)

        # Mutation ΔΔG
        if (
            active_tasks.get("mutation", False)
            and predictions.get("mutation_ddG") is not None
            and targets.get("mutation_ddG") is not None
        ):
            mut_mask = ~torch.isnan(targets["mutation_ddG"])
            if mut_mask.sum() > 0:
                l_mut = F.mse_loss(
                    predictions["mutation_ddG"].squeeze(-1)[mut_mask],
                    targets["mutation_ddG"][mut_mask],
                )
                total = total + self._weighted(l_mut, self.log_var_mutation)
                loss_dict["mutation"] = l_mut.detach()
            else:
                loss_dict["mutation"] = torch.tensor(0.0, device=device)
        else:
            loss_dict["mutation"] = torch.tensor(0.0, device=device)

        # Residue importance (auxiliary)
        if (
            active_tasks.get("residue_importance", False)
            and predictions.get("residue_importance_scores") is not None
            and targets.get("catalytic_residue_mask") is not None
        ):
            l_imp = F.binary_cross_entropy_with_logits(
                predictions["residue_importance_scores"],
                targets["catalytic_residue_mask"].float(),
            )
            total = total + self._weighted(l_imp, self.log_var_residue_imp)
            loss_dict["residue_importance"] = l_imp.detach()
        else:
            loss_dict["residue_importance"] = torch.tensor(0.0, device=device)

        loss_dict["total"] = total
        loss_dict["weights"] = {
            "ec": (0.5 * torch.exp(-self.log_var_ec)).item(),
            "selectivity": (0.5 * torch.exp(-self.log_var_selectivity)).item(),
            "kcat": (0.5 * torch.exp(-self.log_var_kcat)).item(),
            "km": (0.5 * torch.exp(-self.log_var_km)).item(),
            "mutation": (0.5 * torch.exp(-self.log_var_mutation)).item(),
            "residue_importance": (0.5 * torch.exp(-self.log_var_residue_imp)).item(),
        }

        return loss_dict
