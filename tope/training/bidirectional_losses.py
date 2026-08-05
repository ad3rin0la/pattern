"""Losses for the bidirectional physics-informed ToPE model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from tope.models.bidirectional_physics import DiagonalGaussian


@dataclass
class BidirectionalLossConfig:
    activity: float = 1.0
    thermal: float = 1.0
    electronic: float = 1.0
    topology: float = 1.0
    physics: float = 0.25
    cycle: float = 0.5
    kl: float = 1e-3
    epistasis: float = 0.25


def masked_gaussian_nll(prediction: Mapping[str, torch.Tensor], target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL evaluated only where labels exist."""
    if mask is None:
        mask = torch.isfinite(target)
    mask = mask.bool() & torch.isfinite(target)
    if not mask.any():
        return prediction["mean"].sum() * 0.0
    mean, logvar = prediction["mean"], prediction["logvar"]
    safe_target = torch.nan_to_num(target)
    nll = 0.5 * (logvar + (safe_target - mean).square() * torch.exp(-logvar))
    return nll[mask].mean()


def symmetric_gaussian_kl(a: Mapping[str, torch.Tensor],
                          b: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Symmetric KL between diagonal Gaussian posteriors."""
    def kl(p, q):
        return 0.5 * (q["logvar"] - p["logvar"] +
                      (p["logvar"].exp() + (p["mean"] - q["mean"]).square()) /
                      q["logvar"].exp() - 1.0).sum(-1)
    return 0.5 * (kl(a, b) + kl(b, a)).mean()


class BidirectionalPhysicsLoss(nn.Module):
    """Masked multi-fidelity loss with physics and posterior cycle consistency.

    Targets may contain ``thermal``, ``catalytic``, ``topology``, and matching
    ``*_mask`` tensors. Optional direct ``rate``/``rate_mask`` labels constrain
    the integrated physics layer. ``electronic_mask`` selects catalytic columns
    containing electronic/QM observables. Missing groups contribute zero.
    """

    def __init__(self, cfg: Optional[BidirectionalLossConfig] = None):
        super().__init__()
        self.cfg = cfg or BidirectionalLossConfig()

    def forward(self, forward_output: Mapping, targets: Mapping[str, torch.Tensor],
                inverse_output: Optional[Mapping] = None,
                physics_output: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        zero = forward_output["z"].sum() * 0.0
        losses: Dict[str, torch.Tensor] = {}
        thermal_target = targets.get("thermal")
        losses["thermal"] = (masked_gaussian_nll(forward_output["thermal"], thermal_target,
                              targets.get("thermal_mask")) if thermal_target is not None else zero)
        catalytic_target = targets.get("catalytic")
        if catalytic_target is not None:
            cat_mask = targets.get("catalytic_mask")
            losses["activity"] = masked_gaussian_nll(forward_output["catalytic"], catalytic_target, cat_mask)
            electronic_mask = targets.get("electronic_mask")
            losses["electronic"] = (masked_gaussian_nll(forward_output["catalytic"], catalytic_target,
                                      electronic_mask) if electronic_mask is not None else zero)
        else:
            losses["activity"] = losses["electronic"] = zero
        topology_target = targets.get("topology")
        losses["topology"] = (masked_gaussian_nll(forward_output["topology"], topology_target,
                               targets.get("topology_mask")) if topology_target is not None else zero)
        if physics_output is not None and targets.get("rate") is not None:
            mask = targets.get("rate_mask", torch.isfinite(targets["rate"])).bool()
            residual = (physics_output["rate"] - torch.nan_to_num(targets["rate"])).square()
            losses["physics"] = residual[mask].mean() if mask.any() else zero
        else:
            losses["physics"] = zero
        losses["cycle"] = (symmetric_gaussian_kl(forward_output["posterior"], inverse_output["posterior"])
                           if inverse_output is not None else zero)
        losses["kl"] = DiagonalGaussian.kl_standard_normal(
            forward_output["posterior"]["mean"], forward_output["posterior"]["logvar"]
        ).mean()
        losses["epistasis"] = targets.get("epistasis_loss", zero)
        losses["total"] = (
            self.cfg.activity * losses["activity"] + self.cfg.thermal * losses["thermal"] +
            self.cfg.electronic * losses["electronic"] + self.cfg.topology * losses["topology"] +
            self.cfg.physics * losses["physics"] + self.cfg.cycle * losses["cycle"] +
            self.cfg.kl * losses["kl"] + self.cfg.epistasis * losses["epistasis"]
        )
        return losses
