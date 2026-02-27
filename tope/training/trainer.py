"""
ToPE Trainer — Curriculum Learning & Multi-Task Training Loop
==============================================================

Two training modes:

**Active-site mode** (original Phase 3):
    Phase A: EC only → Phase B: EC + selectivity → Phase C: all tasks

**Whole-protein mode** (updated Phase 3):
    Gradually expands the effective receptive field during training.
    Early epochs focus on active site (Zone 1), later epochs incorporate
    the full protein.  Message-passing depth and zone dropout rates are
    controlled by ``WholeProteinCurriculumScheduler``.

Data augmentation includes coordinate noise, subgraph dropout, and
zone-level dropout for the whole-protein encoder.

Validation is run at multiple difficulty levels:
    - Within-family (same EC superfamily)
    - Cross-family  (different superfamily, same fold)
    - Low-similarity (<40% sequence identity)
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    """Training hyper-parameters and curriculum schedule."""

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"          # "cosine" | "onecycle"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Training
    n_epochs: int = 100
    patience: int = 15                 # Early stopping patience
    batch_size: int = 32
    accumulation_steps: int = 1        # Gradient accumulation

    # Curriculum phases (epoch thresholds)
    phase_a_end: int = 20              # EC only
    phase_b_end: int = 50              # EC + selectivity
    # After phase_b_end: all tasks active

    # Data augmentation
    coord_noise_std: float = 0.1       # Å
    subgraph_dropout: float = 0.05     # Fraction of edges to drop
    augment: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5                # Save checkpoint every N epochs

    # Logging
    log_interval: int = 10             # Log every N batches


# ── Data augmentation ─────────────────────────────────────────────────────────

class DataAugmenter:
    """Stochastic augmentations applied to enzyme graph batches."""

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.cfg.augment:
            return batch

        batch = dict(batch)  # shallow copy

        # Coordinate perturbation
        if "pos" in batch and self.cfg.coord_noise_std > 0:
            noise = torch.randn_like(batch["pos"]) * self.cfg.coord_noise_std
            batch["pos"] = batch["pos"] + noise

        # Subgraph dropout: randomly remove edges
        if "edge_index" in batch and self.cfg.subgraph_dropout > 0:
            edge_index = batch["edge_index"]
            n_edges = edge_index.size(1)
            keep_mask = torch.rand(n_edges, device=edge_index.device) > self.cfg.subgraph_dropout
            batch["edge_index"] = edge_index[:, keep_mask]

            # Also mask edge features if present
            if "edge_features" in batch:
                batch["edge_features"] = batch["edge_features"][keep_mask]

        return batch


# ── Curriculum schedule ───────────────────────────────────────────────────────

def get_active_tasks(epoch: int, cfg: TrainerConfig) -> Dict[str, bool]:
    """Determine which tasks are active at a given epoch."""
    if epoch < cfg.phase_a_end:
        return {"ec": True, "selectivity": False, "kinetics": False}
    elif epoch < cfg.phase_b_end:
        return {"ec": True, "selectivity": True, "kinetics": False}
    else:
        return {"ec": True, "selectivity": True, "kinetics": True}


# ── Whole-protein curriculum scheduler ────────────────────────────────────────

class WholeProteinCurriculumScheduler:
    """Gradually expand the effective receptive field during training.

    - Early epochs: focus on active site (Zone 1), few message layers
    - Mid epochs: add second shell (Zone 2), more layers
    - Late epochs: whole protein (all zones), full depth

    Zone dropout selectively masks node features from outer zones so the
    encoder learns a useful active-site representation before incorporating
    distal context.
    """

    def __init__(self, total_epochs: int = 150, max_layers: int = 10):
        self.total_epochs = total_epochs
        self.max_layers = max_layers
        self.current_epoch = 0

    def get_zone_dropout_rates(self) -> Dict[str, float]:
        """Return per-zone dropout rates for the current epoch."""
        progress = self.current_epoch / max(self.total_epochs, 1)

        if progress < 0.3:
            return {"zone1_dropout": 0.0, "zone2_dropout": 0.9, "zone3_dropout": 1.0}
        elif progress < 0.6:
            return {"zone1_dropout": 0.0, "zone2_dropout": 0.3, "zone3_dropout": 0.8}
        else:
            return {"zone1_dropout": 0.0, "zone2_dropout": 0.1, "zone3_dropout": 0.3}

    def get_message_passing_depth(self) -> int:
        """Return the number of active message-passing layers."""
        progress = self.current_epoch / max(self.total_epochs, 1)
        if progress < 0.3:
            return min(4, self.max_layers)
        elif progress < 0.6:
            return min(7, self.max_layers)
        else:
            return self.max_layers

    def get_active_tasks(self) -> Dict[str, bool]:
        """Task activation follows the same phases as zone expansion."""
        progress = self.current_epoch / max(self.total_epochs, 1)
        if progress < 0.3:
            return {
                "ec": True, "selectivity": False, "kcat": False, "km": False,
                "mutation": False, "residue_importance": True,
            }
        elif progress < 0.6:
            return {
                "ec": True, "selectivity": True, "kcat": True, "km": True,
                "mutation": False, "residue_importance": True,
            }
        else:
            return {
                "ec": True, "selectivity": True, "kcat": True, "km": True,
                "mutation": True, "residue_importance": True,
            }

    def step(self) -> None:
        self.current_epoch += 1


def apply_zone_dropout(
    batch: Dict[str, Any],
    zone_dropout_rates: Dict[str, float],
) -> Dict[str, Any]:
    """Apply stochastic zone-level dropout to node features.

    Nodes in dropped zones have their features zeroed, simulating
    absence of that structural context.
    """
    pg_key = "protein_graph"
    if pg_key not in batch:
        return batch

    pg = batch[pg_key]
    zone_assigns = pg.get("zone_assignments")
    if zone_assigns is None:
        return batch

    node_features = pg["node_features"]
    if not node_features.requires_grad:
        node_features = node_features.clone()
        pg["node_features"] = node_features

    for zone_id, key in enumerate(
        ["zone1_dropout", "zone2_dropout", "zone3_dropout"]
    ):
        rate = zone_dropout_rates.get(key, 0.0)
        if rate <= 0.0:
            continue
        zone_mask = zone_assigns == zone_id
        if not zone_mask.any():
            continue
        n_zone = int(zone_mask.sum().item())
        keep = torch.rand(n_zone, device=node_features.device) > rate
        node_features[zone_mask] *= keep.unsqueeze(-1).float()

    return batch


# ── EMA model for validation ─────────────────────────────────────────────────

class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap model params with EMA params; return originals for restore."""
        originals = {}
        for name, param in model.named_parameters():
            originals[name] = param.clone()
            param.data.copy_(self.shadow[name])
        return originals

    def restore(self, model: nn.Module, originals: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(originals[name])


# ── Trainer ───────────────────────────────────────────────────────────────────

class ToPETrainer:
    """Training loop for the ToPE model with curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: Any,
        val_loader: Any,
        cfg: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        val_loaders_by_difficulty: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loaders = val_loaders_by_difficulty or {}
        self.cfg = cfg or TrainerConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Optimiser (include loss_fn params for uncertainty weights)
        self.optimiser = AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )

        # Scheduler
        if self.cfg.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimiser,
                max_lr=self.cfg.lr,
                epochs=self.cfg.n_epochs,
                steps_per_epoch=len(self.train_loader),
            )
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimiser,
                T_0=self.cfg.n_epochs,
                eta_min=self.cfg.min_lr,
            )

        self.augmenter = DataAugmenter(self.cfg)
        self.ema = ExponentialMovingAverage(self.model)

        # Tracking
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: List[Dict[str, float]] = []

        # Checkpoint directory
        self.ckpt_dir = Path(self.cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        active_tasks = get_active_tasks(epoch, self.cfg)
        running_losses: Dict[str, float] = {
            "total": 0.0, "ec": 0.0, "selectivity": 0.0, "kinetics": 0.0
        }
        n_batches = 0

        self.optimiser.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._to_device(batch)

            # Augment
            if "enzyme_pcc" in batch:
                batch["enzyme_pcc"] = self.augmenter(batch["enzyme_pcc"])

            # Forward
            predictions = self.model(batch)
            loss_dict = self.loss_fn(
                predictions, batch.get("targets", {}), active_tasks
            )

            loss = loss_dict["total"] / self.cfg.accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.cfg.accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimiser.step()
                self.optimiser.zero_grad()

                # EMA update
                self.ema.update(self.model)

            # Step-level scheduler for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Accumulate losses
            for key in running_losses:
                if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                    running_losses[key] += loss_dict[key].item()
            n_batches += 1

            if (batch_idx + 1) % self.cfg.log_interval == 0:
                avg_total = running_losses["total"] / n_batches
                logger.info(
                    f"  Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"loss={avg_total:.4f} tasks={active_tasks} "
                    f"weights={loss_dict.get('weights', {})}"
                )

        # Epoch-level scheduler for Cosine
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step(epoch)

        return {k: v / max(n_batches, 1) for k, v in running_losses.items()}

    @torch.no_grad()
    def _validate(
        self,
        loader: Any,
        epoch: int,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """Run validation with optional EMA parameters."""
        active_tasks = get_active_tasks(epoch, self.cfg)

        if use_ema:
            originals = self.ema.apply(self.model)

        self.model.eval()
        running_losses: Dict[str, float] = {
            "total": 0.0, "ec": 0.0, "selectivity": 0.0, "kinetics": 0.0
        }
        n_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            predictions = self.model(batch)
            loss_dict = self.loss_fn(
                predictions, batch.get("targets", {}), active_tasks
            )

            for key in running_losses:
                if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                    running_losses[key] += loss_dict[key].item()
            n_batches += 1

        if use_ema:
            self.ema.restore(self.model, originals)

        return {k: v / max(n_batches, 1) for k, v in running_losses.items()}

    def fit(self, n_epochs: Optional[int] = None) -> List[Dict[str, float]]:
        """Full training loop with curriculum learning and early stopping."""
        n_epochs = n_epochs or self.cfg.n_epochs

        logger.info(
            f"Starting training: {n_epochs} epochs, "
            f"curriculum phases at {self.cfg.phase_a_end}/{self.cfg.phase_b_end}"
        )

        for epoch in range(n_epochs):
            active_tasks = get_active_tasks(epoch, self.cfg)

            # Log phase transitions
            if epoch == 0:
                logger.info("Phase A: EC classification only")
            elif epoch == self.cfg.phase_a_end:
                logger.info("Phase B: Activating selectivity head")
            elif epoch == self.cfg.phase_b_end:
                logger.info("Phase C: Activating kinetics head — all tasks active")

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(self.val_loader, epoch)

            # Validate at multiple difficulty levels
            difficulty_metrics = {}
            for name, loader in self.val_loaders.items():
                difficulty_metrics[name] = self._validate(loader, epoch)

            # Record
            record = {
                "epoch": epoch,
                "active_tasks": active_tasks,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            for name, metrics in difficulty_metrics.items():
                record.update({f"val_{name}_{k}": v for k, v in metrics.items()})
            self.history.append(record)

            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['total']:.4f} "
                f"val_loss={val_metrics['total']:.4f} "
                f"tasks={active_tasks}"
            )

            # Early stopping check
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.cfg.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.cfg.patience})"
                    )
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

        return self.history

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss_fn_state_dict": self.loss_fn.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ema_shadow": self.ema.shadow,
            "history": self.history,
        }
        path = self.ckpt_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"New best model saved: {best_path}")

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the epoch to resume from."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.loss_fn.load_state_dict(state["loss_fn_state_dict"])
        self.optimiser.load_state_dict(state["optimiser_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.best_val_loss = state["best_val_loss"]
        self.ema.shadow = state["ema_shadow"]
        self.history = state.get("history", [])
        logger.info(f"Loaded checkpoint from epoch {state['epoch']}")
        return state["epoch"] + 1

    def _to_device(self, batch: Any) -> Any:
        """Recursively move batch tensors to the target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        return batch
