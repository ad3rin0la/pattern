"""
ToPE Model — Phase 3: Full Topological Deep Learning Architecture
=================================================================

Integrates Phase 2 persistent spectral features into a complete
SE(3)-equivariant model for multi-task enzyme function prediction.

Two model variants:

    **ToPEModel** (active-site only):
        EnzymeTCPNet over 8 Å Enzyme-PCC + cross-attention + multi-task heads.

    **CompleteToPEModel** (whole-protein context):
        WholeProteinTCPNet over three-zone multi-scale graph + enhanced
        task heads (residue-level kinetics, mutation effect prediction).

Components:
    tcpnet                — TCPNet-style message passing over Enzyme-PCC
    whole_protein_tcpnet  — Multi-scale message passing over whole protein
    multi_scale_graph     — Three-zone protein graph construction
    cross_attention       — Substrate-product bipartite cross-attention
    task_heads            — EC, selectivity, kinetics, mutation heads
    losses                — Multi-task loss with uncertainty weighting
    trainer               — Training loop with curriculum learning
    evaluation            — Metrics, attribution, mutation validation

Usage:
    from tope_model import ToPEModel, ToPETrainer

    model = ToPEModel(config)
    trainer = ToPETrainer(model, train_loader, val_loader)
    trainer.fit(n_epochs=100)

    # Whole-protein mode:
    from tope_model import CompleteToPEModel, CompleteToPEConfig

    model = CompleteToPEModel(CompleteToPEConfig())
"""

from tope_model.tope_model import (
    ToPEModel,
    ToPEConfig,
    CompleteToPEModel,
    CompleteToPEConfig,
)
from tope_model.trainer import ToPETrainer

__all__ = [
    "ToPEModel",
    "ToPEConfig",
    "CompleteToPEModel",
    "CompleteToPEConfig",
    "ToPETrainer",
]
