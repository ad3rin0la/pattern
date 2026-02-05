"""
ToPE Model — Phase 3+4: Full Topological Deep Learning Architecture
====================================================================

Integrates Phase 2 persistent spectral features into a complete
SE(3)-equivariant model for multi-task enzyme function prediction.

Two model variants:

    **ToPEModel** (active-site only):
        EnzymeTCPNet over 8 Å Enzyme-PCC + cross-attention + multi-task heads.

    **CompleteToPEModel** (whole-protein context):
        WholeProteinTCPNet over three-zone multi-scale graph + enhanced
        task heads (residue-level kinetics, mutation effect prediction).

Phase 4 Validation & Attribution:

    **MultiScaleAttributionAnalyzer**:
        Four-dimensional attribution (filtration, zone, residue, pathway).

    **StratifiedOODValidator**:
        2D validation matrix (sequence identity × mutation distance).

    **KnownAllostericValidator**:
        Recovery of literature-documented allosteric effects.

    **ExperimentalVariantDesigner**:
        Zone-stratified mutation design for experimental validation.

Components:
    tcpnet                — TCPNet-style message passing over Enzyme-PCC
    whole_protein_tcpnet  — Multi-scale message passing over whole protein
    multi_scale_graph     — Three-zone protein graph construction
    cross_attention       — Substrate-product bipartite cross-attention
    task_heads            — EC, selectivity, kinetics, mutation heads
    losses                — Multi-task loss with uncertainty weighting
    trainer               — Training loop with curriculum learning
    evaluation            — Metrics, attribution, mutation validation
    attribution           — Phase 4 multi-scale attribution & OOD validation

Usage:
    from tope_model import ToPEModel, ToPETrainer

    model = ToPEModel(config)
    trainer = ToPETrainer(model, train_loader, val_loader)
    trainer.fit(n_epochs=100)

    # Whole-protein mode:
    from tope_model import CompleteToPEModel, CompleteToPEConfig

    model = CompleteToPEModel(CompleteToPEConfig())

    # Phase 4 attribution:
    from tope_model import MultiScaleAttributionAnalyzer, StratifiedOODValidator

    analyzer = MultiScaleAttributionAnalyzer(model, device)
    result = analyzer.analyze(protein_graph, target_task="kinetics")
"""

from tope_model.tope_model import (
    ToPEModel,
    ToPEConfig,
    CompleteToPEModel,
    CompleteToPEConfig,
)
from tope_model.trainer import ToPETrainer
from tope_model.attribution import (
    MultiScaleAttributionAnalyzer,
    AttributionConfig,
    ExtendedIoffeRecognition,
    StratifiedOODValidator,
    StratifiedOODConfig,
    ExperimentalVariantDesigner,
    VariantDesignConfig,
    KnownAllostericValidator,
    AllostericMutantEntry,
    AttributionVisualizer,
    validate_attribution_accuracy,
    run_full_phase4_validation,
)

__all__ = [
    # Phase 3: Models
    "ToPEModel",
    "ToPEConfig",
    "CompleteToPEModel",
    "CompleteToPEConfig",
    "ToPETrainer",
    # Phase 4: Attribution
    "MultiScaleAttributionAnalyzer",
    "AttributionConfig",
    "ExtendedIoffeRecognition",
    # Phase 4: Validation
    "StratifiedOODValidator",
    "StratifiedOODConfig",
    "ExperimentalVariantDesigner",
    "VariantDesignConfig",
    "KnownAllostericValidator",
    "AllostericMutantEntry",
    # Phase 4: Visualization
    "AttributionVisualizer",
    # Phase 4: Utilities
    "validate_attribution_accuracy",
    "run_full_phase4_validation",
]
