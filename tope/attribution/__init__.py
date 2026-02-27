"""
ToPE Attribution
================
Multi-scale attribution analysis, OOD validation, and variant design.

Phase 4 interpretability tools for understanding model predictions
at filtration, zone, residue, and pathway scales.
"""

from tope.attribution.attribution import (
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
    "MultiScaleAttributionAnalyzer",
    "AttributionConfig",
    "ExtendedIoffeRecognition",
    "StratifiedOODValidator",
    "StratifiedOODConfig",
    "ExperimentalVariantDesigner",
    "VariantDesignConfig",
    "KnownAllostericValidator",
    "AllostericMutantEntry",
    "AttributionVisualizer",
    "validate_attribution_accuracy",
    "run_full_phase4_validation",
]
