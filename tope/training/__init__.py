"""
ToPE Training
=============
Training utilities, loss functions, and evaluation metrics.

Includes:
    - Multi-task loss with uncertainty weighting
    - Curriculum learning trainer
    - Comprehensive evaluation metrics and validation
"""

from tope.training.trainer import ToPETrainer, TrainerConfig
from tope.training.losses import (
    ECHierarchyConsistencyLoss,
    MultiTaskLoss,
    EnhancedMultiTaskLoss,
)
from tope.training.evaluation import (
    MetricAccumulator,
    FiltrationAttribution,
    IoffeInverseRecognition,
    LongRangeMutationValidator,
    AllostericPathwayVisualiser,
)

__all__ = [
    # Trainer
    "ToPETrainer",
    "TrainerConfig",
    # Losses
    "ECHierarchyConsistencyLoss",
    "MultiTaskLoss",
    "EnhancedMultiTaskLoss",
    # Evaluation
    "MetricAccumulator",
    "FiltrationAttribution",
    "IoffeInverseRecognition",
    "LongRangeMutationValidator",
    "AllostericPathwayVisualiser",
]
