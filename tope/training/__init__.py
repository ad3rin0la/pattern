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
from tope.training.labels import (
    ECVocabulary,
    encode_targets,
    active_tasks_for,
)
from tope.training.bidirectional_losses import (
    BidirectionalLossConfig,
    BidirectionalPhysicsLoss,
    masked_gaussian_nll,
    symmetric_gaussian_kl,
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
    # Label encoding (curated labels → training targets)
    "ECVocabulary",
    "encode_targets",
    "active_tasks_for",
    "BidirectionalLossConfig",
    "BidirectionalPhysicsLoss",
    "masked_gaussian_nll",
    "symmetric_gaussian_kl",
    # Evaluation
    "MetricAccumulator",
    "FiltrationAttribution",
    "IoffeInverseRecognition",
    "LongRangeMutationValidator",
    "AllostericPathwayVisualiser",
]
