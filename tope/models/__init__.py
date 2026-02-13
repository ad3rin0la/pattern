"""
ToPE Models
===========
Neural network architectures for enzyme function prediction.

Includes:
    - TCPNet: Topology-Complete Perceptron Network message passing
    - ToPEModel: Active-site and whole-protein model variants
    - Multi-subunit: Quaternary structure with cooperativity prediction
    - p-Laplacian: Learnable nonlinear diffusion for mechanistic analysis
    - Memory-optimized: TTN + gradient checkpointing for consumer GPUs
"""

from tope.models.tope_model import (
    ToPEModel,
    ToPEConfig,
    CompleteToPEModel,
    CompleteToPEConfig,
)
from tope.models.tcpnet import (
    TCPNetConfig,
    GaussianRBF,
    ScalarEdgeMessage,
    FaceMessagePassing,
    TCPNetLayer,
    GlobalAttentionPooling,
    EnzymeTCPNet,
)
from tope.models.whole_protein_tcpnet import (
    WholeProteinConfig,
    WholeProteinTCPNet,
)
from tope.models.multi_scale_graph import (
    MultiScaleGraphConfig,
    MultiScaleProteinGraph,
)
from tope.models.cross_attention import (
    CrossAttentionConfig,
    MoleculeGNN,
    MultiHeadCrossAttention,
    SubstrateProductCrossAttention,
)
from tope.models.task_heads import (
    TaskHeadsConfig,
    ECClassificationHead,
    SelectivityHead,
    KineticsHead,
    MultiTaskHeads,
    EnhancedKineticEfficiencyHead,
    DistantMutationEffectPredictor,
    WholeProteinTaskHeads,
)
from tope.models.multi_subunit import (
    # PCC Construction
    MultiSubunitPCCBuilder,
    MultiSubunitPCCConfig,
    SubunitCell,
    InterfaceCell,
    # Laplacian
    CrossSubunitLaplacian,
    # Features
    AllostericFeatureExtractor,
    AllostericSpectralFeatures,
    # Message Passing
    MultiSubunitTCPNet,
    MultiSubunitTCPNetLayer,
    SubunitMessagePassing,
    InterfaceMessagePassing,
    # Task Heads
    AllostericCooperativityHead,
    # Complete Model
    MultiSubunitToPEModel,
    MultiSubunitToPEConfig,
    # Utilities
    build_multisubunit_enzyme_pcc,
    extract_allosteric_features_from_pcc,
)
from tope.models.p_laplacian import (
    # Core p-Laplacian
    LearnablePLaplacianToPE,
    PLaplacianConfig,
    PLaplacianEigensolver,
    # Nodal Domains
    NodalDomainExtractor,
    ReactionChannel,
    # Training
    PToPELoss,
    # Interpretability
    MechanisticAnalyzer,
    MechanisticInterpretation,
    # Experimental Predictions
    ExperimentalPredictor,
    # Complete Model
    CompletePToPEModel,
    CompletePToPEConfig,
    # Phonon-Aware
    PhononAwarePLaplacian,
)
from tope.models.memory_optimized import (
    TTNConfig,
    TTNPersistentEncoder,
    CheckpointedTCPNetConfig,
    CheckpointedTCPNet,
    MemoryOptimizedConfig,
    MemoryOptimizedToPE,
    MemoryOptimizedTrainer,
    profile_memory_usage,
    estimate_max_batch_size,
)

__all__ = [
    # Core models
    "ToPEModel",
    "ToPEConfig",
    "CompleteToPEModel",
    "CompleteToPEConfig",
    # TCPNet
    "TCPNetConfig",
    "GaussianRBF",
    "ScalarEdgeMessage",
    "FaceMessagePassing",
    "TCPNetLayer",
    "GlobalAttentionPooling",
    "EnzymeTCPNet",
    # Whole protein
    "WholeProteinConfig",
    "WholeProteinTCPNet",
    # Multi-scale graph
    "MultiScaleGraphConfig",
    "MultiScaleProteinGraph",
    # Cross attention
    "CrossAttentionConfig",
    "MoleculeGNN",
    "MultiHeadCrossAttention",
    "SubstrateProductCrossAttention",
    # Task heads
    "TaskHeadsConfig",
    "ECClassificationHead",
    "SelectivityHead",
    "KineticsHead",
    "MultiTaskHeads",
    "EnhancedKineticEfficiencyHead",
    "DistantMutationEffectPredictor",
    "WholeProteinTaskHeads",
    # Multi-subunit
    "MultiSubunitPCCBuilder",
    "MultiSubunitPCCConfig",
    "SubunitCell",
    "InterfaceCell",
    "CrossSubunitLaplacian",
    "AllostericFeatureExtractor",
    "AllostericSpectralFeatures",
    "MultiSubunitTCPNet",
    "MultiSubunitTCPNetLayer",
    "SubunitMessagePassing",
    "InterfaceMessagePassing",
    "AllostericCooperativityHead",
    "MultiSubunitToPEModel",
    "MultiSubunitToPEConfig",
    "build_multisubunit_enzyme_pcc",
    "extract_allosteric_features_from_pcc",
    # p-Laplacian
    "LearnablePLaplacianToPE",
    "PLaplacianConfig",
    "PLaplacianEigensolver",
    "NodalDomainExtractor",
    "ReactionChannel",
    "PToPELoss",
    "MechanisticAnalyzer",
    "MechanisticInterpretation",
    "ExperimentalPredictor",
    "CompletePToPEModel",
    "CompletePToPEConfig",
    "PhononAwarePLaplacian",
    # Memory-optimized
    "TTNConfig",
    "TTNPersistentEncoder",
    "CheckpointedTCPNetConfig",
    "CheckpointedTCPNet",
    "MemoryOptimizedConfig",
    "MemoryOptimizedToPE",
    "MemoryOptimizedTrainer",
    "profile_memory_usage",
    "estimate_max_batch_size",
]
