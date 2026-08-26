"""
ToPE Models
===========
Neural network architectures for enzyme function prediction.

Training model (Phase 2+):
    - CompleteToPEModel: whole-protein encoder, sole training model

Ablation baseline (deprecated, not for training):
    - ToPEModel: 8 Å active-site crop — Phase 2 ablation reference only

Other components:
    - TCPNet: Topology-Complete Perceptron Network message passing
    - Multi-subunit: Quaternary structure with cooperativity prediction
    - p-Laplacian: Learnable nonlinear diffusion for mechanistic analysis
    - Memory-optimized: TTN + gradient checkpointing for consumer GPUs
"""

# ── Training model ────────────────────────────────────────────────────────────
from tope.models.tope_model import (
    CompleteToPEModel,
    CompleteToPEConfig,
)

# ── Ablation baseline (deprecated — Phase 2 F-score reference only) ──────────
# Do NOT import ToPEModel/ToPEConfig in training scripts or inference paths.
from tope.models.tope_model import (
    ToPEModel,
    ToPEConfig,
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
from tope.models.domain_discovery import (
    DomainDiscoveryConfig,
    DomainDiscoveryHead,
    DomainSubstrateAttention,
    permutation_match,
    soft_domain_pool,
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
from tope.models.cc_attention import (
    CCAttentionPushForward,
    CCAttentionBlock,
    AttentionMergeNode,
    build_zone_adjacency,
    FermionicCCANOConfig,
    FermionicCCAttentionNeuralOperator,
    FermionicCombinatorialComplexAttentionNeuralOperator,
    cell_intersection_size,
    exterior_permutation_sign,
    gyrobarycentric_aggregate,
    wedge_pair,
)
from tope.models.tope_residual import (
    IntraRankResidualBlock,
    InterRankResidualMergeNode,
    SpectralResidualTTN,
    ToPERankStack,
)
from tope.models.bidirectional_physics import (
    BidirectionalPhysicsConfig,
    BidirectionalPhysicsToPE,
    DiagonalGaussian,
    MechanisticHeads,
    NODE_TYPES,
    EDGE_TYPES,
    PHENOTYPE_FIELDS,
    PhenotypeEncoder,
    PhysicsIntegrator,
    TemperatureEncoding,
    TopologyDecoder,
    TypedTopologyEncoder,
    mutation_epistasis,
)

__all__ = [
    # Bidirectional physics-informed latent model
    "BidirectionalPhysicsConfig",
    "BidirectionalPhysicsToPE",
    "DiagonalGaussian",
    "MechanisticHeads",
    "NODE_TYPES",
    "EDGE_TYPES",
    "PHENOTYPE_FIELDS",
    "PhenotypeEncoder",
    "PhysicsIntegrator",
    "TemperatureEncoding",
    "TopologyDecoder",
    "TypedTopologyEncoder",
    "mutation_epistasis",
    # Training model (Phase 2+)
    "CompleteToPEModel",
    "CompleteToPEConfig",
    # Ablation baseline — deprecated, not for training
    "ToPEModel",
    "ToPEConfig",
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
    # Self-supervised latent domains
    "DomainDiscoveryConfig",
    "DomainDiscoveryHead",
    "DomainSubstrateAttention",
    "permutation_match",
    "soft_domain_pool",
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
    # CCANN primitives
    "CCAttentionPushForward",
    "CCAttentionBlock",
    "AttentionMergeNode",
    "build_zone_adjacency",
    "FermionicCCANOConfig",
    "FermionicCCAttentionNeuralOperator",
    "FermionicCombinatorialComplexAttentionNeuralOperator",
    "cell_intersection_size",
    "exterior_permutation_sign",
    "gyrobarycentric_aggregate",
    "wedge_pair",
    # Deep residual learning
    "IntraRankResidualBlock",
    "InterRankResidualMergeNode",
    "SpectralResidualTTN",
    "ToPERankStack",
]
