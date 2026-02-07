"""
ToPE Model — Phase 3-6: Full Topological Deep Learning Architecture
===================================================================

Integrates Phase 2 persistent spectral features into a complete
SE(3)-equivariant model for multi-task enzyme function prediction.

Model Variants:

    **ToPEModel** (active-site only):
        EnzymeTCPNet over 8 Å Enzyme-PCC + cross-attention + multi-task heads.

    **CompleteToPEModel** (whole-protein context):
        WholeProteinTCPNet over three-zone multi-scale graph + enhanced
        task heads (residue-level kinetics, mutation effect prediction).

    **MultiSubunitToPEModel** (quaternary structure):
        4-level hierarchy (atoms → bonds → residues → subunits → interfaces)
        with cross-subunit message passing and cooperativity prediction.

    **CompletePToPEModel** (learnable p-Laplacian):
        Nonlinear diffusion with learnable p-parameters encoding mechanistic
        regimes across topological scales. Connects to Eyring/Arrhenius theory.

Phase 4 Validation & Attribution:

    **MultiScaleAttributionAnalyzer**:
        Four-dimensional attribution (filtration, zone, residue, pathway).

    **StratifiedOODValidator**:
        2D validation matrix (sequence identity × mutation distance).

Phase 5 Multi-Subunit Handling:

    **MultiSubunitPCCBuilder**:
        Constructs 4-level hierarchical complex with 3-cells and 4-cells.

    **AllostericCooperativityHead**:
        Predicts Hill coefficient for cooperative binding.

Phase 6 Learnable p-Laplacian:

    **LearnablePLaplacianToPE**:
        Multi-scale p-parameter grid with T_eff ∝ T/p interpretation.

    **NodalDomainExtractor**:
        Extract Eyring reaction channels from eigenmode nodal domains.

    **MechanisticAnalyzer**:
        Interpretability from learned p-landscape (rate-limiting steps).

    **ExperimentalPredictor**:
        Predictions for KIE, temperature dependence, pressure effects.

Memory-Optimized Architecture:

    **MemoryOptimizedToPE**:
        Combined architecture with TTN persistent homology (60,000x compression),
        gradient-checkpointed TCPNet (10x memory reduction), fits on RTX 3060.

    **TTNPersistentEncoder**:
        Tensor Train Network encoder for multi-parameter persistent homology.

    **CheckpointedTCPNet**:
        TCPNet with gradient checkpointing for memory-efficient training.

    **MemoryOptimizedTrainer**:
        Training with mixed precision (FP16), gradient accumulation, and profiling.

TTN Multi-Parameter Persistent Homology:

    **MultiParameterTTN**:
        GPU-native tensor tree network replacing CPU-bound gudhi persistent homology.
        Spatial (distance) + electronic (VOIP) multi-parameter filtration.

    **MultiParameterFiltration**:
        Computes spectral features across spatial zones and VOIP thresholds.

    **TTNNode**:
        Tree node with Tucker decomposition for tensor contraction.

Components:
    tcpnet                — TCPNet-style message passing over Enzyme-PCC
    whole_protein_tcpnet  — Multi-scale message passing over whole protein
    multi_scale_graph     — Three-zone protein graph construction
    multi_subunit         — Quaternary structure with 3/4-cells
    p_laplacian           — Learnable p-Laplacian for mechanistic interpretability
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

    # Multi-subunit mode (e.g., hemoglobin, ATP synthase):
    from tope_model import MultiSubunitToPEModel, MultiSubunitToPEConfig

    model = MultiSubunitToPEModel(MultiSubunitToPEConfig())
    preds = model(enzyme_pcc)
    hill_coeff = preds["hill_coefficient"]  # Cooperativity

    # Learnable p-Laplacian (mechanistic interpretability):
    from tope_model import CompletePToPEModel, MechanisticAnalyzer

    model = CompletePToPEModel()
    analysis = model.get_mechanistic_analysis(enzyme_pcc)
    print(analysis.interpretations)  # Rate-limiting step identification

    # Phase 4 attribution:
    from tope_model import MultiScaleAttributionAnalyzer, StratifiedOODValidator

    analyzer = MultiScaleAttributionAnalyzer(model, device)
    result = analyzer.analyze(protein_graph, target_task="kinetics")

    # Memory-optimized training (fits on RTX 3060 12GB):
    from tope_model import MemoryOptimizedToPE, MemoryOptimizedTrainer

    model = MemoryOptimizedToPE()  # 45x memory reduction
    trainer = MemoryOptimizedTrainer(model, train_loader)
    trainer.fit(n_epochs=100)  # Mixed precision + gradient accumulation
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
from tope_model.multi_subunit import (
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
from tope_model.p_laplacian import (
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
)
from tope_model.mcp_adapters import (
    # Base Adapter
    MCPAdapter,
    # AlphaFold
    AlphaFoldMCPAdapter,
    AlphaFoldStructure,
    # ChEMBL
    ChEMBLMCPAdapter,
    ChEMBLKineticsData,
    # PubChem
    PubChemMCPAdapter,
    # Pipeline
    MCPEnhancedPipeline,
)
from tope_model.memory_optimized import (
    # TTN Persistent Homology
    TTNConfig,
    TTNPersistentEncoder,
    # Checkpointed TCPNet
    CheckpointedTCPNetConfig,
    CheckpointedTCPNet,
    # Complete Model
    MemoryOptimizedConfig,
    MemoryOptimizedToPE,
    # Trainer
    MemoryOptimizedTrainer,
    # Utilities
    profile_memory_usage,
    estimate_max_batch_size,
)
from tope_model.ttn_persistent_homology import (
    # Configuration
    TTNPHConfig,
    # Tree Nodes
    TTNNode,
    # Filtration
    MultiParameterFiltration,
    # Complete TTN
    MultiParameterTTN,
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
    # Phase 5: Multi-Subunit PCC
    "MultiSubunitPCCBuilder",
    "MultiSubunitPCCConfig",
    "SubunitCell",
    "InterfaceCell",
    # Phase 5: Laplacian
    "CrossSubunitLaplacian",
    # Phase 5: Features
    "AllostericFeatureExtractor",
    "AllostericSpectralFeatures",
    # Phase 5: Message Passing
    "MultiSubunitTCPNet",
    "MultiSubunitTCPNetLayer",
    "SubunitMessagePassing",
    "InterfaceMessagePassing",
    # Phase 5: Task Heads
    "AllostericCooperativityHead",
    # Phase 5: Complete Model
    "MultiSubunitToPEModel",
    "MultiSubunitToPEConfig",
    # Phase 5: Utilities
    "build_multisubunit_enzyme_pcc",
    "extract_allosteric_features_from_pcc",
    # Phase 6: p-Laplacian Core
    "LearnablePLaplacianToPE",
    "PLaplacianConfig",
    "PLaplacianEigensolver",
    # Phase 6: Nodal Domains
    "NodalDomainExtractor",
    "ReactionChannel",
    # Phase 6: Training
    "PToPELoss",
    # Phase 6: Interpretability
    "MechanisticAnalyzer",
    "MechanisticInterpretation",
    # Phase 6: Experimental Predictions
    "ExperimentalPredictor",
    # Phase 6: Complete Model
    "CompletePToPEModel",
    "CompletePToPEConfig",
    # MCP Adapters
    "MCPAdapter",
    "AlphaFoldMCPAdapter",
    "AlphaFoldStructure",
    "ChEMBLMCPAdapter",
    "ChEMBLKineticsData",
    "PubChemMCPAdapter",
    "MCPEnhancedPipeline",
    # Memory-Optimized Architecture
    "TTNConfig",
    "TTNPersistentEncoder",
    "CheckpointedTCPNetConfig",
    "CheckpointedTCPNet",
    "MemoryOptimizedConfig",
    "MemoryOptimizedToPE",
    "MemoryOptimizedTrainer",
    "profile_memory_usage",
    "estimate_max_batch_size",
    # TTN Multi-Parameter Persistent Homology
    "TTNPHConfig",
    "TTNNode",
    "MultiParameterFiltration",
    "MultiParameterTTN",
]
