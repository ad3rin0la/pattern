"""
ToPE: Topological Pattern Recognition for Enzymes
==================================================

A computational biology framework for predicting enzyme function from
structure using persistent sheaf Laplacians and topological deep learning.

Subpackages:
    tope.data        - Data curation pipeline (M-CSA, PDB, BRENDA, TopEC)
    tope.topology    - Topological encoding (TTN, phonon topology, transfer pathways)
    tope.models      - Neural network architectures (TCPNet, multi-subunit, p-Laplacian)
    tope.training    - Training utilities, losses, and evaluation metrics
    tope.attribution - Multi-scale attribution and OOD validation
    tope.utils       - MCP adapters and shared utilities
    tope.quantum     - Differentiable electronic-structure primitives (Phase 5A)

Quick start::

    import tope

    # Load and train a model
    from tope.models import ToPEModel, ToPEConfig
    from tope.training import ToPETrainer

    model = ToPEModel(ToPEConfig())
    trainer = ToPETrainer(model, train_loader, val_loader)
    trainer.fit(n_epochs=100)
"""

from tope.__version__ import __version__

# Core models (most common imports)
from tope.models import (
    ToPEModel,
    ToPEConfig,
    CompleteToPEModel,
    CompleteToPEConfig,
    EnzymeTCPNet,
    TCPNetLayer,
    MultiSubunitToPEModel,
    CompletePToPEModel,
    MemoryOptimizedToPE,
)

# Training
from tope.training import (
    ToPETrainer,
    MultiTaskLoss,
    EnhancedMultiTaskLoss,
)

# Attribution
from tope.attribution import (
    MultiScaleAttributionAnalyzer,
    AttributionVisualizer,
)

# Data pipeline
from tope.data import (
    CurationPipeline,
    CurationConfig,
    MCSAClient,
    PDBClient,
)

# Topology
from tope.topology import (
    MultiParameterTTN,
    TriParameterTTN,
    HodgeLaplacianENM,
    TransferPathwayHead,
)

# Quantum (differentiable VOIP field — Phase 5A)
from tope.quantum import (
    AttentiveVOIPEncoder,
    VOIPSheafSectionBuilder,
    VOIPSIRENConfig,
)

__all__ = [
    "__version__",
    # Models
    "ToPEModel",
    "ToPEConfig",
    "CompleteToPEModel",
    "CompleteToPEConfig",
    "EnzymeTCPNet",
    "TCPNetLayer",
    "MultiSubunitToPEModel",
    "CompletePToPEModel",
    "MemoryOptimizedToPE",
    # Training
    "ToPETrainer",
    "MultiTaskLoss",
    "EnhancedMultiTaskLoss",
    # Attribution
    "MultiScaleAttributionAnalyzer",
    "AttributionVisualizer",
    # Data
    "CurationPipeline",
    "CurationConfig",
    "MCSAClient",
    "PDBClient",
    # Topology
    "MultiParameterTTN",
    "TriParameterTTN",
    "HodgeLaplacianENM",
    "TransferPathwayHead",
    # Quantum
    "AttentiveVOIPEncoder",
    "VOIPSheafSectionBuilder",
    "VOIPSIRENConfig",
]
