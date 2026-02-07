"""Test that all ToPE modules can be imported."""

import pytest


def test_import_tope_model():
    """Test importing main tope_model package."""
    import tope_model

    # Check key classes are exported
    assert hasattr(tope_model, 'ToPEModel')
    assert hasattr(tope_model, 'ToPEConfig')
    assert hasattr(tope_model, 'ToPETrainer')


def test_import_memory_optimized():
    """Test importing memory-optimized components."""
    from tope_model import (
        MemoryOptimizedToPE,
        MemoryOptimizedConfig,
        MemoryOptimizedTrainer,
        TTNConfig,
        TTNPersistentEncoder,
        CheckpointedTCPNet,
    )


def test_import_ttn():
    """Test importing TTN persistent homology."""
    from tope_model import (
        MultiParameterTTN,
        TTNPHConfig,
        MultiParameterFiltration,
        TTNNode,
    )


def test_import_multi_subunit():
    """Test importing multi-subunit components."""
    from tope_model import (
        MultiSubunitToPEModel,
        MultiSubunitToPEConfig,
        MultiSubunitPCCBuilder,
        AllostericCooperativityHead,
    )


def test_import_p_laplacian():
    """Test importing p-Laplacian components."""
    from tope_model import (
        CompletePToPEModel,
        LearnablePLaplacianToPE,
        MechanisticAnalyzer,
        NodalDomainExtractor,
    )


def test_import_attribution():
    """Test importing attribution components."""
    from tope_model import (
        MultiScaleAttributionAnalyzer,
        StratifiedOODValidator,
        AttributionVisualizer,
    )


def test_import_mcp_adapters():
    """Test importing MCP adapters."""
    from tope_model import (
        MCPAdapter,
        AlphaFoldMCPAdapter,
        ChEMBLMCPAdapter,
        PubChemMCPAdapter,
        MCPEnhancedPipeline,
    )


def test_import_data_curation():
    """Test importing data curation package."""
    import data_curation

    # Check key classes are exported
    assert hasattr(data_curation, 'CurationPipeline')
    assert hasattr(data_curation, 'CurationConfig')
