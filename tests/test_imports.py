"""Test that all ToPE modules can be imported."""

import pytest


def test_import_tope_package():
    """Test importing top-level tope package."""
    import tope

    assert hasattr(tope, '__version__')
    assert hasattr(tope, 'ToPEModel')
    assert hasattr(tope, 'ToPEConfig')
    assert hasattr(tope, 'ToPETrainer')


def test_import_models():
    """Test importing model components."""
    from tope.models import (
        ToPEModel,
        ToPEConfig,
        CompleteToPEModel,
        CompleteToPEConfig,
        EnzymeTCPNet,
        TCPNetLayer,
    )


def test_import_memory_optimized():
    """Test importing memory-optimized components."""
    from tope.models import (
        MemoryOptimizedToPE,
        MemoryOptimizedConfig,
        MemoryOptimizedTrainer,
        TTNConfig,
        TTNPersistentEncoder,
        CheckpointedTCPNet,
    )


def test_import_ttn():
    """Test importing TTN persistent homology."""
    from tope.topology import (
        MultiParameterTTN,
        TTNPHConfig,
        MultiParameterFiltration,
        TTNNode,
    )


def test_import_multi_subunit():
    """Test importing multi-subunit components."""
    from tope.models import (
        MultiSubunitToPEModel,
        MultiSubunitToPEConfig,
        MultiSubunitPCCBuilder,
        AllostericCooperativityHead,
    )


def test_import_p_laplacian():
    """Test importing p-Laplacian components."""
    from tope.models import (
        CompletePToPEModel,
        LearnablePLaplacianToPE,
        MechanisticAnalyzer,
        NodalDomainExtractor,
    )


def test_import_attribution():
    """Test importing attribution components."""
    from tope.attribution import (
        MultiScaleAttributionAnalyzer,
        StratifiedOODValidator,
        AttributionVisualizer,
    )


def test_import_mcp_adapters():
    """Test importing MCP adapters."""
    from tope.utils import (
        MCPAdapter,
        AlphaFoldMCPAdapter,
        ChEMBLMCPAdapter,
        PubChemMCPAdapter,
        MCPEnhancedPipeline,
    )


def test_import_data():
    """Test importing data curation package."""
    from tope.data import (
        CurationPipeline,
        CurationConfig,
        MCSAClient,
        PDBClient,
        ActiveSiteExtractor,
    )


def test_import_training():
    """Test importing training components."""
    from tope.training import (
        ToPETrainer,
        MultiTaskLoss,
        EnhancedMultiTaskLoss,
        MetricAccumulator,
    )


def test_import_topology():
    """Test importing topology components."""
    from tope.topology import (
        HodgeLaplacianENM,
        SheafENM,
        TransferPathwayHead,
        TriParameterTTN,
    )


def test_import_phonon():
    """Test importing phonon topology components."""
    from tope.topology import (
        compute_localization_landscape,
        identify_thermal_hotspots,
        build_cofactor_3cells,
        PhononTopologyFeatures,
    )


def test_import_transfer_pathways():
    """Test importing transfer pathway components."""
    from tope.topology import (
        PathwayType,
        TransferPathwayConfig,
        TransferPathwayHead,
        TransferPathwayLoss,
        TransferPathwayVisualizer,
        extract_transfer_pathway_graph,
    )
