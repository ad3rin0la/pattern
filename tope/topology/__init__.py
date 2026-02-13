"""
ToPE Topology
=============
Topological encoding modules for enzyme structures.

Includes:
    - TTN persistent homology (GPU-native tensor tree networks)
    - Phonon topology (Hodge Laplacian ENM, localization landscape)
    - Electron/proton transfer pathway prediction
"""

from tope.topology.ttn_persistent_homology import (
    TTNPHConfig,
    TTNNode,
    MultiParameterFiltration,
    MultiParameterTTN,
    TriParameterConfig,
    TriParameterFiltration,
    TriParameterTTN,
)
from tope.topology.phonon_topology import (
    compute_localization_landscape,
    identify_thermal_hotspots,
    HodgeLaplacianENM,
    SheafENM,
    build_cofactor_3cells,
    PhononTopologyFeatures,
)
from tope.topology.transfer_pathways import (
    PathwayType,
    TransferPathwayConfig,
    TransferPathwayHead,
    TransferPathwayLoss,
    TransferPathwayVisualizer,
    extract_transfer_pathway_graph,
)

__all__ = [
    # TTN persistent homology
    "TTNPHConfig",
    "TTNNode",
    "MultiParameterFiltration",
    "MultiParameterTTN",
    # Tri-parameter filtration
    "TriParameterConfig",
    "TriParameterFiltration",
    "TriParameterTTN",
    # Phonon topology
    "compute_localization_landscape",
    "identify_thermal_hotspots",
    "HodgeLaplacianENM",
    "SheafENM",
    "build_cofactor_3cells",
    "PhononTopologyFeatures",
    # Transfer pathways
    "PathwayType",
    "TransferPathwayConfig",
    "TransferPathwayHead",
    "TransferPathwayLoss",
    "TransferPathwayVisualizer",
    "extract_transfer_pathway_graph",
]
