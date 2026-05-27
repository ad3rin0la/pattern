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
    sheaf_sections_from_voip_tensor,
)
from tope.topology.transfer_pathways import (
    PathwayType,
    TransferPathwayConfig,
    TransferPathwayHead,
    TransferPathwayLoss,
    TransferPathwayVisualizer,
    extract_transfer_pathway_graph,
)
from tope.topology.g_structure import (
    nijenhuis_norm_per_cell,
    build_restriction_maps,
    chern_invariant,
    frust_index_cb,
    voip_covariant_loss,
    almost_tangent_score,
    AlmostTangentDetector,
    nijenhuis_signal,
)
from tope.topology.hsh_restriction_maps import (
    HSHConfig,
    HSHSheafLaplacian,
    HSHRestrictionMap,
    compute_hsh_basis_matrices,
    hsh_coefficients,
    zonal_hsh_kernel,
    verify_addition_theorem,
    run_validation as run_hsh_validation,
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
    "sheaf_sections_from_voip_tensor",
    # Transfer pathways
    "PathwayType",
    "TransferPathwayConfig",
    "TransferPathwayHead",
    "TransferPathwayLoss",
    "TransferPathwayVisualizer",
    "extract_transfer_pathway_graph",
    # G-structure (Clark-Bruckheimer)
    "nijenhuis_norm_per_cell",
    "build_restriction_maps",
    "chern_invariant",
    "frust_index_cb",
    "voip_covariant_loss",
    "almost_tangent_score",
    "AlmostTangentDetector",
    "nijenhuis_signal",
    # HSH restriction maps (Phase 2 sheaf upgrade)
    "HSHConfig",
    "HSHSheafLaplacian",
    "HSHRestrictionMap",
    "compute_hsh_basis_matrices",
    "hsh_coefficients",
    "zonal_hsh_kernel",
    "verify_addition_theorem",
    "run_hsh_validation",
]
