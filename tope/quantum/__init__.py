"""
ToPE Quantum
============
Differentiable electronic-structure primitives for enzyme active-site encoding.

Phase 5A: Attentive VOIP-SIREN architecture that replaces the static NIST
elemental property table with a geometry-dependent, substrate-aware VOIP
field driven by GFN2-xTB descriptors.

Includes:
    - VOIPSIRENConfig:              Hyper-parameter dataclass
    - BandSIREN:                    Per-frequency-band SIREN network
    - XTBContextEncoder:            xTB scalars → latent context vector
    - BandAttention:                substrate-conditioned band selection
    - SpatialGlobalAttention:       log-distance-biased self-attention
    - VOIPSIRENField:               unified differentiable VOIP field
    - SubstrateVOIPCrossAttention:  substrate atoms attend to VOIP field
    - SpectralPosEncoding:          SIREN activations as pos. encodings
    - AttentiveVOIPEncoder:         full drop-in encoder (Phase 5A entry point)
    - VOIPSheafSectionBuilder:      outputs 8-dim Ioffe sections for SheafENM
"""

from tope.quantum.voip_siren import (
    VOIPSIRENConfig,
    BandSIREN,
    XTBContextEncoder,
    BandAttention,
    SpatialGlobalAttention,
    VOIPSIRENField,
    SubstrateVOIPCrossAttention,
    SpectralPosEncoding,
    AttentiveVOIPEncoder,
    VOIPSheafSectionBuilder,
)
from tope.quantum.gyro_dnc import GyroMemory
from tope.quantum.electronic_complex import (
    AdaptiveElectronicReadout,
    ElectronicCochain,
    ElectronicComplexConfig,
    ElectronicComplexEncoder,
    GeometryConditionedElectronicPool,
    GeometryElectronicFeedback,
    MultiresolutionElectronicFingerprint,
    dense_membership_to_incidence,
    hierarchical_candidate_indices,
    local_cell_frames,
)
from tope.quantum.relativistic_holography import (
    CLIFFORD_CHANNELS,
    CLIFFORD_GRADES,
    GYROTRIGONOMETRIC_CHANNELS,
    CliffordCommutatorDynamics,
    CliffordHologramLayer,
    CoupledBoostResult,
    ElectronicCliffordHologram,
    GyroInteractionResult,
    GyrotrigonometricFeatures,
    GyrotrigonometricInteractionKernel,
    clifford_anticommutator,
    clifford_commutator,
    clifford_product,
    clifford_structure_constants,
    clifford_to_matrix,
    composed_lorentz_gamma,
    coupled_boost,
    dirac_basis,
    dirac_hamiltonian,
    dirac_matrices,
    einstein_velocity_to_poincare,
    gyroangle,
    gyrocosine,
    gyrosine,
    gyrotrigonometric_features,
    half_rapidity_gamma,
    hyperbolic_volume_weights,
    mass_shell_four_vector,
    lorentz_gamma_from_poincare,
    matrix_to_clifford,
    mobius_add,
    mobius_gyration,
    mobius_gyration_matrix,
    momentum_to_poincare,
    poincare_gyrolength,
    poincare_to_einstein_velocity,
    poincare_to_momentum,
    project_clifford_hologram,
    spinor_boost,
)

__all__ = [
    "VOIPSIRENConfig",
    "BandSIREN",
    "XTBContextEncoder",
    "BandAttention",
    "SpatialGlobalAttention",
    "VOIPSIRENField",
    "SubstrateVOIPCrossAttention",
    "SpectralPosEncoding",
    "AttentiveVOIPEncoder",
    "VOIPSheafSectionBuilder",
    # Phase 5G
    "GyroMemory",
    # Geometry-aware multirank electronic cochains
    "ElectronicComplexConfig",
    "ElectronicCochain",
    "MultiresolutionElectronicFingerprint",
    "GeometryConditionedElectronicPool",
    "ElectronicComplexEncoder",
    "AdaptiveElectronicReadout",
    "GeometryElectronicFeedback",
    "dense_membership_to_incidence",
    "hierarchical_candidate_indices",
    "local_cell_frames",
    # Relativistic mass-shell / Clifford holography
    "CLIFFORD_CHANNELS",
    "CLIFFORD_GRADES",
    "GYROTRIGONOMETRIC_CHANNELS",
    "CoupledBoostResult",
    "GyroInteractionResult",
    "GyrotrigonometricFeatures",
    "CliffordCommutatorDynamics",
    "CliffordHologramLayer",
    "ElectronicCliffordHologram",
    "GyrotrigonometricInteractionKernel",
    "clifford_anticommutator",
    "clifford_commutator",
    "clifford_product",
    "clifford_structure_constants",
    "clifford_to_matrix",
    "composed_lorentz_gamma",
    "coupled_boost",
    "dirac_basis",
    "dirac_hamiltonian",
    "dirac_matrices",
    "einstein_velocity_to_poincare",
    "gyroangle",
    "gyrocosine",
    "gyrosine",
    "gyrotrigonometric_features",
    "half_rapidity_gamma",
    "hyperbolic_volume_weights",
    "mass_shell_four_vector",
    "lorentz_gamma_from_poincare",
    "matrix_to_clifford",
    "mobius_add",
    "mobius_gyration",
    "mobius_gyration_matrix",
    "momentum_to_poincare",
    "poincare_gyrolength",
    "poincare_to_einstein_velocity",
    "poincare_to_momentum",
    "project_clifford_hologram",
    "spinor_boost",
]
