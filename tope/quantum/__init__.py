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
]
