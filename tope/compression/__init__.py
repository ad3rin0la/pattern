"""
Compression modules for ToPE enzyme feature pyramids.

STATUS — DEFERRED (Phases 5B–5G)
---------------------------------
All modules in this package depend on geometry-dependent VOIP tensors that
Phase 5A (AttentiveVOIPEncoder / VOIPSIRENField) has not yet shipped and
validated.  They are available for review and unit-testing but must NOT be
integrated into the training pipeline until Phase 5A ablations pass.

Deferral checklist:
    [ ] Phase 5A AttentiveVOIPEncoder validated on 5 enzymes
    [ ] VOIP geometry-dependent tensors confirmed numerically stable
    [ ] Phase 2 ablation suite passes (tope.training.evaluation.Phase2AblationSuite)
    Then: lift deferral, integrate TLSCompressor → Tucker ranks → GyroQJL
"""

import warnings as _warnings

# Suppress the per-module FutureWarning cascade — the package-level docstring
# above is the single point of record.  Individual modules still warn on
# direct import outside this package.
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", FutureWarning)

    from tope.compression.gyro_qjl import (
        PolarQuant,
        QJLSketch,
        GyroQJL,
        TangentSpaceQJL,
    )
    from tope.compression.tls_compression import (
        TLSGroup,
        TLSCompressor,
        TLSQuantizer,
    )

__all__ = [
    "PolarQuant",
    "QJLSketch",
    "GyroQJL",
    "TangentSpaceQJL",
    "TLSGroup",
    "TLSCompressor",
    "TLSQuantizer",
]
