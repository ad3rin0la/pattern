"""Compatibility imports for the fermionic CCANO upgrade.

The implementation lives in :mod:`tope.models.cc_attention` because it extends
the existing CC attention formalism.
"""

from tope.models.cc_attention import (
    FermionicCCANOConfig,
    FermionicCCAttentionNeuralOperator,
    FermionicCombinatorialComplexAttentionNeuralOperator,
    cell_intersection_size,
    exterior_permutation_sign,
    gyrobarycentric_aggregate,
    mobius_scalar_mul,
    poincare_exp0,
    poincare_log0,
    project_to_poincare_ball,
    wedge_pair,
)

__all__ = [
    "FermionicCCANOConfig",
    "FermionicCCAttentionNeuralOperator",
    "FermionicCombinatorialComplexAttentionNeuralOperator",
    "cell_intersection_size",
    "exterior_permutation_sign",
    "gyrobarycentric_aggregate",
    "mobius_scalar_mul",
    "poincare_exp0",
    "poincare_log0",
    "project_to_poincare_ball",
    "wedge_pair",
]
