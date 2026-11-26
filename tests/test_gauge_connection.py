"""Gauge invariance, MPS preservation, holonomy and trigger tests."""
import math

import pytest
import torch

from tope.quantum.gauge_connection import (
    GaugeDiagnostics,
    GaugeTier,
    GaugeTriggerPolicy,
    align_global_phase,
    align_mps,
    align_orbitals,
    holonomy_deviation,
    mps_overlap,
    orbital_overlap,
    procrustes_transport,
    wilson_loop,
)


def _unitary(size, *, complex=False):
    dtype = torch.complex128 if complex else torch.float64
    matrix = torch.randn(size, size, dtype=dtype)
    q, _ = torch.linalg.qr(matrix)
    return q


@pytest.mark.parametrize("complex_frame", [False, True])
def test_orbital_transport_removes_phases_permutations_and_rotations(complex_frame):
    torch.manual_seed(2)
    frame, _ = torch.linalg.qr(torch.randn(
        8, 3, dtype=torch.complex128 if complex_frame else torch.float64
    ))
    rotation = _unitary(3, complex=complex_frame)
    current = frame @ rotation
    result = align_orbitals(frame, current)
    torch.testing.assert_close(result.aligned, frame, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result.singular_values, torch.ones(3, dtype=torch.float64))
    assert result.residual_norm < 1e-10


def test_nonorthogonal_metric_and_global_phase_alignment():
    torch.manual_seed(3)
    raw = torch.randn(6, 2, dtype=torch.double)
    metric = torch.diag(torch.linspace(1.0, 2.0, 6, dtype=torch.double))
    gram = raw.T @ metric @ raw
    inverse_root = torch.linalg.inv(torch.linalg.cholesky(gram)).T
    frame = raw @ inverse_root
    current = frame[:, [1, 0]] * torch.tensor([-1.0, 1.0])
    aligned = align_orbitals(frame, current, metric)
    torch.testing.assert_close(aligned.aligned, frame, atol=1e-10, rtol=1e-10)

    state = torch.randn(10, dtype=torch.complex128)
    shifted = state * torch.exp(torch.tensor(0.7j, dtype=torch.complex128))
    torch.testing.assert_close(align_global_phase(state, shifted), state)


def _mps():
    return (
        torch.randn(1, 2, 2, dtype=torch.double),
        torch.randn(2, 2, 2, dtype=torch.double),
        torch.randn(2, 2, 1, dtype=torch.double),
    )


def test_mps_bond_transport_preserves_state_and_aligns_virtual_gauge():
    torch.manual_seed(4)
    previous = _mps()
    g1, g2 = _unitary(2), _unitary(2)
    current = (
        torch.einsum("lpr,rs->lps", previous[0], g1),
        torch.einsum("ab,bpc,cd->apd", g1.T, previous[1], g2),
        torch.einsum("ab,bpc->apc", g2.T, previous[2]),
    )
    before_norm = mps_overlap(current, current)
    result = align_mps(previous, current)
    after_norm = mps_overlap(result.aligned, result.aligned)
    torch.testing.assert_close(after_norm, before_norm)
    torch.testing.assert_close(result.state_overlap.imag if result.state_overlap.is_complex()
                               else torch.zeros_like(result.state_overlap),
                               torch.zeros_like(result.state_overlap))
    assert result.state_overlap > 0
    for reference, aligned in zip(result.reference[:-1], result.aligned[:-1]):
        torch.testing.assert_close(reference, aligned, atol=1e-10, rtol=1e-10)


def test_holonomy_detects_nontrivial_loop_and_is_conjugation_invariant():
    angle = 0.3
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=torch.double)
    identity = torch.eye(2, dtype=torch.double)
    transports = [identity, rotation, identity, identity]
    assert not torch.allclose(wilson_loop(transports), identity)
    curvature = holonomy_deviation(transports, area=0.5)
    gauge = _unitary(2)
    transformed = [gauge.T @ transport @ gauge for transport in transports]
    torch.testing.assert_close(
        holonomy_deviation(transformed, area=0.5), curvature
    )


def test_trigger_policy_gates_curvature_and_requires_many_body_evidence_for_growth():
    policy = GaugeTriggerPolicy()
    assert not policy.should_check_curvature(0.99)
    assert policy.should_check_curvature(0.9)

    singular_only = policy.decide(GaugeDiagnostics(0.7, 0.5, curvature=0.2))
    assert singular_only.tier == GaugeTier.REFINE
    assert not singular_only.increase_bond_dimension

    corroborated = policy.decide(GaugeDiagnostics(
        0.7, 0.5, curvature=0.2, truncation_weight=1e-3
    ))
    assert corroborated.increase_bond_dimension
    assert policy.proposed_bond_dimension(10, 32, corroborated) == 15

    severe = policy.decide(GaugeDiagnostics(0.4, 1.0))
    assert severe.tier == GaugeTier.REFERENCE
    assert severe.reference_calculation and severe.reduce_timestep and severe.update_replay


def test_invalid_overlap_and_loop_inputs_fail_loudly():
    with pytest.raises(ValueError, match="square"):
        procrustes_transport(torch.ones(2, 3))
    with pytest.raises(ValueError, match="at least one"):
        wilson_loop([])
    with pytest.raises(ValueError, match="area"):
        holonomy_deviation([torch.eye(2)], area=0)
    with pytest.raises(ValueError, match="non-negative"):
        GaugeDiagnostics(0.9, -1.0)
