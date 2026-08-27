"""Tests for molecular Hodge, electronic, and coupled-torsion diffusion."""

import math

import pytest
import torch

from tope.models.hodge_diffusion import (
    CoupledTorsionHarmonics,
    ElectronicHodgeDiffusion,
    HodgeDiffusionConfig,
    HodgeHeatDiffusion,
    hodge_decomposition,
    hodge_laplacian,
    torus_heat_decay,
    validate_boundary_operators,
)
from tope.quantum.electronic_complex import (
    ElectronicComplexConfig,
    ElectronicComplexEncoder,
)


def _triangle_boundaries():
    # e01, e12, e20 and their consistently oriented triangular face.
    b1 = torch.tensor([
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ])
    b2 = torch.ones(3, 1)
    return [b1, b2]


def test_hodge_laplacians_are_psd_and_enforce_chain_condition():
    boundaries = _triangle_boundaries()
    validate_boundary_operators(boundaries)
    for rank, expected_size in enumerate([3, 3, 1]):
        laplacian = hodge_laplacian(boundaries, rank)
        assert laplacian.shape == (expected_size, expected_size)
        torch.testing.assert_close(laplacian, laplacian.T)
        assert torch.linalg.eigvalsh(laplacian).min() >= -1e-6

    invalid_b2 = torch.tensor([[1.0], [1.0], [-1.0]])
    with pytest.raises(ValueError, match="chain condition"):
        validate_boundary_operators([boundaries[0], invalid_b2])


def test_heat_semigroup_preserves_harmonic_modes_and_damps_rough_modes():
    heat = HodgeHeatDiffusion(HodgeDiffusionConfig(max_modes=None))
    spectrum = heat.spectra(_triangle_boundaries())[0]
    constant = torch.ones(3, 1)
    constant_result = heat.diffuse(constant, spectrum, torch.tensor([0.0, 5.0]))
    torch.testing.assert_close(
        constant_result.reconstruction[0], constant, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        constant_result.reconstruction[1], constant, atol=1e-5, rtol=1e-5
    )

    rough = torch.tensor([[1.0], [-1.0], [0.0]])
    rough_result = heat.diffuse(rough, spectrum, torch.tensor([0.0, 1.0, 5.0]))
    norms = rough_result.reconstruction.norm(dim=(1, 2))
    assert norms[0] > norms[1] > norms[2]


def test_hodge_decomposition_recovers_exact_and_harmonic_cycle_components():
    b1 = _triangle_boundaries()[0]
    exact = b1.T @ torch.tensor([1.0, 0.0, -1.0])
    harmonic = torch.ones(3)
    decomposition = hodge_decomposition(exact + harmonic, [b1], rank=1)
    torch.testing.assert_close(decomposition.exact, exact, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decomposition.coexact, torch.zeros(3))
    torch.testing.assert_close(decomposition.harmonic, harmonic, atol=1e-5, rtol=1e-5)
    reconstructed = decomposition.exact + decomposition.coexact + decomposition.harmonic
    torch.testing.assert_close(reconstructed, exact + harmonic)
    assert abs(torch.dot(decomposition.exact, decomposition.harmonic)) < 1e-5

    # Filling the triangle turns the cycle into a coexact face-boundary mode.
    coexact = 2.0 * harmonic
    filled = hodge_decomposition(exact + coexact, _triangle_boundaries(), rank=1)
    torch.testing.assert_close(filled.exact, exact, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(filled.coexact, coexact, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(filled.harmonic, torch.zeros(3), atol=1e-5, rtol=1e-5)


def test_complex_cochains_use_real_diffusion_times():
    heat = HodgeHeatDiffusion()
    spectrum = heat.spectra(_triangle_boundaries())[0]
    signal = torch.tensor([[1 + 2j], [-1j], [0.5 - 0.2j]])
    result = heat.diffuse(signal, spectrum, [0.0, 0.25])
    assert not result.times.is_complex()
    torch.testing.assert_close(result.reconstruction[0], signal, atol=1e-5, rtol=1e-5)
    assert result.reconstruction[1].norm() <= result.reconstruction[0].norm() + 1e-6


def test_electronic_fields_project_onto_aligned_hodge_ranks():
    torch.manual_seed(4)
    cfg = ElectronicComplexConfig(
        spectrum_dim=3, hidden_dim=8, max_ranks=3, dropout=0.0
    )
    encoder = ElectronicComplexEncoder(cfg)
    coordinates = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]
    ])
    scalar = torch.randn(3, 3)
    vector = torch.randn(3, 3, 3)
    charge = torch.tensor([0.2, -0.1, -0.1])
    atom_to_edge = torch.tensor([
        [0, 1, 1, 2, 2, 0], [0, 0, 1, 1, 2, 2]
    ])
    edge_to_face = torch.tensor([[0, 1, 2], [0, 0, 0]])
    fingerprint = encoder(
        coordinates, scalar, charge, [atom_to_edge, edge_to_face],
        vector_spectrum=vector,
        rank_names=["atom", "bond", "motif"],
    )
    results = ElectronicHodgeDiffusion()(fingerprint, _triangle_boundaries(), [0.0, 1.0])
    assert len(results) == 3
    assert [result.reconstruction.shape for result in results] == [
        (2, 3, 8), (2, 3, 8), (2, 1, 8)
    ]

    vector_results = ElectronicHodgeDiffusion()(
        fingerprint, _triangle_boundaries(), [0.0], field="vector"
    )
    for cochain, result in zip(fingerprint.cochains, vector_results):
        expected_global = torch.einsum(
            "cij,csj->csi", cochain.frames, cochain.vector
        ).flatten(start_dim=1)
        torch.testing.assert_close(
            result.reconstruction[0], expected_global, atol=1e-5, rtol=1e-5
        )


def test_sparse_coupled_torsions_capture_relative_phase_and_periodicity():
    torch.manual_seed(7)
    model = CoupledTorsionHarmonics(max_frequency=1, hidden_dim=6)
    incidence = torch.tensor([[0, 1], [0, 0]])
    aligned = model(torch.tensor([0.0, 0.0]), incidence)
    opposed = model(torch.tensor([0.0, math.pi]), incidence)
    periodic = model(torch.tensor([2.0 * math.pi, 0.0]), incidence)
    reordered = model(torch.tensor([0.0, 0.0]), incidence.flip(1))

    # Raw layout: independent cos/sin, difference cos/sin, sum cos/sin.
    torch.testing.assert_close(aligned.raw_features[..., 2], torch.ones(1, 1))
    torch.testing.assert_close(opposed.raw_features[..., 2], -torch.ones(1, 1))
    torch.testing.assert_close(
        aligned.raw_features, periodic.raw_features, atol=1e-6, rtol=1e-6
    )
    assert aligned.torsions_per_cell.tolist() == [2]
    torch.testing.assert_close(aligned.raw_features, reordered.raw_features)

    differentiable_angles = torch.tensor([0.2, -0.4], requires_grad=True)
    model(differentiable_angles, incidence).cell_embeddings.sum().backward()
    assert torch.isfinite(differentiable_angles.grad).all()

    with pytest.raises(ValueError, match="duplicate"):
        model(torch.tensor([0.0, 0.0]), torch.tensor([[0, 0], [0, 0]]))
    with pytest.raises(TypeError, match="integer"):
        model(torch.tensor([0.0, 0.0]), incidence.float())


def test_coupled_torus_modes_receive_exact_hodge_heat_decay():
    coefficients = torch.tensor([[2.0], [3.0]])
    wavevectors = torch.tensor([[1, -1], [2, 0]])
    diffused = torus_heat_decay(coefficients, wavevectors, torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(diffused[0], coefficients)
    expected = coefficients * torch.exp(-0.5 * torch.tensor([[2.0], [4.0]]))
    torch.testing.assert_close(diffused[1], expected)

    complex_coefficients = coefficients.to(torch.complex64) * (1 + 1j)
    complex_diffused = torus_heat_decay(complex_coefficients, wavevectors, [0.0, 0.5])
    assert not torch.isnan(complex_diffused).any()
    torch.testing.assert_close(complex_diffused[0], complex_coefficients)


def test_configuration_and_boundary_validation_fail_early():
    with pytest.raises(ValueError, match="max_modes"):
        HodgeDiffusionConfig(max_modes=0)
    with pytest.raises(ValueError, match="at least one"):
        validate_boundary_operators([])
    with pytest.raises(ValueError, match="finite"):
        validate_boundary_operators([torch.tensor([[float("nan")]])])
