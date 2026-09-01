import pytest
import torch

from tope.quantum.electronic_complex import ElectronicCochain
from tope.quantum.relativistic_holography import (
    CliffordCommutatorDynamics,
    CliffordHologramLayer,
    ElectronicCliffordHologram,
    GyrotrigonometricInteractionKernel,
    clifford_product,
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
    lorentz_gamma_from_poincare,
    mass_shell_four_vector,
    matrix_to_clifford,
    mobius_add,
    mobius_gyration_matrix,
    momentum_to_poincare,
    poincare_gyrolength,
    poincare_to_einstein_velocity,
    poincare_to_momentum,
    project_clifford_hologram,
)


def test_gamma_matrices_encode_minkowski_metric():
    gamma, gamma5 = dirac_matrices(dtype=torch.complex128)
    identity = torch.eye(4, dtype=torch.complex128)
    metric = torch.tensor([1.0, -1.0, -1.0, -1.0])
    for mu in range(4):
        for nu in range(4):
            anti = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
            expected = 2 * metric[mu] * identity if mu == nu else torch.zeros_like(identity)
            assert torch.allclose(anti, expected)
    assert torch.allclose(gamma5 @ gamma5, identity)
    assert torch.allclose(gamma5.conj().T, gamma5)


def test_clifford_round_trip_and_fixed_product():
    torch.manual_seed(2)
    coefficients = torch.randn(5, 16, dtype=torch.complex64)
    matrix = clifford_to_matrix(coefficients)
    assert torch.allclose(matrix_to_clifford(matrix), coefficients, atol=1e-5)

    left = torch.zeros(16)
    right = torch.zeros(16)
    left[1] = 1  # gamma0
    right[2] = 1  # gamma1
    expected = dirac_basis()[1] @ dirac_basis()[2]
    assert torch.allclose(clifford_to_matrix(clifford_product(left, right)), expected)


def test_mass_shell_poincare_map_is_exact_and_differentiable():
    momentum = torch.tensor([[0.2, -0.4, 1.1], [2.0, 0.1, -0.3]], requires_grad=True)
    coordinate = momentum_to_poincare(momentum, mass=1.7, c=2.3)
    recovered = poincare_to_momentum(coordinate, mass=1.7, c=2.3)
    assert torch.all(coordinate.norm(dim=-1) < 1)
    assert torch.allclose(recovered, momentum, atol=2e-6)
    shell = mass_shell_four_vector(momentum, mass=1.7, c=2.3)
    norm = shell[:, 0].square() - shell[:, 1:].square().sum(-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=2e-6)
    coordinate.square().sum().backward()
    assert torch.isfinite(momentum.grad).all()


def test_poincare_rapidity_and_einstein_velocity_conventions_are_distinct():
    coordinate = torch.tensor([[0.2, -0.1, 0.04], [-0.3, 0.05, 0.1]])
    rapidity = poincare_gyrolength(coordinate)
    half_gamma = half_rapidity_gamma(coordinate)
    lorentz_gamma = lorentz_gamma_from_poincare(coordinate)
    torch.testing.assert_close(half_gamma, torch.cosh(rapidity / 2))
    torch.testing.assert_close(lorentz_gamma, torch.cosh(rapidity))
    assert torch.all(lorentz_gamma > half_gamma)

    velocity = poincare_to_einstein_velocity(coordinate, c=2.5)
    recovered = einstein_velocity_to_poincare(velocity, c=2.5)
    torch.testing.assert_close(recovered, coordinate, atol=2e-6, rtol=2e-6)


def test_gyroangle_features_are_rooted_and_translation_invariant():
    root = torch.tensor([0.05, -0.07, 0.03])
    left = torch.tensor([0.2, 0.1, 0.0])
    right = torch.tensor([-0.05, 0.15, 0.08])
    translation = torch.tensor([0.1, -0.04, 0.02])
    features = gyrotrigonometric_features(left, right, root=root)
    translated = gyrotrigonometric_features(
        mobius_add(translation, left),
        mobius_add(translation, right),
        root=mobius_add(translation, root),
    )
    torch.testing.assert_close(features.gyrocosine, translated.gyrocosine, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(features.gyrosine, translated.gyrosine, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(features.gyroangle, translated.gyroangle, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gyrocosine(left, right, root=root), features.gyrocosine)
    torch.testing.assert_close(gyrosine(left, right, root=root), features.gyrosine)
    torch.testing.assert_close(gyroangle(left, right, root=root), features.gyroangle)
    torch.testing.assert_close(
        features.gyrocosine.square() + features.gyrosine.square(),
        torch.ones(()), atol=2e-6, rtol=2e-6,
    )
    assert features.as_tensor().shape == (8,)


def test_gamma_composition_uses_einstein_not_half_rapidity_coordinates():
    left = torch.tensor([0.2, 0.1, 0.0])
    right = torch.tensor([-0.05, 0.15, 0.08])
    predicted = composed_lorentz_gamma(left, right)
    actual = lorentz_gamma_from_poincare(mobius_add(left, right))
    torch.testing.assert_close(predicted, actual, atol=2e-6, rtol=2e-6)


def test_mobius_gyration_is_thomas_rotation_and_collinear_limit_is_flat():
    left = torch.tensor([0.2, 0.0, 0.0])
    right = torch.tensor([0.0, 0.25, 0.0])
    rotation = mobius_gyration_matrix(left, right)
    identity = torch.eye(3)
    torch.testing.assert_close(rotation.T @ rotation, identity, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(()), atol=2e-6, rtol=2e-6)
    assert not torch.allclose(rotation, identity)
    collinear = mobius_gyration_matrix(left, 0.5 * left)
    torch.testing.assert_close(collinear, identity, atol=2e-6, rtol=2e-6)


def test_coupled_boost_preserves_spinor_pseudo_norm():
    boost = torch.tensor([0.12, -0.08, 0.03])
    coordinate = torch.tensor([-0.1, 0.04, 0.02])
    spinor = torch.tensor([1 + 0.2j, -0.1j, 0.3, -0.2 + 0.1j])
    state = torch.randn(16, dtype=torch.complex64)
    result = coupled_boost(boost, coordinate, spinor=spinor, clifford=state)
    gamma0 = dirac_matrices()[0][0]
    before = spinor.conj() @ gamma0 @ spinor
    after = result.spinor.conj() @ gamma0 @ result.spinor
    assert result.coordinate.norm() < 1
    assert torch.allclose(after, before, atol=2e-6)
    assert result.clifford.shape == (16,)

    batched_boost = torch.stack([boost, -boost])
    batched_coordinate = torch.stack([coordinate, -coordinate])
    modal_state = torch.randn(2, 5, 16, dtype=torch.complex64)
    modal_result = coupled_boost(
        batched_boost, batched_coordinate, clifford=modal_state
    )
    assert modal_result.clifford.shape == (2, 5, 16)


def test_dirac_hamiltonian_is_hermitian_and_has_expected_dispersion():
    coordinate = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.05]])
    hamiltonian = dirac_hamiltonian(coordinate, mass=1.3, c=2.0)
    assert torch.allclose(hamiltonian, hamiltonian.conj().transpose(-1, -2))
    momentum = poincare_to_momentum(coordinate, mass=1.3, c=2.0)
    energy = torch.sqrt((1.3 * 2.0**2) ** 2 + 2.0**2 * momentum.square().sum(-1))
    eigenvalues = torch.linalg.eigvalsh(hamiltonian)
    assert torch.allclose(eigenvalues[:, :2], -energy[:, None], atol=2e-5)
    assert torch.allclose(eigenvalues[:, 2:], energy[:, None], atol=2e-5)


def test_hologram_projection_layer_and_commutator_dynamics_have_gradients():
    field = torch.randn(7, 16, requires_grad=True)
    modes = torch.randn(7, 4, dtype=torch.complex64)
    hologram = project_clifford_hologram(field, modes, normalize=True)
    assert hologram.shape == (4, 16)

    layer = CliffordHologramLayer(residual=False)
    output = layer(hologram, hologram.roll(1, 0), operation="product")
    dynamics = CliffordCommutatorDynamics()
    identity_h = torch.zeros_like(output)
    identity_h[..., 0] = 1
    assert torch.allclose(dynamics(output, identity_h), torch.zeros_like(output), atol=1e-5)
    output.abs().sum().backward()
    assert torch.isfinite(field.grad).all()


def test_electronic_adapter_preserves_types_and_rejects_quadrupole_relabeling():
    cells, spectra, hidden, observables = 2, 3, 5, 2
    eye = torch.eye(3).expand(cells, -1, -1).clone()
    cochain = ElectronicCochain(
        scalar=torch.randn(cells, spectra),
        vector=torch.randn(cells, spectra, 3),
        quadrupole=torch.randn(cells, spectra, 3, 3),
        embedding=torch.randn(cells, hidden),
        centers=torch.randn(cells, 3), frames=eye,
        charge=torch.randn(cells), dipole=torch.randn(cells, 3),
        charge_quadrupole=torch.randn(cells, 3, 3),
        spectral_density=torch.randn(cells, spectra),
        spectral_dipole=torch.randn(cells, spectra, 3),
        spectral_quadrupole=torch.randn(cells, spectra, 3, 3),
        coherence=torch.rand(cells), observables=torch.randn(cells, observables),
    )
    adapter = ElectronicCliffordHologram(spectra, n_modes=4)
    hologram = adapter(cochain)
    assert hologram.shape == (cells, 4, 16)
    assert torch.count_nonzero(hologram[..., 5:]).item() == 0
    with pytest.raises(ValueError, match="tensor must have shape"):
        adapter(cochain, tensor=cochain.quadrupole)


def test_gyrotrigonometric_kernel_couples_angles_gyrations_and_clifford_channels():
    left_coordinate = torch.tensor(
        [[0.2, 0.0, 0.0], [0.1, -0.04, 0.03]], requires_grad=True
    )
    right_coordinate = torch.tensor(
        [[0.0, 0.25, 0.0], [-0.02, 0.13, 0.06]], requires_grad=True
    )
    left_state = torch.randn(2, 3, 16, requires_grad=True)
    right_state = torch.randn(2, 3, 16, requires_grad=True)
    kernel = GyrotrigonometricInteractionKernel(hidden_dim=12)
    result = kernel(left_coordinate, right_coordinate, left_state, right_state)
    assert result.combined_coordinate.shape == (2, 3)
    assert result.features.gyration.shape == (2, 3, 3)
    assert result.composite_spinor.shape == (2, 4, 4)
    assert result.spinor_clifford.shape == (2, 16)
    assert result.interaction.shape == (2, 3, 16)
    # Orthogonal boosts generate a nonzero xy spatial-bivector/rotation channel.
    assert result.spinor_clifford[0, 8].abs() > 1e-5
    result.interaction.abs().sum().backward()
    for gradient in (
        left_coordinate.grad, right_coordinate.grad, left_state.grad, right_state.grad
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
