import numpy as np
import pytest

from tope.simulation import (
    AnharmonicEnergy,
    CPFactorization,
    HarmonicSystem,
    LosslessTrajectoryCoder,
    MatrixProductState,
    QuantizedTrajectoryCoder,
)


def test_cp_factorization_reconstructs_rank_one_tensor():
    tensor = np.einsum(
        "i,j,k->ijk",
        np.arange(1.0, 5.0),
        np.array([2.0, -1.0, 3.0]),
        np.array([0.5, 1.5]),
    )
    decomposition = CPFactorization(
        rank=1, max_iter=100, tolerance=1e-8, random_state=7
    ).fit_transform(tensor)
    assert decomposition.reconstruct().shape == tensor.shape
    assert np.allclose(decomposition.reconstruct(), tensor, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lossless_float_coding_round_trips_exact_bits(dtype):
    trajectory = np.array(
        [[[0.125, 1.25], [-3.5, 4.25]], [[12.3456789, -0.0], [1e-7, np.nan]]],
        dtype=dtype,
    )
    if dtype == np.float32:
        trajectory = trajectory.astype(np.float32)
    encoded = LosslessTrajectoryCoder().encode(trajectory)
    decoded = LosslessTrajectoryCoder().decode(encoded)
    np.testing.assert_array_equal(decoded.view(np.uint32 if dtype == np.float32 else np.uint64),
                                  trajectory.view(np.uint32 if dtype == np.float32 else np.uint64))


def test_quantized_coding_is_exact_on_grid():
    trajectory = np.array([[[12.345, -2.0]], [[12.346, 0.001]]])
    encoded = QuantizedTrajectoryCoder(spacing=0.001).encode(trajectory)
    decoded = QuantizedTrajectoryCoder(spacing=0.001).decode(encoded)
    integers = np.rint(trajectory / 0.001).astype(np.int64)
    np.testing.assert_array_equal(decoded, integers * 0.001)


def test_harmonic_propagation_and_zero_modes():
    system = HarmonicSystem(np.diag([0.0, 4.0, 9.0]))
    q0 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.5, 1.0, -1.0])
    q, v = system.propagate(q0, v0, 0.25)
    assert q.shape == q0.shape
    assert v.shape == v0.shape
    np.testing.assert_allclose(q[0], q0[0] + v0[0] * 0.25)
    with pytest.raises(ValueError):
        HarmonicSystem(np.diag([-1.0, 1.0]))


def test_anharmonic_energy_and_force_are_conservative():
    quadratic = np.eye(2)
    cubic = np.zeros((2, 2, 2))
    cubic[0, 0, 0] = 2.0
    quartic = np.zeros((2, 2, 2, 2))
    quartic[0, 0, 0, 0] = 4.0
    energy = AnharmonicEnergy(quadratic, cubic, quartic)
    q = np.array([0.7, -0.2])
    eps = 1e-6
    numerical = np.empty(2)
    for i in range(2):
        plus = q.copy()
        minus = q.copy()
        plus[i] += eps
        minus[i] -= eps
        numerical[i] = (energy.energy(plus) - energy.energy(minus)) / (2 * eps)
    np.testing.assert_allclose(energy.gradient(q), numerical, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(energy.forces(q), -numerical)


def test_mps_round_trips_four_level_coefficient_tensor():
    coefficients = np.arange(16, dtype=np.complex128)
    coefficients /= np.linalg.norm(coefficients)
    mps = MatrixProductState.from_coefficients(
        coefficients, physical_dimension=4, max_bond_dimension=8
    )
    reconstructed = mps.to_coefficients()
    assert reconstructed.shape == (4, 4)
    np.testing.assert_allclose(reconstructed, coefficients.reshape(4, 4), atol=1e-8)
