"""Relativistic Clifford holography on the positive-energy mass shell.

This module couples three representations without conflating them:

* Minkowski/Dirac algebra uses the ``(+---)`` metric;
* physical momenta are compactified onto the Poincare ball (the mass shell);
* electronic fields are expanded in the fixed 16-channel Dirac basis.

The routines are differentiable PyTorch operations.  They provide a
physics-typed representation and exact algebraic constraints, not a claim that
the non-relativistic electronic features produced elsewhere in ToPE are a
four-component Dirac wavefunction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


CLIFFORD_CHANNELS: Tuple[str, ...] = (
    "scalar",
    "vector_t", "vector_x", "vector_y", "vector_z",
    "tensor_tx", "tensor_ty", "tensor_tz",
    "tensor_xy", "tensor_xz", "tensor_yz",
    "axial_t", "axial_x", "axial_y", "axial_z",
    "pseudoscalar",
)
CLIFFORD_GRADES: Tuple[str, ...] = (
    "scalar",
    *("vector",) * 4,
    *("tensor",) * 6,
    *("axial",) * 4,
    "pseudoscalar",
)
GYROTRIGONOMETRIC_CHANNELS: Tuple[str, ...] = (
    "left_gyrolength", "right_gyrolength",
    "left_half_rapidity_gamma", "right_half_rapidity_gamma",
    "left_lorentz_gamma", "right_lorentz_gamma",
    "gyrocosine", "gyrosine",
)


def _require_complex(dtype: torch.dtype) -> None:
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Dirac matrices require torch.complex64 or torch.complex128")


def dirac_matrices(
    *, dtype: torch.dtype = torch.complex64, device: Optional[torch.device] = None
) -> Tuple[Tensor, Tensor]:
    """Return ``gamma[mu]`` and ``gamma5`` in the Dirac representation.

    The matrices obey ``{gamma^mu, gamma^nu} = 2 eta^{mu nu}`` for
    ``eta = diag(1, -1, -1, -1)``.
    """
    _require_complex(dtype)
    zero2 = torch.zeros((2, 2), dtype=dtype, device=device)
    eye2 = torch.eye(2, dtype=dtype, device=device)
    sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

    gamma0 = torch.cat(
        [torch.cat([eye2, zero2], -1), torch.cat([zero2, -eye2], -1)], -2
    )
    spatial = []
    for sigma in (sx, sy, sz):
        spatial.append(torch.cat([
            torch.cat([zero2, sigma], -1),
            torch.cat([-sigma, zero2], -1),
        ], -2))
    gamma = torch.stack([gamma0, *spatial])
    gamma5 = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    return gamma, gamma5


def dirac_basis(
    *, dtype: torch.dtype = torch.complex64, device: Optional[torch.device] = None
) -> Tensor:
    """Return the ordered 16-element Dirac/Clifford matrix basis.

    Ordering follows :data:`CLIFFORD_CHANNELS`: ``I``, ``gamma^mu``, the six
    ``sigma^{mu nu}=i[gamma^mu,gamma^nu]/2`` (mu < nu),
    ``gamma5 gamma^mu``, and ``gamma5``.
    """
    gamma, gamma5 = dirac_matrices(dtype=dtype, device=device)
    identity = torch.eye(4, dtype=dtype, device=device)
    tensor = []
    for mu, nu in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        tensor.append(0.5j * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]))
    axial = [gamma5 @ gamma[mu] for mu in range(4)]
    return torch.stack([identity, *list(gamma), *tensor, *axial, gamma5])


def clifford_to_matrix(coefficients: Tensor, basis: Optional[Tensor] = None) -> Tensor:
    """Reconstruct a 4x4 matrix from coefficients with final dimension 16."""
    if coefficients.shape[-1] != 16:
        raise ValueError("Clifford coefficients must have final dimension 16")
    if basis is None:
        dtype = coefficients.dtype if coefficients.is_complex() else torch.complex64
        basis = dirac_basis(dtype=dtype, device=coefficients.device)
    coefficients = coefficients.to(basis.dtype)
    return torch.einsum("...a,aij->...ij", coefficients, basis)


def matrix_to_clifford(matrix: Tensor, basis: Optional[Tensor] = None) -> Tensor:
    """Resolve a 4x4 matrix in the ordered Dirac basis."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError("matrix must have final shape (4, 4)")
    if basis is None:
        dtype = matrix.dtype if matrix.is_complex() else torch.complex64
        basis = dirac_basis(dtype=dtype, device=matrix.device)
    flat_basis = basis.reshape(16, 16)
    dual = torch.linalg.inv(flat_basis)
    return matrix.to(basis.dtype).reshape(*matrix.shape[:-2], 16) @ dual


def clifford_structure_constants(
    *, dtype: torch.dtype = torch.complex64, device: Optional[torch.device] = None
) -> Tensor:
    """Return fixed ``C[a,b,c]`` such that ``Gamma_a Gamma_b=C[a,b,c]Gamma_c``."""
    basis = dirac_basis(dtype=dtype, device=device)
    products = torch.einsum("aij,bjk->abik", basis, basis)
    return matrix_to_clifford(products, basis)


def clifford_product(left: Tensor, right: Tensor, structure: Optional[Tensor] = None) -> Tensor:
    """Multiply two Clifford coefficient fields using the fixed gamma algebra."""
    if left.shape[-1] != 16 or right.shape[-1] != 16:
        raise ValueError("left and right must have final dimension 16")
    if structure is None:
        dtype = torch.promote_types(left.dtype, right.dtype)
        if dtype not in (torch.complex64, torch.complex128):
            dtype = torch.complex64
        structure = clifford_structure_constants(dtype=dtype, device=left.device)
    return torch.einsum(
        "...a,...b,abc->...c", left.to(structure.dtype), right.to(structure.dtype), structure
    )


def clifford_commutator(left: Tensor, right: Tensor, structure: Optional[Tensor] = None) -> Tensor:
    """Return ``[left,right]`` in Clifford coefficient form."""
    return clifford_product(left, right, structure) - clifford_product(right, left, structure)


def clifford_anticommutator(
    left: Tensor, right: Tensor, structure: Optional[Tensor] = None
) -> Tensor:
    """Return ``{left,right}`` in Clifford coefficient form."""
    return clifford_product(left, right, structure) + clifford_product(right, left, structure)


def momentum_to_poincare(momentum: Tensor, mass: float = 1.0, c: float = 1.0) -> Tensor:
    """Map canonical momentum exactly to half-rapidity Poincare coordinates."""
    if mass <= 0 or c <= 0:
        raise ValueError("mass and c must be positive")
    p2 = momentum.square().sum(-1, keepdim=True)
    energy = torch.sqrt(momentum.new_tensor((mass * c * c) ** 2) + c * c * p2)
    return c * momentum / (energy + mass * c * c)


def poincare_to_momentum(coordinate: Tensor, mass: float = 1.0, c: float = 1.0) -> Tensor:
    """Invert :func:`momentum_to_poincare`."""
    if mass <= 0 or c <= 0:
        raise ValueError("mass and c must be positive")
    r2 = coordinate.square().sum(-1, keepdim=True)
    if torch.any(r2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    return 2.0 * mass * c * coordinate / (1.0 - r2)


def mass_shell_four_vector(momentum: Tensor, mass: float = 1.0, c: float = 1.0) -> Tensor:
    """Return normalized ``Q=P/(mc)`` with Minkowski norm +1."""
    p2 = momentum.square().sum(-1, keepdim=True)
    energy = torch.sqrt(momentum.new_tensor((mass * c * c) ** 2) + c * c * p2)
    return torch.cat([energy / (mass * c * c), momentum / (mass * c)], -1)


def mobius_add(left: Tensor, right: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Mobius gyroaddition on the unit Poincare ball."""
    x2 = left.square().sum(-1, keepdim=True)
    y2 = right.square().sum(-1, keepdim=True)
    if torch.any(x2 >= 1) or torch.any(y2 >= 1):
        raise ValueError("Mobius operands must lie strictly inside the unit ball")
    xy = (left * right).sum(-1, keepdim=True)
    numerator = (1 + 2 * xy + y2) * left + (1 - x2) * right
    denominator = (1 + 2 * xy + x2 * y2).clamp_min(epsilon)
    result = numerator / denominator
    norm = result.norm(dim=-1, keepdim=True)
    return result * ((1.0 - epsilon) / norm.clamp_min(epsilon)).clamp(max=1.0)


def poincare_gyrolength(coordinate: Tensor, epsilon: float = 1e-7) -> Tensor:
    """Return rapidity/geodesic radius ``2 artanh(|u|)`` on the unit ball."""
    radius = coordinate.norm(dim=-1)
    if torch.any(radius >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    return 2.0 * torch.atanh(radius.clamp(max=1.0 - epsilon))


def half_rapidity_gamma(coordinate: Tensor) -> Tensor:
    """Return ``cosh(rapidity/2)=1/sqrt(1-|u|^2)`` for Poincare coordinates."""
    radius2 = coordinate.square().sum(-1)
    if torch.any(radius2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    return torch.rsqrt(1.0 - radius2)


def lorentz_gamma_from_poincare(coordinate: Tensor) -> Tensor:
    """Return the physical Lorentz gamma ``cosh(rapidity)``.

    This is ``(1+|u|^2)/(1-|u|^2)``, not the half-rapidity gamma multiplying
    :func:`spinor_boost`.
    """
    radius2 = coordinate.square().sum(-1)
    if torch.any(radius2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    return (1.0 + radius2) / (1.0 - radius2)


def poincare_to_einstein_velocity(coordinate: Tensor, c: float = 1.0) -> Tensor:
    """Map half-rapidity Poincare coordinates to Einstein velocity coordinates."""
    if c <= 0:
        raise ValueError("c must be positive")
    radius2 = coordinate.square().sum(-1, keepdim=True)
    if torch.any(radius2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    return 2.0 * c * coordinate / (1.0 + radius2)


def einstein_velocity_to_poincare(velocity: Tensor, c: float = 1.0) -> Tensor:
    """Map a subluminal Einstein velocity to half-rapidity coordinates."""
    if c <= 0:
        raise ValueError("c must be positive")
    beta = velocity / c
    beta2 = beta.square().sum(-1, keepdim=True)
    if torch.any(beta2 >= 1):
        raise ValueError("Einstein velocities must have norm strictly below c")
    return beta / (1.0 + torch.sqrt(1.0 - beta2))


def mobius_gyration(left: Tensor, right: Tensor, vector: Tensor) -> Tensor:
    """Apply the exact Möbius gyration ``gyr[left,right]`` to a tangent vector."""
    for label, value in (("left", left), ("right", right)):
        if torch.any(value.square().sum(-1) >= 1):
            raise ValueError(f"{label} must lie strictly inside the unit ball")
    ab = (left * right).sum(-1, keepdim=True)
    av = (left * vector).sum(-1, keepdim=True)
    bv = (right * vector).sum(-1, keepdim=True)
    aa = left.square().sum(-1, keepdim=True)
    bb = right.square().sum(-1, keepdim=True)
    denominator = (1.0 + 2.0 * ab + aa * bb).clamp_min(1e-12)
    coefficient_left = (2.0 / denominator) * (
        (1.0 + 2.0 * ab) * bv - bb * av
    )
    coefficient_right = (2.0 / denominator) * (-av - aa * bv)
    return vector + coefficient_left * left + coefficient_right * right


def mobius_gyration_matrix(left: Tensor, right: Tensor) -> Tensor:
    """Return the spatial rotation matrix of ``gyr[left,right]``."""
    if left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError("relativistic momentum gyrations require three-vectors")
    identity = torch.eye(3, dtype=left.dtype, device=left.device)
    target_shape = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    basis = identity.expand(*target_shape, 3, 3)
    a = left.expand(*target_shape, 3).unsqueeze(-2)
    b = right.expand(*target_shape, 3).unsqueeze(-2)
    # Rows are transformed basis vectors; transpose to the conventional
    # column-action rotation matrix.
    return mobius_gyration(a, b, basis).transpose(-1, -2)


@dataclass
class GyrotrigonometricFeatures:
    """Invariant gyroangle features and the associated Thomas/Wigner rotation."""

    left_gyrolength: Tensor
    right_gyrolength: Tensor
    left_half_rapidity_gamma: Tensor
    right_half_rapidity_gamma: Tensor
    left_lorentz_gamma: Tensor
    right_lorentz_gamma: Tensor
    gyrocosine: Tensor
    gyrosine: Tensor
    gyroangle: Tensor
    plane_normal: Tensor
    gyration: Tensor

    def as_tensor(self) -> Tensor:
        """Stack the scalar kernel channels in ``GYROTRIGONOMETRIC_CHANNELS`` order."""
        return torch.stack([
            self.left_gyrolength,
            self.right_gyrolength,
            self.left_half_rapidity_gamma,
            self.right_half_rapidity_gamma,
            self.left_lorentz_gamma,
            self.right_lorentz_gamma,
            self.gyrocosine,
            self.gyrosine,
        ], dim=-1)


def gyrotrigonometric_features(
    left: Tensor,
    right: Tensor,
    *,
    root: Optional[Tensor] = None,
    epsilon: float = 1e-7,
) -> GyrotrigonometricFeatures:
    """Compute rooted gyroangle invariants for two Poincare-ball points.

    A common root is removed by Möbius translation before measuring the angle.
    Because the Poincare metric is conformal, normalized dot/cross products of
    these rooted gyrovectors give the gyrocosine, gyrosine, and oriented plane.
    At a degenerate zero-length side the angle convention is zero.
    """
    if left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError("gyrotrigonometric features require three-vectors")
    rooted_left, rooted_right = left, right
    if root is not None:
        if root.shape[-1] != 3:
            raise ValueError("root must be a three-vector")
        rooted_left = mobius_add(-root, left)
        rooted_right = mobius_add(-root, right)
    left_norm = rooted_left.norm(dim=-1)
    right_norm = rooted_right.norm(dim=-1)
    valid = (left_norm > epsilon) & (right_norm > epsilon)
    denominator = (left_norm * right_norm).clamp_min(epsilon)
    cosine_raw = (rooted_left * rooted_right).sum(-1) / denominator
    cross = torch.linalg.cross(rooted_left, rooted_right, dim=-1)
    sine_raw = cross.norm(dim=-1) / denominator
    cosine = torch.where(valid, cosine_raw.clamp(-1.0, 1.0), torch.ones_like(cosine_raw))
    sine = torch.where(valid, sine_raw.clamp(0.0, 1.0), torch.zeros_like(sine_raw))
    normal = torch.where(
        (cross.norm(dim=-1, keepdim=True) > epsilon),
        cross / cross.norm(dim=-1, keepdim=True).clamp_min(epsilon),
        torch.zeros_like(cross),
    )
    return GyrotrigonometricFeatures(
        poincare_gyrolength(rooted_left, epsilon),
        poincare_gyrolength(rooted_right, epsilon),
        half_rapidity_gamma(rooted_left),
        half_rapidity_gamma(rooted_right),
        lorentz_gamma_from_poincare(rooted_left),
        lorentz_gamma_from_poincare(rooted_right),
        cosine,
        sine,
        torch.atan2(sine, cosine),
        normal,
        mobius_gyration_matrix(rooted_left, rooted_right),
    )


def gyrocosine(left: Tensor, right: Tensor, *, root: Optional[Tensor] = None) -> Tensor:
    """Return the cosine of the rooted gyroangle between two ball points."""
    return gyrotrigonometric_features(left, right, root=root).gyrocosine


def gyrosine(left: Tensor, right: Tensor, *, root: Optional[Tensor] = None) -> Tensor:
    """Return the non-negative sine of the rooted gyroangle."""
    return gyrotrigonometric_features(left, right, root=root).gyrosine


def gyroangle(left: Tensor, right: Tensor, *, root: Optional[Tensor] = None) -> Tensor:
    """Return the rooted gyroangle in radians using ``atan2(gyrosine, gyrocosine)``."""
    return gyrotrigonometric_features(left, right, root=root).gyroangle


def composed_lorentz_gamma(left: Tensor, right: Tensor) -> Tensor:
    """Einstein gamma composition expressed through Poincare gyrovectors.

    The identity is evaluated in Einstein/Klein velocity coordinates, avoiding
    the common error of applying ``gamma_a gamma_b (1+a·b)`` directly to
    half-rapidity Poincare coordinates.
    """
    beta_left = poincare_to_einstein_velocity(left)
    beta_right = poincare_to_einstein_velocity(right)
    return lorentz_gamma_from_poincare(left) * lorentz_gamma_from_poincare(right) * (
        1.0 + (beta_left * beta_right).sum(-1)
    )


def spinor_boost(
    gyrovector: Tensor,
    *,
    inverse: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Spin(1,3) boost parameterized by the same half-rapidity gyrovector.

    ``S(a)=(I-a_i alpha^i)/sqrt(1-|a|^2)``.  Set ``inverse=True`` to reverse
    the active-boost sign convention.
    """
    r2 = gyrovector.square().sum(-1)
    if torch.any(r2 >= 1):
        raise ValueError("boost gyrovector must lie strictly inside the unit ball")
    if dtype is None:
        dtype = torch.complex128 if gyrovector.dtype == torch.float64 else torch.complex64
    gamma, _ = dirac_matrices(dtype=dtype, device=gyrovector.device)
    alpha = torch.stack([gamma[0] @ gamma[i] for i in range(1, 4)])
    generator = torch.einsum("...i,ijk->...jk", gyrovector.to(dtype), alpha)
    identity = torch.eye(4, dtype=dtype, device=gyrovector.device)
    sign = 1.0 if inverse else -1.0
    return (identity + sign * generator) / torch.sqrt(1.0 - r2)[..., None, None]


@dataclass
class CoupledBoostResult:
    coordinate: Tensor
    spinor: Optional[Tensor] = None
    clifford: Optional[Tensor] = None


def coupled_boost(
    boost: Tensor,
    coordinate: Tensor,
    *,
    spinor: Optional[Tensor] = None,
    clifford: Optional[Tensor] = None,
) -> CoupledBoostResult:
    """Apply one gyrovector to momentum, spinor, and Clifford channels together."""
    transformed_coordinate = mobius_add(boost, coordinate)
    boost_matrix = spinor_boost(boost)
    transformed_spinor = None
    if spinor is not None:
        transformed_spinor = torch.einsum("...ij,...j->...i", boost_matrix, spinor.to(boost_matrix.dtype))
    transformed_clifford = None
    if clifford is not None:
        matrix = clifford_to_matrix(clifford)
        inverse = torch.linalg.inv(boost_matrix)
        # A Clifford field may carry extra spectral-mode axes after the boost
        # batch axes.  Insert singleton axes so the same physical boost acts on
        # every mode rather than relying on ambiguous matmul broadcasting.
        extra_axes = matrix.dim() - boost_matrix.dim()
        if extra_axes < 0:
            raise ValueError("clifford field has fewer batch axes than the boost")
        if extra_axes:
            shape = (*boost_matrix.shape[:-2], *((1,) * extra_axes), 4, 4)
            boost_matrix = boost_matrix.reshape(shape)
            inverse = inverse.reshape(shape)
        transformed = boost_matrix @ matrix @ inverse
        transformed_clifford = matrix_to_clifford(transformed)
    return CoupledBoostResult(transformed_coordinate, transformed_spinor, transformed_clifford)


def dirac_hamiltonian(
    coordinate: Tensor,
    mass: float = 1.0,
    c: float = 1.0,
    potential: Optional[Tensor] = None,
) -> Tensor:
    """Free Dirac Hamiltonian expressed on Poincare mass-shell coordinates."""
    r2 = coordinate.square().sum(-1)
    if torch.any(r2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    dtype = torch.complex128 if coordinate.dtype == torch.float64 else torch.complex64
    gamma, _ = dirac_matrices(dtype=dtype, device=coordinate.device)
    alpha = torch.stack([gamma[0] @ gamma[i] for i in range(1, 4)])
    kinetic = torch.einsum("...i,ijk->...jk", coordinate.to(dtype), alpha)
    hamiltonian = mass * c * c * gamma[0] + (
        2.0 * mass * c * c / (1.0 - r2)
    )[..., None, None] * kinetic
    if potential is not None:
        identity = torch.eye(4, dtype=dtype, device=coordinate.device)
        hamiltonian = hamiltonian + potential.to(dtype)[..., None, None] * identity
    return hamiltonian


def hyperbolic_volume_weights(coordinate: Tensor, euclidean_weights: Optional[Tensor] = None) -> Tensor:
    """Poincare-ball volume weights ``(2/(1-|u|^2))^3 d^3u``."""
    r2 = coordinate.square().sum(-1)
    if torch.any(r2 >= 1):
        raise ValueError("Poincare coordinates must lie strictly inside the unit ball")
    weights = (2.0 / (1.0 - r2)) ** 3
    return weights if euclidean_weights is None else weights * euclidean_weights


def project_clifford_hologram(
    field: Tensor,
    modes: Tensor,
    weights: Optional[Tensor] = None,
    *,
    normalize: bool = False,
) -> Tensor:
    """Project a sampled Clifford field onto scalar hyperbolic modes.

    ``field`` has shape ``(..., points, 16)`` and ``modes`` has shape
    ``(..., points, modes)``.  The returned hologram has shape
    ``(..., modes, 16)``.
    """
    if field.shape[-1] != 16 or field.shape[-2] != modes.shape[-2]:
        raise ValueError("field/modes must agree on points and field must have 16 channels")
    if weights is None:
        weights = torch.ones(field.shape[-2], dtype=field.real.dtype, device=field.device)
    dtype = torch.promote_types(field.dtype, modes.dtype)
    projected = torch.einsum(
        "...pa,...pm,...p->...ma",
        field.to(dtype), modes.to(dtype).conj(), weights.to(dtype),
    )
    if normalize:
        denominator = weights.sum(-1).clamp_min(torch.finfo(weights.dtype).eps)
        projected = projected / denominator[..., None, None]
    return projected


class CliffordHologramLayer(nn.Module):
    """Fixed gamma-algebra interaction with learnable per-type response."""

    def __init__(self, learnable_scale: bool = True, residual: bool = True) -> None:
        super().__init__()
        structure = clifford_structure_constants()
        self.register_buffer("structure", structure)
        scale = torch.ones(16)
        if learnable_scale:
            self.channel_scale = nn.Parameter(scale)
        else:
            self.register_buffer("channel_scale", scale)
        self.residual = residual

    def forward(self, left: Tensor, right: Tensor, operation: str = "product") -> Tensor:
        if operation == "product":
            interaction = clifford_product(left, right, self.structure)
        elif operation == "commutator":
            interaction = clifford_commutator(left, right, self.structure)
        elif operation == "anticommutator":
            interaction = clifford_anticommutator(left, right, self.structure)
        else:
            raise ValueError("operation must be product, commutator, or anticommutator")
        output = interaction * self.channel_scale.to(interaction.dtype)
        return output + left.to(output.dtype) if self.residual else output


@dataclass
class GyroInteractionResult:
    """Outputs of a gyrotrigonometric two-state Clifford interaction."""

    combined_coordinate: Tensor
    features: GyrotrigonometricFeatures
    composite_spinor: Tensor
    spinor_clifford: Tensor
    interaction: Tensor


class GyrotrigonometricInteractionKernel(nn.Module):
    """Learned response constrained by analytic gyrogeometry and Clifford algebra.

    For boosts applied first along ``left`` and then ``right``, the kernel uses
    ``right (+) left`` and ``S(right) S(left)``.  Non-collinear composition is
    retained as the exact Möbius gyration and in the spatial-bivector channels
    of the composite spinor; it is not collapsed into a single pure boost.
    """

    def __init__(self, hidden_dim: int = 64, spinor_seed: bool = True) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.response = nn.Sequential(
            nn.Linear(len(GYROTRIGONOMETRIC_CHANNELS), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 16),
        )
        self.spinor_seed = spinor_seed
        self.spinor_seed_gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("structure", clifford_structure_constants())

    def forward(
        self,
        left_coordinate: Tensor,
        right_coordinate: Tensor,
        left_clifford: Tensor,
        right_clifford: Tensor,
        *,
        root: Optional[Tensor] = None,
    ) -> GyroInteractionResult:
        if left_clifford.shape != right_clifford.shape or left_clifford.shape[-1] != 16:
            raise ValueError("Clifford states must have equal shapes ending in 16")
        features = gyrotrigonometric_features(
            left_coordinate, right_coordinate, root=root
        )
        scalar_features = features.as_tensor()
        scales = 1.0 + 0.1 * torch.tanh(self.response(scalar_features))
        left_boost = spinor_boost(left_coordinate)
        right_boost = spinor_boost(right_coordinate)
        composite_spinor = right_boost @ left_boost
        spinor_clifford = matrix_to_clifford(composite_spinor)

        extra_axes = left_clifford.dim() - scales.dim()
        if extra_axes < 0:
            raise ValueError("Clifford states have fewer batch axes than coordinates")
        expanded_seed = spinor_clifford
        for _ in range(extra_axes):
            scales = scales.unsqueeze(-2)
            expanded_seed = expanded_seed.unsqueeze(-2)
        algebraic = clifford_product(left_clifford, right_clifford, self.structure)
        interaction = algebraic * scales.to(algebraic.dtype)
        if self.spinor_seed:
            interaction = interaction + self.spinor_seed_gain.to(algebraic.dtype) * expanded_seed
        return GyroInteractionResult(
            mobius_add(right_coordinate, left_coordinate),
            features,
            composite_spinor,
            spinor_clifford,
            interaction,
        )


class CliffordCommutatorDynamics(nn.Module):
    """Von Neumann dynamics ``dW/dt=-i[H,W]/hbar`` in coefficient space."""

    def __init__(self, hbar: float = 1.0) -> None:
        super().__init__()
        if hbar <= 0:
            raise ValueError("hbar must be positive")
        self.hbar = hbar
        self.register_buffer("structure", clifford_structure_constants())

    def forward(self, state: Tensor, hamiltonian: Tensor) -> Tensor:
        return (-1j / self.hbar) * clifford_commutator(
            hamiltonian, state, self.structure
        )


class ElectronicCliffordHologram(nn.Module):
    """Type-preserving lift from an ``ElectronicCochain`` to Clifford modes.

    Scalar spectra and scalar density are projected independently.  One shared
    spectral projection is applied to all three vector components, preserving
    their spatial type.  Antisymmetric tensor, axial, and pseudoscalar inputs
    must be supplied explicitly; quadrupoles are deliberately not relabeled as
    antisymmetric Clifford tensors.
    """

    def __init__(self, spectrum_dim: int, n_modes: int) -> None:
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.n_modes = n_modes
        self.scalar_projection = nn.Linear(spectrum_dim, n_modes, bias=False)
        self.density_projection = nn.Linear(spectrum_dim, n_modes, bias=False)
        self.vector_projection = nn.Linear(spectrum_dim, n_modes, bias=False)

    def forward(
        self,
        cochain,
        *,
        tensor: Optional[Tensor] = None,
        axial: Optional[Tensor] = None,
        pseudoscalar: Optional[Tensor] = None,
    ) -> Tensor:
        if cochain.scalar.shape[-1] != self.spectrum_dim:
            raise ValueError("cochain spectrum dimension does not match the adapter")
        scalar = self.scalar_projection(cochain.scalar)
        density = self.density_projection(cochain.spectral_density)
        vector_global = torch.einsum("cij,csj->csi", cochain.frames, cochain.vector)
        vector = self.vector_projection(vector_global.transpose(-1, -2)).transpose(-1, -2)
        output = torch.zeros(
            (*scalar.shape, 16), dtype=scalar.dtype, device=scalar.device
        )
        output[..., 0] = scalar
        output[..., 1] = density
        output[..., 2:5] = vector
        if tensor is not None:
            if tensor.shape != (*scalar.shape, 6):
                raise ValueError("tensor must have shape (cells, modes, 6)")
            output[..., 5:11] = tensor
        if axial is not None:
            if axial.shape != (*scalar.shape, 4):
                raise ValueError("axial must have shape (cells, modes, 4)")
            output[..., 11:15] = axial
        if pseudoscalar is not None:
            if pseudoscalar.shape != scalar.shape:
                raise ValueError("pseudoscalar must have shape (cells, modes)")
            output[..., 15] = pseudoscalar
        return output


__all__ = [
    "CLIFFORD_CHANNELS", "CLIFFORD_GRADES", "GYROTRIGONOMETRIC_CHANNELS",
    "CoupledBoostResult", "GyroInteractionResult", "GyrotrigonometricFeatures",
    "CliffordCommutatorDynamics", "CliffordHologramLayer",
    "ElectronicCliffordHologram", "GyrotrigonometricInteractionKernel",
    "clifford_anticommutator",
    "clifford_commutator", "clifford_product", "clifford_structure_constants",
    "clifford_to_matrix", "composed_lorentz_gamma", "coupled_boost",
    "dirac_basis", "dirac_hamiltonian", "dirac_matrices",
    "einstein_velocity_to_poincare", "gyroangle", "gyrocosine", "gyrosine",
    "gyrotrigonometric_features",
    "half_rapidity_gamma", "hyperbolic_volume_weights",
    "lorentz_gamma_from_poincare", "mass_shell_four_vector", "matrix_to_clifford",
    "mobius_add", "mobius_gyration", "mobius_gyration_matrix",
    "momentum_to_poincare", "poincare_gyrolength", "poincare_to_einstein_velocity",
    "poincare_to_momentum", "project_clifford_hologram", "spinor_boost",
]
