"""Matrix-product and adaptive tensor-network research primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class MatrixProductState:
    """A right-canonicalized MPS for a multi-orbital coefficient tensor."""

    tensors: List[np.ndarray]
    bond_dimensions: List[int]
    cutoff: float

    @classmethod
    def from_coefficients(
        cls,
        coefficients: np.ndarray,
        physical_dimension: int = 4,
        max_bond_dimension: int = 128,
        cutoff: float = 1e-12,
    ) -> "MatrixProductState":
        state = np.asarray(coefficients, dtype=np.complex128)
        if state.size == 0:
            raise ValueError("coefficients must not be empty")
        if state.ndim == 1:
            flat = state.reshape(-1)
            n_sites = 0
            remaining = flat.size
            while remaining > 1 and remaining % physical_dimension == 0:
                remaining //= physical_dimension
                n_sites += 1
            if n_sites == 0:
                n_sites = 1
            if flat.size != physical_dimension ** n_sites:
                raise ValueError(
                    "coefficient vector length must be a power of physical_dimension"
                )
            state = flat.reshape((physical_dimension,) * n_sites)
        else:
            n_sites = state.ndim
        if any(d != physical_dimension for d in state.shape):
            raise ValueError(
                f"each orbital axis must have physical dimension {physical_dimension}"
            )
        if max_bond_dimension < 1:
            raise ValueError("max_bond_dimension must be positive")
        residual = state.reshape(physical_dimension, -1)
        tensors: List[np.ndarray] = []
        left_bond = 1
        bond_dimensions = [1]

        for site in range(n_sites - 1):
            rows = left_bond * physical_dimension
            matrix = residual.reshape(rows, -1)
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = int(np.count_nonzero(s > max(cutoff, cutoff * s[0]))) if s.size else 1
            keep = max(1, min(keep, max_bond_dimension))
            u = u[:, :keep]
            vh = vh[:keep, :]
            s = s[:keep]
            tensors.append(u.reshape(left_bond, physical_dimension, keep))
            residual = np.diag(s) @ vh
            left_bond = keep
            bond_dimensions.append(left_bond)

        tensors.append(residual.reshape(left_bond, physical_dimension, 1))
        bond_dimensions.append(1)
        norm = cls.from_tensors(tensors).norm()
        if norm > 0:
            tensors[-1] /= norm
        return cls(tensors=tensors, bond_dimensions=bond_dimensions, cutoff=cutoff)

    @classmethod
    def from_tensors(cls, tensors: List[np.ndarray], cutoff: float = 1e-12) -> "MatrixProductState":
        bond_dimensions = [1]
        bond_dimensions.extend(int(tensor.shape[2]) for tensor in tensors)
        return cls(tensors=tensors, bond_dimensions=bond_dimensions, cutoff=cutoff)

    def to_coefficients(self) -> np.ndarray:
        first = self.tensors[0]
        if first.shape[0] != 1:
            raise ValueError("MPS must begin with a bond dimension of one")
        result = first[0]
        for tensor in self.tensors[1:]:
            result = np.einsum("...r,rps->...ps", result, tensor)
        result = np.squeeze(result)
        if result.ndim == 0:
            return np.asarray(result)
        return result

    def norm(self) -> float:
        state = self.to_coefficients()
        return float(np.sqrt(np.real(np.vdot(state, state))))


MPSWavefunction = MatrixProductState
