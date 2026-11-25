"""Research primitives for tensorized protein simulation.

This module deliberately separates reversible data coding, tensor
factorization, and conservative dynamics.  It is a numerical framework,
not a claim that low-rank approximations are physically validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
import zlib

import numpy as np


@dataclass
class CPDecomposition:
    """Rank-R CP decomposition of a dense 3-D tensor."""

    factors: Tuple[np.ndarray, np.ndarray, np.ndarray]
    rank: int
    relative_error: float = 0.0

    def reconstruct(self) -> np.ndarray:
        a, b, c = self.factors
        result = np.einsum("ir,jr,kr->ijk", a, b, c)
        return result


class CPFactorization:
    """Fit X ~= sum_r A[:,r] B[:,r] C[:,r] by alternating least squares."""

    def __init__(
        self,
        rank: int,
        max_iter: int = 100,
        tolerance: float = 1e-8,
        random_state: Optional[int] = None,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.rank = rank
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.factors: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.relative_error = 0.0

    def fit(self, tensor: np.ndarray) -> "CPFactorization":
        x = np.asarray(tensor, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError("CP factorization requires a 3-D tensor")
        if any(n == 0 for n in x.shape):
            raise ValueError("tensor dimensions must be nonzero")
        rng = np.random.default_rng(self.random_state)
        a = rng.standard_normal((x.shape[0], self.rank))
        b = rng.standard_normal((x.shape[1], self.rank))
        c = rng.standard_normal((x.shape[2], self.rank))
        norm = float(np.linalg.norm(x))
        previous = np.inf

        for _ in range(self.max_iter):
            a = self._update(x, b, c, mode=0)
            b = self._update(x, a, c, mode=1)
            c = self._update(x, a, b, mode=2)
            reconstructed = np.einsum("ir,jr,kr->ijk", a, b, c)
            relative_error = float(np.linalg.norm(x - reconstructed) / max(norm, np.finfo(float).eps))
            if abs(previous - relative_error) <= self.tolerance:
                break
            previous = relative_error

        self.factors = (a, b, c)
        self.relative_error = relative_error
        return self

    def fit_transform(self, tensor: np.ndarray) -> CPDecomposition:
        self.fit(tensor)
        if self.factors is None:
            raise RuntimeError("factorization did not converge")
        return CPDecomposition(self.factors, self.rank, self.relative_error)

    @staticmethod
    def _update(
        tensor: np.ndarray,
        factor1: np.ndarray,
        factor2: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        gram = (factor1.T @ factor1) * (factor2.T @ factor2)
        gram += 1e-12 * np.eye(gram.shape[0])
        unfolded = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        if mode == 0:
            khatri = np.einsum("jr,kr->jkr", factor1, factor2).reshape(-1, factor1.shape[1])
        elif mode == 1:
            khatri = np.einsum("ir,kr->ikr", factor1, factor2).reshape(-1, factor1.shape[1])
        else:
            khatri = np.einsum("ir,jr->ijr", factor1, factor2).reshape(-1, factor1.shape[1])
        return np.linalg.solve(gram, khatri.T @ unfolded.T).T

    def reconstruct(self) -> np.ndarray:
        if self.factors is None:
            raise RuntimeError("call fit before reconstruct")
        a, b, c = self.factors
        return np.einsum("ir,jr,kr->ijk", a, b, c)


@dataclass(frozen=True)
class EncodedTrajectory:
    """Transportable encoded trajectory with exact decoding metadata."""

    payload: bytes
    shape: Tuple[int, ...]
    dtype: str
    mode: str
    spacing: Optional[float] = None
    base_dtype: Optional[str] = None


class LosslessTrajectoryCoder:
    """Losslessly encode trajectory arrays by reversible float-bit residual coding.

    The implementation stores a deterministic first-state representation and
    stores exact residual integer words using per-frame uint64/int64 views.
    Zlib only compresses these reversible integers; no floating-point
    subtraction/readdition is used during decoding.
    """

    def encode(self, trajectory: np.ndarray) -> EncodedTrajectory:
        x = np.asarray(trajectory)
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError("trajectory must contain floating-point values")
        dtype = x.dtype
        if dtype.itemsize not in (4, 8):
            raise TypeError("only float32 or float64 trajectories are supported")
        dtype_name = dtype.name
        if dtype_name == "float32":
            words = x.view(np.uint32).copy()
            residual = np.empty(words.size, dtype=np.uint32)
            residual[0] = words.ravel()[0]
            residual[1:] = words.ravel()[1:] - words.ravel()[:-1]
        else:
            words = x.view(np.uint64).copy()
            residual = np.empty(words.size, dtype=np.uint64)
            residual[0] = words.ravel()[0]
            residual[1:] = words.ravel()[1:] - words.ravel()[:-1]
        payload = zlib.compress(residual.tobytes(order="C"), level=9)
        return EncodedTrajectory(
            payload=payload,
            shape=x.shape,
            dtype=dtype_name,
            mode="lossless",
        )

    def decode(self, encoded: EncodedTrajectory) -> np.ndarray:
        if encoded.mode != "lossless":
            raise ValueError("encoded trajectory is not in lossless mode")
        dtype = np.dtype(encoded.dtype)
        if dtype == np.dtype("float32"):
            residual = np.frombuffer(zlib.decompress(encoded.payload), dtype=np.uint32)
            if residual.size == 0:
                raise ValueError("empty payload")
            words = np.empty_like(residual)
            words[0] = residual[0]
            np.cumsum(residual, out=words)
            out = words.view(np.float32)
        elif dtype == np.dtype("float64"):
            residual = np.frombuffer(zlib.decompress(encoded.payload), dtype=np.uint64)
            if residual.size == 0:
                raise ValueError("empty payload")
            words = np.empty_like(residual)
            words[0] = residual[0]
            np.cumsum(residual, out=words)
            out = words.view(np.float64)
        else:
            raise ValueError("unsupported encoded dtype")
        return out.reshape(encoded.shape).copy()


class QuantizedTrajectoryCoder:
    """Quantize coordinates to a fixed grid, then losslessly encode integers."""

    def __init__(self, spacing: float) -> None:
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        self.spacing = float(spacing)

    def quantize(self, trajectory: np.ndarray) -> np.ndarray:
        x = np.asarray(trajectory)
        if not np.issubdtype(x.dtype, np.floating):
            raise TypeError("trajectory must contain floating-point values")
        return np.rint(x / self.spacing).astype(np.int64)

    def encode(self, trajectory: np.ndarray) -> EncodedTrajectory:
        x = np.asarray(trajectory)
        ints = self.quantize(x)
        payload = zlib.compress(ints.tobytes(order="C"), level=9)
        return EncodedTrajectory(
            payload=payload,
            shape=ints.shape,
            dtype="int64",
            mode="quantized",
            spacing=self.spacing,
            base_dtype=x.dtype.name,
        )

    def decode(self, encoded: EncodedTrajectory) -> np.ndarray:
        if encoded.mode != "quantized" or encoded.spacing is None:
            raise ValueError("encoded trajectory is not in quantized mode")
        size = int(np.prod(encoded.shape, dtype=np.int64))
        raw = zlib.decompress(encoded.payload)
        ints = np.frombuffer(raw, dtype=np.int64, count=size)
        return (ints.reshape(encoded.shape) * encoded.spacing).copy()


@dataclass
class HarmonicSystem:
    """A symmetric harmonic system in mass-weighted coordinates."""

    dynamical_matrix: np.ndarray

    def __post_init__(self) -> None:
        d = np.asarray(self.dynamical_matrix, dtype=np.float64)
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("dynamical_matrix must be square")
        if not np.allclose(d, d.T):
            raise ValueError("dynamical_matrix must be symmetric")
        self.dynamical_matrix = d
        eigenvalues, eigenvectors = np.linalg.eigh(d)
        tolerance = np.max(np.abs(eigenvalues)) * 1e-10 if eigenvalues.size else 0.0
        self.eigenvalues = eigenvalues
        self.eigenvalues_safe = np.maximum(eigenvalues, 0.0)
        self.frequencies = np.sqrt(self.eigenvalues_safe)
        self.modes = eigenvectors
        self.zero_mode_mask = eigenvalues < -tolerance
        if np.any(eigenvalues < -tolerance):
            raise ValueError("dynamical matrix has a negative vibrational eigenvalue")

    def propagate(self, q0: np.ndarray, v0: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q0, dtype=np.float64).reshape(-1)
        v = np.asarray(v0, dtype=np.float64).reshape(-1)
        if q.shape != v.shape or q.shape[0] != self.dynamical_matrix.shape[0]:
            raise ValueError("q0 and v0 must match the dynamical matrix dimension")
        q_modes = self.modes.T @ q
        v_modes = self.modes.T @ v
        omega = self.frequencies
        cos_t = np.cos(omega * time)
        sin_t = np.sin(omega * time)
        inverse_omega = np.empty_like(sin_t)
        inverse_omega[omega > 0] = sin_t[omega > 0] / omega[omega > 0]
        inverse_omega[omega == 0] = time
        q_modes_next = cos_t * q_modes + inverse_omega * v_modes
        v_modes_next = -omega * sin_t * q_modes + cos_t * v_modes
        return self.modes @ q_modes_next, self.modes @ v_modes_next


@dataclass
class AnharmonicEnergy:
    """Scalar anharmonic energy and its conservative coordinate gradient."""

    quadratic: np.ndarray
    cubic: Optional[np.ndarray] = None
    quartic: Optional[np.ndarray] = None
    constant: float = 0.0

    def __post_init__(self) -> None:
        k = np.asarray(self.quadratic, dtype=np.float64)
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError("quadratic Hessian must be square")
        self.quadratic = (k + k.T) * 0.5
        if self.cubic is not None:
            self.cubic = np.asarray(self.cubic, dtype=np.float64)
        if self.quartic is not None:
            self.quartic = np.asarray(self.quartic, dtype=np.float64)

    def energy(self, q: np.ndarray) -> float:
        x = np.asarray(q, dtype=np.float64).reshape(-1)
        if x.size != self.quadratic.shape[0]:
            raise ValueError("coordinate dimension mismatch")
        result = self.constant + 0.5 * x @ self.quadratic @ x
        if self.cubic is not None:
            result += np.einsum("ijk,i,j,k->", self.cubic, x, x, x) / 6.0
        if self.quartic is not None:
            result += np.einsum("ijkl,i,j,k,l->", self.quartic, x, x, x, x) / 24.0
        return float(result)

    def gradient(self, q: np.ndarray) -> np.ndarray:
        x = np.asarray(q, dtype=np.float64).reshape(-1)
        if x.size != self.quadratic.shape[0]:
            raise ValueError("coordinate dimension mismatch")
        result = self.quadratic @ x
        if self.cubic is not None:
            result = result + 0.5 * np.einsum("ijk,j,k->i", self.cubic, x, x)
        if self.quartic is not None:
            result = result + (1.0 / 6.0) * np.einsum("ijkl,j,k,l->i", self.quartic, x, x, x)
        return result

    def forces(self, q: np.ndarray) -> np.ndarray:
        return -self.gradient(q)
