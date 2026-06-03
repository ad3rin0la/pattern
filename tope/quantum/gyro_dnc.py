"""
gyro_dnc.py -- write/read core for the Phase-5G Gyro-DNC.

Memory slots live on the Poincare ball B^W (curvature -c, default c=1). Every
read/write/addressing op is the gyro-analogue of the Graves et al. (2016) DNC:

    content addressing    ->  gyrodistance softmax              (read_weights)
    read aggregation      ->  Einstein gyrocentroid (Klein)     (read)
    erase/add write       ->  Mobius contract + geodesic add    (write)
    usage / allocation    ->  radial coordinate ||M_i||_H       (allocate)
    temporal link matrix  ->  gyration holonomy connection      (frust_index)

The temporal-link refinement is load-bearing: the rotational part of gyro
parallel transport along link edge i->j is gyr[M_j, -M_i] in SO(W). The
composite around a closed cochain loop is a discrete holonomy; its geodesic
distance from the identity is the holonomy frustration index. In the abelian
limit that distance equals the hyperbolic area enclosed by the loop
(Gauss-Bonnet, K=-1) -- the non-integrability of electronic complementarity
that HolonomyDriftSwitch reads as the Wright-drift trigger.

Mobius primitives are implemented locally and norm-guarded so the file is
self-contained; geoopt.PoincareBall is a drop-in for `_Ball` if preferred.

Relation to ``tope.topology.gyro_memory``: that module is the Phase-2 gyro-CC
primitive library (functional ``gyr`` / transport / holonomy FrustIndex over a
combinatorial complex).  This module is the Phase-5G episodic-memory nn.Module;
it keeps its own guarded Mobius primitives by design (the docstring's
"self-contained") so its verified behaviour is independent of that library.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

_EPS = 1e-9
_BOUNDARY = 1.0 - 1e-5  # keep ||x|| <= (1/sqrt(c)) * _BOUNDARY


# ---------------------------------------------------------------------------
# Norm-guarded Mobius primitives on B^W with curvature -c
# ---------------------------------------------------------------------------
class _Ball:
    """Minimal guarded Poincare-ball gyrogroup. Swap for geoopt.PoincareBall."""

    def __init__(self, c: float = 1.0) -> None:
        self.c = float(c)
        self.sqrt_c = math.sqrt(self.c)

    def proj(self, x: Tensor) -> Tensor:
        """Clamp points strictly inside the ball for numerical safety."""
        max_norm = _BOUNDARY / self.sqrt_c
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        scale = (max_norm / norm).clamp(max=1.0)
        return x * scale

    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """x (+)_c y, broadcasting over leading dims."""
        c = self.c
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(_EPS)
        return self.proj(num / den)

    def mobius_neg(self, x: Tensor) -> Tensor:
        return -x

    def mobius_scalar_mul(self, r: "Tensor | float", x: Tensor) -> Tensor:
        """r (x)_c x : move a multiple r along the geodesic through 0 and x."""
        sqrt_c = self.sqrt_c
        norm = x.norm(dim=-1, keepdim=True).clamp(_EPS, _BOUNDARY / sqrt_c)
        ratio = torch.tanh(r * torch.atanh((sqrt_c * norm).clamp(max=_BOUNDARY)))
        return self.proj(ratio / sqrt_c * (x / norm))

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Gyrodistance d_H(x, y)."""
        diff = self.mobius_add(self.mobius_neg(x), y)
        norm = diff.norm(dim=-1).clamp(_EPS, _BOUNDARY / self.sqrt_c)
        return (2.0 / self.sqrt_c) * torch.atanh((self.sqrt_c * norm).clamp(max=_BOUNDARY))

    def logmap0(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(_EPS, _BOUNDARY / self.sqrt_c)
        return torch.atanh((self.sqrt_c * norm).clamp(max=_BOUNDARY)) * x / (self.sqrt_c * norm)

    def expmap0(self, v: Tensor) -> Tensor:
        norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return self.proj(torch.tanh(self.sqrt_c * norm) * v / (self.sqrt_c * norm))

    def gyration(self, a: Tensor, b: Tensor, v: Tensor) -> Tensor:
        """gyr[a,b] v = -(a (+) b) (+) (a (+) (b (+) v)). Linear isometry fixing 0."""
        ab = self.mobius_add(a, b)
        inner = self.mobius_add(a, self.mobius_add(b, v))
        return self.mobius_add(self.mobius_neg(ab), inner)


# ---------------------------------------------------------------------------
# GyroMemory
# ---------------------------------------------------------------------------
class GyroMemory(nn.Module):
    """Episodic memory matrix on B^W with gyro read/write/allocate/holonomy.

    Memory contents are a buffer, not a parameter: written at runtime and reset
    per episode. Slots initialise near the origin = maximally uncommitted (the
    radial coordinate is the commitment / specificity proxy).
    """

    def __init__(self, n_slots: int, mem_dim: int, c: float = 1.0,
                 init_radius: float = 1e-2) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.mem_dim = mem_dim
        self.ball = _Ball(c)
        self.init_radius = init_radius
        self.register_buffer("M", torch.empty(n_slots, mem_dim))
        self.reset_memory()

    # -- state ------------------------------------------------------------
    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        dev = device or self.M.device
        m = torch.randn(self.n_slots, self.mem_dim, device=dev) * self.init_radius
        self.M = self.ball.proj(m)

    # -- content addressing ----------------------------------------------
    def read_weights(self, key: Tensor, beta: "Tensor | float") -> Tensor:
        """w(i) = softmax_i(-beta * d_H(key, M_i)^2). key: (W,) or (B,W)."""
        key = self.ball.proj(key)
        if key.dim() == 1:
            d2 = self.ball.dist(key.unsqueeze(0), self.M) ** 2  # (N,)
            return torch.softmax(-beta * d2, dim=-1)
        d2 = self.ball.dist(key.unsqueeze(1), self.M.unsqueeze(0)) ** 2  # (B,N)
        return torch.softmax(-beta * d2, dim=-1)

    # -- read aggregation (Einstein gyrocentroid in Klein coordinates) ----
    def _einstein_midpoint(self, points: Tensor, weights: Tensor) -> Tensor:
        """Lorentz-weighted gyrocentroid of `points` (N,W) with `weights` (N,)."""
        s2 = (points * points).sum(-1, keepdim=True)             # ||M||^2  (N,1)
        one_minus = (1.0 - self.ball.c * s2).clamp_min(_EPS)
        gamma_u = 2.0 * points / one_minus                       # gamma_i * u_i (Klein)
        gamma = (1.0 + self.ball.c * s2) / one_minus             # Lorentz factor (N,1)
        w = weights.unsqueeze(-1)                                # (N,1)
        num = (w * gamma_u).sum(0)                               # (W,)
        den = (w * gamma).sum(0).clamp_min(_EPS)                 # (1,)
        mK = num / den                                           # Klein midpoint (W,)
        mK2 = (mK * mK).sum(-1, keepdim=True)
        denom = 1.0 + torch.sqrt((1.0 - self.ball.c * mK2).clamp_min(_EPS))
        return self.ball.proj(mK / denom)                        # back to Poincare

    def read(self, weights: Tensor, to_tangent: bool = True) -> Tensor:
        """Read vector = gyrocentroid of slots; returned in T_0 B^W by default."""
        r = self._einstein_midpoint(self.M, weights)
        return self.ball.logmap0(r) if to_tangent else r

    # -- write (Mobius erase-contract then geodesic add) ------------------
    def write(self, content: Tensor, write_weights: Tensor,
              erase_gate: "Tensor | float" = 0.0) -> None:
        """M_i <- (1 - e_i w_i)(x) M_i  (+)  w_i (x) (-M_i (+) content).

        write_weights w_i in [0,1]; erase_gate e_i in [0,1]; content: (W,).
        """
        content = self.ball.proj(content).unsqueeze(0)          # (1,W)
        w = write_weights.view(self.n_slots, 1)                 # (N,1)
        e = (erase_gate if torch.is_tensor(erase_gate)
             else torch.full_like(w, float(erase_gate))).view(self.n_slots, 1)
        # 1. erase = contract slot toward origin by factor (1 - e*w)
        contracted = self.ball.mobius_scalar_mul((1.0 - e * w), self.M)
        # 2. geodesic move a fraction w toward content
        delta = self.ball.mobius_add(self.ball.mobius_neg(contracted),
                                     content.expand_as(contracted))
        step = self.ball.mobius_scalar_mul(w, delta)
        self.M = self.ball.mobius_add(contracted, step)

    # -- allocation (radial coordinate replaces the DNC usage vector) -----
    def allocate(self, sharpness: float = 1.0) -> Tensor:
        """Allocation weight favours low-radius (uncommitted) slots."""
        radius = self.ball.dist(torch.zeros_like(self.M), self.M)  # ||M_i||_H
        return torch.softmax(-sharpness * radius, dim=-1)

    # -- temporal link holonomy (replaces the link matrix L) --------------
    def frust_index(self, cycle: List[int], probe_scale: float = 1e-2,
                    reorthogonalize: bool = False) -> Tensor:
        """Discrete holonomy frustration of a closed slot cycle.

        Hol(C) = prod_k gyr[M_{i_{k+1}}, -M_{i_k}] in SO(W). Since gyration is an
        exact linear isometry, transporting a small in-ball orthonormal frame
        t*I around the loop and dividing by t recovers the rotation matrix R.
        `probe_scale` must be small relative to the slot radii so the Mobius
        realisation stays in its linear regime; shrink it (or set
        reorthogonalize=True, which polar-projects R onto O(W)) for slots near
        the boundary.

        Returns the chordal distance ||Hol - I||_F: conjugation-invariant (so
        base-point independent), differentiable (no eigendecomposition, hence no
        gradient pathology on the W-2 fixed dimensions), and monotone in the
        holonomy angles over [0, pi]. For a single 2-plane rotation by theta it
        equals 2*sqrt(2)*|sin(theta/2)|, so theta is recoverable as
        2*arcsin(||Hol - I||_F / (2*sqrt(2))). In the abelian limit
        theta = enclosed hyperbolic area (Gauss-Bonnet, K=-1).
        """
        if cycle[0] != cycle[-1]:
            cycle = cycle + [cycle[0]]
        W = self.mem_dim
        t = probe_scale
        I = torch.eye(W, device=self.M.device, dtype=self.M.dtype)
        frame = t * I                                                       # rows = t*e_k
        for k in range(len(cycle) - 1):
            a = self.M[cycle[k + 1]].unsqueeze(0)                           # M_j
            b = self.ball.mobius_neg(self.M[cycle[k]]).unsqueeze(0)         # -M_i
            frame = self.ball.gyration(a.expand_as(frame), b.expand_as(frame), frame)
        R = frame / t                                                       # ~orthogonal
        if reorthogonalize:                                                # polar projection
            U, _, Vh = torch.linalg.svd(R)
            R = U @ Vh
        return (R - I).norm()                                              # ||Hol - I||_F
