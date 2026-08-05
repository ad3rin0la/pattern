"""
Grassmannian Pushforward Differential for ToPE
===============================================

Implements the Riemannian pushforward differential (dB)_P on the Grassmannian
Gr(n,k), making incidence matrices B_{r,k} approximate gyromorphisms.

The Grassmannian binary operation (Nguyen NeurIPS 2022):
    U ⊕_gr V = exp_P( PT_{P→I}(log_I(V)) )

The pushforward (dB)_P : T_P Gr(n,k) → T_{B(P)} Gr(n,r) approximates a
gyromorphism to first order, with error O(curvature × distance²).

References
----------
Nguyen NeurIPS 2022 — Gyro-structure of SPD and Grassmann manifolds.
Nguyen ICML 2023 — Building neural networks on matrix manifolds.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sym(A: Tensor) -> Tensor:
    """Symmetrize a matrix: (A + A^T) / 2."""
    return (A + A.transpose(-1, -2)) * 0.5


def _skew(A: Tensor) -> Tensor:
    """Skew-symmetrize a matrix: (A - A^T) / 2."""
    return (A - A.transpose(-1, -2)) * 0.5


def _proj_tangent(V: Tensor, P: Tensor) -> Tensor:
    """Project V onto the horizontal tangent space T_P Gr(n,p).

    T_P Gr(n,p) = { V : VP = PV } (skew of PV, equivalently (I-P)VP + PV(I-P))
    Uses the projector form: proj(V) = (I-P)V P + P V (I-P)
    """
    I = torch.eye(P.size(-1), device=P.device, dtype=P.dtype).expand_as(P)
    ImP = I - P
    return ImP @ V @ P + P @ V @ ImP


# ── Core functions ─────────────────────────────────────────────────────────────

def grassmann_log_at_identity(P: Tensor) -> Tensor:
    """Compute log_I(P) for P ∈ Gr(n,p) in projector form.

    Returns the tangent vector in T_I Gr(n,p) — a skew-symmetric matrix.
    Uses SVD for numerical stability:
        if P = U Σ V^T, then log_I(P) = U arcsin(Σ) V^T

    Parameters
    ----------
    P : (..., n, n)  Symmetric rank-p projector (UU^T)

    Returns
    -------
    log_P : (..., n, n)  Skew-symmetric tangent vector at identity
    """
    # Clamp for numerical stability of arcsin
    U, S, Vh = torch.linalg.svd(P, full_matrices=False)
    S = S.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return U * torch.arcsin(S).unsqueeze(-2) @ Vh


def grassmann_frechet_mean(
    points: List[Tensor],
    weights: Optional[Tensor] = None,
    n_iter: int = 50,
    lr: float = 0.5,
    tol: float = 1e-6,
) -> Tensor:
    """Iterative Riemannian gradient descent for the Fréchet mean.

    Used in compression of restriction map clusters.

    Parameters
    ----------
    points  : list of (n, n) projectors on Gr(n,p)
    weights : (K,) optional non-negative weights (uniform if None)
    n_iter  : maximum iterations
    lr      : step size (0 < lr ≤ 1)
    tol     : convergence tolerance

    Returns
    -------
    mean_P : (n, n) Fréchet mean projector
    """
    K = len(points)
    if weights is None:
        weights = torch.ones(K, device=points[0].device, dtype=points[0].dtype) / K
    else:
        weights = weights / weights.sum()

    P = points[0].clone()

    for _ in range(n_iter):
        # Riemannian gradient: weighted sum of log_P(Q_i)
        grad = torch.zeros_like(P)
        for i, Q in enumerate(points):
            # log_P(Q) via parallel transport: log at identity then PT
            log_Q = grassmann_log_at_identity(Q)
            grad = grad + weights[i] * _proj_tangent(log_Q, P)

        # Geodesic update: P ← exp_P(lr * grad)
        step = lr * grad
        gp = GrassmannPoint(P)
        P_new = gp.exp_map(step)

        if torch.norm(P_new - P) < tol:
            break
        P = P_new

    return P


# ── GrassmannPoint ────────────────────────────────────────────────────────────

class GrassmannPoint:
    """Wraps an orthogonal projector P = UU^T ∈ Gr(n,p).

    Parameters
    ----------
    P : (n, n) symmetric rank-p projector
    """

    def __init__(self, P: Tensor) -> None:
        self.P = P

    def log_map(self, Q: Tensor) -> Tensor:
        """Riemannian log map: log_P(Q) → tangent vector in T_P Gr.

        Computes the geodesic direction from P toward Q.

        Parameters
        ----------
        Q : (n, n) target projector

        Returns
        -------
        V : (n, n) tangent vector at P
        """
        # Parallel-transport log_I(Q) from I to P
        log_Q = grassmann_log_at_identity(Q)
        return _proj_tangent(log_Q, self.P)

    def exp_map(self, V: Tensor) -> Tensor:
        """Riemannian exp map: exp_P(V) → new projector.

        Moves from P along tangent V by unit time.

        Parameters
        ----------
        V : (n, n) tangent vector at P (horizontal component)

        Returns
        -------
        Q : (n, n) new projector
        """
        # Geodesic via matrix exponential on skew-symmetric generator
        # exp_P(V) = exp([V, P]) · P · exp(-[V, P])
        # Simplified: use retraction exp(V - V^T) applied to P
        n = self.P.size(-1)
        generator = _skew(V @ self.P - self.P @ V)
        # Matrix exponential via torch.linalg.matrix_exp
        R = torch.linalg.matrix_exp(generator)
        Q = R @ self.P @ R.transpose(-1, -2)
        # Re-symmetrize for numerical stability
        return _sym(Q)

    def parallel_transport(self, V: Tensor, target_P: Tensor) -> Tensor:
        """Parallel transport tangent vector V from P to target_P.

        Approximation via horizontal lift (first-order for small geodesic).

        Parameters
        ----------
        V       : (n, n) tangent vector at P
        target_P: (n, n) destination projector

        Returns
        -------
        V_transported : (n, n) tangent vector at target_P
        """
        return _proj_tangent(V, target_P)

    def inner_product(self, Q: Tensor) -> Tensor:
        """Grassmannian inner product <P,Q>_gr = tr(P · log_I(Q)).

        Parameters
        ----------
        Q : (n, n) other projector

        Returns
        -------
        scalar inner product
        """
        log_Q = grassmann_log_at_identity(Q)
        return torch.trace(self.P @ log_Q)


# ── RiemannianPushforward ────────────────────────────────────────────────────

class RiemannianPushforward:
    """Riemannian pushforward (dB)_P for an incidence matrix B.

    Constructor takes B_matrix (N_target × N_source) and current manifold
    point P ∈ Gr(n,k).  The forward evaluates (dB)_P on a tangent vector.

    Key property: stores B as a matrix but evaluates it geometrically —
    matrix multiplication that behaves as a gyromorphism.

    Parameters
    ----------
    B_matrix     : (N_target, N_source) incidence / aggregation matrix
    current_point: (n, n) projector P ∈ Gr(n, k) or (N_source, d) feature matrix
    """

    def __init__(self, B_matrix: Tensor, current_point: Tensor) -> None:
        self.B = B_matrix
        self.P = current_point

    def forward(self, cochain: Tensor) -> Tensor:
        """Evaluate (dB)_P on a tangent cochain.

        Implementation:
            (dB)_P(V) = proj_{T_{B(P)} Gr} ( B @ V )

        where proj is horizontal projection onto the target tangent space.

        Parameters
        ----------
        cochain : (N_source, d) tangent cochain (source features in Gr tangent space)

        Returns
        -------
        K_t : (N_target, d) pushed-forward features projected to target tangent space
        """
        # Ambient aggregation
        K_ambient = self.B @ cochain   # (N_target, d)

        # Build target projector from pushed-forward point
        B_P = self.B @ self.P          # (N_target, ...) — target manifold point
        # Compute target projector via SVD
        if B_P.dim() == 2:
            U, _, _ = torch.linalg.svd(B_P, full_matrices=False)
            P_target = U @ U.transpose(-1, -2)
        else:
            P_target = B_P

        # Project K_ambient onto T_{B(P)} Gr
        return _proj_tangent(K_ambient, P_target)
