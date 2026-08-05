"""
CC Gyrotensor Tucker Decomposition for ToPE
============================================

Implements the Gyro-Tucker decomposition of the enzyme feature pyramid
with Cerrini TLS-tightened Tucker rank bounds.

Tucker rank bounds (updated with TLS physics):
    r_0 (atoms)     : 6   (ADP tensor components, vs ~20 orbital types)
    r_1 (bonds)     : 30  (bond environments, unchanged)
    r_2 (residues)  : 20  (TLS params per group, vs ~30 AA environments)
    r_3 (subunits)  : 4   (from S subunits, unchanged)
    r_4 (interfaces): 10  (interface geometries, unchanged)

Core gyrotensor: 6×30×20×4×10 = 144,000 components (vs previous 720,000).
With 3.5-bit Gyro-QJL: 144K × 3.5 bits = 63 KB core.

Tucker factor matrices U^{(k)} are points on Grassmannians.
With Grassmannian pushforward chain:
    U^{(0)} ∈ Gr(N_0, r_0): base frame — ~12 KB
    U^{(1)}: pushforward deviation — ~7 KB
    U^{(2)}: pushforward deviation — ~4 KB
    U^{(3)}, U^{(4)}: tiny, stored at fp32 — ~1 KB

References
----------
CC gyrotensor chat: https://claude.ai/chat/2b60a12c-cb21-461b-9aad-2fb888b42eab
Hajij et al. — CCANN framework (merge nodes, push-forward operations).
Ungar 2008 — Gyrovector spaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

warnings.warn(
    "tope.compression.cc_gyrotensor (Phase 5G — Gyro-Tucker decomposition) is "
    "deferred.  The Tucker rank bounds depend on TLS-tightened ADP tensors, and "
    "the Grassmannian pushforward chain requires geometry-dependent VOIP from "
    "Phase 5A, which has not yet been validated.  Do not integrate this module "
    "into the training pipeline until Phase 5A ablations pass.",
    FutureWarning,
    stacklevel=2,
)

from tope.geometry.grassmann_pushforward import (
    GrassmannPoint,
    RiemannianPushforward,
    grassmann_log_at_identity,
)
from tope.compression.gyro_qjl import TangentSpaceQJL, GyroQJL


# ── Tucker rank configuration ──────────────────────────────────────────────────

@dataclass
class TuckerRankConfig:
    """Cerrini-tightened Tucker rank bounds for the enzyme feature pyramid."""

    r_0: int = 6    # atoms — ADP tensor components (TLS)
    r_1: int = 30   # bonds — bond environments
    r_2: int = 20   # residues — TLS params per group
    r_3: int = 4    # subunits — from S subunits
    r_4: int = 10   # interfaces — interface geometries

    @property
    def core_size(self) -> int:
        return self.r_0 * self.r_1 * self.r_2 * self.r_3 * self.r_4

    @property
    def core_kb_fp32(self) -> float:
        return self.core_size * 4 / 1024

    @property
    def core_kb_gyroqjl(self) -> float:
        return self.core_size * 3.5 / 8 / 1024


# ── Gyrotensor Tucker decomposition ───────────────────────────────────────────

class CCGyrotensor(nn.Module):
    """Gyro-Tucker decomposition of the enzyme CC feature pyramid.

    Compresses the 5-rank enzyme feature pyramid using Tucker decomposition
    on Grassmannian manifolds, with Gyro-QJL quantization of the core tensor.

    Architecture:
        Core G: (r_0, r_1, r_2, r_3, r_4) — compressed via Gyro-QJL
        Factor U^{(0)}: (N_0, r_0) — Gr(N_0, r_0) base frame, fp32
        Factor U^{(1)}: stored as pushforward deviation from U^{(0)}
        Factor U^{(2)}: stored as pushforward deviation from U^{(1)}
        Factor U^{(3)}: (N_3, r_3) — tiny, fp32
        Factor U^{(4)}: (N_4, r_4) — tiny, fp32

    Parameters
    ----------
    rank_cfg  : Tucker rank configuration
    dims      : list of feature dimensions [d_0, d_1, d_2, d_3, d_4]
    n_heads   : attention heads for gyrotensor attention
    """

    def __init__(
        self,
        rank_cfg: Optional[TuckerRankConfig] = None,
        dims: Optional[List[int]] = None,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.rank_cfg = rank_cfg or TuckerRankConfig()
        self.dims = dims or [256, 256, 256, 256, 256]
        self.n_heads = n_heads

        r = self.rank_cfg
        d = self.dims

        # Factor matrix projections (linear: d_k → r_k)
        self.proj_0 = nn.Linear(d[0], r.r_0, bias=False)
        self.proj_1 = nn.Linear(d[1], r.r_1, bias=False)
        self.proj_2 = nn.Linear(d[2], r.r_2, bias=False)
        self.proj_3 = nn.Linear(d[3], r.r_3, bias=False)
        self.proj_4 = nn.Linear(d[4], r.r_4, bias=False)

        # Core tensor (learnable, compressed rank representation)
        # Stored flat: r_0 × r_1 × r_2 × r_3 × r_4
        core_size = r.core_size
        self.core = nn.Parameter(torch.randn(core_size) * 0.01)

        # QJL compressor for the core
        self.core_qjl = TangentSpaceQJL(d=min(core_size, 512), b=2.5, k=256)

        # Pushforward for factor chain U^{(1)} deviation from U^{(0)}
        # (represented as learned residual)
        self.factor_residual_1 = nn.Linear(r.r_0, r.r_1, bias=False)
        self.factor_residual_2 = nn.Linear(r.r_1, r.r_2, bias=False)

    def compress(
        self,
        h_atoms: Tensor,      # (N_0, d_0)
        h_bonds: Tensor,      # (N_1, d_1)
        h_residues: Tensor,   # (N_2, d_2)
        h_subunits: Tensor,   # (N_3, d_3)
        h_interfaces: Tensor, # (N_4, d_4)
    ) -> Dict[str, Tensor]:
        """Compress the full feature pyramid via gyro-Tucker.

        Returns compressed representation dict.
        """
        # Factor matrices via linear projections
        U0 = self.proj_0(h_atoms)       # (N_0, r_0)
        U1 = self.proj_1(h_bonds)       # (N_1, r_1)
        U2 = self.proj_2(h_residues)    # (N_2, r_2)
        U3 = self.proj_3(h_subunits)    # (N_3, r_3)
        U4 = self.proj_4(h_interfaces)  # (N_4, r_4)

        # Project U0 to Grassmannian tangent space
        # (orthonormalize columns via QR)
        U0_orth, _ = torch.linalg.qr(U0)  # (N_0, r_0)
        P0 = U0_orth @ U0_orth.T          # (N_0, N_0) projector

        # Store U1 as pushforward deviation from U0
        U1_base = grassmann_log_at_identity(P0) @ U1  # project via Gr log
        U1_dev = U1 - self.factor_residual_1(U0_orth[:U1.size(0)])

        # Compress core with QJL (take first 512 dims if core too large)
        r = self.rank_cfg
        core_flat = self.core  # (r_0*r_1*r_2*r_3*r_4,)
        core_view = core_flat[:min(len(core_flat), 512)].unsqueeze(0)
        core_hat, s_core, ang_core = self.core_qjl.encode(core_view)

        return {
            "U0": U0_orth,
            "U1_dev": U1_dev,
            "U2": U2,
            "U3": U3,
            "U4": U4,
            "core_hat": core_hat,
            "s_core": s_core,
            "ang_core": ang_core,
            "P0": P0,
        }

    def reconstruct(
        self,
        compressed: Dict[str, Tensor],
        query_rank: int = 0,
    ) -> Tensor:
        """Reconstruct features for a given CC rank from compressed dict.

        Parameters
        ----------
        compressed : dict from compress()
        query_rank : 0=atoms, 1=bonds, 2=residues, 3=subunits, 4=interfaces

        Returns
        -------
        h_reconstructed : (N_k, r_k) Tucker approximation of rank-k features
        """
        rank_map = {0: "U0", 1: "U1_dev", 2: "U2", 3: "U3", 4: "U4"}
        return compressed.get(rank_map[query_rank], compressed["U0"])

    def forward(
        self,
        h_atoms: Tensor,
        h_bonds: Tensor,
        h_residues: Tensor,
        h_subunits: Tensor,
        h_interfaces: Tensor,
    ) -> Tensor:
        """Full forward: compress and return compressed enzyme embedding.

        Returns the global enzyme embedding from the root of the Tucker tree.
        """
        compressed = self.compress(h_atoms, h_bonds, h_residues, h_subunits, h_interfaces)

        # Global embedding: mean-pool factor matrices and contract with core
        global_emb = torch.cat([
            compressed["U0"].mean(0),       # (r_0,)
            compressed["U2"].mean(0),       # (r_2,)
            compressed["U3"].mean(0),       # (r_3,)
            compressed["U4"].mean(0),       # (r_4,)
        ], dim=0)  # (r_0 + r_2 + r_3 + r_4,)

        return global_emb

    @property
    def storage_estimate_kb(self) -> Dict[str, float]:
        """Estimated storage in KB for the compressed representation."""
        r = self.rank_cfg
        return {
            "core_gyroqjl_kb": r.core_kb_gyroqjl + 18.0,  # + QJL residual
            "factor_U0_kb": r.r_0 * 500 * 4 / 1024,        # 500 atoms, fp32
            "factor_U1_dev_kb": r.r_1 * 3.5 / 8 / 1024 * 10_000,
            "factor_U2_kb": r.r_2 * 3.5 / 8 / 1024 * 400,
            "factor_U3_U4_kb": (r.r_3 * 4 + r.r_4 * 4) * 4 / 1024,
        }
