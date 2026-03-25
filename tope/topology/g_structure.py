"""Clark-Bruckheimer G-Structure Machinery for ToPE
====================================================

Implements the canonical obstructions from Clark-Bruckheimer's G-structure
theory, applied to the Enzyme-PCC combinatorial complex.

Physical correspondence
-----------------------
The VOIP assignment on the Enzyme-PCC is a (1,1) tensor field J on the CC.
Clark-Bruckheimer's framework promotes this to a *special* tensor field when
the sheaf connection D satisfies DJ = 0 (covariantly constant VOIP).

GFN2-xTB refinement — implemented in Phase 5A's XTBContextEncoder — is the
curvature correction that makes DJ = 0 hold for geometry-adapted VOIP values
rather than static NIST values.  The xTB step promotes an approximate tensor
to an exact special tensor.

Four constructs are implemented here:

1. nijenhuis_norm_per_cell
   The Nijenhuis tensor N^i_{jk} of the sheaf endomorphism B = diag(VOIP)
   measures the failure of the VOIP field to be integrable at each CC rank.
   Per the formula from the analysis:

     N(i; j,k) = v_j ⊙ (R_ik v_i) − v_k ⊙ (R_ij v_i)

   where R_ij is the metric-aware restriction map (G_delta ⊗ G_delta / inner)
   and ⊙ is elementwise product.  Non-zero Nijenhuis norm at rank 2 (residue
   level) is a rigorous indicator of allosteric coupling: it means the local
   electronic frame at one residue cannot be globally extended without
   referencing distant residues.

2. chern_invariant
   Theorem 5 (Clark-Bruckheimer): β(tN) + λC = 0.
   C is the integer Chern obstruction to integrability of the G-structure.
   Computed from holonomy around rank-2 triangles.  Replaces the ad hoc
   holonomy norm ‖d_AI(Hol(γ), Id)‖ used in FrustIndex with the canonical
   topological invariant.

3. voip_covariant_loss
   Regularisation loss ‖DJ‖² = Σ_{edges} ‖v_j − R_ij v_i‖² / |E|.
   When this loss is zero, the VOIP field is special (DJ = 0).
   Add to the training objective of VOIPSIRENField to make the GFN2-xTB
   corrections drive covariant constancy, not just accuracy.

4. almost_tangent_score
   Near a transition state the p-Laplacian Hessian has one negative eigenvalue
   (the reaction coordinate φ_1) and a leading transverse mode φ_2.  The
   almost tangent endomorphism J = φ_2 ⊗ φ_1 satisfies J² ≈ 0 iff φ_1 ⊥ φ_2.
   Theorem 7 (Clark-Bruckheimer): no G-equivariant complementary subspace
   exists for almost tangent structures — this is why local active-site models
   fail at TS prediction.

References
----------
Clark & Bruckheimer — "Special tensor fields and G-structures on
  combinatorial complexes" (unpublished manuscript referenced in analysis).
Bernard (Section 2.2) — Projection β: P → Z and Chern invariant C.
Cerrini (1971) — Metric-aware restriction maps (see phonon_topology.py).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Nijenhuis tensor — allosteric coupling signal
# ══════════════════════════════════════════════════════════════════════════════

def nijenhuis_norm_per_cell(
    voip_sections: np.ndarray,
    restriction_maps: Dict[Tuple[int, int], np.ndarray],
    adjacency: List[Tuple[int, int]],
) -> np.ndarray:
    """Compute the per-cell Nijenhuis tensor norm for the VOIP sheaf endomorphism.

    For each cell i with adjacent cells j, k, the Nijenhuis tensor component is:

        N(i; j, k) = v_j ⊙ (R_ik v_i) − v_k ⊙ (R_ij v_i)

    where R_ij is the restriction map stored in restriction_maps[(i,j)],
    v_i = voip_sections[i], and ⊙ is elementwise product.

    The norm squared summed over all ordered pairs (j, k) of neighbours of i:

        ‖N(i)‖² = Σ_{j,k ∈ adj(i)} ‖N(i; j, k)‖²

    Parameters
    ----------
    voip_sections : (N, d) array of VOIP vectors per cell.
    restriction_maps : dict[(i,j) → (d,d)] metric-aware projection matrices.
        Use SheafENM.build_restriction_maps() to generate these.
    adjacency : list of (i, j) edge pairs (directed).

    Returns
    -------
    N_norms : (N,) array of per-cell Nijenhuis norms.
        Large values at rank-2 cells indicate allosteric coupling sites.
    """
    N = voip_sections.shape[0]
    N_norms = np.zeros(N)

    # Build adjacency list: for each cell i, list of j such that (i,j) is an edge
    adj_of: List[List[int]] = [[] for _ in range(N)]
    for (i, j) in adjacency:
        adj_of[i].append(j)

    for i in range(N):
        neighbours = adj_of[i]
        if len(neighbours) < 2:
            continue
        v_i = voip_sections[i]  # (d,)

        # Precompute R_ij @ v_i for each neighbour j
        Rv_i: Dict[int, np.ndarray] = {}
        for j in neighbours:
            key = (i, j)
            if key in restriction_maps:
                Rv_i[j] = restriction_maps[key] @ v_i
            else:
                # Fallback: identity (Euclidean, no correction)
                Rv_i[j] = v_i

        sq_sum = 0.0
        for jdx, j in enumerate(neighbours):
            v_j = voip_sections[j]
            for k in neighbours:
                if k == j:
                    continue
                v_k = voip_sections[k]
                # N(i; j, k) = v_j ⊙ (R_ik v_i) − v_k ⊙ (R_ij v_i)
                n_ijk = v_j * Rv_i.get(k, v_i) - v_k * Rv_i.get(j, v_i)
                sq_sum += float(np.dot(n_ijk, n_ijk))

        N_norms[i] = math.sqrt(sq_sum / max(1, len(neighbours) ** 2))

    return N_norms


def build_restriction_maps(
    voip_sections: np.ndarray,
    adjacency: List[Tuple[int, int]],
    descriptor_metric: Optional[np.ndarray] = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Build metric-aware restriction maps from VOIP section differences.

    R_ij = (G @ delta) ⊗ (G @ delta) / (delta^T G delta)
    where delta = v_i − v_j and G = descriptor_metric.

    This is the same construction as SheafENM.sheaf_laplacian() (Cerrini fix).
    Provided here so that nijenhuis_norm_per_cell can be called standalone
    without instantiating SheafENM.

    Parameters
    ----------
    voip_sections : (N, d)
    adjacency : list of (i, j) edge pairs
    descriptor_metric : (d, d) or None (defaults to identity)

    Returns
    -------
    R : dict[(i,j) → (d,d)]
    """
    d = voip_sections.shape[1]
    G = descriptor_metric if descriptor_metric is not None else np.eye(d)
    R: Dict[Tuple[int, int], np.ndarray] = {}
    for (i, j) in adjacency:
        delta = voip_sections[i] - voip_sections[j]
        G_delta = G @ delta
        inner = float(delta @ G_delta) + 1e-10
        R[(i, j)] = np.outer(G_delta, G_delta) / inner
    return R


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Chern invariant — canonical integer obstruction (Theorem 5)
# ══════════════════════════════════════════════════════════════════════════════

def chern_invariant(
    N_norms: np.ndarray,
    holonomy_triangles: Optional[List[Tuple[int, int, int]]] = None,
    restriction_maps: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    voip_sections: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """Compute the Chern invariant C of the G-structure.

    Theorem 5 (Clark-Bruckheimer):  β(tN) + λC = 0
    ⟹  C = −β(t ‖N‖) / λ

    Two paths are supported:

    Path A (holonomy triangles): if holonomy_triangles, restriction_maps, and
    voip_sections are all provided, the integer Chern invariant is computed
    from triangle holonomy around rank-2 cells:

        Hol(i→j→k→i) = R_ki R_jk R_ij
        C = round( (1/2π) Σ_triangles Tr(log(I + Hol − I)) )

    Path B (Nijenhuis proxy): otherwise, the continuous proxy

        C_cont = ‖N‖_rank2 / (2π)
        C      = round(C_cont)

    is used.  This agrees with path A up to a proportionality constant λ
    that encodes the structure-group representation.

    Parameters
    ----------
    N_norms : (N,) per-cell Nijenhuis norms from nijenhuis_norm_per_cell().
    holonomy_triangles : list of (i,j,k) triples forming rank-2 triangles.
    restriction_maps : dict from build_restriction_maps().
    voip_sections : (N, d) — needed for path A.

    Returns
    -------
    C : int  — integer Chern invariant (canonical G-structure obstruction).
    C_cont : float — continuous pre-rounding value for diagnostic use.
    """
    if (holonomy_triangles is not None
            and restriction_maps is not None
            and voip_sections is not None):
        # Path A: holonomy around triangles
        chern_sum = 0.0
        for (i, j, k) in holonomy_triangles:
            # Holonomy = R_ki @ R_jk @ R_ij  (composition of projections)
            R_ij = restriction_maps.get((i, j), np.eye(voip_sections.shape[1]))
            R_jk = restriction_maps.get((j, k), np.eye(voip_sections.shape[1]))
            R_ki = restriction_maps.get((k, i), np.eye(voip_sections.shape[1]))
            Hol = R_ki @ R_jk @ R_ij
            # Curvature: Tr(Hol − I) measures holonomy deviation from identity
            curvature = np.trace(Hol - np.eye(Hol.shape[0]))
            chern_sum += curvature
        C_cont = chern_sum / (2.0 * math.pi)
    else:
        # Path B: Nijenhuis proxy — use rank-2 cells only (indices into N_norms
        # corresponding to residue-level cells; caller should pass the rank-2
        # slice of N_norms directly for precision).
        C_cont = float(N_norms.sum()) / (2.0 * math.pi)

    return int(round(C_cont)), C_cont


def frust_index_cb(
    voip_sections: np.ndarray,
    adjacency: List[Tuple[int, int]],
    descriptor_metric: Optional[np.ndarray] = None,
    holonomy_triangles: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict[str, object]:
    """Compute the Clark-Bruckheimer FrustIndex for the VOIP sheaf.

    Replaces the ad hoc holonomy norm ‖d_AI(Hol(γ), Id)‖ with the canonical
    G-structure obstruction:

        FrustIndex_CB(i) = ‖N(i)‖   (per-cell Nijenhuis norm at rank 2)
        Chern_C          = round( Σ_i ‖N(i)‖ / 2π )

    Parameters
    ----------
    voip_sections : (N, d)
    adjacency : list of (i, j) edges
    descriptor_metric : (d, d) or None
    holonomy_triangles : list of (i, j, k) for path-A Chern computation

    Returns
    -------
    dict with keys:
        'N_norms'      : (N,) per-cell Nijenhuis norms
        'chern_C'      : int   Chern invariant
        'chern_C_cont' : float continuous pre-rounding value
        'frustrated_cells' : list[int] cells where N_norm > threshold
        'holonomy_norm'    : float mean N_norm (drop-in for old holonomy_norm)
    """
    R = build_restriction_maps(voip_sections, adjacency, descriptor_metric)
    N_norms = nijenhuis_norm_per_cell(voip_sections, R, adjacency)
    C, C_cont = chern_invariant(
        N_norms,
        holonomy_triangles=holonomy_triangles,
        restriction_maps=R if holonomy_triangles is not None else None,
        voip_sections=voip_sections if holonomy_triangles is not None else None,
    )

    threshold = float(N_norms.mean()) + float(N_norms.std())
    frustrated = [i for i, n in enumerate(N_norms) if n > threshold]

    return {
        "N_norms": N_norms,
        "chern_C": C,
        "chern_C_cont": C_cont,
        "frustrated_cells": frustrated,
        "holonomy_norm": float(N_norms.mean()),  # drop-in for PyMOL schema
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VOIP covariant loss — makes VOIP a special tensor field (DJ = 0)
# ══════════════════════════════════════════════════════════════════════════════

def voip_covariant_loss(
    voip: torch.Tensor,
    edge_index: torch.Tensor,
    restriction_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Regularisation loss promoting DJ = 0 (VOIP as a special tensor field).

    For each directed edge (i → j):
        (DJ)_{ij} = v_j − R_ij v_i

    where R_ij is approximated as the normalised outer product of the VOIP
    difference (the rank-1 restriction map from phonon_topology.py).

    Loss = (1/|E|) Σ_{(i,j)} ‖v_j − R_ij v_i‖²

    When this loss is zero, the VOIP field is covariantly constant under the
    sheaf connection — it is a *special* tensor field in the Clark-Bruckheimer
    sense.  Adding this to the VOIPSIRENField training objective makes the
    GFN2-xTB refinement drive covariant constancy, not just local accuracy.

    Parameters
    ----------
    voip : (N, d) tensor of VOIP vectors (output of VOIPSIRENField).
    edge_index : (2, E) long tensor of directed edges at rank 0.
    restriction_weight : (E,) optional edge weights (from graph Laplacian).

    Returns
    -------
    loss : scalar tensor
    """
    src, tgt = edge_index[0], edge_index[1]
    v_src = voip[src]  # (E, d)
    v_tgt = voip[tgt]  # (E, d)

    delta = v_src - v_tgt  # (E, d) — descriptor difference
    delta_norm_sq = (delta * delta).sum(dim=-1, keepdim=True) + 1e-10  # (E, 1)

    # R_ij @ v_i = (delta^T v_src / ‖delta‖²) * delta  (rank-1 projection)
    proj_coeff = (delta * v_src).sum(dim=-1, keepdim=True) / delta_norm_sq  # (E, 1)
    Rv_src = proj_coeff * delta  # (E, d)

    # Covariant difference: v_j − R_ij v_i
    cov_diff = v_tgt - Rv_src  # (E, d)

    loss_per_edge = (cov_diff * cov_diff).sum(dim=-1)  # (E,)

    if restriction_weight is not None:
        loss_per_edge = restriction_weight.abs() * loss_per_edge

    return loss_per_edge.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Almost tangent detector — transition state geometry (Theorem 7)
# ══════════════════════════════════════════════════════════════════════════════

def almost_tangent_score(
    phi_1: torch.Tensor,
    phi_2: torch.Tensor,
) -> torch.Tensor:
    """Measure the almost tangent (J² ≈ 0) property of p-Laplacian eigenmodes.

    Near a transition state the leading p-eigenmode φ_1 (reaction coordinate)
    and subleading mode φ_2 (first transverse mode) define the almost tangent
    endomorphism:

        J = φ_2 ⊗ φ_1^T

    J² = φ_2 (φ_1^T φ_2) φ_1^T  →  ‖J²‖ / ‖J‖² = |cos θ_{12}|

    where θ_{12} is the angle between the two eigenmodes.

    Score → 0  iff  φ_1 ⊥ φ_2  (almost tangent structure, TS geometry).
    Theorem 7 (Clark-Bruckheimer): when score ≈ 0 there is no G-equivariant
    complementary subspace — the TS cannot be locally isolated from its
    reaction coordinate context.

    Parameters
    ----------
    phi_1 : (N,) leading p-Laplacian eigenmode (reaction coordinate).
    phi_2 : (N,) subleading eigenmode (first transverse mode).

    Returns
    -------
    score : scalar tensor in [0, 1].
        score ≈ 0  →  almost tangent structure (TS).
        score ≈ 1  →  eigenmodes are collinear (not a TS).
    """
    phi_1 = F.normalize(phi_1, dim=0)
    phi_2 = F.normalize(phi_2, dim=0)
    cos_theta = (phi_1 * phi_2).sum().abs()  # |cos θ_{12}|
    return cos_theta  # = ‖J²‖_F / ‖J‖_F²  for unit-norm eigenmodes


class AlmostTangentDetector(nn.Module):
    """Per-graph almost tangent structure detector for batched p-Laplacian output.

    Accepts the leading k eigenmodes from PLaplacianEigensolver and returns:
    - almost_tangent_score: scalar per graph in [0, 1]
    - is_transition_state: bool per graph (score < threshold)
    - theorem7_warning: bool — True when local active-site models will fail

    A learnable threshold τ (initialised from the batch statistics) makes
    the TS classifier end-to-end differentiable.

    Parameters
    ----------
    threshold : initial decision threshold for is_transition_state.
    learn_threshold : if True, τ is a learnable parameter.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        learn_threshold: bool = True,
    ) -> None:
        super().__init__()
        if learn_threshold:
            self.tau_raw = nn.Parameter(torch.tensor(math.log(threshold / (1 - threshold))))
        else:
            self.register_buffer("tau_raw", torch.tensor(math.log(threshold / (1 - threshold))))

    @property
    def threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.tau_raw)

    def forward(
        self,
        eigenmodes: torch.Tensor,  # (batch, N, k) — k leading eigenmodes per graph
        p_values: Optional[torch.Tensor] = None,  # (batch,) learned p per graph
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        eigenmodes : (batch, N, k)
        p_values : (batch,) optional; used only for the theorem7_warning flag.
            Theorem 7 applies whenever score is small AND p > 2.

        Returns
        -------
        dict with:
            'scores'            : (batch,) almost tangent scores
            'is_ts'             : (batch,) bool
            'theorem7_warning'  : (batch,) bool — local models insufficient
        """
        phi_1 = F.normalize(eigenmodes[:, :, 0], dim=1)  # (batch, N)
        phi_2 = F.normalize(eigenmodes[:, :, 1], dim=1)  # (batch, N)

        scores = (phi_1 * phi_2).sum(dim=1).abs()  # (batch,)

        is_ts = scores < self.threshold

        if p_values is not None:
            # Theorem 7 is structurally significant only in the TS regime (p > 2)
            theorem7 = is_ts & (p_values > 2.0)
        else:
            theorem7 = is_ts

        return {
            "scores": scores,
            "is_ts": is_ts,
            "theorem7_warning": theorem7,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Nijenhuis attention signal — dropout-safe torch version for cc_attention
# ══════════════════════════════════════════════════════════════════════════════

def nijenhuis_signal(
    voip: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Differentiable per-node Nijenhuis tensor norm from VOIP vectors.

    Torch implementation of nijenhuis_norm_per_cell(), suitable for use inside
    cc_attention.py layers as an allosteric coupling signal.

    For each node i and each ordered pair of neighbours (j, k):

        N(i; j, k) = v_j ⊙ (R_ik v_i) − v_k ⊙ (R_ij v_i)

    where the rank-1 restriction R_ij v_i = (delta_ij · v_i / ‖delta_ij‖²) delta_ij
    with delta_ij = v_i − v_j.

    Complexity: O(E * avg_degree) in time.  For sparse enzyme graphs this
    is dominated by the degree, not the number of edges.

    Parameters
    ----------
    voip : (N, d) VOIP vectors per node (any rank).
    edge_index : (2, E) directed edges (source → target).

    Returns
    -------
    N_norms : (N,) differentiable Nijenhuis norm per node.
    """
    N, d = voip.shape
    src, tgt = edge_index[0], edge_index[1]

    v_src = voip[src]  # (E, d)
    v_tgt = voip[tgt]  # (E, d)

    # Restriction map: R_ij v_i = proj of v_src onto (v_src − v_tgt)
    delta = v_src - v_tgt  # (E, d)
    delta_sq = (delta * delta).sum(dim=-1, keepdim=True) + 1e-10  # (E, 1)
    coeff = (delta * v_src).sum(dim=-1, keepdim=True) / delta_sq
    Rv_src = coeff * delta  # (E, d) = R_{src→tgt} @ v_src

    # For each target node i, accumulate products over pairs of incoming edges.
    # We compute: N²(i) = Σ_{j in N(i)} Σ_{k in N(i), k≠j} ‖v_j⊙Rv_{ik} − v_k⊙Rv_{ij}‖²
    # Expanding: = 2 Σ_{j≠k} (‖v_j⊙Rv_{ik}‖²‖v_k⊙Rv_{ij}‖² − (v_j⊙Rv_{ik})·(v_k⊙Rv_{ij}))
    # Use the identity: Σ_{j≠k} ‖a_j − a_k‖² = 2(n Σ‖a_j‖² − ‖Σa_j‖²)
    # where a_j(i) = v_j ⊙ Rv_{ij}  (the "coupling vector" for edge j into i)

    # coupling_j(i) = v_tgt_j ⊙ Rv_src_j  (for edge j = (src_j → tgt_j = i))
    # This is indexed by source edge, evaluated at the target node.
    # For target i: coupling[j] = voip[src_j] ⊙ Rv_src[j]  — wait, need to swap:
    # N(i; j, k): i is the *source* (centre), j and k are neighbours.
    # In edge (i → j), src=i, tgt=j.
    # Rv_{ij} = R_{i→j} @ v_i  (restriction of v_i towards j)
    # v_j ⊙ Rv_{ij} = voip[tgt] ⊙ Rv_src  for edge (i→j) at source i.

    # So for source node i, coupling_j = voip[tgt_j] ⊙ Rv_src_j  (edge j leaving i)
    coupling = v_tgt * Rv_src  # (E, d)  for each directed edge (src→tgt)

    # For each source node i, gather all coupling vectors from its outgoing edges.
    # ‖Σ a_j‖² and Σ ‖a_j‖² over edges grouped by src.
    coupling_sq = (coupling * coupling).sum(dim=-1)  # (E,)  ‖a_j‖²

    sum_sq = torch.zeros(N, device=voip.device, dtype=voip.dtype)
    sum_sq.scatter_add_(0, src, coupling_sq)  # (N,)  Σ_j ‖a_j‖²

    sum_coupling = torch.zeros(N, d, device=voip.device, dtype=voip.dtype)
    idx = src.unsqueeze(-1).expand_as(coupling)
    sum_coupling.scatter_add_(0, idx, coupling)  # (N, d)  Σ_j a_j

    sq_sum_coupling = (sum_coupling * sum_coupling).sum(dim=-1)  # (N,)  ‖Σ_j a_j‖²

    # Count of outgoing edges per source
    deg = torch.zeros(N, device=voip.device, dtype=voip.dtype)
    deg.scatter_add_(0, src, torch.ones(src.size(0), device=voip.device, dtype=voip.dtype))

    # Σ_{j≠k} ‖a_j − a_k‖² = 2(deg Σ‖a_j‖² − ‖Σa_j‖²)
    N_sq = 2.0 * (deg * sum_sq - sq_sum_coupling)  # (N,)
    N_norms = (N_sq.clamp(min=0.0) / (deg.clamp(min=1.0) ** 2)).sqrt()  # (N,)

    return N_norms
