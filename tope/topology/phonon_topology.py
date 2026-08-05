"""
Phonon-Topology Integration for ToPE
=====================================

Integrates insights from Chalopin, Cramer & Arragain (2023), "Phonon-assisted
electron-proton transfer in [FeFe] hydrogenases: Topological role of clusters"
(Biophysical Journal 122, 1557–1567).

Key concepts:
    - Localization landscape u_h: Encodes THz vibrational confinement at each residue
    - Thermal hotspots: Regions with high u_h that confine rate-promoting vibrations
    - Cofactor bridges: FeS clusters that complete topological gaps between hotspots
    - Transfer pathways: Chains of residues at thermal hotspots connected by 3.85Å contacts

This module provides:
    1. Localization landscape computation (cheap: single sparse linear solve)
    2. Multi-rank Hodge Laplacian ENM for combinatorial complex dynamics
    3. Sheaf-valued ENM unifying electronic descriptors with vibrational topology
    4. Cofactor 3-cell construction for explicit topological bridging
    5. Validation utilities for thermal hotspot recovery

References:
    - Chalopin et al. (2023) Biophys. J. 122, 1557–1567
    - Chalopin et al. (2019) Sci. Rep. 9, 12835
    - Chalopin (2020) Sci. Rep. 10, 17465
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    import torch
from scipy.spatial.distance import cdist, pdist, squareform


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhononTopologyConfig:
    """Configuration for phonon topology computations."""

    # Contact network parameters
    contact_cutoff: float = 3.85  # Characteristic Cα spacing (Å)
    contact_tolerance: float = 0.5  # Distance tolerance (Å)

    # Elastic network parameters
    force_constant: float = 1.0  # Uniform γ for basic ENM
    regularization: float = 1e-8  # For numerical stability

    # Cofactor parameters
    cofactor_cutoff: float = 3.85  # Cofactor-residue contact distance
    cofactor_tolerance: float = 0.5

    # Vibrational filtration
    n_vibrational_bins: int = 5  # Number of filtration steps

    # Sheaf ENM
    sheaf_dim: int = 8  # Ioffe descriptor dimension


@dataclass
class PhononTopologyFeatures:
    """Container for phonon-topology feature arrays."""

    localization_landscape: np.ndarray
    vibrational_bins: Optional[np.ndarray] = None
    filtration_values: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None
    hotspot_indices: List[int] = field(default_factory=list)
    contacts: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Localization Landscape Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_localization_landscape(
    ca_coords: np.ndarray,
    cutoff: float = 3.85,
    tolerance: float = 0.5,
    regularization: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
    """
    Compute the phonon localization landscape for a protein backbone.

    Solves L_h @ u_h = 1, where L_h is the dynamic matrix of the
    Cα contact network following Chalopin et al. (2023).

    The localization landscape u_h encodes how much THz vibrational
    energy is confined at each residue position by the fold topology.
    High u_h indicates a "thermal hotspot" where rate-promoting
    vibrations are localized.

    Args:
        ca_coords: (N, 3) array of Cα coordinates
        cutoff: Characteristic Cα spacing (Å), default 3.85
        tolerance: Distance tolerance for contacts (Å), default 0.5
        regularization: Small value to remove null space

    Returns:
        u_h: (N,) localization landscape amplitudes
        contacts: Dict of contact types {'proximal', 'distal', 'total'}
    """
    N = len(ca_coords)
    dist_matrix = squareform(pdist(ca_coords))

    # Build contact network with edge classification
    L_h = lil_matrix((N, N), dtype=np.float64)
    contacts: Dict[str, List[Tuple[int, int]]] = {
        'proximal': [],
        'distal': [],
        'total': [],
    }

    for i in range(N):
        for j in range(i + 1, N):
            d_ij = dist_matrix[i, j]
            if abs(d_ij - cutoff) <= tolerance:
                # Classify contact type by sequence distance
                seq_dist = abs(j - i)
                if seq_dist == 1:
                    contact_type = 'proximal'  # Sequential neighbors
                else:
                    contact_type = 'distal'  # Non-local contacts
                contacts[contact_type].append((i, j))
                contacts['total'].append((i, j))

                # Force constant: uniform γ for basic ENM
                gamma = 1.0
                L_h[i, j] = -gamma
                L_h[j, i] = -gamma

    # Diagonal: sum of off-diagonal entries (degree matrix)
    for i in range(N):
        L_h[i, i] = -L_h[i, :].sum()

    # Solve L_h @ u_h = 1
    L_h_csr = L_h.tocsr()
    ones = np.ones(N)

    # Add small regularization for numerical stability (removes null space)
    L_h_reg = L_h_csr + regularization * np.eye(N)
    u_h = spsolve(L_h_reg, ones)

    return u_h, contacts


def identify_thermal_hotspots(
    u_h: np.ndarray,
    top_k: int = 20,
    threshold: Optional[float] = None,
) -> List[int]:
    """Return residue/atom indices with high localization landscape values."""
    if u_h.size == 0:
        return []
    if threshold is not None:
        return np.where(u_h >= threshold)[0].astype(int).tolist()
    k = min(top_k, u_h.size)
    return np.argsort(u_h)[-k:][::-1].astype(int).tolist()


def compute_cofactor_contacts(
    ca_coords: np.ndarray,
    cofactor_coords: np.ndarray,
    cutoff: float = 3.85,
    tolerance: float = 0.5,
    regularization: float = 1e-8,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Identify cofactor-mediated contacts that bridge topological gaps.

    For FeS clusters, the cofactor center-of-mass or individual Fe/S atoms
    are checked for 3.85 ± 0.5 Å distances to Cα atoms. These contacts
    represent Chalopin's "topological bridges" that complete the vibrational
    network between otherwise disconnected thermal hotspot islands.

    Args:
        ca_coords: (N, 3) Cα coordinates
        cofactor_coords: (M, 3) cofactor atom coordinates
        cutoff, tolerance: Contact criteria
        regularization: For numerical stability

    Returns:
        cofactor_contacts: List of (residue_idx, cofactor_atom_idx) pairs
        enhanced_u_h_residue: (N,) updated landscape for residues
        enhanced_u_h_full: (N+M,) full landscape including cofactors
    """
    dists = cdist(ca_coords, cofactor_coords)

    cofactor_contacts = []
    for i in range(len(ca_coords)):
        for j in range(len(cofactor_coords)):
            if abs(dists[i, j] - cutoff) <= tolerance:
                cofactor_contacts.append((i, j))

    # Rebuild L_h with cofactor contacts included
    N = len(ca_coords)
    M = len(cofactor_coords)

    # Extended system: residues + cofactor atoms
    all_coords = np.vstack([ca_coords, cofactor_coords])
    u_h_extended, _ = compute_localization_landscape(
        all_coords, cutoff, tolerance, regularization
    )

    # Return residue-only portion and full landscape
    return cofactor_contacts, u_h_extended[:N], u_h_extended


def map_residue_to_atom_landscape(
    u_h_residue: np.ndarray,
    residue_to_atoms: Dict[int, List[int]],
    n_atoms: int,
) -> np.ndarray:
    """
    Map residue-level localization landscape to atom-level.

    Each atom inherits the u_h value of its parent residue.
    This is needed for atom-resolution Enzyme-PCC.

    Args:
        u_h_residue: (N_res,) residue-level landscape
        residue_to_atoms: Dict mapping residue index to list of atom indices
        n_atoms: Total number of atoms

    Returns:
        u_h_atom: (n_atoms,) atom-level landscape
    """
    u_h_atom = np.zeros(n_atoms)
    for res_idx, atom_indices in residue_to_atoms.items():
        if res_idx < len(u_h_residue):
            for atom_idx in atom_indices:
                if atom_idx < n_atoms:
                    u_h_atom[atom_idx] = u_h_residue[res_idx]
    return u_h_atom


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Multi-Rank Hodge Laplacian ENM
# ══════════════════════════════════════════════════════════════════════════════

class HodgeLaplacianENM:
    """
    Multi-rank elastic network model using Hodge Laplacians
    on the Enzyme Combinatorial Complex.

    Generalizes the standard Cα ENM to capture collective
    dynamics at bond, residue-cluster, and cofactor-complex scales.

    The Hodge Laplacian at rank k is:
        Δ_k = B_{k-1,k}^T B_{k-1,k} + B_{k,k+1} B_{k,k+1}^T

    where B_{k,k+1} is the boundary operator mapping (k+1)-cells to k-cells.

    Rank | Cells              | Laplacian encodes
    -----|--------------------|-----------------------------------------
    0    | Atoms              | Which atoms confine vibrational energy
    1    | Bonds              | Which contacts are dynamically stiff
    2    | Residue clusters   | Which structural units are vibrational islands
    3    | Cofactor complexes | Which cofactors bridge vibrational islands
    """

    def __init__(
        self,
        enzyme_pcc: Dict[str, Any],
        force_constants: Optional[np.ndarray] = None,
        config: Optional[PhononTopologyConfig] = None,
    ):
        """
        Args:
            enzyme_pcc: Enzyme combinatorial complex with boundary operators
            force_constants: Optional per-edge force constants (default: uniform)
            config: Phonon topology configuration
        """
        self.pcc = enzyme_pcc
        self.force_constants = force_constants
        self.config = config or PhononTopologyConfig()

        # Extract boundary operators from PCC
        self.B_01 = self._get_boundary_operator('boundary_0_1')
        self.B_12 = self._get_boundary_operator('boundary_1_2')
        self.B_23 = self._get_boundary_operator('boundary_2_3')

    def _get_boundary_operator(self, key: str) -> Optional[csr_matrix]:
        """Safely get boundary operator from PCC."""
        B = self.pcc.get(key)
        if B is None:
            return None
        if isinstance(B, np.ndarray):
            return csr_matrix(B)
        return B

    def n_cells(self, rank: int) -> int:
        """Number of cells at given rank."""
        if rank == 0:
            return self.B_01.shape[0] if self.B_01 is not None else 0
        elif rank == 1:
            return self.B_01.shape[1] if self.B_01 is not None else 0
        elif rank == 2:
            return self.B_12.shape[1] if self.B_12 is not None else 0
        elif rank == 3 and self.B_23 is not None:
            return self.B_23.shape[1]
        return 0

    def hodge_laplacian(self, rank: int) -> csr_matrix:
        """
        Compute Hodge Laplacian at given rank.

        Δ_k = B_{k-1,k}^T B_{k-1,k} + B_{k,k+1} B_{k,k+1}^T

        The "down" component (B^T B) captures how k-cells
        are constrained by their boundaries.
        The "up" component (B B^T) captures how k-cells
        are constrained by the higher-order cells they belong to.
        """
        if rank == 0:
            B_down = None
            B_up = self.B_01
        elif rank == 1:
            B_down = self.B_01
            B_up = self.B_12
        elif rank == 2:
            B_down = self.B_12
            B_up = self.B_23
        elif rank == 3:
            B_down = self.B_23
            B_up = None
        else:
            raise ValueError(f"Rank {rank} not supported (max: 3)")

        n = self.n_cells(rank)
        if n == 0:
            raise ValueError(f"No cells at rank {rank}")

        # Down Laplacian: B^T B
        if B_down is not None:
            if self.force_constants is not None:
                # Weight by force constants
                W = self._build_weight_matrix(rank)
                L_down = B_down.T @ W @ B_down
            else:
                L_down = B_down.T @ B_down
        else:
            L_down = csr_matrix((n, n))

        # Up Laplacian: B B^T
        if B_up is not None:
            L_up = B_up @ B_up.T
        else:
            L_up = csr_matrix((n, n))

        return L_down + L_up

    def _build_weight_matrix(self, rank: int) -> csr_matrix:
        """Build diagonal weight matrix from force constants."""
        if self.force_constants is None:
            n = self.n_cells(rank)
            return csr_matrix(np.eye(n))

        # Use force constants as diagonal weights
        return csr_matrix(np.diag(self.force_constants[:self.n_cells(rank)]))

    def localization_landscape(self, rank: int) -> np.ndarray:
        """
        Solve Δ_k u_k = 1 at given rank.

        Returns localization amplitudes for all k-cells.
        This is the core computation from Chalopin's theory.
        """
        L_k = self.hodge_laplacian(rank)
        N = L_k.shape[0]

        # Regularize to remove harmonic subspace
        L_reg = L_k + self.config.regularization * csr_matrix(np.eye(N))
        u_k = spsolve(L_reg, np.ones(N))

        return u_k

    def multi_rank_landscape(self) -> Dict[int, Optional[np.ndarray]]:
        """
        Compute localization landscapes at all available ranks.

        Returns:
            landscapes: Dict[rank → u_k array or None if degenerate]
        """
        landscapes = {}
        max_rank = 3 if self.B_23 is not None else 2

        for k in range(max_rank + 1):
            try:
                if self.n_cells(k) > 0:
                    landscapes[k] = self.localization_landscape(k)
                else:
                    landscapes[k] = None
            except Exception:
                # Some ranks may have degenerate Laplacians
                landscapes[k] = None

        return landscapes

    def spectral_gap(self, rank: int, n_eigenvalues: int = 10) -> Tuple[np.ndarray, float]:
        """
        Compute eigenvalues and spectral gap of Hodge Laplacian.

        The spectral gap (λ₂ - λ₁) indicates how strongly connected
        the vibrational network is at this rank.

        Args:
            rank: Cell rank
            n_eigenvalues: Number of eigenvalues to compute

        Returns:
            eigenvalues: Smallest eigenvalues
            gap: Spectral gap (λ₂ - λ₁)
        """
        L_k = self.hodge_laplacian(rank)

        # For small matrices, use dense eigendecomposition
        if L_k.shape[0] < 500:
            L_dense = L_k.toarray()
            eigenvalues = np.linalg.eigvalsh(L_dense)[:n_eigenvalues]
        else:
            # Use sparse eigensolver
            from scipy.sparse.linalg import eigsh
            eigenvalues, _ = eigsh(L_k, k=min(n_eigenvalues, L_k.shape[0] - 2),
                                   which='SM')
            eigenvalues = np.sort(eigenvalues)

        gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
        return eigenvalues, gap


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Sheaf-Valued Elastic Network Model
# ══════════════════════════════════════════════════════════════════════════════

class SheafENM(HodgeLaplacianENM):
    """
    Sheaf-valued elastic network model.

    Each edge carries a tensor-valued force constant Γ_ij ∈ R^{d×d}
    where d is the sheaf stalk dimension (8 for Ioffe descriptors).

    The dynamic equation becomes a sheaf Laplacian eigenvalue problem:
        Δ_sheaf @ V = V @ Λ

    This encodes how electronic properties (VOIP, electronegativity)
    propagate through the mechanical network — unifying Ioffe's
    electronic descriptors with Chalopin's vibrational topology.
    """

    def __init__(
        self,
        enzyme_pcc: Dict[str, Any],
        sheaf_dim: int = 8,
        config: Optional[PhononTopologyConfig] = None,
        descriptor_metric: Optional[np.ndarray] = None,
        tls_groups: Optional[List[Any]] = None,
        atom_coords: Optional[np.ndarray] = None,
        atom_to_group: Optional[Dict[int, int]] = None,
        use_hsh: bool = False,
        hsh_config: Optional["Any"] = None,
    ):
        super().__init__(enzyme_pcc, config=config)
        self.sheaf_dim = sheaf_dim
        # Descriptor-space metric G: encodes the natural inner product in the
        # stalk space.  Off-diagonal units (VOIP in eV, CN dimensionless,
        # χ in Pauling units, …) make the Euclidean inner product non-invariant;
        # supplying a physics-motivated diagonal G corrects this (Cerrini 1971).
        # Defaults to identity, which reproduces the previous Euclidean behaviour.
        self.descriptor_metric: np.ndarray = (
            descriptor_metric if descriptor_metric is not None else np.eye(sheaf_dim)
        )
        # TLS Hirshfeld constraint: for intra-subunit bonds, restrict to plane ⊥ bond.
        # Reduces restriction map storage from 64 floats (8×8) to 3 floats (n_hat).
        self.tls_groups: Optional[List[Any]] = tls_groups
        self.atom_coords: Optional[np.ndarray] = atom_coords
        self.atom_to_group: Optional[Dict[int, int]] = atom_to_group or {}

        # Phase-2 HSH restriction maps (Avery 1994).  When enabled, the rank-1
        # Cerrini outer product is replaced by an HSH expansion in the 36-element
        # basis of Sym(R^d).  With the default ``init_mode="rank1"`` and an
        # identity descriptor metric the map is byte-identical to the previous
        # construction (zero-diff init).  Non-identity G induces a small
        # normalisation difference — see hsh_restriction_maps.HSHSheafLaplacian.
        self.use_hsh: bool = use_hsh
        self._hsh: Optional[Any] = None
        if use_hsh:
            from tope.topology.hsh_restriction_maps import (
                HSHConfig as _HSHConfig,
                HSHSheafLaplacian as _HSHSheafLaplacian,
            )
            cfg = hsh_config if hsh_config is not None else _HSHConfig(d=sheaf_dim)
            self._hsh = _HSHSheafLaplacian(
                d=sheaf_dim,
                descriptor_metric=self.descriptor_metric,
                config=cfg,
            )

    def sheaf_laplacian(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Compute sheaf Laplacian at given rank.

        The restriction maps along edges are constructed from
        the difference in Ioffe descriptors between connected cells.

        Args:
            rank: Cell rank (0, 1, 2)
            sheaf_sections: Dict[cell_id → R^d feature vector]

        Returns:
            L_sheaf: (N*d, N*d) block-structured Laplacian
        """
        L_k = self.hodge_laplacian(rank)
        N = L_k.shape[0]
        d = self.sheaf_dim

        # Build block Laplacian: each scalar entry L_k[i,j]
        # becomes a d×d block encoding the sheaf restriction map
        L_sheaf = np.zeros((N * d, N * d))

        L_k_dense = L_k.toarray() if hasattr(L_k, 'toarray') else L_k
        rows, cols = np.where(L_k_dense != 0)

        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            w = L_k_dense[i, j]

            if i != j and i in sheaf_sections and j in sheaf_sections:
                # Off-diagonal: restriction map from sheaf sections
                s_i = sheaf_sections[i]
                s_j = sheaf_sections[j]

                # Check if bond is intra-TLS-group (Hirshfeld rigid-bond constraint).
                # For intra-subunit bonds: restriction map = projection onto plane ⊥ bond.
                # Storage: 3 floats (n_hat) instead of 64 (8×8 matrix).
                # Reference: Hirshfeld 1976 — (U_i - U_j)·n̂·n̂^T = 0 constraint.
                use_hirshfeld = (
                    self.tls_groups is not None
                    and self.atom_coords is not None
                    and i < len(self.atom_coords)
                    and j < len(self.atom_coords)
                    and self.atom_to_group.get(i, -1) == self.atom_to_group.get(j, -2)
                    and self.atom_to_group.get(i, -1) >= 0
                )

                if use_hirshfeld:
                    # Hirshfeld: R_ij = I - n̂⊗n̂ (projection onto plane ⊥ to bond)
                    # Embedded in the (d×d) stalk space via Cerrini block structure:
                    # spatial ADP block (first 3 dims), remaining dims use identity.
                    bond_vec = self.atom_coords[j] - self.atom_coords[i]
                    bond_norm = np.linalg.norm(bond_vec)
                    if bond_norm > 1e-7:
                        n_hat = bond_vec / bond_norm
                    else:
                        n_hat = np.array([1.0, 0.0, 0.0])
                    # 3×3 Hirshfeld projection
                    R_spatial = np.eye(3) - np.outer(n_hat, n_hat)
                    # Embed in d×d: spatial block (0:3), identity elsewhere
                    R_ij = np.eye(d)
                    R_ij[:3, :3] = R_spatial
                elif self.use_hsh and self._hsh is not None:
                    # Phase-2 HSH expansion (Avery 1994): replaces the rank-1
                    # outer product with a 36-element basis of Sym(R^d).
                    R_ij = self._hsh.restriction_map(s_i, s_j)
                else:
                    # Restriction map: metric-aware outer product (Cerrini 1971).
                    # G @ delta is the covariant form of the descriptor difference
                    # (index-lowering).  delta^T G delta is the metric-invariant
                    # norm squared; the outer product G_delta ⊗ G_delta gives the
                    # correct rank-1 restriction map that is invariant to separate
                    # scalings of each descriptor axis.
                    delta = s_i - s_j
                    G = self.descriptor_metric
                    G_delta = G @ delta
                    inner = float(delta @ G_delta) + 1e-10
                    R_ij = np.outer(G_delta, G_delta) / inner

                # Scale by graph Laplacian weight
                L_sheaf[i*d:(i+1)*d, j*d:(j+1)*d] = w * (np.eye(d) - R_ij)

        # Fill diagonal blocks: negative sum of off-diagonal blocks
        for i in range(N):
            block_sum = np.zeros((d, d))
            for j in range(N):
                if i != j:
                    block_sum -= L_sheaf[i*d:(i+1)*d, j*d:(j+1)*d]
            L_sheaf[i*d:(i+1)*d, i*d:(i+1)*d] = block_sum

        return L_sheaf

    def build_restriction_maps(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
    ) -> Dict[tuple, np.ndarray]:
        """Return the metric-aware restriction maps as an explicit dict.

        Keys are (i, j) pairs for which a non-zero off-diagonal block exists
        in the sheaf Laplacian.  Values are the (d, d) projection matrices:

            R_ij = (G @ delta) ⊗ (G @ delta) / (delta^T G delta)

        This mirrors the construction in sheaf_laplacian() and is exposed
        so that g_structure.nijenhuis_norm_per_cell() can consume the maps
        without re-computing the Hodge Laplacian.

        Returns
        -------
        R : dict[(i, j) → np.ndarray of shape (d, d)]
        """
        L_k = self.hodge_laplacian(rank)
        d = self.sheaf_dim
        G = self.descriptor_metric
        L_k_dense = L_k.toarray() if hasattr(L_k, "toarray") else L_k
        rows, cols = np.where(L_k_dense != 0)

        R: Dict[tuple, np.ndarray] = {}
        for idx in range(len(rows)):
            i, j = int(rows[idx]), int(cols[idx])
            if i != j and i in sheaf_sections and j in sheaf_sections:
                if self.use_hsh and self._hsh is not None:
                    R[(i, j)] = self._hsh.restriction_map(
                        sheaf_sections[i], sheaf_sections[j]
                    )
                else:
                    delta = sheaf_sections[i] - sheaf_sections[j]
                    G_delta = G @ delta
                    inner = float(delta @ G_delta) + 1e-10
                    R[(i, j)] = np.outer(G_delta, G_delta) / inner
        return R

    # ──────────────────────────────────────────────────────────────────────
    # Phase-2 §5: gyro-transport restriction maps + tangent-space sheaf
    # Laplacian (consolidated spec §5).  These are additive — the existing
    # ``sheaf_laplacian`` / spectral pipeline above is untouched.
    # ──────────────────────────────────────────────────────────────────────

    def _hyperbolic_features(
        self,
        sheaf_sections: Dict[int, np.ndarray],
        hyperbolic_features: Optional[Dict[int, np.ndarray]],
        ball_radius: float,
    ) -> Dict[int, np.ndarray]:
        """Return the ball feature field ``{h_x}`` (spec §5.1).

        If an explicit field is given it is used as-is; otherwise each
        descriptor section is treated as a tangent vector at the origin and
        lifted into the ball via ``h_x = exp_o(s_x)`` — the same field read
        non-linearly by the §2–3 gyro-CCANN head, read here through ``log_o``.
        """
        from tope.topology.gyro_memory import exp_map_origin

        if hyperbolic_features is not None:
            return hyperbolic_features
        return {
            k: exp_map_origin(np.asarray(v, dtype=float), s=ball_radius)
            for k, v in sheaf_sections.items()
        }

    def gyro_restriction_maps(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
        hyperbolic_features: Optional[Dict[int, np.ndarray]] = None,
        ball_radius: float = 1.0,
    ) -> Dict[tuple, np.ndarray]:
        r"""Transport-corrected restriction maps ``F_{x◁y} = K_xy · R_xy`` (§5.2).

        ``K_xy`` is the VOIP-informed stiffness block (the metric-aware /
        HSH map from :meth:`build_restriction_maps`); ``R_xy = gyr[h_y, ⊖h_x]``
        is the gyro-transport rotation (spec §4).  **Dropping ``R_xy`` is the
        silent metric-flattening** the spec flags at this site — the
        Euclidean ``K``-only map is the special case ``h ≡ 0``.

        Returns
        -------
        F : dict[(i, j) → (d, d)]  full restriction maps with transport.
        """
        from tope.topology.gyro_memory import transport_rotation

        K = self.build_restriction_maps(rank, sheaf_sections)
        h = self._hyperbolic_features(sheaf_sections, hyperbolic_features, ball_radius)
        F: Dict[tuple, np.ndarray] = {}
        for (i, j), K_ij in K.items():
            if i in h and j in h:
                R_ij = transport_rotation(h[i], h[j], s=ball_radius)  # gyr[h_j,⊖h_i]
                F[(i, j)] = K_ij @ R_ij
            else:
                F[(i, j)] = K_ij
        return F

    def gyro_sheaf_laplacian(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
        hyperbolic_features: Optional[Dict[int, np.ndarray]] = None,
        ball_radius: float = 1.0,
    ) -> np.ndarray:
        r"""γ-weighted tangent-space sheaf Laplacian ``L = δ* δ`` (spec §5.3–5.4).

        Built in the adjoint-consistent form ``L = D_0^{-1} δ^T D_1 δ`` so it is
        **self-adjoint** w.r.t. the γ-measure inner product ``⟨·,·⟩ = ·^T D_0 ·``
        and has a **real, ≥ 0 spectrum** (PSD), unconditionally at degree 0 —
        no ``∂² = 0`` is required (spec §5.4).  Curvature is front-loaded into
        the chart (``log_o``) and the restriction-map transport ``R_xy``.

        * coboundary  δ : for each undirected edge ``e = (i, j)`` (oriented
          ``i → j``), the block row is ``+I`` at ``i`` and ``−R_ij`` at ``j``,
          where ``R_ij = gyr[h_j, ⊖h_i]`` carries the curvature;
        * stalk measure  ``D_0 = diag(μ(h_i) I_d)``, ``μ(h) = γ_h^{2d}`` (§5.3);
        * coface measure ``D_1 = diag(w_ij I_d)`` from the Hodge edge weight.

        ``ker L`` is the degree-0 sheaf cohomology ``H^0(X; F)``; the low-lying
        spectral gap is the Phase-5H cooperativity descriptor.

        Returns
        -------
        L : (N·d, N·d) self-adjoint PSD sheaf Laplacian.
        """
        from tope.topology.gyro_memory import transport_rotation, conformal_gamma

        L_k = self.hodge_laplacian(rank)
        L_k_dense = L_k.toarray() if hasattr(L_k, "toarray") else np.asarray(L_k)
        N = L_k_dense.shape[0]
        d = self.sheaf_dim

        h = self._hyperbolic_features(sheaf_sections, hyperbolic_features, ball_radius)
        # γ-measure per cell; cells without a section get μ = 1 (Euclidean).
        mu = np.ones(N)
        for i in range(N):
            if i in h:
                gamma = float(conformal_gamma(h[i], s=ball_radius))
                mu[i] = gamma ** (2 * d)

        # Undirected edges with positive coupling weight w_ij = |L_k[i,j]|.
        edges: List[Tuple[int, int, float]] = []
        for i in range(N):
            for j in range(i + 1, N):
                w = abs(float(L_k_dense[i, j]))
                if w > 0.0 and i in h and j in h:
                    edges.append((i, j, w))

        E = len(edges)
        if E == 0:
            return np.zeros((N * d, N * d))

        # Coboundary δ ∈ R^{E·d × N·d}.
        delta = np.zeros((E * d, N * d))
        D1 = np.zeros(E * d)
        for e, (i, j, w) in enumerate(edges):
            R_ij = transport_rotation(h[i], h[j], s=ball_radius)  # gyr[h_j, ⊖h_i]
            rb = slice(e * d, (e + 1) * d)
            delta[rb, i * d:(i + 1) * d] = np.eye(d)
            delta[rb, j * d:(j + 1) * d] = -R_ij
            D1[rb] = w

        # L = D_0^{-1} δ^T D_1 δ  (self-adjoint w.r.t. D_0, PSD).
        D0_inv = np.repeat(1.0 / np.clip(mu, 1e-12, None), d)
        L = (D0_inv[:, None] * delta.T) @ (D1[:, None] * delta)
        return L

    def holonomy_frust_index(
        self,
        loops: List[List[int]],
        sheaf_sections: Dict[int, np.ndarray],
        hyperbolic_features: Optional[Dict[int, np.ndarray]] = None,
        ball_radius: float = 1.0,
    ) -> Dict[str, object]:
        r"""Gyration (holonomy) FrustIndex over cell loops (spec §5.5).

        The Hodge tower needs a *flat* sheaf (``δ_r δ_{r-1} = 0``); failure of
        flatness is the holonomy ``Hol(loop) = ∏ R_xy ≠ I`` — the sheaf
        curvature, which the spec identifies as the FrustIndex carrying
        allostery.  This is the gyro-transport reading; the Nijenhuis/Chern
        reading is :func:`tope.topology.g_structure.frust_index_cb`.

        Returns the dict from
        :func:`tope.topology.gyro_memory.frust_index_gyro`.
        """
        from tope.topology.gyro_memory import frust_index_gyro

        h = self._hyperbolic_features(sheaf_sections, hyperbolic_features, ball_radius)
        keys = sorted(h.keys())
        index_of = {k: idx for idx, k in enumerate(keys)}
        feats = np.stack([h[k] for k in keys])
        remapped = [[index_of[c] for c in loop] for loop in loops]
        return frust_index_gyro(feats, remapped, s=ball_radius)

    def sheaf_endomorphism(
        self,
        sheaf_sections: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Return the VOIP-weighted sheaf endomorphism B for each cell.

        B_i = diag(v_i)  where v_i = sheaf_sections[i] (the VOIP vector).

        This is the (1,1) tensor field J in Clark-Bruckheimer's framework.
        When the GFN2-xTB corrections make D(B) = 0 (covariantly constant),
        B is a *special* tensor field and the G-structure is determined.

        Returns
        -------
        B : (N, d, d) array of per-cell diagonal endomorphisms.
            B[i] = np.diag(sheaf_sections[i]).
        """
        keys = sorted(sheaf_sections.keys())
        d = self.sheaf_dim
        B = np.zeros((len(keys), d, d))
        for idx, k in enumerate(keys):
            B[idx] = np.diag(sheaf_sections[k])
        return B

    def sheaf_localization_landscape(
        self,
        rank: int,
        sheaf_sections: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Solve sheaf Laplacian system for localization.

        Returns (N, d) matrix where each row is the localization
        of the d-dimensional sheaf section at that cell.
        """
        L_sheaf = self.sheaf_laplacian(rank, sheaf_sections)
        N = L_sheaf.shape[0] // self.sheaf_dim
        d = self.sheaf_dim

        # Regularize and solve
        L_reg = L_sheaf + self.config.regularization * np.eye(N * d)
        ones = np.ones(N * d)
        u_sheaf = np.linalg.solve(L_reg, ones)

        # Reshape to (N, d)
        return u_sheaf.reshape(N, d)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Cofactor 3-Cell Construction
# ══════════════════════════════════════════════════════════════════════════════

# Known cofactor descriptors from literature
COFACTOR_DESCRIPTORS = {
    'Fe4S4': {
        'redox_potential_range': (-0.5, 0.1),  # V vs SHE
        'spin_state': 'mixed_valence',
        'n_electrons': 20,  # Total d-electrons
        'voip_effective': 8.5,  # Weighted average of Fe/S VOIP
        'cluster_diameter': 5.7,  # Å
        'coupling_type': 'superexchange',
    },
    'Fe2S2': {
        'redox_potential_range': (-0.4, 0.0),
        'spin_state': 'antiferromagnetic',
        'n_electrons': 10,
        'voip_effective': 8.2,
        'cluster_diameter': 4.1,
        'coupling_type': 'direct_exchange',
    },
    'H_cluster': {
        'redox_potential_range': (-0.4, -0.1),
        'spin_state': 'mixed',
        'n_electrons': 26,  # Fe4S4 + Fe2 subcluster
        'voip_effective': 8.8,
        'cluster_diameter': 8.5,  # Including ADT bridge
        'coupling_type': 'through_bond',
    },
    'heme': {
        'redox_potential_range': (-0.3, 0.4),
        'spin_state': 'variable',
        'n_electrons': 6,
        'voip_effective': 7.5,
        'cluster_diameter': 4.0,
        'coupling_type': 'pi_conjugation',
    },
}


@dataclass
class CofactorDefinition:
    """Definition of a cofactor complex for 3-cell construction."""
    name: str
    cofactor_type: str  # Key into COFACTOR_DESCRIPTORS
    metal_atoms: List[int]  # Indices of metal atoms in PCC
    ligand_residues: List[int]  # Indices of coordinating residues
    additional_atoms: List[int] = field(default_factory=list)  # Non-metal cofactor atoms


def build_cofactor_3cells(
    enzyme_pcc: Dict[str, Any],
    cofactor_definitions: List[CofactorDefinition],
) -> Dict[str, Any]:
    """
    Add cofactor complexes as rank-3 cells to the Enzyme-PCC.

    Each cofactor (e.g., [4Fe-4S] cluster + 4 cysteine ligands)
    becomes a 3-cell whose boundary consists of:
    - The 2-cells (residue clusters) containing the ligand residues
    - The 1-cells (coordination bonds) connecting metals to ligands
    - The 0-cells (metal and ligand atoms)

    This encodes Chalopin's observation that cofactors bridge
    topological gaps between thermal hotspot islands.

    Args:
        enzyme_pcc: Existing 3-level PCC
        cofactor_definitions: List of CofactorDefinition objects

    Returns:
        enhanced_pcc: PCC with rank-3 cofactor cells added
    """
    # Get existing 2-cells
    two_cells = enzyme_pcc.get('2_cells', [])
    if not two_cells:
        # If no 2-cells defined, can't build 3-cells
        enhanced_pcc = enzyme_pcc.copy()
        enhanced_pcc['3_cells'] = []
        return enhanced_pcc

    # Build 3-cells and boundary operator
    three_cells = []
    boundary_3_to_2_data = []

    for cof_def in cofactor_definitions:
        # Find which 2-cells contain the cofactor's ligand residues
        boundary_2cells = set()
        for res_idx in cof_def.ligand_residues:
            for cell_2_idx, cell_2_members in enumerate(two_cells):
                if res_idx in cell_2_members:
                    boundary_2cells.add(cell_2_idx)

        # Get cofactor descriptors
        cof_props = COFACTOR_DESCRIPTORS.get(cof_def.cofactor_type, {})

        three_cell = {
            'name': cof_def.name,
            'type': cof_def.cofactor_type,
            'metal_atoms': cof_def.metal_atoms,
            'ligand_residues': cof_def.ligand_residues,
            'additional_atoms': cof_def.additional_atoms,
            'boundary_2cells': list(boundary_2cells),
            'properties': cof_props,
        }
        three_cells.append(three_cell)
        boundary_3_to_2_data.append(list(boundary_2cells))

    # Build boundary operator B_23: (n_2cells, n_3cells)
    n_2cells = len(two_cells)
    n_3cells = len(three_cells)

    if n_3cells > 0:
        B_23 = np.zeros((n_2cells, n_3cells))
        for j, boundary_2s in enumerate(boundary_3_to_2_data):
            for i in boundary_2s:
                B_23[i, j] = 1.0  # Incidence
        B_23 = csr_matrix(B_23)
    else:
        B_23 = None

    # Create enhanced PCC
    enhanced_pcc = enzyme_pcc.copy()
    enhanced_pcc['3_cells'] = three_cells
    enhanced_pcc['boundary_2_3'] = B_23

    return enhanced_pcc


def extract_cofactor_sheaf_sections(
    three_cells: List[Dict],
    sheaf_dim: int = 8,
) -> Dict[int, np.ndarray]:
    """
    Create sheaf sections for cofactor 3-cells.

    Uses cofactor properties to construct d-dimensional feature vectors.
    """
    sections = {}
    for idx, cell in enumerate(three_cells):
        props = cell.get('properties', {})

        # Build feature vector from cofactor properties
        features = np.zeros(sheaf_dim)
        features[0] = props.get('voip_effective', 8.0)
        features[1] = props.get('n_electrons', 10) / 30.0  # Normalize
        features[2] = props.get('cluster_diameter', 5.0) / 10.0
        redox = props.get('redox_potential_range', (-0.3, 0.1))
        features[3] = (redox[0] + redox[1]) / 2.0  # Mean redox potential
        features[4] = redox[1] - redox[0]  # Redox range

        sections[idx] = features

    return sections


def sheaf_sections_from_voip_tensor(
    voip_sections: "torch.Tensor",
) -> Dict[int, np.ndarray]:
    """
    Convert (N, 8) VOIPSheafSectionBuilder output to the Dict[int, np.ndarray]
    format expected by SheafENM.sheaf_laplacian().

    Parameters
    ----------
    voip_sections : (N, 8) float tensor from AttentiveVOIPEncoder /
                    VOIPSheafSectionBuilder.forward()

    Returns
    -------
    Dict mapping atom index → (8,) numpy array
    """
    arr = voip_sections.detach().cpu().numpy()
    return {i: arr[i] for i in range(arr.shape[0])}


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Vibrational Filtration
# ══════════════════════════════════════════════════════════════════════════════

def compute_vibrational_filtration(
    u_h: np.ndarray,
    n_bins: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert localization landscape to filtration thresholds.

    Atoms with highest u_h (thermal hotspots) have lowest
    filtration value (appear first in complex construction).

    Args:
        u_h: (N,) localization amplitudes
        n_bins: Number of filtration steps

    Returns:
        filt_values: (N,) filtration values ∈ [0, 1]
        thresholds: (n_bins,) bin boundaries
    """
    # Invert: high u_h → low filtration value (appear early)
    u_max = u_h.max()
    if u_max > 0:
        filt_values = 1.0 - (u_h / u_max)
    else:
        filt_values = np.zeros_like(u_h)

    # Quantile-based thresholds for even bin sizes
    thresholds = np.quantile(filt_values, np.linspace(0, 1, n_bins + 1)[1:])

    return filt_values, thresholds


def assign_vibrational_bins(
    u_h: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    Assign atoms to vibrational filtration bins.

    Bin 0 = thermal hotspots (highest u_h)
    Bin n_bins-1 = cold regions (lowest u_h)
    """
    filt_values, thresholds = compute_vibrational_filtration(u_h, n_bins)
    bins = np.digitize(filt_values, thresholds)
    return bins


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Validation Utilities
# ══════════════════════════════════════════════════════════════════════════════

def thermal_hotspot_recovery(
    model_attributions: np.ndarray,
    u_h: np.ndarray,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Test whether model attention/attribution identifies the same
    residues as Chalopin's thermal hotspots.

    Success metric: >80% overlap between top-k attributed residues
    and localization landscape peaks.

    Args:
        model_attributions: (N,) model attribution scores per residue
        u_h: (N,) localization landscape values
        top_k: Number of top residues to compare

    Returns:
        Dict with overlap_fraction, model_hotspots, landscape_hotspots, shared
    """
    # Top-k by model attribution
    model_top_k = set(np.argsort(model_attributions)[-top_k:])

    # Top-k by localization landscape
    landscape_top_k = set(np.argsort(u_h)[-top_k:])

    overlap = len(model_top_k & landscape_top_k) / top_k

    return {
        'overlap_fraction': overlap,
        'model_hotspots': model_top_k,
        'landscape_hotspots': landscape_top_k,
        'shared': model_top_k & landscape_top_k,
        'model_only': model_top_k - landscape_top_k,
        'landscape_only': landscape_top_k - model_top_k,
    }


def vibrational_coupling_analysis(
    u_h_wildtype: np.ndarray,
    u_h_mutant: np.ndarray,
    mutation_site: int,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Analyze how a mutation affects vibrational coupling.

    Chalopin shows that distant mutations can disrupt vibrational
    coupling through the contact network. This provides validation
    for ToPE's Zone 3 (>20 Å) allosteric predictions.

    Args:
        u_h_wildtype: (N,) localization landscape for wild-type
        u_h_mutant: (N,) localization landscape after mutation
        mutation_site: Residue index of mutation
        threshold: Minimum change to consider significant

    Returns:
        Analysis of coupling disruption
    """
    # Compute landscape change
    delta_u_h = u_h_mutant - u_h_wildtype
    abs_delta = np.abs(delta_u_h)

    # Identify affected residues
    affected = np.where(abs_delta > threshold)[0]

    # Classify by distance from mutation site
    distances_from_mutation = np.abs(np.arange(len(u_h_wildtype)) - mutation_site)

    local_affected = affected[distances_from_mutation[affected] < 5]
    medium_affected = affected[(distances_from_mutation[affected] >= 5) &
                               (distances_from_mutation[affected] < 20)]
    distal_affected = affected[distances_from_mutation[affected] >= 20]

    return {
        'total_affected': len(affected),
        'local_affected': list(local_affected),
        'medium_range_affected': list(medium_affected),
        'distal_affected': list(distal_affected),
        'max_change': abs_delta.max(),
        'mean_change': abs_delta.mean(),
        'mutation_site_change': delta_u_h[mutation_site],
        'coupling_disrupted': len(distal_affected) > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Known Test Cases from Chalopin Paper
# ══════════════════════════════════════════════════════════════════════════════

# CpI [FeFe]-hydrogenase thermal hotspot residues (from Chalopin 2023)
CPI_HYDROGENASE_HOTSPOTS = {
    'active_site_pocket': ['M229', 'A230', 'P231', 'S323', 'P324', 'F417', 'M497'],
    'proton_transfer': ['R286', 'E282', 'S319', 'E279', 'C299'],
    'electron_transfer': ['C34', 'C107', 'C150', 'C153', 'C193', 'V202'],
    'water_channel': ['G418', 'A419', 'G502', 'C503'],
}

# E282→D mutation test case
CPI_E282D_MUTATION = {
    'description': 'Glu282→Asp mutation increases H-bond distance from ~3.9 Å to ~6 Å',
    'expected_effect': 'Quench proton transfer channel by decoupling R286 from RPV network',
    'wildtype_residue': 'E282',
    'mutant_residue': 'D282',
    'affected_pathway': 'proton_transfer',
    'reference': 'Chalopin et al. (2023) Biophys. J. 122, 1557–1567',
}


def get_cpi_test_case() -> Dict[str, Any]:
    """
    Get the CpI [FeFe]-hydrogenase test case for validation.

    This enzyme is the primary example in Chalopin's paper and
    provides specific testable predictions.
    """
    return {
        'name': 'CpI [FeFe]-hydrogenase',
        'pdb_id': '4XDC',  # Or '3C8Y' for different form
        'hotspots': CPI_HYDROGENASE_HOTSPOTS,
        'mutation_test': CPI_E282D_MUTATION,
        'cofactors': [
            CofactorDefinition(
                name='H-cluster',
                cofactor_type='H_cluster',
                metal_atoms=[],  # To be filled from structure
                ligand_residues=[],  # Cys residues
            ),
            CofactorDefinition(
                name='FS4A',
                cofactor_type='Fe4S4',
                metal_atoms=[],
                ligand_residues=[],
            ),
            CofactorDefinition(
                name='FS4B',
                cofactor_type='Fe4S4',
                metal_atoms=[],
                ligand_residues=[],
            ),
            CofactorDefinition(
                name='FS2',
                cofactor_type='Fe2S2',
                metal_atoms=[],
                ligand_residues=[],
            ),
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Preprocessing Integration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnzymePCCPhononExtension:
    """Extended Enzyme-PCC fields for phonon topology."""

    # Cα coordinates for ENM
    ca_coords: np.ndarray

    # Localization landscapes at different ranks
    u_h_residue: np.ndarray  # (N_res,) residue-level
    u_h_atom: Optional[np.ndarray] = None  # (N_atoms,) atom-level
    u_h_rank0: Optional[np.ndarray] = None  # Atom-level from Hodge
    u_h_rank1: Optional[np.ndarray] = None  # Bond-level
    u_h_rank2: Optional[np.ndarray] = None  # Cluster-level

    # Contact classification
    contact_types: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    # Cofactor topology
    cofactor_3cells: List[Dict] = field(default_factory=list)
    boundary_2_3: Optional[csr_matrix] = None

    # Vibrational filtration
    vibrational_bins: Optional[np.ndarray] = None
    vibrational_thresholds: Optional[np.ndarray] = None


def extend_enzyme_pcc_with_phonon_topology(
    enzyme_pcc: Dict[str, Any],
    ca_coords: np.ndarray,
    cofactor_definitions: Optional[List[CofactorDefinition]] = None,
    config: Optional[PhononTopologyConfig] = None,
) -> Dict[str, Any]:
    """
    Extend an existing Enzyme-PCC with phonon topology fields.

    Additional cost: ~0.1s per structure (sparse linear solve)
    Additional storage: ~2 KB per structure (u_h arrays)

    Args:
        enzyme_pcc: Existing Enzyme-PCC dict
        ca_coords: (N_res, 3) Cα coordinates
        cofactor_definitions: Optional cofactor definitions for 3-cells
        config: Phonon topology configuration

    Returns:
        Extended PCC with phonon topology fields
    """
    config = config or PhononTopologyConfig()

    # Compute localization landscape (cheap: single sparse solve)
    u_h, contacts = compute_localization_landscape(
        ca_coords,
        cutoff=config.contact_cutoff,
        tolerance=config.contact_tolerance,
        regularization=config.regularization,
    )

    # Add cofactor contacts if cofactors defined
    if cofactor_definitions:
        # Would need cofactor coordinates - placeholder
        pass

    # Build 3-cells if cofactors defined
    if cofactor_definitions:
        enzyme_pcc = build_cofactor_3cells(enzyme_pcc, cofactor_definitions)

    # Compute multi-rank localization landscapes
    try:
        hodge_enm = HodgeLaplacianENM(enzyme_pcc, config=config)
        landscapes = hodge_enm.multi_rank_landscape()
    except Exception:
        landscapes = {}

    # Compute vibrational filtration
    vib_bins = assign_vibrational_bins(u_h, config.n_vibrational_bins)
    _, vib_thresholds = compute_vibrational_filtration(u_h, config.n_vibrational_bins)

    # Add to PCC
    enzyme_pcc['ca_coords'] = ca_coords
    enzyme_pcc['u_h_residue'] = u_h
    enzyme_pcc['u_h_rank0'] = landscapes.get(0)
    enzyme_pcc['u_h_rank1'] = landscapes.get(1)
    enzyme_pcc['u_h_rank2'] = landscapes.get(2)
    enzyme_pcc['contact_types'] = contacts
    enzyme_pcc['vibrational_bins'] = vib_bins
    enzyme_pcc['vibrational_thresholds'] = vib_thresholds

    return enzyme_pcc
