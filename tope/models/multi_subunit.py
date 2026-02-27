"""
Phase 5: Multi-Subunit Enzyme Handling
======================================

Extends the Enzyme-PCC hierarchy to handle quaternary structure:

4-Level Hierarchy:
    0-cells: Atoms
    1-cells: Bonds (covalent, coordination, H-bonds)
    2-cells: Residues (amino acids)
    3-cells: Subunits (protein chains, domains)
    4-cells: Subunit interfaces (inter-chain contact regions)

This enables modelling of:
    - Hemoglobin (4 subunits, allosteric cooperativity)
    - ATP synthase (rotating motor, 24+ subunits)
    - Ribosomes (2 subunits, RNA-protein complexes)
    - Pyruvate dehydrogenase (60 subunits!)

Key Components:
    MultiSubunitPCCBuilder: Constructs 4-level hierarchical complex
    CrossSubunitLaplacian: Sheaf Laplacian with inter-subunit coupling
    AllostericFeatureExtractor: Spectral features for cooperativity
    MultiSubunitTCPNet: Message passing across all cell dimensions
    AllostericCooperativityHead: Hill coefficient prediction

References
----------
Monod, Wyman & Changeux (1965) - Allosteric transitions
Perutz (1970) - Hemoglobin cooperativity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tope.models.cc_attention import CCAttentionBlock

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Multi-Subunit Enzyme-PCC Builder
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SubunitCell:
    """3-cell representing a protein subunit/chain."""

    chain_id: str
    atom_indices: List[int]
    residue_indices: List[int]
    centroid: np.ndarray  # Center of mass
    radius_of_gyration: float
    boundary_vertices: Optional[np.ndarray] = None  # Convex hull


@dataclass
class InterfaceCell:
    """4-cell representing an inter-subunit interface."""

    subunit_pair: Tuple[int, int]  # Indices of interacting subunits
    interface_residues_A: List[int]
    interface_residues_B: List[int]
    contact_area: float  # Buried surface area (Å²)
    interface_type: str  # 'obligate', 'transient', 'catalytic', 'structural'
    hydrogen_bonds: int
    salt_bridges: int
    hydrophobic_contacts: int


@dataclass
class MultiSubunitPCCConfig:
    """Configuration for multi-subunit PCC construction."""

    # Interface detection
    interface_distance_cutoff: float = 5.0  # Å
    min_interface_contacts: int = 5

    # Subunit features
    compute_convex_hull: bool = True

    # Interface classification thresholds
    catalytic_residue_threshold: int = 2  # Min catalytic residues for 'catalytic' type
    obligate_contact_threshold: float = 1500.0  # Å² buried surface area


class MultiSubunitPCCBuilder:
    """
    Builds extended hierarchical complex for quaternary structure.

    Hierarchy:
        0-cells: Atoms
        1-cells: Bonds (covalent, coordination, H-bonds)
        2-cells: Residues (amino acids)
        3-cells: Subunits (protein chains, domains)
        4-cells: Subunit interfaces (inter-chain contact regions)
    """

    def __init__(self, config: Optional[MultiSubunitPCCConfig] = None):
        self.config = config or MultiSubunitPCCConfig()

    def build(
        self,
        pdb_structure: Any,
        active_site_residues: List[Tuple[str, int]],
        base_pcc: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-subunit Enzyme-PCC.

        Parameters
        ----------
        pdb_structure : Bio.PDB.Structure or similar
            Parsed PDB structure with chain information
        active_site_residues : list of (chain_id, res_id)
            Known catalytic residues
        base_pcc : dict, optional
            Pre-built base PCC (0-2 cells). If None, builds from scratch.

        Returns
        -------
        enzyme_pcc : dict with keys:
            '0_cells': atom features
            '1_cells': bond information
            '2_cells': residue clusters
            '3_cells': List[SubunitCell]
            '4_cells': List[InterfaceCell]
            'subunit_membership': residue -> subunit mapping
            'n_subunits': int
            'n_interfaces': int
        """
        # Build or use base PCC (0-2 cells)
        if base_pcc is not None:
            enzyme_pcc = base_pcc.copy()
        else:
            enzyme_pcc = self._build_base_pcc(pdb_structure, active_site_residues)

        # Extract chain information
        chains = self._extract_chains(pdb_structure)

        # Build 3-cells: Subunits
        subunits, subunit_membership = self._build_subunits(
            chains, enzyme_pcc
        )
        enzyme_pcc["3_cells"] = subunits
        enzyme_pcc["subunit_membership"] = subunit_membership
        enzyme_pcc["n_subunits"] = len(subunits)

        # Build 4-cells: Interfaces
        interfaces = self._build_interfaces(
            subunits, enzyme_pcc, active_site_residues
        )
        enzyme_pcc["4_cells"] = interfaces
        enzyme_pcc["n_interfaces"] = len(interfaces)

        # Add quaternary structure metadata
        enzyme_pcc["quaternary_info"] = {
            "n_chains": len(chains),
            "chain_ids": [c["chain_id"] for c in chains],
            "total_interface_area": sum(i.contact_area for i in interfaces),
            "has_catalytic_interface": any(
                i.interface_type == "catalytic" for i in interfaces
            ),
        }

        return enzyme_pcc

    def _build_base_pcc(
        self,
        pdb_structure: Any,
        active_site_residues: List[Tuple[str, int]],
    ) -> Dict[str, Any]:
        """Build base 0-2 cell PCC (placeholder for actual implementation)."""
        # This would normally call the existing Enzyme-PCC builder
        # For now, return a minimal structure
        return {
            "0_cells": {"coords": [], "features": []},
            "1_cells": {"edges": [], "bond_types": []},
            "2_cells": {"residues": []},
            "n_atoms": 0,
            "n_residues": 0,
        }

    def _extract_chains(self, pdb_structure: Any) -> List[Dict[str, Any]]:
        """Extract chain information from PDB structure."""
        chains = []

        try:
            # Bio.PDB style
            for model in pdb_structure:
                for chain in model:
                    residues = list(chain.get_residues())
                    atoms = []
                    for res in residues:
                        atoms.extend(list(res.get_atoms()))

                    if len(residues) > 0:
                        # Compute centroid
                        coords = np.array([a.get_coord() for a in atoms])
                        centroid = coords.mean(axis=0)

                        # Compute radius of gyration
                        r_gyr = np.sqrt(
                            ((coords - centroid) ** 2).sum(axis=1).mean()
                        )

                        chains.append({
                            "chain_id": chain.id,
                            "residues": residues,
                            "atoms": atoms,
                            "coords": coords,
                            "centroid": centroid,
                            "radius_of_gyration": r_gyr,
                        })
        except (AttributeError, TypeError):
            # Fallback for different structure formats
            logger.warning("Could not parse PDB structure; using minimal chain info")

        return chains

    def _build_subunits(
        self,
        chains: List[Dict[str, Any]],
        enzyme_pcc: Dict[str, Any],
    ) -> Tuple[List[SubunitCell], Dict[int, int]]:
        """Build 3-cells (subunits) from chain information."""
        subunits = []
        subunit_membership = {}  # residue_idx -> subunit_idx

        residue_offset = 0

        for subunit_idx, chain in enumerate(chains):
            n_residues = len(chain.get("residues", []))
            n_atoms = len(chain.get("atoms", []))

            # Residue indices for this subunit
            residue_indices = list(range(residue_offset, residue_offset + n_residues))

            # Update membership mapping
            for res_idx in residue_indices:
                subunit_membership[res_idx] = subunit_idx

            # Compute convex hull boundary if requested
            boundary = None
            if self.config.compute_convex_hull and "coords" in chain:
                try:
                    from scipy.spatial import ConvexHull
                    if len(chain["coords"]) >= 4:
                        hull = ConvexHull(chain["coords"])
                        boundary = chain["coords"][hull.vertices]
                except Exception:
                    pass

            subunit = SubunitCell(
                chain_id=chain.get("chain_id", str(subunit_idx)),
                atom_indices=list(range(n_atoms)),  # Would need global mapping
                residue_indices=residue_indices,
                centroid=chain.get("centroid", np.zeros(3)),
                radius_of_gyration=chain.get("radius_of_gyration", 0.0),
                boundary_vertices=boundary,
            )
            subunits.append(subunit)

            residue_offset += n_residues

        return subunits, subunit_membership

    def _build_interfaces(
        self,
        subunits: List[SubunitCell],
        enzyme_pcc: Dict[str, Any],
        active_site_residues: List[Tuple[str, int]],
    ) -> List[InterfaceCell]:
        """Build 4-cells (interfaces) from subunit pairs."""
        interfaces = []
        active_chains = {r[0] for r in active_site_residues}

        for i, subunit_A in enumerate(subunits):
            for j, subunit_B in enumerate(subunits[i + 1:], i + 1):
                # Detect interface residues
                interface_pairs = self._find_interface_residues(
                    subunit_A, subunit_B, enzyme_pcc
                )

                if len(interface_pairs) < self.config.min_interface_contacts:
                    continue

                # Compute interface properties
                contact_area = self._compute_buried_surface_area(
                    subunit_A, subunit_B, interface_pairs
                )

                h_bonds, salt_bridges, hydrophobic = self._count_interface_interactions(
                    interface_pairs, enzyme_pcc
                )

                # Classify interface type
                interface_type = self._classify_interface(
                    subunit_A, subunit_B, interface_pairs,
                    contact_area, active_chains
                )

                interface = InterfaceCell(
                    subunit_pair=(i, j),
                    interface_residues_A=[p[0] for p in interface_pairs],
                    interface_residues_B=[p[1] for p in interface_pairs],
                    contact_area=contact_area,
                    interface_type=interface_type,
                    hydrogen_bonds=h_bonds,
                    salt_bridges=salt_bridges,
                    hydrophobic_contacts=hydrophobic,
                )
                interfaces.append(interface)

        return interfaces

    def _find_interface_residues(
        self,
        subunit_A: SubunitCell,
        subunit_B: SubunitCell,
        enzyme_pcc: Dict[str, Any],
    ) -> List[Tuple[int, int]]:
        """Find residue pairs at the subunit interface."""
        interface_pairs = []

        # Simple distance-based detection
        # In practice, would use actual coordinates from enzyme_pcc
        for res_A in subunit_A.residue_indices:
            for res_B in subunit_B.residue_indices:
                # Placeholder: would compute actual distance
                # For now, use heuristic based on residue index proximity
                # This should be replaced with actual coordinate-based computation
                interface_pairs.append((res_A, res_B))

                if len(interface_pairs) >= 20:  # Limit for placeholder
                    break
            if len(interface_pairs) >= 20:
                break

        return interface_pairs[:self.config.min_interface_contacts + 5]

    def _compute_buried_surface_area(
        self,
        subunit_A: SubunitCell,
        subunit_B: SubunitCell,
        interface_pairs: List[Tuple[int, int]],
    ) -> float:
        """Estimate buried surface area at interface."""
        # Simplified estimation: ~100 Å² per interface contact
        return len(interface_pairs) * 100.0

    def _count_interface_interactions(
        self,
        interface_pairs: List[Tuple[int, int]],
        enzyme_pcc: Dict[str, Any],
    ) -> Tuple[int, int, int]:
        """Count interaction types at interface."""
        # Placeholder counts - would analyze actual atom types
        n_contacts = len(interface_pairs)
        h_bonds = n_contacts // 3
        salt_bridges = n_contacts // 10
        hydrophobic = n_contacts - h_bonds - salt_bridges
        return h_bonds, salt_bridges, hydrophobic

    def _classify_interface(
        self,
        subunit_A: SubunitCell,
        subunit_B: SubunitCell,
        interface_pairs: List[Tuple[int, int]],
        contact_area: float,
        active_chains: Set[str],
    ) -> str:
        """Classify interface type."""
        # Check if interface involves catalytic chains
        if (subunit_A.chain_id in active_chains or
            subunit_B.chain_id in active_chains):
            return "catalytic"

        # Check if obligate (large buried surface area)
        if contact_area >= self.config.obligate_contact_threshold:
            return "obligate"

        return "structural"


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Cross-Subunit Sheaf Laplacian
# ══════════════════════════════════════════════════════════════════════════════


class CrossSubunitLaplacian:
    """
    Extended sheaf Laplacian including inter-subunit interactions.

    Returns:
        L_intra: Laplacian within each subunit (3-cell internal)
        L_inter: Laplacian across subunits (4-cell coupling)
        L_total: Combined multi-scale Laplacian
    """

    def __init__(
        self,
        catalytic_coupling: float = 1.0,
        structural_coupling: float = 0.1,
    ):
        self.catalytic_coupling = catalytic_coupling
        self.structural_coupling = structural_coupling

    def compute(
        self,
        enzyme_pcc: Dict[str, Any],
        filtration_radius: float,
        sheaf_data: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Compute intra-subunit, inter-subunit, and total Laplacians.

        Parameters
        ----------
        enzyme_pcc : dict
            Multi-subunit PCC with 3-cells and 4-cells
        filtration_radius : float
            Current filtration radius (Å)
        sheaf_data : dict, optional
            Per-atom sheaf sections (electronegativity, etc.)

        Returns
        -------
        L_intra : sparse matrix
            Intra-subunit Laplacian
        L_inter : sparse matrix
            Inter-subunit coupling Laplacian
        L_total : sparse matrix
            Combined Laplacian
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            logger.warning("scipy not available; returning None Laplacians")
            return None, None, None

        n_atoms = enzyme_pcc.get("n_atoms", 100)  # Default for placeholder
        interfaces = enzyme_pcc.get("4_cells", [])

        # Build intra-subunit Laplacian (standard graph Laplacian)
        L_intra = self._build_intra_laplacian(enzyme_pcc, filtration_radius)

        # Build inter-subunit coupling Laplacian
        L_inter = sp.lil_matrix((n_atoms, n_atoms), dtype=np.float64)

        for interface in interfaces:
            # Determine coupling strength by interface type
            if interface.interface_type == "catalytic":
                coupling_strength = self.catalytic_coupling
            else:
                coupling_strength = self.structural_coupling

            # Add edges between interface residues
            for res_a, res_b in zip(
                interface.interface_residues_A,
                interface.interface_residues_B
            ):
                # Get atoms in each residue (simplified)
                atoms_A = self._get_atoms_in_residue(enzyme_pcc, res_a)
                atoms_B = self._get_atoms_in_residue(enzyme_pcc, res_b)

                for atom_i in atoms_A:
                    for atom_j in atoms_B:
                        if atom_i >= n_atoms or atom_j >= n_atoms:
                            continue

                        # Compute edge weight
                        ionicity = self._compute_ionicity(
                            sheaf_data, atom_i, atom_j
                        ) if sheaf_data else 1.0

                        weight = coupling_strength * ionicity

                        # Add to Laplacian (symmetric)
                        L_inter[atom_i, atom_j] -= weight
                        L_inter[atom_j, atom_i] -= weight
                        L_inter[atom_i, atom_i] += weight
                        L_inter[atom_j, atom_j] += weight

        L_inter = L_inter.tocsr()

        # Total Laplacian
        if L_intra is not None:
            L_total = L_intra + L_inter
        else:
            L_total = L_inter

        return L_intra, L_inter, L_total

    def _build_intra_laplacian(
        self,
        enzyme_pcc: Dict[str, Any],
        filtration_radius: float,
    ) -> Any:
        """Build standard intra-subunit Laplacian."""
        try:
            import scipy.sparse as sp
        except ImportError:
            return None

        n_atoms = enzyme_pcc.get("n_atoms", 100)
        edges = enzyme_pcc.get("1_cells", {}).get("edges", [])

        L = sp.lil_matrix((n_atoms, n_atoms), dtype=np.float64)

        for edge in edges:
            if len(edge) >= 2:
                i, j = edge[0], edge[1]
                if i < n_atoms and j < n_atoms:
                    L[i, j] -= 1.0
                    L[j, i] -= 1.0
                    L[i, i] += 1.0
                    L[j, j] += 1.0

        return L.tocsr()

    def _get_atoms_in_residue(
        self,
        enzyme_pcc: Dict[str, Any],
        residue_idx: int,
    ) -> List[int]:
        """Get atom indices belonging to a residue."""
        # Simplified: assume ~10 atoms per residue
        base_atom = residue_idx * 10
        return list(range(base_atom, base_atom + 10))

    def _compute_ionicity(
        self,
        sheaf_data: Dict[int, Dict[str, float]],
        atom_i: int,
        atom_j: int,
    ) -> float:
        """Compute ionicity from electronegativity difference."""
        if atom_i not in sheaf_data or atom_j not in sheaf_data:
            return 1.0

        chi_i = sheaf_data[atom_i].get("chi", 2.5)
        chi_j = sheaf_data[atom_j].get("chi", 2.5)
        return abs(chi_i - chi_j) + 0.1  # Add small baseline


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Allosteric Spectral Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AllostericSpectralFeatures:
    """Container for allosteric spectral descriptors."""

    # Per-radius features
    subunit_coupling_strength: List[float] = field(default_factory=list)
    interface_eigenvalue_gap: List[float] = field(default_factory=list)
    cross_subunit_resistance: List[float] = field(default_factory=list)
    allosteric_pathway_count: List[int] = field(default_factory=list)

    # Aggregated features
    mean_coupling: float = 0.0
    min_eigenvalue_gap: float = 0.0
    max_resistance: float = 0.0
    total_pathways: int = 0


class AllostericFeatureExtractor:
    """
    Compute spectral features encoding allosteric communication.

    Features capture:
    - Subunit coupling strength (inter-subunit connectivity)
    - Interface eigenvalue gap (barrier to cross-subunit transfer)
    - Cross-subunit resistance (1/λ_inter, communication cost)
    - Allosteric pathway count (low-eigenvalue modes spanning subunits)
    """

    def __init__(
        self,
        n_eigenvalues: int = 50,
        pathway_threshold_fraction: float = 0.1,
    ):
        self.n_eigenvalues = n_eigenvalues
        self.pathway_threshold_fraction = pathway_threshold_fraction
        self.laplacian_builder = CrossSubunitLaplacian()

    def extract(
        self,
        enzyme_pcc: Dict[str, Any],
        radii: List[float],
        sheaf_data: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> AllostericSpectralFeatures:
        """
        Compute allosteric features across filtration radii.

        Parameters
        ----------
        enzyme_pcc : dict
            Multi-subunit PCC with interfaces
        radii : list of float
            Filtration radii to probe (Å)
        sheaf_data : dict, optional
            Per-atom sheaf sections

        Returns
        -------
        features : AllostericSpectralFeatures
        """
        features = AllostericSpectralFeatures()

        for epsilon in radii:
            L_intra, L_inter, L_total = self.laplacian_builder.compute(
                enzyme_pcc, epsilon, sheaf_data
            )

            if L_inter is None:
                features.subunit_coupling_strength.append(0.0)
                features.interface_eigenvalue_gap.append(0.0)
                features.cross_subunit_resistance.append(float("inf"))
                features.allosteric_pathway_count.append(0)
                continue

            # Coupling strength: sum of inter-subunit edge weights
            coupling = np.abs(L_inter.data).sum() / 2
            features.subunit_coupling_strength.append(float(coupling))

            # Compute eigenvalues of inter-subunit coupling
            try:
                from scipy.sparse.linalg import eigsh

                k = min(self.n_eigenvalues, L_inter.shape[0] - 2)
                if k > 0:
                    evals, _ = eigsh(L_inter.tocsr(), k=k, which="SM")
                else:
                    evals = np.array([0.0])
            except Exception:
                evals = np.array([0.0])

            # Interface gap: smallest positive eigenvalue
            nonzero_evals = evals[np.abs(evals) > 1e-6]
            if len(nonzero_evals) > 0:
                interface_gap = float(nonzero_evals.min())
            else:
                interface_gap = 0.0
            features.interface_eigenvalue_gap.append(interface_gap)

            # Cross-subunit resistance
            if interface_gap > 1e-6:
                resistance = 1.0 / interface_gap
            else:
                resistance = float("inf")
            features.cross_subunit_resistance.append(resistance)

            # Count low-frequency modes (allosteric highways)
            if len(evals) > 0 and evals.max() > 0:
                threshold = self.pathway_threshold_fraction * evals.max()
                n_allosteric = int(np.sum(evals < threshold))
            else:
                n_allosteric = 0
            features.allosteric_pathway_count.append(n_allosteric)

        # Compute aggregated features
        if features.subunit_coupling_strength:
            features.mean_coupling = np.mean(features.subunit_coupling_strength)

        nonzero_gaps = [g for g in features.interface_eigenvalue_gap if g > 0]
        if nonzero_gaps:
            features.min_eigenvalue_gap = min(nonzero_gaps)

        finite_resistance = [r for r in features.cross_subunit_resistance
                           if r != float("inf")]
        if finite_resistance:
            features.max_resistance = max(finite_resistance)

        features.total_pathways = sum(features.allosteric_pathway_count)

        return features


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Multi-Subunit TCPNet Message Passing
# ══════════════════════════════════════════════════════════════════════════════


class SubunitMessagePassing(nn.Module):
    """Message passing from subunits (3-cells) to residues (2-cells)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_subunits: torch.Tensor,  # [n_subunits, H]
        h_residues: torch.Tensor,  # [n_residues, H]
        subunit_membership: Dict[int, int],
    ) -> torch.Tensor:
        """
        Broadcast subunit context to constituent residues.

        Returns
        -------
        msg : [n_residues, H] subunit-to-residue messages
        """
        n_residues = h_residues.size(0)
        device = h_residues.device

        # Project subunit features
        subunit_proj = self.proj(h_subunits)  # [n_subunits, H]

        # Broadcast to residues
        msg = torch.zeros_like(h_residues)
        for res_idx in range(n_residues):
            subunit_idx = subunit_membership.get(res_idx, 0)
            if subunit_idx < h_subunits.size(0):
                # Gated combination
                combined = torch.cat([h_residues[res_idx], subunit_proj[subunit_idx]])
                gate_weight = self.gate(combined)
                msg[res_idx] = gate_weight * subunit_proj[subunit_idx]

        return msg


class ResidueToSubunitAggregation(nn.Module):
    """Aggregate residue features back to subunit level."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h_residues: torch.Tensor,  # [n_residues, H]
        subunit_membership: Dict[int, int],
        n_subunits: int,
    ) -> torch.Tensor:
        """
        Aggregate residue information to subunits with attention.

        Returns
        -------
        msg : [n_subunits, H] residue-to-subunit messages
        """
        device = h_residues.device
        hidden_dim = h_residues.size(1)

        msg = torch.zeros(n_subunits, hidden_dim, device=device)

        # Group residues by subunit
        subunit_residues = defaultdict(list)
        for res_idx, sub_idx in subunit_membership.items():
            if res_idx < h_residues.size(0):
                subunit_residues[sub_idx].append(res_idx)

        for sub_idx, res_indices in subunit_residues.items():
            if sub_idx >= n_subunits:
                continue

            # Get residue features for this subunit
            res_feats = h_residues[res_indices]  # [n_res_in_sub, H]

            # Attention-weighted aggregation
            attn_scores = self.attention(res_feats)  # [n_res_in_sub, 1]
            attn_weights = F.softmax(attn_scores, dim=0)

            # Weighted sum
            aggregated = (attn_weights * res_feats).sum(dim=0)  # [H]
            msg[sub_idx] = self.proj(aggregated)

        return msg


class InterfaceMessagePassing(nn.Module):
    """Message passing across subunit interfaces (4-cells).

    Uses a CCAttentionBlock for bidirectional attention between subunits
    (rank-3 cells) and interfaces (rank-4 cells) via the B_34 incidence matrix.
    When B_34 is not pre-built, it is derived automatically from the
    List[InterfaceCell] objects passed to forward().
    """

    # 4 scalar interface features: area, H-bonds, salt bridges, type
    _N_IFACE_FEATS = 4

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embed raw 4-d interface features → hidden_dim
        self.interface_embed = nn.Linear(self._N_IFACE_FEATS, hidden_dim)

        # CCAttentionBlock: rank-s = subunits (3-cells), rank-t = interfaces (4-cells)
        self.attn_34 = CCAttentionBlock(
            d_s_in=hidden_dim,
            d_t_in=hidden_dim,
            d_out=hidden_dim,
        )

    def _build_B_34_from_cells(
        self,
        interfaces: List["InterfaceCell"],
        n_subunits: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialise interface feature matrix and B_34 incidence tensor.

        Returns
        -------
        h_interfaces : (n_interfaces, hidden_dim)
        B_34         : (2, E_34) — B_34[0] = subunit idx, B_34[1] = interface idx
        """
        raw_feats = []
        src_list: List[int] = []  # subunit indices
        tgt_list: List[int] = []  # interface indices

        for iface_idx, iface in enumerate(interfaces):
            i, j = iface.subunit_pair
            if i >= n_subunits or j >= n_subunits:
                continue
            # Each interface touches two subunits
            src_list.extend([i, j])
            tgt_list.extend([iface_idx, iface_idx])
            raw_feats.append([
                iface.contact_area / 1000.0,
                iface.hydrogen_bonds / 10.0,
                iface.salt_bridges / 5.0,
                1.0 if iface.interface_type == "catalytic" else 0.0,
            ])

        if not raw_feats:
            # No valid interfaces — return empty tensors
            h_iface = torch.zeros(0, self.hidden_dim, device=device)
            B_34 = torch.zeros(2, 0, dtype=torch.long, device=device)
            return h_iface, B_34

        feats = torch.tensor(raw_feats, dtype=torch.float32, device=device)
        h_iface = self.interface_embed(feats)  # (n_interfaces, H)
        B_34 = torch.tensor(
            [src_list, tgt_list], dtype=torch.long, device=device
        )
        return h_iface, B_34

    def forward(
        self,
        h_subunits: torch.Tensor,              # (n_subunits, H)
        interfaces: Optional[List["InterfaceCell"]] = None,
        h_interfaces: Optional[torch.Tensor] = None,   # (n_interfaces, H)
        B_34: Optional[torch.Tensor] = None,           # (2, E_34)
    ) -> torch.Tensor:
        """
        Mediate communication between subunits via interfaces.

        Either provide pre-built (h_interfaces, B_34) **or** a list of
        InterfaceCell objects — in the latter case both are derived here.

        Returns
        -------
        msg : (n_subunits, H) — interface-mediated messages for each subunit
        """
        n_subunits = h_subunits.size(0)
        device = h_subunits.device

        if B_34 is None or h_interfaces is None:
            if interfaces is None or len(interfaces) == 0:
                return torch.zeros_like(h_subunits)
            h_interfaces, B_34 = self._build_B_34_from_cells(
                interfaces, n_subunits, device
            )

        if B_34.size(1) == 0:
            return torch.zeros_like(h_subunits)

        # CCAttentionBlock: H_s=subunits (rank-3), H_t=interfaces (rank-4)
        # K_t = updated interface features (not used further)
        # K_s = updated subunit features = interface-mediated message
        _, K_subunits = self.attn_34(h_subunits, h_interfaces, B_34)
        return K_subunits


class MultiSubunitTCPNetLayer(nn.Module):
    """
    Extended TCPNet with subunit-level and interface-level messaging.

    Performs message passing across all cell dimensions:
    - 0-cells ↔ 1-cells (atoms ↔ bonds)
    - 1-cells ↔ 2-cells (bonds ↔ residues)
    - 2-cells ↔ 3-cells (residues ↔ subunits)
    - 3-cells ↔ 4-cells (subunits ↔ interfaces)
    """

    def __init__(self, hidden_dim: int = 256, max_degree: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # CC-attention for residues (2-cells) ↔ subunits (3-cells)
        self.attn_23 = CCAttentionBlock(
            d_s_in=hidden_dim,
            d_t_in=hidden_dim,
            d_out=hidden_dim,
        )

        # Standard 0↔1↔2 cell messages (simplified versions)
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.face_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3-cell (subunit) messages
        self.subunit_to_residue = SubunitMessagePassing(hidden_dim)
        self.residue_to_subunit = ResidueToSubunitAggregation(hidden_dim)

        # 4-cell (interface) messages
        self.interface_to_subunit = InterfaceMessagePassing(hidden_dim)

        # Subunit self-update
        self.subunit_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Layer norm
        self.norm_nodes = nn.LayerNorm(hidden_dim)
        self.norm_edges = nn.LayerNorm(hidden_dim)
        self.norm_faces = nn.LayerNorm(hidden_dim)
        self.norm_subunits = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_nodes: torch.Tensor,       # (n_atoms, H)
        h_edges: torch.Tensor,       # (n_edges, H)
        h_faces: torch.Tensor,       # (n_residues, H)
        h_subunits: torch.Tensor,    # (n_subunits, H)
        edge_index: torch.Tensor,    # (2, n_edges)
        subunit_membership: Dict[int, int],
        interfaces: List[InterfaceCell],
        B_23: Optional[torch.Tensor] = None,  # (2, E_23) residues→subunits incidence
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-scale message passing across all cell dimensions.

        When B_23 is provided, uses CCAttentionBlock for residue↔subunit messages
        (2-cells ↔ 3-cells); otherwise falls back to the dict-based approach.

        Returns
        -------
        h_nodes_new, h_edges_new, h_faces_new, h_subunits_new
        """
        # ── Standard 0↔1↔2 messages (simplified scatter) ──────────────────────
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        node_msg = torch.zeros_like(h_nodes)
        node_msg.scatter_add_(
            0,
            edge_dst.unsqueeze(-1).expand(-1, h_nodes.size(-1)),
            h_edges,
        )

        edge_msg = h_nodes[edge_src] + h_nodes[edge_dst]

        # ── 2-cells ↔ 3-cells (residues ↔ subunits) ────────────────────────────
        if B_23 is not None and B_23.size(1) > 0:
            # CCAttentionBlock: H_s = residues (rank-2), H_t = subunits (rank-3)
            # K_t = subunits attended by residues = msg_residue→subunit
            # K_s = residues attended by subunits = msg_subunit→residue
            msg_residue_subunit, msg_subunit_residue = self.attn_23(
                h_faces, h_subunits, B_23
            )
        else:
            # Fallback: dict-based broadcast / aggregation
            msg_subunit_residue = self.subunit_to_residue(
                h_subunits, h_faces, subunit_membership
            )
            msg_residue_subunit = self.residue_to_subunit(
                h_faces, subunit_membership, h_subunits.size(0)
            )

        # ── 3-cells ↔ 4-cells (subunits ↔ interfaces) ─────────────────────────
        msg_interface_subunit = self.interface_to_subunit(h_subunits, interfaces)

        # ── Update all cell features ───────────────────────────────────────────
        h_nodes_new = self.norm_nodes(
            h_nodes + self.node_update(torch.cat([h_nodes, node_msg], dim=-1))
        )

        h_edges_new = self.norm_edges(
            h_edges + self.edge_update(torch.cat([h_edges, edge_msg], dim=-1))
        )

        h_faces_new = self.norm_faces(
            h_faces + self.face_update(
                torch.cat([h_faces, msg_subunit_residue], dim=-1)
            )
        )

        h_subunits_new = self.norm_subunits(
            h_subunits + self.subunit_update(
                torch.cat([
                    msg_residue_subunit + msg_interface_subunit,
                    h_subunits,
                ], dim=-1)
            )
        )

        return h_nodes_new, h_edges_new, h_faces_new, h_subunits_new


class MultiSubunitTCPNet(nn.Module):
    """
    Full multi-subunit TCPNet encoder.

    Stacks multiple MultiSubunitTCPNetLayer for deep message passing
    across quaternary structure.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        super().__init__()

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.face_embed = nn.Linear(hidden_dim, hidden_dim)  # From node pooling
        self.subunit_embed = nn.Linear(hidden_dim, hidden_dim)  # From face pooling

        self.layers = nn.ModuleList([
            MultiSubunitTCPNetLayer(hidden_dim)
            for _ in range(n_layers)
        ])

        # Global readout
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # All cell types
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        enzyme_pcc: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode multi-subunit enzyme.

        Returns
        -------
        enzyme_embedding : [H] global enzyme representation
        h_residues : [n_residues, H] residue-level features
        h_subunits : [n_subunits, H] subunit-level features
        """
        device = next(self.parameters()).device

        # Get features from PCC
        node_feats = enzyme_pcc.get("node_features", torch.zeros(100, 64))
        edge_feats = enzyme_pcc.get("edge_features", torch.zeros(200, 32))
        edge_index = enzyme_pcc.get("edge_index", torch.zeros(2, 200, dtype=torch.long))

        if not isinstance(node_feats, torch.Tensor):
            node_feats = torch.tensor(node_feats, dtype=torch.float32)
        if not isinstance(edge_feats, torch.Tensor):
            edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)
        edge_index = edge_index.to(device)

        n_subunits = enzyme_pcc.get("n_subunits", 1)
        subunit_membership = enzyme_pcc.get("subunit_membership", {})
        interfaces = enzyme_pcc.get("4_cells", [])

        # Initial embeddings
        h_nodes = self.node_embed(node_feats)
        h_edges = self.edge_embed(edge_feats)

        # Pool nodes to faces (residues)
        n_residues = len(subunit_membership) if subunit_membership else node_feats.size(0) // 10
        h_faces = torch.zeros(n_residues, self.node_embed.out_features, device=device)
        for res_idx in range(n_residues):
            atom_start = res_idx * 10
            atom_end = min(atom_start + 10, h_nodes.size(0))
            if atom_start < h_nodes.size(0):
                h_faces[res_idx] = h_nodes[atom_start:atom_end].mean(dim=0)
        h_faces = self.face_embed(h_faces)

        # Pool faces to subunits
        h_subunits = torch.zeros(n_subunits, h_faces.size(1), device=device)
        for res_idx, sub_idx in subunit_membership.items():
            if res_idx < h_faces.size(0) and sub_idx < n_subunits:
                h_subunits[sub_idx] = h_subunits[sub_idx] + h_faces[res_idx]
        # Normalize by count
        for sub_idx in range(n_subunits):
            count = sum(1 for r, s in subunit_membership.items() if s == sub_idx)
            if count > 0:
                h_subunits[sub_idx] = h_subunits[sub_idx] / count
        h_subunits = self.subunit_embed(h_subunits)

        # Optional higher-rank incidence matrices from PCC
        B_23 = enzyme_pcc.get("B_23")  # (2, E_23) residues→subunits

        # Message passing layers
        for layer in self.layers:
            h_nodes, h_edges, h_faces, h_subunits = layer(
                h_nodes, h_edges, h_faces, h_subunits,
                edge_index, subunit_membership, interfaces, B_23,
            )

        # Global readout (pool all cell types)
        global_nodes = h_nodes.mean(dim=0)
        global_edges = h_edges.mean(dim=0)
        global_faces = h_faces.mean(dim=0)
        global_subunits = h_subunits.mean(dim=0)

        enzyme_embedding = self.global_pool(
            torch.cat([global_nodes, global_edges, global_faces, global_subunits])
        )

        return enzyme_embedding, h_faces, h_subunits


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Allosteric Cooperativity Prediction Head
# ══════════════════════════════════════════════════════════════════════════════


class AllostericCooperativityHead(nn.Module):
    """
    Predict Hill coefficient (n_H) from subunit interface features.

    Hill coefficient quantifies cooperativity:
    - n_H = 1: No cooperativity (Michaelis-Menten)
    - n_H > 1: Positive cooperativity (e.g., hemoglobin, n_H ≈ 2.8)
    - n_H < 1: Negative cooperativity

    Also predicts:
    - T/R state bias (allosteric two-state model)
    - Inter-subunit communication strength
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Subunit interaction encoder
        self.subunit_pair_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Interface feature encoder
        self.interface_encoder = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 4, hidden_dim // 2),  # +4 for interface props
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Hill coefficient predictor
        self.hill_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # T/R state bias predictor (from MWC model)
        self.tr_bias_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1], negative = T-state, positive = R-state
        )

        # Communication strength predictor
        self.comm_strength_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_subunits: torch.Tensor,    # [batch, n_subunits, H] or [n_subunits, H]
        interfaces: List[InterfaceCell],
        h_global: Optional[torch.Tensor] = None,  # [batch, H] or [H]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict allosteric cooperativity parameters.

        Returns
        -------
        predictions : dict with keys:
            'hill_coefficient': [batch, 1] predicted n_H (0.5 to 4.0)
            'tr_bias': [batch, 1] T/R state preference (-1 to 1)
            'communication_strength': [batch, 1] inter-subunit coupling (0 to 1)
        """
        # Handle batched vs unbatched input
        if h_subunits.dim() == 2:
            h_subunits = h_subunits.unsqueeze(0)  # [1, n_subunits, H]
            batched = False
        else:
            batched = True

        batch_size = h_subunits.size(0)
        device = h_subunits.device

        # Encode pairwise subunit interactions
        interface_feats = []

        for interface in interfaces:
            i, j = interface.subunit_pair

            for b in range(batch_size):
                if i < h_subunits.size(1) and j < h_subunits.size(1):
                    pair_feat = self.subunit_pair_encoder(
                        torch.cat([h_subunits[b, i], h_subunits[b, j]])
                    )

                    # Add interface properties
                    props = torch.tensor([
                        interface.contact_area / 2000.0,
                        interface.hydrogen_bonds / 20.0,
                        interface.salt_bridges / 10.0,
                        1.0 if interface.interface_type == "catalytic" else 0.0,
                    ], device=device)

                    combined = torch.cat([pair_feat, props])
                    interface_feat = self.interface_encoder(combined)
                    interface_feats.append(interface_feat)

        if interface_feats:
            # Aggregate interface features
            interface_embedding = torch.stack(interface_feats).mean(dim=0)
        else:
            # No interfaces - use zero embedding
            interface_embedding = torch.zeros(
                self.interface_encoder[-1].out_features, device=device
            )

        # Predict Hill coefficient
        hill_raw = self.hill_predictor(interface_embedding)
        # Constrain to physically reasonable range (0.5 to 4.0)
        hill_coefficient = torch.sigmoid(hill_raw) * 3.5 + 0.5

        # Predict T/R bias
        if h_global is not None:
            if h_global.dim() == 1:
                h_global = h_global.unsqueeze(0)
            tr_bias = self.tr_bias_predictor(h_global)
        else:
            # Use mean subunit embedding
            tr_bias = self.tr_bias_predictor(h_subunits.mean(dim=1))

        # Predict communication strength
        comm_strength = self.comm_strength_predictor(interface_embedding)

        results = {
            "hill_coefficient": hill_coefficient,
            "tr_bias": tr_bias,
            "communication_strength": comm_strength.expand(batch_size, 1),
        }

        if not batched:
            results = {k: v.squeeze(0) for k, v in results.items()}

        return results


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Complete Multi-Subunit ToPE Model
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MultiSubunitToPEConfig:
    """Configuration for multi-subunit ToPE model."""

    # Encoder
    node_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 256
    n_message_layers: int = 6

    # PCC construction
    interface_distance_cutoff: float = 5.0
    min_interface_contacts: int = 5

    # Tasks
    predict_hill_coefficient: bool = True
    predict_kinetics: bool = True
    predict_ec: bool = True


class MultiSubunitToPEModel(nn.Module):
    """
    Complete ToPE model for multi-subunit enzymes.

    Extends the whole-protein ToPE with:
    - 3-cell (subunit) and 4-cell (interface) representations
    - Cross-subunit message passing
    - Allosteric cooperativity prediction (Hill coefficient)
    """

    def __init__(self, config: Optional[MultiSubunitToPEConfig] = None):
        super().__init__()
        self.config = config or MultiSubunitToPEConfig()

        # Multi-subunit encoder
        self.encoder = MultiSubunitTCPNet(
            node_dim=self.config.node_dim,
            edge_dim=self.config.edge_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_message_layers,
        )

        # Task heads
        if self.config.predict_hill_coefficient:
            self.cooperativity_head = AllostericCooperativityHead(
                hidden_dim=self.config.hidden_dim
            )

        if self.config.predict_kinetics:
            self.kinetics_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 3),  # log_kcat, log_Km, log_kcat/Km
            )

        if self.config.predict_ec:
            # Simplified EC head
            self.ec_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 7),  # 7 EC classes
            )

    def forward(
        self,
        enzyme_pcc: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-subunit enzyme.

        Parameters
        ----------
        enzyme_pcc : dict
            Multi-subunit PCC with 3-cells and 4-cells

        Returns
        -------
        predictions : dict with task predictions
        """
        # Encode
        enzyme_emb, h_residues, h_subunits = self.encoder(enzyme_pcc)

        predictions = {
            "enzyme_embedding": enzyme_emb,
            "residue_embeddings": h_residues,
            "subunit_embeddings": h_subunits,
        }

        # Cooperativity prediction
        if self.config.predict_hill_coefficient:
            interfaces = enzyme_pcc.get("4_cells", [])
            coop_preds = self.cooperativity_head(h_subunits, interfaces, enzyme_emb)
            predictions.update(coop_preds)

        # Kinetics prediction
        if self.config.predict_kinetics:
            kinetics = self.kinetics_head(enzyme_emb)
            predictions["log_kcat"] = kinetics[..., 0]
            predictions["log_km"] = kinetics[..., 1]
            predictions["log_efficiency"] = kinetics[..., 2]

        # EC prediction
        if self.config.predict_ec:
            ec_logits = self.ec_head(enzyme_emb)
            predictions["ec_logits"] = ec_logits

        return predictions


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════════════


def build_multisubunit_enzyme_pcc(
    pdb_path: str,
    active_site_residues: List[Tuple[str, int]],
    config: Optional[MultiSubunitPCCConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function to build multi-subunit PCC from PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    active_site_residues : list of (chain_id, residue_id)
        Known catalytic residues
    config : MultiSubunitPCCConfig, optional

    Returns
    -------
    enzyme_pcc : dict
        Complete multi-subunit PCC
    """
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("enzyme", pdb_path)
    except ImportError:
        logger.warning("Bio.PDB not available; returning minimal PCC")
        structure = None
    except Exception as e:
        logger.warning(f"Could not parse PDB: {e}")
        structure = None

    builder = MultiSubunitPCCBuilder(config)
    return builder.build(structure, active_site_residues)


def extract_allosteric_features_from_pcc(
    enzyme_pcc: Dict[str, Any],
    radii: Optional[List[float]] = None,
) -> AllostericSpectralFeatures:
    """
    Extract allosteric spectral features from multi-subunit PCC.

    Parameters
    ----------
    enzyme_pcc : dict
        Multi-subunit PCC
    radii : list of float, optional
        Filtration radii (default: [3, 4, 5, 6, 7, 8] Å)

    Returns
    -------
    features : AllostericSpectralFeatures
    """
    if radii is None:
        radii = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    extractor = AllostericFeatureExtractor()
    return extractor.extract(enzyme_pcc, radii)
