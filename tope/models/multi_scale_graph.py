"""
Multi-Scale Protein Graph Construction
========================================

Builds a three-zone hierarchical protein representation:

    Zone 1  (0–8 Å from active site):  Atomic-level ToPE spectral features.
    Zone 2  (8–20 Å):                  Residue-level aggregated Ioffe descriptors.
    Zone 3  (>20 Å):                   Residue-level lightweight structural features.

The unified graph contains heterogeneous nodes (different feature
dimensionalities per zone) connected by spatial-proximity and backbone
edges.  This is consumed by ``WholeProteinTCPNet`` for multi-scale
message passing.

References
----------
AlphaFold 3 (Nature 2024) — multi-scale complex prediction.
Benkovic & Hammes-Schiffer (2006) — distant mutations in DHFR.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_AMINO_ACIDS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
_AMINO_ACID_INDEX = {name: i for i, name in enumerate(_AMINO_ACIDS)}

# Optional BioPython
try:
    from Bio.PDB import PDBParser, DSSP
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MultiScaleGraphConfig:
    """Hyper-parameters for multi-scale graph construction."""

    zone1_radius: float = 8.0      # Å — active-site atomic detail
    zone2_radius: float = 20.0     # Å — second-shell residue detail
    edge_cutoff: float = 12.0      # Å — spatial edge threshold
    contact_cutoff: float = 5.0    # Å — tight contact threshold
    zone1_feat_dim: int = 128      # Feature dim for Zone 1 residue nodes
    zone2_feat_dim: int = 64       # Feature dim for Zone 2 residue nodes
    zone3_feat_dim: int = 32       # Feature dim for Zone 3 residue nodes
    edge_feat_dim: int = 4         # Edge feature dim


# ── Zone assignment ───────────────────────────────────────────────────────────

def assign_zones(
    structure: Any,
    catalytic_residues: List[Tuple[str, int]],
    zone1_radius: float = 8.0,
    zone2_radius: float = 20.0,
) -> Dict[str, Any]:
    """Assign each residue to Zone 1, 2, or 3.

    Parameters
    ----------
    structure : Bio.PDB.Structure
    catalytic_residues : list of (chain_id, residue_seq_id)
    zone1_radius, zone2_radius : distance thresholds in Å

    Returns
    -------
    zones : dict with keys
        'zone1_atoms'    : list of Bio.PDB.Atom
        'zone1_residues' : list of (chain_id, res_id)
        'zone2_residues' : list of (chain_id, res_id)
        'zone3_residues' : list of (chain_id, res_id)
        'all_residues'   : list of (chain_id, res_id)
        'catalytic_center' : np.ndarray (3,)
    """
    # Compute catalytic centre
    cat_atoms = []
    model = structure[0]
    for chain_id, res_id in catalytic_residues:
        chain = model[chain_id]
        residue = chain[res_id]
        cat_atoms.extend(list(residue.get_atoms()))

    cat_center = np.mean([a.coord for a in cat_atoms], axis=0)

    zones: Dict[str, Any] = {
        "zone1_atoms": [],
        "zone1_residues": [],
        "zone2_residues": [],
        "zone3_residues": [],
        "all_residues": [],
        "catalytic_center": cat_center,
    }

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue  # skip heteroatoms / water

            res_atoms = list(residue.get_atoms())
            if not res_atoms:
                continue

            res_center = np.mean([a.coord for a in res_atoms], axis=0)
            dist = np.linalg.norm(res_center - cat_center)

            rid = (chain.id, residue.id[1])
            zones["all_residues"].append(rid)

            if dist < zone1_radius:
                zones["zone1_residues"].append(rid)
                zones["zone1_atoms"].extend(res_atoms)
            elif dist < zone2_radius:
                zones["zone2_residues"].append(rid)
            else:
                zones["zone3_residues"].append(rid)

    logger.info(
        "Zone assignment: Z1=%d atoms / %d residues, Z2=%d residues, Z3=%d residues",
        len(zones["zone1_atoms"]),
        len(zones["zone1_residues"]),
        len(zones["zone2_residues"]),
        len(zones["zone3_residues"]),
    )
    return zones


# ── Zone-specific feature builders ───────────────────────────────────────────

def build_zone1_residue_features(
    zones: Dict[str, Any],
    spectral_features: Optional[Dict[str, Any]] = None,
    feat_dim: int = 128,
) -> Dict[Tuple[str, int], torch.Tensor]:
    """Aggregate Zone 1 atomic spectral features to residue level.

    If spectral_features is None (Phase 2 not yet run), returns zero
    vectors so the downstream model can still be tested structurally.
    """
    residue_features: Dict[Tuple[str, int], torch.Tensor] = {}

    # Group atoms by parent residue
    residue_to_atoms: Dict[Tuple[str, int], list] = defaultdict(list)
    for atom in zones["zone1_atoms"]:
        parent = atom.get_parent()
        rid = (parent.get_parent().id, parent.id[1])
        residue_to_atoms[rid].append(atom)

    for rid in zones["zone1_residues"]:
        if spectral_features is not None and rid in spectral_features:
            feat = spectral_features[rid]
            if isinstance(feat, np.ndarray):
                feat = torch.tensor(feat[:feat_dim], dtype=torch.float32)
            elif isinstance(feat, torch.Tensor):
                feat = feat[:feat_dim].float()
            # Pad if shorter
            if feat.size(0) < feat_dim:
                feat = torch.cat([feat, torch.zeros(feat_dim - feat.size(0))])
            residue_features[rid] = feat
        else:
            residue_features[rid] = torch.zeros(feat_dim)

    return residue_features


# Ioffe element descriptors (subset for aggregation)
_ELEMENT_CHI = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58,
    "P": 2.19, "Se": 2.55, "F": 3.98, "Cl": 3.16, "Br": 2.96,
    "Fe": 1.83, "Zn": 1.65, "Mg": 1.31, "Cu": 1.90, "Mn": 1.55,
    "Co": 1.88, "Ni": 1.91,
}

_METALS = {"Fe", "Zn", "Mg", "Cu", "Mn", "Co", "Ni", "Mo", "W", "V"}


def build_zone2_features(
    structure: Any,
    zone2_residues: List[Tuple[str, int]],
    feat_dim: int = 64,
) -> Tuple[Dict[Tuple[str, int], torch.Tensor], Dict[Tuple[str, int], np.ndarray]]:
    """Build medium-detail residue-level features for Zone 2."""
    features = {}
    coords = {}
    model = structure[0]

    for chain_id, res_id in zone2_residues:
        chain = model[chain_id]
        residue = chain[res_id]
        res_atoms = list(residue.get_atoms())
        atom_coords = np.array([a.coord for a in res_atoms])
        atom_types = [a.element.strip() for a in res_atoms]

        coords[(chain_id, res_id)] = np.mean(atom_coords, axis=0)

        chi_values = [_ELEMENT_CHI.get(el, 2.5) for el in atom_types]
        coord_numbers = []
        for a in res_atoms:
            n = sum(
                1 for b in res_atoms
                if b is not a and np.linalg.norm(a.coord - b.coord) < 2.8
            )
            coord_numbers.append(n)

        raw = [
            np.mean(chi_values) if chi_values else 0.0,
            np.var(chi_values) if len(chi_values) > 1 else 0.0,
            max(chi_values) if chi_values else 0.0,
            min(chi_values) if chi_values else 0.0,
            sum(1 for el in atom_types if el == "O") / 10.0,
            sum(1 for el in atom_types if el == "N") / 10.0,
            sum(1 for el in atom_types if el == "S") / 10.0,
            1.0 if any(el in _METALS for el in atom_types) else 0.0,
            np.mean(coord_numbers) / 6.0 if coord_numbers else 0.0,
            len(res_atoms) / 30.0,
            float(residue.resname in ("ARG", "LYS", "ASP", "GLU")),
        ]
        # Pad to feat_dim
        raw = raw + [0.0] * (feat_dim - len(raw))
        features[(chain_id, res_id)] = torch.tensor(raw[:feat_dim], dtype=torch.float32)

    return features, coords


def build_zone3_features(
    structure: Any,
    zone3_residues: List[Tuple[str, int]],
    catalytic_center: np.ndarray,
    pdb_file: Optional[str] = None,
    feat_dim: int = 32,
) -> Tuple[Dict[Tuple[str, int], torch.Tensor], Dict[Tuple[str, int], np.ndarray]]:
    """Build lightweight structural features for Zone 3."""
    features = {}
    coords = {}
    model = structure[0]

    # Try DSSP for secondary structure
    has_dssp = False
    dssp_dict: Dict = {}
    if pdb_file is not None and HAS_BIOPYTHON:
        try:
            dssp_obj = DSSP(model, pdb_file, dssp="mkdssp")
            dssp_dict = dict(dssp_obj)
            has_dssp = True
        except Exception:
            pass

    ss_map = {"H": [1, 0, 0], "E": [0, 1, 0]}

    for chain_id, res_id in zone3_residues:
        chain = model[chain_id]
        residue = chain[res_id]
        res_atoms = list(residue.get_atoms())
        if not res_atoms:
            continue

        res_center = np.mean([a.coord for a in res_atoms], axis=0)
        coords[(chain_id, res_id)] = res_center
        dist_to_active = np.linalg.norm(res_center - catalytic_center)

        # Secondary structure
        ss_onehot = [0, 0, 1]  # default coil
        rasa = 0.5
        if has_dssp:
            dssp_key = (chain_id, (" ", res_id, " "))
            if dssp_key in dssp_dict:
                ss_char = dssp_dict[dssp_key][2]
                rasa = dssp_dict[dssp_key][3]
                ss_onehot = ss_map.get(ss_char, [0, 0, 1])

        mean_b = np.mean([a.bfactor for a in res_atoms])
        resname = residue.resname

        raw = [
            dist_to_active / 50.0,
            *ss_onehot,
            rasa,
            mean_b / 50.0,
            float(resname in ("ARG", "LYS", "ASP", "GLU")),
            float(resname in ("SER", "THR", "ASN", "GLN", "TYR")),
            float(resname in ("ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET")),
        ]
        raw = raw + [0.0] * (feat_dim - len(raw))
        features[(chain_id, res_id)] = torch.tensor(raw[:feat_dim], dtype=torch.float32)

    return features, coords


# ── Edge construction ─────────────────────────────────────────────────────────

def build_protein_edges(
    residues: List[Tuple[str, int]],
    coords: torch.Tensor,
    edge_cutoff: float = 12.0,
    contact_cutoff: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build undirected edges between residues.

    Edge features: [normalised_distance, is_backbone, is_contact, is_spatial]
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords.numpy())
        use_kdtree = True
    except ImportError:
        use_kdtree = False

    src, dst = [], []
    edge_feats = []

    n = len(residues)
    for i in range(n):
        if use_kdtree:
            neighbours = tree.query_ball_point(coords[i].numpy(), r=edge_cutoff)
        else:
            neighbours = range(n)

        for j in neighbours:
            if j <= i:
                continue
            dist = torch.norm(coords[i] - coords[j]).item()
            if dist > edge_cutoff:
                continue

            is_backbone = (
                residues[i][0] == residues[j][0]
                and abs(residues[i][1] - residues[j][1]) == 1
            )
            ef = torch.tensor([
                dist / edge_cutoff,
                1.0 if is_backbone else 0.0,
                1.0 if dist < contact_cutoff else 0.0,
                1.0,
            ])
            # Undirected
            src.extend([i, j])
            dst.extend([j, i])
            edge_feats.extend([ef, ef])

    if not src:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 4), dtype=torch.float32),
        )

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_features = torch.stack(edge_feats)
    return edge_index, edge_features


# ── Unified graph builder ────────────────────────────────────────────────────

class MultiScaleProteinGraph:
    """Three-zone hierarchical protein graph builder.

    Usage::

        builder = MultiScaleProteinGraph(pdb_file, catalytic_residues)
        graph = builder.build()
    """

    def __init__(
        self,
        pdb_file: str,
        catalytic_residues: List[Tuple[str, int]],
        cfg: Optional[MultiScaleGraphConfig] = None,
        spectral_features: Optional[Dict] = None,
        domain_annotations: Optional[Sequence[Any]] = None,
    ):
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required for MultiScaleProteinGraph")

        self.pdb_file = pdb_file
        self.catalytic_residues = catalytic_residues
        self.cfg = cfg or MultiScaleGraphConfig()
        self.spectral_features = spectral_features
        self.domain_annotations = list(domain_annotations or [])

        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("enzyme", pdb_file)
        self.zones = assign_zones(
            self.structure,
            catalytic_residues,
            self.cfg.zone1_radius,
            self.cfg.zone2_radius,
        )

    def build(self) -> Dict[str, Any]:
        """Build the unified protein graph."""
        # Zone 1 — atomic-level aggregated to residue
        z1_feats = build_zone1_residue_features(
            self.zones, self.spectral_features, self.cfg.zone1_feat_dim
        )

        # Zone 2 — residue-level aggregated Ioffe descriptors
        z2_feats, z2_coords = build_zone2_features(
            self.structure, self.zones["zone2_residues"], self.cfg.zone2_feat_dim
        )

        # Zone 3 — lightweight structural features
        z3_feats, z3_coords = build_zone3_features(
            self.structure,
            self.zones["zone3_residues"],
            self.zones["catalytic_center"],
            self.pdb_file,
            self.cfg.zone3_feat_dim,
        )

        # Combine all residues in order
        all_residues = (
            self.zones["zone1_residues"]
            + self.zones["zone2_residues"]
            + self.zones["zone3_residues"]
        )

        node_features = []
        node_coords = []
        zone_assignments = []

        model = self.structure[0]
        for rid in all_residues:
            if rid in z1_feats:
                nf = z1_feats[rid]
                coord = self._residue_center(model, rid)
                zone = 0
            elif rid in z2_feats:
                nf = z2_feats[rid]
                coord = z2_coords[rid]
                zone = 1
            elif rid in z3_feats:
                nf = z3_feats[rid]
                coord = z3_coords[rid]
                zone = 2
            else:
                continue

            node_features.append(nf)
            node_coords.append(
                coord if isinstance(coord, torch.Tensor) else torch.tensor(coord, dtype=torch.float32)
            )
            zone_assignments.append(zone)

        node_features_tensor = torch.stack(node_features)
        node_coords_tensor = torch.stack(node_coords)
        zone_tensor = torch.tensor(zone_assignments, dtype=torch.long)

        edge_index, edge_features = build_protein_edges(
            all_residues,
            node_coords_tensor,
            self.cfg.edge_cutoff,
            self.cfg.contact_cutoff,
        )

        # Catalytic residue mask
        catalytic_set = set(self.catalytic_residues)
        catalytic_mask = torch.tensor(
            [1.0 if rid in catalytic_set else 0.0 for rid in all_residues],
            dtype=torch.float32,
        )
        residue_domain, domains, domain_architecture = self._domain_cells(all_residues)
        chain_vocab = {chain: i for i, chain in enumerate(dict.fromkeys(
            chain for chain, _ in all_residues
        ))}
        chain_index = torch.tensor(
            [chain_vocab[chain] for chain, _ in all_residues], dtype=torch.long
        )
        sequence_index = torch.tensor(
            [residue_number for _, residue_number in all_residues], dtype=torch.long
        )
        # Default sequence modality: residue-identity one-hot from the structure.
        # Callers may replace this tensor with pretrained sequence embeddings.
        sequence_features = torch.zeros((len(all_residues), 21), dtype=torch.float32)
        for i, (chain_id, residue_number) in enumerate(all_residues):
            residue_name = model[chain_id][residue_number].resname
            sequence_features[i, _AMINO_ACID_INDEX.get(residue_name, 20)] = 1.0

        return {
            "num_nodes": len(all_residues),
            "node_features": node_features_tensor,
            "node_coords": node_coords_tensor,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "zone_assignments": zone_tensor,
            "residue_ids": all_residues,
            "catalytic_residues": self.catalytic_residues,
            "catalytic_mask": catalytic_mask,
            "residue_domain": residue_domain,
            "domains": domains,
            "domain_architecture": domain_architecture,
            "chain_index": chain_index,
            "sequence_index": sequence_index,
            "sequence_features": sequence_features,
        }

    def _domain_cells(
        self, residue_ids: List[Tuple[str, int]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, List[int]]]:
        """Materialise ordered domain cells and residue→domain incidence."""
        membership = torch.full((len(residue_ids),), -1, dtype=torch.long)
        domains: List[Dict[str, Any]] = []
        architecture: Dict[str, List[int]] = {}
        previous_end: Dict[str, int] = {}
        ordered = sorted(
            self.domain_annotations,
            key=lambda d: (d.chain_id, d.start, d.end, d.family),
        )
        for annotation in ordered:
            if annotation.start > annotation.end:
                raise ValueError(f"invalid domain interval: {annotation}")
            if annotation.chain_id in previous_end and annotation.start <= previous_end[annotation.chain_id]:
                raise ValueError(f"overlapping domains on chain {annotation.chain_id}")
            previous_end[annotation.chain_id] = annotation.end
            member_indices = [
                i for i, (chain, residue_number) in enumerate(residue_ids)
                if chain == annotation.chain_id
                and annotation.start <= residue_number <= annotation.end
            ]
            if not member_indices:
                raise ValueError(f"domain {annotation.family} contains no graph residues")
            domain_idx = len(domains)
            membership[member_indices] = domain_idx
            domains.append({
                "domain_id": annotation.domain_id or f"{annotation.family}:{domain_idx + 1}",
                "family": annotation.family,
                "chain_id": annotation.chain_id,
                "start": annotation.start,
                "end": annotation.end,
                "residue_indices": member_indices,
            })
            architecture.setdefault(annotation.chain_id, []).append(domain_idx)
        return membership, domains, architecture

    @staticmethod
    def _residue_center(model: Any, rid: Tuple[str, int]) -> torch.Tensor:
        chain_id, res_id = rid
        residue = model[chain_id][res_id]
        atoms = list(residue.get_atoms())
        center = np.mean([a.coord for a in atoms], axis=0)
        return torch.tensor(center, dtype=torch.float32)
