"""
Residue-level active-site validation from PDB structures.

Given a structure file and M-CSA catalytic residue annotations,
validate that annotated residues exist in the structure and record
the residue-level environment. The combinatorial complex (Phase 2)
handles atom decomposition — curation just needs to confirm residues
are present and record their positions.

Replaces the atom-level ActiveSiteExtractor with a residue-level
approach that avoids the chain-ID mismatch problem entirely by
trying both label and auth chain IDs for CIF files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tope.data.config import (
    DEFAULT_ACTIVE_SITE_RADIUS,
    PipelineConfig,
)
from tope.data.mcsa_client import CatalyticResidue

logger = logging.getLogger(__name__)

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Structure import Structure
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AtomRecord:
    """Minimal atom representation for downstream feature computation.

    Atom-level structure consumed by :class:`tope.data.features.FeatureComputer`
    to build Ioffe-style physicochemical descriptors.  The curation-layer
    :class:`ActiveSite` below is residue-level (it only validates that the
    annotated catalytic residues are present); atom decomposition itself is a
    Phase-2 / combinatorial-complex concern, which is why this record is kept
    independent of how ``ActiveSite`` is assembled.
    """

    serial: int
    name: str                # atom name, e.g. "CA", "NZ", "FE"
    element: str             # element symbol, e.g. "C", "N", "FE"
    residue_name: str        # three-letter code
    residue_number: int
    chain_id: str
    coord: np.ndarray        # shape (3,), Cartesian coordinates in Å
    occupancy: float = 1.0
    b_factor: float = 0.0
    is_hetero: bool = False
    is_catalytic: bool = False

    @property
    def residue_id(self) -> str:
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}"


@dataclass
class ResidueRecord:
    """One residue in the active-site environment."""

    chain_id: str            # auth chain ID
    residue_name: str        # three-letter code (e.g. HIS, ASP)
    residue_number: int      # auth residue number
    ca_coord: np.ndarray     # Cα coordinate (3,), or residue centroid if no Cα
    mean_b_factor: float = 0.0
    n_atoms: int = 0
    is_catalytic: bool = False
    role: str = ""           # M-CSA role if catalytic

    @property
    def residue_id(self) -> str:
        return f"{self.chain_id}:{self.residue_name}{self.residue_number}"


@dataclass
class ActiveSite:
    """Validated active-site environment at residue level."""

    pdb_id: str
    residues: List[ResidueRecord] = field(default_factory=list)
    catalytic_residue_ids: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = None
    radius_used: float = DEFAULT_ACTIVE_SITE_RADIUS
    ec_number: str = ""

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def n_catalytic_residues(self) -> int:
        return len(self.catalytic_residue_ids)

    @property
    def n_atoms(self) -> int:
        return sum(r.n_atoms for r in self.residues)

    def ca_coords_array(self) -> np.ndarray:
        """Return (N, 3) Cα coordinate matrix."""
        return np.array([r.ca_coord for r in self.residues])


# ── Extractor ─────────────────────────────────────────────────────────────────

class ActiveSiteExtractor:
    """Validate and extract residue-level active-site environments."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython required: pip install biopython")
        self.config = config or PipelineConfig()
        self._cif_parser = MMCIFParser(QUIET=True)
        self._pdb_parser = PDBParser(QUIET=True)

    def extract(
        self,
        structure_path: Path,
        catalytic_residues: List[CatalyticResidue],
        pdb_id: str = "",
        ec_number: str = "",
        radius: Optional[float] = None,
    ) -> Optional[ActiveSite]:
        """Validate catalytic residues and extract residue-level environment.

        Parameters
        ----------
        structure_path : Path to .cif or .pdb file
        catalytic_residues : M-CSA annotations
        pdb_id : PDB identifier
        ec_number : EC classification
        radius : extraction radius in Å (default from config)

        Returns
        -------
        ActiveSite with residue-level records, or None on failure.
        """
        if radius is None:
            radius = self.config.active_site_radius

        structure = self._parse_structure(structure_path, pdb_id)
        if structure is None:
            return None

        # Build residue lookup: (chain_id, resnum) → residue info
        # For CIF files, index by BOTH label and auth chain IDs
        is_cif = structure_path.suffix.lower() in ('.cif', '.mmcif')
        residue_lookup, auth_map = self._build_residue_lookup(
            structure, structure_path, is_cif
        )

        # M-CSA uses auth chain IDs — build the target set
        cat_keys: Set[Tuple[str, int]] = set()
        cat_roles: Dict[Tuple[str, int], str] = {}
        for cr in catalytic_residues:
            key = (cr.chain_id, cr.residue_number)
            cat_keys.add(key)
            cat_roles[key] = cr.role

        # Match catalytic residues against structure
        matched_cat: List[Tuple[str, int]] = []
        for key in cat_keys:
            if key in residue_lookup:
                matched_cat.append(key)
            elif is_cif:
                # Try mapping auth → label chain ID
                auth_chain, resnum = key
                label_chain = auth_map.get(auth_chain)
                if label_chain and (label_chain, resnum) in residue_lookup:
                    matched_cat.append((label_chain, resnum))
                    # Update the role mapping for the label chain ID
                    cat_roles[(label_chain, resnum)] = cat_roles[key]
                else:
                    # Last resort: match by resnum across all chains
                    for (c, r), _ in residue_lookup.items():
                        if r == resnum:
                            matched_cat.append((c, r))
                            cat_roles[(c, r)] = cat_roles[key]
                            break

        if not matched_cat:
            logger.warning(
                "%s: no catalytic residues matched in structure "
                "(tried %d M-CSA residues, structure has %d residues)",
                pdb_id, len(cat_keys), len(residue_lookup),
            )
            return None

        if len(matched_cat) < len(cat_keys):
            logger.info(
                "%s: matched %d / %d catalytic residues",
                pdb_id, len(matched_cat), len(cat_keys),
            )

        # Compute centroid from catalytic residue Cα positions
        cat_coords = []
        for key in matched_cat:
            info = residue_lookup[key]
            cat_coords.append(info["ca_coord"])
        centroid = np.mean(cat_coords, axis=0)

        # Collect all residues within radius
        matched_set = set(matched_cat)
        records: List[ResidueRecord] = []
        cat_ids: Set[str] = set()

        for key, info in residue_lookup.items():
            dist = np.linalg.norm(info["ca_coord"] - centroid)
            if dist > radius:
                continue

            is_cat = key in matched_set
            rec = ResidueRecord(
                chain_id=key[0],
                residue_name=info["resname"],
                residue_number=key[1],
                ca_coord=info["ca_coord"],
                mean_b_factor=info["mean_b"],
                n_atoms=info["n_atoms"],
                is_catalytic=is_cat,
                role=cat_roles.get(key, ""),
            )
            records.append(rec)
            if is_cat:
                cat_ids.add(rec.residue_id)

        site = ActiveSite(
            pdb_id=pdb_id,
            residues=records,
            catalytic_residue_ids=cat_ids,
            centroid=centroid,
            radius_used=radius,
            ec_number=ec_number,
        )

        logger.debug(
            "%s: %d residues within %.1f Å (%d catalytic)",
            pdb_id, site.n_residues, radius, site.n_catalytic_residues,
        )
        return site

    # ── Batch extraction ─────────────────────────────────────────────────

    def extract_batch(
        self,
        entries: List[Dict],
        structure_dir: Path,
        fmt: str = "pdb",
    ) -> List[ActiveSite]:
        """Extract active sites for a batch of entries."""
        results: List[ActiveSite] = []
        total = len(entries)

        for i, entry in enumerate(entries, 1):
            pdb_id = entry["pdb_id"]

            # Try both formats
            path = structure_dir / f"{pdb_id}.{fmt}"
            if not path.exists():
                alt_fmt = "cif" if fmt == "pdb" else "pdb"
                path = structure_dir / f"{pdb_id}.{alt_fmt}"

            if not path.exists():
                continue

            site = self.extract(
                structure_path=path,
                catalytic_residues=entry["catalytic_residues"],
                pdb_id=pdb_id,
                ec_number=entry.get("ec_number", ""),
            )

            if site is not None:
                results.append(site)

            if i % 100 == 0 or i == total:
                logger.info("Active sites: %d / %d (%d OK)", i, total, len(results))

        logger.info("Done: %d / %d structures", len(results), total)
        return results

    # ── Private ──────────────────────────────────────────────────────────

    def _parse_structure(self, path: Path, pdb_id: str = "") -> Optional[Structure]:
        """Parse a structure file."""
        try:
            if path.suffix.lower() in ('.cif', '.mmcif'):
                return self._cif_parser.get_structure(pdb_id or "X", str(path))
            else:
                return self._pdb_parser.get_structure(pdb_id or "X", str(path))
        except Exception as exc:
            logger.warning("%s: parse failed: %s", pdb_id, exc)
            return None

    def _build_residue_lookup(
        self,
        structure: Structure,
        path: Path,
        is_cif: bool,
    ) -> Tuple[Dict[Tuple[str, int], dict], Dict[str, str]]:
        """Build (chain_id, resnum) → residue info lookup.

        For CIF files, also builds auth_chain → label_chain mapping
        by reading the mmCIF dict directly.

        Returns
        -------
        residue_lookup : dict mapping (chain, resnum) → info dict
        auth_map : dict mapping auth_chain_id → label_chain_id
        """
        residue_lookup: Dict[Tuple[str, int], dict] = {}
        auth_map: Dict[str, str] = {}

        # Build auth → label chain map for CIF
        if is_cif:
            try:
                from Bio.PDB.MMCIF2Dict import MMCIF2Dict
                mmcif_dict = MMCIF2Dict(str(path))
                label_chains = mmcif_dict.get("_atom_site.label_asym_id", [])
                auth_chains = mmcif_dict.get("_atom_site.auth_asym_id", [])
                for lc, ac in zip(label_chains, auth_chains):
                    if ac not in auth_map:
                        auth_map[ac] = lc
            except Exception:
                pass  # Fall back to BioPython chain IDs

        # Iterate structure residues
        model = structure[0]
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                het_flag = residue.get_id()[0]
                if het_flag.strip() not in ("", "H_MSE"):
                    continue  # skip water and most heteroatoms

                resnum = residue.get_id()[1]
                resname = residue.get_resname().strip()

                # Get Cα coord, or residue centroid if no Cα
                atoms = list(residue.get_atoms())
                if not atoms:
                    continue

                ca_coord = None
                for atom in atoms:
                    if atom.get_name() == "CA":
                        ca_coord = np.array(atom.get_vector().get_array(), dtype=np.float64)
                        break

                if ca_coord is None:
                    coords = [a.get_vector().get_array() for a in atoms]
                    ca_coord = np.mean(coords, axis=0).astype(np.float64)

                b_factors = [a.get_bfactor() for a in atoms]
                mean_b = float(np.mean(b_factors)) if b_factors else 0.0

                key = (chain_id, resnum)
                residue_lookup[key] = {
                    "resname": resname,
                    "ca_coord": ca_coord,
                    "mean_b": mean_b,
                    "n_atoms": len(atoms),
                }

        return residue_lookup, auth_map

    # ── Geometry helpers (for compatibility) ──────────────────────────────

    @staticmethod
    def compute_ca_distance_matrix(site: ActiveSite) -> np.ndarray:
        """Pairwise Cα distance matrix (N_res, N_res)."""
        coords = site.ca_coords_array()
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))
