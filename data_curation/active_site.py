"""
Active-site extraction from PDB structures.

Given a structure file and a list of catalytic residue annotations (from M-CSA),
extract the local atomic environment around the active site. The extraction
radius mirrors the filtration shell used in the topology bridge:

    ε = 2 Å  → covalent skeleton
    ε = 4 Å  → H-bond network / catalytic triad
    ε = 6 Å  → active-site pocket / substrate cavity
    ε = 8 Å  → allosteric shell / distal residues

We extract at the largest radius (default 8 Å) so downstream filtration
can sweep through all scales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from data_curation.config import (
    CATALYTIC_RESIDUE_PADDING,
    DEFAULT_ACTIVE_SITE_RADIUS,
    PipelineConfig,
)
from data_curation.mcsa_client import CatalyticResidue

logger = logging.getLogger(__name__)

# BioPython imports — optional but expected for production use.
try:
    from Bio.PDB import MMCIFParser, PDBParser, NeighborSearch, Selection
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Residue import Residue as BioResidue
    from Bio.PDB.Atom import Atom as BioAtom
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    logger.warning(
        "BioPython not installed — active-site extraction will be unavailable. "
        "Install with: pip install biopython"
    )


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AtomRecord:
    """Minimal atom representation for downstream feature computation."""

    serial: int
    name: str               # atom name, e.g. "CA", "NZ", "FE"
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
class ActiveSite:
    """Extracted active-site environment."""

    pdb_id: str
    atoms: List[AtomRecord] = field(default_factory=list)
    catalytic_residue_ids: Set[str] = field(default_factory=set)
    centroid: Optional[np.ndarray] = None       # geometric centre of catalytic atoms
    radius_used: float = DEFAULT_ACTIVE_SITE_RADIUS
    ec_number: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_residues(self) -> int:
        return len({a.residue_id for a in self.atoms})

    @property
    def n_catalytic_residues(self) -> int:
        return len(self.catalytic_residue_ids)

    @property
    def elements(self) -> Set[str]:
        return {a.element for a in self.atoms}

    @property
    def has_metal(self) -> bool:
        metals = {"FE", "ZN", "CU", "MN", "MG", "CA", "CO", "NI", "MO", "W"}
        return bool(self.elements & metals)

    def coords_array(self) -> np.ndarray:
        """Return (N, 3) coordinate matrix."""
        return np.array([a.coord for a in self.atoms])

    def distance_matrix(self) -> np.ndarray:
        """Pairwise Euclidean distance matrix (N, N)."""
        coords = self.coords_array()
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))


# ── Extractor ─────────────────────────────────────────────────────────────────

class ActiveSiteExtractor:
    """Extract active-site atomic environments from PDB/mmCIF files."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        if not HAS_BIOPYTHON:
            raise ImportError(
                "BioPython is required for active-site extraction. "
                "Install with: pip install biopython"
            )
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
        """Extract the active-site environment around catalytic residues.

        Parameters
        ----------
        structure_path : Path
            Path to a .cif or .pdb file.
        catalytic_residues : list of CatalyticResidue
            Annotated catalytic residues from M-CSA.
        pdb_id : str
            PDB identifier (for labelling).
        ec_number : str
            EC classification string.
        radius : float, optional
            Extraction radius in Å. Defaults to config.active_site_radius.

        Returns
        -------
        ActiveSite or None
            Extracted environment, or None if parsing/extraction fails.
        """
        if radius is None:
            radius = self.config.active_site_radius

        # Parse structure
        structure = self._parse_structure(structure_path, pdb_id)
        if structure is None:
            return None

        # Build a set of (chain_id, resnum) for catalytic residues
        cat_keys: Set[Tuple[str, int]] = set()
        for cr in catalytic_residues:
            cat_keys.add((cr.chain_id, cr.residue_number))

        # Find catalytic atoms to compute centroid
        cat_atoms: List[BioAtom] = []
        all_atoms: List[BioAtom] = list(structure.get_atoms())

        for atom in all_atoms:
            res = atom.get_parent()
            chain = res.get_parent()
            if chain is not None:
                chain_id = chain.get_id()
                resnum = res.get_id()[1]
                if (chain_id, resnum) in cat_keys:
                    cat_atoms.append(atom)

        if not cat_atoms:
            logger.warning(
                "%s: no catalytic atoms found for annotated residues %s",
                pdb_id, cat_keys,
            )
            # Fall back: try matching just by residue number across all chains
            cat_resnums = {rn for _, rn in cat_keys}
            for atom in all_atoms:
                res = atom.get_parent()
                if res.get_id()[1] in cat_resnums:
                    cat_atoms.append(atom)

            if not cat_atoms:
                logger.warning("%s: fallback also failed — skipping", pdb_id)
                return None

        # Compute centroid of catalytic atoms
        cat_coords = np.array([a.get_vector().get_array() for a in cat_atoms])
        centroid = cat_coords.mean(axis=0)

        # Neighbour search: find all atoms within radius of centroid
        ns = NeighborSearch(all_atoms)
        nearby_atoms = ns.search(centroid, radius + CATALYTIC_RESIDUE_PADDING, level="A")

        if not nearby_atoms:
            logger.warning("%s: no atoms within %.1f Å of centroid", pdb_id, radius)
            return None

        # Build AtomRecord list
        cat_residue_ids: Set[str] = set()
        records: List[AtomRecord] = []

        for atom in nearby_atoms:
            res: BioResidue = atom.get_parent()
            chain = res.get_parent()
            if chain is None:
                continue

            chain_id = chain.get_id()
            het_flag = res.get_id()[0]
            resnum = res.get_id()[1]
            resname = res.get_resname().strip()

            is_hetero = het_flag.strip() != ""
            is_water = resname in ("HOH", "WAT", "DOD")

            # Skip water unless configured to include it
            if is_water and not self.config.include_water:
                continue

            # Skip non-standard heteroatoms unless configured
            if is_hetero and not is_water and not self.config.include_heteroatoms:
                continue

            is_cat = (chain_id, resnum) in cat_keys
            res_id = f"{chain_id}:{resname}{resnum}"
            if is_cat:
                cat_residue_ids.add(res_id)

            element = atom.element.strip().upper() if atom.element else atom.get_name().strip()[0]

            records.append(AtomRecord(
                serial=atom.get_serial_number(),
                name=atom.get_name(),
                element=element,
                residue_name=resname,
                residue_number=resnum,
                chain_id=chain_id,
                coord=np.array(atom.get_vector().get_array(), dtype=np.float64),
                occupancy=atom.get_occupancy(),
                b_factor=atom.get_bfactor(),
                is_hetero=is_hetero,
                is_catalytic=is_cat,
            ))

        active_site = ActiveSite(
            pdb_id=pdb_id,
            atoms=records,
            catalytic_residue_ids=cat_residue_ids,
            centroid=centroid,
            radius_used=radius,
            ec_number=ec_number,
        )

        logger.debug(
            "%s: extracted %d atoms (%d residues, %d catalytic) within %.1f Å",
            pdb_id, active_site.n_atoms, active_site.n_residues,
            active_site.n_catalytic_residues, radius,
        )
        return active_site

    def extract_batch(
        self,
        entries: List[Dict],
        structure_dir: Path,
        fmt: str = "cif",
    ) -> List[ActiveSite]:
        """Extract active sites for a batch of entries.

        Parameters
        ----------
        entries : list of dict
            Each dict must have keys: "pdb_id", "catalytic_residues", "ec_number".
        structure_dir : Path
            Directory containing downloaded structure files.
        fmt : str
            File format extension ("cif" or "pdb").

        Returns
        -------
        list of ActiveSite
        """
        results: List[ActiveSite] = []
        total = len(entries)

        for i, entry in enumerate(entries, 1):
            pdb_id = entry["pdb_id"]
            ext = ".cif" if fmt == "cif" else ".pdb"
            path = structure_dir / f"{pdb_id}{ext}"

            if not path.exists():
                logger.warning("%s: structure file not found at %s", pdb_id, path)
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
                logger.info("Extracted active sites: %d / %d (%d OK)", i, total, len(results))

        logger.info(
            "Active-site extraction complete: %d / %d structures processed",
            len(results), total,
        )
        return results

    # ── Geometry helpers ─────────────────────────────────────────────────

    @staticmethod
    def compute_adjacency(
        active_site: ActiveSite,
        threshold: float,
    ) -> np.ndarray:
        """Binary adjacency matrix at a given distance threshold.

        This is one filtration step: atoms within `threshold` Å are connected.
        Sweeping threshold across FILTRATION_RADII produces the filtration
        used by persistent Laplacians downstream.
        """
        dist = active_site.distance_matrix()
        return (dist <= threshold).astype(np.int32)

    @staticmethod
    def compute_filtration_adjacencies(
        active_site: ActiveSite,
        radii: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Adjacency matrices at each filtration radius.

        Returns
        -------
        dict mapping radius (float) → adjacency matrix (N, N)
        """
        if radii is None:
            from data_curation.config import FILTRATION_RADII
            radii = FILTRATION_RADII

        dist = active_site.distance_matrix()
        return {r: (dist <= r).astype(np.int32) for r in radii}

    # ── Private ──────────────────────────────────────────────────────────

    def _parse_structure(
        self,
        path: Path,
        pdb_id: str = "",
    ) -> Optional[Structure]:
        """Parse a structure file (mmCIF or PDB)."""
        try:
            if path.suffix.lower() == ".cif":
                return self._cif_parser.get_structure(pdb_id, str(path))
            else:
                return self._pdb_parser.get_structure(pdb_id, str(path))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            return None
