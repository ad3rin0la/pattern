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
import warnings
from dataclasses import dataclass, field, replace
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
    """Validated active-site environment.

    Carries the atom (rank 0) and residue (rank 2) cells of the combinatorial
    complex *simultaneously* — atoms and residues are not an either/or choice
    but two ranks of one structure, linked by ``residue_id``.  Residue-level
    matching is still what validates the M-CSA annotations (it sidesteps the
    chain-ID mismatch problem), while the atom list carries the rank-0 cells
    that ``FeatureComputer`` and the complex's incidence maps consume.
    """

    pdb_id: str
    residues: List[ResidueRecord] = field(default_factory=list)
    atoms: List[AtomRecord] = field(default_factory=list)
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
        # Prefer the materialised rank-0 cells; fall back to the residue
        # summary counts when atoms were not populated (residue-only sites).
        if self.atoms:
            return len(self.atoms)
        return sum(r.n_atoms for r in self.residues)

    @property
    def elements(self) -> Set[str]:
        return {a.element for a in self.atoms if a.element}

    @property
    def has_metal(self) -> bool:
        metals = {
            "LI", "BE", "NA", "MG", "AL", "K", "CA", "SC", "TI", "V", "CR",
            "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "RB", "SR", "Y", "ZR",
            "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
            "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD",
            "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE",
            "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI",
        }
        return any(elem.upper() in metals for elem in self.elements)

    def ca_coords_array(self) -> np.ndarray:
        """Return (N_res, 3) Cα coordinate matrix."""
        return np.array([r.ca_coord for r in self.residues])

    def atoms_array(self) -> np.ndarray:
        """Return (N_atom, 3) atom coordinate matrix (rank-0 cell positions)."""
        if not self.atoms:
            return np.empty((0, 3))
        return np.array([a.coord for a in self.atoms])

    def atom_residue_incidence(self) -> np.ndarray:
        """Atom→residue incidence of the combinatorial complex.

        Returns an integer array ``b`` of length ``len(self.atoms)`` where
        ``b[a]`` is the index into ``self.residues`` of the residue that atom
        ``a`` belongs to (``-1`` if its parent residue is absent — e.g. an
        atom retained without its residue summary).  This is the rank-0 →
        rank-2 membership map the higher-rank cells are assembled from; both
        sides key on ``residue_id``.
        """
        res_index = {r.residue_id: i for i, r in enumerate(self.residues)}
        return np.array(
            [res_index.get(a.residue_id, -1) for a in self.atoms], dtype=int
        )

    @property
    def elements(self) -> Set[str]:
        """Set of element symbols present among the rank-0 atom cells."""
        return {a.element.upper() for a in self.atoms}

    @property
    def has_metal(self) -> bool:
        """True if any atom is a biologically common metal."""
        metals = {"FE", "ZN", "CU", "MN", "MG", "CA", "CO", "NI", "MO", "W"}
        return bool(self.elements & metals)

    def distance_matrix(self) -> np.ndarray:
        """Pairwise atom–atom distance matrix (N_atom, N_atom).

        Over the rank-0 atom cells (``atoms_array``); empty when no atoms are
        populated.  Feeds the radius filtration (see
        ``ActiveSiteExtractor.compute_filtration_adjacencies``).
        """
        coords = self.atoms_array()
        if coords.shape[0] == 0:
            return np.empty((0, 0))
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def atom_coords_array(self) -> np.ndarray:
        """Compatibility alias returning the atom coordinate matrix."""
        return self.atoms_array().astype(np.float64, copy=False)


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

        # Collect all residues within radius, plus their constituent atoms.
        # Both ranks are kept on the same ActiveSite so a single object
        # materialises the atom (rank 0) and residue (rank 2) cells of the
        # combinatorial complex simultaneously, linked by residue_id.
        matched_set = set(matched_cat)
        records: List[ResidueRecord] = []
        atom_records: List[AtomRecord] = []
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
            # Atoms inherit the catalytic flag of their parent residue.
            for atom in info.get("atoms", []):
                atom.is_catalytic = is_cat
                atom_records.append(atom)
            if is_cat:
                cat_ids.add(rec.residue_id)
            for atom_rec in info.get("atoms", []):
                atom_records.append(
                    replace(
                        atom_rec,
                        is_catalytic=is_cat,
                        role=cat_roles.get(key, ""),
                    )
                )

        site = ActiveSite(
            pdb_id=pdb_id,
            residues=records,
            atoms=atom_records,
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
                # Materialise the rank-0 atom cells alongside the residue
                # summary.  Atom and residue share residue_id, which is the
                # atom→residue incidence the combinatorial complex is built on.
                is_hetero = het_flag.strip() != ""
                atom_records: List[AtomRecord] = []
                for a in atoms:
                    try:
                        coord = np.array(a.get_vector().get_array(), dtype=np.float64)
                    except Exception:
                        coord = np.array(a.get_coord(), dtype=np.float64)
                    element = (a.element or "").strip() or a.get_name()[0]
                    atom_records.append(AtomRecord(
                        serial=int(a.get_serial_number() or 0),
                        name=a.get_name().strip(),
                        element=element,
                        residue_name=resname,
                        residue_number=resnum,
                        chain_id=chain_id,
                        coord=coord,
                        occupancy=float(a.get_occupancy() or 1.0),
                        b_factor=float(a.get_bfactor() or 0.0),
                        is_hetero=is_hetero,
                    ))

                key = (chain_id, resnum)
                residue_lookup[key] = {
                    "resname": resname,
                    "ca_coord": ca_coord,
                    "mean_b": mean_b,
                    "n_atoms": len(atoms),
                    "atoms": atom_records,
                }

        return residue_lookup, auth_map

    # ── Geometry helpers (for compatibility) ──────────────────────────────

    @staticmethod
    def compute_ca_distance_matrix(site: ActiveSite) -> np.ndarray:
        """Pairwise Cα distance matrix (N_res, N_res)."""
        coords = site.ca_coords_array()
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    @staticmethod
    def compute_filtration_adjacencies(
        active_site: ActiveSite,
        radii: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Atom-level adjacency matrices at each filtration radius.

        For each radius ``r`` returns the boolean atom–atom adjacency
        ``(dist <= r)`` over the rank-0 atom cells — the two-parameter radius
        sweep the persistent-homology filtration consumes.  Aligned 1:1 with
        ``active_site.atoms`` (and hence with the saved atom coords / features /
        atom_residue arrays).

        Returns
        -------
        dict mapping radius (float) → (N_atom, N_atom) int32 adjacency matrix.
        """
        if radii is None:
            from tope.data.config import FILTRATION_RADII
            radii = list(FILTRATION_RADII)

        dist = active_site.distance_matrix()
        return {float(r): (dist <= r).astype(np.int32) for r in radii}

    @staticmethod
    def compute_atom_distance_matrix(site: ActiveSite) -> np.ndarray:
        """Compatibility helper for pairwise atom distances."""
        return site.distance_matrix()


# ── Deprecated convenience wrapper ───────────────────────────────────────────

def extract_active_site(
    pdb_file,
    catalytic_residues: List,
    radius: float = 8.0,
    pdb_id: str = "",
    ec_number: str = "",
) -> Optional[ActiveSite]:
    """Extract an active-site environment using an 8Å radius crop.

    .. deprecated::
        This function hard-codes a spatial boundary and is incompatible with
        ToPE's whole-protein thesis.  Use
        ``tope.topology.build_whole_protein_pcc()`` for all training-pipeline
        code.  ``extract_active_site`` is retained **only** for visualization
        and debugging; do not call it from the training pipeline.
    """
    warnings.warn(
        "extract_active_site() is deprecated and must not be used in the "
        "training pipeline (Phase 2 migration).  "
        "Use tope.topology.build_whole_protein_pcc() instead, which encodes "
        "active-site membership as a boolean mask on the full protein.  "
        "extract_active_site() is retained for visualization / debugging only.",
        DeprecationWarning,
        stacklevel=2,
    )
    extractor = ActiveSiteExtractor()
    return extractor.extract(
        structure_path=Path(pdb_file),
        catalytic_residues=catalytic_residues,
        pdb_id=pdb_id,
        ec_number=ec_number,
        radius=radius,
    )
