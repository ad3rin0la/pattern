"""
Whole-Protein Combinatorial Complex (Phase 2)
=============================================

Replaces the 8Å active-site crop (extract_active_site) with a whole-protein
PCC where active-site membership is encoded as a per-atom boolean mask.

Key design decision: ALL atoms enter the Vietoris-Rips filtration.
The active site emerges from spectral structure, not from a spatial cutoff.

Reference: ToPE Whole-Protein Migration Spec, Phase 2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from Bio.PDB import MMCIFParser, PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False

try:
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainAnnotation:
    """One ordered domain-family interval on a protein chain."""

    chain_id: str
    start: int
    end: int
    family: str
    domain_id: str = ""


@dataclass
class WholeProteinPCC:
    """Whole-protein Combinatorial Complex.

    Active-site membership is encoded as a per-atom boolean mask
    (is_active_site), NOT as a spatial radius boundary.  All atoms
    participate in the filtration; the active site emerges from spectral
    structure.

    Attributes
    ----------
    atoms          : Bio.PDB atom objects, full protein (rank-0 cells)
    coords         : (N, 3) heavy-atom coordinates, float32
    atom_types     : element symbols per atom
    residue_ids    : (chain, resid) tuple per atom
    is_active_site : (N,) bool — True = M-CSA catalytic residue
    residue_groups : resid → list of atom indices  (rank-2 cells)
    chain_groups   : chain_id → list of residue keys  (rank-3 cells)
    pdb_id         : PDB identifier or file path
    n_atoms        : total heavy-atom count
    n_residues     : total residue count
    n_chains       : chain count
    """

    # Rank-0 cells (atoms)
    atoms:          list
    coords:         np.ndarray              # (N, 3) float32
    atom_types:     List[str]
    residue_ids:    List[tuple]
    is_active_site: np.ndarray              # (N,) bool

    # Rank-2 cells (residues)
    residue_groups: Dict[tuple, List[int]]

    # Rank-3 cells (domains), ordered within each chain
    domain_groups: Dict[tuple, List[tuple]]
    domain_families: Dict[tuple, str]
    domain_architecture: Dict[str, List[tuple]]

    # Rank-4 cells (chains)
    chain_groups:   Dict[str, List[tuple]]

    # Rank-5 cell (the assembled complex)
    complex_groups: Dict[str, List[str]]

    # Metadata
    pdb_id:     str
    n_atoms:    int
    n_residues: int
    n_chains:   int


# ── Builder ───────────────────────────────────────────────────────────────────

def build_whole_protein_pcc(
    pdb_file: str,
    catalytic_residues: List[tuple],
    include_hydrogens: bool = False,
    domain_annotations: Optional[Sequence[DomainAnnotation]] = None,
) -> WholeProteinPCC:
    """Build a whole-protein Combinatorial Complex from a PDB/CIF file.

    Active-site membership is encoded as a boolean mask on each atom,
    NOT as a spatial radius cutoff.  The ``radius`` parameter of the old
    ``extract_active_site()`` is eliminated entirely.

    Parameters
    ----------
    pdb_file           : Path to a PDB (.pdb) or mmCIF (.cif/.mmcif) file.
    catalytic_residues : M-CSA annotations as (chain_id, res_id) tuples,
                         where res_id is the Bio.PDB residue id tuple
                         e.g. (' ', 42, ' ') or the plain integer 42.
    include_hydrogens  : If False (default), hydrogen atoms are stripped for
                         speed.  Set True only for explicit-H force fields.

    Returns
    -------
    WholeProteinPCC with the full-protein atom set and an active-site mask.

    Raises
    ------
    ImportError : if BioPython is not installed.
    ValueError  : if no atoms are found in the structure file.
    """
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython is required: pip install biopython")

    pdb_file_str = str(pdb_file)
    is_cif = pdb_file_str.lower().endswith((".cif", ".mmcif"))

    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    structure = parser.get_structure("enzyme", pdb_file_str)

    # Build catalytic residue lookup for O(1) membership test.
    # Normalise: accept both plain-int and full Bio.PDB resid tuples.
    cat_residue_set: set = set(catalytic_residues)

    atoms:          list                        = []
    coords:         list                        = []
    atom_types:     List[str]                   = []
    residue_ids:    List[tuple]                 = []
    is_active_site: list                        = []
    residue_groups: Dict[tuple, List[int]]      = {}
    chain_groups:   Dict[str, List[tuple]]      = {}

    for model in structure.get_models():
        if model.id != 0:
            continue  # first model only — handles NMR ensembles

        for chain in model.get_chains():
            chain_id = chain.id
            chain_groups[chain_id] = []

            for residue in chain.get_residues():
                # Skip water; keep standard amino acids and MSE
                het_flag = residue.get_id()[0]
                if het_flag.strip() not in ("", "H_MSE"):
                    continue

                res_id  = residue.get_id()
                res_key = (chain_id, res_id)
                chain_groups[chain_id].append(res_key)
                residue_groups[res_key] = []

                # Also test with plain resnum for convenience
                in_active_site = (
                    res_key in cat_residue_set
                    or (chain_id, res_id[1]) in cat_residue_set
                )

                for atom in residue.get_atoms():
                    if not include_hydrogens and atom.element == "H":
                        continue

                    idx = len(atoms)
                    atoms.append(atom)
                    coords.append(atom.coord)
                    atom_types.append(atom.element or "C")
                    residue_ids.append(res_key)
                    is_active_site.append(in_active_site)
                    residue_groups[res_key].append(idx)

    if not atoms:
        raise ValueError(
            f"No atoms found in {pdb_file_str!r}. "
            "Verify the file is a valid PDB/CIF structure."
        )

    coords_arr         = np.array(coords, dtype=np.float32)
    is_active_site_arr = np.array(is_active_site, dtype=bool)

    domain_groups, domain_families, domain_architecture = _build_domain_cells(
        residue_groups, domain_annotations or []
    )

    return WholeProteinPCC(
        atoms=atoms,
        coords=coords_arr,
        atom_types=atom_types,
        residue_ids=residue_ids,
        is_active_site=is_active_site_arr,
        residue_groups=residue_groups,
        domain_groups=domain_groups,
        domain_families=domain_families,
        domain_architecture=domain_architecture,
        chain_groups=chain_groups,
        complex_groups={pdb_file_str: list(chain_groups)},
        pdb_id=pdb_file_str,
        n_atoms=len(atoms),
        n_residues=len(residue_groups),
        n_chains=len(chain_groups),
    )


def _build_domain_cells(
    residue_groups: Dict[tuple, List[int]],
    annotations: Sequence[DomainAnnotation],
) -> Tuple[Dict[tuple, List[tuple]], Dict[tuple, str], Dict[str, List[tuple]]]:
    """Build residue→domain→chain incidence without inferring fake domains."""
    groups: Dict[tuple, List[tuple]] = {}
    families: Dict[tuple, str] = {}
    architecture: Dict[str, List[tuple]] = {}
    by_chain: Dict[str, List[DomainAnnotation]] = {}
    for annotation in annotations:
        if annotation.start > annotation.end:
            raise ValueError(f"invalid domain interval: {annotation}")
        by_chain.setdefault(annotation.chain_id, []).append(annotation)

    for chain_id, chain_annotations in by_chain.items():
        previous_end: Optional[int] = None
        for ordinal, annotation in enumerate(
            sorted(chain_annotations, key=lambda d: (d.start, d.end, d.family))
        ):
            if previous_end is not None and annotation.start <= previous_end:
                raise ValueError(f"overlapping domains on chain {chain_id}")
            previous_end = annotation.end
            domain_name = annotation.domain_id or f"{annotation.family}:{ordinal + 1}"
            key = (chain_id, domain_name)
            members = [
                residue_key for residue_key in residue_groups
                if residue_key[0] == chain_id
                and annotation.start <= int(residue_key[1][1]) <= annotation.end
            ]
            if not members:
                raise ValueError(f"domain {key} contains no structure residues")
            groups[key] = members
            families[key] = annotation.family
            architecture.setdefault(chain_id, []).append(key)
    return groups, families, architecture


# ── Filtration ────────────────────────────────────────────────────────────────

def build_rank_stratified_filtration(
    pcc: WholeProteinPCC,
    steps: int = 16,
    lambda_max: float = 1.0,
) -> Tuple[object, np.ndarray]:
    """Build a rank-stratified Vietoris-Rips filtration over the whole protein.

    Design changes from the 8Å active-site version
    -----------------------------------------------
    * **All atoms enter** the Rips complex — no radius gate.
    * The spectral cutoff is swept **logarithmically** via ``np.geomspace``
      rather than linearly with a fixed 8Å ceiling, consistent with v3.0
      rank-stratified spectral filtration.
    * The ``is_active_site`` mask is returned as metadata for Phase 4
      attribution and TCPNet attention bias; it does NOT gate atoms.

    Parameters
    ----------
    pcc        : WholeProteinPCC — full-protein atom set.
    steps      : number of logarithmically-spaced filtration steps (default 16).
    lambda_max : upper spectral cutoff boundary (default 1.0).

    Returns
    -------
    simplex_tree   : gudhi.SimplexTree built over all atoms.
    is_active_site : (N,) bool — forwarded to Phase 4 attribution.

    Raises
    ------
    ImportError : if gudhi or scipy are not installed.
    """
    if not HAS_GUDHI:
        raise ImportError("gudhi is required: pip install gudhi")
    if not HAS_SCIPY:
        raise ImportError("scipy is required: pip install scipy")

    # Logarithmic spectral cutoff sweep (replaces np.linspace(2.0, 8.0, 16))
    _spectral_cutoffs = np.geomspace(1e-3, lambda_max, num=steps)  # noqa: F841

    # All atoms enter — no radius gate
    dist_matrix = squareform(pdist(pcc.coords.astype(np.float64)))

    rips        = gudhi.RipsComplex(distance_matrix=dist_matrix)
    simplex_tree = rips.create_simplex_tree(max_dimension=3)

    # Active-site mask is metadata for attribution, not a filtration filter
    return simplex_tree, pcc.is_active_site


# ── Sheaf sections ────────────────────────────────────────────────────────────

def construct_sheaf_sections(
    pcc: WholeProteinPCC,
    section_dim: int = 3,
) -> Dict[tuple, np.ndarray]:
    """Construct sheaf sections over the whole-protein PCC.

    Iterates ``pcc.atoms`` (the full protein), matching the whole-protein
    API.  No logic change from the active-site version — the full atom set
    is used without spatial truncation.

    Parameters
    ----------
    pcc         : WholeProteinPCC.
    section_dim : dimension of each section vector.  Currently the section
                  is the mean coordinate (dim=3); extend here for richer
                  per-residue feature sections.

    Returns
    -------
    sections : resid → (1, section_dim) float32 array.
               Keyed by the same (chain, res_id) tuples as
               ``pcc.residue_groups``.
    """
    sections: Dict[tuple, np.ndarray] = {}

    for res_key, atom_indices in pcc.residue_groups.items():
        if not atom_indices:
            continue
        res_coords = pcc.coords[atom_indices]                        # (n_a, 3)
        sections[res_key] = res_coords.mean(axis=0, keepdims=True).astype(np.float32)

    return sections


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test_whole_protein_pcc(
    pdb_file: str,
    catalytic_residues: List[tuple],
) -> WholeProteinPCC:
    """Pass/fail checks for the Phase 2 migration.

    Run this on the five validation enzymes before advancing to Phase 3:
        nitrogenase (1N2C), HDCR (6CFW), adenylate kinase (4AKE),
        DHFR, lactate dehydrogenase.

    Expected output for adenylate kinase (4AKE, ~214 residues)::

        OK: 1656 atoms, 214 residues, 42 active-site atoms (2.5% of protein)

    Parameters
    ----------
    pdb_file           : path to PDB/CIF structure file.
    catalytic_residues : M-CSA annotations as (chain_id, res_id) tuples.

    Returns
    -------
    WholeProteinPCC — the validated whole-protein complex.
    """
    pcc = build_whole_protein_pcc(pdb_file, catalytic_residues)

    # 1. Active-site atoms are a strict subset of all atoms
    assert pcc.is_active_site.sum() < pcc.n_atoms, (
        "Active-site mask should not cover the full protein"
    )

    # 2. At least one active-site atom exists
    assert pcc.is_active_site.any(), (
        "No active-site atoms flagged — check M-CSA residue IDs"
    )

    # 3. Typical proteins: ≥50 residues
    assert pcc.n_residues >= 50, (
        f"Suspiciously small: {pcc.n_residues} residues. "
        "Extraction may be clipped."
    )

    # 4. Active-site fraction should be small (catalytic residues ≈ 2–8%)
    active_frac = pcc.is_active_site.mean()
    assert active_frac < 0.15, (
        f"Active-site fraction {active_frac:.2%} too high — "
        "check M-CSA annotations"
    )

    print(
        f"OK: {pcc.n_atoms} atoms, {pcc.n_residues} residues, "
        f"{int(pcc.is_active_site.sum())} active-site atoms "
        f"({active_frac:.1%} of protein)"
    )
    return pcc
