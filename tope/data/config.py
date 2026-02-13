"""
Configuration and constants for the ToPE data curation pipeline.

Physicochemical descriptors follow Ioffe, Dobrotvorskii & Belozerskikh (1983)
Table 3 — the eight criterion properties found influential for CO oxidation,
generalised here to enzyme catalytic residues.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ── Paths ────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(os.environ.get("TOPE_DATA_ROOT", "data"))
RAW_DIR = DATA_ROOT / "raw"
PDB_DIR = RAW_DIR / "pdb"
MCSA_DIR = RAW_DIR / "mcsa"
KINETICS_DIR = RAW_DIR / "kinetics"
PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_DIR = DATA_ROOT / "features"


# ── RCSB PDB API ─────────────────────────────────────────────────────────────

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


# ── M-CSA (Mechanism and Catalytic Site Atlas) ────────────────────────────────

MCSA_ENTRIES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/"
MCSA_RESIDUES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/"
MCSA_CSV_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/curated_data.csv"


# ── BRENDA & SABIO-RK (Enzyme Kinetics) ───────────────────────────────────────
#
# BRENDA: ~23k kcat entries, ~41k Km entries across all EC classes.
# SABIO-RK: structured reaction kinetics with explicit conditions.
# Together they provide the quantitative activity labels needed for
# the kinetics regression heads (log kcat, log Km, log kcat/Km).

BRENDA_DOWNLOAD_URL = "https://www.brenda-enzymes.org/download.php"
SABIO_RK_API_URL = "http://sabiork.h-its.org/sabioRestWebServices"
SABIO_RK_SEARCH_URL = f"{SABIO_RK_API_URL}/searchKineticLaws/sbml"
SABIO_RK_ENTRY_URL = f"{SABIO_RK_API_URL}/kineticLaws"

# Kinetics parameter types to extract
KINETICS_PARAM_TYPES = ["kcat", "Km", "kcat/Km", "Ki", "Vmax"]

# Expected dataset sizes (approximate, for progress reporting)
EXPECTED_BRENDA_KCAT = 23_000
EXPECTED_BRENDA_KM = 41_000


# ── Performance Targets ───────────────────────────────────────────────────────
#
# Baselines and ToPE targets from the roadmap:
#
# EC classification:
#   TopEC:      F = 0.72  (800+ EC classes)
#   GraphEC:    F = 0.68
#   ToPE target: F > 0.75
#
# Kinetics prediction:
#   CataPro:    R² = 0.67 (kcat), R² = 0.73 (Km)
#   DeepEnzyme: R² = 0.42 (kcat, <50% seq identity)
#   ToPE target: R² > 0.50 (kcat, <40% seq identity)
#
# Selectivity:
#   ToPE target: MAE < 12%

PERFORMANCE_TARGETS = {
    "ec_f_score": 0.75,
    "selectivity_mae": 0.12,
    "kcat_r2_ood": 0.50,         # out-of-distribution (<40% seq identity)
    "kcat_r2_ood_threshold": 0.40,  # sequence identity cutoff
}

BASELINE_PERFORMANCE = {
    "TopEC": {"ec_f_score": 0.72},
    "GraphEC": {"ec_f_score": 0.68},
    "CataPro": {"kcat_r2": 0.67, "km_r2": 0.73},
    "DeepEnzyme": {"kcat_r2_lt50": 0.42},
    "GraphKcat": {},
    "KcatNet": {},
    "CatPred": {},
}


# ── Active-Site Extraction ────────────────────────────────────────────────────

DEFAULT_ACTIVE_SITE_RADIUS = 8.0   # Å — matches filtration ε = 8Å shell
MIN_ACTIVE_SITE_RADIUS = 4.0       # Å — at least H-bond network
CATALYTIC_RESIDUE_PADDING = 2.0    # Å — extra buffer around annotated residues


# ── Ioffe-Style Physicochemical Descriptors ───────────────────────────────────
#
# The original eight influential properties for transition-metal-oxide catalysis:
#   1. Valence-orbital ionisation potential (VOIP)
#   2. Number of d-electrons (Nd)
#   3. Electron affinity (EA)
#   4. Pauling electronegativity (χ)
#   5. Metal ionic radius (r_ion)
#   6. Madelung constant proxy (lattice energy parameter)
#   7. Ratio of crystallographic hole volume to atom volume (V_hole / V_atom)
#   8. Heat of formation of highest oxide (ΔHf)
#
# For enzyme residues we adapt these to amino-acid-level descriptors.
# Items 6–8 become residue-volume, solvent-accessible-surface, and
# sidechain formation-enthalpy proxies.

IOFFE_PROPERTY_KEYS = [
    "voip",                  # Valence-orbital ionisation potential (eV)
    "n_d_electrons",         # d-electron count (0 for light atoms)
    "electron_affinity",     # Electron affinity (eV)
    "electronegativity",     # Pauling electronegativity
    "ionic_radius",          # Shannon ionic radius (Å)
    "residue_volume",        # van-der-Waals volume of residue (Å³)
    "sasa",                  # Solvent-accessible surface area (Å²)
    "sidechain_enthalpy",    # Sidechain transfer free energy (kcal/mol)
]

# Elemental data for atoms commonly found in enzyme active sites.
# Sources: CRC Handbook, NIST, Shannon (1976).

ELEMENT_PROPERTIES: Dict[str, Dict[str, float]] = {
    "H":  {"voip": 13.60, "n_d_electrons": 0, "electron_affinity": 0.75,
            "electronegativity": 2.20, "ionic_radius": 0.25},
    "C":  {"voip": 11.26, "n_d_electrons": 0, "electron_affinity": 1.26,
            "electronegativity": 2.55, "ionic_radius": 0.77},
    "N":  {"voip": 14.53, "n_d_electrons": 0, "electron_affinity": -0.07,
            "electronegativity": 3.04, "ionic_radius": 0.75},
    "O":  {"voip": 13.62, "n_d_electrons": 0, "electron_affinity": 1.46,
            "electronegativity": 3.44, "ionic_radius": 0.73},
    "S":  {"voip": 10.36, "n_d_electrons": 0, "electron_affinity": 2.08,
            "electronegativity": 2.58, "ionic_radius": 1.02},
    "P":  {"voip": 10.49, "n_d_electrons": 0, "electron_affinity": 0.75,
            "electronegativity": 2.19, "ionic_radius": 1.06},
    "Se": {"voip": 9.75,  "n_d_electrons": 0, "electron_affinity": 2.02,
            "electronegativity": 2.55, "ionic_radius": 1.16},
    # Transition metals found in metalloenzymes
    "Fe": {"voip": 7.90,  "n_d_electrons": 6, "electron_affinity": 0.15,
            "electronegativity": 1.83, "ionic_radius": 0.65},
    "Zn": {"voip": 9.39,  "n_d_electrons": 10, "electron_affinity": 0.00,
            "electronegativity": 1.65, "ionic_radius": 0.74},
    "Cu": {"voip": 7.73,  "n_d_electrons": 10, "electron_affinity": 1.24,
            "electronegativity": 1.90, "ionic_radius": 0.73},
    "Mn": {"voip": 7.43,  "n_d_electrons": 5, "electron_affinity": 0.00,
            "electronegativity": 1.55, "ionic_radius": 0.67},
    "Mg": {"voip": 7.65,  "n_d_electrons": 0, "electron_affinity": 0.00,
            "electronegativity": 1.31, "ionic_radius": 0.72},
    "Ca": {"voip": 6.11,  "n_d_electrons": 0, "electron_affinity": 0.02,
            "electronegativity": 1.00, "ionic_radius": 1.00},
    "Co": {"voip": 7.88,  "n_d_electrons": 7, "electron_affinity": 0.66,
            "electronegativity": 1.88, "ionic_radius": 0.65},
    "Ni": {"voip": 7.64,  "n_d_electrons": 8, "electron_affinity": 1.16,
            "electronegativity": 1.91, "ionic_radius": 0.69},
    "Mo": {"voip": 7.09,  "n_d_electrons": 5, "electron_affinity": 0.75,
            "electronegativity": 2.16, "ionic_radius": 0.69},
    "W":  {"voip": 7.86,  "n_d_electrons": 4, "electron_affinity": 0.82,
            "electronegativity": 2.36, "ionic_radius": 0.66},
}

# Residue-level properties (amino acids).
# residue_volume: Å³ (Zamyatnin 1972), sidechain_enthalpy: kcal/mol (Wolfenden 1981)

RESIDUE_PROPERTIES: Dict[str, Dict[str, float]] = {
    "ALA": {"residue_volume":  88.6, "sidechain_enthalpy":  1.94},
    "ARG": {"residue_volume": 173.4, "sidechain_enthalpy": -19.92},
    "ASN": {"residue_volume": 114.1, "sidechain_enthalpy": -9.68},
    "ASP": {"residue_volume": 111.1, "sidechain_enthalpy": -10.95},
    "CYS": {"residue_volume": 108.5, "sidechain_enthalpy": -1.24},
    "GLN": {"residue_volume": 143.8, "sidechain_enthalpy": -9.38},
    "GLU": {"residue_volume": 138.4, "sidechain_enthalpy": -10.20},
    "GLY": {"residue_volume":  60.1, "sidechain_enthalpy":  2.39},
    "HIS": {"residue_volume": 153.2, "sidechain_enthalpy": -10.27},
    "ILE": {"residue_volume": 166.7, "sidechain_enthalpy":  2.15},
    "LEU": {"residue_volume": 166.7, "sidechain_enthalpy":  2.28},
    "LYS": {"residue_volume": 168.6, "sidechain_enthalpy": -9.52},
    "MET": {"residue_volume": 162.9, "sidechain_enthalpy": -1.48},
    "PHE": {"residue_volume": 189.9, "sidechain_enthalpy": -0.76},
    "PRO": {"residue_volume": 112.7, "sidechain_enthalpy":  0.0},
    "SER": {"residue_volume":  89.0, "sidechain_enthalpy": -5.06},
    "THR": {"residue_volume": 116.1, "sidechain_enthalpy": -4.88},
    "TRP": {"residue_volume": 227.8, "sidechain_enthalpy": -5.88},
    "TYR": {"residue_volume": 193.6, "sidechain_enthalpy": -6.11},
    "VAL": {"residue_volume": 140.0, "sidechain_enthalpy":  1.99},
}


# ── Filtration Radii (Å) — mirrors the pipeline visual in the UI ─────────────

FILTRATION_RADII = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


# ── EC Classification ─────────────────────────────────────────────────────────

EC_TOP_LEVEL = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


# ── Pipeline defaults ─────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for a full curation run."""

    # Target dataset size
    min_structures: int = 5000

    # RCSB query filters
    max_resolution: float = 3.0          # Å — X-ray resolution cutoff
    min_chain_length: int = 50           # residues
    experimental_methods: List[str] = field(
        default_factory=lambda: ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    )

    # Active-site extraction
    active_site_radius: float = DEFAULT_ACTIVE_SITE_RADIUS
    include_heteroatoms: bool = True     # cofactors, metal ions
    include_water: bool = False

    # Feature computation
    compute_sasa: bool = True
    ioffe_properties: List[str] = field(
        default_factory=lambda: list(IOFFE_PROPERTY_KEYS)
    )
    filtration_radii: List[float] = field(
        default_factory=lambda: list(FILTRATION_RADII)
    )

    # Kinetics data
    fetch_kinetics: bool = True          # integrate BRENDA/SABIO-RK
    brenda_flat_file: str = ""           # path to BRENDA flat file (if pre-downloaded)
    kinetics_params: List[str] = field(
        default_factory=lambda: ["kcat", "Km", "kcat/Km"]
    )
    min_kinetics_entries: int = 1000     # skip EC classes with fewer entries

    # Parallelism
    n_workers: int = 4
    batch_size: int = 50
    request_delay: float = 0.25          # seconds between API calls

    # Output
    output_format: str = "parquet"       # parquet | csv | hdf5
    data_root: Path = DATA_ROOT
