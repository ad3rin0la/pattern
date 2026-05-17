"""
Physicochemical feature computation — Ioffe-style descriptors on atoms.

Implements the eight-property descriptor scheme from Ioffe, Dobrotvorskii &
Belozerskikh (1983), adapted from transition-metal-oxide surfaces to enzyme
active-site atoms. Each atom receives elemental descriptors (VOIP,
d-electron count, electron affinity, electronegativity, ionic radius) and
each residue contributes bulk descriptors (van-der-Waals volume, SASA,
sidechain transfer enthalpy).

Features are attached as sheaf sections in Phase 2 (persistent sheaf
Laplacian construction). For Phase 1 they serve as flat feature vectors
for baseline models and as ground-truth for inverse-attribution validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tope.data.active_site import ActiveSite, AtomRecord
from tope.data.config import (
    ELEMENT_PROPERTIES,
    IOFFE_PROPERTY_KEYS,
    RESIDUE_PROPERTIES,
)

logger = logging.getLogger(__name__)

# Optional SASA computation via FreeSASA.
try:
    import freesasa

    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False


# ── Feature vector ────────────────────────────────────────────────────────────

@dataclass
class AtomFeatures:
    """Feature vector for a single atom in the active site."""

    atom_index: int
    element: str
    residue_name: str
    residue_id: str
    is_catalytic: bool
    properties: np.ndarray  # shape (8,)
    property_keys: Tuple[str, ...] = tuple(IOFFE_PROPERTY_KEYS)

    def __getitem__(self, key: str) -> float:
        idx = self.property_keys.index(key)
        return float(self.properties[idx])


@dataclass
class ActiveSiteFeatures:
    """Full feature matrix for an active-site environment."""

    pdb_id: str
    ec_number: str
    atom_features: List[AtomFeatures] = field(default_factory=list)
    coords: Optional[np.ndarray] = None           # (N, 3)
    feature_matrix: Optional[np.ndarray] = None   # (N, 8)
    catalytic_mask: Optional[np.ndarray] = None   # (N,) bool

    @property
    def n_atoms(self) -> int:
        return len(self.atom_features)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (coords, features, catalytic_mask) as numpy arrays."""
        assert self.coords is not None
        assert self.feature_matrix is not None
        assert self.catalytic_mask is not None
        return self.coords, self.feature_matrix, self.catalytic_mask


# ── Feature computer ──────────────────────────────────────────────────────────

class FeatureComputer:
    """Compute Ioffe-style physicochemical descriptors for active-site atoms."""

    def __init__(
        self,
        element_properties: Optional[Dict[str, Dict[str, float]]] = None,
        residue_properties: Optional[Dict[str, Dict[str, float]]] = None,
        compute_sasa: bool = True,
        normalise: bool = True,
    ):
        # Case-insensitive element lookup: keep both upper and title forms.
        base_elem = element_properties or ELEMENT_PROPERTIES
        self.element_props = {k.upper(): v for k, v in base_elem.items()}
        self.residue_props = residue_properties or RESIDUE_PROPERTIES
        self.compute_sasa = compute_sasa and HAS_FREESASA
        self.normalise = normalise

    def compute(self, active_site: ActiveSite) -> ActiveSiteFeatures:
        """Compute the 8-element Ioffe descriptor for every atom."""
        atom_feats: List[AtomFeatures] = []
        coords_list: List[np.ndarray] = []
        feat_rows: List[np.ndarray] = []
        cat_mask: List[bool] = []

        sasa_map = self._compute_sasa_map(active_site) if self.compute_sasa else {}

        for i, atom in enumerate(active_site.atoms):
            vec = self._atom_vector(atom, sasa_map)
            af = AtomFeatures(
                atom_index=i,
                element=atom.element,
                residue_name=atom.residue_name,
                residue_id=atom.residue_id,
                is_catalytic=atom.is_catalytic,
                properties=vec,
            )
            atom_feats.append(af)
            coords_list.append(atom.coord)
            feat_rows.append(vec)
            cat_mask.append(atom.is_catalytic)

        coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3))
        features = np.array(feat_rows, dtype=np.float64) if feat_rows else np.empty((0, 8))
        mask = np.array(cat_mask, dtype=bool)

        if self.normalise and features.size > 0:
            features = self._normalise_features(features)

        result = ActiveSiteFeatures(
            pdb_id=active_site.pdb_id,
            ec_number=active_site.ec_number,
            atom_features=atom_feats,
            coords=coords,
            feature_matrix=features,
            catalytic_mask=mask,
        )

        logger.debug(
            "%s: computed %d-dim features for %d atoms (%d catalytic)",
            active_site.pdb_id,
            features.shape[1] if features.ndim == 2 else 0,
            result.n_atoms,
            int(mask.sum()),
        )
        return result

    def compute_batch(
        self, active_sites: List[ActiveSite]
    ) -> List[ActiveSiteFeatures]:
        """Compute features for a batch of active sites."""
        results = []
        for i, site in enumerate(active_sites, 1):
            results.append(self.compute(site))
            if i % 100 == 0:
                logger.info("Features computed: %d / %d", i, len(active_sites))
        return results

    # ── Ioffe inverse-recognition helpers ────────────────────────────────

    @staticmethod
    def criterion_influence(
        features: ActiveSiteFeatures,
        criterion_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (full, ablated) feature matrices with one criterion zeroed."""
        assert features.feature_matrix is not None
        full = features.feature_matrix.copy()
        ablated = full.copy()
        ablated[:, criterion_index] = 0.0
        return full, ablated

    @staticmethod
    def pairwise_criterion_correlation(
        features: ActiveSiteFeatures,
    ) -> np.ndarray:
        """(8, 8) Pearson correlation matrix across the Ioffe properties."""
        assert features.feature_matrix is not None
        fm = features.feature_matrix
        if fm.shape[0] < 2:
            return np.eye(fm.shape[1])
        return np.corrcoef(fm.T)

    # ── Private ──────────────────────────────────────────────────────────

    def _atom_vector(
        self,
        atom: AtomRecord,
        sasa_map: Dict[str, float],
    ) -> np.ndarray:
        """Build the 8-element property vector for one atom."""
        elem = atom.element.upper()
        elem_data = self.element_props.get(elem, {})
        res_data = self.residue_props.get(atom.residue_name.upper(), {})

        voip = elem_data.get("voip", 0.0)
        n_d = elem_data.get("n_d_electrons", 0.0)
        ea = elem_data.get("electron_affinity", 0.0)
        en = elem_data.get("electronegativity", 0.0)
        ir = elem_data.get("ionic_radius", 0.0)
        rv = res_data.get("residue_volume", 0.0)
        sasa = sasa_map.get(atom.residue_id, 0.0)
        seh = res_data.get("sidechain_enthalpy", 0.0)

        return np.array([voip, n_d, ea, en, ir, rv, sasa, seh], dtype=np.float64)

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalisation per property (column)."""
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-12] = 1.0
        return (features - mean) / std

    def _compute_sasa_map(self, active_site: ActiveSite) -> Dict[str, float]:
        """Compute per-residue SASA using FreeSASA (if available)."""
        if not HAS_FREESASA:
            return {}

        try:
            struct = freesasa.Structure()
            for atom in active_site.atoms:
                struct.addAtom(
                    atom.name,
                    atom.residue_name,
                    str(atom.residue_number),
                    atom.chain_id,
                    float(atom.coord[0]),
                    float(atom.coord[1]),
                    float(atom.coord[2]),
                )

            result = freesasa.calc(struct)
            sasa_map: Dict[str, float] = {}
            for chain_key, residues in result.residueAreas().items():
                for resnum_str, area in residues.items():
                    sasa_map[f"{chain_key}:{resnum_str}"] = area.total
            return sasa_map
        except Exception as exc:
            logger.debug("FreeSASA computation failed: %s", exc)
            return {}
