"""
Physicochemical feature computation — Ioffe-style descriptors on residues.

Implements the eight-property descriptor scheme from Ioffe, Dobrotvorskii &
Belozerskikh (1983), adapted from transition-metal-oxide surfaces to enzyme
active-site residues. Each residue receives elemental descriptors derived from
a representative heavy atom (VOIP, d-electron count, electron affinity,
electronegativity, ionic radius) plus residue-level bulk descriptors
(van-der-Waals volume, SASA, sidechain transfer enthalpy).

Note: the upstream `ActiveSite` was refactored to residue-level extraction, so
features are now computed per residue rather than per atom. Output shapes
(coords, feature_matrix, catalytic_mask) are preserved for downstream callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tope.data.active_site import ActiveSite, ResidueRecord
from tope.data.config import (
    ELEMENT_PROPERTIES,
    IOFFE_PROPERTY_KEYS,
    RESIDUE_PROPERTIES,
)

logger = logging.getLogger(__name__)


# Representative heavy-atom element per amino acid. Used to source the five
# elemental Ioffe descriptors from ELEMENT_PROPERTIES when only residue-level
# information is available.
_REPRESENTATIVE_ELEMENT: Dict[str, str] = {
    "ALA": "C", "ARG": "N", "ASN": "N", "ASP": "O", "CYS": "S",
    "GLN": "N", "GLU": "O", "GLY": "C", "HIS": "N", "ILE": "C",
    "LEU": "C", "LYS": "N", "MET": "S", "PHE": "C", "PRO": "C",
    "SER": "O", "THR": "O", "TRP": "N", "TYR": "O", "VAL": "C",
}


# ── Feature vector ────────────────────────────────────────────────────────────

@dataclass
class ResidueFeatures:
    """Feature vector for a single residue in the active site."""

    residue_index: int
    residue_name: str
    residue_id: str
    is_catalytic: bool
    properties: np.ndarray  # shape (8,)
    property_keys: Tuple[str, ...] = tuple(IOFFE_PROPERTY_KEYS)

    def __getitem__(self, key: str) -> float:
        idx = self.property_keys.index(key)
        return float(self.properties[idx])


# Backwards-compatible alias for code that still references the old name.
AtomFeatures = ResidueFeatures


@dataclass
class ActiveSiteFeatures:
    """Full feature matrix for an active-site environment."""

    pdb_id: str
    ec_number: str
    residue_features: List[ResidueFeatures] = field(default_factory=list)
    coords: Optional[np.ndarray] = None           # (N, 3)
    feature_matrix: Optional[np.ndarray] = None   # (N, 8)
    catalytic_mask: Optional[np.ndarray] = None   # (N,) bool

    @property
    def n_residues(self) -> int:
        return len(self.residue_features)

    @property
    def n_atoms(self) -> int:  # legacy alias
        return self.n_residues

    @property
    def atom_features(self) -> List[ResidueFeatures]:  # legacy alias
        return self.residue_features

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (coords, features, catalytic_mask) as numpy arrays."""
        assert self.coords is not None
        assert self.feature_matrix is not None
        assert self.catalytic_mask is not None
        return self.coords, self.feature_matrix, self.catalytic_mask


# ── Feature computer ──────────────────────────────────────────────────────────

class FeatureComputer:
    """Compute Ioffe-style physicochemical descriptors for active-site residues."""

    def __init__(
        self,
        element_properties: Optional[Dict[str, Dict[str, float]]] = None,
        residue_properties: Optional[Dict[str, Dict[str, float]]] = None,
        compute_sasa: bool = True,
        normalise: bool = True,
    ):
        self.element_props = element_properties or ELEMENT_PROPERTIES
        self.residue_props = residue_properties or RESIDUE_PROPERTIES
        # SASA computation requires per-atom records; not available at the
        # residue-level abstraction. Kept as a flag for API compatibility.
        self.compute_sasa = compute_sasa
        self.normalise = normalise

    def compute(self, active_site: ActiveSite) -> ActiveSiteFeatures:
        """Compute the 8-element Ioffe descriptor for every residue."""
        res_feats: List[ResidueFeatures] = []
        coords_list: List[np.ndarray] = []
        feat_rows: List[np.ndarray] = []
        cat_mask: List[bool] = []

        for i, residue in enumerate(active_site.residues):
            vec = self._residue_vector(residue)
            rf = ResidueFeatures(
                residue_index=i,
                residue_name=residue.residue_name,
                residue_id=residue.residue_id,
                is_catalytic=residue.is_catalytic,
                properties=vec,
            )
            res_feats.append(rf)
            coords_list.append(residue.ca_coord)
            feat_rows.append(vec)
            cat_mask.append(residue.is_catalytic)

        coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3))
        features = np.array(feat_rows, dtype=np.float64) if feat_rows else np.empty((0, 8))
        mask = np.array(cat_mask, dtype=bool)

        if self.normalise and features.size > 0:
            features = self._normalise_features(features)

        result = ActiveSiteFeatures(
            pdb_id=active_site.pdb_id,
            ec_number=active_site.ec_number,
            residue_features=res_feats,
            coords=coords,
            feature_matrix=features,
            catalytic_mask=mask,
        )

        logger.debug(
            "%s: computed %d-dim features for %d residues (%d catalytic)",
            active_site.pdb_id,
            features.shape[1] if features.ndim == 2 else 0,
            result.n_residues,
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

    def _residue_vector(self, residue: ResidueRecord) -> np.ndarray:
        """Build the 8-element property vector for one residue."""
        res_name = residue.residue_name.upper()
        elem = _REPRESENTATIVE_ELEMENT.get(res_name, "C")
        elem_data = self.element_props.get(elem, {})
        res_data = self.residue_props.get(res_name, {})

        voip = elem_data.get("voip", 0.0)
        n_d = elem_data.get("n_d_electrons", 0.0)
        ea = elem_data.get("electron_affinity", 0.0)
        en = elem_data.get("electronegativity", 0.0)
        ir = elem_data.get("ionic_radius", 0.0)
        rv = res_data.get("residue_volume", 0.0)
        sasa = 0.0  # not available at residue-level abstraction
        seh = res_data.get("sidechain_enthalpy", 0.0)

        return np.array([voip, n_d, ea, en, ir, rv, sasa, seh], dtype=np.float64)

    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalisation per property (column)."""
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-12] = 1.0
        return (features - mean) / std
