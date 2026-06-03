"""On-disk → model-input loader: closes the curate→train loop.

``DatasetBuilder`` (dataset.py) writes one directory of NumPy arrays per
active-site sample — atom coords, Ioffe features, catalytic mask, radius-
filtration adjacencies, and the atom→residue incidence — indexed by a JSON
manifest.  Nothing previously reconstructed the model's ``enzyme_pcc`` input
dict from those arrays; the trainer just consumed ``batch["enzyme_pcc"]`` from
an unspecified source.

``ToPEDataset`` is that missing bridge.  Each item rebuilds an ``enzyme_pcc``:

  * node_features  ← features.npy            (rank-0 atom descriptors)
  * pos            ← coords.npy              (rank-0 positions)
  * edge_index     ← radius graph over pos   (rank-1 bonds, derived from coords)
  * edge_features  ← Gaussian RBF of bond length
  * atom_residue   ← atom_residue.npy        (rank-0 → rank-2 incidence; the
                     curated membership that drives B_12 in EnzymeTCPNet)

``collate_enzyme_pcc`` batches several graphs into one disjoint-union complex,
offsetting node, edge and residue indices and emitting the ``batch`` membership
vector the encoder pools over.

Label encoding for the multi-task heads (EC-class vocabulary, selectivity) is
deliberately left to the caller; the raw labels (kinetics floats, EC strings)
are carried through on each sample so a downstream encoder can attach them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a core dep in practice
    HAS_TORCH = False
    Dataset = object  # type: ignore

from tope.data.molecule import MoleculeFeaturizer, smiles_to_graph


# ══════════════════════════════════════════════════════════════════════════════
# Edge construction / featurisation
# ══════════════════════════════════════════════════════════════════════════════

def radius_graph_from_coords(
    pos: np.ndarray,
    radius: float,
    max_neighbors: Optional[int] = None,
) -> np.ndarray:
    """Build the rank-1 bond graph from atom positions.

    A directed edge ``(i, j)`` exists when ``0 < ‖pos_i − pos_j‖ ≤ radius``.
    Edges are emitted in both directions (the complex is undirected); an
    optional ``max_neighbors`` keeps only the nearest ``k`` per source atom.

    Parameters
    ----------
    pos : (N, 3) atom coordinates.
    radius : float distance cutoff in Å.
    max_neighbors : optional cap on out-degree.

    Returns
    -------
    edge_index : (2, E) int array [src, dst].
    """
    n = pos.shape[0]
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    adj = dist <= radius

    if max_neighbors is not None and max_neighbors < n:
        # Keep the nearest max_neighbors per row.
        keep = np.argsort(dist, axis=1)[:, :max_neighbors]
        knn = np.zeros_like(adj)
        rows = np.arange(n)[:, None]
        knn[rows, keep] = True
        adj = adj & knn

    src, dst = np.nonzero(adj)
    return np.stack([src, dst]).astype(np.int64)


def gaussian_rbf(dist: np.ndarray, n_basis: int, cutoff: float) -> np.ndarray:
    """Expand scalar distances in ``n_basis`` Gaussian radial bases on [0, cutoff].

    Returns (E, n_basis).  Used to featurise each bond by its length so that
    ``edge_features`` matches the encoder's ``edge_feat_dim``.
    """
    centers = np.linspace(0.0, cutoff, n_basis, dtype=np.float64)
    if n_basis > 1:
        width = cutoff / (n_basis - 1)
    else:
        width = cutoff if cutoff > 0 else 1.0
    gamma = 1.0 / (2.0 * width * width + 1e-12)
    d = dist[:, None]
    return np.exp(-gamma * (d - centers[None, :]) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ToPEDataset(Dataset):
    """Map curated on-disk samples to model-ready ``enzyme_pcc`` graphs.

    Parameters
    ----------
    records : list of record dicts (from the dataset_index.json manifest), or a
        list of DatasetRecord instances.  Each must carry the per-sample array
        paths (``coords_path``, ``features_path``, ``atom_residue_path`` …)
        relative to ``features_dir``, plus the labels.
    features_dir : directory the relative array paths resolve against.
    edge_radius : Å cutoff for the rank-1 bond graph derived from coords.
    edge_feat_dim : number of Gaussian RBF bins for ``edge_features``.
    max_neighbors : optional out-degree cap for the bond graph.
    split : optional filter ("train"/"val"/"test") applied to the records.
    with_molecules : also emit ``substrate``/``product`` graphs (featurised from
        the persisted SMILES) so the cross-attention + kinetics/selectivity
        heads have inputs.  Missing/invalid SMILES degrade to a dummy atom.
    mol_featurizer : molecule featuriser (defaults to MoleculeFeaturizer).
    """

    def __init__(
        self,
        records: Sequence[Union[Dict[str, Any], Any]],
        features_dir: Union[str, Path],
        edge_radius: float = 5.0,
        edge_feat_dim: int = 32,
        max_neighbors: Optional[int] = None,
        split: Optional[str] = None,
        with_molecules: bool = False,
        mol_featurizer: Optional[MoleculeFeaturizer] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("ToPEDataset requires torch.")
        self.features_dir = Path(features_dir)
        self.edge_radius = edge_radius
        self.edge_feat_dim = edge_feat_dim
        self.max_neighbors = max_neighbors
        self.with_molecules = with_molecules
        self.mol_featurizer = mol_featurizer or MoleculeFeaturizer()

        recs = [r if isinstance(r, dict) else _record_to_dict(r) for r in records]
        if split is not None:
            recs = [r for r in recs if r.get("split") == split]
        self.records: List[Dict[str, Any]] = recs

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        features_dir: Union[str, Path],
        **kwargs: Any,
    ) -> "ToPEDataset":
        """Construct from a ``dataset_index.json`` manifest written by DatasetBuilder."""
        records = json.loads(Path(index_path).read_text())
        return cls(records, features_dir, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rel: str) -> np.ndarray:
        return np.load(self.features_dir / rel)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.records[i]
        pos = self._load(r["coords_path"]).astype(np.float64)
        feats = self._load(r["features_path"]).astype(np.float64)

        # Atom→residue incidence (optional for backward compatibility).
        ar_path = r.get("atom_residue_path")
        if ar_path:
            atom_residue = self._load(ar_path).astype(np.int64)
        else:
            atom_residue = np.full(pos.shape[0], -1, dtype=np.int64)

        edge_index = radius_graph_from_coords(pos, self.edge_radius, self.max_neighbors)
        if edge_index.shape[1] > 0:
            d = np.linalg.norm(pos[edge_index[0]] - pos[edge_index[1]], axis=1)
            edge_features = gaussian_rbf(d, self.edge_feat_dim, self.edge_radius)
        else:
            edge_features = np.zeros((0, self.edge_feat_dim), dtype=np.float64)

        n = pos.shape[0]
        n_residues = int(r.get("n_residues") or (int(atom_residue.max()) + 1 if atom_residue.size and atom_residue.max() >= 0 else 0))

        enzyme_pcc = {
            "node_features": torch.tensor(feats, dtype=torch.float32),
            "pos": torch.tensor(pos, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "edge_features": torch.tensor(edge_features, dtype=torch.float32),
            "atom_residue": torch.tensor(atom_residue, dtype=torch.long),
            "batch": torch.zeros(n, dtype=torch.long),
            "n_residues": n_residues,
        }
        sample = {
            "enzyme_pcc": enzyme_pcc,
            "pdb_id": r.get("pdb_id", ""),
            "labels": _labels_from_record(r),
        }
        if self.with_molecules:
            sample["substrate"] = self._mol_graph(r.get("substrate_smiles", ""))
            sample["product"] = self._mol_graph(r.get("product_smiles", ""))
        return sample

    def _mol_graph(self, smiles: str) -> Dict[str, "torch.Tensor"]:
        """Featurise a SMILES string into a substrate/product graph dict."""
        feats, edge_index = smiles_to_graph(smiles, self.mol_featurizer)
        m = feats.shape[0]
        return {
            "node_features": torch.tensor(feats, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "batch": torch.zeros(m, dtype=torch.long),
        }

    @property
    def node_feat_dim(self) -> int:
        """Feature width of the rank-0 node descriptors (from the first sample)."""
        if not self.records:
            return 0
        return int(self._load(self.records[0]["features_path"]).shape[1])

    @property
    def mol_feat_dim(self) -> int:
        """Feature width of the substrate/product node features."""
        return self.mol_featurizer.feat_dim


def _labels_from_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """Carry raw labels through; head-specific encoding is the caller's job."""
    def _f(v: Any) -> float:
        return float(v) if v is not None else float("nan")
    return {
        "ec_number": r.get("ec_number", ""),
        "ec_top_level": r.get("ec_top_level", ""),
        "kinetics": [_f(r.get("log_kcat")), _f(r.get("log_km")), _f(r.get("log_kcat_km"))],
    }


def _record_to_dict(r: Any) -> Dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    if is_dataclass(r):
        return asdict(r)
    raise TypeError(f"Cannot convert record of type {type(r)} to dict")


# ══════════════════════════════════════════════════════════════════════════════
# Collation — disjoint union of graphs
# ══════════════════════════════════════════════════════════════════════════════

def collate_enzyme_pcc(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch samples into one disjoint-union complex.

    Node, edge and residue indices are offset by running cumulative counts so
    the per-graph cells stay disjoint; ``batch`` records graph membership so the
    encoder can pool per active site.  ``atom_residue`` residue indices are
    offset by cumulative residue counts (absent entries, ``-1``, are preserved).
    """
    node_feats, poss, edge_feats = [], [], []
    edge_indices, atom_residues, batch_vec = [], [], []
    pdb_ids, kinetics, ec_numbers, ec_tops = [], [], [], []

    node_offset = 0
    res_offset = 0
    for g, s in enumerate(samples):
        pcc = s["enzyme_pcc"]
        n = pcc["node_features"].size(0)

        node_feats.append(pcc["node_features"])
        poss.append(pcc["pos"])
        edge_feats.append(pcc["edge_features"])
        edge_indices.append(pcc["edge_index"] + node_offset)
        batch_vec.append(torch.full((n,), g, dtype=torch.long))

        ar = pcc["atom_residue"].clone()
        valid = ar >= 0
        ar[valid] = ar[valid] + res_offset
        atom_residues.append(ar)

        node_offset += n
        res_offset += int(pcc.get("n_residues", 0))

        lab = s.get("labels", {})
        pdb_ids.append(s.get("pdb_id", ""))
        kinetics.append(lab.get("kinetics", [float("nan")] * 3))
        ec_numbers.append(lab.get("ec_number", ""))
        ec_tops.append(lab.get("ec_top_level", ""))

    enzyme_pcc = {
        "node_features": torch.cat(node_feats, dim=0),
        "pos": torch.cat(poss, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long),
        "edge_features": torch.cat(edge_feats, dim=0),
        "atom_residue": torch.cat(atom_residues, dim=0),
        "batch": torch.cat(batch_vec, dim=0),
        "n_residues": res_offset,
    }
    batch = {
        "enzyme_pcc": enzyme_pcc,
        "pdb_id": pdb_ids,
        "labels": {
            "ec_number": ec_numbers,
            "ec_top_level": ec_tops,
            "kinetics": torch.tensor(kinetics, dtype=torch.float32),
        },
    }
    # Substrate / product molecule graphs (present iff the dataset emitted them).
    if samples and "substrate" in samples[0]:
        batch["substrate"] = _collate_mol_graphs([s["substrate"] for s in samples])
        batch["product"] = _collate_mol_graphs([s["product"] for s in samples])
    return batch


def _collate_mol_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Disjoint-union batching for substrate/product molecule graphs."""
    node_feats, edge_indices, batch_vec = [], [], []
    offset = 0
    for g, graph in enumerate(graphs):
        m = graph["node_features"].size(0)
        node_feats.append(graph["node_features"])
        edge_indices.append(graph["edge_index"] + offset)
        batch_vec.append(torch.full((m,), g, dtype=torch.long))
        offset += m
    return {
        "node_features": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long),
        "batch": torch.cat(batch_vec, dim=0),
    }
