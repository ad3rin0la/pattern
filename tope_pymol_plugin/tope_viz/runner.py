"""
tope_viz.runner
================
Bridges PyMOL to ToPE inference.

Two modes
---------
1. **Live inference** — imports ToPE modules directly (requires ToPE in the
   same Python environment as PyMOL, e.g. via ``pymol -r`` with a conda env
   that has tope installed).

2. **Load mode** — reads a pre-computed ToPE output file (JSON or NPZ) that
   was produced by ``tope predict --export`` or the RL substrate-retrieval
   pipeline.

Output schema (JSON / dict)
---------------------------
All four layers are optional; a file may contain any subset::

    {
      "pdb_id": "4AKE",
      "chain": "A",
      "n_residues": 214,

      "voip": {
        "residue_scores": [float, ...],          # length n_residues
        "raw_voip": [float, ...],                # in eV, before normalisation
        "thresholds": {"high": 15.0, "low": 10.0}
      },

      "attention": {
        "residue_scores": [float, ...],          # max attention weight per residue
        "substrate_smiles": "C1=CC=CC=C1",
        "per_head": [[float, ...], ...],         # n_heads × n_residues (optional)
        "substrate_atoms": [str, ...]
      },

      "frustration": {
        "residue_scores": [float, ...],          # FrustIndex per residue
        "holonomy_norm": float,                  # |d_AI(Hol(γ), Id)| global
        "frustrated_residues": [int, ...]        # 0-based indices, top-20%
      },

      "attribution": {
        "residue_scores": [float, ...],          # integrated-gradient saliency
        "zone_importance": {"zone1": f, "zone2": f, "zone3": f},
        "filtration_importance": {float: float, ...},
        "pathways": [[int, ...], ...],           # allosteric pathway residue lists
        "catalytic_residues": [int, ...],        # from M-CSA (if available)
        "task": "kinetics",
        "mcsa_overlap": float
      },

      "kinetics": {
        "log_kcat": float,
        "log_km": float,
        "log_kcat_km": float
      }
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public: load pre-computed output
# ---------------------------------------------------------------------------

def load_tope_output(path: str) -> Dict[str, Any]:
    """
    Load a ToPE output file (JSON or NPZ) and return the canonical dict.

    Parameters
    ----------
    path:
        Path to a ``.json`` or ``.npz`` file produced by ToPE.

    Returns
    -------
    dict following the schema documented in this module's docstring.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ToPE output not found: {path}")

    if p.suffix == ".json":
        with open(p) as fh:
            data = json.load(fh)
        return _validate_schema(data)

    if p.suffix == ".npz":
        return _load_npz(p)

    raise ValueError(f"Unsupported ToPE output format: {p.suffix}  (expected .json or .npz)")


def _load_npz(p: Path) -> Dict[str, Any]:
    """Convert NPZ arrays to the canonical JSON-equivalent dict."""
    arrays = np.load(p, allow_pickle=True)
    data: Dict[str, Any] = {}

    if "metadata" in arrays:
        meta = arrays["metadata"].item()  # stored as 0-d object array
        data.update(meta)

    for layer in ("voip", "attention", "frustration", "attribution"):
        key = f"{layer}_residue_scores"
        if key in arrays:
            data.setdefault(layer, {})["residue_scores"] = arrays[key].tolist()

    # Per-layer extras
    if "frustration_holonomy_norm" in arrays:
        data.setdefault("frustration", {})["holonomy_norm"] = float(arrays["frustration_holonomy_norm"])
    if "frustration_frustrated_residues" in arrays:
        data.setdefault("frustration", {})["frustrated_residues"] = arrays["frustration_frustrated_residues"].tolist()
    if "attribution_zone_importance" in arrays:
        data.setdefault("attribution", {})["zone_importance"] = arrays["attribution_zone_importance"].item()
    if "attribution_pathways" in arrays:
        data.setdefault("attribution", {})["pathways"] = arrays["attribution_pathways"].tolist()
    if "attribution_catalytic_residues" in arrays:
        data.setdefault("attribution", {})["catalytic_residues"] = arrays["attribution_catalytic_residues"].tolist()
    if "attention_per_head" in arrays:
        data.setdefault("attention", {})["per_head"] = arrays["attention_per_head"].tolist()

    return _validate_schema(data)


def _validate_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise types (lists → lists, etc.) and add missing keys."""
    for layer in ("voip", "attention", "frustration", "attribution"):
        if layer in data:
            scores = data[layer].get("residue_scores", [])
            data[layer]["residue_scores"] = [float(x) for x in scores]
    return data


# ---------------------------------------------------------------------------
# Public: live ToPE inference
# ---------------------------------------------------------------------------

def run_tope_live(
    pdb_path: str,
    layers: List[str],
    substrate_smiles: Optional[str] = None,
    task: str = "kinetics",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run ToPE inference in-process and return the canonical output dict.

    Parameters
    ----------
    pdb_path:
        Path to the PDB / mmCIF file already loaded in PyMOL.
    layers:
        Which layers to compute: any subset of
        ``["voip", "attention", "frustration", "attribution"]``.
    substrate_smiles:
        SMILES string for the substrate (required for "attention" layer).
    task:
        Attribution task: ``"kinetics"``, ``"ec"``, or ``"selectivity"``.
    device:
        PyTorch device string.

    Returns
    -------
    Canonical output dict (same schema as :func:`load_tope_output`).

    Raises
    ------
    ImportError:
        If ToPE is not installed in the current Python environment.
    """
    _check_tope_installed()

    import torch
    from tope.models.tope_model import ToPEModel, CompleteToPEConfig
    from tope.topology.phase2_topological_encoding import EnzymePCC
    from tope.attribution.attribution import TopeAttributor, AttributorConfig
    from tope.utils.mcp_adapters import load_structure_from_pdb

    device_obj = torch.device(device)
    result: Dict[str, Any] = {}

    print(f"[ToPE] Loading structure from {pdb_path} …")
    protein_graph = load_structure_from_pdb(pdb_path)
    n_residues    = _count_residues(protein_graph)
    result["n_residues"] = n_residues
    result["pdb_path"]   = pdb_path

    print(f"[ToPE] Building Enzyme-PCC for {n_residues} residues …")
    enzyme_pcc = EnzymePCC(protein_graph)

    # ── Model (shared across layers) ────────────────────────────────────────
    cfg   = CompleteToPEConfig()
    model = ToPEModel(cfg).to(device_obj)
    model.eval()

    batch = _graph_to_batch(protein_graph, device_obj)

    with torch.no_grad():
        preds = model(batch)

    # ── VOIP layer ──────────────────────────────────────────────────────────
    if "voip" in layers:
        print("[ToPE] Computing VOIP scores …")
        voip_scores = _extract_voip_scores(enzyme_pcc, n_residues)
        result["voip"] = {
            "residue_scores": voip_scores,
            "raw_voip":       voip_scores,
        }

    # ── Attention layer ─────────────────────────────────────────────────────
    if "attention" in layers:
        print("[ToPE] Computing substrate cross-attention …")
        if substrate_smiles is None:
            print("[ToPE] Warning: no substrate SMILES provided — using uniform attention.")
            attn_scores = [1.0 / n_residues] * n_residues
        else:
            attn_scores = _extract_attention_scores(
                model, batch, substrate_smiles, device_obj, n_residues
            )
        result["attention"] = {
            "residue_scores":  attn_scores,
            "substrate_smiles": substrate_smiles or "",
        }

    # ── Frustration layer ───────────────────────────────────────────────────
    if "frustration" in layers:
        print("[ToPE] Computing holonomy frustration …")
        frust_scores, holonomy_norm, frustrated_resi = _compute_frustration(
            enzyme_pcc, n_residues
        )
        result["frustration"] = {
            "residue_scores":      frust_scores,
            "holonomy_norm":       holonomy_norm,
            "frustrated_residues": frustrated_resi,
        }

    # ── Attribution layer ───────────────────────────────────────────────────
    if "attribution" in layers:
        print(f"[ToPE] Computing attribution (task={task}) …")
        attributor = TopeAttributor(model, AttributorConfig(), device_obj)
        attr_result = attributor.attribute(
            protein_graph=batch["protein_graph"],
            target_task=task,
            target_idx=0,
        )
        attr_scores = _attr_result_to_list(attr_result, n_residues)
        result["attribution"] = {
            "residue_scores":        attr_scores,
            "zone_importance":       attr_result.zone_importance or {},
            "pathways":              attr_result.pathways or [],
            "catalytic_residues":    [],
            "task":                  task,
            "mcsa_overlap":          attr_result.mcsa_overlap or 0.0,
        }

    if "kinetics" in (preds or {}):
        kp = preds["kinetics"]
        result["kinetics"] = {
            "log_kcat":    float(kp[0][0]),
            "log_km":      float(kp[0][1]),
            "log_kcat_km": float(kp[0][2]),
        }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_tope_installed() -> None:
    try:
        import tope  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ToPE is not installed in this Python environment.\n"
            "Install it with:\n"
            "    pip install -e /path/to/tope\n"
            "or run PyMOL from a conda environment that has ToPE installed."
        ) from exc


def _count_residues(protein_graph: Any) -> int:
    """Best-effort residue count from a protein graph dict."""
    if isinstance(protein_graph, dict):
        for key in ("n_residues", "num_residues"):
            if key in protein_graph:
                return int(protein_graph[key])
        if "node_features" in protein_graph:
            import torch
            feats = protein_graph["node_features"]
            return int(feats.shape[0])
    return 100  # fallback


def _graph_to_batch(protein_graph: Any, device: Any) -> Dict[str, Any]:
    import torch
    batch: Dict[str, Any] = {"protein_graph": {}}
    if isinstance(protein_graph, dict):
        for k, v in protein_graph.items():
            if isinstance(v, np.ndarray):
                batch["protein_graph"][k] = torch.from_numpy(v).to(device)
            elif isinstance(v, list):
                try:
                    batch["protein_graph"][k] = torch.tensor(v).to(device)
                except (TypeError, ValueError):
                    batch["protein_graph"][k] = v
            else:
                batch["protein_graph"][k] = v
    return batch


def _extract_voip_scores(enzyme_pcc: Any, n_residues: int) -> List[float]:
    """Extract mean VOIP per residue from the Enzyme-PCC sheaf sections."""
    try:
        voip_by_residue: List[float] = []
        for res_idx in range(n_residues):
            voip = enzyme_pcc.get_residue_voip(res_idx)
            voip_by_residue.append(float(voip))
        return voip_by_residue
    except Exception:
        # Fallback: return zeros (structure loaded but VOIP not yet computed)
        return [0.0] * n_residues


def _extract_attention_scores(
    model: Any,
    batch: Dict[str, Any],
    smiles: str,
    device: Any,
    n_residues: int,
) -> List[float]:
    """Extract per-residue max attention weight from cross-attention module."""
    try:
        import torch
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * n_residues

        # Build a minimal substrate graph (atom features only)
        n_atoms = mol.GetNumAtoms()
        atom_feats = torch.zeros(n_atoms, model.cfg.hidden_dim, device=device)
        batch["substrate"] = {"node_features": atom_feats}

        with torch.no_grad():
            out = model(batch)

        # cross_attention returns per-head weights; take max over heads
        if hasattr(model, "cross_attention") and model.cross_attention is not None:
            # Intercept attention weights via hook
            attn_weights: List[Any] = []
            hook = model.cross_attention.enzyme_to_substrate.register_forward_hook(
                lambda m, inp, out_: attn_weights.append(out_[1])
            )
            model(batch)
            hook.remove()
            if attn_weights:
                w = attn_weights[-1].squeeze(0)  # (n_heads, n_enzyme_atoms, n_substrate_atoms)
                per_residue = w.max(dim=-1).values.mean(dim=0).cpu().tolist()
                if len(per_residue) == n_residues:
                    return per_residue

        return [0.0] * n_residues
    except Exception:
        return [0.0] * n_residues


def _compute_frustration(enzyme_pcc: Any, n_residues: int):
    """
    Compute holonomy frustration (FrustIndex) per residue.

    FrustIndex = ||d_AI(Hol(γ), I_d)|| accumulated over catalytic cycles
    containing residue i, then normalised to [0,1].
    """
    try:
        import torch
        # Ask the PCC for its frustration tensor if available
        frust = enzyme_pcc.compute_frustration_index()  # → (n_residues,) tensor
        frust_list = frust.cpu().tolist()
        holonomy_norm = float(frust.max())
        threshold = float(np.percentile(frust_list, 80))
        frustrated = [i for i, f in enumerate(frust_list) if f >= threshold]
        return frust_list, holonomy_norm, frustrated
    except Exception:
        scores = [0.0] * n_residues
        return scores, 0.0, []


def _attr_result_to_list(attr_result: Any, n_residues: int) -> List[float]:
    """Convert AttributionResult.residue_importance dict → ordered list."""
    rd: Dict[int, float] = getattr(attr_result, "residue_importance", {}) or {}
    return [float(rd.get(i, 0.0)) for i in range(n_residues)]
