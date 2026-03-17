"""
tope_viz.commands
==================
PyMOL command-line interface for ToPE Visualizer.

All commands are registered via ``cmd.extend`` in ``register_commands()``,
which is called by the GUI module on startup.

Commands
--------
tope_voip        [selection [output_json]]
tope_attention   [selection [substrate_smiles [output_json]]]
tope_frustration [selection [output_json]]
tope_attribution [selection [task [output_json]]]
tope_load        <file> [layer] [selection]
tope_kinetics    [selection]   – print predicted kinetics to log
tope_help                      – print quick-reference
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Lazy imports so this module loads even outside a PyMOL session
# ---------------------------------------------------------------------------

def _cmd():
    from pymol import cmd as _c
    return _c


def _print(msg: str) -> None:
    try:
        from pymol import cmd
        cmd.print(msg)
    except Exception:
        print(msg)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_pdb_path(selection: str) -> str:
    """
    Write the current PyMOL selection to a temp PDB file and return the path.
    """
    from pymol import cmd
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.close()
    cmd.save(tmp.name, selection)
    return tmp.name


def _apply_layer(
    data: Dict[str, Any],
    layer: str,
    selection: str,
    palette_override: Optional[str] = None,
) -> None:
    """
    Extract residue scores for *layer* from *data* and paint *selection*.
    """
    from tope_viz.coloring import apply_residue_scores, highlight_catalytic_residues, draw_allosteric_pathway
    from pymol import cmd

    layer_data = data.get(layer)
    if not layer_data:
        _print(f"[ToPE] No '{layer}' data available.")
        return

    scores_list = layer_data.get("residue_scores", [])
    if not scores_list:
        _print(f"[ToPE] '{layer}' scores list is empty.")
        return

    scores_dict = {i: float(v) for i, v in enumerate(scores_list)}
    palette = palette_override or layer  # falls back to layer name
    apply_residue_scores(selection, scores_dict, palette=palette)

    n = len(scores_list)
    nonzero = sum(1 for v in scores_list if abs(v) > 1e-6)
    _print(f"[ToPE] '{layer}' applied to {n} residues ({nonzero} non-zero).")

    # Layer-specific extras
    if layer == "attribution":
        cat_resi = layer_data.get("catalytic_residues", [])
        if cat_resi:
            highlight_catalytic_residues(cmd, selection, cat_resi)
            _print(f"[ToPE] Highlighted {len(cat_resi)} catalytic residues (yellow sticks).")

        pathways = layer_data.get("pathways", [])
        for i, pathway in enumerate(pathways[:3]):  # max 3 pathways drawn
            if len(pathway) >= 2:
                coords = _resi_to_coords(cmd, selection, pathway)
                if coords:
                    colours = [(1.0, 0.4, 0.0), (0.0, 0.8, 0.8), (0.8, 0.0, 0.8)]
                    draw_allosteric_pathway(cmd, coords, name=f"tope_pathway_{i+1}",
                                           colour=colours[i % 3])
            _print(f"[ToPE] Drew allosteric pathway {i+1} ({len(pathway)} residues).")

    if layer == "frustration":
        holonomy = layer_data.get("holonomy_norm", None)
        frustrated = layer_data.get("frustrated_residues", [])
        if holonomy is not None:
            _print(f"[ToPE] Global holonomy norm ||d_AI(Hol(γ),Id)|| = {holonomy:.4f}")
        if frustrated:
            _print(f"[ToPE] {len(frustrated)} highly frustrated residues (top 20%).")

    if layer == "attention":
        smiles = layer_data.get("substrate_smiles", "")
        if smiles:
            _print(f"[ToPE] Substrate: {smiles}")

    if layer == "voip":
        thresholds = layer_data.get("thresholds", {})
        if thresholds:
            _print(f"[ToPE] VOIP thresholds: {thresholds}")


def _resi_to_coords(cmd, selection: str, resi_indices: list):
    """Map 0-based residue indices to Cα coordinates."""
    resi_list = []
    cmd.iterate(f"({selection}) and name CA", "resi_list.append((resi, x, y, z))",
                space={"resi_list": resi_list})
    resi_list.sort(key=lambda t: int(t[0]))
    coords = []
    for idx in resi_indices:
        if 0 <= idx < len(resi_list):
            _, x, y, z = resi_list[idx]
            coords.append((x, y, z))
    return coords


def _maybe_save(data: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)
    _print(f"[ToPE] Output saved to {output_path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_tope_voip(selection: str = "all", output: str = "") -> None:
    """
    Color the structure by VOIP (valence of ionisation potential) scores,
    computed from the Enzyme-PCC sheaf sections.

    Usage:  tope_voip [selection [output_json]]
    """
    from tope_viz.runner import run_tope_live

    pdb_path = _get_pdb_path(selection)
    try:
        data = run_tope_live(pdb_path, layers=["voip"])
        _apply_layer(data, "voip", selection)
        _maybe_save(data, output or None)
    finally:
        os.unlink(pdb_path)


def cmd_tope_attention(
    selection:       str = "all",
    substrate_smiles: str = "",
    output:          str = "",
) -> None:
    """
    Color the structure by substrate cross-attention scores.

    Usage:  tope_attention [selection [substrate_smiles [output_json]]]

    Example:
        tope_attention chain A "C1=CC=CC=C1"
    """
    from tope_viz.runner import run_tope_live

    pdb_path = _get_pdb_path(selection)
    try:
        data = run_tope_live(
            pdb_path,
            layers=["attention"],
            substrate_smiles=substrate_smiles or None,
        )
        _apply_layer(data, "attention", selection)
        _maybe_save(data, output or None)
    finally:
        os.unlink(pdb_path)


def cmd_tope_frustration(selection: str = "all", output: str = "") -> None:
    """
    Color the structure by holonomy frustration (FrustIndex).

    High values (magenta) indicate electronically frustrated regions —
    residues where the catalytic cycle holonomy deviates strongly from the
    identity, per d_AI(Hol(γ), Id).

    Usage:  tope_frustration [selection [output_json]]
    """
    from tope_viz.runner import run_tope_live

    pdb_path = _get_pdb_path(selection)
    try:
        data = run_tope_live(pdb_path, layers=["frustration"])
        _apply_layer(data, "frustration", selection)
        _maybe_save(data, output or None)
    finally:
        os.unlink(pdb_path)


def cmd_tope_attribution(
    selection: str = "all",
    task:      str = "kinetics",
    output:    str = "",
) -> None:
    """
    Color the structure by Phase 4 integrated-gradient attribution saliency.

    Usage:  tope_attribution [selection [task [output_json]]]
    task:   kinetics | ec | selectivity

    Catalytic residues (if known via M-CSA) are shown as yellow sticks.
    Allosteric pathways are drawn as CGO tubes.
    """
    from tope_viz.runner import run_tope_live

    pdb_path = _get_pdb_path(selection)
    try:
        data = run_tope_live(pdb_path, layers=["attribution"], task=task)
        _apply_layer(data, "attribution", selection)
        _maybe_save(data, output or None)

        if "kinetics" in data:
            k = data["kinetics"]
            _print(
                f"[ToPE] Predicted kinetics — "
                f"log(kcat)={k['log_kcat']:.2f}, "
                f"log(Km)={k['log_km']:.2f}, "
                f"log(kcat/Km)={k['log_kcat_km']:.2f}"
            )
    finally:
        os.unlink(pdb_path)


def cmd_tope_all(
    selection:        str = "all",
    substrate_smiles: str = "",
    task:             str = "kinetics",
    output:           str = "",
) -> None:
    """
    Run all four ToPE layers in one pass and open the comparison GUI.

    Usage:  tope_all [selection [substrate_smiles [task [output_json]]]]
    """
    from tope_viz.runner import run_tope_live

    pdb_path = _get_pdb_path(selection)
    layers = ["voip", "attention", "frustration", "attribution"]
    try:
        data = run_tope_live(
            pdb_path,
            layers=layers,
            substrate_smiles=substrate_smiles or None,
            task=task,
        )
        # Default display: VOIP (the always-available layer)
        _apply_layer(data, "voip", selection)
        _maybe_save(data, output or None)
        _print("[ToPE] All layers computed. Use tope_load + layer name to switch views.")
    finally:
        os.unlink(pdb_path)


def cmd_tope_load(
    filepath:  str,
    layer:     str = "voip",
    selection: str = "all",
) -> None:
    """
    Load a pre-computed ToPE output file and apply a visualization layer.

    Usage:  tope_load <file> [layer] [selection]
    layer:  voip | attention | frustration | attribution
    file:   path to a .json or .npz file produced by ToPE

    Example:
        tope_load /data/4ake_tope.json attention chain A
    """
    from tope_viz.runner import load_tope_output

    data = load_tope_output(filepath)
    _apply_layer(data, layer, selection)

    available = [k for k in ("voip", "attention", "frustration", "attribution") if k in data]
    _print(f"[ToPE] File contains layers: {available}")
    if "kinetics" in data:
        k = data["kinetics"]
        _print(
            f"[ToPE] Stored kinetics — "
            f"log(kcat)={k['log_kcat']:.2f}, "
            f"log(Km)={k['log_km']:.2f}, "
            f"log(kcat/Km)={k['log_kcat_km']:.2f}"
        )


def cmd_tope_kinetics(selection: str = "all") -> None:
    """
    Run ToPE kinetics prediction and print results to the PyMOL log.

    Usage:  tope_kinetics [selection]
    """
    from tope_viz.runner import run_tope_live
    import math

    pdb_path = _get_pdb_path(selection)
    try:
        data = run_tope_live(pdb_path, layers=[])
    finally:
        os.unlink(pdb_path)

    if "kinetics" not in data:
        _print("[ToPE] Kinetics not available — model may need re-running with --export.")
        return

    k = data["kinetics"]
    kcat    = math.exp(k["log_kcat"])
    km      = math.exp(k["log_km"])
    eff     = math.exp(k["log_kcat_km"])
    _print(
        f"[ToPE] Kinetics prediction:\n"
        f"  kcat            = {kcat:.3e}  s⁻¹\n"
        f"  Km              = {km:.3e}  M\n"
        f"  kcat/Km         = {eff:.3e}  M⁻¹s⁻¹"
    )


def cmd_tope_help() -> None:
    """Print a quick-reference of all ToPE commands."""
    _print(
        "\n"
        "═══════════════════════════════════════════════════════\n"
        "  ToPE Visualizer — PyMOL commands\n"
        "═══════════════════════════════════════════════════════\n"
        "\n"
        "  tope_voip [sel [out.json]]\n"
        "      Color by VOIP / electronic potential\n"
        "      Blue → low VOIP   Red → high VOIP\n"
        "\n"
        "  tope_attention [sel [SMILES [out.json]]]\n"
        "      Color by substrate cross-attention scores\n"
        "      White → unattended   Green → high attention\n"
        "\n"
        "  tope_frustration [sel [out.json]]\n"
        "      Color by holonomy frustration (FrustIndex)\n"
        "      Cyan → ordered   Magenta → frustrated\n"
        "\n"
        "  tope_attribution [sel [task [out.json]]]\n"
        "      Color by Phase 4 integrated-gradient saliency\n"
        "      task: kinetics | ec | selectivity\n"
        "      Yellow sticks: M-CSA catalytic residues\n"
        "      CGO tubes: allosteric pathways\n"
        "\n"
        "  tope_all [sel [SMILES [task [out.json]]]]\n"
        "      Compute all four layers in one pass\n"
        "\n"
        "  tope_load <file> [layer] [sel]\n"
        "      Load pre-computed .json/.npz and display a layer\n"
        "\n"
        "  tope_kinetics [sel]\n"
        "      Predict and print kcat, Km, kcat/Km\n"
        "\n"
        "  tope_help\n"
        "      Show this message\n"
        "\n"
        "  GUI:  Plugin → ToPE Visualizer\n"
        "═══════════════════════════════════════════════════════\n"
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_commands() -> None:
    """
    Register all ToPE commands with PyMOL's ``cmd`` system.
    Call this once, e.g. from the plugin ``__init__`` or the GUI module.
    """
    from pymol import cmd

    cmd.extend("tope_voip",        cmd_tope_voip)
    cmd.extend("tope_attention",   cmd_tope_attention)
    cmd.extend("tope_frustration", cmd_tope_frustration)
    cmd.extend("tope_attribution", cmd_tope_attribution)
    cmd.extend("tope_all",         cmd_tope_all)
    cmd.extend("tope_load",        cmd_tope_load)
    cmd.extend("tope_kinetics",    cmd_tope_kinetics)
    cmd.extend("tope_help",        cmd_tope_help)

    cmd.print("[ToPE] Commands registered. Type 'tope_help' for usage.")
