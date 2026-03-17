"""
tope_viz.coloring
==================
Maps per-residue scalar scores onto PyMOL's color system.

All four layers (VOIP, attention, frustration, attribution) reduce to the
same problem: assign a float in [0, 1] to each residue and paint the
structure accordingly.  This module owns that mapping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

PALETTES: Dict[str, Tuple[str, str]] = {
    # name → (low-colour, high-colour) for PyMOL's spectrum command
    "voip":        ("blue",  "red"),       # low potential → high potential
    "attention":   ("white", "forest"),    # no attention → high attention
    "frustration": ("cyan",  "magenta"),   # ordered → frustrated
    "attribution": ("white", "orange"),    # neutral → high saliency
    "diverging":   ("blue",  "red"),       # generic diverging
}

# Discrete colour names for individual residue colouring
_SPECTRUM_COLORS = [
    "blue", "cyan", "green", "yellow", "orange", "red", "magenta",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_residue_scores(
    selection: str,
    scores: Dict[int, float],
    palette: str = "diverging",
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
    b_factor: bool = True,
) -> None:
    """
    Paint *selection* in PyMOL using per-residue *scores*.

    Parameters
    ----------
    selection:
        PyMOL atom selection string, e.g. ``"chain A"`` or ``"all"``.
    scores:
        ``{residue_index: float}`` mapping.  Residue indices are
        0-based and correspond to the ordering in the PDB file
        (matching ToPE's internal residue indexing).
    palette:
        Key into ``PALETTES``.  Determines the colour gradient.
    percentile_clip:
        Clip scores to (lo, hi) percentile before normalising, to
        suppress outliers.
    b_factor:
        If True, write normalised scores into the B-factor column so
        external tools (e.g. ChimeraX) can also read them.
    """
    try:
        import pymol
        from pymol import cmd
    except ImportError as exc:
        raise RuntimeError(
            "PyMOL is not importable — this function must be called "
            "from within a live PyMOL session."
        ) from exc

    if not scores:
        cmd.print("ToPE: no scores to display.")
        return

    # ── Normalise scores ────────────────────────────────────────────────────
    indices = np.array(list(scores.keys()), dtype=int)
    values  = np.array([scores[i] for i in indices], dtype=float)

    lo = np.percentile(values, percentile_clip[0])
    hi = np.percentile(values, percentile_clip[1])
    if hi == lo:
        hi = lo + 1e-6
    normed = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    score_map: Dict[int, float] = dict(zip(indices.tolist(), normed.tolist()))

    # ── Write into B-factor column ──────────────────────────────────────────
    if b_factor:
        cmd.alter(selection, "b = 0.0")
        cmd.iterate(
            selection,
            "_tope_score_map = __import__('builtins').__dict__.get('_tope_score_map', {})",
        )
        # Use stored_score approach (thread-safe in PyMOL)
        import pymol.stored as stored  # type: ignore[import]
        stored._tope_score_map = score_map

        cmd.alter(
            selection,
            "b = __import__('pymol').stored._tope_score_map.get("
            "    resi_idx if hasattr(space, 'resi_idx') else 0, 0.0"
            ")",
        )
        # Simpler, robust approach: alter by residue number
        _write_bfactors_by_residue(cmd, selection, score_map)

    # ── Apply colour spectrum ────────────────────────────────────────────────
    low_col, high_col = PALETTES.get(palette, ("blue", "red"))
    cmd.spectrum("b", f"{low_col}_white_{high_col}", selection)
    cmd.rebuild(selection)


def _write_bfactors_by_residue(cmd, selection: str, score_map: Dict[int, float]) -> None:
    """
    Iterate over residues in *selection* and write normalised score into b.

    Uses PyMOL's ``stored`` namespace for safe data transfer.
    """
    import pymol.stored as stored  # type: ignore[import]

    # Collect (resi_string → 0-based_index) mapping
    resi_list: List[str] = []
    cmd.iterate(f"({selection}) and name CA", "resi_list.append(resi)",
                space={"resi_list": resi_list})

    # Build resi → score lookup (resi is a string like "42")
    resi_to_score: Dict[str, float] = {}
    seen = 0
    for resi_str in sorted(set(resi_list), key=lambda x: int(x)):
        resi_to_score[resi_str] = score_map.get(seen, 0.0)
        seen += 1

    stored._tope_resi_score = resi_to_score
    cmd.alter(
        selection,
        "b = __import__('pymol').stored._tope_resi_score.get(resi, 0.0)",
    )


# ---------------------------------------------------------------------------
# Utility: build a per-atom discrete selection for catalytic residues
# ---------------------------------------------------------------------------

def highlight_catalytic_residues(
    cmd,
    selection: str,
    catalytic_resi: List[int],
    name: str = "catalytic_sites",
) -> None:
    """
    Create a named PyMOL selection for catalytic residue indices and
    draw them as sticks with a contrasting colour.
    """
    if not catalytic_resi:
        return

    resi_list: List[str] = []
    cmd.iterate(f"({selection}) and name CA", "resi_list.append(resi)",
                space={"resi_list": resi_list})
    resi_list = sorted(set(resi_list), key=lambda x: int(x))

    sel_parts: List[str] = []
    for idx in catalytic_resi:
        if 0 <= idx < len(resi_list):
            sel_parts.append(f"resi {resi_list[idx]}")

    if not sel_parts:
        return

    sele_expr = f"({selection}) and ({' or '.join(sel_parts)})"
    cmd.select(name, sele_expr)
    cmd.show("sticks", name)
    cmd.color("yellow", name)
    cmd.deselect()


# ---------------------------------------------------------------------------
# Utility: draw an allosteric pathway as a CGO line
# ---------------------------------------------------------------------------

def draw_allosteric_pathway(
    cmd,
    coords: List[Tuple[float, float, float]],
    name: str = "tope_pathway",
    colour: Tuple[float, float, float] = (1.0, 0.5, 0.0),
    width: float = 3.0,
) -> None:
    """
    Draw a CGO tube connecting *coords* in Å-space to represent an
    allosteric pathway identified by Phase 4 attribution.
    """
    from pymol.cgo import CYLINDER, BEGIN, LINES, COLOR, VERTEX, END  # type: ignore[import]

    if len(coords) < 2:
        return

    r, g, b = colour
    obj = []
    for i in range(len(coords) - 1):
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[i + 1]
        obj.extend([
            CYLINDER,
            x0, y0, z0,
            x1, y1, z1,
            width * 0.1,   # radius in Å
            r, g, b,
            r, g, b,
        ])

    cmd.load_cgo(obj, name)
