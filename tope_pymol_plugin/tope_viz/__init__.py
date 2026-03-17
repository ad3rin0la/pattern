"""
ToPE Visualizer — PyMOL Plugin
================================
Topological Protein Encoder visualization suite for PyMOL.

Exposes four visualization layers:
  1. VOIP / electronic potential coloring
  2. Substrate cross-attention scores
  3. Holonomy frustration (FrustIndex) heatmap
  4. Phase 4 attribution / saliency maps

Workflow:
  - Run ToPE inference live (requires ToPE installed in the same Python env)
  - Load pre-computed ToPE outputs (JSON or NPZ files)

Usage (PyMOL command line):
  tope_voip        [selection] [--output path/to/output.json]
  tope_attention   [selection] [--substrate SMILES] [--output ...]
  tope_frustration [selection] [--output ...]
  tope_attribution [selection] [--task kinetics|ec|selectivity] [--output ...]
  tope_load        path/to/output.json [--layer voip|attention|frustration|attribution]

Or open the GUI panel via:
  Plugin → ToPE Visualizer
"""

from __future__ import annotations

import os
import sys

# Register the plugin directory so sub-modules are importable
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def __init_plugin__(app=None):
    """PyMOL plugin entry-point — called once on startup."""
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("ToPE Visualizer", _launch_gui)


def _launch_gui():
    """Open the ToPE Qt GUI panel."""
    from tope_viz.gui import TopeVisualizerDialog
    from pymol.Qt import QtWidgets

    dialog = TopeVisualizerDialog()
    dialog.show()
