"""
ToPE Visualizer — PyMOL Plugin
================================
Entry point loaded by PyMOL's plugin system.

Install
-------
In PyMOL:
    Plugin → Plugin Manager → Install New Plugin → choose tope_pymol_plugin/

Or add to PyMOL startup script (pymolrc):
    run /path/to/tope_pymol_plugin/__init__.py

Or install for all PyMOL sessions:
    Copy the tope_pymol_plugin directory into:
        ~/.pymol/startup/   (Linux/macOS)
        %APPDATA%\\PyMOL\\startup\\   (Windows)
"""

from __future__ import annotations

import os
import sys

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def __init_plugin__(app=None):
    """Called once by PyMOL when the plugin is loaded."""
    try:
        from pymol.plugins import addmenuitemqt
        addmenuitemqt("ToPE Visualizer", _launch_gui)
    except Exception:
        pass   # Headless / no GUI — commands still work

    # Register CLI commands immediately (even without GUI)
    try:
        from tope_viz.commands import register_commands
        register_commands()
    except Exception as exc:
        try:
            from pymol import cmd
            cmd.print(f"[ToPE] Warning: could not register commands: {exc}")
        except Exception:
            print(f"[ToPE] Warning: could not register commands: {exc}")


def _launch_gui():
    """Open the ToPE Qt panel."""
    try:
        from tope_viz.gui import TopeVisualizerDialog
        dialog = TopeVisualizerDialog()
        dialog.show()
        return dialog  # keep reference alive
    except Exception as exc:
        try:
            from pymol import cmd
            cmd.print(f"[ToPE] GUI error: {exc}")
        except Exception:
            print(f"[ToPE] GUI error: {exc}")
        raise
