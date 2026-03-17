"""
tope_viz.gui
=============
Qt5 GUI panel for ToPE Visualizer, docked inside PyMOL.

Tab layout
----------
  [VOIP]  [Attention]  [Frustration]  [Attribution]  [Load File]

Each tab has:
  - Layer-specific controls (substrate SMILES, task selector, etc.)
  - Run button (live inference) or Load button (pre-computed file)
  - B-factor export checkbox
  - Status / log area at the bottom

The panel is modeless so users can interact with PyMOL while it is open.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Qt imports — gracefully degrade if not in PyMOL context
# ---------------------------------------------------------------------------
try:
    from pymol.Qt import QtWidgets, QtCore, QtGui  # type: ignore[import]
    from pymol.Qt.utils import getSaveFileNameWithExt  # type: ignore[import]
    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

    class _Stub:
        """Allow module-level class definitions even outside PyMOL."""
        pass

    class QtWidgets:  # type: ignore[misc]
        QDialog = _Stub
        QWidget = _Stub
        QTabWidget = _Stub

    class QtCore:  # type: ignore[misc]
        class Qt:
            WindowStaysOnTopHint = 0

    class QtGui:  # type: ignore[misc]
        pass


# ---------------------------------------------------------------------------
# Colour constants used in the UI
# ---------------------------------------------------------------------------
_LAYER_COLORS: Dict[str, str] = {
    "voip":        "#3b82f6",   # blue
    "attention":   "#22c55e",   # green
    "frustration": "#a855f7",   # violet
    "attribution": "#f97316",   # orange
}


# ---------------------------------------------------------------------------
# Worker thread for non-blocking inference
# ---------------------------------------------------------------------------

class _InferenceWorker(QtCore.QThread if _QT_AVAILABLE else object):  # type: ignore[misc]
    """Run ToPE inference in a background thread and emit results."""

    finished  = QtCore.pyqtSignal(dict)   if _QT_AVAILABLE else None
    error     = QtCore.pyqtSignal(str)    if _QT_AVAILABLE else None
    log_msg   = QtCore.pyqtSignal(str)    if _QT_AVAILABLE else None

    def __init__(
        self,
        pdb_path:         str,
        layers:           List[str],
        substrate_smiles: Optional[str],
        task:             str,
        device:           str,
        parent=None,
    ):
        super().__init__(parent)
        self.pdb_path         = pdb_path
        self.layers           = layers
        self.substrate_smiles = substrate_smiles
        self.task             = task
        self.device           = device

    def run(self):
        try:
            from tope_viz.runner import run_tope_live
            self.log_msg.emit(f"[ToPE] Starting inference (layers: {self.layers}) …")
            data = run_tope_live(
                self.pdb_path,
                layers=self.layers,
                substrate_smiles=self.substrate_smiles,
                task=self.task,
                device=self.device,
            )
            self.log_msg.emit("[ToPE] Inference complete.")
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Individual tab widgets
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class _VOIPTab(QtWidgets.QWidget):
        run_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Electronic Potential (VOIP)"))
            layout.addWidget(_info(
                "Colors residues by their Valence of Ionisation Potential "
                "computed from the Enzyme-PCC sheaf sections using GFN2-xTB "
                "geometry-dependent Mulliken charges.\n\n"
                "Blue → low VOIP (nucleophilic)\n"
                "Red  → high VOIP (electrophilic)"
            ))

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            self._out = _output_row()
            layout.addWidget(self._out)

            btn = QtWidgets.QPushButton("▶  Run VOIP")
            btn.setStyleSheet(f"background:{_LAYER_COLORS['voip']}; color:white; font-weight:bold; padding:6px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _emit(self):
            self.run_requested.emit({
                "layers": ["voip"],
                "selection": self._sel.text(),
                "output": self._out.text(),
            })

    # ────────────────────────────────────────────────────────────────────────

    class _AttentionTab(QtWidgets.QWidget):
        run_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Substrate Cross-Attention"))
            layout.addWidget(_info(
                "Shows which enzyme residues attend most strongly to the "
                "substrate via SubstrateVOIPCrossAttention, implementing "
                "Ioffe's electronic complementarity criterion.\n\n"
                "White → unattended\nForest green → high attention"
            ))

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            smi_row = QtWidgets.QHBoxLayout()
            smi_row.addWidget(QtWidgets.QLabel("Substrate SMILES:"))
            self._smiles = QtWidgets.QLineEdit()
            self._smiles.setPlaceholderText("e.g. C1CCCCC1  (leave blank for uniform)")
            smi_row.addWidget(self._smiles)
            layout.addLayout(smi_row)

            self._out = _output_row()
            layout.addWidget(self._out)

            btn = QtWidgets.QPushButton("▶  Run Attention")
            btn.setStyleSheet(f"background:{_LAYER_COLORS['attention']}; color:white; font-weight:bold; padding:6px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _emit(self):
            self.run_requested.emit({
                "layers": ["attention"],
                "selection": self._sel.text(),
                "substrate_smiles": self._smiles.text().strip() or None,
                "output": self._out.text(),
            })

    # ────────────────────────────────────────────────────────────────────────

    class _FrustrationTab(QtWidgets.QWidget):
        run_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Holonomy Frustration (FrustIndex)"))
            layout.addWidget(_info(
                "FrustIndex = ‖d_AI(Hol(γ), Id)‖ — the deviation of the "
                "holonomy around a catalytic cycle from the identity.\n"
                "High values indicate electronically frustrated residues where "
                "combinatorial criterion interactions create irresolvable "
                "competing demands.\n\n"
                "Cyan → ordered\nMagenta → frustrated"
            ))

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            self._out = _output_row()
            layout.addWidget(self._out)

            btn = QtWidgets.QPushButton("▶  Run Frustration")
            btn.setStyleSheet(f"background:{_LAYER_COLORS['frustration']}; color:white; font-weight:bold; padding:6px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _emit(self):
            self.run_requested.emit({
                "layers": ["frustration"],
                "selection": self._sel.text(),
                "output": self._out.text(),
            })

    # ────────────────────────────────────────────────────────────────────────

    class _AttributionTab(QtWidgets.QWidget):
        run_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Phase 4 Attribution / Saliency"))
            layout.addWidget(_info(
                "Integrated-gradient saliency with respect to the chosen "
                "prediction task.  Highlights which residues drive the "
                "predicted output — including Zone 3 (>20 Å) allosteric "
                "sites inaccessible to local models.\n\n"
                "White → neutral\nOrange → high saliency\n"
                "Yellow sticks → M-CSA catalytic residues\n"
                "CGO tubes → allosteric pathways"
            ))

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            task_row = QtWidgets.QHBoxLayout()
            task_row.addWidget(QtWidgets.QLabel("Task:"))
            self._task = QtWidgets.QComboBox()
            self._task.addItems(["kinetics", "ec", "selectivity"])
            task_row.addWidget(self._task)
            layout.addLayout(task_row)

            self._out = _output_row()
            layout.addWidget(self._out)

            btn = QtWidgets.QPushButton("▶  Run Attribution")
            btn.setStyleSheet(f"background:{_LAYER_COLORS['attribution']}; color:white; font-weight:bold; padding:6px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _emit(self):
            self.run_requested.emit({
                "layers": ["attribution"],
                "selection": self._sel.text(),
                "task": self._task.currentText(),
                "output": self._out.text(),
            })

    # ────────────────────────────────────────────────────────────────────────

    class _LoadTab(QtWidgets.QWidget):
        load_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Load Pre-Computed Output"))
            layout.addWidget(_info(
                "Load a ToPE output file (.json or .npz) produced by:\n"
                "  tope predict --export output.json\n\n"
                "Select which layer to display and which PyMOL selection to "
                "apply it to."
            ))

            # File picker
            file_row = QtWidgets.QHBoxLayout()
            self._file = QtWidgets.QLineEdit()
            self._file.setPlaceholderText("path/to/tope_output.json")
            browse_btn = QtWidgets.QPushButton("Browse…")
            browse_btn.clicked.connect(self._browse)
            file_row.addWidget(self._file)
            file_row.addWidget(browse_btn)
            layout.addLayout(file_row)

            layer_row = QtWidgets.QHBoxLayout()
            layer_row.addWidget(QtWidgets.QLabel("Layer:"))
            self._layer = QtWidgets.QComboBox()
            self._layer.addItems(["voip", "attention", "frustration", "attribution"])
            layer_row.addWidget(self._layer)
            layout.addLayout(layer_row)

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            btn = QtWidgets.QPushButton("📂  Load & Display")
            btn.setStyleSheet("background:#475569; color:white; font-weight:bold; padding:6px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _browse(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open ToPE Output", "",
                "ToPE Output (*.json *.npz);;All files (*)"
            )
            if fname:
                self._file.setText(fname)

        def _emit(self):
            self.load_requested.emit({
                "filepath":  self._file.text().strip(),
                "layer":     self._layer.currentText(),
                "selection": self._sel.text(),
            })

    # ────────────────────────────────────────────────────────────────────────

    class _AllTab(QtWidgets.QWidget):
        """Run all four layers at once."""
        run_requested = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)

            layout.addWidget(_section_label("Run All Layers"))
            layout.addWidget(_info(
                "Computes VOIP, Attention, Frustration, and Attribution in "
                "one pass and saves them to a single JSON file.\n\n"
                "After running, use other tabs (or tope_load) to switch "
                "between layer visualizations."
            ))

            self._sel = _selection_row()
            layout.addWidget(self._sel)

            smi_row = QtWidgets.QHBoxLayout()
            smi_row.addWidget(QtWidgets.QLabel("Substrate SMILES:"))
            self._smiles = QtWidgets.QLineEdit()
            self._smiles.setPlaceholderText("optional")
            smi_row.addWidget(self._smiles)
            layout.addLayout(smi_row)

            task_row = QtWidgets.QHBoxLayout()
            task_row.addWidget(QtWidgets.QLabel("Attribution task:"))
            self._task = QtWidgets.QComboBox()
            self._task.addItems(["kinetics", "ec", "selectivity"])
            task_row.addWidget(self._task)
            layout.addLayout(task_row)

            self._out = _output_row(label="Save to:")
            layout.addWidget(self._out)

            btn = QtWidgets.QPushButton("▶▶  Run All")
            btn.setStyleSheet("background:#0f172a; color:white; font-weight:bold; padding:8px; font-size:14px;")
            btn.clicked.connect(self._emit)
            layout.addWidget(btn)
            layout.addStretch()

        def _emit(self):
            self.run_requested.emit({
                "layers": ["voip", "attention", "frustration", "attribution"],
                "selection": self._sel.text(),
                "substrate_smiles": self._smiles.text().strip() or None,
                "task": self._task.currentText(),
                "output": self._out.text(),
            })


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

if _QT_AVAILABLE:

    class TopeVisualizerDialog(QtWidgets.QDialog):
        """Main ToPE Visualizer panel — modeless dialog."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("ToPE Visualizer")
            self.setMinimumWidth(520)
            self.setMinimumHeight(540)
            self.setWindowFlags(
                self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint
            )

            from tope_viz.commands import register_commands
            register_commands()

            self._worker: Optional[_InferenceWorker] = None
            self._last_data: Optional[Dict[str, Any]] = None

            root = QtWidgets.QVBoxLayout(self)

            # ── Header ──────────────────────────────────────────────────────
            header = QtWidgets.QLabel(
                "<b style='font-size:16px'>ToPE Visualizer</b>"
                "<span style='color:#64748b; font-size:11px'>"
                "  Topological Protein Encoder</span>"
            )
            header.setTextFormat(QtCore.Qt.RichText)
            root.addWidget(header)

            # ── Device row ──────────────────────────────────────────────────
            dev_row = QtWidgets.QHBoxLayout()
            dev_row.addWidget(QtWidgets.QLabel("Device:"))
            self._device = QtWidgets.QComboBox()
            self._device.addItems(["cpu", "cuda", "mps"])
            dev_row.addWidget(self._device)
            dev_row.addStretch()
            root.addLayout(dev_row)

            # ── Tabs ─────────────────────────────────────────────────────────
            self._tabs = QtWidgets.QTabWidget()

            self._voip_tab     = _VOIPTab()
            self._attn_tab     = _AttentionTab()
            self._frust_tab    = _FrustrationTab()
            self._attr_tab     = _AttributionTab()
            self._load_tab     = _LoadTab()
            self._all_tab      = _AllTab()

            self._tabs.addTab(self._voip_tab,  "⚡ VOIP")
            self._tabs.addTab(self._attn_tab,  "🔍 Attention")
            self._tabs.addTab(self._frust_tab, "🌀 Frustration")
            self._tabs.addTab(self._attr_tab,  "🔥 Attribution")
            self._tabs.addTab(self._load_tab,  "📂 Load File")
            self._tabs.addTab(self._all_tab,   "▶▶ Run All")

            root.addWidget(self._tabs)

            # ── Status / log ─────────────────────────────────────────────────
            root.addWidget(QtWidgets.QLabel("Log:"))
            self._log = QtWidgets.QPlainTextEdit()
            self._log.setReadOnly(True)
            self._log.setMaximumHeight(120)
            self._log.setStyleSheet("background:#0f172a; color:#94a3b8; font-family:monospace; font-size:11px;")
            root.addWidget(self._log)

            # ── Progress bar ─────────────────────────────────────────────────
            self._progress = QtWidgets.QProgressBar()
            self._progress.setRange(0, 0)  # indeterminate
            self._progress.setVisible(False)
            root.addWidget(self._progress)

            # ── Close button ─────────────────────────────────────────────────
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(self.close)
            root.addWidget(close_btn)

            # ── Signal connections ────────────────────────────────────────────
            self._voip_tab.run_requested.connect(self._run_layer)
            self._attn_tab.run_requested.connect(self._run_layer)
            self._frust_tab.run_requested.connect(self._run_layer)
            self._attr_tab.run_requested.connect(self._run_layer)
            self._all_tab.run_requested.connect(self._run_layer)
            self._load_tab.load_requested.connect(self._load_file)

        # ── Slot: run inference ───────────────────────────────────────────────

        def _run_layer(self, params: Dict[str, Any]) -> None:
            if self._worker is not None and self._worker.isRunning():
                self._log_msg("[ToPE] Inference already running — please wait.")
                return

            selection = params.get("selection", "all") or "all"

            import tempfile, os
            from pymol import cmd
            tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            tmp.close()
            cmd.save(tmp.name, selection)

            self._worker = _InferenceWorker(
                pdb_path=tmp.name,
                layers=params.get("layers", ["voip"]),
                substrate_smiles=params.get("substrate_smiles"),
                task=params.get("task", "kinetics"),
                device=self._device.currentText(),
            )
            self._worker.finished.connect(lambda data: self._on_done(data, tmp.name, params))
            self._worker.error.connect(lambda e: self._on_error(e, tmp.name))
            self._worker.log_msg.connect(self._log_msg)

            self._progress.setVisible(True)
            self._worker.start()

        def _on_done(self, data: Dict[str, Any], tmp_path: str, params: Dict[str, Any]) -> None:
            os.unlink(tmp_path)
            self._progress.setVisible(False)
            self._last_data = data

            from tope_viz.commands import _apply_layer, _maybe_save
            selection = params.get("selection", "all") or "all"
            for layer in (params.get("layers") or ["voip"]):
                _apply_layer(data, layer, selection)

            if out := params.get("output"):
                _maybe_save(data, out)

            if "kinetics" in data:
                import math
                k = data["kinetics"]
                self._log_msg(
                    f"Kinetics → kcat={math.exp(k['log_kcat']):.3e} s⁻¹  "
                    f"Km={math.exp(k['log_km']):.3e} M  "
                    f"kcat/Km={math.exp(k['log_kcat_km']):.3e} M⁻¹s⁻¹"
                )

        def _on_error(self, message: str, tmp_path: str) -> None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            self._progress.setVisible(False)
            self._log_msg(f"[ToPE ERROR] {message}")
            QtWidgets.QMessageBox.critical(self, "ToPE Error", message)

        # ── Slot: load file ───────────────────────────────────────────────────

        def _load_file(self, params: Dict[str, Any]) -> None:
            filepath = params.get("filepath", "")
            if not filepath:
                self._log_msg("[ToPE] No file specified.")
                return

            from tope_viz.runner import load_tope_output
            from tope_viz.commands import _apply_layer

            try:
                data = load_tope_output(filepath)
                self._last_data = data
                layer     = params.get("layer", "voip")
                selection = params.get("selection", "all") or "all"
                _apply_layer(data, layer, selection)

                available = [k for k in ("voip", "attention", "frustration", "attribution") if k in data]
                self._log_msg(f"Loaded {filepath}")
                self._log_msg(f"Available layers: {available}")
            except Exception as exc:
                self._log_msg(f"[ToPE ERROR] {exc}")
                QtWidgets.QMessageBox.critical(self, "ToPE Error", str(exc))

        # ── Log helper ────────────────────────────────────────────────────────

        def _log_msg(self, msg: str) -> None:
            self._log.appendPlainText(msg)
            self._log.verticalScrollBar().setValue(
                self._log.verticalScrollBar().maximum()
            )


# ---------------------------------------------------------------------------
# UI helper constructors
# ---------------------------------------------------------------------------

def _section_label(text: str) -> "QtWidgets.QLabel":
    lbl = QtWidgets.QLabel(f"<b>{text}</b>")
    lbl.setTextFormat(QtCore.Qt.RichText)
    return lbl


def _info(text: str) -> "QtWidgets.QLabel":
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#475569; font-size:11px; padding:4px 0;")
    return lbl


def _selection_row(default: str = "all") -> "QtWidgets.QLineEdit":
    row = QtWidgets.QHBoxLayout()
    # This returns a widget but we embed inside a wrapper
    container = QtWidgets.QWidget()
    layout    = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel("Selection:"))
    le = QtWidgets.QLineEdit(default)
    le.setPlaceholderText("e.g.  chain A  or  all")
    layout.addWidget(le)
    # Attach text() to the container so callers can do self._sel.text()
    container.text = le.text  # type: ignore[attr-defined]
    return container           # type: ignore[return-value]


def _output_row(label: str = "Save output:") -> "QtWidgets.QLineEdit":
    container = QtWidgets.QWidget()
    layout    = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QtWidgets.QLabel(label))
    le = QtWidgets.QLineEdit()
    le.setPlaceholderText("optional: path/to/output.json")
    layout.addWidget(le)
    browse = QtWidgets.QPushButton("…")
    browse.setMaximumWidth(28)

    def _browse():
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save ToPE Output", "", "JSON (*.json);;All (*)"
        )
        if fname:
            le.setText(fname)

    browse.clicked.connect(_browse)
    layout.addWidget(browse)
    container.text = le.text  # type: ignore[attr-defined]
    return container           # type: ignore[return-value]
