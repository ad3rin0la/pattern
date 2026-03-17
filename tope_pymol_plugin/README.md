# ToPE Visualizer — PyMOL Plugin

A PyMOL plugin for the **Topological Protein Encoder (ToPE)** that maps
electronic-topological features directly onto protein 3D structure.

## Visualization layers

| Layer | What it shows | Colour ramp |
|---|---|---|
| **VOIP** | Valence of Ionisation Potential per residue from Enzyme-PCC sheaf sections (GFN2-xTB Mulliken charges) | Blue → Red |
| **Attention** | Substrate cross-attention scores — where the enzyme "reads" the substrate via `SubstrateVOIPCrossAttention` | White → Forest green |
| **Frustration** | Holonomy frustration FrustIndex = ‖d_AI(Hol(γ), Id)‖ per residue | Cyan → Magenta |
| **Attribution** | Phase 4 integrated-gradient saliency (kinetics / EC / selectivity task) | White → Orange |

Attribution also draws:
- **Yellow sticks** — M-CSA catalytic residues
- **CGO tubes** — allosteric pathways identified by attention flow

---

## Installation

### Option 1 — PyMOL Plugin Manager (recommended)

1. In PyMOL: **Plugin → Plugin Manager → Install New Plugin**
2. Select the `tope_pymol_plugin/` directory
3. Restart PyMOL

### Option 2 — Startup script

Add to `~/.pymolrc`:

```python
run /path/to/tope_pymol_plugin/__init__.py
```

### Option 3 — Startup directory

Copy `tope_pymol_plugin/` into:
- `~/.pymol/startup/` (Linux / macOS)
- `%APPDATA%\PyMOL\startup\` (Windows)

---

## Requirements

```
PyMOL >= 2.5   (open-source or incentive builds both work)
Python >= 3.9  (same interpreter as PyMOL)
numpy
```

For **live inference** (optional — load-mode works without this):

```
tope            # the ToPE package itself
torch >= 2.0
torch-geometric
rdkit
geoopt
```

Install ToPE in the same environment as PyMOL:

```bash
conda activate pymol-env
pip install -e /path/to/tope
```

---

## Usage

### GUI

```
Plugin → ToPE Visualizer
```

Opens a docked Qt panel with tabs for each layer, a device selector (cpu / cuda / mps),
a log window, and a "Load File" tab for pre-computed outputs.

### Command line

```
tope_voip        [selection [output.json]]
tope_attention   [selection [SMILES [output.json]]]
tope_frustration [selection [output.json]]
tope_attribution [selection [task [output.json]]]
tope_all         [selection [SMILES [task [output.json]]]]
tope_load        <file> [layer] [selection]
tope_kinetics    [selection]
tope_help
```

#### Examples

```python
# Colour DHFR by VOIP and save output
fetch 1RX2
tope_voip all /data/dhfr_tope.json

# Highlight adenylate kinase residues that attend to ATP
fetch 4AKE
tope_attention chain A "C1=NC(=NC2=C1N=CN=C2N)N" /data/4ake_atp.json

# View holonomy-frustrated residues in nitrogenase
fetch 1N2C
tope_frustration all

# Load pre-computed HDCR output and switch between layers
tope_load /data/6cfw_tope.json voip all
tope_load /data/6cfw_tope.json attention all
tope_load /data/6cfw_tope.json frustration all
tope_load /data/6cfw_tope.json attribution all
```

---

## Output file schema

All commands accept an optional output path that saves a `.json` file:

```json
{
  "pdb_id": "4AKE",
  "n_residues": 214,

  "voip": {
    "residue_scores": [8.2, ...],
    "raw_voip": [8.2, ...],
    "thresholds": {"high": 15.0, "low": 10.0}
  },

  "attention": {
    "residue_scores": [0.1, ...],
    "substrate_smiles": "C1=NC...",
    "per_head": [[...], ...]
  },

  "frustration": {
    "residue_scores": [0.3, ...],
    "holonomy_norm": 0.42,
    "frustrated_residues": [12, 47, 88]
  },

  "attribution": {
    "residue_scores": [0.01, ...],
    "zone_importance": {"zone1": 0.6, "zone2": 0.3, "zone3": 0.1},
    "pathways": [[0, 12, 47, 88]],
    "catalytic_residues": [47],
    "task": "kinetics",
    "mcsa_overlap": 0.85
  },

  "kinetics": {
    "log_kcat": 3.2,
    "log_km": -5.1,
    "log_kcat_km": 8.3
  }
}
```

NPZ format is also supported with the same layer keys prefixed:
`voip_residue_scores`, `attention_residue_scores`, etc.

---

## Architecture

```
tope_pymol_plugin/
├── __init__.py          # PyMOL plugin entry point
└── tope_viz/
    ├── __init__.py
    ├── commands.py      # CLI: tope_voip, tope_attention, …
    ├── coloring.py      # Score → PyMOL spectrum mapping, CGO pathways
    ├── gui.py           # Qt5 panel (tabs, worker thread, log)
    └── runner.py        # ToPE inference bridge + JSON/NPZ loader
```

---

## Running the tests

No PyMOL session needed:

```bash
cd tope_pymol_plugin
python test_tope_plugin.py
```

---

## Validation targets

The canonical test structures for ToPE visualisation:

| PDB | System | Expected signal |
|---|---|---|
| `1RX2` | DHFR G121V | High frustration at G121 / Zone-3 attribution |
| `4AKE` / `1AKE` | Adenylate kinase open/closed | Attention shift between conformations |
| `1N2C` | Nitrogenase | VOIP gradient toward Fe-Mo cofactor |
| `6CFW` | HDCR | High frustration at [NiFeSe] cluster |
