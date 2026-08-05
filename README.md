# ToPE: Topological Pattern Recognition for Enzymes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A deep learning architecture for enzyme function prediction using persistent homology and geometric deep learning. Bridges Ioffe's 1983 criterion-space framework with modern topological deep learning.

## Overview

ToPE combines topological data analysis with SE(3)-equivariant neural networks to predict enzyme properties:

- **EC Classification**: Enzyme Commission number prediction (800+ classes)
- **Substrate Selectivity**: Multi-label substrate/product prediction
- **Kinetic Parameters**: k_cat, K_M, and k_cat/K_M regression
- **Mutation Effects**: DDG and activity change prediction
- **Cooperativity**: Hill coefficient prediction for multi-subunit enzymes

## Architecture

```
Enzyme Structure (PDB)
         |
         v
+-----------------------------+
|   Multi-Parameter TTN       |  <- GPU-native persistent homology
|   (Spatial + Electronic)    |     60,000x memory reduction
+-----------------------------+
         |
         v
+-----------------------------+
|   TCPNet Message Passing    |  <- SE(3)-equivariant GNN
|   (Gradient Checkpointed)   |     10x memory reduction
+-----------------------------+
         |
         v
+-----------------------------+
|   Multi-Task Heads          |  <- EC, kinetics, mutation
|   + p-Laplacian Analysis    |     mechanistic interpretability
+-----------------------------+
```

## Installation

```bash
# From source
git clone https://github.com/ad3rin0la/pattern.git
cd pattern
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Development install
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- CUDA >= 11.8 (recommended for GPU training)

## Quick Start

### Data Curation

```python
from tope.data import CurationPipeline, CurationConfig

config = CurationConfig(
    output_dir="./data",
    active_site_radius=8.0,
    include_kinetics=True,
)

pipeline = CurationPipeline(config)
dataset = pipeline.run()
```

### Training

```python
from tope.models import ToPEModel, ToPEConfig
from tope.training import ToPETrainer

# Configure model
config = ToPEConfig(
    hidden_dim=256,
    n_layers=6,
    use_persistent_homology=True,
)

# Create model and train
model = ToPEModel(config)
trainer = ToPETrainer(model, train_loader, val_loader)
trainer.fit(n_epochs=100)
```

### Memory-Optimized Training (RTX 3060 12GB)

```python
from tope.models import MemoryOptimizedToPE, MemoryOptimizedTrainer

# 45x memory reduction: fits on consumer GPU
model = MemoryOptimizedToPE()
trainer = MemoryOptimizedTrainer(model, train_loader)
trainer.fit(n_epochs=100)
```

### Multi-Subunit Enzymes (Cooperativity)

```python
from tope.models import MultiSubunitToPEModel, MultiSubunitToPEConfig

config = MultiSubunitToPEConfig(
    max_subunits=4,
    predict_hill_coefficient=True,
)

model = MultiSubunitToPEModel(config)
predictions = model(enzyme_pcc)
hill_coeff = predictions["hill_coefficient"]
```

### Mechanistic Interpretability

```python
from tope.models import CompletePToPEModel, MechanisticAnalyzer

model = CompletePToPEModel()
analysis = model.get_mechanistic_analysis(enzyme_pcc)

print(analysis.rate_limiting_step)
print(analysis.reaction_channels)
```

### Multi-Parameter Filtration

```python
from tope.topology import MultiParameterTTN, TTNPHConfig

cfg = TTNPHConfig(
    spatial_zones=[8.0, 20.0],      # Distance zones (A)
    voip_thresholds=[10.0, 15.0],   # VOIP ranges (eV)
    k_eigenvalues=16,
)

ttn = MultiParameterTTN(cfg)
features = ttn(coords, batch, sheaf_features)
```

### CLI Tools

```bash
# Curate enzyme dataset
tope-curate --output-dir ./data --max-entries 100

# Train model
tope-train config.yaml --data-dir ./data --output-dir ./checkpoints

# Run predictions
tope-predict enzyme.pdb --checkpoint model.pt --task all

# Ingest TopEC dataset
tope-ingest-topec ./TopEC/data/csv --output-dir ./data
```

## Package Structure

```
tope/
├── __init__.py              # Top-level API
├── __version__.py           # Version info
├── data/                    # Data curation pipeline
│   ├── pipeline.py          # Main curation orchestrator
│   ├── pdb_client.py        # PDB structure fetching
│   ├── mcsa_client.py       # M-CSA active site data
│   ├── kinetics_client.py   # BRENDA/SABIO-RK kinetics
│   ├── active_site.py       # Active site extraction
│   ├── features.py          # Ioffe descriptors
│   ├── dataset.py           # PyTorch dataset builder
│   ├── topec_ingestion.py   # TopEC dataset ingestion
│   ├── config.py            # Configuration
│   └── cli.py               # CLI entry points
├── models/                  # Neural network architectures
│   ├── tope_model.py        # Main ToPE model
│   ├── tcpnet.py            # TCPNet message passing
│   ├── whole_protein_tcpnet.py  # Multi-scale TCPNet
│   ├── multi_scale_graph.py # Three-zone graph construction
│   ├── cross_attention.py   # Substrate-product attention
│   ├── task_heads.py        # EC, kinetics, mutation heads
│   ├── multi_subunit.py     # Phase 5: Quaternary structure
│   ├── p_laplacian.py       # Phase 6: Learnable p-Laplacian
│   ├── memory_optimized.py  # Memory-efficient architecture
│   └── cli.py               # Prediction CLI
├── topology/                # Topological encoding
│   ├── ttn_persistent_homology.py  # GPU-native TTN
│   ├── phonon_topology.py   # Hodge Laplacian ENM
│   └── transfer_pathways.py # Electron/proton transfer
├── training/                # Training utilities
│   ├── trainer.py           # Training loop + curriculum
│   ├── losses.py            # Multi-task loss functions
│   ├── evaluation.py        # Metrics and validation
│   └── cli.py               # Training CLI
├── attribution/             # Phase 4: Interpretability
│   └── attribution.py       # Multi-scale attribution & OOD
└── utils/                   # Shared utilities
    └── mcp_adapters.py      # AlphaFold, ChEMBL, PubChem
```

## Key Features

### Bidirectional physics-informed modeling

`BidirectionalPhysicsToPE` implements a probabilistic topology ↔ phenotype model
for mutation and temperature studies. It provides:

- typed residue/cofactor/ligand/water/subunit graphs with MD/QM edge features;
- temperature-conditioned global, interface, active-site and pathway pooling;
- topology and phenotype encoders that infer diagonal-Gaussian latent posteriors;
- uncertain thermal, catalytic, electronic and topology predictions;
- Arrhenius kinetics, thermal inactivation and integrated productivity;
- masked multi-fidelity, physics, KL and cycle-consistency losses.

```python
from tope.models import BidirectionalPhysicsToPE
from tope.training import BidirectionalPhysicsLoss

model = BidirectionalPhysicsToPE(topology_target_dim=16)
forward = model.forward_topology(temperature_dependent_graph)
inverse = model.infer_topology(phenotype_values, phenotype_mask)
losses = BidirectionalPhysicsLoss()(forward, targets, inverse)
```

The inverse decoder predicts distributions over topology descriptors or candidate
edge changes. It must not be interpreted as uniquely reconstructing a structure
from catalytic measurements.

### GPU-Native Persistent Homology (TTN)

Replaces CPU-bound `gudhi` with GPU-accelerated Tensor Tree Networks:

- **Memory**: ~50MB/batch vs ~500MB/protein (60,000x reduction)
- **Speed**: ~50ms/batch vs ~5s/protein (100x speedup)
- **Multi-parameter**: Simultaneous spatial + electronic filtration
- **Differentiable**: End-to-end gradient flow

### Learnable p-Laplacian

Connects topology to reaction mechanism via Eyring theory:

- **p -> 1**: Tunneling regime (quantum effects)
- **p = 2**: Standard diffusion (thermal activation)
- **p -> infinity**: Conformational gating (rate-limiting motion)

### Phase 4-6 Extensions

| Phase | Feature | Description |
|-------|---------|-------------|
| 4 | Attribution | 4D attribution (filtration x zone x residue x pathway) |
| 4 | OOD Validation | Stratified by sequence identity x mutation distance |
| 5 | Multi-Subunit | Quaternary structure with Hill coefficient prediction |
| 6 | p-Laplacian | Mechanistic interpretability via nodal domains |

## Data Sources

1. **M-CSA** (Mechanism and Catalytic Site Atlas): ~1,000 enzymes with catalytic annotations
2. **PDB** (Protein Data Bank): 200,000+ protein structures
3. **BRENDA / SABIO-RK**: ~23k kcat, ~41k Km measurements
4. **AlphaFold Database** (optional): 200M+ predicted structures
5. **TopEC**: Pre-curated enzyme classification dataset

## Citation

```bibtex
@software{tope2025,
  title={ToPE: Topological Pattern Recognition for Enzymes},
  author={Fasipe, Derin},
  year={2025},
  url={https://github.com/ad3rin0la/pattern}
}
```

## License

MIT License

## Contact

- **Issues**: https://github.com/ad3rin0la/pattern/issues
- **Email**: derin@anabaena.co
