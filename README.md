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

### Self-supervised latent domains

`CompleteToPEModel` learns a soft residue-to-domain incidence matrix directly
from TCPNet residue embeddings. Training uses normalized structural cut,
sequence continuity, domain-bottleneck reconstruction, contact reconstruction,
permutation-matched perturbation consistency, optional sequence/structure
agreement, and anti-collapse penalties. No CATH, ECOD, Pfam, or EC labels enter
the discovery head.

```python
from tope.models import CompleteToPEConfig, CompleteToPEModel

model = CompleteToPEModel(CompleteToPEConfig(
    use_domain_discovery=True,
    max_latent_domains=8,
    sequence_feat_dim=21,  # default residue identity; replace with richer embeddings
))

outputs = model(batch)
Q = outputs["latent_domains"]["assignments"]
domain_embeddings = outputs["latent_domains"]["domain_embeddings"]
domain_substrate_attention = outputs["domain_substrate_attention"]

# Delta_k(s) = y(enzyme, substrate) - y(enzyme without domain k, substrate)
domain_specificity = model.counterfactual_domain_specificity(batch)
```

Known domain databases are supported only through evaluation utilities such as
`tope.training.evaluate_domains`, which reports boundary overlap, domain-count
error, and perturbation stability.

### Multiresolution electronic cochains

`ElectronicComplexEncoder` keeps electronic structure as a cochain on every
complex rank instead of reducing all atoms to one global fingerprint. Scalar
spectra, directional orbital channels, quadrupolar channels, charge multipoles,
spectral multipoles, and coherence are lifted through sparse incidence maps
using geometry-conditioned weights and cell-local coordinate frames.

```python
from tope.quantum import ElectronicComplexConfig, ElectronicComplexEncoder

encoder = ElectronicComplexEncoder(ElectronicComplexConfig(
    spectrum_dim=64,
    hidden_dim=128,
    max_ranks=6,
))

fingerprint = encoder(
    coords=atom_coords,
    scalar_spectrum=atom_pdos,
    charge=mulliken_charge,
    vector_spectrum=directional_pdos,
    quadrupole_spectrum=orbital_quadrupoles,
    incidences=[B_atom_bond, B_bond_motif, B_motif_residue],
    rank_names=["atom", "bond", "motif", "residue"],
)
```

`AdaptiveElectronicReadout` selects both cells and resolution for a query.
`hierarchical_candidate_indices` first selects nearby coarse cells and then
descends through their incidence links, bounding the fine-scale candidate set
without constructing every query-atom pair.
`append_soft_rank` accepts the learned residue-to-domain matrix `Q`, allowing
electronic domain fingerprints to condition domain–substrate attention.
`GeometryElectronicFeedback.energy_and_forces` supplies a differentiable
geometry→electronic-state→energy→force interface; it is not a substitute for a
force-trained potential or molecular-dynamics validation.

### Relativistic Clifford holography

`tope.quantum.relativistic_holography` implements the relativistic Pattern
state as a Clifford-valued field on the positive-energy mass shell. Canonical
momenta are mapped exactly to half-rapidity Poincare coordinates, while a fixed
16-element Dirac basis types every holographic mode as scalar, four-vector,
antisymmetric tensor, axial vector, or pseudoscalar.

```python
from tope.quantum import (
    CliffordHologramLayer,
    coupled_boost,
    momentum_to_poincare,
    project_clifford_hologram,
)

u = momentum_to_poincare(momentum, mass=electron_mass, c=speed_of_light)
H = project_clifford_hologram(clifford_field, hyperbolic_modes, volume_weights)
H_next = CliffordHologramLayer()(H, interaction_H, operation="commutator")
boosted = coupled_boost(boost_u, u, spinor=psi, clifford=H)
```

The same gyrovector controls Mobius translation on the mass-shell ball and the
corresponding `Spin(1,3)` transformation of the spinor/Clifford state. The
gamma multiplication table is fixed; only typed response coefficients are
learned. `ElectronicCliffordHologram` connects existing multiresolution
`ElectronicCochain` objects without mislabeling their symmetric quadrupoles as
antisymmetric Clifford tensors. These components are structural inductive
biases and differentiable research primitives, not an X2C implementation or a
validated relativistic electronic-structure calculation.

For interactions between multiple momentum states,
`GyrotrigonometricInteractionKernel` adds rooted gyroangles, gyrocosines,
gyrosines, and exact Möbius gyrations:

```python
from tope.quantum import GyrotrigonometricInteractionKernel

kernel = GyrotrigonometricInteractionKernel(hidden_dim=64)
pair = kernel(u_i, u_j, clifford_i, clifford_j, root=domain_origin)
# pair.features.gyrocosine / gyrosine / gyration
# pair.composite_spinor == S(u_j) @ S(u_i)
```

The implementation distinguishes the half-rapidity factor
`1/sqrt(1-|u|^2)` used by a Dirac spinor boost from the physical Lorentz gamma
`(1+|u|^2)/(1-|u|^2)`. Einstein gamma-composition identities are evaluated
after mapping Poincare coordinates to Einstein velocity coordinates. This
prevents a gamma law for velocities from being incorrectly applied directly to
half-rapidity coordinates. Gyrocosine and gyrosine describe rooted gyroangles;
they do not replace the `cosh` and `sinh` generated by exponentiating a Lorentz
boost. Non-collinear boosts retain both their Thomas/Wigner gyration matrix and
their spatial-bivector Clifford channels.

### Hodge spectral diffusion and coupled torsions

`HodgeHeatDiffusion` generalizes circle heat diffusion to every signed boundary
rank of a molecular chain complex:

```python
from tope.models import HodgeHeatDiffusion

# B1: atoms x bonds, B2: bonds x motifs, with B1 @ B2 == 0
heat = HodgeHeatDiffusion()
spectra = heat.spectra([B1, B2])
atom_path = heat.diffuse(atom_field, spectra[0], times=[0.0, 0.1, 1.0])
```

Each modal coefficient is multiplied by `exp(-t * eigenvalue)`, so large
diffusion times retain global/harmonic organization while small times restore
finer structure. `ElectronicHodgeDiffusion` applies the same operation to
aligned fields in a `MultiresolutionElectronicFingerprint`. Local-frame vector
and tensor fields are rotated into a common global frame before diffusion by
default. Real and complex cochains are supported with real-valued diffusion
times.

`hodge_decomposition` separates any cochain into exact, coexact, and harmonic
parts, making conserved cycle/cocycle content directly inspectable rather than
only implicit in zero Laplacian eigenvalues.

`CoupledTorsionHarmonics` uses topology-selected torsion cells rather than the
complete torsion torus. It includes independent circle modes and pairwise sum
and difference modes such as `cos(theta_i - theta_j)`. `torus_heat_decay`
provides the exact `exp(-t * ||k||^2)` decay for arbitrary integer coupled-mode
wavevectors.

Hodge boundary operators are signed and must satisfy `B_k @ B_{k+1} == 0`.
Pooling incidences and learned residue-to-domain assignments `Q` remain useful
memberships, but are not silently treated as boundaries; a valid oriented cell
complex must be supplied for Hodge diffusion.

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
