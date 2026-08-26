"""Multirank geometry-aware electronic fingerprint regression tests."""

import torch

from tope.quantum.electronic_complex import (
    AdaptiveElectronicReadout,
    ElectronicComplexConfig,
    ElectronicComplexEncoder,
    GeometryElectronicFeedback,
    hierarchical_candidate_indices,
)


def _incidence(groups):
    pairs = [(child, parent) for parent, children in enumerate(groups) for child in children]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _inputs(spectrum_dim=4):
    coords = torch.tensor([
        [0.0, 0.0, 0.0], [1.2, 0.0, 0.0],
        [2.0, 0.8, 0.1], [3.1, 1.0, 0.7],
    ])
    scalar = torch.tensor([
        [1.0, 0.2, 0.0, 0.0], [0.8, 0.3, 0.1, 0.0],
        [0.1, 0.7, 0.3, 0.1], [0.0, 0.2, 0.8, 0.4],
    ])
    charge = torch.tensor([0.4, -0.2, 0.3, -0.5])
    vector = scalar[..., None] * torch.tensor([1.0, 0.2, -0.1])
    quadrupole = torch.zeros(4, spectrum_dim, 3, 3)
    quadrupole[..., 0, 0] = scalar
    quadrupole[..., 1, 1] = -0.5 * scalar
    quadrupole[..., 2, 2] = -0.5 * scalar
    return coords, scalar, charge, vector, quadrupole


def test_structured_fingerprint_preserves_ranks_and_extensive_moments():
    torch.manual_seed(2)
    cfg = ElectronicComplexConfig(spectrum_dim=4, hidden_dim=8, max_ranks=5, dropout=0.0)
    encoder = ElectronicComplexEncoder(cfg)
    coords, scalar, charge, vector, quadrupole = _inputs()
    atom_to_residue = _incidence([[0, 1], [2, 3]])
    residue_to_region = _incidence([[0, 1]])
    region_to_molecule = _incidence([[0]])
    fingerprint = encoder(
        coords, scalar, charge,
        [atom_to_residue, residue_to_region, region_to_molecule],
        vector_spectrum=vector, quadrupole_spectrum=quadrupole,
        rank_observables=[
            torch.tensor([
                [1.1, 0.8, -0.3, 0.1, 0.0, 0.0, 0.2, -0.1],
                [0.9, 0.5, -0.2, 0.0, 0.7, 0.4, 0.0, 0.3],
            ]),
            None,
            None,
        ],
        rank_names=["atom", "residue", "region", "molecule"],
    )
    assert [c.rank_name for c in fingerprint.cochains] == [
        "atom", "residue", "region", "molecule"
    ]
    assert [c.scalar.size(0) for c in fingerprint.cochains] == [4, 2, 1, 1]
    torch.testing.assert_close(fingerprint.cochains[-1].charge, charge.sum().view(1))
    torch.testing.assert_close(
        fingerprint.cochains[-1].spectral_density,
        scalar.sum(dim=0, keepdim=True),
    )
    assert fingerprint.cochains[1].vector.shape == (2, 4, 3)
    assert fingerprint.cochains[1].quadrupole.shape == (2, 4, 3, 3)
    assert fingerprint.cochains[1].observables.shape == (2, 8)
    assert fingerprint.cochains[1].observables[0, 0] == 1.1
    assert torch.all((fingerprint.cochains[1].coherence >= 0)
                     & (fingerprint.cochains[1].coherence <= 1))


def test_local_frame_channels_are_rotation_invariant():
    torch.manual_seed(3)
    cfg = ElectronicComplexConfig(spectrum_dim=4, hidden_dim=8, max_ranks=2, dropout=0.0)
    encoder = ElectronicComplexEncoder(cfg).eval()
    coords, scalar, charge, vector, quadrupole = _inputs()
    incidence = _incidence([[0, 1, 2, 3]])
    first = encoder(
        coords, scalar, charge, [incidence],
        vector_spectrum=vector, quadrupole_spectrum=quadrupole,
    ).cochains[-1]
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    ])
    rotated_vector = torch.einsum("ij,nsj->nsi", rotation, vector)
    rotated_quadrupole = torch.einsum(
        "ia,nsab,jb->nsij", rotation, quadrupole, rotation
    )
    second = encoder(
        coords @ rotation.T, scalar, charge, [incidence],
        vector_spectrum=rotated_vector,
        quadrupole_spectrum=rotated_quadrupole,
    ).cochains[-1]
    torch.testing.assert_close(first.scalar, second.scalar, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first.vector, second.vector, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first.quadrupole, second.quadrupole, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(first.coherence, second.coherence, atol=2e-5, rtol=2e-5)


def test_learned_domains_and_adaptive_resolution_are_first_class_ranks():
    torch.manual_seed(5)
    cfg = ElectronicComplexConfig(spectrum_dim=4, hidden_dim=8, max_ranks=4, dropout=0.0)
    encoder = ElectronicComplexEncoder(cfg)
    coords, scalar, charge, vector, quadrupole = _inputs()
    atom_to_residue = _incidence([[0, 1], [2, 3]])
    fingerprint = encoder(
        coords, scalar, charge, [atom_to_residue],
        vector_spectrum=vector, quadrupole_spectrum=quadrupole,
        rank_names=["atom", "residue"],
    )
    q = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    fingerprint = encoder.append_soft_rank(fingerprint, q)
    assert fingerprint.cochains[-1].rank_name == "learned_domain"
    assert fingerprint.cochains[-1].scalar.shape == (2, 4)

    readout = AdaptiveElectronicReadout(cfg)
    queries = torch.tensor([
        [0.1, 0.0, 0.0], [2.5, 1.0, 0.2], [8.0, 8.0, 8.0]
    ])
    candidates = hierarchical_candidate_indices(
        queries, fingerprint, top_coarse=1, max_children=2
    )
    context, rank_weights, cell_weights = readout(
        torch.randn(3, 8), queries, fingerprint, candidate_indices=candidates,
    )
    assert context.shape == (3, 8)
    assert rank_weights.shape == (3, 3)
    torch.testing.assert_close(rank_weights.sum(-1), torch.ones(3))
    assert len(cell_weights) == 3
    assert all(indices.size(1) <= 2 for indices in candidates)


def test_geometry_electronic_feedback_produces_finite_forces():
    torch.manual_seed(8)
    cfg = ElectronicComplexConfig(spectrum_dim=4, hidden_dim=8, max_ranks=2, dropout=0.0)
    encoder = ElectronicComplexEncoder(cfg)
    feedback = GeometryElectronicFeedback(encoder)
    coords, scalar, charge, vector, quadrupole = _inputs()
    energy, forces, fingerprint = feedback.energy_and_forces(
        coords=coords,
        scalar_spectrum=scalar,
        charge=charge,
        incidences=[_incidence([[0, 1, 2, 3]])],
        vector_spectrum=vector,
        quadrupole_spectrum=quadrupole,
    )
    assert energy.ndim == 0
    assert forces.shape == coords.shape
    assert torch.isfinite(energy) and torch.isfinite(forces).all()
    assert len(fingerprint.cochains) == 2
