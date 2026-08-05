import torch

from tope.models.bidirectional_physics import (
    BidirectionalPhysicsConfig, BidirectionalPhysicsToPE, mutation_epistasis,
)
from tope.training.bidirectional_losses import BidirectionalPhysicsLoss


def _graph(cfg):
    return {
        "node_features": torch.randn(6, cfg.node_feature_dim),
        "node_type": torch.tensor([0, 0, 1, 2, 3, 4]),
        "edge_index": torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        "edge_features": torch.randn(6, cfg.edge_feature_dim),
        "edge_type": torch.tensor([0, 1, 2, 3, 4, 5]),
        "batch": torch.zeros(6, dtype=torch.long),
        "temperature": torch.tensor([310.0]),
        "active_site_mask": torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
        "interface_mask": torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        "pathway_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
    }


def test_bidirectional_model_and_physics_are_differentiable():
    cfg = BidirectionalPhysicsConfig(node_feature_dim=12, edge_feature_dim=5,
                                     hidden_dim=32, latent_dim=8, n_message_layers=2,
                                     phenotype_dim=10)
    model = BidirectionalPhysicsToPE(cfg, topology_target_dim=7)
    forward = model.forward_topology(_graph(cfg), sample=False)
    assert forward["topology"]["mean"].shape == (1, 7)
    phenotype = torch.randn(1, 10)
    phenotype[0, 3] = float("nan")
    inverse = model.infer_topology(phenotype, sample=False)
    physics = model.integrate(forward, torch.tensor([290.0, 300.0, 310.0]),
                              torch.tensor([0.0, 10.0, 20.0]), torch.tensor([1.0]),
                              torch.tensor([0.01]))
    assert physics["rate"].shape == (1, 3)
    loss = physics["productivity"].sum() + inverse["topology"]["mean"].sum()
    loss.backward()
    assert model.topology_encoder.node_input.weight.grad is not None


def test_masked_multifidelity_loss_and_epistasis():
    cfg = BidirectionalPhysicsConfig(node_feature_dim=8, edge_feature_dim=4,
                                     hidden_dim=32, latent_dim=8, phenotype_dim=10)
    model = BidirectionalPhysicsToPE(cfg, topology_target_dim=4)
    output = model.forward_topology(_graph(cfg), sample=False)
    inverse = model.infer_topology(torch.randn(1, 10), sample=False)
    targets = {
        "thermal": torch.full((1, 5), float("nan")),
        "topology": torch.randn(1, 4),
        "topology_mask": torch.tensor([[1, 1, 0, 1]], dtype=torch.bool),
    }
    losses = BidirectionalPhysicsLoss()(output, targets, inverse)
    assert torch.isfinite(losses["total"])
    assert torch.equal(mutation_epistasis(torch.tensor(9.), torch.tensor(5.),
                                          torch.tensor(6.), torch.tensor(3.)), torch.tensor(1.))
