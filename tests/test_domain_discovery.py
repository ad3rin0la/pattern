"""Label-free latent-domain discovery and specificity integration tests."""

import torch
import torch.nn as nn

from tope.data.molecule import MOL_EDGE_FEAT_DIM, MOL_FEAT_DIM, smiles_to_graph
from tope.models.domain_discovery import (
    DomainDiscoveryConfig,
    DomainDiscoveryHead,
    permutation_match,
)
from tope.models.tope_model import CompleteToPEConfig, CompleteToPEModel
from tope.training.losses import EnhancedMultiTaskLoss
from tope.training.domain_evaluation import evaluate_domains


def _edges(*pairs):
    directed = list(pairs) + [(b, a) for a, b in pairs]
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def _molecule(smiles):
    nodes, edges, edge_features = smiles_to_graph(smiles)
    return {
        "node_features": torch.tensor(nodes, dtype=torch.float32),
        "edge_index": torch.tensor(edges, dtype=torch.long),
        "edge_features": torch.tensor(edge_features, dtype=torch.float32),
        "batch": torch.zeros(nodes.shape[0], dtype=torch.long),
    }


def test_domain_head_produces_soft_incidence_and_all_losses():
    torch.manual_seed(4)
    cfg = DomainDiscoveryConfig(
        hidden_dim=8, sequence_feat_dim=5, max_domains=3, dropout=0.0
    )
    head = DomainDiscoveryHead(cfg)
    h = torch.randn(6, 8, requires_grad=True)
    sequence = torch.randn(6, 5)
    edge_index = _edges((0, 1), (1, 2), (3, 4), (4, 5))
    outputs = head(h, sequence_features=sequence)
    q = outputs["assignments"]
    assert q.shape == (6, 3)
    torch.testing.assert_close(q.sum(dim=-1), torch.ones(6))
    assert outputs["domain_embeddings"].shape == (1, 3, 8)

    losses = head.losses(
        outputs, h, edge_index,
        sequence_index=torch.arange(6),
        perturbed_assignments=q[:, [2, 0, 1]],
    )
    assert set(losses) == {
        "total", "cut", "continuity", "reconstruction", "contact",
        "consistency", "agreement", "complexity",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_permutation_matching_recovers_equivalent_domains():
    reference = torch.tensor([
        [1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.1, 0.9],
    ])
    permuted = reference[:, [2, 0, 1]]
    torch.testing.assert_close(permutation_match(reference, permuted), reference)
    metrics = evaluate_domains(
        reference, reference.argmax(-1),
        sequence_index=torch.arange(reference.size(0)),
        perturbed_assignments=permuted,
        tolerance=0,
    )
    assert metrics["boundary_f1"] == 1.0
    assert metrics["domain_count_error"] == 0.0
    assert metrics["perturbation_stability_mse"] == 0.0


class _DummyWholeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph):
        h = self.proj(graph["precomputed_residue_embeddings"])
        batch = graph.get("batch")
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = torch.stack([h[batch == i].mean(0) for i in batch.unique()])
        return pooled, h


def test_complete_model_routes_latent_domains_into_specificity():
    torch.manual_seed(7)
    cfg = CompleteToPEConfig(
        hidden_dim=8, n_heads=2, n_message_layers=1,
        zone1_feat_dim=8, zone2_feat_dim=8, zone3_feat_dim=8,
        ec_levels=[2, 2, 2, 2], head_hidden_dim=8,
        mol_feat_dim=MOL_FEAT_DIM, mol_hidden_dim=8, mol_n_layers=1,
        max_latent_domains=3, dropout=0.0,
        use_electronic_complex=True, electronic_spectrum_dim=4,
        electronic_hidden_dim=8, electronic_max_ranks=3,
    )
    model = CompleteToPEModel(cfg)
    model.encoder = _DummyWholeEncoder(8)
    graph = {
        "precomputed_residue_embeddings": torch.randn(6, 8),
        "edge_index": _edges((0, 1), (1, 2), (3, 4), (4, 5), (2, 3)),
        "zone_assignments": torch.tensor([0, 0, 1, 1, 2, 2]),
        "sequence_index": torch.arange(6),
        "chain_index": torch.zeros(6, dtype=torch.long),
        "batch": torch.zeros(6, dtype=torch.long),
    }
    batch = {
        "protein_graph": graph,
        "substrate": _molecule("CCO"),
        "product": _molecule("CC=O"),
        "compute_domain_losses": True,
        "electronic_complex": {
            "coords": torch.randn(6, 3),
            "scalar_spectrum": torch.rand(6, 4),
            "charge": torch.tensor([0.2, -0.1, 0.0, 0.1, -0.3, 0.1]),
            "incidences": [torch.stack([torch.arange(6), torch.arange(6)])],
            "rank_names": ["atom", "residue"],
            "residue_rank": 1,
        },
    }
    outputs = model(batch)
    assert outputs["latent_domains"]["assignments"].shape == (6, 3)
    assert outputs["domain_substrate_attention"].shape == (1, 3)
    assert outputs["log_efficiency"].shape == (1, 1)
    assert torch.isfinite(outputs["domain_losses"]["total"])
    assert outputs["electronic_fingerprint"].cochains[-1].rank_name == "learned_domain"
    assert outputs["electronic_fingerprint"].cochains[-1].embedding.shape == (3, 8)

    deltas = model.counterfactual_domain_specificity(batch)
    assert deltas.shape == (1, 3)
    assert torch.isfinite(deltas).all()


def test_enhanced_loss_includes_self_supervised_domain_objective():
    criterion = EnhancedMultiTaskLoss(domain_discovery_weight=2.0)
    domain_total = torch.tensor(1.25, requires_grad=True)
    predictions = {
        "ec_logits": [torch.zeros(1, 2) for _ in range(4)],
        "domain_losses": {"total": domain_total, "cut": domain_total / 2},
    }
    result = criterion(
        predictions, {},
        active_tasks={
            "ec": False, "selectivity": False, "kcat": False, "km": False,
            "mutation": False, "residue_importance": False,
            "domain_discovery": True,
        },
    )
    torch.testing.assert_close(result["total"], torch.tensor(2.5))
    result["total"].backward()
    torch.testing.assert_close(domain_total.grad, torch.tensor(2.0))
