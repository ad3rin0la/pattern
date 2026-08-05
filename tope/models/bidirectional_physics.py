"""Bidirectional, physics-informed latent-variable model for enzyme design.

This module implements the probabilistic model layer.  Structure/MD/QM data are
represented by a typed graph; a topology encoder maps it to a distribution over
latent topology, while a phenotype encoder provides the inverse posterior.  A
shared decoder and mechanistic heads make both directions comparable and permit
cycle-consistent training with partially observed labels.

The module deliberately does not prescribe a structure-preparation pipeline.
Callers can supply residue, cofactor, ligand, water and subunit nodes, and typed
edges derived from crystallography, MD, electronic structure or network analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]

# Stable IDs used by structure/MD preprocessing. Additional project-specific
# types can occupy the remaining configured embedding IDs.
NODE_TYPES = {
    "residue": 0, "cofactor_atom": 1, "homocitrate": 2, "substrate": 3,
    "water": 4, "subunit": 5, "intermediate": 6, "other": 7,
}
EDGE_TYPES = {
    "covalent": 0, "contact": 1, "hydrogen_bond": 2, "salt_bridge": 3,
    "hydrophobic": 4, "interface": 5, "coordination": 6, "electrostatic": 7,
    "correlated_motion": 8, "proton_pathway": 9, "electron_pathway": 10,
    "other": 11,
}
PHENOTYPE_FIELDS = (
    "log_kcat", "log_km", "tm", "log_kd", "residual_activity",
    "nh3_h2", "delta_q_fe", "r_nn", "nu_nn", "aggregation",
)


@dataclass
class BidirectionalPhysicsConfig:
    """Configuration for :class:`BidirectionalPhysicsToPE`."""

    node_feature_dim: int = 128
    edge_feature_dim: int = 16
    sequence_feature_dim: int = 0
    phenotype_dim: int = 10
    hidden_dim: int = 256
    latent_dim: int = 64
    n_node_types: int = 8
    n_edge_types: int = 16
    n_message_layers: int = 4
    n_temperature_frequencies: int = 8
    dropout: float = 0.1
    min_temperature: float = 250.0
    max_temperature: float = 400.0
    gas_constant: float = 8.314462618  # J mol^-1 K^-1


class DiagonalGaussian:
    """Small reparameterised diagonal-Gaussian helper."""

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    @staticmethod
    def kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(dim=-1)


class TemperatureEncoding(nn.Module):
    """Continuous temperature encoding, normalized to the configured range."""

    def __init__(self, cfg: BidirectionalPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(cfg.n_temperature_frequencies).float()
        )

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.cfg.n_temperature_frequencies

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        t = (temperature.float() - self.cfg.min_temperature) / (
            self.cfg.max_temperature - self.cfg.min_temperature
        )
        phase = math.pi * t.unsqueeze(-1) * self.frequencies
        return torch.cat([t.unsqueeze(-1), phase.sin(), phase.cos()], dim=-1)


def _pool_mean(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = x.new_zeros((n_graphs, x.size(-1)))
    out.index_add_(0, batch, x)
    counts = x.new_zeros(n_graphs)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / counts.clamp_min(1).unsqueeze(-1)


def _masked_pool(
    x: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    return _pool_mean(x * mask.float().unsqueeze(-1), batch, n_graphs) / (
        _pool_mean(mask.float().unsqueeze(-1), batch, n_graphs).clamp_min(1e-6)
    )


class TypedTopologyEncoder(nn.Module):
    """Message-passing encoder for temperature-dependent heterogeneous graphs.

    Expected graph keys are ``node_features``, ``edge_index``, ``edge_features``,
    ``node_type`` and ``edge_type``.  Optional keys are ``batch``, ``temperature``,
    ``sequence_features`` and boolean scale masks named ``global_mask``,
    ``interface_mask``, ``active_site_mask`` and ``pathway_mask``.
    """

    SCALE_NAMES = ("global", "interface", "active_site", "pathway")

    def __init__(self, cfg: BidirectionalPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.temperature = TemperatureEncoding(cfg)
        self.node_type_embedding = nn.Embedding(cfg.n_node_types, h // 4)
        self.edge_type_embedding = nn.Embedding(cfg.n_edge_types, h // 4)
        self.node_input = nn.Linear(cfg.node_feature_dim + h // 4, h)
        edge_in = cfg.edge_feature_dim + h // 4 + self.temperature.output_dim
        self.edge_input = nn.Linear(edge_in, h)
        self.messages = nn.ModuleList(
            [nn.Sequential(nn.Linear(3 * h, h), nn.SiLU(), nn.Linear(h, h))
             for _ in range(cfg.n_message_layers)]
        )
        self.updates = nn.ModuleList(
            [nn.GRUCell(h, h) for _ in range(cfg.n_message_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(cfg.n_message_layers)])
        pooled_dim = 4 * h + cfg.sequence_feature_dim + self.temperature.output_dim
        self.posterior = nn.Sequential(
            nn.Linear(pooled_dim, h), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, 2 * cfg.latent_dim),
        )

    def forward(self, graph: Mapping[str, torch.Tensor]) -> TensorDict:
        node = graph["node_features"]
        edge_index = graph["edge_index"].long()
        edge_features = graph["edge_features"]
        node_type = graph["node_type"].long()
        edge_type = graph["edge_type"].long()
        batch = graph.get("batch")
        if batch is None:
            batch = torch.zeros(node.size(0), dtype=torch.long, device=node.device)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        temperature = graph.get("temperature")
        if temperature is None:
            temperature = node.new_full((n_graphs,), 298.15)
        temperature = temperature.reshape(n_graphs)
        t_graph = self.temperature(temperature)

        h = self.node_input(torch.cat([node, self.node_type_embedding(node_type)], -1))
        src, dst = edge_index
        edge_batch = batch[src]
        e = self.edge_input(torch.cat([
            edge_features, self.edge_type_embedding(edge_type), t_graph[edge_batch]
        ], -1))
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            m = message(torch.cat([h[src], h[dst], e], dim=-1))
            aggregate = h.new_zeros(h.shape)
            aggregate.index_add_(0, dst, m)
            degree = h.new_zeros(h.size(0))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
            h = norm(update(aggregate, h))

        pooled = []
        for scale in self.SCALE_NAMES:
            mask = graph.get(f"{scale}_mask")
            if scale == "global" or mask is None:
                pooled.append(_pool_mean(h, batch, n_graphs))
            else:
                pooled.append(_masked_pool(h, batch, mask.bool(), n_graphs))
        sequence = graph.get("sequence_features")
        if self.cfg.sequence_feature_dim:
            if sequence is None:
                sequence = node.new_zeros((n_graphs, self.cfg.sequence_feature_dim))
            pooled.append(sequence)
        pooled.append(t_graph)
        context = torch.cat(pooled, dim=-1)
        mean, logvar = self.posterior(context).chunk(2, dim=-1)
        return {"mean": mean, "logvar": logvar.clamp(-12.0, 8.0),
                "node_embeddings": h, "scale_embedding": context}


class PhenotypeEncoder(nn.Module):
    """Infer a topology posterior from partially observed phenotypes."""

    def __init__(self, cfg: BidirectionalPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        self.temperature = TemperatureEncoding(cfg)
        input_dim = 2 * cfg.phenotype_dim + self.temperature.output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.latent_dim),
        )

    def forward(
        self, phenotype: torch.Tensor, mask: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        if mask is None:
            mask = torch.isfinite(phenotype)
        values = torch.nan_to_num(phenotype) * mask.float()
        if temperature is None:
            temperature = phenotype.new_full((phenotype.size(0),), 298.15)
        encoded = torch.cat([values, mask.float(), self.temperature(temperature)], dim=-1)
        mean, logvar = self.network(encoded).chunk(2, dim=-1)
        return {"mean": mean, "logvar": logvar.clamp(-12.0, 8.0)}


class TopologyDecoder(nn.Module):
    """Decode probabilistic mutation-induced topology descriptors.

    ``decode_edges`` supports candidate-edge reconstruction.  Candidate edge
    features can encode edge type, baseline occupancy, distance or orientation.
    The returned mean/log-variance represent changes rather than absolute structure.
    """

    def __init__(self, cfg: BidirectionalPhysicsConfig, topology_dim: int):
        super().__init__()
        h = cfg.hidden_dim
        self.summary = nn.Sequential(nn.Linear(cfg.latent_dim, h), nn.SiLU(),
                                     nn.Linear(h, 2 * topology_dim))
        self.edge_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.edge_feature_dim, h), nn.SiLU(),
            nn.Linear(h, 2),
        )

    def forward(self, z: torch.Tensor) -> TensorDict:
        mean, logvar = self.summary(z).chunk(2, dim=-1)
        return {"mean": mean, "logvar": logvar.clamp(-12.0, 8.0)}

    def decode_edges(
        self, z: torch.Tensor, candidate_features: torch.Tensor,
        candidate_batch: torch.Tensor,
    ) -> TensorDict:
        raw = self.edge_decoder(torch.cat([z[candidate_batch], candidate_features], -1))
        mean, logvar = raw.chunk(2, dim=-1)
        return {"mean": mean.squeeze(-1), "logvar": logvar.squeeze(-1).clamp(-12, 8)}


class MechanisticHeads(nn.Module):
    """Predict interpretable thermal, catalytic and electronic parameters."""

    THERMAL_NAMES = ("tm", "ddg_fold", "log_kd_ref", "delta_cp", "aggregation_logit")
    CATALYTIC_NAMES = (
        "log_kcat_ref", "log_km_ref", "ea", "delta_q_fe", "r_nn", "nu_nn", "nh3_h2_logit"
    )

    def __init__(self, cfg: BidirectionalPhysicsConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.shared = nn.Sequential(nn.Linear(cfg.latent_dim, h), nn.SiLU(), nn.Dropout(cfg.dropout))
        self.thermal = nn.Linear(h, 2 * len(self.THERMAL_NAMES))
        self.catalytic = nn.Linear(h, 2 * len(self.CATALYTIC_NAMES))

    @staticmethod
    def _distribution(raw: torch.Tensor) -> TensorDict:
        mean, logvar = raw.chunk(2, dim=-1)
        return {"mean": mean, "logvar": logvar.clamp(-12.0, 8.0)}

    def forward(self, z: torch.Tensor) -> Dict[str, TensorDict]:
        h = self.shared(z)
        thermal = self._distribution(self.thermal(h))
        catalytic = self._distribution(self.catalytic(h))
        # Enforce physical domains without hiding raw predictive uncertainty.
        thermal["raw_mean"] = thermal["mean"]
        thermal["mean"] = torch.stack([
            273.15 + 100.0 * torch.sigmoid(thermal["mean"][:, 0]),
            thermal["mean"][:, 1], thermal["mean"][:, 2], thermal["mean"][:, 3],
            thermal["mean"][:, 4],
        ], dim=-1)
        catalytic["raw_mean"] = catalytic["mean"]
        catalytic["mean"] = torch.stack([
            catalytic["mean"][:, 0], catalytic["mean"][:, 1],
            1_000.0 + 199_000.0 * torch.sigmoid(catalytic["mean"][:, 2]),
            catalytic["mean"][:, 3],
            0.8 + 2.2 * torch.sigmoid(catalytic["mean"][:, 4]),
            F.softplus(catalytic["mean"][:, 5]), catalytic["mean"][:, 6],
        ], dim=-1)
        thermal["physical_mean"] = thermal["mean"]
        catalytic["physical_mean"] = catalytic["mean"]
        return {"thermal": thermal, "catalytic": catalytic}


class PhysicsIntegrator(nn.Module):
    """Differentiable Arrhenius/inactivation/productivity integration layer."""

    def __init__(self, cfg: BidirectionalPhysicsConfig):
        super().__init__()
        self.cfg = cfg
        self.log_kd_ea = nn.Parameter(torch.tensor(10.0))  # softplus → kJ/mol

    def forward(
        self, thermal: torch.Tensor, catalytic: torch.Tensor,
        temperatures: torch.Tensor, times: torch.Tensor,
        substrate_concentration: torch.Tensor, enzyme_concentration: torch.Tensor,
    ) -> TensorDict:
        """Evaluate rates on a broadcastable ``(B, P)`` temperature/time grid."""
        t = temperatures
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(thermal.size(0), -1)
        time = times
        if time.dim() == 1:
            time = time.unsqueeze(0).expand_as(t)
        t_ref = 298.15
        log_kcat_ref, log_km_ref, ea = catalytic[:, 0:1], catalytic[:, 1:2], catalytic[:, 2:3]
        log_kd_ref = thermal[:, 2:3]
        log_kcat = log_kcat_ref + ea / self.cfg.gas_constant * (1.0 / t_ref - 1.0 / t)
        kd_ea = 1000.0 * F.softplus(self.log_kd_ea)
        log_kd = log_kd_ref + kd_ea / self.cfg.gas_constant * (1.0 / t_ref - 1.0 / t)
        kcat, km, kd = log_kcat.exp(), log_km_ref.exp(), log_kd.exp()
        active_fraction = torch.exp(-kd * time)
        substrate = torch.as_tensor(substrate_concentration, device=t.device, dtype=t.dtype)
        enzyme = torch.as_tensor(enzyme_concentration, device=t.device, dtype=t.dtype)
        while substrate.dim() < 2:
            substrate = substrate.unsqueeze(-1)
        while enzyme.dim() < 2:
            enzyme = enzyme.unsqueeze(-1)
        rate = enzyme * active_fraction * kcat * substrate / (km + substrate)
        productivity = torch.trapezoid(rate, time, dim=-1) if time.size(-1) > 1 else rate[..., 0] * time[..., 0]
        return {"log_kcat": log_kcat, "log_kd": log_kd, "active_fraction": active_fraction,
                "rate": rate, "productivity": productivity}


class BidirectionalPhysicsToPE(nn.Module):
    """Complete topology ↔ phenotype probabilistic framework."""

    def __init__(self, cfg: Optional[BidirectionalPhysicsConfig] = None,
                 topology_target_dim: int = 16):
        super().__init__()
        self.cfg = cfg or BidirectionalPhysicsConfig()
        self.topology_encoder = TypedTopologyEncoder(self.cfg)
        self.phenotype_encoder = PhenotypeEncoder(self.cfg)
        self.topology_decoder = TopologyDecoder(self.cfg, topology_target_dim)
        self.mechanistic_heads = MechanisticHeads(self.cfg)
        self.physics = PhysicsIntegrator(self.cfg)

    def _decode(self, posterior: TensorDict, sample: bool) -> TensorDict:
        z = DiagonalGaussian.sample(posterior["mean"], posterior["logvar"], sample)
        mechanisms = self.mechanistic_heads(z)
        return {"z": z, "posterior": posterior, "topology": self.topology_decoder(z),
                **mechanisms}

    def forward_topology(self, graph: Mapping[str, torch.Tensor], sample: bool = True) -> TensorDict:
        posterior = self.topology_encoder(graph)
        output = self._decode(posterior, sample)
        output["node_embeddings"] = posterior["node_embeddings"]
        return output

    def infer_topology(
        self, phenotype: torch.Tensor, phenotype_mask: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None, sample: bool = True,
    ) -> TensorDict:
        return self._decode(self.phenotype_encoder(phenotype, phenotype_mask, temperature), sample)

    def integrate(self, output: TensorDict, temperatures: torch.Tensor,
                  times: torch.Tensor, substrate_concentration: torch.Tensor,
                  enzyme_concentration: torch.Tensor) -> TensorDict:
        return self.physics(output["thermal"]["physical_mean"],
                            output["catalytic"]["physical_mean"], temperatures, times,
                            substrate_concentration, enzyme_concentration)


def mutation_epistasis(
    double_mutant: torch.Tensor, mutation_a: torch.Tensor, mutation_b: torch.Tensor,
    wild_type: torch.Tensor,
) -> torch.Tensor:
    """Return ``J_AB - J_A - J_B + J_WT`` for any broadcastable output tensors."""
    return double_mutant - mutation_a - mutation_b + wild_type
