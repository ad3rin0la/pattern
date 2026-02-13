"""
Substrate–Product Bipartite Cross-Attention
============================================

Encodes substrate and product molecular graphs with a lightweight GNN,
then performs multi-head cross-attention between the enzyme active-site
representation (from EnzymeTCPNet) and each small-molecule graph.

The enzyme-to-substrate and enzyme-to-product attention maps provide the
basis for selectivity prediction and mechanistic attribution.

References
----------
Vaswani et al., Attention Is All You Need, NeurIPS 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class CrossAttentionConfig:
    """Hyper-parameters for substrate–product cross-attention."""

    mol_feat_dim: int = 64         # Input atom feature dim for small molecules
    mol_hidden_dim: int = 128      # GNN hidden dim for small molecules
    mol_n_layers: int = 3          # GNN message-passing layers
    enzyme_hidden_dim: int = 256   # Enzyme TCPNet hidden dim (must match)
    n_heads: int = 8               # Multi-head attention heads
    dropout: float = 0.1


# ── Lightweight molecule GNN ──────────────────────────────────────────────────

class MoleculeGNNLayer(nn.Module):
    """One GNN layer for small-molecule encoding (message-passing + update)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (N, H)
        edge_index : (2, E)

        Returns
        -------
        h_new : (N, H)
        """
        row, col = edge_index
        # Build messages from pairs
        msg_input = torch.cat([h[row], h[col]], dim=-1)  # (E, 2H)
        msgs = self.msg_mlp(msg_input)                   # (E, H)

        # Aggregate messages (scatter-add)
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msgs), msgs)

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_new = h + self.dropout(h_new)
        h_new = self.ln(h_new)
        return h_new


class MoleculeGNN(nn.Module):
    """Lightweight GNN encoder for substrate / product molecules."""

    def __init__(self, cfg: CrossAttentionConfig):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(cfg.mol_feat_dim, cfg.mol_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.mol_hidden_dim, cfg.mol_hidden_dim),
        )
        self.layers = nn.ModuleList([
            MoleculeGNNLayer(cfg.mol_hidden_dim, cfg.dropout)
            for _ in range(cfg.mol_n_layers)
        ])
        # Project to enzyme hidden dim for cross-attention compatibility
        self.proj = nn.Linear(cfg.mol_hidden_dim, cfg.enzyme_hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (M, mol_feat_dim)
        edge_index    : (2, E_mol)

        Returns
        -------
        h_mol : (M, enzyme_hidden_dim)
        """
        h = self.embed(node_features)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.proj(h)


# ── Multi-head cross-attention ────────────────────────────────────────────────

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: queries from one set, keys/values from another."""

    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_batch: Optional[torch.Tensor] = None,
        kv_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query      : (N_q, H)  — e.g. enzyme atom embeddings
        key_value  : (N_kv, H) — e.g. substrate atom embeddings
        query_batch : (N_q,)   — graph membership for queries (optional)
        kv_batch    : (N_kv,)  — graph membership for keys/values (optional)

        Returns
        -------
        attended : (N_q, H)
        attn_weights : (n_heads, N_q, N_kv) — for attribution
        """
        N_q = query.size(0)
        N_kv = key_value.size(0)

        Q = self.q_proj(query).view(N_q, self.n_heads, self.head_dim)
        K = self.k_proj(key_value).view(N_kv, self.n_heads, self.head_dim)
        V = self.v_proj(key_value).view(N_kv, self.n_heads, self.head_dim)

        # (n_heads, N_q, head_dim) x (n_heads, head_dim, N_kv) → (n_heads, N_q, N_kv)
        Q = Q.permute(1, 0, 2)  # (n_heads, N_q, head_dim)
        K = K.permute(1, 0, 2)  # (n_heads, N_kv, head_dim)
        V = V.permute(1, 0, 2)  # (n_heads, N_kv, head_dim)

        attn_logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (n_heads, N_q, N_kv)

        # Mask cross-graph interactions when batching multiple graphs
        if query_batch is not None and kv_batch is not None:
            # Build mask: query i can attend to kv j only if same graph
            mask = query_batch.unsqueeze(1) != kv_batch.unsqueeze(0)  # (N_q, N_kv)
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)  # (n_heads, N_q, N_kv)
        attn_weights = self.dropout(attn_weights)

        attended = torch.bmm(attn_weights, V)  # (n_heads, N_q, head_dim)
        attended = attended.permute(1, 0, 2).contiguous().view(N_q, self.hidden_dim)
        attended = self.out_proj(attended)

        # Residual + layer norm
        attended = self.ln(query + attended)

        return attended, attn_weights


# ── Full substrate–product cross-attention module ─────────────────────────────

class SubstrateProductCrossAttention(nn.Module):
    """Bipartite cross-attention between enzyme and substrate/product molecules.

    Produces:
    - Enzyme embeddings enriched with substrate context
    - Enzyme embeddings enriched with product context
    - Attention maps for mechanistic attribution
    """

    def __init__(self, cfg: Optional[CrossAttentionConfig] = None):
        super().__init__()
        self.cfg = cfg or CrossAttentionConfig()
        H = self.cfg.enzyme_hidden_dim

        # Substrate and product encoders
        self.substrate_gnn = MoleculeGNN(self.cfg)
        self.product_gnn = MoleculeGNN(self.cfg)

        # Cross-attention: enzyme queries, molecule keys/values
        self.enzyme_to_substrate = MultiHeadCrossAttention(
            H, self.cfg.n_heads, self.cfg.dropout
        )
        self.enzyme_to_product = MultiHeadCrossAttention(
            H, self.cfg.n_heads, self.cfg.dropout
        )

        # Fusion of substrate-attended and product-attended enzyme embeddings
        self.fusion = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )
        self.ln = nn.LayerNorm(H)

    def forward(
        self,
        h_enzyme: torch.Tensor,
        substrate_data: Dict[str, torch.Tensor],
        product_data: Dict[str, torch.Tensor],
        enzyme_batch: Optional[torch.Tensor] = None,
        substrate_batch: Optional[torch.Tensor] = None,
        product_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        h_enzyme : (N_enz, H)
            Per-atom enzyme embeddings from EnzymeTCPNet.
        substrate_data : dict with keys
            node_features : (M_sub, mol_feat_dim)
            edge_index    : (2, E_sub)
        product_data : dict with keys
            node_features : (M_prod, mol_feat_dim)
            edge_index    : (2, E_prod)
        enzyme_batch    : (N_enz,)  optional graph membership
        substrate_batch : (M_sub,)  optional graph membership
        product_batch   : (M_prod,) optional graph membership

        Returns
        -------
        h_fused : (N_enz, H)
            Enzyme embeddings enriched with substrate + product context.
        attn_maps : dict
            'substrate': (n_heads, N_enz, M_sub)
            'product':   (n_heads, N_enz, M_prod)
        """
        # Encode molecules
        h_sub = self.substrate_gnn(
            substrate_data["node_features"],
            substrate_data["edge_index"],
        )
        h_prod = self.product_gnn(
            product_data["node_features"],
            product_data["edge_index"],
        )

        # Cross-attend: enzyme ← substrate
        h_enz_sub, attn_sub = self.enzyme_to_substrate(
            h_enzyme, h_sub, enzyme_batch, substrate_batch
        )
        # Cross-attend: enzyme ← product
        h_enz_prod, attn_prod = self.enzyme_to_product(
            h_enzyme, h_prod, enzyme_batch, product_batch
        )

        # Fuse substrate-attended and product-attended representations
        h_fused = self.fusion(torch.cat([h_enz_sub, h_enz_prod], dim=-1))
        h_fused = self.ln(h_enzyme + h_fused)

        attn_maps = {
            "substrate": attn_sub,
            "product": attn_prod,
        }

        return h_fused, attn_maps
