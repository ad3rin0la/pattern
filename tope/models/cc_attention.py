"""CC-attention primitives for CCANN (CC-Attention Neural Network) formalism.

Implements the learned, normalized attention push-forward operations
described in Definitions 32, 33, and 36 of the CCANN formalism.
These primitives replace isotropic scatter_add / uniform-sum message
aggregation with attention-weighted, per-neighborhood normalized operations.

References
----------
Definition 32: Equal-rank CC-attention push-forward (coadjacency attention).
Definition 33: Unequal-rank bidirectional CC-attention block (incidence attention).
Definition 36: Inter-neighborhood attention merge node.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_scatter import scatter_softmax
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


# ── Internal utilities ────────────────────────────────────────────────────────

def _scatter_softmax_1d(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Numerically stable per-group softmax.  src, index: (E,)."""
    if HAS_TORCH_SCATTER:
        return scatter_softmax(src, index, dim=0)
    # Manual fallback (torch ≥ 2.0 scatter_reduce_ path)
    max_vals = src.new_full((dim_size,), float("-inf"))
    max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    shifted = src - max_vals[index]
    exp_s = shifted.exp()
    sum_e = src.new_zeros(dim_size)
    sum_e.scatter_add_(0, index, exp_s)
    return exp_s / (sum_e[index] + 1e-8)


# ── Equal-rank attention push-forward ─────────────────────────────────────────

class CCAttentionPushForward(nn.Module):
    """CC-attention push-forward for cells of equal rank (Definition 32).

    Used for node→node and edge→edge messages through (co)adjacency matrices.

    Inputs
    ------
    H_s : (N, d_in)            — source cochain
    adj : (2, E) sparse         — adj[0] = source j, adj[1] = target i

    Output
    ------
    K_t : (N, d_out)

    Attention score for pair (i, j):
        e_ij = LeakyReLU( a^T [ W·H_s[i] || W·H_s[j] (|| edge_attr) ] )
        att_ij = softmax over j in N(i) of e_ij
        K_t[i] = sum_j att_ij * (value_j or W·H_s[j])

    Parameters
    ----------
    d_in           : input feature dimension
    d_out          : output feature dimension
    n_heads        : number of attention heads (default 1)
    dropout        : dropout on attention weights (default 0.0)
    negative_slope : LeakyReLU slope (default 0.2)
    d_edge         : optional edge feature dim added to attention score (0 = disabled)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        d_edge: int = 0,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.d_edge = d_edge

        self.W = nn.Linear(d_in, d_out, bias=False)
        # Attention vector per head: [W·H[i] || W·H[j] (|| edge_feat)]
        attn_in = 2 * self.head_dim + d_edge
        self.a = nn.Parameter(torch.empty(n_heads, attn_in))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        H_s: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        H_s       : (N, d_in)
        adj       : (2, E) — adj[0] = source, adj[1] = target
        edge_attr : (E, d_edge) optional features added to attention score
        value     : (E, d_out) optional; if given, replaces W·H_s[source]
                    as the aggregated value (e.g. pre-computed messages)

        Returns
        -------
        K_t : (N, d_out)
        """
        N = H_s.size(0)
        src, tgt = adj[0], adj[1]

        Wh = self.W(H_s).view(N, self.n_heads, self.head_dim)  # (N, n_heads, hd)
        Wh_i = Wh[tgt]  # (E, n_heads, hd) — target features
        Wh_j = Wh[src]  # (E, n_heads, hd) — source features

        # Build attention input: [W·H[i] || W·H[j] (|| edge_attr)]
        cat_attn = torch.cat([Wh_i, Wh_j], dim=-1)  # (E, n_heads, 2*hd)
        if edge_attr is not None and self.d_edge > 0:
            ea = edge_attr.unsqueeze(1).expand(-1, self.n_heads, -1)
            cat_attn = torch.cat([cat_attn, ea], dim=-1)

        e = (cat_attn * self.a.unsqueeze(0)).sum(-1)  # (E, n_heads)
        e = self.leaky_relu(e)

        # Per-target-node softmax
        att = torch.stack(
            [_scatter_softmax_1d(e[:, h], tgt, N) for h in range(self.n_heads)],
            dim=1,
        )  # (E, n_heads)
        att = self.dropout(att)

        # Values: external or W·H_s[source]
        if value is not None:
            vals = value.view(-1, self.n_heads, self.head_dim)
        else:
            vals = Wh_j  # (E, n_heads, hd)

        weighted = att.unsqueeze(-1) * vals  # (E, n_heads, hd)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=H_s.device, dtype=H_s.dtype)
        idx = tgt.unsqueeze(1).unsqueeze(2).expand_as(weighted)
        out.scatter_add_(0, idx, weighted)
        return out.view(N, self.d_out)


# ── Unequal-rank bidirectional attention block ────────────────────────────────

class CCAttentionBlock(nn.Module):
    """Bidirectional CC-attention push-forward for cells of unequal rank
    (Definition 33).

    Simultaneously computes:
      - forward  (s→t): K_t = attention-weighted aggregation of rank-s onto rank-t
      - reverse  (t→s): K_s = attention-weighted aggregation of rank-t onto rank-s

    Inputs
    ------
    H_s : (N_s, d_s_in)  — rank-s cochain (lower rank, e.g. atoms)
    H_t : (N_t, d_t_in)  — rank-t cochain (higher rank, e.g. bonds)
    B   : (2, E_st) sparse — B[0] = rank-s indices, B[1] = rank-t indices

    Outputs
    -------
    K_t : (N_t, d_out)  — updated rank-t cochain
    K_s : (N_s, d_out)  — updated rank-s cochain

    Forward attention (s→t):
        e_ij = phi( a^T [ W_s·H_s[j] || W_t·H_t[i] ] )
               where i = rank-t cell, j ∈ {B[0] : B[1]==i}
        att_ij = softmax over {j : B[1]==i}
        K_t[i] = sum_j att_ij * W_s * H_s[j]

    Reverse attention (t→s):
        f_ji = phi( rev_a^T [ W_t·H_t[i] || W_s·H_s[j] ] )
               rev_a = [a[head_dim:] || a[:head_dim]]
        att_ji = softmax over {i : B[0]==j}
        K_s[j] = sum_i att_ji * W_t * H_t[i]

    Parameters
    ----------
    d_s_in, d_t_in : input dims for rank-s and rank-t
    d_out          : output dim (same for both directions)
    n_heads        : number of attention heads (default 1)
    dropout        : attention dropout (default 0.0)
    negative_slope : LeakyReLU slope (default 0.2)
    """

    def __init__(
        self,
        d_s_in: int,
        d_t_in: int,
        d_out: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert d_out % n_heads == 0
        self.d_s_in = d_s_in
        self.d_t_in = d_t_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_s = nn.Linear(d_s_in, d_out, bias=False)
        self.W_t = nn.Linear(d_t_in, d_out, bias=False)

        # Shared attention vector per head (2 * head_dim).
        # Forward uses a^T [W_s·H_s[j] || W_t·H_t[i]].
        # Reverse uses rev_a^T [W_t·H_t[i] || W_s·H_s[j]]
        #   where rev_a = [a[head_dim:] || a[:head_dim]].
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        H_s: Tensor,
        H_t: Tensor,
        B: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        H_s : (N_s, d_s_in)
        H_t : (N_t, d_t_in)
        B   : (2, E) — B[0] = rank-s indices, B[1] = rank-t indices

        Returns
        -------
        K_t : (N_t, d_out)
        K_s : (N_s, d_out)
        """
        N_s, N_t = H_s.size(0), H_t.size(0)
        s_idx = B[0]  # (E,) rank-s indices
        t_idx = B[1]  # (E,) rank-t indices

        Ws_h = self.W_s(H_s).view(N_s, self.n_heads, self.head_dim)
        Wt_h = self.W_t(H_t).view(N_t, self.n_heads, self.head_dim)

        Ws_j = Ws_h[s_idx]  # (E, n_heads, hd)
        Wt_i = Wt_h[t_idx]  # (E, n_heads, hd)

        # ── Forward (s→t): e_ij = phi( a^T [ W_s·H_s[j] || W_t·H_t[i] ] ) ──
        e_fwd = (torch.cat([Ws_j, Wt_i], dim=-1) * self.a.unsqueeze(0)).sum(-1)
        e_fwd = self.leaky_relu(e_fwd)  # (E, n_heads)

        att_fwd = torch.stack(
            [_scatter_softmax_1d(e_fwd[:, h], t_idx, N_t) for h in range(self.n_heads)],
            dim=1,
        )  # (E, n_heads)
        att_fwd = self.dropout(att_fwd)

        weighted_fwd = att_fwd.unsqueeze(-1) * Ws_j  # (E, n_heads, hd)
        K_t = torch.zeros(N_t, self.n_heads, self.head_dim, device=H_s.device, dtype=H_s.dtype)
        K_t.scatter_add_(0, t_idx.unsqueeze(1).unsqueeze(2).expand_as(weighted_fwd), weighted_fwd)
        K_t = K_t.view(N_t, self.d_out)

        # ── Reverse (t→s): rev_a^T [ W_t·H_t[i] || W_s·H_s[j] ] ──────────
        rev_a = torch.cat([self.a[:, self.head_dim:], self.a[:, :self.head_dim]], dim=-1)
        e_rev = (torch.cat([Wt_i, Ws_j], dim=-1) * rev_a.unsqueeze(0)).sum(-1)
        e_rev = self.leaky_relu(e_rev)  # (E, n_heads)

        att_rev = torch.stack(
            [_scatter_softmax_1d(e_rev[:, h], s_idx, N_s) for h in range(self.n_heads)],
            dim=1,
        )  # (E, n_heads)
        att_rev = self.dropout(att_rev)

        weighted_rev = att_rev.unsqueeze(-1) * Wt_i  # (E, n_heads, hd)
        K_s = torch.zeros(N_s, self.n_heads, self.head_dim, device=H_s.device, dtype=H_s.dtype)
        K_s.scatter_add_(0, s_idx.unsqueeze(1).unsqueeze(2).expand_as(weighted_rev), weighted_rev)
        K_s = K_s.view(N_s, self.d_out)

        return K_t, K_s


# ── Inter-neighborhood attention merge ───────────────────────────────────────

class AttentionMergeNode(nn.Module):
    """Attention merge node combining messages from multiple neighborhoods
    (Definition 36, inter-neighborhood weights b^k).

    Inputs
    ------
    messages : list of K tensors, each (N, d)

    Output
    ------
    merged : (N, d)

    b^k = softmax(learned_weights) over k neighborhoods
    merged = sum_k b^k * messages[k]

    Parameters
    ----------
    n_neighborhoods : number of neighborhoods to merge
    hidden_dim      : feature dimension (unused in computation, kept for API clarity)
    """

    def __init__(self, n_neighborhoods: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_neighborhoods = n_neighborhoods
        self.hidden_dim = hidden_dim
        self.weights = nn.Parameter(torch.ones(n_neighborhoods))

    def forward(self, messages: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        messages : list of (N, d) tensors, one per neighborhood

        Returns
        -------
        merged : (N, d)
        """
        assert len(messages) == self.n_neighborhoods, (
            f"Expected {self.n_neighborhoods} message tensors, got {len(messages)}"
        )
        b_k = torch.softmax(self.weights, dim=0)  # (K,)
        return sum(b_k[k] * messages[k] for k in range(self.n_neighborhoods))


# ── Zone adjacency helper ─────────────────────────────────────────────────────

def build_zone_adjacency(
    zone_mask: Tensor,
    global_edge_index: Tensor,
) -> Tensor:
    """Filter edge_index to edges with both endpoints in a zone, then
    remap node indices to zone-local (contiguous) indices.

    Parameters
    ----------
    zone_mask         : (N_total,) BoolTensor — True for nodes in this zone
    global_edge_index : (2, E) — global edge_index (source, target)

    Returns
    -------
    local_edge_index : (2, E_zone) — zone-local edge index
    """
    src, tgt = global_edge_index[0], global_edge_index[1]
    in_zone = zone_mask[src] & zone_mask[tgt]

    if not in_zone.any():
        return global_edge_index.new_zeros(2, 0)

    node_map = torch.full(
        (zone_mask.size(0),), -1, dtype=torch.long, device=zone_mask.device
    )
    zone_nodes = zone_mask.nonzero(as_tuple=True)[0]
    node_map[zone_nodes] = torch.arange(zone_nodes.size(0), device=zone_mask.device)

    return torch.stack([node_map[src[in_zone]], node_map[tgt[in_zone]]], dim=0)
