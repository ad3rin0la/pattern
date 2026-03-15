"""
ToPE Phase 5A: Attentive VOIP-SIREN Architecture
=================================================
Replaces the static NIST + GFN2-xTB correction pipeline with a unified,
differentiable electronic potential field represented as a spectrally-attended
SIREN (Sinusoidal Implicit Neural Representation).

Architecture:
    1.  SpectralBandSIREN       — one SIREN per frequency band (core/valence/lone-pair)
    2.  BandAttention           — substrate-conditioned soft selection of frequency bands
    3.  SpatialGlobalAttention  — global context modulates local VOIP queries
    4.  VOIPSIRENField          — unified VOIP field  (replaces NIST + xTB lookup)
    5.  SubstrateVOIPCrossAttn  — substrate atoms attend directly to the VOIP field
    6.  SpectralPosEncoding     — SIREN activations as physics-informed pos. encodings
    7.  AttentiveVOIPEncoder    — full encoder that feeds the sheaf Laplacian pipeline
Design principles:
    - xTB descriptors (Mulliken charges, Wiberg bond orders, coordination numbers)
      are the *context signal* for band attention — not additive corrections.
    - Sine activations are natural for oscillatory electronic wavefunctions.
    - Spectral bias (low-freq first) mirrors core → valence → long-range ordering.
    - Substrate embedding drives which frequency regime the field exposes;
      this is the continuous analogue of sweeping Λᵣ in rank-stratified filtration.
    - All operations are GPU-native (torch.linalg, torch.einsum, geoopt-compatible).
References:
    Sitzmann et al. (2020) — SIREN: Implicit Neural Representations with
        Periodic Activation Functions
    Ioffe (1983)           — Electronic criterion-space framework for catalysis
    ToPE Phase 3           — SubstrateProductCrossAttention, MoleculeGNN
    ToPE Phase 5A          — XTBInterface, geometry-dependent VOIP
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# §0  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VOIPSIRENConfig:
    """Hyper-parameters for the attentive VOIP-SIREN architecture."""

    # ── SIREN bands ──────────────────────────────────────────────────────────
    # Frequency bands correspond to electronic structure scales:
    #   core_omega    ~1–3   : core electron density, slow-varying backbone
    #   valence_omega ~8–15  : valence electrons, bonding, coordination shell
    #   lp_omega      ~25–40 : lone pairs, polarisation, fine orbital structure
    freq_bands: List[float] = field(default_factory=lambda: [2.0, 10.0, 30.0])
    siren_hidden_dim: int = 128
    siren_n_layers: int = 4          # depth of each band-SIREN

    # ── Coordinate / atom-type input ─────────────────────────────────────────
    coord_dim: int = 3               # (x, y, z)
    atom_type_vocab: int = 100       # max atomic number
    atom_type_embed_dim: int = 16

    # ── xTB context (drives band attention) ──────────────────────────────────
    xtb_dim: int = 32                # dim of xTB descriptor vector per atom
    # raw xTB scalars: Mulliken charge, Wiberg bond order sum,
    #                  coordination number, dispersion coeff C6, HOMO gap
    xtb_scalar_features: int = 5

    # ── Attention ─────────────────────────────────────────────────────────────
    n_heads: int = 8
    attn_dropout: float = 0.1

    # ── Substrate / cross-attention ───────────────────────────────────────────
    substrate_embed_dim: int = 256   # must match Phase-3 MoleculeGNN output dim
    enzyme_hidden_dim: int = 256     # must match EnzymeTCPNet hidden dim

    # ── Output ────────────────────────────────────────────────────────────────
    voip_out_dim: int = 8            # s, p, d × scalar + 5 auxiliary scalars


# ══════════════════════════════════════════════════════════════════════════════
# §1  Sinusoidal activation + SIREN layer primitives
# ══════════════════════════════════════════════════════════════════════════════

class Sine(nn.Module):
    """Sinusoidal activation with learnable frequency scale."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


def _siren_init(layer: nn.Linear, omega_0: float, is_first: bool) -> None:
    """Weight initialisation from Sitzmann et al. (2020) §3.2."""
    fan_in = layer.weight.shape[1]
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega_0
    nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.uniform_(layer.bias, -bound, bound)


class SIRENLayer(nn.Module):
    """One linear + sine layer for a SIREN network."""

    def __init__(self, in_dim: int, out_dim: int, omega_0: float, is_first: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = Sine(omega_0)
        _siren_init(self.linear, omega_0, is_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


# ══════════════════════════════════════════════════════════════════════════════
# §2  Per-band SIREN — one network per electronic frequency scale
# ══════════════════════════════════════════════════════════════════════════════

class BandSIREN(nn.Module):
    """
    Single-band SIREN that maps (coords, atom_type_embed) → VOIP activations.

    Inputs
    ------
    coords        : (N, 3)  Cartesian coordinates
    atom_embed    : (N, E)  atom-type embedding

    Output
    ------
    activations   : (N, H, n_layers)  intermediate + final activations
                    used both as the VOIP field and as positional encodings
    """

    def __init__(self, cfg: VOIPSIRENConfig, omega_0: float):
        super().__init__()
        self.omega_0 = omega_0
        in_dim = cfg.coord_dim + cfg.atom_type_embed_dim
        H = cfg.siren_hidden_dim
        layers: List[nn.Module] = [SIRENLayer(in_dim, H, omega_0, is_first=True)]
        for _ in range(cfg.siren_n_layers - 1):
            layers.append(SIRENLayer(H, H, omega_0, is_first=False))
        self.layers = nn.ModuleList(layers)
        # Final projection to VOIP output space
        self.out_proj = nn.Linear(H, cfg.voip_out_dim)

    def forward(
        self,
        coords: torch.Tensor,           # (N, 3)
        atom_embed: torch.Tensor,       # (N, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        voip   : (N, voip_out_dim)  final VOIP values for this band
        acts   : (N, H, L)          per-layer activations for pos. encoding
        """
        x = torch.cat([coords, atom_embed], dim=-1)          # (N, 3+E)
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)                            # each (N, H)
        voip = self.out_proj(x)                              # (N, voip_out_dim)
        acts = torch.stack(activations, dim=-1)              # (N, H, L)
        return voip, acts


# ══════════════════════════════════════════════════════════════════════════════
# §3  xTB context encoder — maps raw xTB scalars → context vector
# ══════════════════════════════════════════════════════════════════════════════

class XTBContextEncoder(nn.Module):
    """
    Encodes per-atom GFN2-xTB descriptors into a context vector that drives
    band attention.

    Input scalars (per atom):
        q_Mulliken  : partial charge  (dimensionless)
        WBO_sum     : Wiberg bond order sum
        CN          : coordination number
        C6          : dispersion coefficient (a.u.)
        gap_HOMO    : HOMO-LUMO gap (eV)

    These are the same quantities used by XTBInterface in
    tope/quantum/xtb_interface.py.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.xtb_scalar_features, cfg.xtb_dim),
            nn.SiLU(),
            nn.Linear(cfg.xtb_dim, cfg.xtb_dim),
            nn.LayerNorm(cfg.xtb_dim),
        )

    def forward(self, xtb_scalars: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xtb_scalars : (N, 5)

        Returns
        -------
        context : (N, xtb_dim)
        """
        return self.mlp(xtb_scalars)


# ══════════════════════════════════════════════════════════════════════════════
# §4  Band Attention — substrate + xTB jointly select frequency bands
# ══════════════════════════════════════════════════════════════════════════════

class BandAttention(nn.Module):
    """
    Computes per-atom soft weights over frequency bands.

    Context signal:
        - xTB descriptor  (encodes local electronic environment)
        - substrate summary embedding  (encodes what the substrate "needs")

    High CN / diffuse environment  →  low-ω band dominates.
    Lone-pair donor / sharp orbital  →  high-ω band dominates.
    The substrate can shift these preferences to expose the right
    electronic scale for complementarity scoring.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        n_bands = len(cfg.freq_bands)
        context_dim = cfg.xtb_dim + cfg.substrate_embed_dim
        self.attn_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.SiLU(),
            nn.Linear(context_dim // 2, n_bands),
        )

    def forward(
        self,
        xtb_context: torch.Tensor,          # (N, xtb_dim)
        substrate_summary: torch.Tensor,    # (1, sub_dim) or (N, sub_dim)
    ) -> torch.Tensor:
        """
        Returns
        -------
        band_weights : (N, n_bands)  normalised attention weights
        """
        # Broadcast substrate summary to all atoms
        if substrate_summary.dim() == 2 and substrate_summary.shape[0] == 1:
            substrate_summary = substrate_summary.expand(xtb_context.shape[0], -1)
        ctx = torch.cat([xtb_context, substrate_summary], dim=-1)  # (N, ctx_dim)
        logits = self.attn_mlp(ctx)                                # (N, n_bands)
        return F.softmax(logits, dim=-1)                           # (N, n_bands)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Spatial Global Attention — global context modulates local VOIP queries
# ══════════════════════════════════════════════════════════════════════════════

class SpatialGlobalAttention(nn.Module):
    """
    Self-attention over atom positions so that the VOIP at each site is
    informed by the whole-protein electronic environment.

    This is the mechanism that preserves allosteric long-range coupling:
    a residue >20 Å from the active site can modulate VOIP values inside
    the active site if they are electronically coupled.

    Implementation: standard multi-head self-attention with rotary-like
    distance bias injected as an additive log-distance penalty.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        H = cfg.siren_hidden_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=H,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(H)

    def forward(
        self,
        h: torch.Tensor,                    # (N, H) per-atom features
        coords: torch.Tensor,               # (N, 3) for distance bias
        batch: Optional[torch.Tensor] = None,  # (N,) graph membership
    ) -> torch.Tensor:
        """
        Returns
        -------
        h_global : (N, H)  globally-attended per-atom features
        """
        # For batched graphs we process each protein independently
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        batch_size = int(batch.max().item()) + 1
        outputs = []
        for gid in range(batch_size):
            mask = batch == gid
            h_g = h[mask].unsqueeze(0)          # (1, n, H)
            c_g = coords[mask]                  # (n, 3)

            # Pairwise distance bias: -log(1 + d_ij) discourages very
            # distant atoms from dominating attention
            dist = torch.cdist(c_g, c_g)        # (n, n)
            attn_bias = -torch.log1p(dist)       # (n, n)

            # MHA expects attn_mask of shape (n, n) or (n*heads, n, n)
            # We replicate across heads
            n = c_g.shape[0]
            attn_bias_expanded = attn_bias.unsqueeze(0).expand(
                self.mha.num_heads, -1, -1
            ).reshape(self.mha.num_heads, n, n)

            out, _ = self.mha(
                h_g, h_g, h_g,
                attn_mask=attn_bias_expanded.reshape(
                    self.mha.num_heads * n, n
                )[:n, :],               # (n, n) after reshaping trick
            )
            outputs.append(out.squeeze(0))      # (n, H)

        h_global = torch.cat(outputs, dim=0)    # (N, H)
        return self.ln(h + h_global)            # residual


# ══════════════════════════════════════════════════════════════════════════════
# §6  Spectrally-Attended VOIP Field — the unified VOIP generator
# ══════════════════════════════════════════════════════════════════════════════

class VOIPSIRENField(nn.Module):
    """
    The unified VOIP field.  Replaces NIST lookup + xTB additive correction.

    For each atom i:
        1. Query all band-SIRENs at position xᵢ
        2. Compute band weights from xTB context + substrate summary
        3. Return weighted sum of band outputs as VOIP(xᵢ)
        4. Apply global spatial attention to inject whole-protein context

    The result is a geometry-dependent, substrate-aware VOIP vector that
    satisfies Phase 5A's requirement: static NIST VOIP is the frozen-atom
    limit obtained when xTB context is zero and substrate_summary is zero.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        self.cfg = cfg

        # Atom-type embedding (shared across all bands)
        self.atom_embed = nn.Embedding(cfg.atom_type_vocab, cfg.atom_type_embed_dim)

        # One SIREN per frequency band
        self.band_sirens = nn.ModuleList([
            BandSIREN(cfg, omega_0=f) for f in cfg.freq_bands
        ])

        # xTB context encoder
        self.xtb_encoder = XTBContextEncoder(cfg)

        # Band attention
        self.band_attn = BandAttention(cfg)

        # Global spatial attention applied to aggregated hidden state
        self.spatial_attn = SpatialGlobalAttention(cfg)

        # Project aggregated hidden → same dim as voip_out for residual path
        H = cfg.siren_hidden_dim
        self.hidden_to_voip = nn.Linear(H, cfg.voip_out_dim)

        # Final layer norm on VOIP output
        self.out_ln = nn.LayerNorm(cfg.voip_out_dim)

    def forward(
        self,
        coords: torch.Tensor,               # (N, 3)
        atom_types: torch.Tensor,           # (N,) int64 atomic numbers
        xtb_scalars: torch.Tensor,          # (N, 5)
        substrate_summary: torch.Tensor,    # (1 or N, substrate_embed_dim)
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        voip        : (N, voip_out_dim)    geometry-dependent VOIP field values
        band_weights: (N, n_bands)         interpretable band attention weights
        pos_enc     : (N, H*L)            physics-informed positional encodings
        """
        atom_emb = self.atom_embed(atom_types)          # (N, E)
        xtb_ctx  = self.xtb_encoder(xtb_scalars)        # (N, xtb_dim)

        # ── Band attention weights ───────────────────────────────────────────
        band_weights = self.band_attn(xtb_ctx, substrate_summary)  # (N, B)

        # ── Query each band SIREN ────────────────────────────────────────────
        band_voips = []
        band_acts  = []
        for siren in self.band_sirens:
            v, a = siren(coords, atom_emb)       # (N, D), (N, H, L)
            band_voips.append(v)
            band_acts.append(a)

        band_voip_stack = torch.stack(band_voips, dim=1)  # (N, B, D)
        band_acts_stack = torch.stack(band_acts,  dim=1)  # (N, B, H, L)

        # ── Weighted sum over bands ──────────────────────────────────────────
        # band_weights: (N, B) → (N, B, 1) for broadcasting
        voip = (band_weights.unsqueeze(-1) * band_voip_stack).sum(dim=1)  # (N, D)

        # ── Global spatial attention on the aggregated hidden state ──────────
        # Use the weighted-average hidden state from the last SIREN layer
        last_acts = band_acts_stack[:, :, :, -1]         # (N, B, H)
        h_agg = (band_weights.unsqueeze(-1) * last_acts).sum(dim=1)  # (N, H)
        h_global = self.spatial_attn(h_agg, coords, batch)           # (N, H)

        # Add global context to VOIP via residual
        voip = self.out_ln(voip + self.hidden_to_voip(h_global))     # (N, D)

        # ── Physics-informed positional encoding ─────────────────────────────
        # Flatten all band-layer activations → rich positional descriptor
        # Shape: (N, B, H, L) → (N, B*H*L)
        pos_enc = band_acts_stack.reshape(coords.shape[0], -1)

        return voip, band_weights, pos_enc


# ══════════════════════════════════════════════════════════════════════════════
# §7  Substrate–VOIP Cross-Attention
#     Substrate atoms attend directly to the VOIP field (not residue embeddings)
# ══════════════════════════════════════════════════════════════════════════════

class SubstrateVOIPCrossAttention(nn.Module):
    """
    Cross-attention where:
        queries = substrate atom embeddings  (from MoleculeGNN)
        keys    = VOIP field values          (from VOIPSIRENField)
        values  = VOIP field values

    The attention score measures electronic complementarity between
    substrate orbital characteristics and the enzyme's VOIP landscape —
    the Ioffe complementarity criterion made architectural.

    Returns both enriched substrate embeddings and the attention map,
    which gives per-substrate-atom / per-enzyme-atom importance scores
    suitable for Phase 4 attribution.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        # Project VOIP field (voip_out_dim) → enzyme_hidden_dim for attention
        self.voip_key_proj = nn.Linear(cfg.voip_out_dim, cfg.enzyme_hidden_dim)
        self.voip_val_proj = nn.Linear(cfg.voip_out_dim, cfg.enzyme_hidden_dim)

        # Substrate projection (assumed already in substrate_embed_dim space)
        self.sub_query_proj = nn.Linear(cfg.substrate_embed_dim, cfg.enzyme_hidden_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.enzyme_hidden_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(cfg.enzyme_hidden_dim)

    def forward(
        self,
        h_substrate: torch.Tensor,  # (M, substrate_embed_dim)
        voip_field: torch.Tensor,   # (N, voip_out_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_sub_enriched : (M, enzyme_hidden_dim)
            Substrate embeddings enriched with VOIP-field context.
        attn_weights   : (n_heads, M, N)
            Per-head attention map for attribution.
        """
        Q = self.sub_query_proj(h_substrate).unsqueeze(0)   # (1, M, H)
        K = self.voip_key_proj(voip_field).unsqueeze(0)     # (1, N, H)
        V = self.voip_val_proj(voip_field).unsqueeze(0)     # (1, N, H)

        out, attn = self.mha(Q, K, V, need_weights=True, average_attn_weights=False)
        # out:  (1, M, H)
        # attn: (1, n_heads, M, N)

        h_sub_enriched = self.ln(
            h_substrate + out.squeeze(0)
        )                                                    # (M, H)
        return h_sub_enriched, attn.squeeze(0)               # (M,H), (heads,M,N)


# ══════════════════════════════════════════════════════════════════════════════
# §8  Spectral Positional Encoding bridge
#     Injects physics-informed pos. encodings into the transformer token stream
# ══════════════════════════════════════════════════════════════════════════════

class SpectralPosEncoding(nn.Module):
    """
    Projects the SIREN activation stack (band × hidden × layer) down to
    enzyme_hidden_dim and uses it as a learned positional encoding.

    Two atoms at the same Cartesian position but different electronic
    environments receive *different* positional encodings — a crucial
    property for enzyme active sites with multiple metal-coordinated
    residues at similar distances.
    """

    def __init__(self, cfg: VOIPSIRENConfig):
        super().__init__()
        n_bands  = len(cfg.freq_bands)
        n_layers = cfg.siren_n_layers
        raw_dim  = n_bands * cfg.siren_hidden_dim * n_layers
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, cfg.enzyme_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.enzyme_hidden_dim * 2, cfg.enzyme_hidden_dim),
            nn.LayerNorm(cfg.enzyme_hidden_dim),
        )

    def forward(self, pos_enc_raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_enc_raw : (N, raw_dim)

        Returns
        -------
        pos_enc : (N, enzyme_hidden_dim)
        """
        return self.proj(pos_enc_raw)


# ══════════════════════════════════════════════════════════════════════════════
# §9  AttentiveVOIPEncoder — the full encoder that feeds the sheaf pipeline
# ══════════════════════════════════════════════════════════════════════════════

class AttentiveVOIPEncoder(nn.Module):
    """
    Full Phase 5A encoder.  Drop-in replacement for the static NIST VOIP
    lookup + xTB additive correction used in Phase 2 sheaf section construction.

    Inputs
    ------
    coords            : (N, 3)   atom Cartesian coordinates
    atom_types        : (N,)     atomic numbers (int64)
    xtb_scalars       : (N, 5)   [q_Mulliken, WBO_sum, CN, C6, HOMO_gap]
    h_residue         : (N, H)   residue-level embeddings from TCPNet
    substrate_graph   : dict     {node_features: (M, sub_feat), edge_index: (2,E)}
    substrate_summary : (1, sub_dim)  global substrate embedding (from MolGNN pool)
    batch             : (N,)     graph membership (optional)

    Outputs
    -------
    voip_field        : (N, voip_out_dim)  geometry-dependent VOIP values
    h_enzyme_enriched : (N, enzyme_hidden_dim)  residue embeddings + pos enc
    h_sub_enriched    : (M, enzyme_hidden_dim)  substrate + VOIP cross-attn
    attn_maps         : dict {
        'band_weights'  : (N, n_bands),
        'substrate_voip': (n_heads, M, N),
    }
    """

    def __init__(self, cfg: Optional[VOIPSIRENConfig] = None):
        super().__init__()
        self.cfg = cfg or VOIPSIRENConfig()

        self.voip_field    = VOIPSIRENField(self.cfg)
        self.pos_enc       = SpectralPosEncoding(self.cfg)
        self.cross_attn    = SubstrateVOIPCrossAttention(self.cfg)

        # Combine residue embedding + spectral pos enc
        H = self.cfg.enzyme_hidden_dim
        self.fusion_ln = nn.LayerNorm(H)
        self.fusion_proj = nn.Linear(H, H)   # residual gate

    def forward(
        self,
        coords:            torch.Tensor,
        atom_types:        torch.Tensor,
        xtb_scalars:       torch.Tensor,
        h_residue:         torch.Tensor,
        h_substrate:       torch.Tensor,
        substrate_summary: torch.Tensor,
        batch:             Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:

        # ── 1. Compute VOIP field ────────────────────────────────────────────
        voip, band_weights, pos_enc_raw = self.voip_field(
            coords, atom_types, xtb_scalars, substrate_summary, batch
        )

        # ── 2. Physics-informed positional encoding ──────────────────────────
        pos_enc = self.pos_enc(pos_enc_raw)                   # (N, H)

        # ── 3. Enrich residue embeddings with spectral pos enc ───────────────
        h_enzyme_enriched = self.fusion_ln(
            h_residue + self.fusion_proj(pos_enc)
        )                                                      # (N, H)

        # ── 4. Substrate → VOIP cross-attention ─────────────────────────────
        h_sub_enriched, sub_voip_attn = self.cross_attn(
            h_substrate, voip
        )                                                      # (M,H), (heads,M,N)

        attn_maps = {
            "band_weights":   band_weights,    # (N, n_bands)
            "substrate_voip": sub_voip_attn,   # (heads, M, N)
        }

        return voip, h_enzyme_enriched, h_sub_enriched, attn_maps


# ══════════════════════════════════════════════════════════════════════════════
# §10  Integration shim: feeding AttentiveVOIPEncoder into the sheaf pipeline
# ══════════════════════════════════════════════════════════════════════════════

class VOIPSheafSectionBuilder(nn.Module):
    """
    Thin wrapper that converts AttentiveVOIPEncoder outputs into the sheaf
    section format expected by phase2_topological_encoding.py.

    Sheaf section at vertex i = [VOIP_s, VOIP_p, VOIP_d,
                                  electronegativity, ionisation_energy,
                                  electron_affinity, polarisability,
                                  oxidation_state]     ← Ioffe's 8-dim descriptor

    The first 3 dimensions are the s/p/d VOIP values produced by the SIREN.
    The remaining 5 are learned projections of the xTB context.
    """

    def __init__(self, cfg: Optional[VOIPSIRENConfig] = None):
        super().__init__()
        self.cfg = cfg or VOIPSIRENConfig()
        assert self.cfg.voip_out_dim == 8, (
            "voip_out_dim must be 8 to match Ioffe descriptor dimensionality"
        )
        # Map xTB context to the remaining 5 Ioffe descriptors
        self.xtb_to_ioffe = nn.Linear(self.cfg.xtb_dim, 5)

    def forward(
        self,
        voip_field:  torch.Tensor,    # (N, 8) from AttentiveVOIPEncoder
        xtb_context: torch.Tensor,    # (N, xtb_dim) from XTBContextEncoder
    ) -> torch.Tensor:
        """
        Returns
        -------
        sheaf_sections : (N, 8)   ready for SheafENM / persistent sheaf Laplacian
        """
        ioffe_extra = self.xtb_to_ioffe(xtb_context)        # (N, 5)
        # First 3 dims from VOIP field (s, p, d orbital potentials)
        # Last 5 dims from xTB → Ioffe projection
        sections = torch.cat([voip_field[:, :3], ioffe_extra], dim=-1)  # (N, 8)
        return sections


# ══════════════════════════════════════════════════════════════════════════════
# §11  Quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running self-test on {device}\n")

    cfg = VOIPSIRENConfig(
        freq_bands=[2.0, 10.0, 30.0],
        siren_hidden_dim=64,
        siren_n_layers=3,
        voip_out_dim=8,
    )

    N, M = 120, 18   # 120 enzyme atoms, 18 substrate atoms
    coords       = torch.randn(N, 3, device=device)
    atom_types   = torch.randint(1, 80, (N,), device=device)
    xtb_scalars  = torch.randn(N, 5, device=device)
    h_residue    = torch.randn(N, cfg.enzyme_hidden_dim, device=device)
    h_substrate  = torch.randn(M, cfg.substrate_embed_dim, device=device)
    sub_summary  = torch.randn(1, cfg.substrate_embed_dim, device=device)
    batch        = torch.zeros(N, dtype=torch.long, device=device)

    encoder = AttentiveVOIPEncoder(cfg).to(device)
    builder = VOIPSheafSectionBuilder(cfg).to(device)

    voip, h_enz, h_sub, attn = encoder(
        coords, atom_types, xtb_scalars,
        h_residue, h_substrate, sub_summary, batch
    )
    xtb_ctx = XTBContextEncoder(cfg).to(device)(xtb_scalars)
    sections = builder(voip, xtb_ctx)

    print("AttentiveVOIPEncoder outputs:")
    print(f"  voip field      : {voip.shape}")           # (120, 8)
    print(f"  enzyme enriched : {h_enz.shape}")          # (120, 256)
    print(f"  substrate enrich: {h_sub.shape}")          # (18, 256)
    print(f"  band_weights    : {attn['band_weights'].shape}")    # (120, 3)
    print(f"  sub-voip attn   : {attn['substrate_voip'].shape}")  # (8, 18, 120)
    print(f"\nSheaf sections   : {sections.shape}")      # (120, 8)

    # Verify gradient flow end-to-end
    loss = voip.sum() + h_enz.sum() + h_sub.sum() + sections.sum()
    loss.backward()
    print("\nGradient flow: OK")

    # Parameter count
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"AttentiveVOIPEncoder params: {n_params:,}")
