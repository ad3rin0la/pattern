"""Configuration for the protein engineering engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Canonical amino acid alphabet (three-letter codes).
STANDARD_AA: List[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


@dataclass
class EngineConfig:
    """Knobs that govern mutation enumeration, search, and ranking."""

    # ── Mutation space ────────────────────────────────────────────────────
    target_amino_acids: List[str] = field(default_factory=lambda: list(STANDARD_AA))
    exclude_catalytic: bool = True
    exclude_self_mutation: bool = True
    max_simultaneous_mutations: int = 3

    # ── Saturation scan ───────────────────────────────────────────────────
    saturation_top_k: int = 20

    # ── Beam search ───────────────────────────────────────────────────────
    beam_width: int = 8
    beam_depth: int = 3  # number of compound mutations

    # ── MCMC ──────────────────────────────────────────────────────────────
    mcmc_steps: int = 200
    mcmc_temperature: float = 1.0
    mcmc_anneal: bool = True       # geometric cool-down toward 0
    mcmc_seed: Optional[int] = None

    # ── Objective direction ───────────────────────────────────────────────
    # When the scorer returns a scalar, "maximize" treats higher as better.
    objective: str = "maximize"   # {"maximize", "minimize"}

    # ── Misc ──────────────────────────────────────────────────────────────
    return_top_n: int = 50
    verbose: bool = False
