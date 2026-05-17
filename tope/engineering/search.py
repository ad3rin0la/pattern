"""Search strategies over the mutation space.

All strategies share the same interface: given an `ActiveSite` and a
`ScorerProtocol`, return a list of `ScoredMutationSet`s sorted under the
configured objective (max-first by default).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np

from tope.data.active_site import ActiveSite, ResidueRecord
from tope.engineering.config import EngineConfig, STANDARD_AA
from tope.engineering.mutation import Mutation, MutationSet
from tope.engineering.scoring import ScoredMutationSet, ScorerProtocol

logger = logging.getLogger(__name__)


class SearchStrategy(Protocol):
    """A search produces ranked `ScoredMutationSet` candidates."""

    def search(
        self,
        active_site: ActiveSite,
        scorer: ScorerProtocol,
        cfg: EngineConfig,
    ) -> List[ScoredMutationSet]: ...


# ── Shared helpers ────────────────────────────────────────────────────────────

def _eligible_residues(
    active_site: ActiveSite,
    cfg: EngineConfig,
) -> List[ResidueRecord]:
    out = []
    for r in active_site.residues:
        if r.residue_name.upper() not in STANDARD_AA:
            continue  # don't try to mutate cofactors / ligands
        if cfg.exclude_catalytic and r.is_catalytic:
            continue
        out.append(r)
    return out


def _candidate_aas(
    residue: ResidueRecord,
    cfg: EngineConfig,
) -> List[str]:
    out = []
    src = residue.residue_name.upper()
    for aa in cfg.target_amino_acids:
        aa_u = aa.upper()
        if cfg.exclude_self_mutation and aa_u == src:
            continue
        out.append(aa_u)
    return out


def _better(a: float, b: float, cfg: EngineConfig) -> bool:
    return a > b if cfg.objective == "maximize" else a < b


def _sort_key(cfg: EngineConfig):
    return (lambda s: -s.score) if cfg.objective == "maximize" else (lambda s: s.score)


# ── 1. Saturation scan ────────────────────────────────────────────────────────

@dataclass
class SaturationScan:
    """Single-point saturation: every eligible residue × every target AA.

    Returns the `top_k` (defaults to `cfg.saturation_top_k`) best single
    mutations under the objective. This is the recommended baseline.
    """

    top_k: Optional[int] = None

    def search(
        self,
        active_site: ActiveSite,
        scorer: ScorerProtocol,
        cfg: EngineConfig,
    ) -> List[ScoredMutationSet]:
        results: List[ScoredMutationSet] = []
        residues = _eligible_residues(active_site, cfg)
        for res in residues:
            for aa in _candidate_aas(res, cfg):
                m = Mutation(res.chain_id, res.residue_number, aa)
                results.append(scorer.score(active_site, [m]))
        results.sort(key=_sort_key(cfg))
        k = self.top_k or cfg.saturation_top_k
        if cfg.verbose:
            logger.info("Saturation scan: %d candidates, top score = %.4f",
                        len(results), results[0].score if results else float("nan"))
        return results[:k]


# ── 2. Beam search ────────────────────────────────────────────────────────────

@dataclass
class BeamSearch:
    """Beam search over multi-point mutation sets.

    Starts from saturation top-`beam_width`, then at each depth extends
    each beam by every additional eligible single mutation, keeping the
    top `beam_width` candidates by combined score.
    """

    beam_width: Optional[int] = None
    beam_depth: Optional[int] = None

    def search(
        self,
        active_site: ActiveSite,
        scorer: ScorerProtocol,
        cfg: EngineConfig,
    ) -> List[ScoredMutationSet]:
        width = self.beam_width or cfg.beam_width
        depth = self.beam_depth or cfg.beam_depth
        depth = min(depth, cfg.max_simultaneous_mutations)

        # Seed beam with best single-point mutations.
        seed = SaturationScan(top_k=width).search(active_site, scorer, cfg)
        if depth <= 1 or not seed:
            return seed

        beam = seed
        history: List[ScoredMutationSet] = list(seed)

        residues = _eligible_residues(active_site, cfg)
        for d in range(2, depth + 1):
            extended: List[ScoredMutationSet] = []
            for cand in beam:
                used_keys = {m.key for m in cand.mutations}
                for res in residues:
                    if (res.chain_id, res.residue_number) in used_keys:
                        continue
                    for aa in _candidate_aas(res, cfg):
                        new_mut = Mutation(res.chain_id, res.residue_number, aa)
                        new_set = list(cand.mutations) + [new_mut]
                        extended.append(scorer.score(active_site, new_set))
            if not extended:
                break
            extended.sort(key=_sort_key(cfg))
            beam = extended[:width]
            history.extend(beam)
            if cfg.verbose:
                logger.info(
                    "Beam depth=%d: %d extensions, beam-top=%.4f",
                    d, len(extended), beam[0].score,
                )

        history.sort(key=_sort_key(cfg))
        # Deduplicate by mutation set string.
        seen = set()
        unique: List[ScoredMutationSet] = []
        for s in history:
            tag = str(s.mutations)
            if tag in seen:
                continue
            seen.add(tag)
            unique.append(s)
        return unique[: cfg.return_top_n]


# ── 3. MCMC / simulated annealing ─────────────────────────────────────────────

@dataclass
class MCMCSearch:
    """Metropolis-Hastings walk over mutation sets.

    Proposes single-residue edits (add / remove / swap target AA) and
    accepts under a Metropolis criterion. With `mcmc_anneal=True`, the
    temperature geometrically decays toward zero across `mcmc_steps`,
    yielding simulated annealing.
    """

    n_steps: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None

    def search(
        self,
        active_site: ActiveSite,
        scorer: ScorerProtocol,
        cfg: EngineConfig,
    ) -> List[ScoredMutationSet]:
        steps = self.n_steps or cfg.mcmc_steps
        t0 = self.temperature if self.temperature is not None else cfg.mcmc_temperature
        seed = self.seed if self.seed is not None else cfg.mcmc_seed
        rng = random.Random(seed)

        residues = _eligible_residues(active_site, cfg)
        if not residues:
            return []

        # Start from a single random mutation.
        current_mut = self._random_mutation(rng, residues, cfg)
        current = scorer.score(active_site, [current_mut])
        best = current
        trace: List[ScoredMutationSet] = [current]

        sign = 1.0 if cfg.objective == "maximize" else -1.0

        for step in range(1, steps + 1):
            T = self._temperature(t0, step, steps, cfg.mcmc_anneal)
            proposal_muts = self._propose(rng, current.mutations, residues, cfg)
            if not proposal_muts:
                continue
            proposal = scorer.score(active_site, proposal_muts)

            delta = sign * (proposal.score - current.score)
            if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
                current = proposal
                if sign * (current.score - best.score) > 0:
                    best = current
            trace.append(current)

        trace.append(best)
        trace.sort(key=_sort_key(cfg))
        seen = set()
        unique: List[ScoredMutationSet] = []
        for s in trace:
            tag = str(s.mutations)
            if tag in seen:
                continue
            seen.add(tag)
            unique.append(s)
        return unique[: cfg.return_top_n]

    @staticmethod
    def _temperature(t0: float, step: int, total: int, anneal: bool) -> float:
        if not anneal:
            return t0
        # Geometric cool-down to ~1% of initial T over the run.
        return t0 * (0.01 ** (step / max(total, 1)))

    @staticmethod
    def _random_mutation(
        rng: random.Random,
        residues: Sequence[ResidueRecord],
        cfg: EngineConfig,
    ) -> Mutation:
        res = rng.choice(list(residues))
        targets = [aa for aa in cfg.target_amino_acids
                   if not (cfg.exclude_self_mutation
                           and aa.upper() == res.residue_name.upper())]
        return Mutation(res.chain_id, res.residue_number, rng.choice(targets).upper())

    def _propose(
        self,
        rng: random.Random,
        current: Sequence[Mutation],
        residues: Sequence[ResidueRecord],
        cfg: EngineConfig,
    ) -> List[Mutation]:
        muts = list(current)
        moves = ["swap"]
        if len(muts) < cfg.max_simultaneous_mutations:
            moves.append("add")
        if len(muts) > 1:
            moves.append("remove")
        move = rng.choice(moves)

        if move == "add":
            occupied = {m.key for m in muts}
            free = [r for r in residues
                    if (r.chain_id, r.residue_number) not in occupied]
            if not free:
                return muts
            muts.append(self._random_mutation(rng, free, cfg))
        elif move == "remove":
            idx = rng.randrange(len(muts))
            muts.pop(idx)
        else:  # swap target AA on an existing site
            idx = rng.randrange(len(muts))
            old = muts[idx]
            targets = [aa.upper() for aa in cfg.target_amino_acids
                       if aa.upper() != old.target_aa.upper()]
            if not targets:
                return muts
            muts[idx] = Mutation(old.chain, old.resnum, rng.choice(targets))
        return muts
