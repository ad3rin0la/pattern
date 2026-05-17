"""Top-level orchestrator for the protein engineering pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from tope.data.active_site import ActiveSite
from tope.engineering.config import EngineConfig
from tope.engineering.mutation import Mutation
from tope.engineering.scoring import ScoredMutationSet, ScorerProtocol
from tope.engineering.search import (
    SaturationScan,
    BeamSearch,
    MCMCSearch,
    SearchStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineeringEngine:
    """Drive forward scoring and inverse proposal of mutations.

    Parameters
    ----------
    scorer : ScorerProtocol
        Required. Either an `AttributionScorer` or a `ToPEModelScorer`
        wrapping a trained model.
    search : SearchStrategy, optional
        Used by `.propose()`. Defaults to `SaturationScan()`.
    cfg : EngineConfig, optional
        Knobs shared across scoring and search.
    """

    scorer: ScorerProtocol
    search: Optional[SearchStrategy] = None
    cfg: Optional[EngineConfig] = None

    def __post_init__(self) -> None:
        self.cfg = self.cfg or EngineConfig()
        self.search = self.search or SaturationScan()

    # ── Forward scoring ──────────────────────────────────────────────────

    def score(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ScoredMutationSet:
        """Score a single mutation set."""
        return self.scorer.score(active_site, mutations)

    def score_batch(
        self,
        active_site: ActiveSite,
        mutation_sets: Iterable[Sequence[Mutation]],
    ) -> List[ScoredMutationSet]:
        """Score many mutation sets, returning them in input order."""
        return [self.scorer.score(active_site, ms) for ms in mutation_sets]

    # ── Inverse search ───────────────────────────────────────────────────

    def propose(
        self,
        active_site: ActiveSite,
        objective: Optional[str] = None,
        strategy: Optional[SearchStrategy] = None,
    ) -> List[ScoredMutationSet]:
        """Run the configured search and return ranked candidates.

        `objective` overrides `cfg.objective` for this call. `strategy`
        overrides the default search strategy.
        """
        cfg = self.cfg
        if objective is not None:
            if objective not in {"maximize", "minimize"}:
                raise ValueError(f"objective must be max/min, got {objective!r}")
            cfg = _replace(cfg, objective=objective)

        used = strategy or self.search
        return used.search(active_site, self.scorer, cfg)

    # ── Convenience: run all three search strategies and merge ──────────

    def propose_ensemble(
        self,
        active_site: ActiveSite,
        objective: Optional[str] = None,
    ) -> List[ScoredMutationSet]:
        """Run saturation + beam + MCMC and return a merged ranked list.

        Useful when you want broad coverage and don't yet know which
        search shape best matches your scorer's landscape.
        """
        strategies: List[SearchStrategy] = [
            SaturationScan(),
            BeamSearch(),
            MCMCSearch(),
        ]
        merged: List[ScoredMutationSet] = []
        for s in strategies:
            merged.extend(self.propose(active_site, objective=objective, strategy=s))

        # Deduplicate by mutation-set tag, keep the best score under cfg.
        cfg = self.cfg if objective is None else _replace(self.cfg, objective=objective)
        sort_max = cfg.objective == "maximize"
        merged.sort(key=lambda s: -s.score if sort_max else s.score)
        seen = set()
        unique: List[ScoredMutationSet] = []
        for s in merged:
            tag = str(s.mutations)
            if tag in seen:
                continue
            seen.add(tag)
            unique.append(s)
        return unique[: cfg.return_top_n]


def _replace(cfg: EngineConfig, **kwargs) -> EngineConfig:
    """Shallow copy + override of an EngineConfig."""
    import dataclasses
    return dataclasses.replace(cfg, **kwargs)
