"""
ToPE Protein Engineering Engine
================================

Forward (scoring) and inverse (proposing) mutation design on top of the
ToPE active-site representation.

Quick start
-----------
    from tope.data.active_site import ActiveSite
    from tope.engineering import (
        EngineeringEngine, EngineConfig, Mutation, AttributionScorer,
        SaturationScan, BeamSearch, MCMCSearch,
    )

    # Score a specific mutation set (forward)
    engine = EngineeringEngine(
        scorer=AttributionScorer(),
        search=SaturationScan(top_k=20),
    )
    score = engine.score(active_site, [Mutation(chain="A", resnum=57, target_aa="ALA")])

    # Propose mutations (inverse)
    candidates = engine.propose(active_site, objective="maximize")

The engine is deliberately decoupled from any specific neural scoring
backend: the `ToPEModelScorer` takes a user-supplied callable mapping
`ActiveSite -> float`, so it can wrap whatever featurisation pipeline the
caller has already wired up to `ToPEModel.forward`.
"""

from tope.engineering.config import EngineConfig
from tope.engineering.mutation import (
    Mutation,
    MutationSet,
    Repacker,
    apply_mutations,
)
from tope.engineering.scoring import (
    ScorerProtocol,
    ScoredMutationSet,
    AttributionScorer,
    ToPEModelScorer,
)
from tope.engineering.search import (
    SearchStrategy,
    SaturationScan,
    BeamSearch,
    MCMCSearch,
)
from tope.engineering.engine import EngineeringEngine

__all__ = [
    "EngineConfig",
    "Mutation",
    "MutationSet",
    "Repacker",
    "apply_mutations",
    "ScorerProtocol",
    "ScoredMutationSet",
    "AttributionScorer",
    "ToPEModelScorer",
    "SearchStrategy",
    "SaturationScan",
    "BeamSearch",
    "MCMCSearch",
    "EngineeringEngine",
]
