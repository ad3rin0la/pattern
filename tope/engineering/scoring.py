"""Mutation scorers.

Two backends are provided:

* `ToPEModelScorer` — wraps any callable `ActiveSite -> float | dict`
  (typically a closure over a featurised `ToPEModel.forward`). The
  engine is decoupled from the exact tensor schema the model expects
  because that schema varies across heads/configs in this repo.

* `AttributionScorer` — model-free, uses Ioffe-style per-atom features
  re-computed on the mutant active site and aggregated with a simple
  attribution-style weighting (catalytic neighbourhood emphasised).

Both produce a `ScoredMutationSet` carrying the scalar score plus the
per-component breakdown for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from tope.data.active_site import ActiveSite
from tope.data.features import ActiveSiteFeatures, FeatureComputer
from tope.engineering.mutation import Mutation, MutationSet, Repacker, apply_mutations


@dataclass
class ScoredMutationSet:
    """A scored mutation candidate."""

    mutations: MutationSet
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "ScoredMutationSet") -> bool:
        return self.score < other.score


class ScorerProtocol(Protocol):
    """A scorer turns an ActiveSite + mutations into a scalar."""

    def score(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ScoredMutationSet: ...


# ── Model-based scorer ─────────────────────────────────────────────────────────

class ToPEModelScorer:
    """Wrap a user-supplied callable that scores an `ActiveSite`.

    The callable returns either a scalar (combined objective) or a dict
    with at least a `"score"` key plus arbitrary breakdown components.
    Mutations are applied via `apply_mutations` (using `repacker` when
    provided) before the callable is invoked, so the user-supplied
    function never has to deal with the WT baseline difference itself.

    Parameters
    ----------
    score_fn : callable
        `ActiveSite -> float | dict`. If a dict, must contain `"score"`.
    repacker : Repacker, optional
        External sidechain repacker; if `None`, falls back to the cheap
        in-place backbone-preserving swap.
    relative_to_wildtype : bool
        If True (default), score is reported as `score(mutant) - score(WT)`
        so positive numbers mean "better than wild-type" under
        `objective="maximize"`.
    """

    def __init__(
        self,
        score_fn: Callable[[ActiveSite], Any],
        repacker: Optional[Repacker] = None,
        relative_to_wildtype: bool = True,
    ):
        self.score_fn = score_fn
        self.repacker = repacker
        self.relative_to_wildtype = relative_to_wildtype
        self._wt_cache: Dict[int, Tuple[float, Dict[str, float]]] = {}

    def score(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ScoredMutationSet:
        mut_site, canonical = apply_mutations(active_site, mutations, self.repacker)
        s_mut, br_mut = self._invoke(mut_site)

        if not self.relative_to_wildtype:
            return ScoredMutationSet(MutationSet(tuple(canonical)), s_mut, br_mut)

        wt_key = id(active_site)
        if wt_key not in self._wt_cache:
            self._wt_cache[wt_key] = self._invoke(active_site)
        s_wt, br_wt = self._wt_cache[wt_key]

        delta_br = {f"delta_{k}": br_mut.get(k, 0.0) - br_wt.get(k, 0.0)
                    for k in br_mut.keys() | br_wt.keys()}
        return ScoredMutationSet(
            MutationSet(tuple(canonical)),
            s_mut - s_wt,
            delta_br,
            metadata={"mutant": br_mut, "wildtype": br_wt},
        )

    def _invoke(self, site: ActiveSite) -> Tuple[float, Dict[str, float]]:
        out = self.score_fn(site)
        if isinstance(out, dict):
            score = float(out.get("score"))
            br = {k: float(v) for k, v in out.items() if k != "score"
                  and isinstance(v, (int, float))}
            return score, br
        return float(out), {}


# ── Attribution-based scorer ───────────────────────────────────────────────────

@dataclass
class AttributionScorerConfig:
    """Knobs for the model-free attribution scorer."""

    catalytic_radius: float = 6.0    # Å, neighbourhood around catalytic residues
    catalytic_weight: float = 3.0    # multiplier for catalytic-shell atoms
    feature_weights: Optional[Dict[str, float]] = None
    objective_feature: str = "voip"  # column whose change drives the score


class DynamicsScorer:
    """ΔΔG_unfold scorer using the SPD-manifold dynamics pipeline.

    Wraps :func:`tope.dynamics.delta_delta_g_unfold` so the cross-
    protein N-bias from the unit offset cancels by construction
    (matched-N point mutations). For indels, the offset persists and
    the caller should be aware of the calibration caveat.

    The scorer's sign convention matches the engine's
    ``objective="maximize"`` default: a *destabilising* mutation
    produces a *negative* ΔΔG (mutant unfolds more readily ⇒ less
    stable ⇒ ΔG_unfold goes down ⇒ ΔΔG < 0). Flip via the engine's
    objective knob to search for destabilisers instead.
    """

    def __init__(
        self,
        cfg=None,
        T: float = 298.15,
        repacker=None,
    ):
        from tope.dynamics.free_energy import FreeEnergyConfig
        from tope.engineering.mutation import apply_mutations as _apply
        self.cfg = cfg or FreeEnergyConfig()
        self.T = float(T)
        self.repacker = repacker
        self._apply = _apply

    def score(
        self,
        active_site,
        mutations,
    ):
        from tope.dynamics.free_energy import delta_delta_g_unfold
        mut_site, canonical = self._apply(active_site, mutations, self.repacker)
        ddg = delta_delta_g_unfold(active_site, mut_site, self.T, self.cfg)
        breakdown = {
            "delta_delta_g_unfold_kcal": ddg,
            "T_K": self.T,
            "n_mutations": float(len(canonical)),
            "entropy_mode": float(hash(self.cfg.entropy_mode) % 1_000_000),
        }
        return ScoredMutationSet(
            MutationSet(tuple(canonical)),
            score=ddg,
            breakdown=breakdown,
        )


class AttributionScorer:
    """Model-free scorer using Ioffe descriptors weighted by proximity to
    the catalytic shell.

    The score is the catalytic-weighted L1 change in the chosen Ioffe
    descriptor (default: VOIP) between mutant and wild-type, which acts
    as a cheap stand-in for "did the mutation perturb the electronic
    environment of the active site?". Higher score ⇒ bigger perturbation.

    This is intentionally simple: it's a useful baseline / smoke-test
    scorer that doesn't require a trained model.
    """

    def __init__(
        self,
        cfg: Optional[AttributionScorerConfig] = None,
        feature_computer: Optional[FeatureComputer] = None,
        repacker: Optional[Repacker] = None,
    ):
        self.cfg = cfg or AttributionScorerConfig()
        self.feature_computer = feature_computer or FeatureComputer(
            compute_sasa=False, normalise=False,
        )
        self.repacker = repacker
        self._wt_cache: Dict[int, ActiveSiteFeatures] = {}

    def score(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ScoredMutationSet:
        wt_feats = self._cached_wt(active_site)
        mut_site, canonical = apply_mutations(active_site, mutations, self.repacker)
        mut_feats = self.feature_computer.compute(mut_site)

        weights = self._catalytic_weights(wt_feats)
        col = self._feature_index(self.cfg.objective_feature)

        wt_col = self._feature_column(wt_feats, col, default=0.0)
        mut_col = self._feature_column(mut_feats, col, default=0.0)
        n = min(len(wt_col), len(mut_col))
        w = weights[:n]
        delta = np.abs(mut_col[:n] - wt_col[:n])
        score = float((w * delta).sum())

        breakdown = {
            "n_atoms_wt": float(len(wt_col)),
            "n_atoms_mut": float(len(mut_col)),
            "delta_l1": float(delta.sum()),
            "weighted_delta": score,
            "catalytic_atoms": float((weights > 1.0).sum()),
        }
        return ScoredMutationSet(MutationSet(tuple(canonical)), score, breakdown)

    # ── helpers ──────────────────────────────────────────────────────────

    def _cached_wt(self, site: ActiveSite) -> ActiveSiteFeatures:
        key = id(site)
        if key not in self._wt_cache:
            self._wt_cache[key] = self.feature_computer.compute(site)
        return self._wt_cache[key]

    def _catalytic_weights(self, feats: ActiveSiteFeatures) -> np.ndarray:
        coords = feats.coords
        mask = feats.catalytic_mask
        if coords is None or mask is None or coords.shape[0] == 0:
            return np.ones(0)
        if not mask.any():
            return np.ones(coords.shape[0])
        cat_coords = coords[mask]
        diff = coords[:, None, :] - cat_coords[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)
        w = np.ones_like(d)
        near = d <= self.cfg.catalytic_radius
        w[near] = self.cfg.catalytic_weight
        return w

    def _feature_index(self, name: str) -> int:
        from tope.data.config import IOFFE_PROPERTY_KEYS
        return IOFFE_PROPERTY_KEYS.index(name)

    @staticmethod
    def _feature_column(
        feats: ActiveSiteFeatures, col: int, default: float = 0.0,
    ) -> np.ndarray:
        fm = feats.feature_matrix
        if fm is None or fm.size == 0:
            return np.zeros(0)
        return fm[:, col]
