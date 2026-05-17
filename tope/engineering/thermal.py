"""Temperature-aware composite scorer for the engineering engine.

Decomposes observed activity into three multiplicative factors that map
onto distinct ToPE representations:

    kcat_obs(T)  ∝  f_folded(T)  ·  exp(−Ea / R T)  ·  γ_coop(T)

* ``f_folded``     : Boltzmann population of the folded state, governed by
                    ΔG_unfold. Whole-protein PCC topology signal.
* ``exp(−Ea/RT)``  : Arrhenius gain at the chemical step. Active-site
                    Hodge spectrum signal (see
                    ``tope.models.p_laplacian.predict_arrhenius_slope``).
* ``γ_coop``       : Folding-cooperativity correction (NOT allosteric
                    Hill cooperativity — that lives in
                    ``AllostericCooperativityHead``). Rank-2 non-active-
                    site spectral signal.

This file ships **stub** implementations of each sub-scorer so the
engine API is wired end-to-end before the supervised heads land. Each
stub takes a user-supplied callable returning the relevant
thermodynamic quantity (ΔG_unfold, Ea, ΔG_open); the default callables
return constants so the factor degenerates to 1.0 and the composite
reduces to whatever sub-scorers the user actually wires up.

The composite reports the **log-ratio** mutant/wild-type so search
strategies see an additive objective:

    score = log f_mut/f_wt  +  (Ea_wt − Ea_mut)/(R T)  +  log γ_mut/γ_wt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from tope.data.active_site import ActiveSite
from tope.engineering.mutation import Mutation, MutationSet, Repacker, apply_mutations
from tope.engineering.scoring import ScoredMutationSet


# Gas constant in kcal/(mol·K) — matches Ea reported in kcal/mol throughout
# the BRENDA / SABIO-RK literature.
R_KCAL = 1.987204e-3


# ── Sub-scorer protocols ──────────────────────────────────────────────────────

class FoldedFractionScorer(Protocol):
    """Return f_folded ∈ (0, 1] for an active site at temperature T."""

    def __call__(self, active_site: ActiveSite, T: float) -> float: ...


class ArrheniusEaScorer(Protocol):
    """Return the activation energy Ea (kcal/mol) at the chemical step."""

    def __call__(self, active_site: ActiveSite, T: float) -> float: ...


class CooperativityScorer(Protocol):
    """Return the cooperativity correction γ ∈ (0, 1] at temperature T."""

    def __call__(self, active_site: ActiveSite, T: float) -> float: ...


# ── Default stubs ─────────────────────────────────────────────────────────────

@dataclass
class BoltzmannFoldedFraction:
    """f_folded(T) = 1 / (1 + exp(−ΔG_unfold(T) / R T)).

    The user supplies a callable returning ΔG_unfold in kcal/mol. The
    default constant ΔG = +5 kcal/mol gives a high but non-saturated
    folded fraction near room T — a deliberately sane stub.
    """

    delta_g_unfold: Callable[[ActiveSite, float], float] = field(
        default=lambda site, T: 5.0
    )

    def __call__(self, active_site: ActiveSite, T: float) -> float:
        dG = float(self.delta_g_unfold(active_site, T))
        return 1.0 / (1.0 + math.exp(-dG / (R_KCAL * T)))


@dataclass
class ConstantEa:
    """Stub Ea scorer returning a fixed activation energy."""

    Ea_kcal_per_mol: float = 15.0

    def __call__(self, active_site: ActiveSite, T: float) -> float:
        return self.Ea_kcal_per_mol


@dataclass
class ConstantCooperativity:
    """Stub γ_coop scorer; degenerate to 1.0 by default."""

    gamma: float = 1.0

    def __call__(self, active_site: ActiveSite, T: float) -> float:
        return self.gamma


# ── Composite scorer ──────────────────────────────────────────────────────────

@dataclass
class ThermalCompositeScorer:
    """Compose folded-fraction, Arrhenius, and cooperativity factors.

    Implements the ``ScorerProtocol`` from ``tope.engineering.scoring``.
    Score is the log-ratio mutant/wild-type at ``T``; positive means
    "better than WT" under the engine's default ``maximize`` objective.

    Sub-scorers default to stubs returning constants — wire them up to
    trained heads as they land. The repacker hook is delegated to
    ``apply_mutations`` so user-supplied PyRosetta/FoldX behaviour is
    consistent with the rest of the engineering pipeline.
    """

    folded: FoldedFractionScorer = field(default_factory=BoltzmannFoldedFraction)
    ea: ArrheniusEaScorer = field(default_factory=ConstantEa)
    coop: CooperativityScorer = field(default_factory=ConstantCooperativity)
    T: float = 298.15
    repacker: Optional[Repacker] = None

    # Weighting of each factor in the log composite. Defaults to 1.0 each;
    # set a weight to 0.0 to disable that factor entirely.
    w_folded: float = 1.0
    w_arrhenius: float = 1.0
    w_coop: float = 1.0

    def __post_init__(self) -> None:
        self._wt_cache: Dict[int, Dict[str, float]] = {}

    def score(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ScoredMutationSet:
        wt = self._cached_wt(active_site)
        mut_site, canonical = apply_mutations(active_site, mutations, self.repacker)
        m = self._factors(mut_site)

        # Log-ratio per factor; (Ea_wt - Ea_mut)/(R T) follows from
        # log(exp(-Ea_mut/RT) / exp(-Ea_wt/RT)) = (Ea_wt - Ea_mut)/(R T).
        log_f_ratio = math.log(max(m["f_folded"], 1e-30)) \
                      - math.log(max(wt["f_folded"], 1e-30))
        log_arrh = (wt["Ea"] - m["Ea"]) / (R_KCAL * self.T)
        log_g_ratio = math.log(max(m["gamma"], 1e-30)) \
                      - math.log(max(wt["gamma"], 1e-30))

        score = (
            self.w_folded * log_f_ratio
            + self.w_arrhenius * log_arrh
            + self.w_coop * log_g_ratio
        )

        breakdown: Dict[str, float] = {
            "log_f_folded_ratio": log_f_ratio,
            "log_arrhenius_gain": log_arrh,
            "log_gamma_coop_ratio": log_g_ratio,
            "f_folded_wt": wt["f_folded"],
            "f_folded_mut": m["f_folded"],
            "Ea_wt": wt["Ea"],
            "Ea_mut": m["Ea"],
            "gamma_wt": wt["gamma"],
            "gamma_mut": m["gamma"],
            "T_K": self.T,
        }
        return ScoredMutationSet(
            MutationSet(tuple(canonical)),
            score,
            breakdown,
            metadata={"wildtype": wt, "mutant": m},
        )

    def kcat_curve(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
        temperatures_K: Sequence[float],
    ) -> List[Dict[str, float]]:
        """Sweep T and return the per-T composite log-ratio (volcano plot)."""
        out: List[Dict[str, float]] = []
        original_T = self.T
        try:
            for T in temperatures_K:
                self.T = float(T)
                self._wt_cache.clear()  # f, Ea, γ are T-dependent
                s = self.score(active_site, mutations)
                out.append({"T_K": T, "log_kcat_ratio": s.score, **s.breakdown})
        finally:
            self.T = original_T
            self._wt_cache.clear()
        return out

    # ── helpers ──────────────────────────────────────────────────────────

    def _cached_wt(self, site: ActiveSite) -> Dict[str, float]:
        key = id(site)
        if key not in self._wt_cache:
            self._wt_cache[key] = self._factors(site)
        return self._wt_cache[key]

    def _factors(self, site: ActiveSite) -> Dict[str, float]:
        return {
            "f_folded": float(self.folded(site, self.T)),
            "Ea": float(self.ea(site, self.T)),
            "gamma": float(self.coop(site, self.T)),
        }
