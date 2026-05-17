"""Smoke tests for the protein engineering engine."""

import numpy as np
import pytest

from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord
from tope.engineering import (
    AttributionScorer,
    BeamSearch,
    EngineConfig,
    EngineeringEngine,
    MCMCSearch,
    Mutation,
    MutationSet,
    SaturationScan,
    ToPEModelScorer,
    apply_mutations,
)


def _make_site() -> ActiveSite:
    """Tiny synthetic active site: HIS (catalytic) + ASP + SER + LEU."""
    residues = []
    spec = [
        ("HIS", 57, True, "C"),
        ("ASP", 102, False, "O"),
        ("SER", 195, False, "O"),
        ("LEU", 240, False, "C"),
    ]
    for name, num, cat, side_elem in spec:
        atoms = [
            AtomRecord("N", "N", np.array([num + 0.0, 0., 0.]), "A", name, num),
            AtomRecord("CA", "C", np.array([num + 1.0, 0., 0.]), "A", name, num),
            AtomRecord("C", "C", np.array([num + 2.0, 0., 0.]), "A", name, num),
            AtomRecord("O", "O", np.array([num + 2.0, 1., 0.]), "A", name, num),
            AtomRecord("CB", "C", np.array([num + 1.0, 1., 0.]), "A", name, num),
            AtomRecord("CG", side_elem, np.array([num + 1.0, 2., 0.]), "A", name, num),
        ]
        for a in atoms:
            a.is_catalytic = cat
        residues.append(ResidueRecord(
            chain_id="A", residue_name=name, residue_number=num,
            ca_coord=atoms[1].coord, n_atoms=len(atoms),
            is_catalytic=cat, atoms=atoms,
        ))
    return ActiveSite(pdb_id="test", residues=residues, ec_number="3.4.21.1")


def test_apply_mutations_drops_sidechain():
    site = _make_site()
    mut = Mutation("A", 240, "ALA")
    new_site, canonical = apply_mutations(site, [mut])
    assert len(canonical) == 1
    assert canonical[0].source_aa == "LEU"
    leu = [r for r in new_site.residues if r.residue_number == 240][0]
    assert leu.residue_name == "ALA"
    atom_names = {a.name.upper() for a in leu.atoms}
    assert "CG" not in atom_names           # sidechain dropped
    assert {"N", "CA", "C", "O", "CB"} <= atom_names | {"CB"}  # backbone kept


def test_mutation_set_rejects_duplicates():
    with pytest.raises(ValueError):
        MutationSet((Mutation("A", 57, "ALA"), Mutation("A", 57, "GLY")))


def test_attribution_scorer_signs():
    site = _make_site()
    scorer = AttributionScorer()
    s_same = scorer.score(site, [Mutation("A", 240, "VAL")]).score
    s_big = scorer.score(site, [Mutation("A", 240, "TRP")]).score
    assert s_same >= 0 and s_big >= 0
    # Mutating to a chemically different residue should perturb features
    # at least as much as a near-isosteric one.
    assert s_big >= s_same - 1e-6


def test_saturation_scan_runs():
    site = _make_site()
    engine = EngineeringEngine(
        scorer=AttributionScorer(),
        search=SaturationScan(top_k=5),
        cfg=EngineConfig(target_amino_acids=["ALA", "GLY", "VAL"], saturation_top_k=5),
    )
    out = engine.propose(site)
    assert 1 <= len(out) <= 5
    # All proposals should be single-point mutations.
    assert all(len(s.mutations) == 1 for s in out)
    # Catalytic HIS should be excluded.
    for s in out:
        for m in s.mutations:
            assert m.resnum != 57


def test_beam_search_returns_compound_mutations():
    site = _make_site()
    cfg = EngineConfig(
        target_amino_acids=["ALA", "VAL"], beam_width=3, beam_depth=2,
        return_top_n=20, saturation_top_k=3,
    )
    engine = EngineeringEngine(
        scorer=AttributionScorer(),
        search=BeamSearch(),
        cfg=cfg,
    )
    out = engine.propose(site)
    assert any(len(s.mutations) == 2 for s in out)


def test_mcmc_search_runs():
    site = _make_site()
    cfg = EngineConfig(
        target_amino_acids=["ALA", "GLY", "VAL", "LEU"],
        mcmc_steps=20, mcmc_seed=42, return_top_n=5,
    )
    engine = EngineeringEngine(
        scorer=AttributionScorer(), search=MCMCSearch(), cfg=cfg,
    )
    out = engine.propose(site)
    assert len(out) >= 1
    assert out[0].score >= out[-1].score   # sorted max-first


def test_tope_model_scorer_with_user_callable():
    """ToPEModelScorer wraps an arbitrary score_fn returning a dict."""
    site = _make_site()

    def fake_model(site_arg):
        # "Bigger residues are better": sum of residue volumes.
        from tope.data.config import RESIDUE_PROPERTIES
        score = sum(
            RESIDUE_PROPERTIES.get(r.residue_name, {}).get("residue_volume", 0.0)
            for r in site_arg.residues
        )
        return {"score": score, "n_residues": float(len(site_arg.residues))}

    scorer = ToPEModelScorer(fake_model, relative_to_wildtype=True)
    res = scorer.score(site, [Mutation("A", 240, "TRP")])
    # LEU (166.7) → TRP (227.8) ⇒ Δ ≈ +61.1
    assert res.score > 50
    assert "delta_n_residues" in res.breakdown


def test_engine_propose_ensemble():
    site = _make_site()
    cfg = EngineConfig(
        target_amino_acids=["ALA", "GLY", "VAL"],
        saturation_top_k=3, beam_width=2, beam_depth=2,
        mcmc_steps=10, mcmc_seed=1, return_top_n=5,
    )
    engine = EngineeringEngine(scorer=AttributionScorer(), cfg=cfg)
    out = engine.propose_ensemble(site)
    assert 1 <= len(out) <= 5
