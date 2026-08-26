"""Regression tests for substrate-specific observations and domain cells."""

from tope.data.kinetics_client import KineticEntry, KineticsAggregator
from tope.topology.whole_protein_pcc import DomainAnnotation, _build_domain_cells
from tope.models.multi_scale_graph import MultiScaleProteinGraph


def _entry(substrate, value, *, ph=7.0, temperature=30.0):
    return KineticEntry(
        ec_number="1.1.1.1", uniprot_id="P12345", pdb_id="1ABC",
        organism="test", substrate=substrate, param_type="kcat/Km",
        value=value, unit="s^-1 mM^-1", ph=ph, temperature=temperature,
    )


def test_kinetics_medians_do_not_cross_substrates_or_conditions(tmp_path):
    aggregator = KineticsAggregator(cache_dir=tmp_path)
    observations = aggregator.build_pdb_kinetics_map([
        _entry("ethanol", 10.0),
        _entry("ethanol", 1000.0),
        _entry("propanol", 1.0),
        _entry("ethanol", 100.0, ph=8.0),
    ])
    assert len(observations) == 3
    ethanol_ph7 = next(
        value for key, value in observations.items()
        if key.substrate_id == "ethanol" and key.ph == 7.0
    )
    assert ethanol_ph7["kcat/Km"] == 2.0  # median of log10(10), log10(1000)


def test_below_detection_assay_remains_a_separate_observation(tmp_path):
    aggregator = KineticsAggregator(cache_dir=tmp_path)
    negative = _entry("butanol", 0.0)
    negative.detected = False
    negative.detection_limit = 0.01
    observations = aggregator.build_pdb_kinetics_map([negative])
    assert len(observations) == 1
    assay = next(iter(observations.values()))
    assert assay == {"detected": False, "detection_limit_log": -2.0}


def test_domain_cells_preserve_family_and_chain_order():
    residues = {
        ("A", (" ", 1, " ")): [0],
        ("A", (" ", 2, " ")): [1],
        ("A", (" ", 3, " ")): [2],
        ("A", (" ", 4, " ")): [3],
    }
    groups, families, architecture = _build_domain_cells(residues, [
        DomainAnnotation("A", 3, 4, "PF00002", "C-terminal"),
        DomainAnnotation("A", 1, 2, "PF00001", "N-terminal"),
    ])
    assert architecture["A"] == [("A", "N-terminal"), ("A", "C-terminal")]
    assert families[("A", "N-terminal")] == "PF00001"
    assert len(groups[("A", "C-terminal")]) == 2

    graph_builder = object.__new__(MultiScaleProteinGraph)
    graph_builder.domain_annotations = [
        DomainAnnotation("A", 1, 2, "PF00001", "N-terminal"),
        DomainAnnotation("A", 3, 4, "PF00002", "C-terminal"),
    ]
    membership, graph_domains, graph_architecture = graph_builder._domain_cells(
        [("A", 1), ("A", 2), ("A", 3), ("A", 4)]
    )
    assert membership.tolist() == [0, 0, 1, 1]
    assert [d["family"] for d in graph_domains] == ["PF00001", "PF00002"]
    assert graph_architecture == {"A": [0, 1]}
