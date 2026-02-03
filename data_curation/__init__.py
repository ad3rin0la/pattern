"""
ToPE Data Curation Pipeline
============================
Curate enzyme active-site datasets from M-CSA + PDB + BRENDA/SABIO-RK
for topological pattern recognition in enzyme catalysis.

Phase 1 of the ToPE roadmap: Baseline & Data.
"""

from data_curation.pipeline import CurationPipeline
from data_curation.kinetics_client import KineticsAggregator

__all__ = ["CurationPipeline", "KineticsAggregator"]
