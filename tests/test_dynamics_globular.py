"""Globular-fold sanity test for the SPD ΔG_unfold formulation.

Catches sign errors that the collinear-chain regression can't see: on a
non-trivial tertiary geometry, ΔG_unfold must be positive at 298 K and
monotonically decreasing with T. Fails loudly if either the sorted-
pairing heuristic or the spectral sign convention is upside-down.
"""

import numpy as np
import pytest

from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord
from tope.dynamics import predict_delta_g_unfold, FreeEnergyConfig, NMAConfig


def _alpha_helix_coords(n_res: int = 20) -> np.ndarray:
    """Canonical α-helix Cα coordinates.

    Standard parameters: radius 2.3 Å, rise 1.5 Å/residue, 3.6
    residues/turn (≈100° per residue). Gives a non-collinear, locally-
    contacting geometry with nontrivial i,i+3 and i,i+4 stiffness —
    the simplest globular proxy that's defensible for a unit test.
    """
    r = 2.3
    rise = 1.5
    twist = np.deg2rad(100.0)
    i = np.arange(n_res)
    return np.stack(
        [r * np.cos(twist * i), r * np.sin(twist * i), rise * i],
        axis=1,
    )


def _site_from_coords(coords: np.ndarray) -> ActiveSite:
    residues = []
    for i, c in enumerate(coords):
        atom = AtomRecord("CA", "C", c, "A", "ALA", 100 + i)
        residues.append(ResidueRecord(
            chain_id="A", residue_name="ALA", residue_number=100 + i,
            ca_coord=c, n_atoms=1, atoms=[atom],
        ))
    return ActiveSite(pdb_id="helix", residues=residues, ec_number="")


def test_globular_helix_delta_g_unfold_sign_at_298K():
    """Trp-cage-scale α-helix: ΔG_unfold > 0 at room T."""
    site = _site_from_coords(_alpha_helix_coords(20))
    cfg = FreeEnergyConfig()
    dG = predict_delta_g_unfold(site, T=298.15, cfg=cfg)
    assert dG > 0, f"ΔG_unfold should be positive for folded helix; got {dG:.2f}"


def test_globular_helix_delta_g_monotone_in_T():
    """ΔG_unfold must decrease monotonically across a sweep of T."""
    site = _site_from_coords(_alpha_helix_coords(20))
    cfg = FreeEnergyConfig()
    Ts = np.linspace(260.0, 400.0, 8)
    dGs = np.array([predict_delta_g_unfold(site, T=float(T), cfg=cfg) for T in Ts])
    diffs = np.diff(dGs)
    assert (diffs < 0).all(), (
        f"ΔG_unfold(T) must be strictly decreasing; got dGs={dGs}, diffs={diffs}"
    )


def test_globular_helix_at_least_one_signed_zero_crossing():
    """Across a wide T sweep, ΔG_unfold should change sign somewhere
    (otherwise the model has no notion of a melting temperature)."""
    site = _site_from_coords(_alpha_helix_coords(20))
    cfg = FreeEnergyConfig()
    # Sweep up to very high T to ensure sign change is captured even
    # for the deliberately stiff helix.
    Ts = np.linspace(200.0, 2000.0, 40)
    dGs = np.array([predict_delta_g_unfold(site, T=float(T), cfg=cfg) for T in Ts])
    signs = np.sign(dGs)
    assert (signs[0] > 0) and (signs[-1] < 0), (
        f"ΔG must start positive and end negative under wide-T sweep; "
        f"endpoints = ({dGs[0]:.2f}, {dGs[-1]:.2f})"
    )
