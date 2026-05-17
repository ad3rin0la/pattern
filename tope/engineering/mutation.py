"""Mutation representation and structural application.

A `Mutation` is a (chain, residue_number, target_aa) triple. Applying a
mutation produces a new `ActiveSite` whose residue identity is updated.
The default `apply_mutations` performs an in-place residue swap that
relabels the residue and drops sidechain atoms beyond Cβ; for higher
fidelity, plug in an external repacker (PyRosetta, FoldX, …) via the
`Repacker` protocol.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from tope.data.active_site import ActiveSite, AtomRecord, ResidueRecord


# Backbone atoms preserved during a cheap residue swap.
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "CB"})


@dataclass(frozen=True)
class Mutation:
    """A point mutation: replace a residue's identity at a given location."""

    chain: str
    resnum: int
    target_aa: str   # three-letter, e.g. "ALA"
    source_aa: str = ""  # filled in by `apply_mutations`; optional input

    def __str__(self) -> str:
        src = self.source_aa or "???"
        return f"{src}{self.resnum}{self.target_aa}@{self.chain}"

    @property
    def key(self) -> Tuple[str, int]:
        return (self.chain, self.resnum)


@dataclass(frozen=True)
class MutationSet:
    """An ordered, hashable bundle of point mutations."""

    mutations: Tuple[Mutation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Guard against two mutations targeting the same residue.
        seen: set = set()
        for m in self.mutations:
            if m.key in seen:
                raise ValueError(
                    f"Conflicting mutations at {m.chain}:{m.resnum}"
                )
            seen.add(m.key)

    def __iter__(self):
        return iter(self.mutations)

    def __len__(self) -> int:
        return len(self.mutations)

    def __add__(self, other: "MutationSet | Mutation") -> "MutationSet":
        if isinstance(other, Mutation):
            return MutationSet(self.mutations + (other,))
        return MutationSet(self.mutations + other.mutations)

    def __str__(self) -> str:
        return "+".join(str(m) for m in self.mutations) or "WT"


class Repacker(Protocol):
    """Pluggable interface for external sidechain repackers.

    Implementations receive the wild-type `ActiveSite`, the requested
    mutations, and return a fully-repacked `ActiveSite` (with new
    sidechain atoms). The default engine pipeline calls a `Repacker`
    when supplied; otherwise it falls back to the in-place swap.
    """

    def repack(
        self,
        active_site: ActiveSite,
        mutations: Sequence[Mutation],
    ) -> ActiveSite: ...


def apply_mutations(
    active_site: ActiveSite,
    mutations: Iterable[Mutation],
    repacker: Optional[Repacker] = None,
) -> Tuple[ActiveSite, List[Mutation]]:
    """Return a new `ActiveSite` with the requested mutations applied.

    Returns the mutated site and the canonicalised list of mutations
    (with `source_aa` filled in). Mutations targeting residues not
    present in the site are silently dropped — callers should check
    the returned list length.

    If `repacker` is supplied, delegates to it. Otherwise performs an
    in-place swap: residue name is updated, sidechain atoms (anything
    beyond backbone N/CA/C/O/CB) are dropped from the atom record so
    downstream features no longer reflect the wild-type sidechain.
    """
    mutations = list(mutations)
    if repacker is not None:
        # Fill in source AA first so the repacker has full context.
        canonical = _canonicalise(active_site, mutations)
        return repacker.repack(active_site, canonical), canonical

    new_residues: List[ResidueRecord] = []
    canonical: List[Mutation] = []
    mut_by_key = {m.key: m for m in mutations}

    for res in active_site.residues:
        key = (res.chain_id, res.residue_number)
        mut = mut_by_key.get(key)
        if mut is None:
            new_residues.append(res)
            continue

        target = mut.target_aa.upper()
        source = res.residue_name.upper()
        canonical.append(
            Mutation(mut.chain, mut.resnum, target, source_aa=source)
        )

        kept_atoms: List[AtomRecord] = []
        for atom in res.atoms:
            if atom.name.upper() not in _BACKBONE_ATOMS:
                continue
            kept_atoms.append(AtomRecord(
                name=atom.name,
                element=atom.element,
                coord=np.array(atom.coord, dtype=np.float64),
                chain_id=atom.chain_id,
                residue_name=target,
                residue_number=atom.residue_number,
                b_factor=atom.b_factor,
                is_catalytic=atom.is_catalytic,
                role=atom.role,
            ))

        new_residues.append(ResidueRecord(
            chain_id=res.chain_id,
            residue_name=target,
            residue_number=res.residue_number,
            ca_coord=np.array(res.ca_coord, dtype=np.float64),
            mean_b_factor=res.mean_b_factor,
            n_atoms=len(kept_atoms) if kept_atoms else res.n_atoms,
            is_catalytic=res.is_catalytic,
            role=res.role,
            atoms=kept_atoms,
        ))

    new_site = ActiveSite(
        pdb_id=active_site.pdb_id,
        residues=new_residues,
        catalytic_residue_ids=set(active_site.catalytic_residue_ids),
        centroid=(
            np.array(active_site.centroid, dtype=np.float64)
            if active_site.centroid is not None else None
        ),
        radius_used=active_site.radius_used,
        ec_number=active_site.ec_number,
    )
    return new_site, canonical


def _canonicalise(
    active_site: ActiveSite,
    mutations: Iterable[Mutation],
) -> List[Mutation]:
    by_key = {(r.chain_id, r.residue_number): r for r in active_site.residues}
    out: List[Mutation] = []
    for m in mutations:
        res = by_key.get(m.key)
        if res is None:
            continue
        out.append(Mutation(
            m.chain, m.resnum, m.target_aa.upper(),
            source_aa=res.residue_name.upper(),
        ))
    return out
