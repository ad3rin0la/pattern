r"""Self-contained molecule featurisation: SMILES → graph (no RDKit).

The curation pipeline records substrate / product identities as SMILES strings
(from BRENDA / SABIO-RK), but nothing turned them into the atom graphs the
substrate/product cross-attention consumes.  RDKit is not available in all
deployments, so this module provides a dependency-free parser for the common
organic SMILES subset — enough to build a connectivity graph
(``node_features``, ``edge_index``), which is all ``MoleculeGNN`` needs.

Supported: organic-subset bare atoms (B C N O P S F Cl Br I and aromatic
lowercase), bracket atoms ``[...]`` (element, explicit H count, formal charge,
and ``@``/``@@`` chirality), bond symbols ``- = # : / \``,
branches ``( )``, ring-closure digits and ``%nn``, and disconnected components
``.``. Aromaticity *perception* is out of scope — lower-case atoms are flagged
aromatic, nothing more. Empty input is an explicit missing-molecule placeholder;
malformed non-empty input raises instead of silently changing the chemistry.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Element vocabulary for the atom one-hot; everything else → the "other" slot.
ATOM_VOCAB: Tuple[str, ...] = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H")
_VOCAB_INDEX = {e: i for i, e in enumerate(ATOM_VOCAB)}
_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "H"}
_AROMATIC_ORGANIC = {"b", "c", "n", "o", "p", "s"}
_BOND_ORDER = {"-": 1, "=": 2, "#": 3, ":": 4, "/": 1, "\\": 1}

# Atom features append aromaticity, charge, H count, degree, and the two
# bracket-atom tetrahedral markers (@ and @@). Bond features preserve order
# and directional stereo rather than reducing chemistry to connectivity.
MOL_FEAT_DIM = len(ATOM_VOCAB) + 1 + 6
MOL_EDGE_FEAT_DIM = 7  # order one-hot (4) + stereo one-hot (none, /, \\)


class _Atom:
    __slots__ = ("element", "aromatic", "charge", "n_h", "chirality")

    def __init__(self, element: str, aromatic: bool, charge: int, n_h: int,
                 chirality: int = 0) -> None:
        self.element = element
        self.aromatic = aromatic
        self.charge = charge
        self.n_h = n_h
        self.chirality = chirality


def _parse_bracket(token: str) -> _Atom:
    """Parse the inside of a bracket atom ``[...]`` (without the brackets)."""
    s = token
    i = 0
    # Skip leading isotope digits.
    while i < len(s) and s[i].isdigit():
        i += 1
    # Element symbol: one upper + optional lower, or aromatic lower.
    aromatic = False
    if i < len(s) and s[i].islower():
        element = s[i].upper()
        aromatic = True
        i += 1
    elif i < len(s):
        element = s[i]
        i += 1
        if i < len(s) and s[i].islower() and (element + s[i]) in _ORGANIC:
            element += s[i]
            i += 1
    else:
        element = "C"
    # Remainder: chirality (@/@@), H count, charge — scan for H and +/-.
    n_h = 0
    charge = 0
    chirality = 0
    while i < len(s):
        c = s[i]
        if c == "@":
            chirality = 2 if i + 1 < len(s) and s[i + 1] == "@" else 1
            i += chirality
        elif c == "H":
            i += 1
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            n_h = int(num) if num else 1
        elif c in "+-":
            sign = 1 if c == "+" else -1
            i += 1
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            if num:
                charge = sign * int(num)
            else:
                # Count repeated signs (e.g. "++").
                charge = sign
                while i < len(s) and s[i] == c:
                    charge += sign
                    i += 1
        else:
            i += 1  # skip chirality / unsupported tokens
    return _Atom(element, aromatic, charge, n_h, chirality)


def parse_smiles(smiles: str) -> Tuple[List[_Atom], List[Tuple[int, int, int, int]]]:
    """Parse a SMILES string into (atoms, bonds).

    Bonds are ``(i, j, order, stereo)`` with order 1/2/3/4 (aromatic)
    and stereo 0/1/2 (none, ``/``, ``\\``).
    Raises ``ValueError`` on a structurally invalid string.
    """
    atoms: List[_Atom] = []
    bonds: List[Tuple[int, int, int, int]] = []
    branch_stack: List[int] = []
    ring_bonds: Dict[str, Tuple[int, int]] = {}   # label → (atom_idx, order)
    prev: Optional[int] = None
    pending_bond: Optional[int] = None
    pending_stereo = 0

    s = smiles.strip()
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "(":
            if prev is None:
                raise ValueError("branch opens before any atom")
            branch_stack.append(prev)
            i += 1
        elif c == ")":
            if not branch_stack:
                raise ValueError("unbalanced ')'")
            prev = branch_stack.pop()
            i += 1
        elif c in _BOND_ORDER:
            if prev is None or pending_bond is not None:
                raise ValueError(f"bond token {c!r} has no following atom")
            pending_bond = _BOND_ORDER[c]
            pending_stereo = 1 if c == "/" else (2 if c == "\\" else 0)
            i += 1
        elif c == ".":
            prev = None
            pending_bond = None
            pending_stereo = 0
            i += 1
        elif c == "%":
            label = s[i + 1 : i + 3]
            i += 3
            prev = _close_ring(label, prev, pending_bond, pending_stereo, ring_bonds, bonds, atoms)
            pending_bond = None
            pending_stereo = 0
        elif c.isdigit():
            prev = _close_ring(c, prev, pending_bond, pending_stereo, ring_bonds, bonds, atoms)
            pending_bond = None
            pending_stereo = 0
            i += 1
        elif c == "[":
            j = s.index("]", i)
            atom = _parse_bracket(s[i + 1 : j])
            i = j + 1
            prev, pending_bond = _add_atom(
                atom, atoms, bonds, prev, pending_bond, pending_stereo
            )
            pending_stereo = 0
        else:
            element, aromatic, consumed = _read_organic(s, i)
            if element is None:
                raise ValueError(f"unexpected token {c!r} at {i}")
            i += consumed
            prev, pending_bond = _add_atom(
                _Atom(element, aromatic, 0, 0), atoms, bonds, prev, pending_bond,
                pending_stereo,
            )
            pending_stereo = 0

    if branch_stack:
        raise ValueError("unbalanced '('")
    if ring_bonds:
        raise ValueError(f"unclosed ring bond(s): {list(ring_bonds)}")
    if pending_bond is not None:
        raise ValueError("SMILES ends with a bond token")
    if not atoms:
        raise ValueError("empty molecule")
    return atoms, bonds


def _read_organic(s: str, i: int):
    """Read a bare organic-subset atom at position i → (element, aromatic, len)."""
    two = s[i : i + 2]
    if two in ("Cl", "Br"):
        return two, False, 2
    c = s[i]
    if c in "BCNOPSFI":
        return c, False, 1
    if c in _AROMATIC_ORGANIC:
        return c.upper(), True, 1
    return None, False, 0


def _add_atom(atom, atoms, bonds, prev, pending_bond, pending_stereo=0):
    idx = len(atoms)
    atoms.append(atom)
    if prev is not None:
        order = pending_bond if pending_bond is not None else (
            4 if atoms[prev].aromatic and atom.aromatic else 1
        )
        bonds.append((prev, idx, order, pending_stereo))
    return idx, None


def _close_ring(label, prev, pending_bond, pending_stereo, ring_bonds, bonds, atoms):
    if prev is None:
        raise ValueError("ring closure before any atom")
    if label in ring_bonds:
        other, other_order = ring_bonds.pop(label)
        order = pending_bond or other_order or (
            4 if atoms[other].aromatic and atoms[prev].aromatic else 1
        )
        bonds.append((other, prev, order, pending_stereo))
    else:
        ring_bonds[label] = (prev, pending_bond or 0)
    return prev


class MoleculeFeaturizer:
    """Featurise parsed atoms into a fixed-width node-feature matrix."""

    feat_dim: int = MOL_FEAT_DIM

    def featurize(
        self, atoms: List[_Atom], bonds: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        m = len(atoms)
        feats = np.zeros((m, self.feat_dim), dtype=np.float64)
        degree = np.zeros(m, dtype=np.float64)
        for a, b, *_ in bonds:
            degree[a] += 1
            degree[b] += 1
        for k, atom in enumerate(atoms):
            slot = _VOCAB_INDEX.get(atom.element, len(ATOM_VOCAB))  # "other"
            feats[k, slot] = 1.0
            feats[k, len(ATOM_VOCAB) + 1] = 1.0 if atom.aromatic else 0.0
            feats[k, len(ATOM_VOCAB) + 2] = float(atom.charge)
            feats[k, len(ATOM_VOCAB) + 3] = float(atom.n_h)
            feats[k, len(ATOM_VOCAB) + 4] = degree[k]
            feats[k, len(ATOM_VOCAB) + 5] = float(atom.chirality == 1)
            feats[k, len(ATOM_VOCAB) + 6] = float(atom.chirality == 2)
        return feats


def smiles_to_graph(
    smiles: str,
    featurizer: Optional[MoleculeFeaturizer] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a SMILES string to node, connectivity, and bond features.

    Edges are undirected (each bond emitted both ways). Empty input is an
    explicit missing-molecule placeholder. Invalid non-empty input raises;
    malformed chemistry must never silently become carbon during curation.

    Returns
    -------
    node_features : (M, MOL_FEAT_DIM)
    edge_index    : (2, 2E) int
    edge_features : (2E, MOL_EDGE_FEAT_DIM)
    """
    featurizer = featurizer or MoleculeFeaturizer()
    if not smiles or not smiles.strip():
        atoms, bonds = [_Atom("C", False, 0, 0)], []
    else:
        atoms, bonds = parse_smiles(smiles)

    feats = featurizer.featurize(atoms, bonds)
    if bonds:
        src = [b[0] for b in bonds] + [b[1] for b in bonds]
        dst = [b[1] for b in bonds] + [b[0] for b in bonds]
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_features = np.zeros((2 * len(bonds), MOL_EDGE_FEAT_DIM), dtype=np.float64)
        for k, bond in enumerate(bonds + bonds):
            order, stereo = bond[2], bond[3]
            edge_features[k, min(max(order, 1), 4) - 1] = 1.0
            edge_features[k, 4 + min(max(stereo, 0), 2)] = 1.0
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, MOL_EDGE_FEAT_DIM), dtype=np.float64)
    return feats, edge_index, edge_features
