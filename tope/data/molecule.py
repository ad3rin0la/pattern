"""Self-contained molecule featurisation: SMILES → graph (no RDKit).

The curation pipeline records substrate / product identities as SMILES strings
(from BRENDA / SABIO-RK), but nothing turned them into the atom graphs the
substrate/product cross-attention consumes.  RDKit is not available in all
deployments, so this module provides a dependency-free parser for the common
organic SMILES subset — enough to build a connectivity graph
(``node_features``, ``edge_index``), which is all ``MoleculeGNN`` needs.

Supported: organic-subset bare atoms (B C N O P S F Cl Br I and aromatic
lowercase), bracket atoms ``[...]`` (element, explicit H count, formal charge;
isotope / chirality are parsed and ignored), bond symbols ``- = # : / \``,
branches ``( )``, ring-closure digits and ``%nn``, and disconnected components
``.``.  Stereochemistry and aromaticity *perception* are out of scope — lower
case atoms are flagged aromatic, nothing more.  Unparseable input degrades to a
single dummy carbon so the model still runs.
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

# node feature width: element one-hot (+ "other") + [aromatic, charge, n_h, degree]
MOL_FEAT_DIM = len(ATOM_VOCAB) + 1 + 4


class _Atom:
    __slots__ = ("element", "aromatic", "charge", "n_h")

    def __init__(self, element: str, aromatic: bool, charge: int, n_h: int) -> None:
        self.element = element
        self.aromatic = aromatic
        self.charge = charge
        self.n_h = n_h


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
    while i < len(s):
        c = s[i]
        if c == "H":
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
    return _Atom(element, aromatic, charge, n_h)


def parse_smiles(smiles: str) -> Tuple[List[_Atom], List[Tuple[int, int, int]]]:
    """Parse a SMILES string into (atoms, bonds).

    bonds are ``(i, j, order)`` with order 1/2/3/4 (4 = aromatic).
    Raises ``ValueError`` on a structurally invalid string.
    """
    atoms: List[_Atom] = []
    bonds: List[Tuple[int, int, int]] = []
    branch_stack: List[int] = []
    ring_bonds: Dict[str, Tuple[int, int]] = {}   # label → (atom_idx, order)
    prev: Optional[int] = None
    pending_bond: Optional[int] = None

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
            pending_bond = _BOND_ORDER[c]
            i += 1
        elif c == ".":
            prev = None
            pending_bond = None
            i += 1
        elif c == "%":
            label = s[i + 1 : i + 3]
            i += 3
            prev = _close_ring(label, prev, pending_bond, ring_bonds, bonds)
            pending_bond = None
        elif c.isdigit():
            prev = _close_ring(c, prev, pending_bond, ring_bonds, bonds)
            pending_bond = None
            i += 1
        elif c == "[":
            j = s.index("]", i)
            atom = _parse_bracket(s[i + 1 : j])
            i = j + 1
            prev, pending_bond = _add_atom(atom, atoms, bonds, prev, pending_bond)
        else:
            element, aromatic, consumed = _read_organic(s, i)
            if element is None:
                raise ValueError(f"unexpected token {c!r} at {i}")
            i += consumed
            prev, pending_bond = _add_atom(
                _Atom(element, aromatic, 0, 0), atoms, bonds, prev, pending_bond
            )

    if branch_stack:
        raise ValueError("unbalanced '('")
    if ring_bonds:
        raise ValueError(f"unclosed ring bond(s): {list(ring_bonds)}")
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


def _add_atom(atom, atoms, bonds, prev, pending_bond):
    idx = len(atoms)
    atoms.append(atom)
    if prev is not None:
        order = pending_bond if pending_bond is not None else 1
        bonds.append((prev, idx, order))
    return idx, None


def _close_ring(label, prev, pending_bond, ring_bonds, bonds):
    if prev is None:
        raise ValueError("ring closure before any atom")
    if label in ring_bonds:
        other, other_order = ring_bonds.pop(label)
        order = pending_bond or other_order or 1
        bonds.append((other, prev, order))
    else:
        ring_bonds[label] = (prev, pending_bond or 0)
    return prev


class MoleculeFeaturizer:
    """Featurise parsed atoms into a fixed-width node-feature matrix."""

    feat_dim: int = MOL_FEAT_DIM

    def featurize(
        self, atoms: List[_Atom], bonds: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        m = len(atoms)
        feats = np.zeros((m, self.feat_dim), dtype=np.float64)
        degree = np.zeros(m, dtype=np.float64)
        for a, b, _ in bonds:
            degree[a] += 1
            degree[b] += 1
        for k, atom in enumerate(atoms):
            slot = _VOCAB_INDEX.get(atom.element, len(ATOM_VOCAB))  # "other"
            feats[k, slot] = 1.0
            feats[k, len(ATOM_VOCAB) + 1] = 1.0 if atom.aromatic else 0.0
            feats[k, len(ATOM_VOCAB) + 2] = float(atom.charge)
            feats[k, len(ATOM_VOCAB) + 3] = float(atom.n_h)
            feats[k, len(ATOM_VOCAB) + 4 - 1] = degree[k]  # last slot = degree
        return feats


def smiles_to_graph(
    smiles: str,
    featurizer: Optional[MoleculeFeaturizer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a SMILES string to ``(node_features, edge_index)``.

    Edges are undirected (each bond emitted both ways).  Invalid or empty
    input degrades to a single dummy carbon (so downstream attention still has
    a key to attend to).

    Returns
    -------
    node_features : (M, MOL_FEAT_DIM)
    edge_index    : (2, 2E) int
    """
    featurizer = featurizer or MoleculeFeaturizer()
    try:
        atoms, bonds = parse_smiles(smiles)
    except (ValueError, IndexError):
        atoms, bonds = [_Atom("C", False, 0, 0)], []

    feats = featurizer.featurize(atoms, bonds)
    if bonds:
        src = [b[0] for b in bonds] + [b[1] for b in bonds]
        dst = [b[1] for b in bonds] + [b[0] for b in bonds]
        edge_index = np.array([src, dst], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return feats, edge_index
