"""Label encoding: curated record labels → multi-task training targets.

``ToPEDataset`` / ``collate_enzyme_pcc`` carry raw labels through on each batch
(EC-number strings and the three log-kinetics floats), deliberately leaving the
head-specific encoding to a vocabulary that must be fit across the dataset.
This module is that encoder — it turns those raw labels into the ``targets``
dict ``MultiTaskLoss`` consumes:

    {
      "ec_levels"    : [ (B,), (B,), (B,), (B,) ]  hierarchical class indices,
      "kinetics"     : (B, 3)  log[kcat, Km, kcat/Km] (missing → 0),
      "kinetics_mask": (B, 3)  1.0 where the value is present,
      "selectivity"  : None,   # not produced by the curation pipeline
    }

EC numbers are hierarchical (``"3.4.21.1"``): level ``k`` is keyed on the first
``k+1`` fields (``"3"``, ``"3.4"``, ``"3.4.21"``, ``"3.4.21.1"``).  Index 0 at
every level is reserved for ``<unk>`` so EC numbers / partial labels unseen at
fit time (or with ``-`` placeholder fields) degrade gracefully.  The fitted
``level_sizes`` are what the EC head's ``ec_levels`` must be configured with.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


_UNK = 0  # reserved index for unknown / missing tokens at every level


def _ec_fields(ec_number: str) -> List[str]:
    """Split an EC string into clean fields, stopping at the first blank/'-'."""
    fields: List[str] = []
    for part in str(ec_number).strip().split("."):
        part = part.strip()
        if part == "" or part == "-":
            break
        fields.append(part)
    return fields


def _level_keys(ec_number: str, n_levels: int) -> List[Optional[str]]:
    """Cumulative level keys: [ "3", "3.4", "3.4.21", "3.4.21.1" ].

    Levels beyond the available fields are ``None`` (→ ``<unk>``).
    """
    fields = _ec_fields(ec_number)
    keys: List[Optional[str]] = []
    for k in range(n_levels):
        keys.append(".".join(fields[: k + 1]) if k < len(fields) else None)
    return keys


class ECVocabulary:
    """Hierarchical EC-number vocabulary fitted over a dataset.

    Index 0 is ``<unk>`` at each of the ``n_levels`` levels.
    """

    def __init__(self, level_maps: List[Dict[str, int]], n_levels: int = 4) -> None:
        self.n_levels = n_levels
        self.level_maps = level_maps

    @classmethod
    def build(cls, ec_numbers: Sequence[str], n_levels: int = 4) -> "ECVocabulary":
        level_maps: List[Dict[str, int]] = [dict() for _ in range(n_levels)]
        for ec in ec_numbers:
            for k, key in enumerate(_level_keys(ec, n_levels)):
                if key is None:
                    continue
                if key not in level_maps[k]:
                    level_maps[k][key] = len(level_maps[k]) + 1  # 0 reserved for <unk>
        return cls(level_maps, n_levels=n_levels)

    @property
    def level_sizes(self) -> List[int]:
        """Class count per level (incl. the reserved ``<unk>`` slot).

        Use these as the EC head's ``ec_levels`` so every encoded index is in
        range.
        """
        return [len(m) + 1 for m in self.level_maps]

    def encode(self, ec_number: str) -> List[int]:
        """Encode one EC string to ``n_levels`` class indices (0 = ``<unk>``)."""
        return [
            self.level_maps[k].get(key, _UNK) if key is not None else _UNK
            for k, key in enumerate(_level_keys(ec_number, self.n_levels))
        ]

    def encode_batch(self, ec_numbers: Sequence[str]) -> List["torch.Tensor"]:
        """Encode a batch to a list of ``n_levels`` ``(B,)`` long tensors."""
        if not HAS_TORCH:
            raise ImportError("encode_batch requires torch.")
        rows = [self.encode(ec) for ec in ec_numbers]  # (B, n_levels)
        cols = list(zip(*rows)) if rows else [()] * self.n_levels
        return [torch.tensor(c, dtype=torch.long) for c in cols]


def encode_targets(
    labels: Dict[str, object],
    vocab: ECVocabulary,
) -> Dict[str, object]:
    """Build the ``MultiTaskLoss`` targets dict from a collated batch's labels.

    Parameters
    ----------
    labels : the ``batch["labels"]`` dict from ``collate_enzyme_pcc`` —
        ``ec_number`` (list[str]) and ``kinetics`` ((B, 3) tensor, NaN where the
        value is missing).
    vocab : a fitted :class:`ECVocabulary`.

    Returns
    -------
    targets dict with ``ec_levels``, ``kinetics``, ``kinetics_mask`` and
    ``selectivity`` (None).
    """
    if not HAS_TORCH:
        raise ImportError("encode_targets requires torch.")
    ec_numbers = list(labels.get("ec_number", []))
    ec_levels = vocab.encode_batch(ec_numbers)

    kin = labels.get("kinetics")
    if kin is None:
        B = len(ec_numbers)
        kinetics = torch.zeros(B, 3)
        kinetics_mask = torch.zeros(B, 3)
    else:
        kin = kin if torch.is_tensor(kin) else torch.tensor(kin, dtype=torch.float32)
        kinetics_mask = (~torch.isnan(kin)).float()
        kinetics = torch.nan_to_num(kin, nan=0.0)

    return {
        "ec_levels": ec_levels,
        "kinetics": kinetics,
        "kinetics_mask": kinetics_mask,
        "selectivity": None,
    }


def active_tasks_for(targets: Dict[str, object]) -> Dict[str, bool]:
    """Curriculum flags implied by a targets dict.

    EC is always supervised; kinetics is active only when at least one value is
    present in the batch; selectivity is never produced by the curation pipeline.
    """
    kin_mask = targets.get("kinetics_mask")
    kinetics_active = bool(kin_mask is not None and float(kin_mask.sum()) > 0)
    return {"ec": True, "kinetics": kinetics_active, "selectivity": False}
