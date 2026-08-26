"""Evaluation-only metrics for label-free latent domain discovery.

Reference labels from CATH, ECOD, or Pfam enter only here; the discovery head
and its training objectives never accept them.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tope.models.domain_discovery import permutation_match


def boundary_f1(
    assignments: torch.Tensor,
    reference_labels: torch.Tensor,
    sequence_index: Optional[torch.Tensor] = None,
    chain_index: Optional[torch.Tensor] = None,
    tolerance: int = 2,
) -> Dict[str, float]:
    """Boundary precision/recall/F1 with a residue-number tolerance."""
    n = assignments.size(0)
    sequence_index = sequence_index if sequence_index is not None else torch.arange(n)
    chain_index = chain_index if chain_index is not None else torch.zeros(n, dtype=torch.long)
    order = sorted(
        range(n), key=lambda i: (int(chain_index[i]), int(sequence_index[i]))
    )
    predicted_labels = assignments.argmax(dim=-1)

    def boundaries(labels):
        result = []
        for left, right in zip(order[:-1], order[1:]):
            if chain_index[left] != chain_index[right]:
                continue
            if labels[left] != labels[right]:
                result.append(int(sequence_index[left]))
        return result

    predicted = boundaries(predicted_labels)
    reference = boundaries(reference_labels)
    matched_reference = set()
    true_positive = 0
    for boundary in predicted:
        candidates = [
            (abs(boundary - ref), i) for i, ref in enumerate(reference)
            if i not in matched_reference and abs(boundary - ref) <= tolerance
        ]
        if candidates:
            _, best = min(candidates)
            matched_reference.add(best)
            true_positive += 1
    precision = true_positive / len(predicted) if predicted else float(not reference)
    recall = true_positive / len(reference) if reference else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def evaluate_domains(
    assignments: torch.Tensor,
    reference_labels: torch.Tensor,
    sequence_index: Optional[torch.Tensor] = None,
    chain_index: Optional[torch.Tensor] = None,
    perturbed_assignments: Optional[torch.Tensor] = None,
    tolerance: int = 2,
) -> Dict[str, float]:
    """Evaluate boundaries, domain count, and optional perturbation stability."""
    metrics = boundary_f1(
        assignments, reference_labels, sequence_index, chain_index, tolerance
    )
    predicted_count = int(assignments.argmax(dim=-1).unique().numel())
    reference_count = int(reference_labels.unique().numel())
    metrics.update({
        "predicted_domain_count": float(predicted_count),
        "reference_domain_count": float(reference_count),
        "domain_count_error": float(abs(predicted_count - reference_count)),
    })
    if perturbed_assignments is not None:
        matched = permutation_match(assignments, perturbed_assignments)
        metrics["perturbation_stability_mse"] = float(
            torch.mean((assignments - matched).square()).detach()
        )
    return metrics
