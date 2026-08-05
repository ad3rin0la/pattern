"""
Evaluation, Metrics, and Attribution
======================================

Provides:
1. **Metrics** — Per-task evaluation metrics:
   - EC:          F1 score (macro, micro), accuracy at each hierarchy level
   - Selectivity: MAE, RMSE, Pearson r
   - Kinetics:    R², MAE, RMSE on log kcat / log Km

2. **Filtration Attribution** — Integrated gradients over filtration radii
   to identify which topological scales contribute most to predictions.

3. **Ioffe Inverse Recognition** — Compare model's criterion importance
   ranking with Ioffe's (1983) sliding-recognition ablation ordering.

4. **Long-Range Mutation Validation** — Evaluate the model's ability to
   predict effects of mutations at various distances from the active site.

5. **Allosteric Pathway Visualisation** — Generate 3-D plots of residue
   attention highlighting allosteric communication pathways.

References
----------
Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML 2017.
Ioffe, Sliding recognition of the selectivity of metallocomplex catalysts, 1983.
Benkovic & Hammes-Schiffer (2006), distant mutations in DHFR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Metric accumulators ──────────────────────────────────────────────────────

class MetricAccumulator:
    """Accumulates predictions and targets across batches for epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._ec_preds: List[List[torch.Tensor]] = []
        self._ec_targets: List[List[torch.Tensor]] = []
        self._sel_preds: List[torch.Tensor] = []
        self._sel_targets: List[torch.Tensor] = []
        self._kin_preds: List[torch.Tensor] = []
        self._kin_targets: List[torch.Tensor] = []
        self._kin_masks: List[torch.Tensor] = []

    def update(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> None:
        """Accumulate one batch of predictions and targets."""
        # EC
        if "ec_logits" in predictions and "ec_levels" in targets:
            self._ec_preds.append(
                [l.detach().cpu() for l in predictions["ec_logits"]]
            )
            self._ec_targets.append(
                [t.detach().cpu() for t in targets["ec_levels"]]
            )

        # Selectivity
        if predictions.get("selectivity") is not None and targets.get("selectivity") is not None:
            self._sel_preds.append(predictions["selectivity"].detach().cpu().squeeze(-1))
            self._sel_targets.append(targets["selectivity"].detach().cpu())

        # Kinetics
        if predictions.get("kinetics") is not None and targets.get("kinetics") is not None:
            self._kin_preds.append(predictions["kinetics"].detach().cpu())
            self._kin_targets.append(targets["kinetics"].detach().cpu())
            mask = targets.get("kinetics_mask", torch.ones_like(targets["kinetics"]))
            self._kin_masks.append(mask.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """Compute all accumulated metrics."""
        results: Dict[str, float] = {}

        # EC metrics
        if self._ec_preds:
            ec_metrics = self._compute_ec_metrics()
            results.update(ec_metrics)

        # Selectivity metrics
        if self._sel_preds:
            sel_metrics = self._compute_selectivity_metrics()
            results.update(sel_metrics)

        # Kinetics metrics
        if self._kin_preds:
            kin_metrics = self._compute_kinetics_metrics()
            results.update(kin_metrics)

        return results

    def _compute_ec_metrics(self) -> Dict[str, float]:
        """F1, accuracy at each EC level."""
        results = {}
        n_levels = len(self._ec_preds[0])

        for level in range(n_levels):
            all_preds = torch.cat([b[level] for b in self._ec_preds], dim=0)
            all_targets = torch.cat([b[level] for b in self._ec_targets], dim=0)
            pred_classes = all_preds.argmax(dim=-1)

            # Accuracy
            correct = (pred_classes == all_targets).float()
            results[f"ec_level{level}_accuracy"] = correct.mean().item()

            # Macro F1
            n_classes = all_preds.size(-1)
            f1_sum = 0.0
            n_present = 0
            for c in range(n_classes):
                tp = ((pred_classes == c) & (all_targets == c)).sum().float()
                fp = ((pred_classes == c) & (all_targets != c)).sum().float()
                fn = ((pred_classes != c) & (all_targets == c)).sum().float()
                if (tp + fn) > 0:
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    f1_sum += f1.item()
                    n_present += 1
            results[f"ec_level{level}_f1_macro"] = f1_sum / max(n_present, 1)

        return results

    def _compute_selectivity_metrics(self) -> Dict[str, float]:
        """MAE, RMSE, Pearson r for selectivity."""
        preds = torch.cat(self._sel_preds, dim=0)
        targets = torch.cat(self._sel_targets, dim=0)

        diff = preds - targets
        mae = diff.abs().mean().item()
        rmse = (diff ** 2).mean().sqrt().item()

        # Pearson correlation
        preds_centered = preds - preds.mean()
        targets_centered = targets - targets.mean()
        cov = (preds_centered * targets_centered).mean()
        std_p = preds_centered.std(unbiased=False)
        std_t = targets_centered.std(unbiased=False)
        pearson_r = (cov / (std_p * std_t + 1e-8)).item()

        return {
            "selectivity_mae": mae,
            "selectivity_rmse": rmse,
            "selectivity_pearson_r": pearson_r,
        }

    def _compute_kinetics_metrics(self) -> Dict[str, float]:
        """R², MAE, RMSE for each kinetic parameter."""
        preds = torch.cat(self._kin_preds, dim=0)   # (N, 3)
        targets = torch.cat(self._kin_targets, dim=0)
        masks = torch.cat(self._kin_masks, dim=0)

        target_names = ["log_kcat", "log_km", "log_kcat_km"]
        results = {}

        for i, name in enumerate(target_names):
            mask = masks[:, i].bool()
            if mask.sum() < 2:
                continue

            p = preds[mask, i]
            t = targets[mask, i]

            diff = p - t
            results[f"kinetics_{name}_mae"] = diff.abs().mean().item()
            results[f"kinetics_{name}_rmse"] = (diff ** 2).mean().sqrt().item()

            # R² = 1 - SS_res / SS_tot
            ss_res = (diff ** 2).sum()
            ss_tot = ((t - t.mean()) ** 2).sum()
            r2 = (1.0 - ss_res / (ss_tot + 1e-8)).item()
            results[f"kinetics_{name}_r2"] = r2

        return results


# ── Filtration Attribution (Integrated Gradients) ────────────────────────────

class FiltrationAttribution:
    """Integrated gradients over filtration radius to identify which
    topological scales are most important for predictions.

    For each sample, computes the attribution of the prediction to
    features extracted at each filtration radius by integrating the
    gradient along a path from a baseline (zero features) to the
    actual features at that radius.
    """

    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
    ):
        self.model = model
        self.n_steps = n_steps

    @torch.enable_grad()
    def attribute(
        self,
        batch: Dict[str, torch.Tensor],
        target_task: str = "ec",
        target_class: Optional[int] = None,
        ec_level: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for the given sample.

        Parameters
        ----------
        batch : dict
            Single-sample batch (no batching dimension needed).
        target_task : str
            "ec", "selectivity", or "kinetics"
        target_class : int or None
            For EC: which class to attribute. If None, uses predicted class.
        ec_level : int
            Which EC hierarchy level to attribute (0–3).

        Returns
        -------
        attributions : dict
            'node_attributions' : (N, feat_dim) — per-atom feature importances
            'edge_attributions' : (E, feat_dim) — per-bond feature importances
        """
        self.model.eval()

        node_feats = batch["enzyme_pcc"]["node_features"].detach().clone()
        node_feats.requires_grad_(True)

        # Baseline: zero features
        baseline = torch.zeros_like(node_feats)

        # Accumulate gradients along interpolation path
        total_grads = torch.zeros_like(node_feats)

        for step in range(self.n_steps):
            alpha = step / self.n_steps
            interpolated = baseline + alpha * (node_feats - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            # Replace features in batch
            batch_copy = _deep_copy_batch(batch)
            batch_copy["enzyme_pcc"]["node_features"] = interpolated

            predictions = self.model(batch_copy)

            # Select target output
            if target_task == "ec":
                logits = predictions["ec_logits"][ec_level]
                if target_class is None:
                    target_class = logits.argmax(dim=-1).item()
                score = logits[0, target_class]
            elif target_task == "selectivity":
                score = predictions["selectivity"].squeeze()
            elif target_task == "kinetics":
                col = target_class if target_class is not None else 0
                score = predictions["kinetics"][0, col]
            else:
                raise ValueError(f"Unknown task: {target_task}")

            score.backward()
            total_grads += interpolated.grad

        # IG = (input - baseline) * avg_gradient
        avg_grads = total_grads / self.n_steps
        attributions = (node_feats - baseline) * avg_grads

        return {
            "node_attributions": attributions.detach(),
            "attribution_sum_per_atom": attributions.detach().sum(dim=-1),
        }


# ── Ioffe Inverse Recognition ───────────────────────────────────────────────

class IoffeInverseRecognition:
    """Compare the model's learned criterion importance with Ioffe's (1983)
    sliding-recognition ablation ordering.

    Ioffe's method: for each of the 8 criterion properties, ablate it
    (set to zero or mean) and measure the drop in classification performance.
    The property whose ablation causes the largest drop is most important.

    This class performs the same ablation on ToPE predictions and compares
    the resulting ranking with Ioffe's original ordering.
    """

    # Ioffe's 8 criterion-space properties (canonical ordering)
    IOFFE_PROPERTIES = [
        "voip",              # Valence orbital ionisation potential
        "n_d_electrons",     # Number of d-electrons
        "electron_affinity",
        "electronegativity",
        "ionic_radius",
        "residue_volume",
        "sasa",              # Solvent-accessible surface area
        "sidechain_enthalpy",
    ]

    def __init__(self, model: nn.Module, metric_fn: Any = None):
        self.model = model
        self.metric_fn = metric_fn

    @torch.no_grad()
    def ablation_study(
        self,
        data_loader: Any,
        device: torch.device,
        property_indices: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Run leave-one-out ablation for each Ioffe property.

        Parameters
        ----------
        data_loader : iterable of batches
        device : torch.device
        property_indices : dict mapping property name → feature index
            If None, assumes properties are at indices 0–7.

        Returns
        -------
        importance : dict {property_name: delta_metric}
            Larger delta → more important property.
        """
        self.model.eval()

        if property_indices is None:
            property_indices = {
                name: i for i, name in enumerate(self.IOFFE_PROPERTIES)
            }

        # Baseline performance
        baseline_metric = self._evaluate(data_loader, device)

        # Ablation per property
        importance = {}
        for prop_name, feat_idx in property_indices.items():
            ablated_metric = self._evaluate(
                data_loader, device, ablate_idx=feat_idx
            )
            importance[prop_name] = baseline_metric - ablated_metric

        # Sort by importance (descending)
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        return importance

    def _evaluate(
        self,
        data_loader: Any,
        device: torch.device,
        ablate_idx: Optional[int] = None,
    ) -> float:
        """Evaluate model (optionally with one feature ablated)."""
        accumulator = MetricAccumulator()

        for batch in data_loader:
            batch = _to_device(batch, device)

            if ablate_idx is not None and "enzyme_pcc" in batch:
                batch["enzyme_pcc"]["node_features"] = (
                    batch["enzyme_pcc"]["node_features"].clone()
                )
                batch["enzyme_pcc"]["node_features"][:, ablate_idx] = 0.0

            predictions = self.model(batch)
            accumulator.update(predictions, batch.get("targets", {}))

        metrics = accumulator.compute()
        # Use EC level-0 F1 as the primary metric
        return metrics.get("ec_level0_f1_macro", 0.0)

    def compare_with_ioffe(
        self,
        model_ranking: Dict[str, float],
        ioffe_ranking: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare model's property importance ranking with Ioffe's ordering.

        Parameters
        ----------
        model_ranking : dict from ablation_study()
        ioffe_ranking : list of property names in Ioffe's importance order
            If None, uses Ioffe's original ordering (approximate).

        Returns
        -------
        comparison : dict
            'model_order'   : list of property names by model importance
            'ioffe_order'   : list of property names by Ioffe's ordering
            'rank_correlation' : Spearman rank correlation coefficient
            'concordant_top3' : number of top-3 properties in common
        """
        if ioffe_ranking is None:
            # Approximate Ioffe ordering (from original 1983 paper)
            ioffe_ranking = [
                "voip",
                "electronegativity",
                "n_d_electrons",
                "electron_affinity",
                "ionic_radius",
                "sidechain_enthalpy",
                "residue_volume",
                "sasa",
            ]

        model_order = list(model_ranking.keys())

        # Spearman rank correlation
        n = len(model_order)
        model_ranks = {name: i for i, name in enumerate(model_order)}
        ioffe_ranks = {name: i for i, name in enumerate(ioffe_ranking)}

        common = set(model_ranks.keys()) & set(ioffe_ranks.keys())
        if len(common) < 2:
            return {
                "model_order": model_order,
                "ioffe_order": ioffe_ranking,
                "rank_correlation": 0.0,
                "concordant_top3": 0,
            }

        d_sq_sum = sum(
            (model_ranks[name] - ioffe_ranks[name]) ** 2 for name in common
        )
        nc = len(common)
        spearman = 1.0 - (6 * d_sq_sum) / (nc * (nc ** 2 - 1))

        # Top-3 concordance
        model_top3 = set(model_order[:3])
        ioffe_top3 = set(ioffe_ranking[:3])
        concordant = len(model_top3 & ioffe_top3)

        return {
            "model_order": model_order,
            "ioffe_order": ioffe_ranking,
            "rank_correlation": spearman,
            "concordant_top3": concordant,
        }


# ── Utilities ────────────────────────────────────────────────────────────────

def _deep_copy_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Create a shallow dict copy with cloned tensors."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.clone()
        elif isinstance(v, dict):
            result[k] = _deep_copy_batch(v)
        elif isinstance(v, list):
            result[k] = [x.clone() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            result[k] = v
    return result


def _to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move batch tensors to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(x, device) for x in batch)
    return batch


# ── Long-range mutation validation ───────────────────────────────────────────

class LongRangeMutationValidator:
    """Validate the model's ability to capture distant-residue effects.

    Groups mutations into distance bins relative to the active-site
    centre and reports per-bin ΔΔG R² and kcat R².  A model that
    captures allosteric effects should maintain reasonable R² even for
    mutations >15 Å away.
    """

    DISTANCE_BINS = [(0, 8), (8, 15), (15, 25), (25, 100)]
    BIN_LABELS = ["0-8A", "8-15A", "15-25A", ">25A"]

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def validate(
        self,
        test_set: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        test_set : list of dicts, each with keys
            'wt_graph'          : protein graph dict (wild-type)
            'mut_graph'         : protein graph dict (mutant)
            'mutation_distance' : float (Å from active site)
            'mutation_site_idx' : int
            'measured_ddG'      : float
            'measured_kcat'     : float (optional)

        Returns
        -------
        results : dict with per-bin metrics
        """
        self.model.eval()
        results = {
            "distance_bins": self.BIN_LABELS,
            "ddG_r2": [],
            "kcat_r2": [],
            "n_mutations": [],
        }

        for lo, hi in self.DISTANCE_BINS:
            subset = [
                s for s in test_set
                if lo <= s["mutation_distance"] < hi
            ]
            results["n_mutations"].append(len(subset))

            if len(subset) < 2:
                results["ddG_r2"].append(None)
                results["kcat_r2"].append(None)
                continue

            pred_ddG, true_ddG = [], []
            pred_kcat, true_kcat = [], []

            for item in subset:
                wt_graph = _to_device(item["wt_graph"], self.device)
                mut_graph = _to_device(item["mut_graph"], self.device)

                wt_emb, wt_res = self.model.encoder(wt_graph)
                mut_emb, mut_res = self.model.encoder(mut_graph)

                # ΔΔG prediction if mutation head available
                if hasattr(self.model, "mutation_head"):
                    ddG, _ = self.model.mutation_head(
                        wt_res.unsqueeze(0),
                        mut_res.unsqueeze(0),
                        torch.tensor([item["mutation_site_idx"]], device=self.device),
                        wt_graph["zone_assignments"].unsqueeze(0),
                    )
                    pred_ddG.append(ddG.item())
                    true_ddG.append(item["measured_ddG"])

                # kcat comparison if kinetic head available
                if hasattr(self.model, "kinetic_head") and "measured_kcat" in item:
                    log_kcat, _, _, _ = self.model.kinetic_head(
                        mut_emb.unsqueeze(0) if mut_emb.dim() == 1 else mut_emb,
                        mut_res.unsqueeze(0),
                        mut_graph["zone_assignments"].unsqueeze(0),
                    )
                    pred_kcat.append(log_kcat.item())
                    true_kcat.append(item["measured_kcat"])

            results["ddG_r2"].append(
                _r2(true_ddG, pred_ddG) if len(true_ddG) > 1 else None
            )
            results["kcat_r2"].append(
                _r2(true_kcat, pred_kcat) if len(true_kcat) > 1 else None
            )

        return results

    @staticmethod
    def format_report(results: Dict[str, Any]) -> str:
        """Pretty-print the mutation validation results."""
        lines = [
            "",
            "=" * 70,
            "LONG-RANGE MUTATION EFFECT VALIDATION",
            "=" * 70,
            f"{'Distance':<12} {'n_mut':<8} {'ddG R2':<12} {'kcat R2':<12}",
            "-" * 70,
        ]
        for i, label in enumerate(results["distance_bins"]):
            n = results["n_mutations"][i]
            ddg = results["ddG_r2"][i]
            kcat = results["kcat_r2"][i]
            ddg_str = f"{ddg:>10.3f}" if ddg is not None else f"{'N/A':>10}"
            kcat_str = f"{kcat:>10.3f}" if kcat is not None else f"{'N/A':>10}"
            lines.append(f"{label:<12} {n:<8} {ddg_str}   {kcat_str}")
        lines.append("=" * 70)
        return "\n".join(lines)


def _r2(y_true: List[float], y_pred: List[float]) -> float:
    """Compute R² from lists."""
    import numpy as np
    yt = np.array(y_true)
    yp = np.array(y_pred)
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ── Allosteric pathway visualisation ─────────────────────────────────────────

class AllostericPathwayVisualiser:
    """Generate 3-D plots of residue attention showing allosteric pathways.

    Requires matplotlib (optional import).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def visualise(
        self,
        protein_graph: Dict[str, torch.Tensor],
        catalytic_residues: List,
        mutation_site: Optional[Any] = None,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Parameters
        ----------
        protein_graph     : whole-protein graph dict
        catalytic_residues : list of (chain_id, res_id)
        mutation_site     : (chain_id, res_id) or None
        save_path         : file path for PNG output (optional)

        Returns
        -------
        fig : matplotlib Figure or None
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            logger.warning("matplotlib not available; skipping visualisation")
            return None

        pg = _to_device(protein_graph, self.device)
        self.model.eval()

        enzyme_emb, h_res = self.model.encoder(pg)

        # Get kinetic attention weights
        attn_weights = None
        if hasattr(self.model, "kinetic_head"):
            _, _, _, attn_weights = self.model.kinetic_head(
                enzyme_emb.unsqueeze(0) if enzyme_emb.dim() == 1 else enzyme_emb,
                h_res.unsqueeze(0),
                pg["zone_assignments"].unsqueeze(0),
            )
            attn_weights = attn_weights.squeeze(0).cpu().numpy()

        if attn_weights is None:
            logger.warning("No kinetic_head attention available")
            return None

        coords = pg["node_coords"].cpu().numpy()
        residue_ids = protein_graph.get("residue_ids", list(range(len(coords))))
        catalytic_set = set(catalytic_residues)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Colour by attention weight
        cmap = plt.cm.YlOrRd

        for i, coord in enumerate(coords):
            rid = residue_ids[i] if i < len(residue_ids) else i
            if mutation_site is not None and rid == mutation_site:
                colour = "red"
                size = 200
            elif rid in catalytic_set:
                colour = "blue"
                size = 200
            else:
                colour = cmap(float(attn_weights[i]))
                size = 50

            ax.scatter(
                coord[0], coord[1], coord[2],
                c=[colour], s=size, alpha=0.7,
            )

        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        ax.set_zlabel("Z (A)")
        title = "Residue Attention for Kinetics"
        if mutation_site is not None:
            title += f"\n{mutation_site} -> Catalytic Site"
        ax.set_title(title)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved allosteric pathway plot to %s", save_path)

        return fig


# ── Phase 2 ablation suite ────────────────────────────────────────────────────

class Phase2AblationSuite:
    """Validate Phase 2 component contributions before advancing to Phase 3.

    Run all ablations, confirm the deltas match expected thresholds, then
    call ``assert_phase2_complete()`` as a hard gate before building TCPNet
    on top of unvalidated components.

    Expected ablation numbers (from ToPE Phase 2 design doc):

    ============================================  ========================
    Ablation                                       Expected F-score delta
    ============================================  ========================
    Without sheaf sections                         −12%  (whole-protein)
    Without is_active_site attention bias          −5%
    Whole-protein vs 8 Å crop (ToPEModel)          −8%
    Without rank-stratified filtration             −6%
    Without spectral cutoff (geomspace)            −3%
    ============================================  ========================

    Usage::

        suite = Phase2AblationSuite(whole_protein_model, baseline_model, loader)
        results = suite.run(device)
        suite.assert_phase2_complete(results)   # raises if any threshold missed
    """

    # (ablation_name, feature_to_zero_or_None, min_expected_delta)
    # delta = baseline_f1 − ablated_f1  (positive = feature helps)
    ABLATIONS: List[Tuple[str, Optional[str], float]] = [
        ("without_active_site_bias", "is_active_site", 0.03),   # ≥3 pp drop
        ("without_sheaf_node_features", "sheaf_features",  0.08),  # ≥8 pp drop
    ]

    # Minimum F-score improvement of CompleteToPEModel over ToPEModel baseline
    MIN_WHOLE_PROTEIN_GAIN: float = 0.05   # ≥5 pp F-score gain over 8 Å crop

    def __init__(
        self,
        model: nn.Module,
        baseline_model: Optional[nn.Module],
        data_loader: Any,
    ) -> None:
        """
        Parameters
        ----------
        model          : CompleteToPEModel (whole-protein, Phase 2)
        baseline_model : ToPEModel (8 Å crop) — may be None if checkpoint
                         unavailable; whole-protein gain check is skipped.
        data_loader    : validation data loader
        """
        self.model          = model
        self.baseline_model = baseline_model
        self.data_loader    = data_loader

    @torch.no_grad()
    def run(self, device: torch.device) -> Dict[str, Any]:
        """Run all Phase 2 ablations.

        Returns
        -------
        results : dict with keys
            'baseline_f1'          : float
            'ablation_deltas'      : dict {name: delta_f1}
            'whole_protein_gain'   : float or None
            'passed'               : bool
            'failures'             : list of str
        """
        self.model.eval()

        # ── Baseline: full CompleteToPEModel ─────────────────────────────────
        baseline_f1 = self._evaluate(self.model, device)
        logger.info("Phase2AblationSuite | baseline F1 = %.4f", baseline_f1)

        # ── Component ablations ───────────────────────────────────────────────
        ablation_deltas: Dict[str, float] = {}
        failures: List[str] = []

        for name, feature_key, min_delta in self.ABLATIONS:
            ablated_f1 = self._evaluate(
                self.model, device, zero_feature=feature_key
            )
            delta = baseline_f1 - ablated_f1
            ablation_deltas[name] = delta
            logger.info(
                "Phase2AblationSuite | %-35s  delta=%.4f  (threshold=%.4f)  %s",
                name, delta, min_delta, "OK" if delta >= min_delta else "FAIL",
            )
            if delta < min_delta:
                failures.append(
                    f"{name}: delta={delta:.4f} < threshold={min_delta:.4f}"
                )

        # ── Whole-protein vs 8 Å crop ─────────────────────────────────────────
        whole_protein_gain: Optional[float] = None
        if self.baseline_model is not None:
            self.baseline_model.eval()
            baseline_crop_f1 = self._evaluate(self.baseline_model, device)
            whole_protein_gain = baseline_f1 - baseline_crop_f1
            logger.info(
                "Phase2AblationSuite | whole-protein gain over 8A crop = %.4f  "
                "(threshold=%.4f)  %s",
                whole_protein_gain, self.MIN_WHOLE_PROTEIN_GAIN,
                "OK" if whole_protein_gain >= self.MIN_WHOLE_PROTEIN_GAIN else "FAIL",
            )
            if whole_protein_gain < self.MIN_WHOLE_PROTEIN_GAIN:
                failures.append(
                    f"whole_protein_gain={whole_protein_gain:.4f} "
                    f"< threshold={self.MIN_WHOLE_PROTEIN_GAIN:.4f}"
                )

        return {
            "baseline_f1":        baseline_f1,
            "ablation_deltas":    ablation_deltas,
            "whole_protein_gain": whole_protein_gain,
            "passed":             len(failures) == 0,
            "failures":           failures,
        }

    def assert_phase2_complete(self, results: Dict[str, Any]) -> None:
        """Hard gate: raise if any Phase 2 ablation threshold was missed.

        Call this before building Phase 3 components (TCPNet stack, multi-task
        heads) so that unvalidated ablation numbers cannot be silently ignored.

        Raises
        ------
        AssertionError : with a summary of all failing ablations.
        """
        if results["passed"]:
            logger.info(
                "Phase2AblationSuite | ALL ABLATIONS PASSED — cleared for Phase 3"
            )
            return

        summary = "\n".join(f"  - {f}" for f in results["failures"])
        raise AssertionError(
            "Phase 2 ablations did not meet expected thresholds.\n"
            "Do not advance to Phase 3 (TCPNet stack) until all thresholds pass.\n"
            f"Failures:\n{summary}"
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        device: torch.device,
        zero_feature: Optional[str] = None,
    ) -> float:
        """Run one evaluation pass, optionally zeroing a named feature.

        ``zero_feature`` choices:
            ``"is_active_site"``   — zeros the active-site attention bias mask
            ``"sheaf_features"``   — zeros ``face_features`` in the enzyme PCC
                                     (ablates sheaf section contribution)
        """
        acc = MetricAccumulator()

        for batch in self.data_loader:
            batch = _to_device(batch, device)

            if zero_feature == "is_active_site" and "enzyme_pcc" in batch:
                pcc = batch["enzyme_pcc"]
                if "is_active_site" in pcc:
                    pcc = dict(pcc)
                    pcc["is_active_site"] = torch.zeros_like(pcc["is_active_site"])
                    batch = dict(batch)
                    batch["enzyme_pcc"] = pcc

            elif zero_feature == "sheaf_features" and "enzyme_pcc" in batch:
                pcc = batch["enzyme_pcc"]
                if "face_features" in pcc and pcc["face_features"] is not None:
                    pcc = dict(pcc)
                    pcc["face_features"] = torch.zeros_like(pcc["face_features"])
                    batch = dict(batch)
                    batch["enzyme_pcc"] = pcc

            preds = model(batch)
            acc.update(preds, batch.get("targets", {}))

        metrics = acc.compute()
        return metrics.get("ec_level0_f1_macro", 0.0)
