"""Node-budget optimisation for the Biohub cell-tracking metric.

The competition's adjusted edge Jaccard is

    J_adj = max(0, J_edge * (1 - 0.1 * (N_pred - N_true) / N_true))

and the ``max`` clamps only the bottom, so emitting *fewer* nodes than the estimated
true count multiplies the score upward. Pruning the least-confident detections trades
edge recall against that multiplier; because the low-confidence tail is enriched in
false positives, the raw Jaccard frequently rises over the first stretch of pruning as
well. This module finds the optimum empirically.

See docs/METRIC_ANALYSIS.md, property 1.
"""

from __future__ import annotations

from dataclasses import dataclass

ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1


def adjusted_edge_jaccard(edge_jaccard: float, n_pred: int, n_true: float) -> float:
    """Adjusted edge Jaccard for one sample. Mirrors ``per_sample_metrics``."""
    if n_true <= 0:
        return float("nan")
    ratio = (n_pred - n_true) / n_true
    return max(0.0, edge_jaccard * (1.0 - ADJUSTMENT_ALPHA * ratio))


def combined_score(adj_edge_jaccard: float, division_jaccard: float) -> float:
    """Run-level combined score."""
    if division_jaccard != division_jaccard:  # NaN => no divisions anywhere
        return adj_edge_jaccard
    return adj_edge_jaccard + SCORE_DIVISION_WEIGHT * division_jaccard


@dataclass
class PruneResult:
    keep_fraction: float
    threshold: float
    edge_jaccard: float
    adj_edge_jaccard: float
    n_pred: int


def sweep_node_budget(
    node_confidences,
    evaluate_subset,
    n_true: float,
    steps: int = 40,
    min_keep: float = 0.55,
) -> list[PruneResult]:
    """Sweep a confidence cut-off and report the adjusted Jaccard at each level.

    Parameters
    ----------
    node_confidences
        Sequence of per-node confidence scores for one sample, any scale.
    evaluate_subset
        Callable ``(threshold) -> (edge_tp, edge_fp, edge_fn, n_pred)`` that rebuilds the
        tracking graph using only nodes whose confidence is >= ``threshold`` and scores it
        against ground truth. This is the expensive part and is supplied by the caller so
        that this module stays free of tracksdata/zarr dependencies.
    n_true
        ``estimated_number_of_nodes`` from the sample's ``.geff`` metadata. Note this is
        the estimated *true total* cell count, not the number of annotated nodes.
    steps
        Number of cut-offs to try between ``min_keep`` and keeping everything.
    min_keep
        Lowest keep-fraction to explore. Below ~0.5 the multiplier gain cannot offset the
        recall loss for any realistic detector, so there is no point going further.

    Returns
    -------
    list[PruneResult] ordered from most aggressive pruning to no pruning.
    """
    conf = sorted(node_confidences)
    if not conf:
        return []

    results: list[PruneResult] = []
    for i in range(steps + 1):
        keep_frac = min_keep + (1.0 - min_keep) * i / steps
        idx = int(round((1.0 - keep_frac) * (len(conf) - 1)))
        threshold = conf[idx]

        tp, fp, fn, n_pred = evaluate_subset(threshold)
        denom = tp + fp + fn
        j = tp / denom if denom > 0 else float("nan")
        results.append(
            PruneResult(
                keep_fraction=keep_frac,
                threshold=threshold,
                edge_jaccard=j,
                adj_edge_jaccard=adjusted_edge_jaccard(j, n_pred, n_true),
                n_pred=n_pred,
            )
        )
    return results


def best_of(results: list[PruneResult]) -> PruneResult | None:
    """Pick the sweep point with the highest adjusted Jaccard.

    Prefers the *least* aggressive cut among near-ties (within 1e-4), because a flatter
    operating point generalises better to an unseen embryo than a sharp peak does.
    """
    valid = [r for r in results if r.adj_edge_jaccard == r.adj_edge_jaccard]
    if not valid:
        return None
    peak = max(r.adj_edge_jaccard for r in valid)
    near = [r for r in valid if r.adj_edge_jaccard >= peak - 1e-4]
    return max(near, key=lambda r: r.keep_fraction)
