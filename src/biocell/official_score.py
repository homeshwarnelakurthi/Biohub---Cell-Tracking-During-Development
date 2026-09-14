"""Score plain node/edge dicts with the organisers' metric code, unmodified.

Imports `tracking_cellmot.metrics` and `tracking_cellmot.division_metrics` from a
checkout of royerlab/kaggle-cell-tracking-competition at or after commit aa65e90
(the 2026-07-17 fix for the weakly-connected-component division exploit). Point
BIOCELL_METRIC_REPO at that checkout.

Coordinates are rounded and clipped the way the submission writer does, because
that is what the leaderboard scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo = os.environ.get("BIOCELL_METRIC_REPO")
if _repo:
    sys.path.insert(0, str(Path(_repo) / "src"))

import tracksdata as td  # noqa: E402
from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise  # noqa: E402

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


def load_gt(geff_path: Path):
    g = td.graph.IndexedRXGraph.from_geff(geff_path)
    return g[0] if isinstance(g, tuple) else g


def estimated_nodes(geff_path: Path) -> float:
    from geff import GeffMetadata
    try:
        meta = GeffMetadata.read(geff_path)
    except Exception:
        return float("nan")
    v = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(v) if v is not None else float("nan")


def plain_to_graph(nodes_by_id: dict, edges: list, rounded: bool = True):
    import polars as pl
    g = td.graph.IndexedRXGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, dtype=pl.Int64 if rounded else pl.Float64, default_value=0 if rounded else 0.0)
    ids = sorted(int(n) for n in nodes_by_id)
    rows = []
    for nid in ids:
        n = nodes_by_id[nid]
        rows.append({
            td.DEFAULT_ATTR_KEYS.T: int(n["t"]),
            "z": max(0, int(round(float(n["z"])))) if rounded else max(0.0, float(n["z"])),
            "y": max(0, int(round(float(n["y"])))) if rounded else max(0.0, float(n["y"])),
            "x": max(0, int(round(float(n["x"])))) if rounded else max(0.0, float(n["x"])),
        })
    if rows:
        g.bulk_add_nodes(rows, indices=ids)
    if edges:
        g.bulk_add_edges([{td.DEFAULT_ATTR_KEYS.EDGE_SOURCE: int(e["source_id"]),
                           td.DEFAULT_ATTR_KEYS.EDGE_TARGET: int(e["target_id"])} for e in edges])
    return g


def score_sample(nodes_by_id: dict, edges: list, gt_geff: Path, scale=DEFAULT_SCALE, rounded: bool = True) -> dict:
    pred = plain_to_graph(nodes_by_id, edges, rounded=rounded)
    gt = load_gt(gt_geff)
    er = evaluate(pred, gt, scale=scale, max_distance=7.0)
    rec = node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
    return per_sample_metrics(er, estimated_nodes(gt_geff), rec)


__all__ = ["score_sample", "summarise", "plain_to_graph", "load_gt", "estimated_nodes"]
