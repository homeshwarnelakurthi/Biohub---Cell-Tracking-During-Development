"""Submission writing and validation for the Biohub cell-tracking competition.

The important behaviour here is :func:`write_submission`'s edge ordering. The official
metric truncates out-degree by *edge ID*, not by confidence:

    edge_attrs.with_columns(pl.col(EDGE_ID).rank("ordinal").over(EDGE_SOURCE).alias("_out_rank"))
    edge_attrs.filter(pl.col("_out_rank") <= 2)

A node with three or more outgoing edges keeps whichever two were written first. The same
lowest-ID-wins rule decides which edge survives the merge-collapse step. Writing edges in
descending confidence order therefore converts a random loss into a deliberate choice, at
no cost. See docs/METRIC_ANALYSIS.md, property 3.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)  # (z, y, x)


def write_submission(path, per_dataset, *, sort_edges_by_confidence=True):
    """Write a competition submission CSV.

    Parameters
    ----------
    path
        Destination CSV path. Must be named ``submission.csv`` for Kaggle to accept it.
    per_dataset
        Mapping ``{dataset_name: (nodes, edges)}`` where

        * ``nodes`` is an iterable of ``(node_id, t, z, y, x)`` with integer voxel coords;
        * ``edges`` is an iterable of ``(source_id, target_id, confidence)``. Confidence may
          be ``None`` when unavailable, in which case ordering is left untouched.

        ``dataset_name`` must match the test folder name without the ``.zarr`` suffix.
    sort_edges_by_confidence
        Sort each dataset's edges by descending confidence before assigning row ids, so the
        metric's lowest-edge-id tie-breaks keep our best links.
    """
    path = Path(path)
    rows = []
    idx = 0

    for dataset in sorted(per_dataset):
        nodes, edges = per_dataset[dataset]

        for node_id, t, z, y, x in nodes:
            rows.append([idx, dataset, "node", int(node_id), int(t),
                         int(z), int(y), int(x), -1, -1])
            idx += 1

        edges = list(edges)
        if sort_edges_by_confidence and edges and edges[0][2] is not None:
            edges.sort(key=lambda e: e[2], reverse=True)

        for source_id, target_id, _conf in edges:
            rows.append([idx, dataset, "edge", -1, -1, -1, -1, -1,
                         int(source_id), int(target_id)])
            idx += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SUBMISSION_COLUMNS)
        w.writerows(rows)
    return path


def cap_out_degree(edges, max_children=2):
    """Drop the lowest-confidence outgoing edges beyond ``max_children`` per source.

    The metric silently truncates to two children anyway. Doing it ourselves — by
    confidence rather than by write order — means the surviving pair is the pair we chose,
    and it keeps ``num_pred_nodes`` accounting honest downstream.
    """
    by_source = defaultdict(list)
    for e in edges:
        by_source[e[0]].append(e)
    kept = []
    for source, group in by_source.items():
        if len(group) > max_children and group[0][2] is not None:
            group = sorted(group, key=lambda e: e[2], reverse=True)[:max_children]
        elif len(group) > max_children:
            group = group[:max_children]
        kept.extend(group)
    return kept


def validate_submission(path, expected_datasets=None):
    """Check a submission for the failure modes that cost a whole day of submissions.

    Returns a list of human-readable problems; empty means it looks structurally sound.
    This does not check tracking quality, only that the file will score at all.
    """
    path = Path(path)
    problems = []
    if path.name != "submission.csv":
        problems.append(f"file must be named submission.csv, got {path.name}")

    seen_nodes = defaultdict(set)
    edge_endpoints = defaultdict(list)
    node_times = defaultdict(dict)
    n_rows = 0
    prev_id = -1

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            problems.append(f"header mismatch: {reader.fieldnames}")
            return problems
        for row in reader:
            n_rows += 1
            cur = int(row["id"])
            if cur != prev_id + 1:
                problems.append(f"id column must be consecutive; jumped {prev_id}->{cur}")
                prev_id = cur
            else:
                prev_id = cur
            ds = row["dataset"]
            if row["row_type"] == "node":
                nid = int(row["node_id"])
                if nid in seen_nodes[ds]:
                    problems.append(f"{ds}: duplicate node_id {nid}")
                seen_nodes[ds].add(nid)
                node_times[ds][nid] = int(row["t"])
            elif row["row_type"] == "edge":
                edge_endpoints[ds].append((int(row["source_id"]), int(row["target_id"])))
            else:
                problems.append(f"unknown row_type {row['row_type']!r}")

    for ds, pairs in edge_endpoints.items():
        for s, t in pairs:
            if s not in seen_nodes[ds]:
                problems.append(f"{ds}: edge source {s} has no node row")
                break
            if t not in seen_nodes[ds]:
                problems.append(f"{ds}: edge target {t} has no node row")
                break
        bad_dt = sum(
            1 for s, t in pairs
            if s in node_times[ds] and t in node_times[ds]
            and node_times[ds][t] - node_times[ds][s] != 1
        )
        if bad_dt:
            problems.append(
                f"{ds}: {bad_dt} edges do not span exactly t->t+1; the metric discards these"
            )

    if expected_datasets:
        missing = set(expected_datasets) - set(seen_nodes)
        if missing:
            problems.append(f"missing datasets: {sorted(missing)}")

    if n_rows == 0:
        problems.append("submission is empty")
    return problems
