"""Local replay of the 0.948 notebook's post-processing tail, from a divlab snapshot.

The divlab kernel records the graph as it enters `add_safe_divisions_postlink`
plus a DeepCenter score for every node. Everything after that point is pure
Python, so this module lifts those functions *verbatim* out of the base notebook
(by AST, not by copy-paste) and re-runs them locally. DeepCenter calls are
answered from the recorded scores.

`replay(snapshot)` must reproduce the kernel's recorded final graph exactly;
`check_reproduction` asserts that before any variant is trusted.
"""

from __future__ import annotations

import ast
import copy
import gzip
import json
import os
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_NB = Path(os.environ.get("DIVLAB_BASE_NB", REPO / "bases" / "biohub-948-sew20.ipynb"))

TAIL_FUNCS = [
    "edge_distance_um", "point_distance_um", "node_point", "edge_sort_key", "_position_um",
    "add_safe_divisions_postlink", "filter_short_track_components", "linefit_smooth_output_graph",
]


def _code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def build_namespace(env_overrides: dict[str, str] | None = None, nb_path: Path | None = None) -> dict:
    """Namespace holding the notebook's config constants and tail functions."""
    import numpy as np
    from scipy.spatial import cKDTree

    cells = _code_cells(nb_path or BASE_NB)
    saved = dict(os.environ)
    try:
        # Cell 0: env assignments only.
        for node in ast.parse(cells[0]).body:
            seg = ast.get_source_segment(cells[0], node)
            if seg and seg.startswith("os.environ["):
                exec(seg, {"os": os})
        for k, v in (env_overrides or {}).items():
            os.environ[k] = str(v)

        import math
        ns: dict = {"os": os, "np": np, "math": math, "cKDTree": cKDTree, "Path": Path, "json": json}
        # Upper-case constants from every cell, evaluated in order; skip anything that
        # needs runtime state (artifact paths, torch, ...).
        for src in cells[1:]:
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and all(
                    isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
                ):
                    try:
                        exec(ast.get_source_segment(src, node), ns)
                    except Exception:
                        pass
        cell5 = next(c for c in cells if "def add_safe_divisions_postlink" in c)
        defs = {n.name: n for n in ast.parse(cell5).body if isinstance(n, ast.FunctionDef)}
        for name in TAIL_FUNCS:
            exec(ast.get_source_segment(cell5, defs[name]), ns)
        # filter_output_graph: keep only the part after the safe-division call.
        fog = ast.get_source_segment(cell5, defs["filter_output_graph"])
        start = fog.index("    _geo_cands = stats['safe_division_geometric_candidates']")
        end = fog.index("    return nodes_by_id, edges, stats")
        body = fog[start:end]
        tail_src = (
            "def _tail(nodes_by_id, edges, stats, dataset=None):\n"
            + body
            + "    return nodes_by_id, edges, stats\n"
        )
        exec(tail_src, ns)
        stats_src = fog[fog.index("    stats = {"): fog.index("    edges: list[dict[str, object]] = []")]
        exec("def _fresh_stats(raw_edges=()):\n" + stats_src + "    return stats\n", ns)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return ns


def load_snapshot(path: Path) -> dict:
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def install_dc_lookup(ns: dict, snap: dict) -> None:
    """Answer DeepCenter veto calls from the recorded per-node scores."""
    by_point = {}
    for nid, node in snap["nodes_by_id"].items():
        s = snap["dc_scores"].get(int(nid))
        if s is not None:
            by_point[(int(node["t"]), ns["node_point"](node))] = s

    def deepcenter_accept_repair_point(dataset, t, point, detector_bundle, frame_cache, heatmap_cache,
                                       stats, prefix, threshold):
        if not ns["USE_DEEPCENTER_VETO"]:
            return True
        score = by_point.get((int(t), tuple(point)))
        if score is None:
            stats[f"deepcenter_{prefix}_missing"] += 1
            return True
        stats[f"deepcenter_{prefix}_checked"] += 1
        if score < float(threshold):
            stats[f"deepcenter_{prefix}_rejected"] += 1
            return False
        stats[f"deepcenter_{prefix}_accepted"] += 1
        return True

    ns["deepcenter_accept_repair_point"] = deepcenter_accept_repair_point
    ns["deepcenter_score_point"] = lambda dataset, t, point, *a, **k: by_point.get((int(t), tuple(point)))
    ns["_dc_by_node"] = {int(k): v for k, v in snap["dc_scores"].items()}


def replay(snap: dict, ns: dict, safe_div_fn=None):
    """Run safe divisions + the notebook tail on a copy of the snapshot."""
    install_dc_lookup(ns, snap)
    nodes = copy.deepcopy(snap["nodes_by_id"])
    edges = copy.deepcopy(snap["edges"])
    stats = ns["_fresh_stats"]()
    fn = safe_div_fn or ns["add_safe_divisions_postlink"]
    edges = fn(nodes, edges, stats, dataset=snap["dataset"], deepcenter_bundle={"recorded": True},
               frame_cache={}, deepcenter_cache={})
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        nodes, edges, stats = ns["_tail"](nodes, edges, stats, dataset=snap["dataset"])
    return nodes, edges, stats


def graph_signature(nodes: dict, edges: list) -> tuple:
    n = sorted((int(k), int(v["t"]), round(float(v["z"]), 4), round(float(v["y"]), 4), round(float(v["x"]), 4))
               for k, v in nodes.items())
    e = sorted((int(x["source_id"]), int(x["target_id"])) for x in edges)
    return tuple(n), tuple(e)


def check_reproduction(snap: dict, final: dict, ns: dict) -> bool:
    nodes, edges, _ = replay(snap, ns)
    return graph_signature(nodes, edges) == graph_signature(final["nodes_by_id"], final["edges"])
