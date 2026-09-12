"""Division-repair research on divlab snapshots, scored with the official metric.

    python experiments/divlab/divlab.py check     # replay reproduces the kernel exactly
    python experiments/divlab/divlab.py features  # proposal table with GT labels -> proposals.csv
    python experiments/divlab/divlab.py sweep     # variants, official metric, per embryo

Needs BIOCELL_METRIC_REPO (official metric checkout) and DIVLAB (unzipped divlab output).
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from biocell import replay948 as R  # noqa: E402

DIVLAB = Path(os.environ.get("DIVLAB", REPO / "experiments" / "divlab" / "data"))
OUT = Path(__file__).resolve().parent


def stems() -> list[str]:
    return sorted(p.name[: -len(".pkl.gz")] for p in (DIVLAB / "replay").glob("*.pkl.gz"))


# --------------------------------------------------------------------------- variant
def chain_len(start: int, succ: dict[int, list[int]], limit: int = 12) -> int:
    n, cur = 1, start
    while n < limit:
        nxt = succ.get(cur, [])
        if len(nxt) != 1:
            break
        cur = nxt[0]
        n += 1
    return n


def back_len(start: int, pred: dict[int, int], limit: int = 12) -> int:
    n, cur = 1, start
    while n < limit and cur in pred:
        cur = pred[cur]
        n += 1
    return n


def make_safe_div(ns: dict, p: dict, collect: list | None = None):
    """Parameterised add_safe_divisions_postlink. p keys:
    parent_max, sister_max, child_max, mutual_nn, diverge (None=off), dc_thr,
    frame_cap, global_cap, rank (callable(features)->key, lower first),
    min_cand_len, min_child_len.
    With p == baseline params and rank == baseline key this is the notebook function.
    """
    edge_distance_um = ns["edge_distance_um"]
    _position_um = ns["_position_um"]

    def fn(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        dc = ns["_dc_by_node"]
        out_by_source: dict[int, list[dict]] = {}
        incoming: set[int] = set()
        pred_of: dict[int, int] = {}
        for e in edges:
            out_by_source.setdefault(int(e["source_id"]), []).append(e)
            incoming.add(int(e["target_id"]))
            pred_of[int(e["target_id"])] = int(e["source_id"])
        succ = {k: [int(e["target_id"]) for e in v] for k, v in out_by_source.items()}
        ids_by_t: dict[int, list[int]] = {}
        for nid, node in nodes_by_id.items():
            ids_by_t.setdefault(int(node["t"]), []).append(nid)
        existing = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
        global_cap = max(1, int(round(max(1, len(edges)) * p["global_cap"])))
        added, used_t, used_s = [], set(), set()

        for t in sorted(ids_by_t):
            child_ids = ids_by_t.get(t + 1, [])
            if not child_ids:
                continue
            source_ids = [n for n in ids_by_t[t] if len(out_by_source.get(n, [])) == 1]
            cand_ids = [n for n in child_ids if n not in incoming and n not in used_t]
            if not source_ids or not cand_ids:
                continue
            tree = cKDTree(np.stack([_position_um(nodes_by_id[c]) for c in cand_ids])) if p["mutual_nn"] else None
            frame_cap = max(1, int(round(len(source_ids) * p["frame_cap"])))
            props = []
            for sid in source_ids:
                src = nodes_by_id[sid]
                cid = int(out_by_source[sid][0]["target_id"])
                child = nodes_by_id.get(cid)
                if child is None or int(child["t"]) != t + 1:
                    continue
                child_dist = edge_distance_um(src, child)
                if child_dist > p["child_max"]:
                    continue
                nn_id = None
                if tree is not None:
                    _, k = tree.query(_position_um(child))
                    nn_id = cand_ids[int(k)]
                for qid in cand_ids:
                    if (sid, qid) in existing:
                        continue
                    q = nodes_by_id[qid]
                    pd_ = edge_distance_um(src, q)
                    if pd_ > p["parent_max"]:
                        continue
                    sd = edge_distance_um(child, q)
                    if sd > p["sister_max"]:
                        continue
                    if p["mutual_nn"] and qid != nn_id:
                        continue
                    div = float("nan")
                    c1, q1 = out_by_source.get(cid, []), out_by_source.get(qid, [])
                    if len(c1) == 1 and len(q1) == 1:
                        g1 = nodes_by_id.get(int(c1[0]["target_id"]))
                        g2 = nodes_by_id.get(int(q1[0]["target_id"]))
                        if g1 is not None and g2 is not None and int(g1["t"]) == t + 2 and int(g2["t"]) == t + 2:
                            div = edge_distance_um(g1, g2) - sd
                    if p["diverge"] is not None and not (div == div and div >= p["diverge"]):
                        continue
                    dq = dc.get(int(qid))
                    if p["dc_thr"] is not None and dq is not None and dq < p["dc_thr"]:
                        continue
                    f = {
                        "t": t, "source": sid, "child": cid, "cand": qid,
                        "parent_dist": pd_, "sister_dist": sd, "child_dist": child_dist, "diverge": div,
                        "dc_cand": dq if dq is not None else float("nan"),
                        "dc_child": dc.get(cid, float("nan")),
                        "cand_len": chain_len(qid, succ), "child_len": chain_len(cid, succ),
                        "src_back": back_len(sid, pred_of),
                        "child_prob": float(out_by_source[sid][0].get("edge_prob") or float("nan")),
                        "n_sources": len(source_ids),
                    }
                    if f["cand_len"] < p.get("min_cand_len", 0) or f["child_len"] < p.get("min_child_len", 0):
                        continue
                    props.append((p["rank"](f), f))
            if collect is not None:
                collect.extend(f for _, f in props)
            props.sort(key=lambda x: x[0])
            n_frame = 0
            for _, f in props:
                if len(added) >= global_cap or n_frame >= frame_cap:
                    break
                if f["cand"] in used_t or f["cand"] in incoming or f["source"] in used_s:
                    continue
                added.append({"source_id": f["source"], "target_id": f["cand"], "edge_prob": None,
                              "distance_um": f["parent_dist"], "safe_division": 1})
                used_t.add(f["cand"])
                used_s.add(f["source"])
                n_frame += 1
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added] if added else edges

    return fn


def baseline_params(ns: dict) -> dict:
    return {
        "parent_max": ns["SAFE_DIV_MAX_UM"], "sister_max": ns["SAFE_DIV_SISTER_MAX_UM"],
        "child_max": ns["SAFE_DIV_EXISTING_CHILD_MAX_UM"], "mutual_nn": ns["SAFE_DIV_REQUIRE_MUTUAL_NN"],
        "diverge": ns["SAFE_DIV_DIVERGE_UM"] if ns["SAFE_DIV_REQUIRE_DIVERGENCE"] else None,
        "dc_thr": ns["DEEPCENTER_SAFE_DIV_THRESHOLD"] if ns["DEEPCENTER_SAFE_DIV_VETO"] else None,
        "frame_cap": ns["SAFE_DIV_FRAME_FRAC_CAP"], "global_cap": ns["SAFE_DIV_GLOBAL_FRAC_CAP"],
        "rank": lambda f: f["parent_dist"] + 0.15 * f["sister_dist"],
    }


# --------------------------------------------------------------------------- tasks
def _load(stem: str):
    snap = R.load_snapshot(DIVLAB / "replay" / f"{stem}.pkl.gz")
    final_p = DIVLAB / "final" / f"{stem}.pkl.gz"
    final = R.load_snapshot(final_p) if final_p.exists() else None
    return snap, final


def task_check(stem: str) -> tuple[str, bool, bool]:
    ns = R.build_namespace()
    snap, final = _load(stem)
    verbatim = R.check_reproduction(snap, final, ns)
    nodes, edges, _ = R.replay(snap, ns, safe_div_fn=make_safe_div(ns, baseline_params(ns)))
    mine = R.graph_signature(nodes, edges) == R.graph_signature(final["nodes_by_id"], final["edges"])
    return stem, verbatim, mine


def gt_path(stem: str) -> Path:
    return DIVLAB / "gt" / f"{stem}.geff"


def task_features(stem: str) -> list[dict]:
    from biocell import official_score as O
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    import tracksdata as td

    ns = R.build_namespace()
    snap, _ = _load(stem)
    R.install_dc_lookup(ns, snap)
    p = baseline_params(ns)
    # Wide-open pool so the labels describe what a better ranker could choose from.
    p.update(parent_max=12.0, sister_max=18.0, child_max=12.0, mutual_nn=False, diverge=None, dc_thr=None,
             frame_cap=1.0, global_cap=1.0)
    rows: list[dict] = []
    fn = make_safe_div(ns, p, collect=rows)
    fn(dict(snap["nodes_by_id"]), list(snap["edges"]), ns["_fresh_stats"]())

    gt = O.load_gt(gt_path(stem))
    pred = O.plain_to_graph(snap["nodes_by_id"], snap["edges"])
    m = _matched_node_attrs(_match_full(pred, gt, O.DEFAULT_SCALE, 7.0))
    p2g = dict(zip(m[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), m[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()))
    gt_succ = {}
    for s, t in zip(*[gt.edge_attrs()[k].to_list() for k in (td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET)]):
        gt_succ.setdefault(int(s), set()).add(int(t))
    for r in rows:
        gs, gc, gq = p2g.get(r["source"]), p2g.get(r["child"]), p2g.get(r["cand"])
        kids = gt_succ.get(gs, set()) if gs is not None else set()
        r["label"] = int(len(kids) >= 2 and gc in kids and gq in kids and gc != gq)
        r["gt_src_divides"] = int(len(kids) >= 2)
        r["cand_matched"] = int(gq is not None)
        r["cand_dup_of_child"] = int(gq is not None and gq == gc)
        r["stem"], r["embryo"] = stem, stem.split("_")[0]
    return rows


def run_variant(stem: str, name: str, overrides: dict, rank_name: str) -> dict:
    from biocell import official_score as O
    ns = R.build_namespace()
    snap, _ = _load(stem)
    R.install_dc_lookup(ns, snap)
    p = baseline_params(ns)
    p.update(overrides)
    p["rank"] = RANKERS[rank_name]
    nodes, edges, stats = R.replay(snap, ns, safe_div_fn=make_safe_div(ns, p))
    row = O.score_sample(nodes, edges, gt_path(stem))
    row.update(stem=stem, embryo=stem.split("_")[0], variant=name, safe_divs=stats["safe_divisions_added"])
    return row


RANKERS = {
    "baseline": lambda f: f["parent_dist"] + 0.15 * f["sister_dist"],
}
VARIANTS: dict[str, tuple[dict, str]] = {
    "baseline": ({}, "baseline"),
}


def _pool(fn, args_list, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, *zip(*args_list)))


def main() -> int:
    cmd = sys.argv[1]
    ss = stems()
    print(f"{len(ss)} snapshots in {DIVLAB}")
    t0 = time.time()
    if cmd == "check":
        for stem, verbatim, mine in _pool(task_check, [(s,) for s in ss]):
            print(f"  {stem:<28} verbatim={verbatim} param-copy={mine}")
    elif cmd == "features":
        rows = [r for rs in _pool(task_features, [(s,) for s in ss]) for r in rs]
        with open(OUT / "proposals.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} proposals, {sum(r['label'] for r in rows)} positive -> proposals.csv")
    elif cmd == "sweep":
        from biocell import official_score as O
        names = sys.argv[2:] or list(VARIANTS)
        jobs = [(s, n, VARIANTS[n][0], VARIANTS[n][1]) for n in names for s in ss]
        rows = _pool(run_variant, jobs)
        with open(OUT / "sweep_rows.csv", "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        print(f"{'variant':<22}{'fold':<6}{'score':>8}{'adj':>8}{'divJ':>8}{'TP/FP/FN':>13}{'added':>7}")
        for n in names:
            vr = [r for r in rows if r["variant"] == n]
            for fold in sorted({r["embryo"] for r in vr}) + ["ALL"]:
                sel = [r for r in vr if fold == "ALL" or r["embryo"] == fold]
                s = O.summarise(sel)
                print(f"{n:<22}{fold:<6}{s['score']:>8.4f}{s['adj_edge_jaccard']:>8.4f}{s['division_jaccard']:>8.4f}"
                      f"{str(s['division_tp'])+'/'+str(s['division_fp'])+'/'+str(s['division_fn']):>13}"
                      f"{sum(r['safe_divs'] for r in sel):>7}")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
