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
OUT = Path(__file__).resolve().parent / os.environ.get("DIVLAB_TAG", "")
OUT.mkdir(exist_ok=True)


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
        removed: set[tuple[int, int]] = set()
        steal = bool(p.get("steal"))

        for t in sorted(ids_by_t):
            child_ids = ids_by_t.get(t + 1, [])
            if not child_ids:
                continue
            source_ids = [n for n in ids_by_t[t] if len(out_by_source.get(n, [])) == 1]
            cand_ids = [n for n in child_ids if n not in incoming and n not in used_t]
            pool_ids = cand_ids
            if steal:
                # Also daughters currently owned by a single-child parent ("thief").
                pool_ids = [n for n in child_ids if n not in used_t and (
                    n not in incoming or (len(out_by_source.get(pred_of[n], [])) == 1 and pred_of[n] not in used_s))]
            if not source_ids or not pool_ids:
                continue
            tree = (cKDTree(np.stack([_position_um(nodes_by_id[c]) for c in cand_ids]))
                    if (p["mutual_nn"] and cand_ids) else None)
            pool_tree = cKDTree(np.stack([_position_um(nodes_by_id[c]) for c in pool_ids]))
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
                # Ball query keeps pool order (sorted indices), so ties resolve as in the notebook.
                for k_ in sorted(pool_tree.query_ball_point(_position_um(src), p["parent_max"] + 1e-6)):
                    qid = pool_ids[k_]
                    if (sid, qid) in existing:
                        continue
                    stolen = qid in incoming
                    q = nodes_by_id[qid]
                    pd_ = edge_distance_um(src, q)
                    if pd_ > p["parent_max"]:
                        continue
                    sd = edge_distance_um(child, q)
                    if sd > p["sister_max"]:
                        continue
                    if not stolen and p["mutual_nn"] and qid != nn_id:
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
                    if p.get("sym_tau", 0.0) > 0.0 and not stolen:
                        if abs(child_dist - pd_) / max((child_dist + pd_) / 2.0, 1e-6) > p["sym_tau"]:
                            continue
                    f = {
                        "t": t, "source": sid, "child": cid, "cand": qid, "stolen": int(stolen),
                        "thief_dist": edge_distance_um(nodes_by_id[pred_of[qid]], q) if stolen else float("nan"),
                        "thief_back": back_len(pred_of[qid], pred_of) if stolen else 0,
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
                    if stolen:
                        if p.get("steal_rank") is None:  # feature collection only
                            if collect is not None:
                                collect.append(f)
                            continue
                        key = p["steal_rank"](f)
                        if p.get("steal_min_rank") is not None and key > p["steal_min_rank"]:
                            continue
                    else:
                        key = p["rank"](f)
                        if p.get("min_rank") is not None and key > p["min_rank"]:
                            continue
                    props.append((key, f))
            if collect is not None:
                collect.extend(f for _, f in props)
            props.sort(key=lambda x: x[0])
            n_frame = 0
            for _, f in props:
                if len(added) >= global_cap or n_frame >= frame_cap:
                    break
                if f["cand"] in used_t or f["source"] in used_s:
                    continue
                if f["stolen"]:
                    th = pred_of.get(f["cand"])
                    if th is None or th in used_s or th == f["source"]:
                        continue
                    removed.add((th, f["cand"]))
                    used_s.add(th)
                elif f["cand"] in incoming:
                    continue
                added.append({"source_id": f["source"], "target_id": f["cand"], "edge_prob": None,
                              "distance_um": f["parent_dist"], "safe_division": 1})
                used_t.add(f["cand"])
                used_s.add(f["source"])
                n_frame += 1
        stats["safe_divisions_added"] = len(added)
        stats["safe_divisions_reclaimed"] = len(removed)
        if removed:
            edges = [e for e in edges if (int(e["source_id"]), int(e["target_id"])) not in removed]
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
        "sym_tau": float(ns.get("SAFE_DIV_SISTER_SYMMETRY_TAU", 0.0)),
    }


# --------------------------------------------------------------------------- learned ranker
FEATS = ["parent_dist", "child_dist", "diverge", "dc_cand", "child_prob", "cand_len", "sister_dist"]
FEATS_STEAL = FEATS + ["thief_dist", "thief_back"]


def _fx(f: dict, feats: list[str] = FEATS) -> list[float]:
    out = []
    for k in feats:
        x = float(f[k])
        if x != x:
            x = -3.0 if k == "diverge" else (0.85 if k == "child_prob" else 0.22)
        out.append(x)
    return out


def fit_logit(rows: list[dict], l2: float = 1.0, feats: list[str] = FEATS) -> dict:
    X = np.array([_fx(r, feats) for r in rows]); y = np.array([int(r["label"]) for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    sw = np.where(y == 1, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = (p - y) * sw
        w -= 0.1 * (Z.T @ g / len(y) + l2 * w / len(y)); b -= 0.1 * g.mean()
    # express in raw units: logit = c0 + sum(c_i * x_i)
    c = w / sd
    return {"feats": feats, "coef": c.tolist(), "intercept": float(b - (c * mu).sum())}


def logit_of(model: dict, f: dict) -> float:
    return model["intercept"] + sum(c * x for c, x in zip(model["coef"], _fx(f, model["feats"])))


def task_fit() -> None:
    import json
    rows = [r for r in csv.DictReader(open(OUT / "proposals.csv"))
            if r["label"] == "1" or r["src_gt_has_child"] == "1"]
    orphan = [r for r in rows if r.get("stolen", "0") == "0"]
    stolen = [r for r in rows if r.get("stolen", "0") == "1"]
    print(f"orphan rows {len(orphan)} ({sum(r['label'] == '1' for r in orphan)} pos), "
          f"stolen rows {len(stolen)} ({sum(r['label'] == '1' for r in stolen)} pos)")
    models = {"ALL": fit_logit(orphan), "steal_ALL": fit_logit(stolen, feats=FEATS_STEAL)}
    for e in sorted({r["embryo"] for r in rows}):
        models[f"not_{e}"] = fit_logit([r for r in orphan if r["embryo"] != e])
        models[f"steal_not_{e}"] = fit_logit([r for r in stolen if r["embryo"] != e], feats=FEATS_STEAL)
    (OUT / "ranker_models.json").write_text(json.dumps(models, indent=1))
    for k, m in models.items():
        print(k, {f: round(c, 3) for f, c in zip(m["feats"], m["coef"])}, "b", round(m["intercept"], 2))


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
    p.update(parent_max=12.0, sister_max=18.0, child_max=12.0, mutual_nn=False, diverge=None, dc_thr=None, sym_tau=0.0,
             frame_cap=1.0, global_cap=1.0, steal=True)
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
        r["src_matched"] = int(gs is not None)
        r["src_gt_has_child"] = int(gs is not None and len(kids) >= 1)
        r["cand_matched"] = int(gq is not None)
        r["cand_dup_of_child"] = int(gq is not None and gq == gc)
        r["stem"], r["embryo"] = stem, stem.split("_")[0]
    return rows


def run_tail_variant(stem: str, name: str, env: dict) -> dict:
    """Verbatim notebook safe divisions + tail, with notebook env overrides (BIOHUB_* keys)."""
    from biocell import official_score as O
    ns = R.build_namespace(env_overrides=env)
    snap, _ = _load(stem)
    nodes, edges, stats = R.replay(snap, ns)
    rounded = env.get("_ROUNDED", "1") == "1"
    row = O.score_sample(nodes, edges, gt_path(stem), rounded=rounded)
    row.update(stem=stem, embryo=stem.split("_")[0], variant=name, safe_divs=stats["safe_divisions_added"])
    return row


TAIL_VARIANTS: dict[str, dict] = {
    "tail_base": {},
    "minlen5": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "5"},
    "minlen7": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "7"},
    "minlen8": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "8"},
    "minlen4": {"BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"},
    "no_rescue": {"BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "0"},
    "no_keepdiv": {"BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "0"},
    "linefit_off": {"BIOHUB_OUTPUT_LINEFIT_SMOOTH": "0"},
    "linefit_w06": {"BIOHUB_OUTPUT_LINEFIT_WEIGHT": "0.6"},
    "linefit_w10": {"BIOHUB_OUTPUT_LINEFIT_WEIGHT": "1.0"},
    "linefit_win3": {"BIOHUB_OUTPUT_LINEFIT_WINDOW": "3"},
    "linefit_win1": {"BIOHUB_OUTPUT_LINEFIT_WINDOW": "1"},
    "divgeom_on": {"BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER": "1"},
    "float_coords": {"_ROUNDED": "0"},
    "float_linefit_w10": {"_ROUNDED": "0", "BIOHUB_OUTPUT_LINEFIT_WEIGHT": "1.0"},
    "float_linefit_win3": {"_ROUNDED": "0", "BIOHUB_OUTPUT_LINEFIT_WINDOW": "3"},
}


def run_variant(stem: str, name: str, overrides: dict, rank_name: str) -> dict:
    from biocell import official_score as O
    ns = R.build_namespace()
    snap, _ = _load(stem)
    R.install_dc_lookup(ns, snap)
    p = baseline_params(ns)
    p.update(overrides)
    if rank_name.startswith("logit"):
        import json
        models = json.loads((OUT / "ranker_models.json").read_text())
        model = models[f"not_{stem.split('_')[0]}"] if rank_name == "logit_loeo" else models["ALL"]
        min_logit = p.pop("min_logit", None)
        p["rank"] = lambda f, m=model: -logit_of(m, f)
        if min_logit is not None:
            p["min_rank"] = -min_logit
        steal_min = p.pop("steal_min_logit", None)
        if steal_min is not None:
            sm = models[f"steal_not_{stem.split('_')[0]}"] if rank_name == "logit_loeo" else models["steal_ALL"]
            p["steal"] = True
            p["steal_rank"] = lambda f, m=sm: -logit_of(m, f)
            p["steal_min_rank"] = -steal_min
    else:
        p["rank"] = RANKERS[rank_name]
    nodes, edges, stats = R.replay(snap, ns, safe_div_fn=make_safe_div(ns, p))
    row = O.score_sample(nodes, edges, gt_path(stem))
    row.update(stem=stem, embryo=stem.split("_")[0], variant=name, safe_divs=stats["safe_divisions_added"],
               reclaimed=stats.get("safe_divisions_reclaimed", 0))
    return row


RANKERS = {
    "baseline": lambda f: f["parent_dist"] + 0.15 * f["sister_dist"],
}
OPEN = dict(parent_max=10.0, sister_max=16.0, child_max=12.0, mutual_nn=False, diverge=None, dc_thr=None, sym_tau=0.0)
VARIANTS: dict[str, tuple[dict, str]] = {
    "baseline": ({}, "baseline"),
    "logit_gated": ({}, "logit_loeo"),
    "logit_open": (OPEN, "logit_loeo"),
    "logit_open_cap2x": ({**OPEN, "frame_cap": 0.0152, "global_cap": 0.0075}, "logit_loeo"),
    "logit_open_min0": ({**OPEN, "min_logit": 0.0}, "logit_loeo"),
    "logit_open_min1": ({**OPEN, "min_logit": 1.0}, "logit_loeo"),
    "logit_open_min2": ({**OPEN, "min_logit": 2.0}, "logit_loeo"),
    "logit_open_min1_cap2x": ({**OPEN, "min_logit": 1.0, "frame_cap": 0.0152, "global_cap": 0.0075}, "logit_loeo"),
    "open_baseline_key": (OPEN, "baseline"),
    "keepgates_min1": ({**OPEN, "sym_tau": 0.6, "dc_thr": 0.20, "min_logit": 1.0}, "logit_loeo"),
    "keepgates_min0": ({**OPEN, "sym_tau": 0.6, "dc_thr": 0.20, "min_logit": 0.0}, "logit_loeo"),
    "logit_open_min3": ({**OPEN, "min_logit": 3.0}, "logit_loeo"),
    "logit_open_min4": ({**OPEN, "min_logit": 4.0}, "logit_loeo"),
    "logit_open_min5": ({**OPEN, "min_logit": 5.0}, "logit_loeo"),
    "logit_open_min6": ({**OPEN, "min_logit": 6.0}, "logit_loeo"),
    "ALLmodel_open_min3": ({**OPEN, "min_logit": 3.0}, "logit_all"),
    "ALLmodel_open_min4": ({**OPEN, "min_logit": 4.0}, "logit_all"),
    "gate_tight_min3": ({**OPEN, "parent_max": 8.0, "sister_max": 14.0, "child_max": 10.0, "min_logit": 3.0}, "logit_loeo"),
    "gate_wide_min3": ({**OPEN, "parent_max": 12.0, "sister_max": 18.0, "child_max": 14.0, "min_logit": 3.0}, "logit_loeo"),
    "gate_parent9_min3": ({**OPEN, "parent_max": 9.0, "min_logit": 3.0}, "logit_loeo"),
    "gate_sister13_min3": ({**OPEN, "sister_max": 13.0, "min_logit": 3.0}, "logit_loeo"),
    "min3_cap2x": ({**OPEN, "min_logit": 3.0, "frame_cap": 0.0152, "global_cap": 0.0075}, "logit_loeo"),
    "min3_cap_half": ({**OPEN, "min_logit": 3.0, "frame_cap": 0.0038, "global_cap": 0.001875}, "logit_loeo"),
    "min1_steal0": ({**OPEN, "min_logit": 1.0, "steal_min_logit": 0.0}, "logit_loeo"),
    "min1_steal1": ({**OPEN, "min_logit": 1.0, "steal_min_logit": 1.0}, "logit_loeo"),
    "min1_steal2": ({**OPEN, "min_logit": 1.0, "steal_min_logit": 2.0}, "logit_loeo"),
    "min1_steal3": ({**OPEN, "min_logit": 1.0, "steal_min_logit": 3.0}, "logit_loeo"),
    "min2_steal2": ({**OPEN, "min_logit": 2.0, "steal_min_logit": 2.0}, "logit_loeo"),
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
    elif cmd == "fit":
        task_fit()
    elif cmd == "features":
        rows = [r for rs in _pool(task_features, [(s,) for s in ss]) for r in rs]
        with open(OUT / "proposals.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} proposals, {sum(r['label'] for r in rows)} positive -> proposals.csv")
    elif cmd == "tailsweep":
        from biocell import official_score as O
        names = sys.argv[2:] or list(TAIL_VARIANTS)
        rows = _pool(run_tail_variant, [(s, n, TAIL_VARIANTS[n]) for n in names for s in ss])
        with open(OUT / "tail_rows.csv", "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        base = {f: O.summarise([r for r in rows if r["variant"] == names[0] and (f == "ALL" or r["embryo"] == f)])
                for f in ["44b6", "6bba", "ALL"]}
        print(f"{'variant':<14}" + "".join(f"{f:>18}" for f in ["44b6", "6bba", "ALL"]) + "   (score, delta vs first)")
        for n in names:
            cells = []
            for f in ["44b6", "6bba", "ALL"]:
                s = O.summarise([r for r in rows if r["variant"] == n and (f == "ALL" or r["embryo"] == f)])
                cells.append(f"{s['score']:.4f} {s['score'] - base[f]['score']:+.4f}")
            print(f"{n:<14}" + "".join(f"{c:>18}" for c in cells))
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
