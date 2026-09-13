"""Why are GT divisions missed? Classify each GT division by the state of the snapshot graph
(entering safe-division repair) around its matched nodes."""
import os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(Path(__file__).parent))
import divlab as D
from biocell import replay948 as R


def classify(stem):
    from biocell import official_score as O
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    import tracksdata as td
    snap = R.load_snapshot(D.DIVLAB / "replay" / f"{stem}.pkl.gz")
    nodes, edges = snap["nodes_by_id"], snap["edges"]
    ns = R.build_namespace()
    dist = ns["edge_distance_um"]
    gt = O.load_gt(D.gt_path(stem))
    m = _matched_node_attrs(_match_full(O.plain_to_graph(nodes, edges), gt, O.DEFAULT_SCALE, 7.0))
    p2g = dict(zip(m[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), m[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()))
    g2p = {g: p for p, g in p2g.items()}
    ea = gt.edge_attrs()
    gsucc = {}
    for s, t in zip(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(), ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()):
        gsucc.setdefault(s, []).append(t)
    psucc, ppred = {}, {}
    for e in edges:
        psucc.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
        ppred[int(e["target_id"])] = int(e["source_id"])
    out = []
    for d, kids in gsucc.items():
        if len(kids) < 2:
            continue
        pd_ = g2p.get(d); pc = [g2p.get(k) for k in kids[:2]]
        rec = {"stem": stem}
        if pd_ is None:
            cls = "parent_undetected"
        elif pc[0] is None and pc[1] is None:
            cls = "both_daughters_undetected"
        elif pc[0] is None or pc[1] is None:
            cls = "one_daughter_undetected"
        else:
            links = [ppred.get(c) for c in pc]
            nout = len(psucc.get(pd_, []))
            if links[0] == pd_ and links[1] == pd_:
                cls = "already_division"
            elif pd_ in links:
                other = pc[1] if links[0] == pd_ else pc[0]
                olink = ppred.get(other)
                if olink is None:
                    cls = "orphan_daughter(reachable)"
                else:
                    cls = "daughter_stolen_by_other_parent"
                    rec["thief_dist"] = dist(nodes[olink], nodes[other])
                    rec["pd_dist"] = dist(nodes[pd_], nodes[other])
                rec["sister"] = dist(nodes[pc[0]], nodes[pc[1]])
                rec["parent_other"] = dist(nodes[pd_], nodes[other])
            elif nout == 0:
                cls = "parent_track_ends"
            else:
                cls = "parent_linked_elsewhere"
        rec["cls"] = cls
        out.append(rec)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(7) as ex:
        rows = [r for rs in ex.map(classify, D.stems()) for r in rs]
    print(len(rows), "GT divisions")
    for k, v in Counter(r["cls"] for r in rows).most_common():
        print(f"  {k:<34}{v}")
    import statistics as st
    for cls in ["orphan_daughter(reachable)", "daughter_stolen_by_other_parent"]:
        rs = [r for r in rows if r["cls"] == cls]
        if rs:
            print(cls, "sister um:", sorted(round(r["sister"], 1) for r in rs))
            print(cls, "parent->other um:", sorted(round(r["parent_other"], 1) for r in rs))
            if "thief_dist" in rs[0]:
                print("  thief link um:", sorted(round(r["thief_dist"], 1) for r in rs))
