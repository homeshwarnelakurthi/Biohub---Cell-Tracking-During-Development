"""For 'stolen daughter' GT divisions: what is the thief node?"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).parent))
import divlab as D
from biocell import replay948 as R


def run(stem):
    from biocell import official_score as O
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    import tracksdata as td
    snap = R.load_snapshot(D.DIVLAB / "replay" / f"{stem}.pkl.gz")
    nodes, edges = snap["nodes_by_id"], snap["edges"]
    dc = snap["dc_scores"]
    dist = R.build_namespace()["edge_distance_um"]
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

    def back(n):
        k = 0
        while n in ppred and k < 30:
            n = ppred[n]; k += 1
        return k
    out = []
    for d, kids in gsucc.items():
        if len(kids) < 2: continue
        pd_ = g2p.get(d); pc = [g2p.get(k) for k in kids[:2]]
        if pd_ is None or None in pc: continue
        links = [ppred.get(c) for c in pc]
        if pd_ not in links or links[0] == links[1]: continue
        other = pc[1] if links[0] == pd_ else pc[0]
        th = ppred.get(other)
        if th is None: continue
        out.append(dict(stem=stem,
            thief_matched=th in p2g, thief_gt_is_pred_of_other=(p2g.get(th) is not None and p2g.get(th) == d),
            thief_to_parent=round(dist(nodes[th], nodes[pd_]), 1),
            thief_back=back(th), parent_back=back(pd_), parent_out=len(psucc.get(pd_, [])),
            thief_out=len(psucc.get(th, [])),
            pp_exists=pd_ in ppred, dc_thief=round(dc.get(th, -1), 2),
            thief_to_pp=round(dist(nodes[th], nodes[ppred[pd_]]), 1) if pd_ in ppred and int(nodes[ppred[pd_]]['t'])==int(nodes[th]['t'])-1 else None))
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(7) as ex:
        rows = [r for rs in ex.map(run, D.stems()) for r in rs]
    keys = list(rows[0])
    print(" ".join(f"{k[:14]:>14}" for k in keys[1:]))
    for r in rows:
        print(" ".join(f"{str(r[k]):>14}" for k in keys[1:]))
