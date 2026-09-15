"""C020's shipped ranker vs the 128-clip refit, on clips C020's ranker never saw."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).parent))
import divlab as D
from biocell import replay948 as R

HERE = Path(__file__).resolve().parent
C020 = json.loads((HERE / "v020" / "ranker_models.json").read_text())["ALL"]
BIG = json.loads((HERE / "v020big" / "ranker_models.json").read_text())
SEEN = set(json.loads((D.REPO / "experiments/divlab/data020/val_stems.json").read_text()))


def one(stem):
    from biocell import official_score as O
    out = {}
    emb = stem.split("_")[0]
    arms = {
        "baseline": None,
        "c020_min3": (C020, 3.0),
        "big_loeo_min3": (BIG[f"not_{emb}"], 3.0),
        "big_all_min3": (BIG["ALL"], 3.0),
        "big_all_min2.5": (BIG["ALL"], 2.5),
        "big_all_min3.5": (BIG["ALL"], 3.5),
    }
    snap, _ = D._load(stem)
    for name, arm in arms.items():
        ns = R.build_namespace()
        R.install_dc_lookup(ns, snap)
        p = D.baseline_params(ns)
        if arm is not None:
            p.update(D.OPEN)
            m, thr = arm
            p["rank"] = lambda f, m=m: -D.logit_of(m, f)
            p["min_rank"] = -thr
        nodes, edges, _ = R.replay(snap, ns, safe_div_fn=D.make_safe_div(ns, p))
        row = O.score_sample(nodes, edges, D.gt_path(stem))
        row.update(embryo=emb, stem=stem)
        out[name] = row
    return out


if __name__ == "__main__":
    from biocell.official_score import summarise
    stems = [s for s in D.stems() if s not in SEEN]
    print(len(stems), "unseen clips")
    with ProcessPoolExecutor(7) as ex:
        res = list(ex.map(one, stems))
    json.dump(res, open(HERE / "v020big" / "compare_rows.json", "w"))
    for arm in res[0]:
        cells = []
        for f in ["44b6", "6bba", "ALL"]:
            s = summarise([r[arm] for r in res if f == "ALL" or r[arm]["embryo"] == f])
            cells.append(f"{f} {s['score']:.4f} ({s['division_tp']}/{s['division_fp']}/{s['division_fn']})")
        print(f"{arm:<16}", "  ".join(cells))
