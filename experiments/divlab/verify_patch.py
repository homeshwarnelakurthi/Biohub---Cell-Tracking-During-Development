"""The B948 notebook's patched add_safe_divisions_postlink must equal the lab variant (ALL model)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).parent))
import json
import os
import divlab as D
from biocell import replay948 as R
NB = Path(os.environ.get("VERIFY_NB", Path(__file__).resolve().parents[2] / "notebooks/b948-divranker/biohub-b948-divranker.ipynb"))
MIN_LOGIT = float(os.environ.get("VERIFY_MIN_LOGIT", "1.0"))


def one(stem):
    snap, _ = D._load(stem)
    ns_nb = R.build_namespace(nb_path=NB)
    a = R.replay(snap, ns_nb)
    ns = R.build_namespace()
    R.install_dc_lookup(ns, snap)
    p = D.baseline_params(ns); p.update(D.OPEN)
    m = json.loads((D.OUT / "ranker_models.json").read_text())["ALL"]
    p["rank"] = lambda f: -D.logit_of(m, f); p["min_rank"] = -MIN_LOGIT
    b = R.replay(snap, ns, safe_div_fn=D.make_safe_div(ns, p))
    return stem, R.graph_signature(a[0], a[1]) == R.graph_signature(b[0], b[1]), a[2]["safe_divisions_added"], b[2]["safe_divisions_added"]


if __name__ == "__main__":
    with ProcessPoolExecutor(7) as ex:
        res = list(ex.map(one, D.stems()))
    for r in res:
        if not r[1]:
            print("MISMATCH", r)
    print(sum(r[1] for r in res), "/", len(res), "identical; divisions added", sum(r[2] for r in res), sum(r[3] for r in res))
