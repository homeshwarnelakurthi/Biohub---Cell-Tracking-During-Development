"""E002 - leave-one-embryo-out CV harness + node-budget characterisation.

Source of truth for the Kaggle notebook `homeshwarrao/biohub-cv-harness`. Kept here as a
plain script so it is reviewable and diffable in git; `tools/sync_notebook.py` wraps it
into a notebook and pushes it.

This notebook needs internet ON (it pip-installs the official scorer). It is a *validation*
notebook, never a submission notebook, so that is fine.

Three stages:

1. **Plumbing check** - score ground truth against itself. Edge Jaccard must come out at
   1.0. If it does not, the harness is wrong and every number downstream is worthless.
2. **Node-budget curve** - degrade ground truth by dropping nodes and re-score. This
   measures the adjusted-Jaccard/node-count trade-off (METRIC_ANALYSIS property 1)
   *without running the tracking pipeline at all*, so we learn the shape of the curve for
   the cost of a CPU notebook.
3. **Pipeline scoring** - the same harness pointed at real predicted geffs, per embryo.

Stages 1-2 are the ones that unblock everything else.
"""

# %% [markdown]
# # E002 - CV harness
#
# Train has **two embryos** (`44b6` x 71, `6bba` x 24) and train/test are embryo-disjoint,
# so the only honest validation is leave-one-embryo-out. See `docs/MISTAKES.md` M004.

# %%
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(REPO)],
        check=True,
    )
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tracksdata", "geff", "polars"],
               check=True)
sys.path.insert(0, str(REPO / "src"))

DATA = Path("/kaggle/input/biohub-cell-tracking-during-development")
TRAIN = DATA / "train"
print("train geffs:", len(list(TRAIN.glob("*.geff"))))

# %% [markdown]
# ## biocell helpers
#
# Inlined rather than imported so the notebook runs standalone on Kaggle. Keep in sync with
# `src/biocell/cv.py` - that file is the canonical copy.

# %%
import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.io import DEFAULT_SCALE, open_dataset
from tracking_cellmot.metrics import (
    ADJUSTMENT_ALPHA,
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)


def embryo_of(name):
    return name.split("_", 1)[0]


def load_graph(p):
    r = td.graph.IndexedRXGraph.from_geff(p)
    return r[0] if isinstance(r, tuple) else r


def read_scale(gt_dir, name):
    try:
        return open_dataset(Path(gt_dir) / name, load_image=False).scale
    except FileNotFoundError:
        return DEFAULT_SCALE


def read_n_total(geff_path):
    try:
        meta = GeffMetadata.read(geff_path)
    except Exception:
        return float("nan")
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def score_one(pred_graph, gt_path, gt_dir, name, max_distance=7.0):
    gt = load_graph(gt_path)
    er = compute_metric(pred_graph, gt, scale=read_scale(gt_dir, name),
                        max_distance=max_distance)
    rec = (node_recall(pred_graph, gt)
           if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0 else 0.0)
    row = per_sample_metrics(er, read_n_total(gt_path), rec)
    row["sample"], row["embryo"] = name, embryo_of(name)
    return row


def format_report(summaries, title=""):
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    header = (f"{'fold':<8}{'n':>4}{'score':>9}{'adj_edge':>10}{'edge_J':>9}"
              f"{'div_J':>8}{'div TP/FP/FN':>16}{'node_rec':>10}")
    lines += [header, "-" * len(header)]
    for fold, s in summaries.items():
        div = f"{s['division_tp']}/{s['division_fp']}/{s['division_fn']}"
        lines.append(
            f"{fold:<8}{s['n']:>4}{s['score']:>9.4f}{s['adj_edge_jaccard']:>10.4f}"
            f"{s['edge_jaccard']:>9.4f}{s['division_jaccard']:>8.4f}{div:>16}"
            f"{s['node_recall']:>10.4f}")
    return "\n".join(lines)


def fold_summaries(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["embryo"]].append(r)
    out = {e: summarise(rs) for e, rs in sorted(by.items())}
    out["ALL"] = summarise(list(rows))
    return out


# %% [markdown]
# ## Stage 1 - plumbing check
#
# Score ground truth against itself. **Edge Jaccard must be 1.0.** Anything else means the
# harness is broken and no downstream number can be trusted.
#
# Note `adj_edge` will *not* be 1.0: ground truth is sparsely annotated, so the GT node
# count sits well below `estimated_number_of_nodes` and the node multiplier pushes the
# adjusted score *above* the raw Jaccard. That gap is itself the measurement we want - it
# quantifies how much headroom METRIC_ANALYSIS property 1 leaves.

# %%
SANITY_N = 6
sanity_names = sorted(p.stem for p in TRAIN.glob("*.geff"))[:SANITY_N]

sanity_rows = []
for nm in sanity_names:
    gt_path = TRAIN / f"{nm}.geff"
    sanity_rows.append(score_one(load_graph(gt_path), gt_path, TRAIN, nm))

print(format_report(fold_summaries(sanity_rows), "Stage 1 - GT vs GT (edge_J must be 1.0)"))
print()
for r in sanity_rows:
    print(f"  {r['sample']:<32} edge_J={r['edge_jaccard']:.4f} "
          f"n_pred={r['num_pred_nodes']:>6} ratio={r['total_node_ratio']:+.4f} "
          f"adj={r['adj_edge_jaccard']:.4f}")

assert all(abs(r["edge_jaccard"] - 1.0) < 1e-9 for r in sanity_rows), \
    "HARNESS BROKEN: GT vs GT did not give edge Jaccard 1.0"
print("\nplumbing OK")

# %% [markdown]
# ## Stage 2 - node-budget curve
#
# The adjusted Jaccard is `max(0, J * (1 - 0.1 * (N_pred - N_true)/N_true))`, clamped only
# at the bottom, so under-predicting nodes multiplies the score upward.
#
# Here we drop a fraction of GT nodes (and every edge touching them) and re-score. This
# gives the **pure cost** side of the trade: how fast Jaccard falls per node removed, with
# no FP-removal benefit, because ground truth has no false positives to remove. Real
# predictions have an FP-enriched tail, so the true optimum sits at *more* aggressive
# pruning than whatever this curve suggests.
#
# In other words this measures the conservative bound. If pruning already looks profitable
# against clean GT, it is unambiguously profitable against real predictions.

# %%
import random


def drop_nodes(graph, keep_fraction, seed=0):
    """Return a new graph keeping a random `keep_fraction` of nodes, edges induced."""
    ids = list(graph.node_ids())
    rng = random.Random(seed)
    keep = set(rng.sample(ids, max(1, int(round(len(ids) * keep_fraction)))))

    attrs = graph.node_attrs()
    cols = {c: attrs[c].to_list() for c in attrs.columns}
    id_col = td.DEFAULT_ATTR_KEYS.NODE_ID

    new = td.graph.IndexedRXGraph()
    remap = {}
    for i, nid in enumerate(cols[id_col]):
        if nid not in keep:
            continue
        payload = {c: cols[c][i] for c in cols if c != id_col}
        remap[nid] = new.add_node(payload)

    for s, t in graph.edge_ids_as_pairs() if hasattr(graph, "edge_ids_as_pairs") else []:
        if s in remap and t in remap:
            new.add_edge(remap[s], remap[t], {})
    return new


KEEP_GRID = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
curve = []
for keep in KEEP_GRID:
    rows = []
    for nm in sanity_names:
        gt_path = TRAIN / f"{nm}.geff"
        pred = drop_nodes(load_graph(gt_path), keep) if keep < 1.0 else load_graph(gt_path)
        try:
            rows.append(score_one(pred, gt_path, TRAIN, nm))
        except Exception as exc:
            print(f"  keep={keep} {nm}: {type(exc).__name__}: {exc}")
    if rows:
        s = summarise(rows)
        curve.append((keep, s))
        print(f"keep={keep:.2f}  score={s['score']:.4f}  "
              f"adj_edge={s['adj_edge_jaccard']:.4f}  edge_J={s['edge_jaccard']:.4f}")

if curve:
    best = max(curve, key=lambda kv: kv[1]["adj_edge_jaccard"])
    print(f"\nbest keep_fraction on clean GT: {best[0]:.2f} "
          f"(adj_edge={best[1]['adj_edge_jaccard']:.4f})")
    print("Real predictions carry an FP-enriched tail, so the true optimum is at or below "
          "this keep fraction - never above it.")

# %% [markdown]
# ## Stage 3 - score real predictions
#
# Point `PRED_DIR` at a directory of predicted `.geff` files and this reports per-embryo.
# Ship a change only if **both** folds improve - see `verdict()` in `src/biocell/cv.py`.

# %%
PRED_DIR = Path("/kaggle/input/biohub-predictions/geffs")  # set per experiment

if PRED_DIR.exists():
    names = sorted({p.stem for p in PRED_DIR.glob("*.geff")}
                   & {p.stem for p in TRAIN.glob("*.geff")})
    rows = []
    for nm in names:
        try:
            rows.append(score_one(load_graph(PRED_DIR / f"{nm}.geff"),
                                  TRAIN / f"{nm}.geff", TRAIN, nm))
        except Exception as exc:
            print(f"  SKIP {nm}: {type(exc).__name__}: {exc}")
    if rows:
        print(format_report(fold_summaries(rows), "Stage 3 - predictions, per embryo"))
        ratios = defaultdict(list)
        for r in rows:
            if r["total_node_ratio"] == r["total_node_ratio"]:
                ratios[r["embryo"]].append(r["total_node_ratio"])
        print("\nnode budget:")
        for emb, vals in sorted(ratios.items()):
            m = sum(vals) / len(vals)
            print(f"  {emb:<8} mean ratio {m:+.4f} -> multiplier {1 - ADJUSTMENT_ALPHA * m:.4f}"
                  f"  ({'penalised' if m > 0 else 'rewarded'})")
else:
    print(f"PRED_DIR not attached ({PRED_DIR}) - stages 1-2 are the deliverable for E002.")
