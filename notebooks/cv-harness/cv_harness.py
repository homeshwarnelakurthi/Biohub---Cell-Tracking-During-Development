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

import numpy as np

# Pin numpy to whatever the Kaggle image already ships. Installing tracksdata/geff
# unpinned drags in a newer numpy, and every pre-compiled extension in the image was
# built against the old ABI - the whole session then dies on
# `numpy._core._multiarray_umath has no attribute _blas_supports_fpe`.
# Same class of failure as docs/MISTAKES.md M005 and M007.
NUMPY_PIN = f"numpy=={np.__version__}"
print("pinning", NUMPY_PIN)

# Clone to /tmp, NOT /kaggle/working: anything under working becomes notebook output,
# and a cloned repo there makes the output archive huge and slow to retrieve.
REPO = Path("/tmp/kaggle-cell-tracking-competition")
if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(REPO)],
        check=True,
    )

with open("/tmp/constraints.txt", "w") as fh:
    fh.write(NUMPY_PIN + "\n")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-c", "/tmp/constraints.txt", "tracksdata", "geff", "polars"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)],
               check=True)
sys.path.insert(0, str(REPO / "src"))

import numpy as _np_after  # noqa: E402
assert _np_after.__version__ == np.__version__, (
    f"numpy moved {np.__version__} -> {_np_after.__version__}; the image ABI is now broken"
)
print("numpy still", _np_after.__version__)


def find_train_dir(max_depth: int = 4) -> Path:
    """Locate the competition train directory without hardcoding the mount layout.

    Two runs have now been lost to path assumptions: first a hardcoded slug path that
    silently yielded zero samples, then a one-level search that missed the real nesting
    (`/kaggle/input/competitions/<slug>/train`). So walk the tree instead of guessing, and
    prefer a directory literally named `train` when several contain geffs — the others
    would be `test`, which has no ground truth.
    """
    root = Path("/kaggle/input")
    print("mounted inputs:", [p.name for p in sorted(root.glob("*"))])

    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            children = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        if any(c.name.endswith(".geff") for c in children):
            found.append(d)
            continue  # no need to descend further into a sample directory
        stack.extend((c, depth + 1) for c in children if c.is_dir())

    if not found:
        raise FileNotFoundError(f"no directory containing *.geff under {root} (depth<={max_depth})")

    print("dirs containing geffs:", [str(p) for p in found])
    for p in found:
        if p.name == "train":
            return p
    return found[0]


TRAIN = find_train_dir()
geffs = sorted(TRAIN.glob("*.geff"))
zarrs = sorted(TRAIN.glob("*.zarr"))
print(f"TRAIN = {TRAIN}\n  {len(geffs)} .geff, {len(zarrs)} .zarr")

embryos = defaultdict(int)
for g in geffs:
    embryos[g.stem.split("_", 1)[0]] += 1
print("  embryos:", dict(embryos))
assert geffs, "no ground-truth geffs found - nothing to validate against"

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
SANITY_PER_EMBRYO = 4

# Stratify across embryos. Taking the first N sorted names put every sample in `44b6`,
# which defeats the entire point of a per-embryo harness.
_by_embryo = defaultdict(list)
for p in sorted(TRAIN.glob("*.geff")):
    _by_embryo[embryo_of(p.stem)].append(p.stem)
sanity_names = [nm for e in sorted(_by_embryo) for nm in _by_embryo[e][:SANITY_PER_EMBRYO]]
print("sanity samples:", sanity_names)

# How sparse is the ground truth really? estimated_number_of_nodes is what the node
# penalty is measured against, and the gap between it and the annotated count decides
# whether the node-budget lever is worth anything.
print("\nannotation density (annotated nodes vs estimated true total):")
for nm in sanity_names:
    gp = TRAIN / f"{nm}.geff"
    g = load_graph(gp)
    nt = read_n_total(gp)
    print(f"  {nm:<32} annotated={g.num_nodes():>6}  est_true={nt:>12,.0f}  "
          f"{100 * g.num_nodes() / nt:>6.2f}%" if nt == nt and nt > 0 else
          f"  {nm:<32} annotated={g.num_nodes():>6}  est_true=NaN")

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
# ## Stage 2 - node-budget curve (degenerate worst case, read the caveat)
#
# The adjusted Jaccard is `max(0, J * (1 - 0.1 * (N_pred - N_true)/N_true))`, clamped only
# at the bottom, so under-predicting nodes multiplies the score upward.
#
# Dropping GT nodes measures the **cost** side. But note what this curve is and is not:
# in ground truth, *every* node is annotated, so every removal destroys real edges. In real
# predictions the annotation density above is on the order of 1%, so ~99% of predicted
# nodes match no GT node at all and are invisible to edge TP/FP/FN while still counting
# against `num_pred_nodes`.
#
# So this curve is a **degenerate worst case**, not a usable bound - it is the most hostile
# possible setting for pruning. It is worth running only to confirm the mechanism and the
# slope. The number that actually matters comes from stage 3 on real predictions.

# %%
import random

# API verified against tracksdata 0.1.0rc7 locally: IndexedRXGraph has no subgraph(), but
# it does have copy() and bulk_remove_nodes(Sequence[int]). Removing a node drops its
# incident edges, which is the semantics we want.


def drop_nodes(graph, keep_fraction, seed=0):
    """Return a copy of `graph` with a random `keep_fraction` of its nodes kept.

    Deterministic given `seed`, so the sweep compares like with like across keep levels.
    """
    ids = list(graph.node_ids())
    n_drop = len(ids) - max(1, int(round(len(ids) * keep_fraction)))
    if n_drop <= 0:
        return graph.copy()

    rng = random.Random(seed)
    drop = rng.sample(ids, n_drop)

    new = graph.copy()
    new.bulk_remove_nodes(sorted(drop))
    return new


KEEP_GRID = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
curve = []
stage2_failed = None

for keep in KEEP_GRID:
    rows = []
    for nm in sanity_names:
        gt_path = TRAIN / f"{nm}.geff"
        try:
            pred = load_graph(gt_path) if keep >= 1.0 else drop_nodes(load_graph(gt_path), keep)
            rows.append(score_one(pred, gt_path, TRAIN, nm))
        except NotImplementedError as exc:
            stage2_failed = str(exc)
            break
        except Exception as exc:
            print(f"  keep={keep} {nm}: {type(exc).__name__}: {exc}")
    if stage2_failed:
        break
    if rows:
        s = summarise(rows)
        curve.append((keep, s))
        print(f"keep={keep:.2f}  score={s['score']:.4f}  "
              f"adj_edge={s['adj_edge_jaccard']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
              f"n_pred={sum(r['num_pred_nodes'] for r in rows)}")

if stage2_failed:
    print(f"\nSTAGE 2 SKIPPED: {stage2_failed}")
    print("Stage 1 is unaffected and remains the deliverable for this run.")
elif curve:
    best = max(curve, key=lambda kv: kv[1]["adj_edge_jaccard"])
    baseline = curve[0][1]["adj_edge_jaccard"]
    print(f"\nbest keep_fraction on clean GT: {best[0]:.2f} "
          f"(adj_edge={best[1]['adj_edge_jaccard']:.4f} vs {baseline:.4f} at keep=1.0, "
          f"delta {best[1]['adj_edge_jaccard'] - baseline:+.4f})")
    print("Expected result is keep=1.00: on fully-annotated GT every removal destroys real")
    print("edges, so pruning can only lose. This confirms the slope, and says nothing about")
    print("real predictions where ~99% of nodes are metric-invisible. See stage 3.")

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
