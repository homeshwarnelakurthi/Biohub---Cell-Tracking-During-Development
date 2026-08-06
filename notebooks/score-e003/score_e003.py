"""Score the E003 predictions (biohub-validation-e003) against real ground truth,
per embryo. This is the first real measurement of the current pipeline's
edge/division/node-budget terms - everything downstream (levers 2-4) depends
on this actually producing numbers.

Attach `homeshwarrao/biohub-validation-e003` as a kernel source so its output
(/kaggle/working/biocell_pred_geffs/*.geff from that run) is mounted here.
CPU only - no GPU needed for scoring, just internet to install the scorer.
"""

# %% [markdown]
# # Score E003 predictions
#
# Reuses the plumbing-checked helpers from `biohub-cv-harness` (E002). Ground truth vs
# ground truth already proved edge Jaccard = 1.0 there, so the scoring path itself is
# trusted; this run is about the *predictions*, not re-validating the harness.

# %%
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NUMPY_PIN = f"numpy=={np.__version__}"
print("pinning", NUMPY_PIN)

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

# %%
def find_geff_dirs(max_depth: int = 6) -> list[Path]:
    """Walk /kaggle/input for every directory that directly contains *.geff files."""
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
            continue  # a geff directory's own subtree (nodes/, edges/) isn't a candidate
        stack.extend((c, depth + 1) for c in children if c.is_dir())
    return found


# Two mounted directories are expected to contain *.geff: the real training set (paired
# with matching *.zarr volumes) and this run's predictions (Patch B wrote geff only, no
# zarr, into /kaggle/working/biocell_pred_geffs on the E003 kernel). Discriminate on that,
# not on a guessed mount name - Kaggle's exact path for a kernel_source isn't documented
# and guessing it wrong already cost two runs on the CV harness (MISTAKES M007/M008).
candidates = find_geff_dirs()
print("directories containing *.geff:", [str(p) for p in candidates])
if len(candidates) < 2:
    raise FileNotFoundError(
        f"expected >=2 geff directories (train + predictions), found {len(candidates)}: "
        f"{candidates}"
    )

train_candidates = [p for p in candidates if any(p.glob("*.zarr"))]
pred_candidates = [p for p in candidates if not any(p.glob("*.zarr"))]
if len(train_candidates) != 1 or len(pred_candidates) != 1:
    raise FileNotFoundError(
        f"could not uniquely tell train from predictions by zarr-pairing: "
        f"train_candidates={train_candidates} pred_candidates={pred_candidates}"
    )
TRAIN, PRED_DIR = train_candidates[0], pred_candidates[0]
print(f"TRAIN    = {TRAIN}  ({len(list(TRAIN.glob('*.geff')))} geffs, has zarr)")
print(f"PRED_DIR = {PRED_DIR}  ({len(list(PRED_DIR.glob('*.geff')))} geffs, no zarr)")

# %% [markdown]
# ## Scoring helpers
#
# Same functions as `biohub-cv-harness` / `src/biocell/cv.py`, inlined so this notebook
# runs standalone.

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
    er = compute_metric(pred_graph, gt, scale=read_scale(gt_dir, name), max_distance=max_distance)
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
# ## Score every predicted sample against its matching ground truth

# %%
names = sorted({p.stem for p in PRED_DIR.glob("*.geff")} & {p.stem for p in TRAIN.glob("*.geff")})
print(f"{len(names)} samples to score: {names}")
missing_pred = sorted({p.stem for p in TRAIN.glob("*.geff")} - {p.stem for p in PRED_DIR.glob("*.geff")})
missing_gt = sorted({p.stem for p in PRED_DIR.glob("*.geff")} - {p.stem for p in TRAIN.glob("*.geff")})
if missing_gt:
    print(f"WARNING: {len(missing_gt)} predictions have no matching GT: {missing_gt[:10]}")

rows = []
for nm in names:
    try:
        pred = load_graph(PRED_DIR / f"{nm}.geff")
        rows.append(score_one(pred, TRAIN / f"{nm}.geff", TRAIN, nm))
        r = rows[-1]
        print(f"  {nm}: edge_J={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f} "
              f"n_pred={r['num_pred_nodes']} ratio={r['total_node_ratio']:+.4f} "
              f"div_tp/fp/fn={r['division_tp']}/{r['division_fp']}/{r['division_fn']} "
              f"node_recall={r['node_recall']:.4f}")
    except Exception as exc:
        print(f"  SKIP {nm}: {type(exc).__name__}: {exc}")

assert rows, "no samples scored - nothing to report"

print("\n" + format_report(fold_summaries(rows), "E003 - baseline predictions, per embryo"))

# %% [markdown]
# ## Node budget: how far are we from the target, per embryo?
#
# `total_node_ratio > 0` means we over-predict and the multiplier is working against us;
# `< 0` means we under-predict and it is working for us. See docs/METRIC_ANALYSIS.md
# property 1 - this is the number that was previously unmeasurable (MISTAKES M009).

# %%
by_embryo_ratio = defaultdict(list)
for r in rows:
    if r["total_node_ratio"] == r["total_node_ratio"]:
        by_embryo_ratio[r["embryo"]].append(r["total_node_ratio"])

print("node budget (total_node_ratio = (N_pred - N_true_est) / N_true_est):")
for emb, vals in sorted(by_embryo_ratio.items()):
    mean = sum(vals) / len(vals)
    mult = 1 - ADJUSTMENT_ALPHA * mean
    print(f"  {emb:<8} mean {mean:+.4f}  ->  multiplier {mult:.4f}  "
          f"({'penalised' if mean > 0 else 'rewarded'}, n={len(vals)})")
