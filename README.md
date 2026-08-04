# Biohub — Cell Tracking During Development

Working repository for the Kaggle competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
detect and track zebrafish embryo cells through 3D + time microscopy, reconstruct lineages,
and identify divisions.

Team `Homii_N` (`homeshwarrao`).

## Current standing

| | |
| --- | --- |
| Public LB | **0.913 — rank 200 / 1935** |
| Top score | 0.947 |
| Prize cutoff (7th) | 0.933 |
| Gap to prize / to #1 | +0.020 / +0.034 |
| Deadline | 2026-09-29 |

Refresh at any time:

```bash
python tools/kaggle_status.py
```

**Read this first:** our 0.913 is shared by **162 teams** (ranks 151–312) — it is the score of an
unmodified public notebook, not of anything we built. The board immediately above is thin: +0.003
moves us roughly 150 places. See [docs/STRATEGY.md](docs/STRATEGY.md).

## What is here

```
docs/
  METRIC_ANALYSIS.md   Exploitable structure in the official scorer — read before changing anything
  STRATEGY.md          Standing, score decomposition, ranked levers, 57-day plan
  COMPETITION.md       Task, data format, rules, timeline, submission format
  LITERATURE.md        Methods from the literature mapped onto our specific levers
  MISTAKES.md          Running log of errors and what changed as a result
  source/Biohub.md     Archived scrape of the competition page (2026-07-03) — stale, see M002
src/biocell/
  cv.py                Leave-one-embryo-out CV; ships a change only if both folds improve
  node_budget.py       Node-count/Jaccard trade-off optimiser (metric property 1)
  submission.py        Submission writer with confidence-ordered edge ids (metric property 3)
tools/
  kaggle_status.py     Live leaderboard, standing and submission history
  sync_notebook.py     Push a `# %%` script to Kaggle as a notebook
notebooks/
  cv-harness/          E002 validation harness (source of truth for the Kaggle notebook)
  <others>             Notebooks pulled from our Kaggle account
experiments/           Per-experiment results, appended over time
```

## The three facts that drive everything

1. **`score = adj_edge_jaccard + 0.1 × division_jaccard`**, where
   `adj = max(0, J × (1 − 0.1 × (N_pred − N_true)/N_true))`. The `max` clamps only the bottom, so
   *under-predicting nodes multiplies the score upward*. Node count is a tunable parameter, not a
   byproduct — though its size is still unresolved (MISTAKES M009).
2. **False positives are only charged on edges touching an annotated ground-truth node.** Measured
   annotation density is **0.16–1.34%**, so ~99% of predicted nodes are invisible to the edge
   Jaccard while still being charged against the node count. Conservative
   linking is priced for a penalty that partly does not exist.
3. **Training data contains two embryos** (`44b6` × 71 samples, `6bba` × 128), train/test are
   embryo-disjoint, and the hidden test is a third embryo. The only honest validation is
   leave-one-embryo-out, 2 folds. Random-sample CV leaks embryo identity.

Derivations and the scorer excerpts they come from are in
[docs/METRIC_ANALYSIS.md](docs/METRIC_ANALYSIS.md).

## Setup

The Anaconda base environment on this machine has a numpy/pandas ABI mismatch that breaks
`import pandas` (see MISTAKES M005). Use a dedicated environment:

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

Kaggle API credentials are read from `~/.kaggle/kaggle.json`.

Because the dataset is 87.6 GB and there is no local GPU, all training and inference runs on Kaggle
notebooks. This repo holds the code, the analysis and the experiment record; Kaggle holds the
compute.

## Working rules

- No threshold is ever tuned on the public leaderboard. CV decides; the LB only confirms.
- A change ships only if it wins on **both** embryo folds. One-fold wins are noise.
- Every experiment gets a row in `experiments/`, including the ones that failed.
- Read the scorer source before trusting the competition prose about how anything is counted.
