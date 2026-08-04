# Biohub — Cell Tracking During Development

Work on the Kaggle competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
detect and track zebrafish embryo cells through 3D + time microscopy, reconstruct lineages, and
identify divisions.

Team `Homii_N`.

## Approach

The scoring function, not tracking quality alone, decides placement here — so the work is driven by a
reading of the official scorer source rather than the competition description. Three facts shape
everything:

- Node count is a scored parameter in its own right, not a byproduct of detection.
- Ground truth is annotated extremely sparsely, so most of what a pipeline predicts is invisible to
  the edge metric while still being charged elsewhere.
- Training data spans only two embryos, and the hidden test set is a different one — so validation
  has to be leave-one-embryo-out, and a change ships only if both folds agree.

## Repository

| Path | Contents |
| --- | --- |
| `docs/` | Metric analysis, strategy, competition brief, literature, and a running mistake log |
| `src/biocell/` | Cross-validation, node-budget and submission utilities |
| `tools/` | Kaggle status and notebook sync |
| `notebooks/` | Notebooks, stored as `# %%` scripts so they diff in git |
| `experiments/` | Per-experiment record, including the failures |

Start with [docs/STRATEGY.md](docs/STRATEGY.md) for current standing and the plan, and
[docs/METRIC_ANALYSIS.md](docs/METRIC_ANALYSIS.md) before changing anything that affects scoring.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
```

Kaggle credentials are read from `~/.kaggle/kaggle.json`. The dataset is ~88 GB and there is no local
GPU, so training and inference run on Kaggle; this repo holds the code, the analysis and the record.

Check live standing with:

```bash
python tools/kaggle_status.py
```

## Working rules

- No threshold is tuned on the public leaderboard. CV decides; the leaderboard only confirms.
- A change ships only if it wins on both embryo folds. One-fold wins are noise.
- Every experiment gets a row in `experiments/`, including the ones that failed.
- Read the scorer source, not the competition prose, before trusting how anything is counted.
