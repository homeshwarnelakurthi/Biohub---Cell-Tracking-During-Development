# Metric Analysis — where the score actually lives

Source of truth: [`royerlab/kaggle-cell-tracking-competition`](https://github.com/royerlab/kaggle-cell-tracking-competition),
`src/tracking_cellmot/metrics.py` and `division_metrics.py`. Everything below is read off that
implementation, not off the competition prose. Where the prose and the code disagree, the code wins.

## The scoring formula

```
per sample:   total_node_ratio = (N_pred - N_true_est) / N_true_est
              J_edge           = TP / (TP + FP + FN)
              J_adj            = max(0, J_edge * (1 - 0.1 * total_node_ratio))

run level:    adj_edge_jaccard = weighted mean of J_adj, weights w_i = TP_i + FP_i + FN_i
              division_jaccard = micro-averaged  div_TP / (div_TP + div_FP + div_FN)
              score            = adj_edge_jaccard + 0.1 * division_jaccard
```

`N_true_est` is the `estimated_number_of_nodes` field in each `.geff`'s zarr metadata — **not** the
count of annotated nodes. Ground truth is sparsely labelled; that field estimates the *true total*
cell count including unlabelled cells.

Score ceiling is therefore ~1.1 before the node bonus, which is why the organizers note scores can
exceed 1.0. Current #1 is 0.947.

---

## Five structural properties worth exploiting

### 1. The node-count multiplier is uncapped on the upside

```python
adj_edge_jaccard = max(0.0, edge_jaccard * (1 - ADJUSTMENT_ALPHA * total_node_ratio))
```

`max(0, ...)` clamps the **bottom** only. When `N_pred < N_true`, `total_node_ratio` is negative and
the multiplier exceeds 1.0:

| `N_pred / N_true` | multiplier |
| ----------------- | ---------- |
| 1.30              | 0.970      |
| 1.00              | 1.000      |
| 0.90              | 1.010      |
| 0.75              | 1.025      |
| 0.50              | 1.050      |

So **emitting fewer nodes is directly rewarded**, independently of whether those nodes were correct.
Pruning the least-confident tail of detections trades edge recall (Jaccard down) against the
multiplier (up), and because the low-confidence tail is FP-enriched, the Jaccard often goes *up* too
over the first stretch of pruning. There is a well-defined interior optimum.

This is the single cheapest lever available: it is pure post-processing on an existing submission,
needs no retraining, and no GPU. `src/biocell/node_budget.py` finds the optimum from held-out counts.

> Caveat: the size of the gain depends entirely on the FP-composition of *our* confidence tail,
> which has to be measured on real predictions. Do not assume a number before measuring it.

### 2. False positives are only charged in annotated territory

```python
edge_attrs = edge_attrs.with_columns(
    (pl.col("out_valid") | pl.col("in_valid")).alias("pred_valid"),
)
...
edge_fp = edge_valid_pred - edge_tp
```

`out_valid` / `in_valid` are `True` only when the predicted endpoint matched a GT node that itself
has degree > 0; unmatched endpoints are `fill_null(False)`. An edge whose **both** endpoints match no
GT node has `pred_valid == False` and is excluded from the FP count entirely.

Because GT is sparse, **predicted edges in unannotated regions are invisible to the edge Jaccard.**
They cost nothing in FP. Their only cost is through `num_pred_nodes` in property 1.

Consequence: the linking threshold should not be tuned as if every spurious edge were punished. The
real currency is *node budget*, not edge precision. Aggressive linking on top of a tight node budget
is strictly better than timid linking on a loose one.

### 3. Out-degree is truncated by edge ID, not by confidence

```python
edge_attrs = edge_attrs.with_columns(
    pl.col(EDGE_ID).rank("ordinal").over(EDGE_SOURCE).alias("_out_rank")
)
edge_attrs = edge_attrs.filter(pl.col("_out_rank") <= 2)
```

A node with three or more outgoing edges keeps **the two lowest edge IDs** — i.e. whichever two we
happened to write first. Nothing about confidence enters. If our writer emits edges in arbitrary
order, we are letting the metric discard good links at random.

**Fix: sort edges by descending confidence before assigning IDs.** Costless, and strictly
non-negative in expectation. Same applies to the merge-collapse rule just above it, which keeps the
lowest edge ID per matched GT edge pair.

### 4. Only `t -> t+1` edges survive

```python
.filter(pl.col("_target_t") - pl.col("_source_t") == 1)
```

Edges spanning more than one frame are dropped silently — they do not even count as FP. Gap closing
across a missed frame therefore *must* materialize an intermediate node, and that node is charged
against the node budget in property 1. Every gap closed at distance 2 costs one node's worth of
multiplier. Worth it only if it recovers two edges that were otherwise FN.

### 5. Divisions are worth 0.1 — small, but the same size as the gap we need

`score = adj_edge_jaccard + 0.1 * division_jaccard`, micro-averaged over all samples.

We need +0.034 to reach #1. If our division Jaccard is around 0.25 and the leaders sit near 0.60,
that difference alone is +0.035 — the entire gap. Divisions are rare events, so this term is noisy
and under-optimized by most teams, which is exactly why it is where the leaders likely separate.

The division matcher (`division_metrics.py`) requires: local parent anchoring, two *distinct*
daughter branches, correct directed local topology, valid branch evidence, and unmerged branches. A
division that is topologically right but whose daughters merge back is scored as FP **and** FN.

---

## What this implies for a run plan

Ranked by (expected gain) / (GPU hours), highest first:

1. **Confidence-ordered edge IDs** — zero cost, removes a random-truncation loss (property 3).
2. **Node-budget sweep** on existing predictions — no GPU, potentially the largest single gain
   (property 1). Must be validated per-embryo, not on the public LB.
3. **Division precision/recall tuning** — the term is only 0.1-weighted but the gap we need is 0.034
   (property 5).
4. **Re-tuning link aggressiveness upward** now that FP is known to be nearly free in unannotated
   regions (property 2).
5. Retraining detectors — most expensive, do last.

## Validation constraint that governs all of it

Train contains **two embryos only**: `44b6` (71 samples) and `6bba` (24 samples). Train and test are
embryo-disjoint, and the hidden test set is a *different* embryo. Any CV that splits randomly across
samples leaks embryo identity and will overstate every result above.

The only honest protocol is **leave-one-embryo-out (2-fold)**: fit on `44b6`, score on `6bba`, and
the reverse. Two folds is a weak signal, so prefer changes that win on *both* folds and are robust to
threshold perturbation, over changes that win on one fold by a lot.

See [MISTAKES.md](MISTAKES.md) for the failure modes this is meant to prevent.
