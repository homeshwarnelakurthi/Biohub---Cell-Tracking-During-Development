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

### How big is this really? (measured 2026-08-03, run 3 of `biohub-cv-harness`)

`N_true` is `estimated_number_of_nodes`, and it is enormous relative to the annotations:

| sample | annotated nodes | `estimated_number_of_nodes` | density |
| --- | --- | --- | --- |
| `44b6_0113de3b` | 52 | 26,000 | 0.20% |
| `44b6_0b24845f` | 51 | 31,875 | 0.16% |
| `44b6_0c582fdc` | 71 | 28,400 | 0.25% |
| `44b6_0db75fae` | 157 | 15,392 | 1.02% |
| `44b6_12dfb391` | 788 | 58,806 | 1.34% |
| `44b6_144b256d` | 121 | 63,684 | 0.19% |

**Ground truth annotates 0.16–1.34% of cells.** Two consequences, and they pull in opposite
directions:

The multiplier coefficient per node is `0.1 / N_true`, which at `N_true ≈ 26,000` is 3.8e-6 —
tiny per node. Dropping 500 nodes is worth about +0.0017; dropping 10,000 is worth about +0.035.
So the lever only pays at *large* node counts, not from trimming a thin confidence tail.

But combined with property 2 below, ~99% of predicted nodes match no GT node at all. Those nodes
contribute nothing to edge TP/FP/FN — they are invisible to the Jaccard — **while still counting
against `num_pred_nodes`**. Every metric-invisible node we emit is pure multiplier loss with no
offsetting benefit.

That is the real shape of this lever: not "trim the low-confidence tail", but "we are probably
paying multiplier for a very large number of nodes that cannot earn anything back".

### What the GT experiment showed, and why it does not settle the question

Dropping nodes from clean ground truth and re-scoring gave a strictly decreasing curve — best
keep-fraction 1.00, i.e. pruning only ever lost:

| keep | edge J | adj edge J |
| --- | --- | --- |
| 1.00 | 1.000 | 1.099 |
| 0.95 | 0.901 | 0.990 |
| 0.90 | 0.809 | 0.890 |
| 0.75 | 0.552 | 0.607 |

This is a **degenerate worst case, not a bound**. In ground truth every node is annotated, so every
removal destroys real edges (−0.099 J for +0.0002 multiplier at the 5% step). Real predictions are
~99% metric-invisible nodes, which is the opposite regime. The experiment confirms the mechanism and
the slope; it says nothing about the operating point.

> An earlier version of this document called it "the highest expected-value item" on the strength of
> a simulation with invented FP fractions. That was overstated — see MISTAKES M009.

### Measured against real predictions (E003-score, 2026-08-06, 12 stratified train samples)

```
44b6   mean total_node_ratio = -0.3060  ->  multiplier 1.0306  (rewarded)
6bba   mean total_node_ratio = -0.0339  ->  multiplier 1.0034  (rewarded)
```

**Both folds are already net-rewarded** — the baseline under-predicts nodes on average, banking a
small multiplier bonus (+3.1% on `44b6`, +0.3% on `6bba`) without any tuning. The upside remaining is
small and asymmetric: `44b6` has more room, `6bba` almost none.

The per-sample picture is mixed, not uniform: 9 of 12 samples under-predict (negative ratio,
rewarded), 3 over-predict (`6bba_6479435d` +0.046, `6bba_d6ecebbb` +0.097, `6bba_f8ffd5e7` +0.018).
Those three are exactly where pruning would help; the other nine are already past the point where
pruning pays. A single global threshold cannot serve both regimes — see `node_budget.py`'s per-sample
sweep for why this needs to be tuned per operating point, not as one global cut.

**Conclusion: this lever is measured, and the remaining upside is small.** Worth a light per-sample
tune once other levers are exhausted, not worth prioritising.

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

Measured annotation density is **0.16–1.34%** (table in property 1), so this is not a marginal
effect: the overwhelming majority of what the pipeline predicts is scored only through the node
count, never through edge precision.

Consequence: the linking threshold should not be tuned as if every spurious edge were punished. The
real currency is *node budget*, not edge precision. Aggressive linking on top of a tight node budget
is strictly better than timid linking on a loose one.

A second consequence, less obvious: because only ~1% of nodes can earn TP, the edge Jaccard is
estimated from a very small annotated subsample. Per-sample Jaccards will be noisy, which is another
reason to require both embryo folds to agree before shipping anything.

### 3. Out-degree is truncated by edge ID, not by confidence

```python
edge_attrs = edge_attrs.with_columns(
    pl.col(EDGE_ID).rank("ordinal").over(EDGE_SOURCE).alias("_out_rank")
)
edge_attrs = edge_attrs.filter(pl.col("_out_rank") <= 2)
```

A node with three or more outgoing edges keeps **the two lowest edge IDs** — i.e. whichever two we
happened to write first. Nothing about confidence enters.

### Tested, and worth nothing on this pipeline (E004, 2026-08-04)

The obvious fix — sort edges by descending confidence before assigning IDs — was implemented and
submitted. **Public LB came back 0.913, identical to the baseline.** Predicted in advance; see the
pre-registration in `experiments/EXPERIMENTS.md`.

The rule is real, but it has a precondition this pipeline never meets:

- Our added cap logged **zero** drops, and the baseline's own `run_stats.csv` reports
  `dropped_multi_child_edges = 0` and `dropped_multi_parent_edges = 0` on every test sample.
  `filter_output_graph` already enforces out-degree ≤ 2 upstream, so `_out_rank <= 2` can never fire.
- The merge-collapse rule is likewise inert for scoring: among predicted edges collapsing onto the
  *same* matched GT edge pair, whichever survives is a TP either way, so ID order cannot move TP/FP.

**Lesson worth more than the points:** a rule found by reading the scorer is only exploitable if our
own output actually violates it. Check the pipeline's existing behaviour against the rule before
ranking it as a lever. This one was ranked "sign known" purely on the scorer's text.

The property stays documented because it constrains any *future* change that raises out-degree — for
example a more permissive division policy under lever 3. If we ever emit three children, the ordering
fix becomes live again.

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

We need +0.034 to reach #1. Divisions are rare events, so this term is noisy and under-optimized by
most teams, which is exactly why it is where the leaders likely separate.

The division matcher (`division_metrics.py`) requires: local parent anchoring, two *distinct*
daughter branches, correct directed local topology, valid branch evidence, and unmerged branches. A
division that is topologically right but whose daughters merge back is scored as FP **and** FN.

### Measured: division_jaccard = 0.0000 on both embryo folds (E003-score, 2026-08-06)

This was a guess ("around 0.25") until now. The real number, from 12 stratified train samples scored
against real ground truth:

```
44b6   div TP/FP/FN = 0/15/3    division_jaccard = 0.0000
6bba   div TP/FP/FN = 0/3/4     division_jaccard = 0.0000
ALL    div TP/FP/FN = 0/18/7    division_jaccard = 0.0000
```

**Zero true-positive divisions across 25 division-related events, on both folds.** Not "weak" —
literally none of the 18 predicted divisions in this sample matched a real one, and none of the 7 real
divisions were recovered. This is not noise: it is the same result on two embryos with very different
node counts and sample sizes, which is exactly the kind of cross-fold agreement the leave-one-embryo
discipline (MISTAKES M004) is meant to surface as trustworthy.

The FP/FN split says the failure isn't one-directional — the pipeline both invents divisions that
aren't there (18 FP, concentrated in a few large/dense samples — `44b6_7a302da0` alone accounts for 7)
and misses real ones (7 FN spread more evenly). A single threshold nudge in either direction trades
one failure mode for the other; whatever is wrong here is structural, not a tuning knob.

**Caveat on strength of evidence:** n=12 is thin for a rare event — 25 division-events total is a
small sample to characterize a failure mode precisely. Zero-on-both-folds is strong enough to justify
investigating now, not yet strong enough to claim the true division Jaccard is exactly zero on the
full test distribution. The next step is to inspect specific FP/FN cases, not just the aggregate.

If the leaders are anywhere above ~0.35 division Jaccard, this term alone accounts for the entire gap
to #1.

### Root cause (E005, 2026-08-06): the entire division-prediction capability is one heuristic

`run_stats.csv` from E003 has a `division_like_sources` column (out-degree ≥ 2 nodes in the final
output) and a `safe_divisions_added` column (edges added by the post-link repair function
`add_safe_divisions_postlink`). They are **exactly equal on all 12 samples**:

```
44b6_7a302da0   division_like_sources=159   safe_divisions_added=159
44b6_eb2880fc   division_like_sources=125   safe_divisions_added=125
... (10 more, all exact matches, including 0==0 on the one sample with no divisions)
```

**The base linker/ILP produces zero natural forks.** Every predicted division in every sample comes
from one late-stage repair step, not from tracking or detection.

Reading `add_safe_divisions_postlink` explains why it has 0% precision where measurable. For every
node with exactly one existing child, it searches the next frame for an *unmatched, orphan* detection
— one the primary linker failed to attach to anything — within `SAFE_DIV_MAX_UM` (4.7 µm) of the
source and `SAFE_DIV_SISTER_MAX_UM` (7.2 µm) of the existing child, and attaches it as a second child.
**No confidence score, no morphological signal — purely geometric proximity to a leftover detection
the main linker already gave up on.** In densely packed embryo tissue, an unrelated neighboring cell
sitting within a few µm of an existing track is unremarkable; the heuristic cannot tell that apart
from a genuine division.

The scale confirms it: on the largest sample (`44b6_7a302da0`, 40,718 nodes), the heuristic added 159
divisions against a `SAFE_DIV_GLOBAL_FRAC_CAP`-implied ceiling of ≈163 — **it is nearly saturating the
cap**, meaning it wants to fire even more often and is being stopped by an arbitrary global limit, not
by any evidence of a real division.

There is a compounding side effect: a component that "has a division" is exempted from the
`OUTPUT_MIN_TRACK_LEN` short-track filter (`OUTPUT_KEEP_DIVISION_COMPONENTS`). So this heuristic isn't
just adding wrong edges — it is actively rescuing noisy orphan detections from being pruned, by
mislabelling them as division daughters.

See MISTAKES.md for the fix tested against this diagnosis (E007) and its CV result.

### Why the base linker never divides: the ILP arithmetic (E010, 2026-08-06)

Removing the bad heuristic (E007) left division_jaccard at 0 because the underlying tracker never
proposes a division at all. That is not a bug — it is what the configured ILP costs mathematically
require.

`ILPSolver` **minimises** total cost. The baseline configures:

```
edge_weight       = -1.0 * edge_prob     # a reward, in [-1, 0]
division_weight   = +1.0                 # a penalty
appearance_weight = +0.1                 # penalty for a track starting fresh
```

For a parent already linked to one child, a second daughter is either a **division**
(cost `1.0 - edge_prob`) or just a **new track appearing** (cost `0.1`). The solver takes the cheaper
one, so a division is only ever chosen when

```
1.0 - edge_prob < 0.1     ->     edge_prob > 0.90
```

**Divisions require a second-child edge probability above 0.90 under the default weights.** In
practice nothing clears that bar, which is why `division_like_sources` is exactly 0 once the
heuristic is off. It is not a detection failure or a threshold that needs nudging — it is the
objective function declining to ever pay for a division.

`division_weight` moves that bar directly: the division is preferred when
`edge_prob > division_weight - appearance_weight`.

| `division_weight` | division accepted when |
| --- | --- |
| 1.0 (baseline) | `edge_prob > 0.90` |
| 0.6 | `edge_prob > 0.50` |
| 0.4 | `edge_prob > 0.30` |

Being explicit about the risk: lowering it far enough will start producing divisions, but nothing
here guarantees they are the *right* ones — too low and every confident edge pair forks, trading a
0-TP/0-FP state for a 0-TP/many-FP one, which is strictly worse. E010 (0.4) and E011 (0.6) test this
on the same 12 samples, and `biocell.cv.verdict()` decides on both folds as usual. That makes it the clearest priority target this project has had so far.

---

## What this implies for a run plan

Ranked by (expected gain) / (GPU hours), highest first:

1. ~~**Confidence-ordered edge IDs**~~ — **tested, no-op** (E004, LB 0.913 unchanged). The pipeline
   already satisfies the rule; see property 3.
2. **Node-budget sweep** on real predictions — no GPU. Size unknown; the mechanism is confirmed but
   the operating point is not (property 1).
3. **Division precision/recall tuning** — the term is only 0.1-weighted but the gap we need is ~0.035
   (property 5).
4. **Re-tuning link aggressiveness upward** now that FP is known to be nearly free in unannotated
   regions (property 2).
5. Retraining detectors — most expensive, do last.

With lever 1 dead, **nothing remaining has a known sign**, and levers 2–4 all depend on the same
missing input: predicted geffs for the training samples. Producing them is the critical path, and
until they exist no lever can be evaluated at all.

## Validation constraint that governs all of it

Train contains **two embryos only**: `44b6` (71 samples) and `6bba` (128 samples), 199 total —
counted from the actual Kaggle mount, which supersedes the truncated file-listing estimate of 95 that
earlier drafts used. Train and test are embryo-disjoint, and the hidden test set is a *different*
embryo. Any CV that splits randomly across samples leaks embryo identity and will overstate every
result above.

The only honest protocol is **leave-one-embryo-out (2-fold)**: fit on `44b6`, score on `6bba`, and
the reverse. Two folds is a weak signal, so prefer changes that win on *both* folds and are robust to
threshold perturbation, over changes that win on one fold by a lot.

See [MISTAKES.md](MISTAKES.md) for the failure modes this is meant to prevent.
