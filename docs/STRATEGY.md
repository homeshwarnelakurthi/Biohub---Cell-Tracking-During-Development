# Strategy — 0.913 to the top of the board

Last refreshed: 2026-08-03 (live Kaggle API pull).

## Where we actually stand

| | |
| --- | --- |
| Team | `Homii_N` (`homeshwarrao`) |
| Public LB | **0.913 — rank 200 / 1935 teams** |
| Submissions used | 3 (0.143, 0.913, 0.913) |
| #1 | 0.947 |
| Prize cutoff (7th) | 0.933 |
| Gap to prize | **+0.020** |
| Gap to #1 | **+0.034** |
| Final deadline | 2026-09-29 (57 days) |
| Team merger deadline | 2026-09-22 |
| Budget | 5 submissions/day, 12h notebook runtime, no internet at scoring |

## The single most important fact

**162 teams are tied at exactly 0.913**, spanning ranks 151–312. That is not a coincidence and it is
not a plateau in the problem — it is the score of an unmodified public notebook. Our current
submission is a fork of `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` running its
default preset, so we are one of those 162.

Two consequences:

1. Any genuine, non-cosmetic improvement moves us past ~150 teams immediately, because the
   distribution above us is thin: 0.913 → 0.916 is rank 200 → ~50.
2. Everyone in that band will be tuning the same knobs against the same public LB. The public
   leaderboard is 29% of the hidden test set; the private 71% decides the prizes. A field this
   compressed and this correlated is set up for a large shakeup, and robustness across embryos will
   decide it — not public-LB micro-tuning.

## Score decomposition — where the 0.034 can come from

`score = adj_edge_jaccard + 0.1 * division_jaccard`. Full derivation in
[METRIC_ANALYSIS.md](METRIC_ANALYSIS.md). Four levers, ordered by expected gain per GPU-hour:

### Lever 1 — confidence-ordered edge IDs (cost: ~0 GPU hours)
The metric truncates out-degree > 2 by *edge ID*, keeping whichever two edges were written first.
Sorting edges by descending confidence before writing converts a random loss into a chosen one.
Implemented in `src/biocell/submission.py`. Strictly non-negative in expectation.

### Lever 2 — node budget (cost: ~0 GPU hours, CPU sweep only)
The adjusted-Jaccard multiplier `(1 - 0.1 * (N_pred - N_true)/N_true)` is clamped only at the bottom,
so under-predicting nodes multiplies the score *up*, while the FP-enriched low-confidence tail often
raises the raw Jaccard as it is removed. There is an interior optimum. `src/biocell/node_budget.py`
locates it. This is the highest expected-value item on the list and needs no retraining.

### Lever 3 — divisions (cost: moderate)
Only 0.1-weighted, but the gap we need is 0.034. Moving division Jaccard from ~0.25 to ~0.60 is worth
about that entire gap on its own. Divisions are rare, noisy, and consequently under-tuned by the
field. The current pipeline gates them behind `SAFE_DIV_*` caps (`SAFE_DIV_FRAME_FRAC_CAP = 0.008`,
`SAFE_DIV_GLOBAL_FRAC_CAP = 0.004`) that were tuned for precision; whether that trade is right is an
open, measurable question.

### Lever 4 — link aggressiveness (cost: low)
False positives are only charged on edges touching an annotated GT node (METRIC_ANALYSIS property 2).
Since GT is sparse, a large share of spurious edges is invisible to the metric. The pipeline's
conservative linking thresholds are priced for a penalty that partly does not exist. Re-tune upward,
paid for out of the node budget from lever 2.

### Lever 5 — retraining detectors (cost: high — do last)
Only after 1–4 are exhausted and measured.

## The validation problem, and why it dominates everything

Train has **two embryos**: `44b6` (71 samples), `6bba` (24 samples). Train/test are embryo-disjoint
and the hidden test is a *different* embryo again.

So the only honest protocol is **leave-one-embryo-out, 2 folds**. Two folds is a weak signal. The
discipline that follows:

- A change ships only if it wins on **both** folds.
- A change that wins on one fold and loses on the other is noise; discard it, do not average it.
- Prefer flat optima to sharp ones. `best_of()` in `node_budget.py` deliberately picks the least
  aggressive cut among near-ties for this reason.
- Never tune a threshold on the public LB. With 5 submissions/day and 29% test coverage, that is
  fitting noise, and it is precisely how the 162-team cluster got stuck.

## Plan for the 57 days

**Phase 1 — instrumentation (days 1–7).** Get a local scorer running that reports `edge_jaccard`,
`adj_edge_jaccard`, `division_jaccard` and `total_node_ratio` *separately*, per embryo. Until we can
see which term is costing us, every change is guesswork. This is the gate for everything else.

**Phase 2 — free wins (days 7–14).** Levers 1 and 2. No retraining. Expect this to clear the 0.913
cluster.

**Phase 3 — divisions (days 14–35).** Lever 3, then lever 4. This is where the prize-zone gap is.

**Phase 4 — modelling (days 35–50).** Lever 5, only if phases 2–3 have been fully measured.

**Phase 5 — freeze (days 50–57).** No new ideas. Pick two final submissions: the best CV score and
the most robust (flattest-optimum) configuration. Given the expected shakeup, do **not** pick two
variants of the same tuning — pick one aggressive and one robust.

## Selecting final submissions

Kaggle scores the private LB on 71% of the test set that nobody has seen. With 162 teams tied and a
metric this sensitive to node counts, public rank is a poor predictor of private rank. Choose:

- **Submission A** — best 2-fold CV score, regardless of public LB.
- **Submission B** — the configuration whose score degrades least under ±20% perturbation of its
  main thresholds.

If A and B are the same configuration, deliberately weaken B until it isn't.

## Honest assessment of "position 1"

First place is 0.947 against our 0.913, with 1935 teams and 57 days. Levers 1–2 are close to free and
should move us out of the tied cluster; that part is low-risk. Closing the full 0.034 to #1 requires
lever 3 to work, and that is a genuine research question, not an engineering certainty — the leaders
are not standing still, and the top of the board will keep moving. A realistic target is the prize
zone (top 7, +0.020) with a real shot at higher if the division work lands. I will keep the estimate
updated against measured CV rather than restating the goal.
