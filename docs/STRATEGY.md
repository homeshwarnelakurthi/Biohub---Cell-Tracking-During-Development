# Strategy — 0.913 to the top of the board

Last refreshed: 2026-08-04 (live Kaggle API pull).

## Where we actually stand

| | |
| --- | --- |
| Team | `Homii_N` (`homeshwarrao`) |
| Public LB | **0.913 — rank 238 / 1935 teams** (was 200 on 2026-08-03) |
| Submissions used | 4 (0.143, 0.913, 0.913, 0.913) |
| #1 | 0.948 |
| Prize cutoff (7th) | 0.933 |
| Gap to prize | **+0.020** |
| Gap to #1 | **+0.035** |
| Final deadline | 2026-09-29 (56 days) |
| Team merger deadline | 2026-09-22 |
| Budget | 5 submissions/day, 12h notebook runtime, no internet at scoring |

## The single most important fact

**165 teams are tied at exactly 0.913**, spanning ranks 189–353. That is not a coincidence and it is
not a plateau in the problem — it is the score of an unmodified public notebook. Our current
submission is a fork of `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` running its
default preset, so we are one of those 165.

The cluster is also *drifting downward*: the same 0.913 was rank 200 yesterday and is 238 today,
because teams outside it keep improving. A tied score is not a stable position.

Two consequences:

1. Any genuine, non-cosmetic improvement moves us past ~180 teams immediately, because the
   distribution above us is thin: 0.913 → 0.916 is rank 238 → ~50.
2. Everyone in that band will be tuning the same knobs against the same public LB. The public
   leaderboard is 29% of the hidden test set; the private 71% decides the prizes. A field this
   compressed and this correlated is set up for a large shakeup, and robustness across embryos will
   decide it — not public-LB micro-tuning.

## Score decomposition — where the 0.035 can come from

`score = adj_edge_jaccard + 0.1 * division_jaccard`. Full derivation in
[METRIC_ANALYSIS.md](METRIC_ANALYSIS.md). Four levers, ordered by expected gain per GPU-hour:

### Lever 1 — confidence-ordered edge IDs — DEAD (tested 2026-08-04)

Implemented, submitted, **LB 0.913 — unchanged**, exactly as pre-registered. The baseline already
enforces out-degree <= 2 upstream, so the metric rule this targets never fires, and the
merge-collapse rule cannot move TP/FP regardless of ordering. See METRIC_ANALYSIS property 3.

Kept in `src/biocell/submission.py` because it becomes live again if a future change (a more
permissive division policy under lever 3) ever emits three children.

### Lever 2 — node budget — MEASURED, small remaining upside
Measured on real predictions (E003-score, 2026-08-06, 12 stratified train samples): both embryo folds
are already net-rewarded — `44b6` mean ratio -0.306 (multiplier 1.031), `6bba` mean ratio -0.034
(multiplier 1.003). The baseline already under-predicts nodes on average and banks a small bonus for
free. 3 of 12 samples over-predict and would benefit from pruning; the other 9 are already past the
point where pruning pays. Full detail in METRIC_ANALYSIS property 1.

**Downgraded from "unresolved" to "small and mostly banked already."** Worth a light per-sample tune
later, not worth prioritising now.

### Lever 3 — divisions — MEASURED, and now the clear priority
Measured on the same 12 samples: **division_jaccard = 0.0000 on both folds** — 0 true positives out
of 25 division-related events (18 FP, 7 FN), on two embryos with very different sample sizes. This
was a guess ("~0.25") before 2026-08-06; it is now a hard number with cross-fold agreement.

The current pipeline gates divisions behind `SAFE_DIV_*` caps (`SAFE_DIV_FRAME_FRAC_CAP = 0.008`,
`SAFE_DIV_GLOBAL_FRAC_CAP = 0.004`) and geometry thresholds. Zero TP with both FP and FN present means
the failure is not a simple over/under-tuned threshold — tightening fixes FP at the cost of more FN
and vice versa. Needs inspection of specific failure cases, not a threshold sweep. Full detail,
including the caveat on evidence strength at n=12, in METRIC_ANALYSIS property 5.

**This is now the clearest priority in the project.** If the leaders are anywhere above ~0.35 division
Jaccard, this term alone explains the entire gap to #1.

### Lever 4 — link aggressiveness (cost: low)
False positives are only charged on edges touching an annotated GT node (METRIC_ANALYSIS property 2).
Since GT is sparse, a large share of spurious edges is invisible to the metric. The pipeline's
conservative linking thresholds are priced for a penalty that partly does not exist. Re-tune upward,
paid for out of the node budget from lever 2.

### Lever 5 — retraining detectors (cost: high — do last)
Only after 1–4 are exhausted and measured.

## The validation problem, and why it dominates everything

Train has **two embryos**: `44b6` (71 samples), `6bba` (128 samples), 199 total — counted from the
mounted dataset, correcting an earlier truncated-API estimate of 95 (MISTAKES M010). Train/test are
embryo-disjoint and the hidden test is a *different* embryo again.

So the only honest protocol is **leave-one-embryo-out, 2 folds**. Two folds is a weak signal. The
discipline that follows:

- A change ships only if it wins on **both** folds.
- A change that wins on one fold and loses on the other is noise; discard it, do not average it.
- Prefer flat optima to sharp ones. `best_of()` in `node_budget.py` deliberately picks the least
  aggressive cut among near-ties for this reason.
- Never tune a threshold on the public LB. With 5 submissions/day and 29% test coverage, that is
  fitting noise, and it is precisely how the tied cluster got stuck.

## Plan for the 57 days

**Phase 1 — instrumentation (days 1–7).** Get a local scorer running that reports `edge_jaccard`,
`adj_edge_jaccard`, `division_jaccard` and `total_node_ratio` *separately*, per embryo. Until we can
see which term is costing us, every change is guesswork. This is the gate for everything else.

**Phase 2 — free wins (days 7–14).** Lever 1 tested and dead. The phase collapses to its second
half: produce predicted geffs over the training set. Everything else is blocked on it.

**Phase 3 — divisions (days 14–35).** Lever 3, then lever 4. This is where the prize-zone gap is.

**Phase 4 — modelling (days 35–50).** Lever 5, only if phases 2–3 have been fully measured.

**Phase 5 — freeze (days 50–57).** No new ideas. Pick two final submissions: the best CV score and
the most robust (flattest-optimum) configuration. Given the expected shakeup, do **not** pick two
variants of the same tuning — pick one aggressive and one robust.

## Selecting final submissions

Kaggle scores the private LB on 71% of the test set that nobody has seen. With 165 teams tied and a
metric this sensitive to node counts, public rank is a poor predictor of private rank. Choose:

- **Submission A** — best 2-fold CV score, regardless of public LB.
- **Submission B** — the configuration whose score degrades least under ±20% perturbation of its
  main thresholds.

If A and B are the same configuration, deliberately weaken B until it isn't.

## Honest assessment of "position 1"

First place is 0.948 against our 0.913, with 1935 teams and 56 days.

Lever 1 tested dead. Lever 2 turned out small once measured. But E003-score (2026-08-06) changed the
picture: **division_jaccard = 0.0000 on both folds, with cross-fold agreement.** That is no longer a
guess feeding a ranking — it is a measured, structural failure, sized almost exactly to the gap that
separates us from #1. This is the first genuinely promising finding this project has had.

The field is still moving — our score hasn't changed since 29 July and rank drifted 200 -> 238 on the
same 0.913 — so there's real time pressure. But for the first time the next step isn't "measure and
see"; it's to go fix a specific, quantified problem.

Closing the full 0.035 to #1 still requires lever 3 to actually improve, not just be diagnosed — a
measured 0-TP baseline tells us where the problem is, not that it's fixable within 56 days. But it is
a real, well-defined engineering target now rather than an open research question about where the gap
lives. A realistic target is the prize zone (top 7, +0.020), with real upside if division detection
can be brought off zero. This estimate gets revised against measured CV, not restated as a goal.
