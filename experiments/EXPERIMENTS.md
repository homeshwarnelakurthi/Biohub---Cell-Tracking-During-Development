# Experiment log

One row per experiment, including failures. An experiment without a row did not happen.

CV columns are **leave-one-embryo-out**: `44b6` means trained/tuned on `6bba` and scored on `44b6`,
and vice versa. A change ships only if both folds improve — see MISTAKES M004.

| ID | Date | Change | CV `44b6` | CV `6bba` | LB | Shipped | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E000 | 2026-07-29 | Fork of `pilkwang/...learned-graph-w-gap-recovery`, default preset `dual_seed_near_balanced_center_confirmed_synthetic_gap` | — | — | 0.913 | baseline | Tied with 161 other teams. No local CV existed — see MISTAKES M001. |
| E001 | 2026-08-02 | Re-run of E000 | — | — | 0.913 | no | Identical result; confirmed no effective change was made. |
| E004 | 2026-08-04 | Confidence-ordered edge ids + explicit out-degree cap, patched onto the E000 baseline (`biohub-submission-v1`) | — | — | **0.913** | **no — no-op** | **Prediction confirmed exactly.** Cap logged zero drops; `run_stats.csv` shows `dropped_multi_child_edges = 0` on every sample. The baseline's `filter_output_graph` already enforces out-degree <= 2, so the metric rule this targets never fires. Lever 1 is dead for this pipeline. Run cost: P100 failure (M012) + submit 403 (M013). Did validate the pinned T4 + docker reproduction path end to end. |
| E002 | 2026-08-03 | CV harness stood up on Kaggle (`biohub-cv-harness`) | n/a | n/a | — | **done** | Stage 1 passes: GT vs GT = edge Jaccard 1.0000 exactly. Took 3 runs (M007 numpy ABI, M008 data path). Measured GT annotation density **0.16–1.34%** and corrected the split to 199 samples (44b6×71, 6bba×128, M010). Stage 2 refuted the node-budget sizing (M009). |

## Queued

Ordered by expected gain per GPU-hour. Rationale for the ordering is in
[../docs/STRATEGY.md](../docs/STRATEGY.md).

| ID | Change | Lever | Cost | Status |
| --- | --- | --- | --- | --- |
| E003 | **Produce predicted geffs for the training samples** — stratified subset (6/embryo, 12 total), via `biohub-validation-e003` | critical path | GPU | **done** — 12/12 geffs produced, ~24 min predict time (~2 min/sample, scales to ~6.6h for the full 199) |
| E003-score | 2026-08-06 | Score E003 predictions (12 stratified train samples) against real GT, per embryo | 44b6: 0.9349 | 6bba: 0.9306 | n/a (measurement, not a change) | **done** | **division_jaccard = 0.0000 on BOTH folds** (0 TP / 18 FP / 7 FN across 25 events) — was a guess (~0.25), now measured and cross-fold-consistent. Node budget: both folds net-rewarded already (44b6 mult 1.031, 6bba mult 1.003), upside mostly banked. This is a baseline measurement of E000, not a lever test — no ship/no-ship verdict applies. |
| E005 | 2026-08-06 | Root-cause the 0-TP division result — read `add_safe_divisions_postlink` and cross-check `run_stats.csv` | n/a | n/a | n/a (analysis, not a run) | **done** | `division_like_sources == safe_divisions_added` exactly on 12/12 samples: the base linker produces zero natural forks, every predicted division comes from this one late-stage heuristic. It attaches any unmatched orphan detection within a few µm of an existing single-child node as a second child — pure geometric proximity, no confidence or morphology check. Largest sample nearly saturates `SAFE_DIV_GLOBAL_FRAC_CAP` (159 added vs ~163 cap), meaning it wants to fire even more. |
| E007 | 2026-08-06 | Disable `add_safe_divisions_postlink` (`BIOHUB_OUTPUT_SAFE_DIVISIONS=0`), same 12 samples, re-score | **44b6: 0.9374 (+0.0025)** | **6bba: 0.9308 (+0.0002)** | not yet submitted | **SHIP** (`biocell.cv.verdict()` confirmed) | division FP 18→0 on both folds; division_jaccard unchanged at 0.0 (TP still 0 — this removes noise, doesn't fix detection). Edge Jaccard +0.0023 on 44b6 (spurious edges had been touching annotated GT nodes), unchanged on 6bba (none of its removed edges were annotated-adjacent — its whole gain is from the node-budget multiplier ticking up). **Small, real, safe. Does not close the division gap by itself** — TP=0 is still unsolved. |
| E014 | Confirm E013 at 24 samples (12/embryo) before submitting — current CV rests on 7 GT divisions | 3 | GPU ~72min | **next** |
| E015 | Sweep `BIOHUB_ILP_DIVISION_MIN_PROB` if division FP needs tightening (11 FP vs 2 TP today) | 3 | GPU | after E014 |
| E006 | Node-budget per-sample tune on the 3 over-predicting samples | 2 | CPU | low priority |
| E008 | Raise link aggressiveness, paid for from the node budget | 4 | GPU | low priority |
| E009 | Detector retraining | 5 | GPU × many | last resort |

**Critical path note.** E003 unblocked everything and surfaced a measured problem: zero true-positive
divisions. E005 traced 100% of predicted divisions to one geometric heuristic, and E007 removed it
(shipped, both folds up). E010/E011 now attack the remaining half: the base ILP never proposes a
division at all, because under `division_weight = 1.0` a division only beats a fresh track
appearance when `edge_prob > 0.90` (see METRIC_ANALYSIS property 5). These sweep that bar down.

### Division experiments, 2026-08-06

| ID | Change | 44b6 | 6bba | verdict |
| --- | --- | --- | --- | --- |
| E011 | `division_weight` 0.6, relink ON | — | — | **0 forks** — weight is irrelevant while relink runs |
| E010 | `division_weight` 0.4, relink ON | — | — | same, 0 forks |
| E012 | `division_weight` 0.6, **relink OFF** | 0.8249 (-0.1125) | 0.9212 (-0.0096) | **REJECT** — degrades both folds |
| **E013** | **relink ON + restore ILP-proposed divisions** | **0.9380 (+0.0006)** | **0.9563 (+0.0255)** | **SHIP** — improves both folds |

**E013 is the first substantial gain this project has produced.** Keeping motion relink preserved
edge quality exactly (`44b6` edge Jaccard 0.9104 vs E007's 0.9099, node recall identical at 0.9866),
while restoring 539 ILP-proposed divisions lifted division Jaccard from a flat 0.0000 to **0.2500 on
`6bba`** — worth +0.0255 on that fold.

It also beat E012 on divisions despite E012 having *more* forks: 2 TP / 11 FP here vs 2 TP / 15 FP
there. Requiring the target to have no existing parent filtered 215 candidates and removed 4 false
positives without costing a true one — the guard-safety constraint turned out to also be a precision
filter.

**Why the folds differ so much (+0.0006 vs +0.0255):** all 3 of `44b6`'s ground-truth divisions are
still missed (0 TP / 7 FP), while `6bba` recovers 2 of 4. That asymmetry is unexplained and is the
main reason E014 re-runs at 24 samples before this is submitted — 7 GT divisions across 12 samples is
too thin to conclude the method genuinely works better on one embryo than the other.

**E012 is the informative one.** Disabling motion relink let the ILP's divisions survive: 461 natural
forks, and **2 true-positive divisions — the first this project has ever produced** (`6bba` division
Jaccard 0.2857, up from a flat 0.0000). The mechanism is proven: real divisions *are* recoverable
from the ILP.

But the cost is far larger than the gain. Motion relink turns out to be doing heavy lifting for edge
quality — without it, node recall fell 0.9866 -> 0.8840 on `44b6` and edge Jaccard 0.9099 -> 0.7941.
Divisions are weighted 0.1; edge Jaccard is ~92% of the score. Trading one for the other is a bad
deal at these magnitudes.

**Synthesis for E013:** the two useful signals are in different stages. Motion relink must stay (edge
quality), *and* the ILP knows where divisions are (2 TP proves it). So: run relink as now, then add a
second child **only where the ILP proposed a division**, rather than where a leftover detection
happens to be nearby. That replaces `add_safe_divisions_postlink`'s geometric proximity — measured at
0% precision in E005 — with a learned signal measured to produce real true positives.

## Pre-registered predictions

Written before the result is known, so the finding cannot be rationalised afterwards.

**E004 (recorded 2026-08-04, before submitting).** Predicted public LB = **0.913, unchanged**.
**RESULT: 0.913. Prediction correct.** The reasoning below held; lever 1 is a confirmed no-op and
METRIC_ANALYSIS property 3 has been amended with its missing precondition.

Reason: the run's own `run_stats.csv` reports `dropped_multi_child_edges = 0` and
`dropped_multi_parent_edges = 0` on every test sample, and our added cap logged zero drops. The
baseline's `filter_output_graph` already enforces out-degree <= 2 upstream, so the metric's
`_out_rank <= 2` truncation — the rule lever 1 targets — can never fire on this output.

The other id-order rule, merge-collapse, keeps the lowest edge id among predicted edges mapping onto
the same matched GT edge pair. All such duplicates map to the *same* GT edge, so whichever survives
is a TP either way; ordering cannot change TP/FP there either.

If this comes back at 0.913, lever 1 is a confirmed no-op for this pipeline and METRIC_ANALYSIS
property 3 needs its precondition stated: the rule is real, but the baseline never violates it.
If it comes back different, my reading of the scorer is wrong somewhere and that is worth more than
the points.

**E007 submission (recorded 2026-08-06, before submitting).** Predicted public LB: a small
improvement over 0.913, in the neighborhood of +0.001 to +0.003 (0.914-0.916), not a large jump.

Reason: on CV the fix improved both folds by +0.0025 (44b6) and +0.0002 (6bba) — real but modest,
because it only removes division false positives and cleans up a few rescued-orphan nodes; TP for
divisions is still 0, so `division_jaccard` itself is unmoved. The actual test-set distribution and
sample composition differ from the 12-sample CV subset, so the transferred magnitude is uncertain,
but the *sign* should hold — nothing about the mechanism (spurious edges/nodes in a heuristic that
demonstrably never produces a correct division) is specific to the CV samples.

If this comes back below 0.913, the CV-to-LB relationship for this fix doesn't transfer and needs
investigating before trusting CV again. If it lands in the predicted range, that is real confirmation
CV can be trusted here. If it lands much higher than +0.003, something about the real test
distribution differs from CV in a way worth understanding.

## What to record per experiment

- The exact config diff, not a description of it.
- Both fold scores, separately — never their average alone.
- The **decomposed** metric terms. A score that moved without a term moving is a measurement bug.
- If it was submitted: the LB delta, and whether it matched the CV prediction. Divergence between CV
  and LB is itself a finding and belongs in MISTAKES.md.
