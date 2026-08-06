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
| E003 | **Produce predicted geffs for the training samples** — stratified subset (6/embryo, 12 total), first pass via `biohub-validation-e003` | critical path | GPU | running |
| E005 | Node-budget sweep on real predictions — resolve the M009 question | 2 | CPU | blocked on E003 |
| E006 | Division threshold sweep (`SAFE_DIV_FRAME_FRAC_CAP`, `SAFE_DIV_GLOBAL_FRAC_CAP`, geometry gates) | 3 | GPU | blocked on E003 |
| E007 | Raise link aggressiveness, paid for from the node budget | 4 | GPU | blocked on E005 |
| E008 | Detector retraining | 5 | GPU × many | last resort |

**Critical path note.** E005–E007 all depend on the same missing input: predicted geffs over the
training set. Producing them (E003) is the bottleneck, not any individual lever. The harness is ready
and idle until it has predictions to score.

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

## What to record per experiment

- The exact config diff, not a description of it.
- Both fold scores, separately — never their average alone.
- The **decomposed** metric terms. A score that moved without a term moving is a measurement bug.
- If it was submitted: the LB delta, and whether it matched the CV prediction. Divergence between CV
  and LB is itself a finding and belongs in MISTAKES.md.
