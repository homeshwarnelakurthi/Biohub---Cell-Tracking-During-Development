# Experiment log

One row per experiment, including failures. An experiment without a row did not happen.

CV columns are **leave-one-embryo-out**: `44b6` means trained/tuned on `6bba` and scored on `44b6`,
and vice versa. A change ships only if both folds improve — see MISTAKES M004.

| ID | Date | Change | CV `44b6` | CV `6bba` | LB | Shipped | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E000 | 2026-07-29 | Fork of `pilkwang/...learned-graph-w-gap-recovery`, default preset `dual_seed_near_balanced_center_confirmed_synthetic_gap` | — | — | 0.913 | baseline | Tied with 161 other teams. No local CV existed — see MISTAKES M001. |
| E001 | 2026-08-02 | Re-run of E000 | — | — | 0.913 | no | Identical result; confirmed no effective change was made. |
| E004 | 2026-08-03 | Confidence-ordered edge ids + explicit out-degree cap, patched onto the E000 baseline (`biohub-submission-v1`) | pending | pending | pending | running | Baseline writes edges in filter order and its only out-degree cap sits behind `OUTPUT_DIVISION_GEOMETRY_FILTER`, which defaults off — so the metric was truncating our links by write order. Only lever with a known sign. First run died on a P100 (M012); re-pushed pinned to NvidiaTeslaT4 + the baseline docker image. |
| E002 | 2026-08-03 | CV harness stood up on Kaggle (`biohub-cv-harness`) | n/a | n/a | — | **done** | Stage 1 passes: GT vs GT = edge Jaccard 1.0000 exactly. Took 3 runs (M007 numpy ABI, M008 data path). Measured GT annotation density **0.16–1.34%** and corrected the split to 199 samples (44b6×71, 6bba×128, M010). Stage 2 refuted the node-budget sizing (M009). |

## Queued

Ordered by expected gain per GPU-hour. Rationale for the ordering is in
[../docs/STRATEGY.md](../docs/STRATEGY.md).

| ID | Change | Lever | Cost | Status |
| --- | --- | --- | --- | --- |
| E003 | **Produce predicted geffs for the training samples** by running the forked pipeline over `train/` | critical path | GPU | queued |
| E005 | Node-budget sweep on real predictions — resolve the M009 question | 2 | CPU | blocked on E003 |
| E006 | Division threshold sweep (`SAFE_DIV_FRAME_FRAC_CAP`, `SAFE_DIV_GLOBAL_FRAC_CAP`, geometry gates) | 3 | GPU | blocked on E003 |
| E007 | Raise link aggressiveness, paid for from the node budget | 4 | GPU | blocked on E005 |
| E008 | Detector retraining | 5 | GPU × many | last resort |

**Critical path note.** E005–E007 all depend on the same missing input: predicted geffs over the
training set. Producing them (E003) is the bottleneck, not any individual lever. The harness is ready
and idle until it has predictions to score.

## What to record per experiment

- The exact config diff, not a description of it.
- Both fold scores, separately — never their average alone.
- The **decomposed** metric terms. A score that moved without a term moving is a measurement bug.
- If it was submitted: the LB delta, and whether it matched the CV prediction. Divergence between CV
  and LB is itself a finding and belongs in MISTAKES.md.
