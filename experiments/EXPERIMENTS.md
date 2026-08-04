# Experiment log

One row per experiment, including failures. An experiment without a row did not happen.

CV columns are **leave-one-embryo-out**: `44b6` means trained/tuned on `6bba` and scored on `44b6`,
and vice versa. A change ships only if both folds improve — see MISTAKES M004.

| ID | Date | Change | CV `44b6` | CV `6bba` | LB | Shipped | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E000 | 2026-07-29 | Fork of `pilkwang/...learned-graph-w-gap-recovery`, default preset `dual_seed_near_balanced_center_confirmed_synthetic_gap` | — | — | 0.913 | baseline | Tied with 161 other teams. No local CV existed — see MISTAKES M001. |
| E001 | 2026-08-02 | Re-run of E000 | — | — | 0.913 | no | Identical result; confirmed no effective change was made. |

## Queued

Ordered by expected gain per GPU-hour. Rationale for the ordering is in
[../docs/STRATEGY.md](../docs/STRATEGY.md).

| ID | Change | Lever | Cost | Status |
| --- | --- | --- | --- | --- |
| E002 | Stand up 2-fold embryo-disjoint local scorer reporting `edge_jaccard`, `adj_edge_jaccard`, `division_jaccard`, `total_node_ratio` separately | gate for everything | CPU | **next** |
| E003 | Confidence-ordered edge ids + explicit out-degree cap | 1 | ~0 | blocked on E002 |
| E004 | Node-budget sweep on existing predictions | 2 | CPU | blocked on E002 |
| E005 | Division threshold sweep (`SAFE_DIV_FRAME_FRAC_CAP`, `SAFE_DIV_GLOBAL_FRAC_CAP`, geometry gates) | 3 | GPU | blocked on E002 |
| E006 | Raise link aggressiveness, paid for from the node budget | 4 | GPU | blocked on E004 |
| E007 | Detector retraining | 5 | GPU × many | last resort |

## What to record per experiment

- The exact config diff, not a description of it.
- Both fold scores, separately — never their average alone.
- The **decomposed** metric terms. A score that moved without a term moving is a measurement bug.
- If it was submitted: the LB delta, and whether it matched the CV prediction. Divergence between CV
  and LB is itself a finding and belongs in MISTAKES.md.
