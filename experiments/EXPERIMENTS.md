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

### Lineage Forge era, 2026-09-10 onward

Five weeks idle left us at rank 1361/3358 on a stale 0.914 while the field moved hard: top
0.949 -> 0.970, prize cutoff 0.935 -> 0.958. Adopted `biohub-lf-dctta-v020.ipynb` (public 0.939
declared) as the new base after checking it carries no metric exploit.

| ID | Change | LB | Notes |
| --- | --- | --- | --- |
| LF-v020 | Adopt Lineage Forge base unmodified | pending | Re-run reproduced the original output exactly (241,356 rows). Dual-seed ensemble, bidirectional harmonic fusion, DeepCenter TTA, built-in 8-sample holdout validator, automatic post-process sweep. |
| LF-v021 | v020 + restore ILP-proposed divisions (our E013 patch) | pending | +64 lines, 0 removed, one hunk inside `filter_output_graph`. Targets the base's 9 FN divisions. |

**Why v021 targets divisions specifically.** v020's own validator reports divisions at
3 TP / 1 FP / 9 FN - precision is already near-perfect, the loss is pure recall, ~75% missed. It
still calls `motion_relink_edges` with scipy `linear_sum_assignment`, the strict 1:1 Hungarian match
we showed in E011/E012 cannot express a division and which discards the ILP solution outright. So the
same structural cause we diagnosed in August is present in a 0.939 pipeline, and the same fix applies.

The patch sits inside `filter_output_graph`, which v020 calls from both the submission loop and its
holdout validator - so the validator scores the change for free. That does not repair M015 (the
holdout is still drawn from the distribution the weights were fit on), but it gives a directly
comparable base-vs-patched number from one harness.

**LF-v020 RESULT: 0.947** - higher than the 0.939 the notebook declared. Rank 1361 -> **209 / 3378**,
tied with 157 teams, gap to prize now **+0.011** (was +0.044).

**LF-v021 RESULT: confirmed no-op, and not submitted.** The patch logged
`ILP divisions restored: 0 (skipped, target already had a parent: 0)` on all 72 sample-runs - both
counters zero, so the loop never even found a source with two ILP children. Its validator reproduces
v020 exactly (base PROXY_SCORE 0.9490, divisions 3 TP / 1 FP / 9 FN), so v021 is functionally
identical to v020 and submitting it would spend a slot to re-measure 0.947.

The cause is arithmetic, not a bug in the patch. v020 sets
`ILP_DIVISION_WEIGHT = 1.2` and `ILP_APPEARANCE_WEIGHT = 0.0`. The solver minimises cost, so a
division costs `1.2 - edge_prob` against `0.0` for letting the daughter start a fresh track, and a
division only wins when `edge_prob > 1.2`. Probabilities cap at 1.0, so **v020's ILP cannot propose a
division at all** - its division machinery is dead code as configured, and every one of its 3 true
positives comes from the geometric `add_safe_divisions_postlink` heuristic.

That is the same cost comparison derived in August (METRIC_ANALYSIS property 5); v020's settings just
push it from "hard" to "impossible". Our patch is correct and does nothing here because there is
nothing to restore.

**Pre-registered for LF-v021 (recorded before the run finished).** Expect division recall to rise off
3 TP with FP staying low, since the no-existing-parent guard doubled as a precision filter in E013
(it removed 4 FP without costing a TP). Expect the validator to *overstate* the LB gain, per M015 -
E013 predicted +0.0255 on a fold and delivered +0.001 real. Sign trustworthy, magnitude not. A result
at or below v020 would mean the ILP divisions this pipeline proposes are lower quality than the old
baseline's, which would be worth knowing before spending more on the division line.

### LF-v022 division sweep: hypothesis refuted, 2026-09-12

Expanded the post-process sweep with 8 division-recall candidates on the argument that
`division_jaccard = TP/(TP+FP+FN)` weighs FP and FN equally, so a base sitting at low FP against high
FN should profit from loosening - break-even around one recovered division per four new false
positives.

**It did not.** Loosening lost on essentially every axis:

| config | proxy | delta | div_J | TP/FP/FN |
| --- | --- | --- | --- | --- |
| **tight55** (relink, not division) | **0.9406** | **+0.0031** | 0.2083 | 5/6/13 |
| dcdiv005 | 0.9386 | +0.0010 | 0.2083 | 5/6/13 |
| dcdiv010 | 0.9385 | +0.0009 | 0.2083 | 5/6/13 |
| base | 0.9375 | — | 0.2000 | 5/7/13 |
| diverge100 | 0.9368 | -0.0008 | 0.1875 | 6/14/12 |
| diverge150 | 0.9348 | -0.0028 | 0.1724 | 5/11/13 |
| loose_div_mild | 0.9343 | -0.0032 | 0.1667 | 5/12/13 |
| symtau050 | 0.9338 | -0.0037 | 0.1667 | 4/6/14 |
| loose_div_strong | 0.9329 | -0.0046 | 0.1515 | 5/15/13 |

The break-even arithmetic was right; the assumed exchange rate was wrong. `diverge100` was the only
candidate to buy a true positive at all, and it cost **7 false positives** for it - well past the 1:4
break-even. `loose_div_strong` added 8 FP and no TP. `symtau050` actually *lost* a true positive.

**Conclusion: v020's division filters are not over-tuned for precision, they are correctly tuned.**
The 13 false negatives are not divisions being wrongly rejected - they are divisions the geometric
candidate generator never proposes, so no amount of filter loosening can reach them. Recovering them
needs a different *candidate source*, not a looser gate.

The sweep's winner, `tight55`, is a motion-relink parameter and is the same config v020 already
selected, so v022's `submission.csv` is byte-identical in size (241,356 rows) and should score the
same 0.947.

Widening the holdout was worth it independently: at 8 samples the base read 3 TP / 1 FP / 9 FN
(div_J 0.2308); at 12 samples it reads 5/7/13 (div_J 0.2000). The smaller sample was optimistic, and
the whole loosening argument rested on that optimistic 1 FP.

### LF-v022 and LF-v023 leaderboard results, 2026-09-12

| submission | change vs v020 | validator (8-sample base) | public LB |
| --- | --- | --- | --- |
| LF-v020 (x2) | — | 0.9490 | **0.947, 0.947** |
| LF-v022 | division sweep, same selected config | (12-sample, not comparable) | 0.946 |
| LF-v023 | 350-epoch primary weights | **0.9646** (+0.0156) | **0.941** (-0.006) |

**v022 is noise, not a regression.** Its visible-test `submission.csv` is byte-identical to v020's
(same SHA256, 100% of node rows and 0 differing edges), and it selected the same `tight55` config.
Code-competition scores come from a hidden-set rerun and are shown to three decimals, so a 0.001
difference on identical visible output is either rounding of a sub-0.001 difference or rerun
nondeterminism. **Practical noise floor: changes of ~0.001 on the public LB are indistinguishable.**

**v023 is a real regression, and a clean M015 confirmation with the sign inverted.** The validator
said the 350-epoch weights were better by +0.0156 at base and ~+0.023 after the sweep; the leaderboard
says -0.006. On the held-out training samples v023 detects more cells (node ratio 0.905 -> 0.990) and
wins edge Jaccard on 5 of 8 samples. On the four visible test samples it emits 11% more nodes, with one
sample jumping from 20,729 to 34,907 (+68%).

The 350-epoch checkpoint is seven times more training on the two training embryos. The validator
holds out samples from those same embryos, so it rewards exactly that extra fit; the hidden test set
is a different embryo and punishes it. The sweep then compounded it by auto-selecting a three-way combo
(`tight55+dcgap035+relaxed9`) on the inflated signal - `relaxed9` had *lost* on v020's validator.

v020 (50-epoch weights) remains the best real score at 0.947.

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

**RESULT: 0.904. Prediction WRONG — and wrong in sign, not just magnitude.** The removed heuristic
was earning roughly 0.09 division Jaccard on the hidden test set while measuring 0.000 on our
validation samples. Root cause in MISTAKES M015: the pre-trained weights were fit on the training
set our validation samples are drawn from, so error-recovery heuristics look worthless on data the
model memorised. E007 is reverted; 0.913 (v1/E004) remains the best real score.

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

**v3 submission (recorded 2026-08-06, before submitting).** Predicted public LB: **sign genuinely
uncertain**, most likely a small move in either direction, roughly 0.905-0.918.

**RESULT: 0.914 — a new best, and the first score this project has moved off 0.913.** The "> 0.913"
branch below is the one that happened: the 155 ILP-proposed divisions add real true positives on
unseen data rather than net false positives. The division lever transfers, and it is worth pushing
further. Note the gain (+0.001) is far smaller than CV suggested (+0.0255 on `6bba`), which is
consistent with M015 — CV overstates division work, but does not invert its sign the way it did for
removals.

I am not claiming a direction, and the reason is specific rather than hedging. Division Jaccard is
`TP / (TP + FP + FN)`, so the 155 added divisions are *not* free: every one that is wrong and lands
on annotated ground truth enlarges the denominator and pushes the term **down**. v1 already earns
roughly 0.09 there from its 318 safe divisions (inferred from the v2 regression), so this can lose
ground as easily as gain it.

Two things damp the risk: most predicted divisions sit in unannotated regions where the metric cannot
see them at all (METRIC_ANALYSIS property 2), and everything that earned 0.913 is untouched.

Outcomes and what each would mean:
- **> 0.913** - ILP divisions add real true positives on unseen data; the division lever is live and
  worth pushing further (tighten with `BIOHUB_ILP_DIVISION_MIN_PROB`, then re-test).
- **~0.913** - the added divisions are mostly metric-invisible; neutral, and the lever needs
  targeting rather than volume.
- **< 0.913** - they are net false positives on real data. That would mean the ILP division signal
  does not transfer either, and the whole division line of attack is weaker than E012/E013 suggested.

## What to record per experiment

- The exact config diff, not a description of it.
- Both fold scores, separately — never their average alone.
- The **decomposed** metric terms. A score that moved without a term moving is a measurement bug.
- If it was submitted: the LB delta, and whether it matched the CV prediction. Divergence between CV
  and LB is itself a finding and belongs in MISTAKES.md.

---

# Public-code survey and the division lab (2026-09-12)

## Survey findings
Read the leaderboard-relevant public notebooks (index of 369 with title scores). What matters:

1. **0.963-0.966 notebooks are not reproducible.** They exploited the weakly-connected-component
   division rule patched 2026-07-17; their scores stand but cannot be re-earned. The reproducible
   public frontier is ~0.948 (`rishabhr0y/biohub-948-sew20`).
2. **The 0.948 notebook has fewer features than our 0.947 base**, not more: no DeepCenter TTA, no
   sister-symmetry gate, no validator-driven sweep; tighter safe-division caps (parent 7.0 / sister
   12.0 um vs 9.0 / 14.0), gap close 5.8 vs 5.0, DeepCenter `checkpoint_last.pt`, safe-div DC
   threshold 0.12.
3. **Both notebooks' validators use the retired division rule** (M017).
4. **Safe-division ranking prefers duplicates.** `score = parent_dist + 0.15 * sister_dist`, sorted
   ascending, fills the small per-frame budget with the tightest pairs - typically two detections of
   one cell - while real sisters sit ~10 um apart (megayak's GT audit: 35 of 151 GT divisions
   reachable under shipped gates). Opening the gates with this ranker hurt (0.9508 -> 0.9341); a
   better ranker with open gates has not been tested publicly.
5. **Knob landscape is flat.** busyaprime's 328 LB-scored versions show most single-knob changes
   within +/-0.001, our noise floor. Config mixing between the 0.947 and 0.948 lines cannot reach the
   prize line (0.959).

## Plan
- **A948** (`homeshwarrao/biohub-a948-base`): the 0.948 notebook verbatim. Reference + small real gain.
- **DIVLAB** (`homeshwarrao/biohub-divlab-948`): 0.948 pipeline on 40 held-out TRAIN clips, recording
  the graph entering `add_safe_divisions_postlink` plus a DeepCenter score per node. Not a submission.
- Locally, `src/biocell/replay948.py` lifts the notebook's post-division tail verbatim by AST and must
  reproduce the kernel's final graphs exactly before any variant is read.
  `experiments/divlab/divlab.py` then (a) labels every candidate proposal against GT, (b) replays
  ranker/gate variants, scoring with the official metric per embryo.
- Only variants that improve division Jaccard on **both** embryos without costing adjusted edge
  Jaccard go to a GPU run. M015 still applies: TRAIN clips were seen by the weights, so the
  expected bias is fewer duplicates than on test - which, if anything, understates a duplicate-aware
  ranker.
