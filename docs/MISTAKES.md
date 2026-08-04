# Mistake log

Running record of errors, wrong turns and their corrections. Entries are appended, never
edited away — the point is to stop repeating them. Newest last.

Format: what happened → why it happened → what changes as a result.

---

## M001 — Submitting an unmodified public notebook and reading 0.913 as progress
**Date:** 2026-07-29 / 2026-08-02 (both submissions)
**What happened.** Our two scoring submissions are byte-equivalent in result (0.913, 0.913) and come
from a fork of `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` at its default preset.
The leaderboard shows **162 teams tied at exactly 0.913**, ranks 151–312. The score measured the
public notebook, not our work.
**Why.** No local evaluation existed, so the public LB was the only feedback channel. With no way to
tell "our change helped" from "the baseline is what it is", running the baseline felt like progress.
**Change.** Instrumentation before iteration. No further submissions until a local 2-fold
embryo-disjoint scorer reports edge/division/node-ratio terms separately. A submission that cannot be
predicted by CV within a reasonable band is not information, it is a coin flip against a 5/day quota.

---

## M002 — Working from a month-stale snapshot of the competition
**Date:** 2026-08-03
**What happened.** `Biohub.md` is a scrape of the competition page taken ~2026-07-03. It reports the
top public notebook at 0.861 and "3 months to go". Live API pull on 2026-08-03: top of leaderboard is
0.947, 1935 teams, 57 days left. Planning against the snapshot would have aimed at a bar that had
already moved by ~0.09.
**Why.** A static document was treated as the state of the world rather than as a point-in-time
capture.
**Change.** `tools/kaggle_status.py` pulls live standing, submissions and score distribution on
demand. Run it before any planning conversation. Treat `Biohub.md` as an archived artifact
(`docs/source/`), not as a reference.

---

## M003 — Tuning against the public leaderboard
**Date:** standing risk, not yet realised
**What happened.** Not yet — logged pre-emptively because it is the dominant failure mode for this
competition's shape.
**Why it would happen.** 5 submissions/day makes LB probing feel cheap. The public LB is 29% of the
test set; the private 71% decides prizes. 162 teams tied at one value means the field is collectively
overfitting the same signal.
**Change.** Thresholds are set on 2-fold embryo-disjoint CV only. The LB is used to confirm that a
CV-selected change did not break, never to choose between candidates.

---

## M004 — Assuming random-sample CV is valid here
**Date:** standing risk
**What happened.** Not yet realised.
**Why it would happen.** The sample count looks like enough for 5-fold CV. It is not: they come from
only **two embryos**, train/test are embryo-disjoint, and the hidden test is a third embryo. Random
splits leak embryo identity and will overstate every result.

> Counts corrected 2026-08-03: this entry originally said 95 samples (`44b6` × 71, `6bba` × 24). The
> real split is 199 (`44b6` × 71, `6bba` × 128) — see M010. The argument is unaffected; only the
> numbers were wrong.
**Change.** Leave-one-embryo-out only. Ship a change only if it wins on both folds. One-fold wins are
noise and get discarded rather than averaged.

---

## M005 — Broken base Python environment (numpy 2.5.1 vs numpy-1.x-compiled pandas)
**Date:** 2026-08-03
**What happened.** `import pandas` fails in the Anaconda base env with
`numpy.core.multiarray failed to import`; pandas 2.2.2 and pyarrow are compiled against numpy 1.x
while numpy 2.5.1 is installed. Pre-existing, not caused by this project's installs — but it silently
breaks any analysis script that touches pandas.
**Why.** Shared base environment mutated over time by unrelated projects.
**Change.** This project pins its own virtual environment (see README, Setup). Analysis code in
`src/` and `tools/` avoids pandas where the stdlib `csv` module suffices, so that tooling keeps
working even in a degraded interpreter.

---

## M006 — Letting the metric choose which edges to discard
**Date:** 2026-08-03 (identified by reading the official scorer)
**What happened.** The metric truncates out-degree > 2 by keeping the two **lowest edge IDs**, and
resolves merge-collapse the same way. Emitting edges in arbitrary order hands that choice to write
order rather than to confidence. This has been silently costing us on every division-adjacent node.
**Why.** The behaviour is in `metrics.py`, not in the competition description, so it is invisible
unless the scorer source is read.
**Change.** `src/biocell/submission.py` sorts edges by descending confidence before assigning ids,
and `cap_out_degree()` applies the cap ourselves by confidence. Read the scorer source, not the prose,
before trusting any assumption about how predictions are counted.

---

## M007 — Unpinned `pip install` broke numpy's ABI inside the Kaggle image
**Date:** 2026-08-03 (first run of `biohub-cv-harness`, killed the notebook)
**What happened.** `pip install tracksdata geff polars` pulled numpy up to 2.4.6. Every
pre-compiled extension in the Kaggle image was built against the older numpy, so the next
`import tracksdata` died with
`AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'`.
The whole run failed at cell 3.
**Why.** Treated the Kaggle image as a normal environment where installs are additive. It is a
pinned binary image, and *any* transitive numpy bump invalidates it. Exactly the same failure class
as M005 locally — noticed there, then walked into it again on a different machine.
**Change.** The harness now reads `numpy.__version__` first, writes it to a pip constraints file,
installs against `-c` that file, and **asserts numpy did not move** afterwards. The official repo is
installed with `--no-deps`. Any Kaggle notebook that pip-installs anything gets the same treatment.
The failure mode of a silent numpy bump is a dead kernel 10 minutes in, so assert early and loudly.

---

## M008 — Hardcoded the competition data path and did not check it
**Date:** 2026-08-03 (same run)
**What happened.** The notebook assumed
`/kaggle/input/biohub-cell-tracking-during-development/train` and printed `train geffs: 0`. Even if
the imports had worked, the harness would have scored an empty set and reported summary statistics
over nothing.
**Why.** The path was written from the competition slug rather than from the actual mount, and the
zero count was printed but not asserted on — so it would have scrolled past as informational output.
**Change.** `find_train_dir()` discovers the mount by globbing `/kaggle/input` for a directory that
actually contains `*.geff`, prints what it found and the per-embryo counts, and asserts non-empty.
General rule: every input path is discovered and asserted, never assumed. A count of zero is an
error, not a log line.

---

## M009 — Ranked the node-budget lever first on the strength of a simulation
**Date:** 2026-08-03 (claimed), corrected same day by run 3 of `biohub-cv-harness`
**What happened.** METRIC_ANALYSIS and STRATEGY both called the node-budget sweep "the highest
expected-value item" and "potentially the largest single gain". That ranking came from a sweep with
*invented* inputs — an assumed 35/65 TP/FP split in the pruned tail — which produced a headline
+0.047. No part of that was measured.

Then the harness measured the real quantities. `estimated_number_of_nodes` is 15k–64k per sample
against 51–788 annotated nodes, so the multiplier coefficient is `0.1/N_true ≈ 3.8e-6` per node.
Trimming a thin confidence tail is worth ~0.0017, not 0.047. The lever is real but its shape is
completely different from what was claimed: it pays only at large node counts, and the actual
argument for it is that ~99% of predicted nodes are metric-invisible yet still charged.

**Why.** A plausible mechanism was found in the scorer source and then quantified with placeholder
numbers, and the placeholder number got carried into a ranking as if it were evidence. The simulation
was labelled illustrative in the code comment but not in the conclusion, which is where it mattered.

**Change.** Levers do not get ranked before they are measured — an unmeasured lever is listed as
unresolved, with the mechanism stated and the magnitude explicitly marked unknown. When a number
comes from assumed inputs, the assumption goes in the same sentence as the number, not in a footnote.

---

## M010 — Counted training samples from a truncated API listing
**Date:** 2026-08-03
**What happened.** Enumerating competition files via the Kaggle API returned 12,000 entries, from
which the training set was reported as 95 samples (`44b6` × 71, `6bba` × 24). The actual mount shows
**199 samples — `44b6` × 71, `6bba` × 128**. The listing had been silently truncated by pagination,
and `6bba` was undercounted by more than 5×.
**Why.** The pagination loop terminated on a falsy `next_page_token` without checking whether the
returned count had hit a server-side cap, and the resulting number looked plausible so it was not
questioned.
**Change.** Counts that come from a paginated API are cross-checked against the mounted filesystem
before being used. The corrected split is materially better news for CV — `6bba` is a substantial
fold, not a 24-sample afterthought — which is exactly why the wrong number would have skewed
planning toward distrusting that fold.

---

## M011 — Kaggle derives the kernel slug from the title, not from the id you set
**Date:** 2026-08-03
**What happened.** `kernel-metadata.json` set `"id": "homeshwarrao/biohub-submission-v1"` with
`"title": "Biohub Submission v1 - confidence-ordered edges"`. Kaggle created the kernel at
**`biohub-submission-v1-confidence-ordered-edges`** — slugified from the title — and ignored the id,
emitting only a warning. The push reported success and even printed the intended URL, so it looked
fine. Every later `kernels_status()` call against the intended slug then failed with a 403, whose
message suggests a permissions problem rather than a naming one.
**Why.** The id field was assumed to be authoritative, and the warning on push was skimmed past
because the command exited zero and printed a plausible URL.
**Change.** `tools/make_submission_notebook.py` now has a `slugify()` helper and refuses to build
unless `slugify(TITLE) == SLUG`. Titles are kept short and slug-shaped; descriptive detail goes in the
commit message and the experiment log, not the notebook title. General rule: when a push warns, read
the warning even if the exit code is zero.

---

## M012 — `enable_gpu` is deprecated; without `machine_shape` Kaggle picked an unusable GPU
**Date:** 2026-08-04
**What happened.** The E004 submission notebook was pushed with `"enable_gpu": "true"` and nothing
else. Kaggle assigned a **Tesla P100 (sm_60)**, and the image's PyTorch ships kernels only for
sm_70–sm_120, so the first forward pass died with
`CUDA error: no kernel image is available for execution on the device`. Five minutes of GPU quota
burnt, and the traceback pointed at `torch.relu_`, which reads like a model bug rather than a
hardware-selection bug.
**Why.** `enable_gpu` is marked *DEPRECATED: use `machine_shape` instead* in the API type
definitions, and with `machine_shape` unset the accelerator falls back to Kaggle's default. The
existing baseline notebook had `machine_shape: "NvidiaTeslaT4"` set — inherited from having been
configured in the web UI — so it never hit this. Building fresh metadata from scratch silently
dropped a setting the baseline depended on.
**Change.** `make_submission_notebook.py` now reads `machine_shape` **and** `docker_image` from the
baseline's own `kernel-metadata.json` and carries both forward. Pinning the image matters
independently: Kaggle's rolling image could otherwise move the score under an experiment, making
environment drift indistinguishable from a real CV-vs-LB divergence.

General rule: when forking a working configuration, diff the *whole* metadata against it rather than
re-deriving the fields that seem relevant. The setting that breaks you is the one you did not know
was load-bearing.

