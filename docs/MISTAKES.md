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
**Why it would happen.** 95 training samples looks like enough for 5-fold CV. It is not: they come
from only **two embryos** (`44b6` × 71, `6bba` × 24), train/test are embryo-disjoint, and the hidden
test is a third embryo. Random splits leak embryo identity and will overstate every result.
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
