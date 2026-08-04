# Literature — methods worth stealing

Focused on lever 3 (divisions) from [STRATEGY.md](STRATEGY.md), because that is where the 0.034 gap
most plausibly lives. Entries are here because they map onto a concrete change we could make, not
because they are famous.

---

## Mitosis-aware multi-hypothesis tracking with aleatoric uncertainty

[Cell Tracking According to Biological Needs — Strong Mitosis-Aware Multi-Hypothesis Tracker With
Aleatoric Uncertainty](https://consensus.app/papers/details/07831549fb2754549171009a9ccd35f0/?utm_source=claude_code)
(Kaiser et al., 2024, IEEE Transactions on Medical Imaging)

The most directly relevant paper found. It targets exactly our failure mode: trackers that score well
on local association metrics while reconstructing lineage trees badly. Two ideas, both portable:

**1. Test-time augmentation as an uncertainty estimator.** Rather than committing to a single
predicted centroid per cell, they run problem-specific TTA and relax the point estimate into a
probabilistic spatial density. Our pipeline currently produces point centroids and then applies hard
geometric gates (`SAFE_DIV_MAX_UM = 4.7`, `SAFE_DIV_SISTER_MAX_UM = 7.2`, `DIV_PARENT_MAX_UM = 10.5`).
A hard gate on a noisy point estimate is exactly the thing a density replaces. Cheap version: run the
detector under a few flips/shifts, use the spread as a per-node confidence, and feed that confidence
into the division gate instead of a fixed micron threshold.

This also feeds lever 2 directly — TTA spread is a *principled* per-node confidence, which is what
`node_budget.py` needs to rank nodes for pruning.

**2. Mitosis-aware assignment with long-term conflict resolution.** They formulate the assignment so
divisions are modelled explicitly and false associations are resolved against *long-term* conflicts
rather than frame-to-frame cost. Our pipeline caps divisions with global frequency priors
(`SAFE_DIV_FRAME_FRAC_CAP = 0.008`, `SAFE_DIV_GLOBAL_FRAC_CAP = 0.004`) — a blunt instrument that
throws away true divisions in dense frames and admits false ones in sparse frames at the same rate.

Reported improvement is roughly 6× on biologically inspired metrics over the then state of the art.
Our division Jaccard is the same *kind* of metric, so this is the strongest single lead we have.

**Caveat before acting on it.** Their evaluation is on 2D Cell Tracking Challenge datasets. Ours is
3D and strongly anisotropic (z is 4× coarser than x/y), and the metric matches on *scaled* centroid
distance at 7.0 µm. Any density or TTA scheme has to be anisotropy-aware or it will be systematically
wrong in z. Do not port the thresholds; port the formulation.

---

## Tui — ILP tracking with lineage-aware division correction

[Tui: A Multigenerational and Expert-Correctable Tracker for Cellular
Dynamics](https://consensus.app/papers/details/29854271ffcc5c6e8bd41316166785c3/?utm_source=claude_code)
(Chang et al., 2026, Computational and Structural Biotechnology Journal). Source:
<https://github.com/hftsai/tui>

Integer linear programming plus explicit lineage-aware post-hoc correction of division and fusion
events. Our pipeline already runs an ILP (`USE_ILP=1`, with `ILP_DIVISION_WEIGHT = 1.0`), so the
architecture matches and the correction stage is the transferable part: they treat division
assignment as something to *repair* after global optimisation rather than to gate during it.

Relevant because the metric punishes a specific failure they target — a division whose daughters
later merge back scores as FP *and* FN. Their unmerged-branch handling addresses that case head-on.
Code is available, which makes this cheap to evaluate.

Caveat: validated on 2D datasets (DIC-C2DH-HeLa, Fluo-N2DH-SIM+) and synthetic data, at accuracies
(0.98–0.99) that reflect far easier tracking problems than dense 3D zebrafish embryos. Treat the
reported numbers as irrelevant and the mechanism as interesting.

---

## Spatiotemporal context for lineage construction

[Cell population tracking and lineage construction with spatiotemporal
context](https://consensus.app/papers/details/a26d8ebb0fbc5d9b873610386b49a856/?utm_source=claude_code)
(Li et al., 2008, Medical Image Analysis, 416 citations)

The classical reference. Combines bottom-up and top-down analysis with interacting multiple models
(IMM) motion filtering and spatiotemporal trajectory optimisation.

The IMM idea is the useful part for us: cells in an embryo are not one motion class — interphase
cells drift, dividing cells move differently, and the current pipeline models all of them with a
single motion prior (`MOTION_RELINK_VELOCITY_WEIGHT = 0.5`, one tight and one relaxed radius). A
two-mode motion model, where the second mode is division-like, is a modest change to the existing
relink stage and directly targets division recall.

Dated (2008, phase-contrast 2D, 87–93% accuracy) — it is here for the motion-model formulation, not
for its results.

---

## Reading queue

- Cell Tracking Challenge benchmark papers — the metric family our division Jaccard belongs to, and
  the standard failure taxonomy for lineage reconstruction.
- `trackastra` (transformer-based association) — a public notebook in this competition already uses
  it; worth understanding what it does better and worse than our current edge predictor.
- The `tracksdata` documentation, since the official scorer is built on it and its matching semantics
  determine what counts as a hit.

---

*Create or connect a free Consensus account to return more than 3 results per search in Claude Code.: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code*
