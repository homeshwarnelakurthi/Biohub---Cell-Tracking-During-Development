"""Leave-one-embryo-out cross-validation for the Biohub cell-tracking metric.

Why this exists: the training set contains only two embryos — ``44b6`` (71 samples) and
``6bba`` (24) — and train/test are embryo-disjoint, with the hidden test set drawn from a
third embryo. Random-sample CV leaks embryo identity and will overstate every result. The
only honest protocol is leave-one-embryo-out, which here means two folds.

The official ``scripts/evaluate.py`` aggregates over all samples at once and drops the
sample names, so it cannot report per-embryo. This module reruns the same per-sample
scoring (identical calls into ``tracking_cellmot``) while tagging each row with its sample
and embryo, then summarises per fold.

Two folds is a weak signal. :func:`verdict` therefore refuses to average them: a change
ships only if it improves both.

Requires the official package on the path::

    pip install -e /kaggle/working/kaggle-cell-tracking-competition

See docs/STRATEGY.md and docs/MISTAKES.md M004.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def embryo_of(sample_name: str) -> str:
    """Embryo id from a sample folder name.

    Folder names are ``{embryo_id}_{field_of_view}``, e.g.
    ``44b6_0049_0438_1330_1273`` -> ``44b6``.
    """
    return sample_name.split("_", 1)[0]


def group_by_embryo(names) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        groups[embryo_of(n)].append(n)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def score_samples(pred_dir, gt_dir, max_distance: float = 7.0) -> list[dict]:
    """Score every sample present in both directories, tagged with sample and embryo.

    Returns ``per_sample_metrics`` dicts with two extra keys, ``sample`` and ``embryo``.
    Unreadable samples are skipped with a message rather than aborting the run.
    """
    import tracksdata as td
    from geff import GeffMetadata
    from tracking_cellmot.io import DEFAULT_SCALE, open_dataset
    from tracking_cellmot.metrics import (
        evaluate as compute_metric,
        node_recall,
        per_sample_metrics,
    )

    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)

    def _load(p):
        r = td.graph.IndexedRXGraph.from_geff(p)
        return r[0] if isinstance(r, tuple) else r

    def _scale(name):
        try:
            return open_dataset(gt_dir / name, load_image=False).scale
        except FileNotFoundError:
            return DEFAULT_SCALE

    def _n_total(geff_path):
        try:
            meta = GeffMetadata.read(geff_path)
        except Exception:
            return float("nan")
        val = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(val) if val is not None else float("nan")

    names = sorted({p.stem for p in pred_dir.glob("*.geff")}
                   & {p.stem for p in gt_dir.glob("*.geff")})

    rows: list[dict] = []
    for name in names:
        gt_path = gt_dir / f"{name}.geff"
        try:
            pred, gt = _load(pred_dir / f"{name}.geff"), _load(gt_path)
            er = compute_metric(pred, gt, scale=_scale(name), max_distance=max_distance)
            recall = (
                node_recall(pred, gt)
                if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
            )
            row = per_sample_metrics(er, _n_total(gt_path), recall)
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the run
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")
            continue
        row["sample"], row["embryo"] = name, embryo_of(name)
        rows.append(row)
    return rows


def fold_summaries(rows) -> dict[str, dict]:
    """Summarise per embryo, plus a pooled ``ALL`` entry.

    ``ALL`` is reported for continuity with the official script only. Do not use it to
    make ship/no-ship decisions — see :func:`verdict`.
    """
    from tracking_cellmot.metrics import summarise

    by_embryo = defaultdict(list)
    for r in rows:
        by_embryo[r["embryo"]].append(r)

    out = {emb: summarise(rs) for emb, rs in sorted(by_embryo.items())}
    out["ALL"] = summarise(list(rows))
    return out


def verdict(baseline: dict[str, dict], candidate: dict[str, dict],
            min_delta: float = 1e-4) -> tuple[bool, str]:
    """Decide whether a change ships, given two ``fold_summaries`` results.

    Ships only when every embryo fold improves by at least ``min_delta``. A change that
    wins on one fold and loses on the other is noise at two folds, so it is rejected
    rather than averaged.
    """
    folds = [k for k in baseline if k != "ALL" and k in candidate]
    if not folds:
        return False, "no comparable folds"

    deltas = {f: candidate[f]["score"] - baseline[f]["score"] for f in folds}
    detail = "  ".join(f"{f}: {d:+.4f}" for f, d in sorted(deltas.items()))

    if all(d >= min_delta for d in deltas.values()):
        return True, f"SHIP - improves every fold ({detail})"
    if all(d <= -min_delta for d in deltas.values()):
        return False, f"REJECT - degrades every fold ({detail})"
    return False, f"REJECT - inconsistent across folds, treat as noise ({detail})"


def format_report(summaries: dict[str, dict], title: str = "") -> str:
    """Render fold summaries with the metric terms broken out separately.

    The decomposition is the point: a combined score that moved without any underlying
    term moving is a measurement bug, not an improvement.
    """
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    header = (f"{'fold':<8}{'n':>4}{'score':>9}{'adj_edge':>10}{'edge_J':>9}"
              f"{'div_J':>8}{'div TP/FP/FN':>16}{'node_rec':>10}")
    lines += [header, "-" * len(header)]
    for fold, s in summaries.items():
        div = f"{s['division_tp']}/{s['division_fp']}/{s['division_fn']}"
        lines.append(
            f"{fold:<8}{s['n']:>4}{s['score']:>9.4f}{s['adj_edge_jaccard']:>10.4f}"
            f"{s['edge_jaccard']:>9.4f}{s['division_jaccard']:>8.4f}{div:>16}"
            f"{s['node_recall']:>10.4f}"
        )
    return "\n".join(lines)


def node_ratio_report(rows) -> str:
    """Per-embryo mean ``total_node_ratio`` — how far our node count sits from the target.

    Positive means we over-predict and are being penalised; negative means we under-predict
    and the adjusted-Jaccard multiplier is working in our favour. This is the readout that
    tells us whether the node-budget lever has any room left.
    """
    by_embryo = defaultdict(list)
    for r in rows:
        v = r.get("total_node_ratio")
        if v == v:  # not NaN
            by_embryo[r["embryo"]].append(v)

    lines = ["node budget (total_node_ratio = (N_pred - N_true)/N_true)"]
    for emb, vals in sorted(by_embryo.items()):
        mean = sum(vals) / len(vals)
        mult = 1 - 0.1 * mean
        lines.append(
            f"  {emb:<8} mean {mean:+.4f}  ->  multiplier {mult:.4f}  "
            f"({'penalised' if mean > 0 else 'rewarded'}, n={len(vals)})"
        )
    return "\n".join(lines)
