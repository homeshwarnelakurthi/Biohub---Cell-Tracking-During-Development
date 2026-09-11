"""Patch the Lineage Forge notebook (public 0.939 base) with our division fix.

The base notebook is `biohub-lf-dctta-v020.ipynb` - 206k characters of someone
else's pipeline, adopted 2026-09-10 because our own best was 0.914 and five weeks
stale. As with the earlier baseline, we patch it by asserted anchor rather than
vendoring a modified copy, so an upstream change fails loudly instead of silently
no-opping.

What the patch does, and why this notebook specifically needs it:

Its own holdout validator reports divisions at 3 TP / 1 FP / 9 FN. Precision is
already excellent; the failure is recall - it misses ~75% of divisions. We
diagnosed the structural cause in August (E011/E012): `motion_relink_edges`
rebuilds every frame-to-frame link with scipy `linear_sum_assignment`, a strict
one-to-one Hungarian match that cannot express a division and that *replaces*
rather than refines the ILP's solution. Confirmed still present in this notebook.

So this restores a second child wherever the ILP proposed one and the relink
discarded it. Tested as E013 on the old baseline (+0.0006 / +0.0255 CV) and
shipped as submission v3, which moved the real leaderboard 0.913 -> 0.914.

Two things make this a better integration than E013 was:

  * It patches *inside* `filter_output_graph`, which this notebook calls from both
    the submission loop and its own holdout validator. So the validator measures
    the change for free - no separate CV run. That does not repair M015 (the
    holdout is still drawn from the distribution the weights were fit on), but it
    does give a directly comparable base-vs-patched number from the same harness.
  * It is gated by BIOHUB_RESTORE_ILP_DIVISIONS so the notebook's own
    post-process sweep can treat it as just another candidate.

Guard-safe by construction, as in E013: a division is added only when the target
currently has no parent (max in-degree stays 1 - the constraint that crashed E012)
and the source has exactly one child (max out-degree becomes 2, never 3).

Usage:
    python tools/make_lineage_forge_notebook.py --dry-run
    python tools/make_lineage_forge_notebook.py --push
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "biohub-lf-dctta-v020.ipynb"
COMPETITION = "biohub-cell-tracking-during-development"

DATASET_SOURCES = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]


def slugify(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


ANCHOR = (
    '    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n'
    '    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")\n'
    '\n'
    '    return nodes_by_id, edges, stats'
)

PATCH_BODY = '''
    # --- BIOCELL PATCH: restore ILP-proposed divisions discarded by motion relink ---
    # motion_relink_edges rebuilds links as a strict 1:1 Hungarian assignment, which
    # cannot express a division and replaces the ILP solution outright. This puts back
    # the second child wherever the ILP proposed one. Purely additive - no existing edge
    # or node is removed - so it cannot reproduce the E007 regression (MISTAKES M015).
    if os.environ.get("BIOHUB_RESTORE_ILP_DIVISIONS", "1") != "0":
        _ilp_min_prob = float(os.environ.get("BIOHUB_ILP_DIVISION_MIN_PROB", "0.0"))
        _ilp_children = {}
        for _re in raw_edges:
            _p = _re.get("edge_prob")
            _ilp_children.setdefault(int(_re["source_id"]), []).append(
                (int(_re["target_id"]), 0.0 if _p is None else float(_p))
            )

        _out_c = {}
        _in_c = {}
        _existing = set()
        for _e in edges:
            _s = int(_e["source_id"])
            _t = int(_e["target_id"])
            _out_c[_s] = _out_c.get(_s, 0) + 1
            _in_c[_t] = _in_c.get(_t, 0) + 1
            _existing.add((_s, _t))

        _added = 0
        _skipped_parent = 0
        for _src, _kids in _ilp_children.items():
            if len(_kids) < 2 or _out_c.get(_src, 0) != 1:
                continue                                  # keep max out-degree at 2
            _src_node = nodes_by_id.get(_src)
            if _src_node is None:
                continue                                  # source pruned upstream
            for _tgt, _prob in sorted(_kids, key=lambda kv: kv[1], reverse=True):
                if (_src, _tgt) in _existing:
                    continue
                _tgt_node = nodes_by_id.get(_tgt)
                if _tgt_node is None:
                    continue                              # target pruned upstream
                if int(_tgt_node["t"]) != int(_src_node["t"]) + 1:
                    continue                              # metric only counts t -> t+1
                if _in_c.get(_tgt, 0) != 0:
                    _skipped_parent += 1
                    continue                              # never create a merge
                if _prob < _ilp_min_prob:
                    continue
                edges.append({
                    "source_id": _src,
                    "target_id": _tgt,
                    "edge_prob": _prob,
                    "distance_um": edge_distance_um(_src_node, _tgt_node),
                    "ilp_division": 1,
                })
                _out_c[_src] = _out_c.get(_src, 0) + 1
                _in_c[_tgt] = 1
                _existing.add((_src, _tgt))
                _added += 1
                break                                     # at most one extra child
        stats["ilp_divisions_added"] = _added
        stats["ilp_divisions_skipped_has_parent"] = _skipped_parent
        print(f"  [{dataset}] ILP divisions restored: {_added} "
              f"(skipped, target already had a parent: {_skipped_parent})")
    # --- END BIOCELL PATCH ---
'''

REPLACEMENT = (
    '    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)\n'
    + PATCH_BODY
    + '\n    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")\n'
    '\n'
    '    return nodes_by_id, edges, stats'
)


def build(slug: str, title: str, strip_outputs: bool = True) -> Path:
    if slugify(title) != slug:
        raise SystemExit(f"title {title!r} slugifies to {slugify(title)!r}, not {slug!r}")
    if not BASE_NB.exists():
        raise SystemExit(f"base notebook not found: {BASE_NB}")

    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    applied = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        n = src.count(ANCHOR)
        if n == 0:
            continue
        if n > 1:
            raise SystemExit(f"anchor found {n} times, expected 1")
        cell["source"] = src.replace(ANCHOR, REPLACEMENT).splitlines(keepends=True)
        applied += 1
    if applied != 1:
        raise SystemExit(
            f"patch applied {applied} times, expected exactly 1 - "
            "the base notebook changed and the anchor needs updating"
        )
    print("  applied: restore ILP-proposed divisions (inside filter_output_graph)")

    if strip_outputs:
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
        nb["metadata"].pop("papermill", None)

    out_dir = REPO / "notebooks" / slug.replace("biohub-", "", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        # M012: enable_gpu alone lets Kaggle choose the accelerator, and a P100 (sm_60)
        # is incompatible with the image's PyTorch. This notebook hard-requires CUDA.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "false",
        "competition_sources": [COMPETITION],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out_dir / (slug + '.ipynb')}")
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--slug", default="biohub-lineage-forge-v021-ilpdiv")
    p.add_argument("--title", default="Biohub Lineage Forge V021 Ilpdiv")
    args = p.parse_args()

    out_dir = build(args.slug, args.title)

    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        meta = json.loads((out_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["enable_internet"] == "false", "submission notebook must be offline"
        print(f"pushing {meta['id']} ...")
        api.kernels_push(str(out_dir))
        print(f"https://www.kaggle.com/code/{meta['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
