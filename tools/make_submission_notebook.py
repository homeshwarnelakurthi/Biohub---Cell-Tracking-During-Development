"""Build the submission notebook by patching the forked baseline.

The baseline (`notebooks/biohub/biohub.ipynb`, a fork of pilkwang's public notebook) is
138k characters of someone else's code. Rather than vendor a modified copy and lose track
of what we changed, this applies named, asserted patches to it. Every patch fails loudly if
its anchor text is not found exactly once, so a silent no-op is impossible.

Usage:
    python tools/make_submission_notebook.py --dry-run
    python tools/make_submission_notebook.py --push
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "notebooks" / "biohub" / "biohub.ipynb"
OUT_DIR = REPO / "notebooks" / "submission-v1"

SLUG = "biohub-submission-v1"
# Kaggle derives the kernel slug from the TITLE and quietly ignores a non-matching id in
# kernel-metadata.json - it only emits a warning. A title of
# "Biohub Submission v1 - confidence-ordered edges" created the kernel at
# `biohub-submission-v1-confidence-ordered-edges`, and every subsequent status call against
# the intended slug returned 403. Keep the title such that slugify(TITLE) == SLUG; the
# assertion below enforces it.
TITLE = "Biohub Submission v1"
COMPETITION = "biohub-cell-tracking-during-development"


def slugify(title: str) -> str:
    """Approximate Kaggle's title -> slug derivation."""
    out = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


if slugify(TITLE) != SLUG:
    raise SystemExit(
        f"TITLE {TITLE!r} slugifies to {slugify(TITLE)!r}, not {SLUG!r}. "
        "Kaggle would create the kernel at the slugified title and the id would be ignored."
    )

# Attached model/support datasets, carried over from the baseline notebook. Required
# because submission notebooks run with internet disabled.
DATASET_SOURCES = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
]

# ---------------------------------------------------------------------------
# Patch 1 - emit edges in descending confidence order, and cap out-degree ourselves.
#
# The official scorer truncates out-degree > 2 by keeping the two LOWEST edge ids, and
# resolves merge-collapse the same way (metrics.py, `_out_rank` and `_is_merge_dup`).
# Nothing about confidence enters. The baseline writes `edges` in filter order, and its
# only out-degree cap sits behind OUTPUT_DIVISION_GEOMETRY_FILTER, which defaults to "0"
# - so today the metric is discarding our links essentially at random.
#
# `edge_sort_key` already exists in the baseline as (edge_prob, -distance_um); it is used
# inside filter_output_graph but never for the final write order.
#
# See docs/METRIC_ANALYSIS.md property 3.
# ---------------------------------------------------------------------------

ANCHOR_1 = """        division_sources: dict[int, int] = {}
        for edge in edges:"""

REPLACEMENT_1 = """        # --- BIOCELL PATCH 1: confidence-ordered edge ids + explicit out-degree cap ---
        # The metric keeps the two LOWEST edge ids per source when out-degree > 2, and the
        # lowest edge id per merged GT edge pair. Edge ids here are assigned in write
        # order, so writing highest-confidence first makes those tie-breaks keep our best
        # links rather than whichever edge happened to be emitted first.
        edges = sorted(edges, key=edge_sort_key, reverse=True)

        # Apply the out-degree cap ourselves, by confidence. The metric truncates to two
        # children regardless; doing it here means the surviving pair is the pair we chose,
        # and keeps the reported edge counts honest.
        _kept_edges: list[dict[str, object]] = []
        _out_count: dict[int, int] = {}
        _dropped_outdeg = 0
        for _edge in edges:
            _src = int(_edge["source_id"])
            if _out_count.get(_src, 0) >= 2:
                _dropped_outdeg += 1
                continue
            _out_count[_src] = _out_count.get(_src, 0) + 1
            _kept_edges.append(_edge)
        if _dropped_outdeg:
            print(f"  {dataset}: capped out-degree, dropped {_dropped_outdeg} lowest-confidence edge(s)")
        edges = _kept_edges
        # --- END BIOCELL PATCH 1 ---

        division_sources: dict[int, int] = {}
        for edge in edges:"""

PATCHES = [("confidence-ordered edges + out-degree cap", ANCHOR_1, REPLACEMENT_1)]


def apply_patches(nb: dict) -> dict:
    applied = {name: 0 for name, _, _ in PATCHES}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        changed = False
        for name, anchor, replacement in PATCHES:
            n = src.count(anchor)
            if n == 0:
                continue
            if n > 1:
                raise SystemExit(f"patch {name!r}: anchor found {n} times, expected 1")
            src = src.replace(anchor, replacement)
            applied[name] += 1
            changed = True
        if changed:
            cell["source"] = src.splitlines(keepends=True)

    for name, count in applied.items():
        if count != 1:
            raise SystemExit(
                f"patch {name!r} applied {count} times, expected exactly 1 - "
                "the baseline notebook has changed and the anchor needs updating"
            )
        print(f"  applied: {name}")
    return nb


def build() -> Path:
    if not BASELINE.exists():
        raise SystemExit(f"baseline not found: {BASELINE}")
    nb = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"baseline: {BASELINE.name} ({len(nb['cells'])} cells)")
    nb = apply_patches(nb)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nb = OUT_DIR / f"{SLUG}.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{SLUG}",
        "title": TITLE,
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        # Submission notebooks are scored with no network access. Must stay false.
        "enable_internet": "false",
        "competition_sources": [COMPETITION],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
    }, indent=1), encoding="utf-8")

    print(f"wrote {out_nb}")
    return OUT_DIR


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true", help="push to Kaggle after building")
    p.add_argument("--dry-run", action="store_true", help="build only (default)")
    args = p.parse_args()

    out_dir = build()

    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        meta = json.loads((out_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["enable_internet"] == "false", "submission notebook must have internet disabled"
        print(f"pushing {meta['id']} ...")
        api.kernels_push(str(out_dir))
        print(f"https://www.kaggle.com/code/{meta['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
