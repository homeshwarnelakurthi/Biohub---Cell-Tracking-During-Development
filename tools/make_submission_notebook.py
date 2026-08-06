"""Build the submission notebook by patching the forked baseline.

The baseline (`notebooks/biohub/biohub.ipynb`, a fork of pilkwang's public notebook) is
138k characters of someone else's code. Rather than vendor a modified copy and lose track
of what we changed, this applies named, asserted patches to it. Every patch fails loudly if
its anchor text is not found exactly once, so a silent no-op is impossible.

Usage:
    python tools/make_submission_notebook.py --dry-run
    python tools/make_submission_notebook.py --push
    python tools/make_submission_notebook.py --push --slug biohub-submission-v2 \\
        --title "Biohub Submission v2" --disable-safe-divisions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "notebooks" / "biohub" / "biohub.ipynb"

DEFAULT_SLUG = "biohub-submission-v1"
DEFAULT_TITLE = "Biohub Submission v1"
COMPETITION = "biohub-cell-tracking-during-development"


def slugify(title: str) -> str:
    """Approximate Kaggle's title -> slug derivation."""
    out = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")

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

# ---------------------------------------------------------------------------
# Patch 2 (optional, --disable-safe-divisions) - turn off add_safe_divisions_postlink.
#
# E005 diagnosis (docs/METRIC_ANALYSIS.md property 5): division_like_sources exactly
# equals safe_divisions_added on every sample tested - the base linker produces zero
# natural forks, and every predicted division comes from this one heuristic, which
# attaches any unmatched orphan detection within a few microns of an existing track as
# a second child, purely on geometric proximity. Tested as E007 on 12 stratified train
# samples: division FP dropped 18 -> 0 with TP unchanged at 0 (this doesn't fix
# detection, it removes noise), and both embryo folds improved
# (44b6 +0.0025, 6bba +0.0002) - shipped per biocell.cv.verdict().
# ---------------------------------------------------------------------------

ANCHOR_2 = """COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = ["""

REPLACEMENT_2 = """# --- BIOCELL PATCH 2: disable add_safe_divisions_postlink (E007, shipped) ---
os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"
print("BIOCELL PATCH 2: BIOHUB_OUTPUT_SAFE_DIVISIONS -> 0 (heuristic disabled)")
# --- END BIOCELL PATCH 2 ---

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = ["""


def apply_patches(nb: dict, disable_safe_divisions: bool) -> dict:
    patches = [("confidence-ordered edges + out-degree cap", ANCHOR_1, REPLACEMENT_1)]
    if disable_safe_divisions:
        patches.append(("disable add_safe_divisions_postlink", ANCHOR_2, REPLACEMENT_2))

    applied = {name: 0 for name, _, _ in patches}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        changed = False
        for name, anchor, replacement in patches:
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


def build(slug: str, title: str, disable_safe_divisions: bool) -> Path:
    if slugify(title) != slug:
        raise SystemExit(
            f"title {title!r} slugifies to {slugify(title)!r}, not {slug!r}. "
            "Kaggle would create the kernel at the slugified title and the id would be ignored."
        )

    if not BASELINE.exists():
        raise SystemExit(f"baseline not found: {BASELINE}")
    nb = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"baseline: {BASELINE.name} ({len(nb['cells'])} cells)")
    nb = apply_patches(nb, disable_safe_divisions)

    out_dir = REPO / "notebooks" / slug.replace("biohub-", "", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"{slug}.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]

    # Both of these are inherited from the baseline notebook's server-side metadata and
    # both matter:
    #
    # machine_shape - `enable_gpu` alone is deprecated and leaves the accelerator to
    #   Kaggle's default, which handed the first run a Tesla P100 (sm_60). The image's
    #   PyTorch only ships kernels for sm_70+, so every forward pass died with
    #   "no kernel image is available for execution on the device". The T4 is sm_75.
    #
    # docker_image - pinned to the exact image the baseline scored with, so that a change
    #   in Kaggle's rolling image cannot silently move the score underneath an experiment.
    #   Without this, a CV-vs-LB divergence would be indistinguishable from environment
    #   drift.
    baseline_meta = json.loads(
        (BASELINE.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    machine_shape = baseline_meta.get("machine_shape") or "NvidiaTeslaT4"
    docker_image = baseline_meta.get("docker_image", "")
    print(f"  machine_shape: {machine_shape}")
    print(f"  docker_image:  {docker_image[:60]}{'...' if len(docker_image) > 60 else ''}")

    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "machine_shape": machine_shape,
        # Submission notebooks are scored with no network access. Must stay false.
        "enable_internet": "false",
        "competition_sources": [COMPETITION],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
    }
    if docker_image:
        meta["docker_image"] = docker_image

    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"wrote {out_nb}")
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true", help="push to Kaggle after building")
    p.add_argument("--dry-run", action="store_true", help="build only (default)")
    p.add_argument("--slug", default=DEFAULT_SLUG)
    p.add_argument("--title", default=DEFAULT_TITLE)
    p.add_argument("--disable-safe-divisions", action="store_true",
                    help="apply patch 2: disable add_safe_divisions_postlink (E007, shipped)")
    args = p.parse_args()

    out_dir = build(args.slug, args.title, args.disable_safe_divisions)

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
