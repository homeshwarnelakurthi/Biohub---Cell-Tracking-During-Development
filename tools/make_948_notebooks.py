"""Build notebooks on the public 0.948 base (rishabhr0y/biohub-948-sew20).

Two builds:

  a948    The base notebook as-is under our account. It scores 0.948 publicly
          against our 0.947, so it is both a small real gain and the reference
          every variant below is measured against.

  divlab  A research run, not a submission. It runs the 0.948 pipeline on a large
          held-out slice of TRAIN and records the graph exactly as it enters
          `add_safe_divisions_postlink`, together with a DeepCenter score for every
          node. Everything downstream of that point (safe divisions, geometry
          filter, prune, short-track filter, line-fit smoothing) is pure Python, so
          division-ranker and gate variants can then be replayed locally in seconds
          and scored with the *official* metric.

Why the replay instead of the notebook's own validator: its
`compute_division_confusion` still implements the weakly-connected-component rule
the organisers removed on 2026-07-17 (commit aa65e90), and reads division Jaccard
roughly 2x too high. Division variants judged by it are judged by the wrong rule.

Patches are applied by asserted anchor, as in make_lineage_forge_notebook.py.

Usage:
    python tools/make_948_notebooks.py a948 [--push]
    python tools/make_948_notebooks.py divlab [--push]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_NB = REPO / "bases" / "biohub-948-sew20.ipynb"
COMPETITION = "biohub-cell-tracking-during-development"
DATASET_SOURCES = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
]

BUILDS = {
    "a948": ("biohub-a948-base", "Biohub A948 Base"),
    "divlab": ("biohub-divlab-948", "Biohub Divlab 948"),
}

ENV_ANCHOR = 'os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"\n'
DIVLAB_ENV = ENV_ANCHOR + (
    "# divlab: a large held-out slice so division statistics are not a handful of events\n"
    'os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "20"\n'
)

SCORING_CELL_MARKER = "# CELL 9 (NEW) -- VALIDATOR: scoring against the official metric"

RECORDER_CELL = r'''# ============================================================
# DIVLAB -- record the graph entering add_safe_divisions_postlink
# ============================================================
# filter_output_graph resolves add_safe_divisions_postlink as a module global at
# call time, so rebinding the global intercepts the validator loop below without
# touching the pipeline. The snapshot is deep-copied because linefit smoothing
# later mutates node dicts in place.
import copy as _dl_copy
import gzip as _dl_gzip
import pickle as _dl_pickle
import time as _dl_time

DIVLAB_DIR = WORKING_DIR / "divlab"
(DIVLAB_DIR / "replay").mkdir(parents=True, exist_ok=True)
(DIVLAB_DIR / "final").mkdir(parents=True, exist_ok=True)

_dl_orig_safe_div = add_safe_divisions_postlink
_dl_orig_filter = filter_output_graph


def _dl_score_all_nodes(dataset, nodes_by_id, bundle, frame_cache, dc_cache):
    scores = {}
    if bundle is None or dataset is None:
        return scores
    for node_id in sorted(nodes_by_id, key=lambda nid: (int(nodes_by_id[nid]["t"]), nid)):
        node = nodes_by_id[node_id]
        s = deepcenter_score_point(dataset, int(node["t"]), node_point(node), bundle, frame_cache, dc_cache)
        if s is not None:
            scores[int(node_id)] = float(s)
    return scores


def add_safe_divisions_postlink(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None,
                                frame_cache=None, deepcenter_cache=None):
    _t0 = _dl_time.time()
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    snap = {
        "dataset": dataset,
        "nodes_by_id": _dl_copy.deepcopy(nodes_by_id),
        "edges": _dl_copy.deepcopy(edges),
        "dc_scores": _dl_score_all_nodes(dataset, nodes_by_id, deepcenter_bundle, frame_cache, deepcenter_cache),
    }
    with _dl_gzip.open(DIVLAB_DIR / "replay" / f"{dataset}.pkl.gz", "wb") as fh:
        _dl_pickle.dump(snap, fh, protocol=4)
    print(f"  [divlab] {dataset}: snapshot {len(nodes_by_id)} nodes, {len(edges)} edges, "
          f"{len(snap['dc_scores'])} DC scores in {_dl_time.time() - _t0:.1f}s")
    return _dl_orig_safe_div(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                             frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)


def filter_output_graph(nodes_by_id, raw_edges, dataset=None, deepcenter_bundle=None):
    out_nodes, out_edges, out_stats = _dl_orig_filter(nodes_by_id, raw_edges, dataset=dataset,
                                                      deepcenter_bundle=deepcenter_bundle)
    with _dl_gzip.open(DIVLAB_DIR / "final" / f"{dataset}.pkl.gz", "wb") as fh:
        _dl_pickle.dump({"nodes_by_id": out_nodes, "edges": out_edges, "stats": out_stats}, fh, protocol=4)
    return out_nodes, out_edges, out_stats


print("DIVLAB: recorder installed")
'''

EXPORT_CELL = r'''# ============================================================
# DIVLAB -- export ground truth, config and raw predictions
# ============================================================
import shutil as _dl_shutil

_dl_gt = DIVLAB_DIR / "gt"
_dl_raw = DIVLAB_DIR / "raw_pred"
_dl_gt.mkdir(parents=True, exist_ok=True)
_dl_raw.mkdir(parents=True, exist_ok=True)
for _stem in val_stems:
    _src = TRAIN_DIR / f"{_stem}.geff"
    if _src.exists():
        _dl_shutil.copytree(_src, _dl_gt / f"{_stem}.geff", dirs_exist_ok=True)
    # Zarr metadata only (no image chunks) so the official scorer can read the scale.
    _zsrc = TRAIN_DIR / f"{_stem}.zarr"
    for _meta in list(_zsrc.glob("*.json")) + list(_zsrc.glob(".z*")) + list(_zsrc.glob("0/*.json")) + list(_zsrc.glob("0/.z*")):
        _dst = _dl_gt / f"{_stem}.zarr" / _meta.relative_to(_zsrc)
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dl_shutil.copy2(_meta, _dst)
    _p = next((REPO_DIR / "predictions").rglob(f"{_stem}.geff"), None)
    if _p is not None:
        _dl_shutil.copytree(_p, _dl_raw / f"{_stem}.geff", dirs_exist_ok=True)

_dl_cfg = {k: v for k, v in globals().items()
           if k.isupper() and isinstance(v, (int, float, str, bool, tuple)) and not k.startswith("_")}
(DIVLAB_DIR / "config.json").write_text(json.dumps(_dl_cfg, indent=1, default=str))
(DIVLAB_DIR / "val_stems.json").write_text(json.dumps(val_stems, indent=1))

# Is any of TRAIN a genuine holdout for these weights? (M015)
for _root in [ARTIFACTS, REPO_DIR]:
    for _j in sorted(Path(_root).rglob("*split*.json"))[:20]:
        try:
            print("SPLITS FILE:", _j, _j.read_text()[:600])
        except Exception as _exc:
            print("SPLITS FILE (unreadable):", _j, _exc)

_dl_shutil.make_archive(str(WORKING_DIR / "divlab"), "zip", str(DIVLAB_DIR))
print("DIVLAB: wrote", WORKING_DIR / "divlab.zip")
'''


def slugify(title: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in title)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def _code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def build(kind: str) -> Path:
    slug, title = BUILDS[kind]
    assert slugify(title) == slug, (title, slug)
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))

    if kind == "divlab":
        codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
        env_cells = [c for c in codes if ENV_ANCHOR in "".join(c["source"])]
        assert len(env_cells) == 1, f"env anchor found in {len(env_cells)} cells"
        src = "".join(env_cells[0]["source"])
        assert src.count(ENV_ANCHOR) == 1
        env_cells[0]["source"] = src.replace(ENV_ANCHOR, DIVLAB_ENV).splitlines(keepends=True)

        idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and SCORING_CELL_MARKER in "".join(c["source"])]
        assert len(idx) == 1, f"scoring cell marker found {len(idx)} times"
        nb["cells"].insert(idx[0] + 1, _code_cell(EXPORT_CELL))
        nb["cells"].insert(idx[0], _code_cell(RECORDER_CELL))
        print("  applied: validator N=20, recorder cell, export cell")

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb.get("metadata", {}).pop("papermill", None)

    out_dir = REPO / "notebooks" / slug.replace("biohub-", "", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    meta = {
        "id": f"{username}/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4",  # M012
        "enable_internet": "false",
        "competition_sources": [COMPETITION], "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out_dir / (slug + '.ipynb')}")
    return out_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=sorted(BUILDS))
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    out_dir = build(args.kind)
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        meta = json.loads((out_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["enable_internet"] == "false"
        api.kernels_push(str(out_dir))
        print(f"https://www.kaggle.com/code/{meta['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
