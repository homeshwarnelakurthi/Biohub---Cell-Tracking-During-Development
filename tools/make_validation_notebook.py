"""Build a validation notebook (E003): run the E000 baseline pipeline over a
stratified subset of TRAIN instead of the real test set, and save the final
predicted graphs as .geff so the CV harness can score them against real
ground truth.

Every other lever (node budget, division tuning, link aggressiveness) is
blocked on having real predictions to measure against - see
experiments/EXPERIMENTS.md, "Critical path note". This notebook produces
that input.

Two patches, both applied to the unmodified E000 baseline
(notebooks/biohub/biohub.ipynb), deliberately NOT the E004-patched copy -
E004 tested as a confirmed no-op (see docs/MISTAKES.md M014), and comparisons
for future levers should be against the same pure baseline documented as E000
in EXPERIMENTS.md, not muddied by an inert change.

Patch A - redirect TEST_DIR at a stratified subset of TRAIN by setting the
existing BIOHUB_TEST_DIR environment override before it is read. The pipeline
already discovers sample stems and builds its own splits file dynamically
from whatever is in TEST_DIR (verified by reading the baseline source), and
every downstream guard rail computes its expectations from TEST_DIR too - so
nothing else needs to change.

Patch B - after filter_output_graph produces the final (nodes_by_id, edges)
for a dataset, also build a tracksdata graph from them and save it as .geff
under /kaggle/working. This is the same object the CSV writer serialises, so
it faithfully represents what a real submission would score.

Usage:
    python tools/make_validation_notebook.py --dry-run
    python tools/make_validation_notebook.py --push --samples-per-embryo 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "notebooks" / "biohub" / "biohub.ipynb"

DEFAULT_SLUG = "biohub-validation-e003"
DEFAULT_TITLE = "Biohub Validation E003"
COMPETITION = "biohub-cell-tracking-during-development"

DATASET_SOURCES = [
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
]


def slugify(title: str) -> str:
    out = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


# ---------------------------------------------------------------------------
# Patch A - stratified TRAIN subset via the existing BIOHUB_TEST_DIR override
# ---------------------------------------------------------------------------

ANCHOR_A = """COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = ["""

REPLACEMENT_A_TEMPLATE = """# --- BIOCELL PATCH E003a: point TEST_DIR at a stratified TRAIN subset ---
# The pipeline discovers sample stems from TEST_DIR.iterdir() and builds its own
# splits file dynamically (see list_test_stems() below) - nothing else needs to
# change for it to run over training data instead of the real test set.
import random as _rnd_e003

_train_root_e003 = None
for _root_e003 in sorted(Path("/kaggle/input").glob("*")):
    for _cand_e003 in _root_e003.rglob("train"):
        if _cand_e003.is_dir() and any(_cand_e003.glob("*.geff")):
            _train_root_e003 = _cand_e003
            break
    if _train_root_e003:
        break
if _train_root_e003 is None:
    raise FileNotFoundError("E003: no train directory with .geff files found under /kaggle/input")

_per_embryo_e003 = {samples_per_embryo}
_by_embryo_e003: dict[str, list[str]] = {{}}
for _p_e003 in sorted(_train_root_e003.glob("*.geff")):
    _emb_e003 = _p_e003.stem.split("_", 1)[0]
    _by_embryo_e003.setdefault(_emb_e003, []).append(_p_e003.stem)

_rng_e003 = _rnd_e003.Random(0)
_chosen_e003: list[str] = []
for _emb_e003 in sorted(_by_embryo_e003):
    _stems_e003 = sorted(_by_embryo_e003[_emb_e003])
    _rng_e003.shuffle(_stems_e003)
    _chosen_e003.extend(_stems_e003[:_per_embryo_e003])
_chosen_e003 = sorted(_chosen_e003)
print(f"E003: {{len(_chosen_e003)}} samples across {{len(_by_embryo_e003)}} embryos: {{_chosen_e003}}")
if not _chosen_e003:
    raise RuntimeError("E003: stratified sample selection produced zero samples")

_subset_dir_e003 = Path("/kaggle/working/biocell_train_subset")
_subset_dir_e003.mkdir(parents=True, exist_ok=True)
for _stem_e003 in _chosen_e003:
    _src_e003 = _train_root_e003 / f"{{_stem_e003}}.zarr"
    _dst_e003 = _subset_dir_e003 / f"{{_stem_e003}}.zarr"
    if not _dst_e003.exists():
        _dst_e003.symlink_to(_src_e003, target_is_directory=True)

os.environ["BIOHUB_TEST_DIR"] = str(_subset_dir_e003)
print("E003: BIOHUB_TEST_DIR ->", _subset_dir_e003)
# --- END BIOCELL PATCH E003a ---

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = ["""

# ---------------------------------------------------------------------------
# Patch B - save the final predicted graph (post filter_output_graph) as .geff
# ---------------------------------------------------------------------------

ANCHOR_B = """        division_sources: dict[int, int] = {}
        for edge in edges:"""

REPLACEMENT_B = """        # --- BIOCELL PATCH E003b: save the final predicted graph as .geff ---
        # Same (nodes_by_id, edges) the CSV writer below serialises, so this is
        # exactly what a real submission would score - not the raw pre-filter
        # model output.
        import polars as _pl_e003b

        _pred_dir_e003 = Path("/kaggle/working/biocell_pred_geffs")
        _pred_dir_e003.mkdir(parents=True, exist_ok=True)
        _g_e003 = td.graph.IndexedRXGraph()
        for _key_e003 in ("z", "y", "x"):
            _g_e003.add_node_attr_key(_key_e003, dtype=_pl_e003b.Float64, default_value=0.0)
        _id_map_e003: dict[int, int] = {}
        for _node_id_e003 in sorted(nodes_by_id):
            _node_e003 = nodes_by_id[_node_id_e003]
            _id_map_e003[_node_id_e003] = _g_e003.add_node(dict(
                t=int(_node_e003["t"]),
                z=float(_node_e003["z"]),
                y=float(_node_e003["y"]),
                x=float(_node_e003["x"]),
            ))
        _skipped_e003 = 0
        for _edge_e003 in edges:
            _s_e003 = int(_edge_e003["source_id"])
            _t_e003 = int(_edge_e003["target_id"])
            if _s_e003 in _id_map_e003 and _t_e003 in _id_map_e003:
                _g_e003.add_edge(_id_map_e003[_s_e003], _id_map_e003[_t_e003], {})
            else:
                _skipped_e003 += 1
        _g_e003.to_geff(str(_pred_dir_e003 / f"{dataset}.geff"), overwrite=True)
        print(f"  {dataset}: saved prediction geff "
              f"({_g_e003.num_nodes()} nodes, {_g_e003.num_edges()} edges"
              f"{f', {_skipped_e003} dangling edges skipped' if _skipped_e003 else ''})")
        # --- END BIOCELL PATCH E003b ---

        division_sources: dict[int, int] = {}
        for edge in edges:"""

# ---------------------------------------------------------------------------
# Patch C (optional) - disable add_safe_divisions_postlink via its existing
# BIOHUB_OUTPUT_SAFE_DIVISIONS override.
#
# Evidence this heuristic is the entire problem (E005, 2026-08-06): on all 12
# stratified samples, division_like_sources == safe_divisions_added exactly - the
# base linker/ILP produces zero natural forks, every predicted division comes from
# this one late-stage repair step, and it scored 0 TP / 18 FP / 7 FN on real GT.
#
# Reading the function itself explains why: for every node with exactly one
# existing child, it looks for an unmatched orphan detection in the next frame
# within SAFE_DIV_MAX_UM of the source and SAFE_DIV_SISTER_MAX_UM of the existing
# child, and attaches it as a second child - purely geometric proximity, no
# confidence or morphological check. On the largest sample (44b6_7a302da0) it added
# 159 divisions against a SAFE_DIV_GLOBAL_FRAC_CAP-implied ceiling of ~163 - the cap
# is nearly saturated, meaning the heuristic wants to fire even more and is only
# being stopped by an arbitrary global limit, not by evidence of a real division.
# ---------------------------------------------------------------------------

ANCHOR_C = ANCHOR_A  # same insertion point as Patch A - both must run before CONFIG cell


def _env_override_block(env: dict[str, str]) -> str:
    """Emit an os.environ block that runs before the CONFIG cell reads any override.

    The baseline reads every tunable through os.environ.get(...) at CONFIG time, so
    setting them here is equivalent to changing the constants, without touching 138k
    characters of upstream code.
    """
    lines = ["# --- BIOCELL PATCH E00Xc: config overrides via the baseline's own env vars ---"]
    for key, value in env.items():
        lines.append(f'os.environ["{key}"] = "{value}"')
        lines.append(f'print("BIOCELL override: {key} -> {value}")')
    lines.append("# --- END BIOCELL PATCH E00Xc ---")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Patch D (--ilp-divisions) - re-add ILP-proposed divisions after motion relink.
#
# E012 established both halves of the problem:
#   * motion_relink must stay - removing it cost 0.10 node recall and 0.12 edge Jaccard
#   * the ILP genuinely knows where divisions are - it produced the project's first 2
#     true-positive divisions once its output was allowed to survive
# and E005 established that add_safe_divisions_postlink, which picks second children by
# geometric proximity to a leftover detection, has 0% precision.
#
# So: keep relink exactly as-is, then restore a second child ONLY where the ILP proposed
# one. `raw_edges` is still in scope at this point and holds the ILP solution with its
# edge probabilities.
#
# Guard-safe by construction: a candidate is only added when the target currently has no
# parent (keeps max in-degree 1, the constraint E012 violated) and the source currently has
# exactly one child (makes max out-degree 2, never 3).
# ---------------------------------------------------------------------------

ANCHOR_D = """        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")"""

REPLACEMENT_D = """        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
        if not nodes_by_id:
            raise AssertionError(f"{dataset}: post-processing removed every node")

        # --- BIOCELL PATCH E013d: restore ILP-proposed divisions after motion relink ---
        _ilp_children_e013: dict[int, list[tuple[int, float]]] = {}
        for _re_e013 in raw_edges:
            _s_e013 = int(_re_e013["source_id"])
            _t_e013 = int(_re_e013["target_id"])
            _p_e013 = _re_e013.get("edge_prob")
            _ilp_children_e013.setdefault(_s_e013, []).append(
                (_t_e013, 0.0 if _p_e013 is None else float(_p_e013))
            )

        _out_e013: dict[int, int] = {}
        _in_e013: dict[int, int] = {}
        _existing_e013 = set()
        for _e_e013 in edges:
            _s_e013 = int(_e_e013["source_id"])
            _t_e013 = int(_e_e013["target_id"])
            _out_e013[_s_e013] = _out_e013.get(_s_e013, 0) + 1
            _in_e013[_t_e013] = _in_e013.get(_t_e013, 0) + 1
            _existing_e013.add((_s_e013, _t_e013))

        _added_e013 = 0
        _skipped_parent_e013 = 0
        for _src_e013, _kids_e013 in _ilp_children_e013.items():
            if len(_kids_e013) < 2:
                continue                                   # ILP saw no division here
            if _out_e013.get(_src_e013, 0) != 1:
                continue                                   # keep max out-degree at 2
            _src_node_e013 = nodes_by_id.get(_src_e013)
            if _src_node_e013 is None:
                continue
            for _tgt_e013, _prob_e013 in sorted(_kids_e013, key=lambda kv: kv[1], reverse=True):
                if (_src_e013, _tgt_e013) in _existing_e013:
                    continue
                _tgt_node_e013 = nodes_by_id.get(_tgt_e013)
                if _tgt_node_e013 is None:
                    continue
                if int(_tgt_node_e013["t"]) != int(_src_node_e013["t"]) + 1:
                    continue                               # metric only counts t -> t+1
                if _in_e013.get(_tgt_e013, 0) != 0:
                    _skipped_parent_e013 += 1
                    continue                               # never create a merge
                if _prob_e013 < ILP_DIVISION_MIN_PROB:
                    continue
                edges.append({
                    "source_id": _src_e013,
                    "target_id": _tgt_e013,
                    "edge_prob": _prob_e013,
                    "distance_um": edge_distance_um(_src_node_e013, _tgt_node_e013),
                    "ilp_division": 1,
                })
                _out_e013[_src_e013] = _out_e013.get(_src_e013, 0) + 1
                _in_e013[_tgt_e013] = 1
                _existing_e013.add((_src_e013, _tgt_e013))
                _added_e013 += 1
                break                                      # at most one extra child
        filter_stats["ilp_divisions_added"] = _added_e013
        filter_stats["ilp_divisions_skipped_has_parent"] = _skipped_parent_e013
        print(f"  {dataset}: ILP divisions restored={_added_e013} "
              f"(skipped, target already had a parent: {_skipped_parent_e013})")
        # --- END BIOCELL PATCH E013d ---"""

REPLACEMENT_D_CONST = """ILP_DIVISION_MIN_PROB = float(os.environ.get("BIOHUB_ILP_DIVISION_MIN_PROB", "0.0"))
"""


def apply_patches(
    nb: dict,
    samples_per_embryo: int,
    disable_safe_divisions: bool,
    env: dict[str, str] | None = None,
    ilp_divisions: bool = False,
) -> dict:
    replacement_a = REPLACEMENT_A_TEMPLATE.format(samples_per_embryo=samples_per_embryo)

    overrides: dict[str, str] = {}
    if disable_safe_divisions:
        # See docs/METRIC_ANALYSIS.md property 5: this heuristic supplied 100% of the
        # baseline's divisions at 0% precision (E005/E007).
        overrides["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"
    overrides.update(env or {})

    if overrides:
        replacement_a = _env_override_block(overrides) + replacement_a
    if ilp_divisions:
        # the threshold constant must be defined before the output loop reads it
        replacement_a = REPLACEMENT_D_CONST + replacement_a
    patches = [
        ("A: stratified TRAIN subset via BIOHUB_TEST_DIR", ANCHOR_A, replacement_a),
        ("B: save predicted graph as .geff", ANCHOR_B, REPLACEMENT_B),
    ]
    if ilp_divisions:
        patches.append(("D: restore ILP-proposed divisions after relink", ANCHOR_D, REPLACEMENT_D))

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


def build(samples_per_embryo: int, slug: str, title: str, disable_safe_divisions: bool,
          env: dict[str, str] | None = None, ilp_divisions: bool = False) -> Path:
    if slugify(title) != slug:
        raise SystemExit(f"title {title!r} slugifies to {slugify(title)!r}, not {slug!r}")

    if not BASELINE.exists():
        raise SystemExit(f"baseline not found: {BASELINE}")
    nb = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"baseline: {BASELINE.name} ({len(nb['cells'])} cells)")
    nb = apply_patches(nb, samples_per_embryo, disable_safe_divisions, env, ilp_divisions)

    out_dir = REPO / "notebooks" / slug.replace("biohub-", "", 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"{slug}.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]

    # machine_shape + docker_image inherited from the baseline for the same reason as
    # make_submission_notebook.py (see MISTAKES.md M012): enable_gpu alone leaves the
    # accelerator to Kaggle's default, which handed a prior run an incompatible P100.
    baseline_meta = json.loads((BASELINE.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    machine_shape = baseline_meta.get("machine_shape") or "NvidiaTeslaT4"
    docker_image = baseline_meta.get("docker_image", "")
    print(f"  machine_shape: {machine_shape}")
    print(f"  samples_per_embryo: {samples_per_embryo}")
    print(f"  disable_safe_divisions: {disable_safe_divisions}")
    if env:
        print(f"  env overrides: {env}")
    print(f"  ilp_divisions: {ilp_divisions}")

    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "machine_shape": machine_shape,
        # This notebook never submits and only needs the same offline model weights the
        # real submission uses - keep internet off so it measures the actual scoring
        # environment, not a different one.
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
    p.add_argument("--push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--samples-per-embryo", type=int, default=6,
                    help="stratified sample count per embryo (default 6 = ~12 total, "
                         "a deliberately small first pass - see docs/STRATEGY.md)")
    p.add_argument("--slug", default=DEFAULT_SLUG)
    p.add_argument("--title", default=DEFAULT_TITLE)
    p.add_argument("--disable-safe-divisions", action="store_true",
                    help="E007: disable add_safe_divisions_postlink (see docs/MISTAKES.md)")
    p.add_argument("--ilp-divisions", action="store_true",
                    help="E013: re-add ILP-proposed divisions after motion relink")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="override any BIOHUB_* config var the baseline reads from os.environ; "
                         "repeatable, e.g. --env BIOHUB_ILP_DIVISION_WEIGHT=0.4")
    args = p.parse_args()

    env: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key.startswith("BIOHUB_"):
            raise SystemExit(f"--env key {key!r} does not start with BIOHUB_ - likely a typo")
        env[key] = value

    out_dir = build(args.samples_per_embryo, args.slug, args.title,
                    args.disable_safe_divisions, env, args.ilp_divisions)

    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        meta = json.loads((out_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        print(f"pushing {meta['id']} ...")
        api.kernels_push(str(out_dir))
        print(f"https://www.kaggle.com/code/{meta['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
