"""Wrap a `# %%`-delimited Python script into a Kaggle notebook and push it.

Notebooks live in git as plain scripts so they diff cleanly; Kaggle wants `.ipynb`. This
converts one to the other and pushes via the Kaggle API.

Cell markers follow the jupytext percent format:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (leading '# ' stripped from each line)

Usage:
    python tools/sync_notebook.py notebooks/cv-harness/cv_harness.py \
        --slug biohub-cv-harness --title "Biohub CV Harness" --internet

Pushing creates or updates a **private** notebook. It never submits to the competition.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"


def split_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (kind, body) pairs."""
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def to_notebook(cells: list[tuple[str, str]], gpu: bool = False) -> dict:
    nb_cells = []
    for kind, body in cells:
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#")
                     for ln in body.splitlines()]
            src = "\n".join(lines).strip()
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": src.splitlines(keepends=True)})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": body.splitlines(keepends=True)})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU" if gpu else "None",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def push(script: Path, slug: str, title: str, *, internet: bool, gpu: bool,
         username: str | None = None, kernel_sources: list[str] | None = None,
         dataset_sources: list[str] | None = None,
         competition_sources: list[str] | None = None) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    if username is None:
        username = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]

    nb = to_notebook(split_cells(script.read_text(encoding="utf-8")), gpu=gpu)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{username}/{slug}",
            "title": title,
            "code_file": f"{slug}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true" if gpu else "false",
            "enable_internet": "true" if internet else "false",
            "competition_sources": competition_sources or [COMPETITION],
            "dataset_sources": dataset_sources or [],
            "kernel_sources": kernel_sources or [],
        }, indent=1), encoding="utf-8")

        print(f"pushing {username}/{slug} (gpu={gpu}, internet={internet}) ...")
        api.kernels_push(str(d))
        print(f"https://www.kaggle.com/code/{username}/{slug}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("script", type=Path)
    p.add_argument("--slug", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--internet", action="store_true", help="enable internet (validation only)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--kernel-source", action="append", default=[],
                    help="<owner>/<kernel-slug> to attach as input; repeatable")
    p.add_argument("--dataset-source", action="append", default=[],
                    help="<owner>/<dataset-slug> to attach as input; repeatable")
    p.add_argument("--dry-run", action="store_true", help="write the ipynb locally, do not push")
    args = p.parse_args()

    if args.dry_run:
        nb = to_notebook(split_cells(args.script.read_text(encoding="utf-8")), gpu=args.gpu)
        out = args.script.with_suffix(".ipynb")
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        kinds = [c["cell_type"] for c in nb["cells"]]
        print(f"wrote {out}  ({kinds.count('code')} code, {kinds.count('markdown')} markdown)")
        return 0

    push(args.script, args.slug, args.title, internet=args.internet, gpu=args.gpu,
         kernel_sources=args.kernel_source, dataset_sources=args.dataset_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
