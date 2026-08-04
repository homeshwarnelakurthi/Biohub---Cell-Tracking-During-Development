"""Verify a completed Kaggle notebook, then submit it to the code competition.

Submissions are capped at 5/day and the public leaderboard is only 29% of the test set, so
a wasted slot is expensive and a silently-degenerate submission is worse than none. This
refuses to submit unless the run completed and its `submission.csv` passes structural
validation.

Usage:
    python tools/submit.py --kernel homeshwarrao/biohub-submission-v1 --check
    python tools/submit.py --kernel homeshwarrao/biohub-submission-v1 --message "E004 ..."
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COMPETITION = "biohub-cell-tracking-during-development"
SUBMISSION_FILE = "submission.csv"


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def kernel_status(api, kernel: str) -> str:
    return str(getattr(api.kernels_status(kernel), "status", ""))


def fetch_output(api, kernel: str, dest: Path) -> Path | None:
    """Download the kernel's output; return the path to submission.csv if present."""
    dest.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, path=str(dest))
    hits = list(dest.rglob(SUBMISSION_FILE))
    return hits[0] if hits else None


def summarise_log(dest: Path) -> None:
    """Print the patch's own diagnostics from the run log, if we can find them."""
    logs = list(dest.rglob("*.log"))
    if not logs:
        return
    try:
        entries = json.loads(logs[0].read_text(encoding="utf-8"))
        text = "\n".join(e.get("data", "") for e in entries)
    except Exception:
        return
    interesting = [ln for ln in text.splitlines()
                   if "capped out-degree" in ln or "Wrote submission" in ln
                   or "total nodes" in ln.lower() or "total edges" in ln.lower()]
    if interesting:
        print("\nrun diagnostics:")
        for ln in interesting[:20]:
            print("  " + ln.strip())
    else:
        print("\nrun diagnostics: no out-degree-cap lines found "
              "(cap may not have fired on this test set)")


def verify(csv_path: Path) -> list[str]:
    from biocell.submission import validate_submission
    problems = validate_submission(csv_path)
    size = csv_path.stat().st_size
    with csv_path.open(encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1
    print(f"\n{SUBMISSION_FILE}: {size:,} bytes, {rows:,} data rows")
    if rows <= 0:
        problems.append("submission has no data rows")
    return problems


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, help="<owner>/<notebook>")
    p.add_argument("--message", default="", help="submission description")
    p.add_argument("--check", action="store_true", help="verify only, never submit")
    p.add_argument("--keep", action="store_true", help="keep the downloaded output")
    p.add_argument("--max-version", type=int, default=8,
                   help="highest kernel version to try when submitting")
    args = p.parse_args()

    api = _api()

    status = kernel_status(api, args.kernel)
    print(f"kernel : {args.kernel}\nstatus : {status}")
    if "COMPLETE" not in status:
        print("\nNot submitting: the run has not completed successfully.")
        return 1

    dest = Path(tempfile.mkdtemp(prefix="biocell_sub_"))
    try:
        print(f"downloading output -> {dest}")
        csv_path = fetch_output(api, args.kernel, dest)
        if csv_path is None:
            print(f"\nNot submitting: no {SUBMISSION_FILE} in the kernel output.")
            return 1

        summarise_log(dest)
        problems = verify(csv_path)
        if problems:
            print("\nNot submitting - validation problems:")
            for prob in problems:
                print(f"  - {prob}")
            return 1
        print("validation: OK")

        if args.check:
            print("\n--check given; not submitting.")
            return 0

        if not args.message:
            print("\nRefusing to submit without --message.")
            return 1

        print(f"\nsubmitting to {COMPETITION} ...")
        # kernel_version is typed Optional, but omitting it makes CreateCodeSubmission
        # return a bare 403 with no explanation - it is effectively required. Kaggle does
        # not expose the version number through kernels_status or the pulled metadata, so
        # walk upward from 1 until one is accepted.
        last_error: Exception | None = None
        for version in range(1, args.max_version + 1):
            try:
                resp = api.competition_submit_code(
                    file_name=SUBMISSION_FILE,
                    message=args.message,
                    competition=COMPETITION,
                    kernel=args.kernel,
                    kernel_version=version,
                )
            except Exception as exc:  # noqa: BLE001 - reported below if all versions fail
                last_error = exc
                print(f"  version {version}: rejected ({type(exc).__name__})")
                continue
            print(f"  version {version}: accepted")
            print("response:", getattr(resp, "message", resp) or "(empty)")
            return 0

        print(f"\nAll versions 1..{args.max_version} rejected. Last error:\n  {last_error}")
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
