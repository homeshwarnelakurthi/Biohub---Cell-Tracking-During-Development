"""Wait for a Kaggle kernel to finish, then verify and submit it - detached.

Session-bound background tasks in this project have died with exit 4 twice when
the session ended, each time leaving a finished Kaggle run sitting unsubmitted
for hours. This is meant to be launched detached (nohup / Start-Process) so the
wait-verify-submit chain completes regardless.

It reuses tools/submit.py for the actual submission, so the same gate applies:
the kernel must report COMPLETE, its output must contain submission.csv, and that
file must pass structural validation. A failed kernel is never submitted.

Usage (detached):
    nohup python tools/autosubmit.py --kernel <owner>/<slug> --message "..." \\
        > scratchpad/autosubmit.log 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def kernel_status(api, kernel: str) -> str:
    return str(getattr(api.kernels_status(kernel), "status", ""))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, help="<owner>/<slug>")
    p.add_argument("--message", required=True, help="submission description")
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--max-wait", type=int, default=6 * 60 * 60, help="give up after N seconds")
    args = p.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    waited = 0
    last = None
    while waited < args.max_wait:
        try:
            status = kernel_status(api, args.kernel)
        except Exception as exc:  # noqa: BLE001 - transient API errors are expected
            print(f"[{waited//60}m] poll error: {type(exc).__name__}", flush=True)
            time.sleep(args.poll)
            waited += args.poll
            continue

        if status != last:
            print(f"[{waited//60}m] {status}", flush=True)
            last = status

        if "RUNNING" not in status and "QUEUED" not in status:
            if "COMPLETE" not in status:
                print(f"kernel finished as {status} - NOT submitting.", flush=True)
                return 1
            break

        time.sleep(args.poll)
        waited += args.poll
    else:
        print(f"gave up after {args.max_wait//3600}h still running - NOT submitting.", flush=True)
        return 1

    print("kernel COMPLETE - handing off to submit.py (which re-verifies)", flush=True)
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "submit.py"),
         "--kernel", args.kernel, "--message", args.message],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(result.stdout, flush=True)
    if result.stderr.strip():
        print("STDERR:", result.stderr[-2000:], flush=True)
    print(f"submit.py exit={result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
