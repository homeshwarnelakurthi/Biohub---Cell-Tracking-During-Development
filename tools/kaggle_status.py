"""Pull live competition standing from the Kaggle API.

Deliberately avoids pandas: the Anaconda base environment on this machine has a
numpy/pandas ABI mismatch (see docs/MISTAKES.md M005), and this tool must keep working
regardless. Uses the stdlib csv module instead.

Usage:
    python tools/kaggle_status.py            # standing + submissions + distribution
    python tools/kaggle_status.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
PRIZE_RANKS = (1, 2, 3, 4, 5, 6, 7)


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_leaderboard(api) -> list[dict]:
    with tempfile.TemporaryDirectory() as td:
        api.competition_leaderboard_download(COMPETITION, path=td)
        zips = list(Path(td).glob("*.zip"))
        if not zips:
            raise RuntimeError("leaderboard download produced no archive")
        with zipfile.ZipFile(zips[0]) as z:
            raw = z.read(z.namelist()[0]).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))
    rows.sort(key=lambda r: float(r["Score"]), reverse=True)
    return rows


def my_username() -> str:
    env = os.environ.get("KAGGLE_USERNAME")
    if env:
        return env
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(cfg.read_text())["username"]


def collect() -> dict:
    api = _api()
    rows = fetch_leaderboard(api)
    user = my_username()

    mine = None
    for i, r in enumerate(rows, start=1):
        if user.lower() in (r.get("TeamMemberUserNames") or "").lower():
            mine = {
                "rank": i,
                "team": r["TeamName"],
                "score": float(r["Score"]),
                "submissions": int(r["SubmissionCount"]),
                "last_submission": r["LastSubmissionDate"],
            }
            break

    out: dict = {
        "competition": COMPETITION,
        "teams": len(rows),
        "top_score": float(rows[0]["Score"]),
        "prize_cutoff": float(rows[PRIZE_RANKS[-1] - 1]["Score"]) if len(rows) >= 7 else None,
        "me": mine,
    }

    if mine:
        out["gap_to_first"] = round(out["top_score"] - mine["score"], 4)
        if out["prize_cutoff"] is not None:
            out["gap_to_prize"] = round(out["prize_cutoff"] - mine["score"], 4)
        tied = [i for i, r in enumerate(rows, 1)
                if abs(float(r["Score"]) - mine["score"]) < 5e-4]
        out["tied_at_my_score"] = {
            "count": len(tied), "rank_from": min(tied), "rank_to": max(tied),
        }

    out["distribution"] = {
        str(r): float(rows[r - 1]["Score"])
        for r in (1, 3, 7, 10, 25, 50, 100, 200, 500, 1000)
        if r <= len(rows)
    }

    try:
        resp = api.competition_submissions(COMPETITION)
        subs = resp.submissions if hasattr(resp, "submissions") else resp
        out["my_submissions"] = [
            {
                "date": str(s.date),
                "file": getattr(s, "file_name", ""),
                "public_score": getattr(s, "public_score", None),
                "status": str(getattr(s, "status", "")),
            }
            for s in subs
        ]
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        out["my_submissions_error"] = f"{type(exc).__name__}: {exc}"

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args()

    data = collect()
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    me = data.get("me")
    print(f"Competition : {data['competition']}")
    print(f"Teams       : {data['teams']}")
    print(f"Top score   : {data['top_score']}")
    print(f"Prize (7th) : {data['prize_cutoff']}")
    if me:
        print(f"\nUs          : {me['team']}  rank {me['rank']}  score {me['score']}"
              f"  ({me['submissions']} subs, last {me['last_submission']})")
        print(f"Gap to 1st  : +{data['gap_to_first']}")
        print(f"Gap to prize: +{data.get('gap_to_prize')}")
        t = data["tied_at_my_score"]
        print(f"Tied with   : {t['count']} teams at our exact score "
              f"(ranks {t['rank_from']}-{t['rank_to']})")
    else:
        print("\nNo leaderboard row found for this Kaggle user.")

    print("\nScore by rank:")
    for rank, score in data["distribution"].items():
        print(f"  #{rank:>5}  {score:.4f}")

    if data.get("my_submissions"):
        print("\nOur submissions:")
        for s in data["my_submissions"]:
            print(f"  {s['date']}  {s['public_score']}  {s['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
