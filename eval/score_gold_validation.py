"""
eval/score_gold_validation.py — accuracy from the gold validation pass.

Reads eval/gold_validation.csv (verdicts filled in by gold_validation_app.py)
and reports, overall and per hazard stratum:

  accuracy     correct / judged
  useful_rate  (correct + partial) / judged   # mirrors Groundsource's "useful" split
  counts       correct / partial / wrong / unjudged

Only in_queue == Y rows count. Writes eval/gold_validation_scores.csv.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "eval" / "gold_validation.csv"
DEFAULT_OUT = REPO_ROOT / "eval" / "gold_validation_scores.csv"

VERDICTS = ("correct", "partial", "wrong")


def _summary(rows: list[dict]) -> dict:
    c = {v: 0 for v in VERDICTS}
    unjudged = 0
    for r in rows:
        v = (r.get("verdict") or "").strip().lower()
        if v in c:
            c[v] += 1
        else:
            unjudged += 1
    judged = sum(c.values())
    return {
        "judged": judged,
        "correct": c["correct"],
        "partial": c["partial"],
        "wrong": c["wrong"],
        "unjudged": unjudged,
        "accuracy": round(c["correct"] / judged, 3) if judged else "",
        "useful_rate": round((c["correct"] + c["partial"]) / judged, 3) if judged else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.inp.exists():
        ap.error(f"Validation CSV not found: {args.inp}")

    rows = [r for r in csv.DictReader(args.inp.open(encoding="utf-8"))
            if str(r.get("in_queue", "")).strip().upper() == "Y"]
    if not rows:
        ap.error("No in_queue == Y rows to score.")

    overall = {"stratum": "ALL", **_summary(rows)}

    by_stratum: dict[str, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r.get("stratum", "(unknown)") or "(unknown)", []).append(r)
    strata = [{"stratum": s, **_summary(g)} for s, g in by_stratum.items()]
    strata.sort(key=lambda x: x["judged"], reverse=True)

    fields = ["stratum", "judged", "correct", "partial", "wrong", "unjudged", "accuracy", "useful_rate"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(overall)
        for s in strata:
            w.writerow(s)

    print(f"Wrote {args.out}\n")
    print(f"{'stratum':<22}{'judged':>7}{'acc':>7}{'useful':>8}  (✓/~/✗/?)")
    def _line(d):
        return (f"{d['stratum']:<22}{d['judged']:>7}{str(d['accuracy']):>7}{str(d['useful_rate']):>8}"
                f"  {d['correct']}/{d['partial']}/{d['wrong']}/{d['unjudged']}")
    print(_line(overall))
    print("-" * 60)
    for s in strata:
        print(_line(s))
    if overall["unjudged"]:
        print(f"\nNote: {overall['unjudged']} queued events still unjudged.")


if __name__ == "__main__":
    main()
