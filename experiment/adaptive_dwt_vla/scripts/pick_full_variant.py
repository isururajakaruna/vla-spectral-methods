"""Print the variant to train on full LIBERO: dwt_fixed_slw if its pooled 3-seed success at step 2000 on the training
task beats plain dwt_fixed, otherwise dwt_fixed (ties and missing seeds go to the plain loss). Reason goes to stderr."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pooled(rows, v):
    pat = re.compile(rf"^{v}@lc_s(\d+)_{v}:step_02000$")
    hits = [r for r in rows if r["task"] == "libero_object#0" and pat.match(r["model"])]
    return sum(r["successes"] for r in hits), sum(r["episodes"] for r in hits), len(hits)


def main() -> None:
    rows = json.loads((ROOT / "results" / "eval.json").read_text())["rows"]
    ks, ns, ss = pooled(rows, "dwt_fixed_slw")
    kp, np_, sp = pooled(rows, "dwt_fixed")
    pick = "dwt_fixed_slw" if ss == 3 and sp == 3 and ks * np_ > kp * ns else "dwt_fixed"
    print(f"soft+weighted {ks}/{ns} ({ss} seeds) vs plain {kp}/{np_} ({sp} seeds) at step 2000 -> {pick}", file=sys.stderr)
    print(pick)


if __name__ == "__main__":
    main()
