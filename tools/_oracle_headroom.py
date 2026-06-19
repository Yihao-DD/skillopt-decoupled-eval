#!/usr/bin/env python3
"""LOCAL-ONLY diagnostic (delete after): per-benchmark CAPTURABLE-HEADROOM map.

Reads cached posthoc per-item test/val (results.jsonl) and reports, per seed and pooled:
  greedy = best_skill test-mean              (what the gate ships)
  C      = test-mean of the argmax-VAL skill (Method C, anti-oracle, realizable)
  oracle = MAX test-mean over trajectory skills (UPPER BOUND; peeks at TEST = DIAGNOSTIC ONLY,
           never shippable -- same role as posthoc_select.py's printed oracle ceiling)
  gaps:  C-greedy (realized) ; oracle-greedy (total headroom) ; oracle-C (UNREALIZED capturable)

oracle-C answers the gain-hunt routing question: how much MORE could a perfect selector capture
than Method C does now? Large => eval-side SNR levers (control variates / allocation) have room.
~0 => capture is already maxed and the only remaining lever is generation-side headroom.
Excludes skill_v0000 (the hand-written seed), matching decoupled_ship / posthoc_select.
"""
from __future__ import annotations

import glob
import json
import os
import sys

OUT = os.environ.get("SKILLOPT_OUT") or sys.exit("set SKILLOPT_OUT")


def mean_hard(d: str):
    p = os.path.join(d, "results.jsonl")
    if not os.path.exists(p):
        return None
    xs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") is not None:
                xs.append(float(r.get("hard", 0) or 0))
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    tags = [t.strip() for t in sys.argv[1].split(",")]
    shipped = sys.argv[2] if len(sys.argv) > 2 else "best_skill"
    gd, cd, od = [], [], []
    for tag in tags:
        skills = {}
        for vd in sorted(glob.glob(os.path.join(OUT, f"posthoc_{tag}_skill_v*_val"))):
            name = os.path.basename(vd)[len(f"posthoc_{tag}_"):-len("_val")]
            if name == "skill_v0000":
                continue
            v = mean_hard(vd)
            t = mean_hard(os.path.join(OUT, f"posthoc_{tag}_{name}_test"))
            if v is not None and t is not None:
                skills[name] = (v, t)
        g = mean_hard(os.path.join(OUT, f"posthoc_{tag}_{shipped}_test"))
        if not skills or g is None:
            print(f"{tag}: MISSING (skills={len(skills)} greedy={g is not None})")
            continue
        c_skill = max(skills, key=lambda k: (skills[k][0], k))  # argmax VAL (anti-oracle)
        o_skill = max(skills, key=lambda k: skills[k][1])        # argmax TEST (oracle ceiling)
        c, o = skills[c_skill][1], skills[o_skill][1]
        gd.append(g); cd.append(c); od.append(o)
        print(f"{tag}: greedy={g:.4f}  C(val->{c_skill})={c:.4f} d={c - g:+.4f}  "
              f"oracle(test->{o_skill})={o:.4f} head={o - g:+.4f}  UNREALIZED(oracle-C)={o - c:+.4f}")
    n = len(gd)
    if n:
        mg, mc, mo = sum(gd) / n, sum(cd) / n, sum(od) / n
        print(f"=== MEAN/{n}seeds: greedy={mg:.4f}  C d={mc - mg:+.4f}  oracle head={mo - mg:+.4f}  "
              f"UNREALIZED capturable (oracle-C)={mo - mc:+.4f}")
    return 0


raise SystemExit(main())
