#!/usr/bin/env python3
"""Pooled paired per-item McNemar across seeds for the selection A/B (small-test envs).

Pairs arm-A's shipped best_skill vs arm-B's shipped best_skill PER test item, WITHIN
each seed (identical test set), then POOLS the discordant pairs across all seeds into a
single exact two-sided McNemar over the pooled (n_seeds * n_test) pairs. This is the
right test for "does arm B beat arm A" across seeds — pool discordant pairs, do NOT
average per-seed accuracies (the user's nuance: compare items, not means).

Unlike repro/official/mcnemar_compare.py this does NOT require n>=200 (works for the
82-item LiveMath test) and pools a paired list of tags (one per seed) instead of a
single pair.

Usage (box2, repo root):
  SKILLOPT_OUT=/root/skillopt-fullrun-gatesweep/SkillOpt/outputs \
  /root/miniconda3/bin/python tools/mcnemar_pooled.py \
    --a lm_selA_v18_s1,lm_selA_v18_s2,lm_selA_v18_s3 \
    --b lm_selB_v60_s1,lm_selB_v60_s2,lm_selB_v60_s3 \
    --name-a val18 --name-b val60
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb

DEFAULT_OUT = "/root/skillopt-fullrun-gatesweep/SkillOpt/outputs"


def load(out: str, tag: str) -> dict[str, float]:
    # full driver runs put per-item under <tag>/test_eval/; eval_only --out_root <tag>
    # writes <tag>/results.jsonl directly — accept either.
    cands = [os.path.join(out, tag, "test_eval", "results.jsonl"),
             os.path.join(out, tag, "results.jsonl")]
    p = next((c for c in cands if os.path.exists(c)), None)
    d: dict[str, float] = {}
    if not p:
        return d
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[str(r["id"])] = float(r.get("hard", 0) or 0)
    return d


def exact_mcnemar_p(a_wins: int, b_wins: int) -> float:
    """Two-sided exact (sign-test) McNemar p over the discordant pairs."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="comma list of arm-A tags (one per seed)")
    ap.add_argument("--b", required=True, help="comma list of arm-B tags (one per seed)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    args = ap.parse_args()
    out = os.environ.get("SKILLOPT_OUT", DEFAULT_OUT)
    a_tags = [t.strip() for t in args.a.split(",")]
    b_tags = [t.strip() for t in args.b.split(",")]
    if len(a_tags) != len(b_tags):
        raise SystemExit("[mcnemar] need the same number of A and B tags (paired by seed)")

    pool_a_wins = pool_b_wins = 0
    n_pool = a_correct = b_correct = 0
    na, nb = args.name_a, args.name_b
    print("=== per-seed (paired on the identical test set) ===")
    for i, (ta, tb) in enumerate(zip(a_tags, b_tags), 1):
        da, db = load(out, ta), load(out, tb)
        if not da or not db:
            print(f"  seed{i}: MISSING ({ta}:{bool(da)} {tb}:{bool(db)})")
            continue
        ids = sorted(set(da) & set(db))
        a_wins = sum(1 for x in ids if da[x] > db[x])   # A correct, B wrong
        b_wins = sum(1 for x in ids if db[x] > da[x])   # B correct, A wrong
        sa = sum(da[x] for x in ids)
        sb = sum(db[x] for x in ids)
        p = exact_mcnemar_p(a_wins, b_wins)
        print(f"  seed{i}: n={len(ids):3d}  {na}={sa/len(ids):.4f}  {nb}={sb/len(ids):.4f}  "
              f"d(B-A)={(sb-sa)/len(ids):+.4f}  {na}+only={a_wins} {nb}+only={b_wins}  p={p:.4f}")
        pool_a_wins += a_wins
        pool_b_wins += b_wins
        n_pool += len(ids)
        a_correct += sa
        b_correct += sb

    if n_pool == 0:
        raise SystemExit("[mcnemar] no paired seeds loaded — check tags / SKILLOPT_OUT")
    p = exact_mcnemar_p(pool_a_wins, pool_b_wins)
    sig = "SIG" if p < 0.05 else "ns"
    print(f"=== POOLED ({n_pool} paired items) ===")
    print(f"  {na} acc={a_correct/n_pool:.4f}   {nb} acc={b_correct/n_pool:.4f}   "
          f"d(B-A)={(b_correct-a_correct)/n_pool:+.4f}")
    print(f"  discordant: {na}+only={pool_a_wins}  {nb}+only={pool_b_wins}  "
          f"net(B-A)={pool_b_wins-pool_a_wins}")
    print(f"  exact two-sided McNemar p={p:.4f} [{sig}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
