#!/usr/bin/env python3
"""Offline decoupled ship (Method C): apply selection rules over a run's trajectory using
CACHED per-item val/test results (zero new API) and report the shipped TEST per rule vs the
greedy ship, per seed + pooled McNemar.

Reads outputs/posthoc_<tag>_<skill>_{val,test}/results.jsonl (written by posthoc_select.py).
Selection uses qd.decoupled_select {argmax, copeland}. Anti-oracle: rules select on the VAL;
we only REPORT the picked skill's TEST. Greedy ship = <shipped> (default best_skill).

Usage (box2, repo root):
  SKILLOPT_OUT=/root/skillopt-fullrun-gatesweep/SkillOpt/outputs \
  /root/miniconda3/bin/python tools/decoupled_ship.py \
    --tags lm_selA_v18_s1,lm_selA_v18_s2,lm_selA_v18_s3 --rules argmax,copeland
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qd.decoupled_select import INCUMBENT, select  # noqa: E402

OUT = os.environ.get("SKILLOPT_OUT", str(REPO / "SkillOpt" / "outputs"))


def load_items(d: str) -> dict[str, float] | None:
    p = os.path.join(d, "results.jsonl")
    if not os.path.exists(p):
        return None
    m: dict[str, float] = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id")
            if rid is not None:
                m[str(rid)] = float(r.get("hard", 0) or 0)
    return m or None


def exact_mcnemar(a: dict, b: dict):
    ids = sorted(set(a) & set(b))
    nb = sum(1 for i in ids if a[i] > b[i])  # a(greedy)-only
    nc = sum(1 for i in ids if b[i] > a[i])  # b(rule)-only
    n = nb + nc
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(min(nb, nc) + 1)) / (2 ** n)) if n else 1.0
    return len(ids), nb, nc, p


def parse_rule(spec: str) -> tuple[str, str, dict]:
    """'siggate@0.10' -> ('siggate@0.10', 'siggate', {'alpha': 0.10}); 'lcb@2' -> (.., 'lcb', {'z': 2.0}).

    Bare 'argmax'/'copeland'/'siggate'/'lcb' -> ({}). The optional @param tunes the inference-aware
    rules (siggate: alpha; lcb: z) so a whole sweep runs in one invocation. The full spec is the label.
    """
    base, _, param = spec.partition("@")
    kw: dict = {}
    if param:
        if base == "siggate":
            kw["alpha"] = float(param)
        elif base == "lcb":
            kw["z"] = float(param)
        else:
            raise SystemExit(f"[ship] rule {base!r} takes no @param (got {spec!r})")
    return spec, base, kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma list of run tags (one per seed)")
    ap.add_argument("--shipped", default="best_skill", help="the greedy ship skill stem")
    ap.add_argument("--rules", default="argmax,copeland,siggate,lcb",
                    help="comma list; inference-aware rules take an optional @param "
                         "(siggate@<alpha>, lcb@<z>), e.g. argmax,siggate@0.05,siggate@0.10,lcb@1,lcb@2")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",")]
    specs = [parse_rule(s.strip()) for s in args.rules.split(",")]  # (label, base, kwargs)
    labels = [lbl for lbl, _, _ in specs]
    needs_inc = any(base in ("siggate", "lcb") for _, base, _ in specs)

    pooled: dict[str, dict[str, float]] = {"greedy": {}, **{lbl: {} for lbl in labels}}
    seed_wins: dict[str, list[int]] = {lbl: [] for lbl in labels}   # per-seed 1 if rule>greedy (sign test)
    seed_d: dict[str, list[float]] = {lbl: [] for lbl in labels}    # per-seed held-out test delta
    abstain: dict[str, int] = {lbl: 0 for lbl in labels}            # seeds the rule shipped the incumbent
    print(f"[ship] out={OUT}")
    for si, tag in enumerate(tags):
        pool_val: dict[str, dict[str, float]] = {}
        test_of: dict[str, dict[str, float]] = {}
        for vd in sorted(glob.glob(os.path.join(OUT, f"posthoc_{tag}_skill_v*_val"))):
            name = os.path.basename(vd)[len(f"posthoc_{tag}_"):-len("_val")]
            if name == "skill_v0000":  # exclude the hand-written seed; C selects over the optimizer's EDITS
                continue
            vi = load_items(vd)
            ti = load_items(os.path.join(OUT, f"posthoc_{tag}_{name}_test"))
            if vi and ti:
                pool_val[name], test_of[name] = vi, ti
        gtest = load_items(os.path.join(OUT, f"posthoc_{tag}_{args.shipped}_test"))
        gval = load_items(os.path.join(OUT, f"posthoc_{tag}_{args.shipped}_val"))  # incumbent val (sig ref)
        if not pool_val or gtest is None:
            print(f"[ship] {tag}: MISSING cached results (pool={len(pool_val)} greedy={gtest is not None})")
            continue
        if needs_inc and gval is None:
            print(f"[ship] {tag}: WARN no cached {args.shipped}_val -> siggate/lcb abstain (ship greedy)")
        gm_seed = sum(gtest.values()) / len(gtest)
        line = f"[ship] {tag}: greedy_full={gm_seed:.4f}"
        for lbl, base, kw in specs:
            inc = gval if base in ("siggate", "lcb") else None
            if base in ("siggate", "lcb") and inc is None:
                pick = INCUMBENT  # no incumbent val cached -> cannot run the test -> abstain
            else:
                pick = select(pool_val, rule=base, incumbent=inc, **kw)
            if pick == INCUMBENT:
                ptest, shown = gtest, "ABSTAIN->greedy"
                abstain[lbl] += 1
            else:
                ptest, shown = test_of[pick], pick
            ids = sorted(set(ptest) & set(gtest))  # compare on the shared items only
            pmean = sum(ptest[i] for i in ids) / len(ids)
            gmean = sum(gtest[i] for i in ids) / len(ids)
            d = pmean - gmean
            line += f" | {lbl}->{shown} test={pmean:.4f} (d={d:+.4f})"
            seed_wins[lbl].append(1 if d > 0 else 0)
            seed_d[lbl].append(d)
            for k, v in ptest.items():
                pooled[lbl][f"{si}:{k}"] = v
        for k, v in gtest.items():
            pooled["greedy"][f"{si}:{k}"] = v
        print(line)

    g = pooled["greedy"]
    if not g:
        print("[ship] nothing pooled")
        return 1
    gm = sum(g.values()) / len(g)
    n_seeds = len(seed_wins[labels[0]]) if labels else 0
    print(f"=== POOLED ({len(g)} items across {n_seeds} seeds) ===  greedy acc={gm:.4f}")
    print("  [note] pooled-item McNemar is ANTICONSERVATIVE (test items reused across seeds, pairs not")
    print("         independent). seed-level sign test treats seeds as the unit; noLose counts d>=0 seeds")
    print("         (siggate/lcb abstain to greedy => d=0 => never a strict loss).")
    for lbl in labels:
        rr = pooled[lbl]
        rm = sum(rr.values()) / len(rr)
        _, nb, nc, p = exact_mcnemar(g, rr)
        wins = sum(seed_wins[lbl])
        nolose = sum(1 for d in seed_d[lbl] if d >= -1e-9)
        sp = (sum(comb(n_seeds, i) for i in range(wins, n_seeds + 1)) / (2 ** n_seeds)) if n_seeds else 1.0
        mean_d = sum(seed_d[lbl]) / len(seed_d[lbl]) if seed_d[lbl] else 0.0
        sig = "SIG" if p < 0.05 else "ns"
        ssig = "SIG" if sp < 0.05 else "ns"
        print(f"  {lbl:12s} acc={rm:.4f} d_pool={rm - gm:+.4f} d_seedmean={mean_d:+.4f}  "
              f"discord {nb}:{nc} net={nc - nb} pooledMcN p={p:.4f}[{sig}]  "
              f"seed {wins}/{n_seeds}>g sign_p={sp:.4f}[{ssig}] noLose {nolose}/{n_seeds} abstain {abstain[lbl]}/{n_seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
