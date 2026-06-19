#!/usr/bin/env python3
"""Paired per-item McNemar over SkillOpt 280-test results.jsonl.

Usage (on the server, where outputs/ live):
    python mcnemar_compare.py LABEL=tag LABEL2=tag2 ... --pairs A:B,C:D

Each tag's per-item file is the best-skill 280-test results.jsonl (fields id, hard).
We pair by id, count B = first-wins (first=1,second=0), C = second-wins, and report
the two-sided EXACT binomial McNemar p = min(1, 2*Σ_{i<=min(B,C)} C(n,i) / 2^n).
"""
from __future__ import annotations
import json, glob, os, sys
from math import comb

OUT = os.environ.get("SKILLOPT_OUT", "/root/autodl-tmp/skillopt-fullrun-gatesweep/SkillOpt/outputs")


def find_results(tag: str) -> str | None:
    cands = [f"{OUT}/{tag}/test_eval/results.jsonl"]
    cands += sorted(glob.glob(f"{OUT}/{tag}/test_eval/sel/*/results.jsonl"))
    cands += sorted(glob.glob(f"{OUT}/{tag}/test_eval/**/results.jsonl", recursive=True))
    for p in cands:
        if os.path.exists(p):
            n = sum(1 for _ in open(p))
            if n >= 200:
                return p
    return None


def load(tag: str) -> dict[str, float]:
    p = find_results(tag)
    if not p:
        return {}
    d: dict[str, float] = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d[str(r["id"])] = float(r.get("hard", 0) or 0)
    return d


def mcnemar(a: dict, b: dict):
    ids = sorted(set(a) & set(b))
    B = sum(1 for i in ids if a[i] > b[i])
    C = sum(1 for i in ids if b[i] > a[i])
    n = B + C
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(min(B, C) + 1)) / (2 ** n)) if n else 1.0
    return len(ids), B, C, p


def main():
    labels: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for arg in sys.argv[1:]:
        if arg.startswith("--pairs"):
            spec = arg.split("=", 1)[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1]
            for pr in spec.split(","):
                a, b = pr.split(":")
                pairs.append((a.strip(), b.strip()))
        elif "=" in arg:
            k, v = arg.split("=", 1)
            labels[k.strip()] = v.strip()
    data = {k: load(t) for k, t in labels.items()}
    print("=== per-arm 280-test accuracy ===")
    for k, t in labels.items():
        d = data[k]
        if d:
            print(f"  {k:12} {t:24} n={len(d):3}  test_hard={sum(d.values())/len(d):.4f}")
        else:
            print(f"  {k:12} {t:24} MISSING results.jsonl")
    print("=== paired exact McNemar ===")
    for a, b in pairs:
        if not data.get(a) or not data.get(b):
            print(f"  {a} vs {b}: MISSING")
            continue
        n, B, C, p = mcnemar(data[a], data[b])
        acc_a = sum(data[a][i] for i in (set(data[a]) & set(data[b]))) / n
        acc_b = sum(data[b][i] for i in (set(data[a]) & set(data[b]))) / n
        sig = "SIG" if p < 0.05 else "ns"
        print(f"  {a:10} vs {b:10}: n={n} acc {acc_a:.4f} vs {acc_b:.4f} "
              f"Δ={acc_a-acc_b:+.4f} B({a}+)={B} C({b}+)={C} p={p:.4f} [{sig}]")


if __name__ == "__main__":
    main()
