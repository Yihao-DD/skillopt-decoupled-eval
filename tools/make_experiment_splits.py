#!/usr/bin/env python3
"""Create train/val/test split variants for the Phase-2 SELECTION experiment.

Holds the held-out TEST and TRAIN FIXED, varies ONLY the val (D_sel) size, so an
A/B comparison cleanly isolates the effect of selection-set size on the greedy gate
(the diagnostic showed the 18-item gate discards genuinely-better skills the optimizer
finds). Vals are NESTED (smaller val is a prefix of larger) so A sees a subset of B's val.

Reads SkillOpt/data/<env>_split/{train,val,test}/items.json, pools ALL items, shuffles
with a fixed seed, carves test+train+val_pool, and writes one split dir per val size
under SkillOpt/data/<out_prefix>_v<N>/ (same item dicts, only re-partitioned).

Usage (box2, repo root):
  python tools/make_experiment_splits.py --env livemathematicianbench \
      --test-size 82 --train-size 35 --val-sizes 18 60 --seed 7 --out-prefix lm_sel
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLOPT_DIR = REPO_ROOT / "SkillOpt"


def load_items(split_dir: Path, name: str) -> list:
    p = split_dir / name / "items.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


def write_split(out: Path, train: list, val: list, test: list) -> None:
    for name, items in (("train", train), ("val", val), ("test", test)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def detect_format(src: Path) -> str:
    if (src / "test" / "items.json").is_file():
        return "json"
    if any((src / "test").glob("*.csv")):
        return "csv"
    raise SystemExit(f"[split] {src}/test has neither items.json nor *.csv")


def load_csv_pool(src: Path):
    """Pool data rows from train/val/test CSVs (shared header). Returns (header, rows)."""
    header = None
    rows: list = []
    for name in ("train", "val", "test"):
        cs = sorted((src / name).glob("*.csv"))
        if not cs:
            continue
        with open(cs[0], newline="", encoding="utf-8") as fh:
            r = list(csv.reader(fh))
        if r:
            if header is None:
                header = r[0]
            rows.extend(r[1:])
    return header, rows


def write_csv_split(out: Path, header, train: list, val: list, test: list) -> None:
    for name, data in (("train", train), ("val", val), ("test", test)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if header is not None:
                w.writerow(header)
            w.writerows(data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--test-size", type=int, required=True)
    ap.add_argument("--train-size", type=int, required=True)
    ap.add_argument("--val-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    src = SKILLOPT_DIR / "data" / f"{args.env}_split"
    fmt = detect_format(src)
    if fmt == "csv":
        header, pool = load_csv_pool(src)
    else:
        header = None
        pool = load_items(src, "train") + load_items(src, "val") + load_items(src, "test")
    n = len(pool)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    max_val = max(args.val_sizes)
    need = args.test_size + args.train_size + max_val
    if need > n:
        raise SystemExit(f"[split] need {need} items (test+train+maxval) but pool has only {n}")

    test = pool[: args.test_size]
    train = pool[args.test_size: args.test_size + args.train_size]
    base = args.test_size + args.train_size
    val_pool = pool[base: base + max_val]
    print(f"[split] env={args.env} pool={n} test={len(test)} train={len(train)} "
          f"val_pool={len(val_pool)} seed={args.seed}")
    for vs in sorted(args.val_sizes):
        out = SKILLOPT_DIR / "data" / f"{args.out_prefix}_v{vs}"
        if fmt == "csv":
            write_csv_split(out, header, train, val_pool[:vs], test)
        else:
            write_split(out, train, val_pool[:vs], test)  # nested val (prefix)
        print(f"[split]   {out.name}: train {len(train)} / val {vs} / test {len(test)}")
    print("[split] done — test+train FIXED across variants; val nested (A's val in B's val).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
