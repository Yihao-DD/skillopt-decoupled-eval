#!/usr/bin/env python3
"""Materialize the SearchQA split for SkillOpt from HuggingFace.

官方仓库只发布 ID 清单（data/searchqa_id_split/{train,val,test}/items.json，
每个是 [{"id": <32位 key>}, ...]）。SkillOpt 的 SearchQA env 需要全字段 items
（id/question/context/answers）放在 data/searchqa_split/{train,val,test}/items.json。
本脚本用 key==id 关联 HF `lucadiliello/searchqa` 并写出完整 split。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

MANIFEST_COUNTS = {"train": 400, "val": 200, "test": 1400}
SPLITS = ("train", "val", "test")


def load_manifest_ids(id_split_dir: str, split: str) -> list[str]:
    path = os.path.join(id_split_dir, split, "items.json")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [str(it["id"]) for it in items]


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(a) for a in value]


def build_split(manifest_ids: list[str], key_to_row: dict[str, dict]) -> list[dict]:
    """Join manifest IDs to HF rows. Raises KeyError if any ID is missing."""
    missing = [i for i in manifest_ids if i not in key_to_row]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest IDs not found in source dataset; "
            f"first few: {missing[:5]}"
        )
    out: list[dict] = []
    for i in manifest_ids:
        row = key_to_row[i]
        out.append({
            "id": i,
            "question": row["question"],
            "context": row.get("context", ""),
            "answers": coerce_answers(row.get("answers")),
        })
    return out


def validate_counts(split: str, items: list[dict]) -> None:
    expected = MANIFEST_COUNTS[split]
    if len(items) != expected:
        raise ValueError(f"split={split}: expected {expected} items, got {len(items)}")


def write_split(out_dir: str, split: str, items: list[dict]) -> str:
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    out_path = os.path.join(split_dir, "items.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return out_path


def load_key_to_row() -> dict[str, dict]:
    """Load HF lucadiliello/searchqa (train+validation) into {key: row}."""
    from datasets import load_dataset

    key_to_row: dict[str, dict] = {}
    for hf_split in ("train", "validation"):
        ds = load_dataset("lucadiliello/searchqa", split=hf_split)
        for row in ds:
            key_to_row[str(row["key"])] = row
    return key_to_row


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize SearchQA split from HF.")
    ap.add_argument("--id-split-dir", default="SkillOpt/data/searchqa_id_split")
    ap.add_argument("--out-dir", default="SkillOpt/data/searchqa_split")
    ap.add_argument("--force", action="store_true", help="rebuild even if output exists")
    args = ap.parse_args()

    if not args.force and all(
        os.path.exists(os.path.join(args.out_dir, s, "items.json")) for s in SPLITS
    ):
        print(f"{args.out_dir} already complete; use --force to rebuild.")
        return

    print("Loading HF lucadiliello/searchqa (train+validation)...")
    key_to_row = load_key_to_row()
    print(f"  loaded {len(key_to_row)} source rows")

    for split in SPLITS:
        ids = load_manifest_ids(args.id_split_dir, split)
        items = build_split(ids, key_to_row)
        validate_counts(split, items)
        out_path = write_split(args.out_dir, split, items)
        print(f"  {split}: wrote {len(items)} items -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
