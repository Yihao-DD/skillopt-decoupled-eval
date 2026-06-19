#!/usr/bin/env python3
"""Materialize the LiveMathematicianBench split for SkillOpt from HuggingFace.

The released ``livemathematicianbench_id_split/`` is ID-only
(``{train,val,test}/items.json`` each a list of ``{"id": "<month>:<no>", ...}``).
This tool mirrors ``materialize_searchqa.py``:

  1. downloads the 4 raw ``qa_*_final.json`` source files from HF
     ``LiveMathematicianBench/LiveMathematicianBench`` (pinned revision).
     Respects ``HF_ENDPOINT`` — in CN set ``HF_ENDPOINT=https://hf-mirror.com``;
  2. normalizes them with the env's OWN ``dataloader.load_items`` (so the schema —
     ``id`` = ``"<month>:<no>"``, ``question`` / ``choices`` / ``correct_choice`` /
     ``theorem`` / ... — is exactly what the rollout + evaluator consume);
  3. joins the ID manifest and writes the runnable split at
     ``livemathematicianbench_split/{train,val,test}/items.json`` (the path the
     config's ``env.split_dir`` points at).

Idempotent: re-running re-downloads only missing raw files and rewrites the split.

Usage (CN):
    HF_ENDPOINT=https://hf-mirror.com python tools/materialize_livemathematicianbench.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = "LiveMathematicianBench/LiveMathematicianBench"
REVISION = "b72450f6ce96c26158d64d945a5d31ef7727be41"
SOURCE_FILES = (
    "data/202511/qa_202511_final.json",
    "data/202512/qa_202512_final.json",
    "data/202601/qa_202601_final.json",
    "data/202602/qa_202602_final.json",
)
SPLITS = ("train", "val", "test")
EXPECTED = {"train": 35, "val": 18, "test": 124}  # from id_split manifest counts


def download_raw(raw_dir: Path) -> Path:
    """Download the monthly qa_*_final.json source files into ``raw_dir`` (flat)."""
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    for rel in SOURCE_FILES:
        cached = hf_hub_download(REPO, rel, repo_type="dataset", revision=REVISION)
        dst = raw_dir / Path(rel).name
        if not dst.exists() or dst.stat().st_size != Path(cached).stat().st_size:
            shutil.copy(cached, dst)
        print(f"  raw: {dst.name} ({dst.stat().st_size} bytes)")
    return raw_dir


def materialize(data_dir: Path, repo_root: Path) -> dict[str, int]:
    id_split = data_dir / "livemathematicianbench_id_split"
    raw_dir = data_dir / "livemathematicianbench_raw"
    split_out = data_dir / "livemathematicianbench_split"

    if not id_split.exists():
        raise FileNotFoundError(f"ID manifest not found: {id_split}")

    download_raw(raw_dir)

    # Reuse the env's own normalizer so items match rollout/evaluator exactly
    # (id == "<month>:<no>", question, choices, correct_choice, ...).
    sys.path.insert(0, str(repo_root / "SkillOpt"))
    from skillopt.envs.livemathematicianbench.dataloader import load_items

    items = load_items(str(raw_dir))
    by_id = {str(it["id"]): it for it in items}
    print(f"  normalized {len(items)} source items")

    counts: dict[str, int] = {}
    for split in SPLITS:
        manifest = json.loads(
            (id_split / split / "items.json").read_text(encoding="utf-8")
        )
        ids = [str(it["id"]) for it in manifest]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(
                f"split={split}: {len(missing)} manifest ids missing from source; "
                f"first few: {missing[:5]}"
            )
        out_items = [by_id[i] for i in ids]
        if len(out_items) != EXPECTED[split]:
            print(
                f"  WARN split={split}: got {len(out_items)}, "
                f"manifest counts expect {EXPECTED[split]}"
            )
        out_dir = split_out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "items.json").write_text(
            json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts[split] = len(out_items)
        print(f"  {split}: wrote {len(out_items)} items -> {out_dir / 'items.json'}")
    return counts


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_data = repo_root / "SkillOpt" / "data"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(default_data), help="SkillOpt fork data/ dir")
    args = ap.parse_args()

    counts = materialize(Path(args.data_dir), repo_root)
    print(f"materialized LiveMathematicianBench: {counts} (total={sum(counts.values())})")
    print(f"  split_dir = {Path(args.data_dir) / 'livemathematicianbench_split'}")


if __name__ == "__main__":
    main()
