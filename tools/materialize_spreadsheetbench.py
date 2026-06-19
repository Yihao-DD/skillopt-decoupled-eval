"""Materialize SpreadsheetBench (Verified 400) for SkillOpt.

The released ``spreadsheetbench_id_split/`` is ID-only. This tool:
  1. downloads ``KAKA22/SpreadsheetBench`` -> ``spreadsheetbench_verified_400.tar.gz`` (~15 MB),
  2. extracts it to ``<data>/spreadsheetbench_verified_400/`` (the env ``data_root``;
     each task dir ``spreadsheet/<id>/`` holds ``prompt.txt`` + ``*_init.xlsx`` + ``*_golden.xlsx``),
  3. joins the ID manifest with the dataset's ``dataset.json`` metadata and writes the
     runnable split at ``<data>/spreadsheetbench_split/{train,val,test}/items.json``
     (the path the config's ``env.split_dir`` points at).

Idempotent: re-running skips extraction when already present and rewrites the split.

Usage (defaults to the sibling ``SkillOpt/data`` fork dir):
    python tools/materialize_spreadsheetbench.py
    python tools/materialize_spreadsheetbench.py --data-dir E:/skillopt/SkillOpt/data
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "KAKA22/SpreadsheetBench"
TARBALL = "spreadsheetbench_verified_400.tar.gz"
SPLITS = ("train", "val", "test")
# Fields the SpreadsheetBench env (rollout + evaluator) consumes per item.
ITEM_FIELDS = (
    "id",
    "instruction",
    "spreadsheet_path",
    "instruction_type",
    "answer_position",
    "answer_sheet",
    "data_position",
)


def materialize(data_dir: Path) -> dict[str, int]:
    id_split = data_dir / "spreadsheetbench_id_split"
    verified = data_dir / "spreadsheetbench_verified_400"
    split_out = data_dir / "spreadsheetbench_split"

    if not id_split.exists():
        raise FileNotFoundError(f"ID manifest not found: {id_split}")

    # 1-2. download + extract (tar top-level dir == spreadsheetbench_verified_400/)
    tar_path = hf_hub_download(REPO, TARBALL, repo_type="dataset")
    if not (verified / "dataset.json").exists():
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_dir, filter="data")  # safe extraction (no path traversal)
    if not (verified / "dataset.json").exists():
        raise FileNotFoundError(f"extraction did not produce {verified / 'dataset.json'}")

    # 3. join id manifest with full metadata
    meta = json.loads((verified / "dataset.json").read_text(encoding="utf-8"))
    by_id = {str(r["id"]): r for r in meta}

    counts: dict[str, int] = {}
    for split in SPLITS:
        items_in = json.loads((id_split / split / "items.json").read_text(encoding="utf-8"))
        out_items = []
        for it in items_in:
            sid = str(it["id"])
            if sid not in by_id:
                raise KeyError(f"id {sid!r} ({split}) missing from dataset.json")
            rec = by_id[sid]
            task_dir = verified / rec["spreadsheet_path"]
            if not task_dir.is_dir():
                raise FileNotFoundError(f"missing task dir for id {sid!r}: {task_dir}")
            out_items.append({k: rec.get(k, "") for k in ITEM_FIELDS} | {"id": sid})
        out_dir = split_out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "items.json").write_text(
            json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts[split] = len(out_items)
    return counts


def main() -> None:
    default_data = Path(__file__).resolve().parent.parent / "SkillOpt" / "data"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(default_data), help="SkillOpt fork data/ dir")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    counts = materialize(data_dir)
    total = sum(counts.values())
    print(f"materialized SpreadsheetBench: {counts} (total={total})")
    print(f"  split_dir = {data_dir / 'spreadsheetbench_split'}")
    print(f"  data_root = {data_dir / 'spreadsheetbench_verified_400'}")


if __name__ == "__main__":
    main()
