#!/usr/bin/env python3
"""Materialize the OfficeQA split for SkillOpt from HuggingFace (databricks/officeqa).

The released ``officeqa_id_split/`` is ID-only ({train,val,test}/items.json, each a
list of {id, uid, category, source_files, source_docs, ...} — NO question/answer
and NO document corpus). The official QA CSV + the parsed Treasury-Bulletin corpus
are GATED on HF, so this tool needs an authorized token (env ``HF_TOKEN``).

  1. snapshot_download ``databricks/officeqa`` (token, mirror-aware), pulling only
     ``officeqa_full.csv`` + ``treasury_bulletins_parsed/{jsons,transformed}`` —
     the multi-GB raw PDFs are skipped;
  2. place the corpus at ``data/officeqa_docs_official/{transformed,jsons}`` (the
     path the env's ``resolve_docs_roots`` + ``_locate_parsed_json`` consume);
  3. join the ID manifest (uid) against ``officeqa_full.csv`` and write the runnable
     CSV split at ``data/officeqa_split/{train,val,test}/<split>.csv`` (the path the
     config's ``env.split_dir`` points at; the env runs ``search_mode=offline`` so
     NO external search API is needed).

Idempotent: re-running re-downloads only missing files and rewrites corpus + splits.

Usage (CN):
    HF_TOKEN=hf_xxx HF_ENDPOINT=https://hf-mirror.com python tools/materialize_officeqa.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

REPO = "databricks/officeqa"
REVISION = "8ecbf18d3833daf4750a903d14963e4c4c1d4cd8"
QA_CSV = "officeqa_full.csv"
CORPUS_PREFIX = "treasury_bulletins_parsed"
ALLOW = [QA_CSV, f"{CORPUS_PREFIX}/jsons/*", f"{CORPUS_PREFIX}/transformed/*"]
SPLITS = ("train", "val", "test")
EXPECTED = {"train": 50, "val": 24, "test": 172}  # from id_split manifest counts


def collect_needed_files(id_split: Path) -> set[str]:
    """Basenames of the source_files referenced by the train/val/test items."""
    needed: set[str] = set()
    for split in SPLITS:
        items = json.loads((id_split / split / "items.json").read_text(encoding="utf-8"))
        for it in items:
            raw = it.get("source_files")
            parts = raw if isinstance(raw, list) else str(raw or "").replace("\n", ",").split(",")
            for p in parts:
                name = str(p).strip()
                if name:
                    needed.add(name)
    return needed


def download_raw(raw_dir: Path, token: str | None, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=allow_patterns,
        local_dir=str(raw_dir),
        token=token,
        max_workers=4,
        etag_timeout=30,
    )
    return raw_dir


def place_corpus(raw_dir: Path, docs_dir: Path) -> dict[str, int]:
    """Move parsed corpus to data/officeqa_docs_official/{transformed,jsons}."""
    src = raw_dir / CORPUS_PREFIX
    counts: dict[str, int] = {}
    for sub in ("transformed", "jsons"):
        src_sub = src / sub
        dst_sub = docs_dir / sub
        dst_sub.mkdir(parents=True, exist_ok=True)
        n = 0
        if src_sub.is_dir():
            for f in src_sub.iterdir():
                if not f.is_file():
                    continue
                dst = dst_sub / f.name
                if not dst.exists() or dst.stat().st_size != f.stat().st_size:
                    shutil.copy2(f, dst)
                n += 1
        counts[sub] = n
    return counts


def write_splits(id_split: Path, qa_csv: Path, split_out: Path) -> dict[str, int]:
    with qa_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        by_uid = {str(row.get("uid") or "").strip(): row for row in reader}
    print(f"  loaded {len(by_uid)} QA rows from {qa_csv.name} (cols={fieldnames})")

    counts: dict[str, int] = {}
    for split in SPLITS:
        manifest = json.loads((id_split / split / "items.json").read_text(encoding="utf-8"))
        uids = [str(it.get("uid") or it.get("id") or "").strip() for it in manifest]
        missing = [u for u in uids if u not in by_uid]
        if missing:
            raise KeyError(f"split={split}: {len(missing)} uids missing from QA CSV; first: {missing[:5]}")
        out_dir = split_out / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{split}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for u in uids:
                writer.writerow(by_uid[u])
        counts[split] = len(uids)
        flag = "" if counts[split] == EXPECTED[split] else f"  (WARN expected {EXPECTED[split]})"
        print(f"  {split}: wrote {counts[split]} rows -> {out_csv}{flag}")
    return counts


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "SkillOpt" / "data"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(data_dir), help="SkillOpt fork data/ dir")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF token (or env HF_TOKEN)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    id_split = data_dir / "officeqa_id_split"
    if not id_split.exists():
        raise FileNotFoundError(f"ID manifest not found: {id_split}")
    token = (args.token or "").strip() or None

    needed = collect_needed_files(id_split)
    stems = {Path(n).stem for n in needed}
    allow = [QA_CSV]
    allow += [f"{CORPUS_PREFIX}/transformed/{n}" for n in sorted(needed)]
    allow += [f"{CORPUS_PREFIX}/jsons/{s}.json" for s in sorted(stems)]
    raw_dir = data_dir / "officeqa_raw"
    print(f"  downloading {REPO}: QA + {len(needed)} referenced docs (transformed+jsons) -> {raw_dir}")
    download_raw(raw_dir, token, allow)

    docs_dir = data_dir / "officeqa_docs_official"
    corpus_counts = place_corpus(raw_dir, docs_dir)
    print(f"  corpus placed: {corpus_counts} -> {docs_dir}")

    counts = write_splits(id_split, raw_dir / QA_CSV, data_dir / "officeqa_split")
    print(f"materialized OfficeQA: {counts} (total={sum(counts.values())})")
    print(f"  split_dir = {data_dir / 'officeqa_split'}")
    print(f"  docs_dir  = {docs_dir}  (transformed={corpus_counts.get('transformed')}, jsons={corpus_counts.get('jsons')})")


if __name__ == "__main__":
    main()
