#!/usr/bin/env python3
"""One command to materialize ALL benchmark data the pipeline runs on.

    python tools/materialize_all.py                       # all 4 benchmarks
    python tools/materialize_all.py --only livemathematicianbench searchqa
    HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=hf_xxx python tools/materialize_all.py

Each benchmark's data is downloaded from HuggingFace into SkillOpt/data/<env>_split via the
per-benchmark materializer. After this, `python run.py --env <benchmark>` runs with NO
further setup.

Notes:
  - officeqa needs a HuggingFace token (env HF_TOKEN or --token).
  - Inside mainland China, set HF_ENDPOINT=https://hf-mirror.com.
  - One benchmark failing does not stop the others; the summary lists what succeeded.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS = ["livemathematicianbench", "officeqa", "searchqa", "spreadsheetbench"]


def run_materializer(env: str, token: str) -> tuple[bool, str]:
    script = REPO_ROOT / "tools" / f"materialize_{env}.py"
    if not script.is_file():
        return False, f"missing {script.name}"
    cmd = [sys.executable, str(script)]
    if env == "officeqa" and token:
        cmd += ["--token", token]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return r.returncode == 0, f"rc={r.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=ENVS, default=ENVS, help="Subset of benchmarks (default: all 4).")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF token for officeqa (or env HF_TOKEN).")
    args = ap.parse_args()

    if "officeqa" in args.only and not args.token:
        print("[materialize] NOTE: officeqa needs a HuggingFace token (HF_TOKEN or --token); "
              "it will fail without one.", file=sys.stderr)

    results: dict[str, tuple[bool, str]] = {}
    for env in args.only:
        print(f"\n===== materializing {env} =====", flush=True)
        results[env] = run_materializer(env, args.token)

    print("\n===== summary =====")
    for env, (ok, info) in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {env}  ({info})")
    failed = [e for e, (ok, _) in results.items() if not ok]
    if failed:
        print(f"\n[materialize] {len(failed)} failed: {failed}. See messages above "
              "(common cause: missing HF_TOKEN for officeqa, or no HF mirror inside CN).")
        return 1
    print("\n[materialize] all done -> `python run.py --env <benchmark>` is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
