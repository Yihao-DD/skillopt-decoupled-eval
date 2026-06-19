#!/usr/bin/env python3
"""One command for the whole evaluation — serial over all benchmarks, each:
    greedy (3 seeds)  ->  posthoc per seed  ->  decoupled_ship (the 3 optimizers)  ->  verdict
and a final per-benchmark verdict table. ALL on the ONE model in .env.

LiveMath / SearchQA / SpreadsheetBench data is bundled in the repo (offline-ready).
OfficeQA's documents are ~788M and are NOT bundled — it is materialized at runtime from
HuggingFace using HF_TOKEN in .env (skipped automatically if HF_TOKEN is unset).

    python run_all.py                                   # all benchmarks, serial
    python run_all.py --only livemathematicianbench searchqa
    python run_all.py --epochs 4 --workers 32
    python run_all.py --skip-officeqa                   # the 3 bundled ones only

Benchmarks run independently: if one fails (e.g. an executor dependency missing), the others
still run and the failure is reported in the final table.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import model_config

REPO = Path(__file__).resolve().parent
PY = sys.executable
ENV_FILE = REPO / ".env"
# bundled (no network) first; OfficeQA last (needs HF).
BENCHES = ["livemathematicianbench", "searchqa", "spreadsheetbench", "officeqa"]
SEEDS = [1, 2, 3]


def sh(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO), env=env)


def load_hf_into_env() -> None:
    """Put HF_TOKEN / HF_ENDPOINT from .env into os.environ so materialize_officeqa sees them."""
    dot = model_config.load_dotenv(ENV_FILE)
    for k in ("HF_TOKEN", "HF_ENDPOINT", "HUGGINGFACE_HUB_TOKEN"):
        if dot.get(k) and not os.environ.get(k):
            os.environ[k] = dot[k]


def materialize_officeqa() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    if not token:
        print("[run_all] OfficeQA needs HF_TOKEN in .env — skipping it.", flush=True)
        return False
    print("[run_all] materializing OfficeQA from HuggingFace (~788M docs, one-time)...", flush=True)
    return sh([PY, "tools/materialize_officeqa.py", "--token", token]).returncode == 0


def run_benchmark(env_name: str, epochs: int, workers: int | None) -> tuple[str, str]:
    """Greedy 3 seeds -> posthoc per seed -> decoupled_ship. Returns (status, verdict_text)."""
    if env_name == "officeqa" and not materialize_officeqa():
        return "SKIPPED", "OfficeQA not materialized (HF_TOKEN unset or download failed)."

    # 1) greedy, 3 seeds
    cmd = [PY, "run.py", "--env", env_name, "--seeds", *(str(s) for s in SEEDS), "--epochs", str(epochs)]
    if workers:
        cmd += ["--workers", str(workers)]
    if sh(cmd).returncode != 0:
        return "FAILED", "greedy run.py returned non-zero (see log above)."
    tags = [f"run_{env_name}_s{s}" for s in SEEDS]

    # 2) posthoc per seed (uses the bundled standard split's val/test)
    for tag in tags:
        sh([PY, "tools/posthoc_select.py", "--tag", tag, "--env", env_name,
            "--val-split-dir", f"data/{env_name}_split", "--val-split", "val",
            "--test-split-dir", f"data/{env_name}_split", "--test-split", "test"])

    # 3) decoupled_ship across the 3 seeds -> the 3-optimizer verdict (capture it)
    env2 = dict(os.environ)
    env2["SKILLOPT_OUT"] = str(REPO / "SkillOpt" / "outputs")
    r = subprocess.run(
        [PY, "tools/decoupled_ship.py", "--tags", ",".join(tags), "--rules", "argmax,siggate,lcb@1"],
        cwd=str(REPO), env=env2, capture_output=True, text=True,
    )
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        return "PARTIAL", "decoupled_ship failed:\n" + (r.stderr or "")[-400:]
    # keep the POOLED 3-optimizer lines for the final table
    keep = [ln for ln in r.stdout.splitlines()
            if ln.strip().startswith(("argmax", "siggate", "lcb")) or "POOLED" in ln]
    return "OK", "\n".join(keep) if keep else r.stdout[-500:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=BENCHES, default=None, help="Subset of benchmarks (default: all).")
    ap.add_argument("--skip-officeqa", action="store_true", help="Run only the 3 bundled benchmarks.")
    ap.add_argument("--epochs", type=int, default=4, help="Epochs per seed (default 4).")
    ap.add_argument("--workers", type=int, default=None, help="Rollout concurrency (default: each benchmark config).")
    args = ap.parse_args()

    if not (REPO / "SkillOpt").is_dir():
        sys.exit("[run_all] SkillOpt/ not found — run from the repo root.")

    cfg = model_config.resolve(dotenv_path=ENV_FILE)   # fail fast if .env model is unset
    load_hf_into_env()
    benches = args.only or [b for b in BENCHES if not (args.skip_officeqa and b == "officeqa")]

    print("=" * 72)
    print(f"[run_all] MODEL = {cfg.model} (provider={cfg.provider})  — one model for everything")
    print(f"[run_all] benchmarks (serial): {benches}   seeds={SEEDS}  epochs={args.epochs}")
    print("=" * 72)

    results: dict[str, tuple[str, str]] = {}
    for env_name in benches:
        print("\n" + "#" * 72 + f"\n#  BENCHMARK: {env_name}\n" + "#" * 72)
        try:
            results[env_name] = run_benchmark(env_name, args.epochs, args.workers)
        except Exception as e:  # noqa: BLE001 — one benchmark must not kill the rest
            results[env_name] = ("ERROR", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    print("[run_all] FINAL — 3-optimizer (argmax / siggate / lcb) verdict per benchmark")
    print("=" * 72)
    for env_name, (status, verdict) in results.items():
        print(f"\n### {env_name}  [{status}]")
        print(verdict)
    print("\n[run_all] (argmax=Method-C the gain engine; siggate/lcb=inference-aware never-lose. "
          "d_seedmean>0 means the selector beat greedy averaged over the 3 seeds.)")
    return 0 if all(s in ("OK", "SKIPPED") for s, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
