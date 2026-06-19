#!/usr/bin/env python3
"""One-command training driver — runs vanilla SkillOpt (greedy) on a benchmark with a
SINGLE model doing EVERYTHING (optimizer reflection + target rollouts), read from .env.

By DEFAULT it runs THREE seeds (1, 2, 3) for the one (model, benchmark) — the experiment
unit — producing one greedy trajectory per seed. Then score them with the decoupled
selectors: tools/posthoc_select.py per seed + tools/decoupled_ship.py across the 3 tags
(the NEXT step printed at the end does exactly this).

    python run.py --env livemathematicianbench                 # seeds 1 2 3 (default)
    python run.py --env spreadsheetbench --seeds 1 2 3 4 5      # custom seeds
    python run.py --env officeqa --seeds 1                      # a single seed
    python run.py --env searchqa --smoke                        # tiny 1-seed auth/data/executor check
    python run.py --env officeqa --dry                          # print the commands, spend nothing

The model is .env's MODEL_PROVIDER / MODEL_API_KEY / MODEL_NAME (see model_config.py).
Optimizer AND target both use that one model, so a run is NEVER split across two providers.

Output per seed: SkillOpt/outputs/<tag>_s<seed>/{skills/, best_skill.md, summary.json, driver.log}.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import model_config

REPO_ROOT = Path(__file__).resolve().parent
SKILLOPT_DIR = REPO_ROOT / "SkillOpt"
ENV_FILE = REPO_ROOT / ".env"
ENVS = ["livemathematicianbench", "officeqa", "searchqa", "spreadsheetbench"]
DEFAULT_SEEDS = [1, 2, 3]


def build_train_args(env: str, model: str, out_root: str, seed: int, epochs: int,
                     smoke: bool, workers: int | None) -> list[str]:
    # One model for both roles via the OpenAI-compatible backend.
    common = [
        "--config", f"configs/{env}/default.yaml",
        "--optimizer_backend", "openai_chat",
        "--target_backend", "openai_chat",
        "--optimizer_model", model,
        "--target_model", model,
        "--reasoning_effort", "",
        "--out_root", out_root,
        "--split_dir", f"data/{env}_split",
    ]
    if smoke:
        args = common + [
            "--num_epochs", "1",
            "--use_slow_update", "false", "--use_meta_skill", "false",
            "--workers", str(workers or 4),
        ]
    else:
        # train_size / batch_size / max_turns come from configs/<env>/default.yaml (faithful).
        args = common + [
            "--num_epochs", str(epochs), "--seed", str(seed),
            "--use_slow_update", "true", "--use_meta_skill", "true",
        ]
        if workers:
            args += ["--workers", str(workers)]
    if env == "spreadsheetbench":
        # SSB references task xlsx under a separate data_root; the other envs have none.
        args += ["--data_root", "data/spreadsheetbench_verified_400"]
    return args


def run_one_seed(cfg, env: str, tag: str, seed: int, epochs: int, smoke: bool,
                 workers: int | None, dry: bool) -> int:
    out_root = f"outputs/{tag}"
    train_args = build_train_args(env, cfg.model, out_root, seed, epochs, smoke, workers)
    cmd = [sys.executable, "scripts/train.py", *train_args]

    # Child env: ONE model for both roles via the OpenAI-compatible backend.
    child = os.environ.copy()
    child["AZURE_OPENAI_ENDPOINT"] = cfg.base_url
    child["AZURE_OPENAI_API_KEY"] = cfg.api_key
    child["AZURE_OPENAI_AUTH_MODE"] = "openai_compatible"
    # Role-aware decoding: frozen target (temp 0 + seed → reproducible eval); diverse
    # optimizer (temp 0.8 → varied reflections). SAME model, just different sampling.
    child.setdefault("TARGET_TEMPERATURE", "0")
    child.setdefault("TARGET_SEED", "42")
    child.setdefault("OPTIMIZER_TEMPERATURE", "0.8")
    child["OPTIMIZER_SEED"] = str(seed)
    child["PYTHONIOENCODING"] = "utf-8"            # train.py prints unicode to the pipe
    child["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + child.get("PYTHONPATH", "")

    print("-" * 72)
    print(f"[run] seed={seed}  tag={tag}  -> SkillOpt/{out_root}")
    print("[run] command: python " + " ".join(repr(a) if a == "" else a for a in cmd[1:]))
    if dry:
        print("[run] --dry: not executing.")
        return 0

    out_dir = SKILLOPT_DIR / out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "driver.log", "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(SKILLOPT_DIR), env=child,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            logf.write(line); logf.flush()
        rc = proc.wait()
    print(f"[run] seed {seed} exited rc={rc}. Summary: SkillOpt/{out_root}/summary.json")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=ENVS, required=True, help="Which benchmark to run.")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="Seeds to run, one greedy trajectory each (default: 1 2 3).")
    ap.add_argument("--tag", default="", help="Tag prefix; each run -> <tag>_s<seed> (default run_<env>_s<seed>).")
    ap.add_argument("--epochs", type=int, default=4, help="Training epochs per seed (default 4).")
    ap.add_argument("--workers", type=int, default=None, help="Rollout concurrency (default: the benchmark config).")
    ap.add_argument("--smoke", action="store_true", help="Tiny 1-epoch, SINGLE-seed auth/data/executor check.")
    ap.add_argument("--dry", action="store_true", help="Print the per-seed commands and exit; spend nothing.")
    args = ap.parse_args()

    if not SKILLOPT_DIR.is_dir():
        sys.exit(f"[run] SkillOpt/ not found at {SKILLOPT_DIR}")

    cfg = model_config.resolve(dotenv_path=ENV_FILE)
    seeds = [args.seeds[0]] if args.smoke else list(args.seeds)   # smoke = one quick seed

    def tag_for(seed: int) -> str:
        if args.tag:
            return f"{args.tag}_s{seed}"
        return f"{'smoke' if args.smoke else 'run'}_{args.env}_s{seed}"

    print("=" * 72)
    print(f"[run] {args.env}: {len(seeds)} seed(s) {seeds}   MODEL={cfg.model} "
          f"(provider={cfg.provider}, key={cfg.masked_key()})")
    print(f"[run]   base_url={cfg.base_url}  — optimizer AND target both use this one model.")
    print("=" * 72)

    results: dict[str, int] = {}
    for seed in seeds:
        tag = tag_for(seed)
        results[tag] = run_one_seed(cfg, args.env, tag, seed, args.epochs, args.smoke, args.workers, args.dry)

    print("=" * 72)
    print(f"[run] DONE — {len(seeds)} seed(s):")
    for tag, rc in results.items():
        print(f"    {tag}: rc={rc}  (SkillOpt/outputs/{tag}/summary.json)")
    if not args.dry and not args.smoke:
        tags = ",".join(results)
        seedstr = " ".join(str(s) for s in seeds)
        prefix = args.tag or f"run_{args.env}"
        print("[run] NEXT — decoupled selection (the 3 optimizers) across the seeds:")
        print(f"  for s in {seedstr}; do python tools/posthoc_select.py --tag {prefix}_s$s --env {args.env} \\")
        print(f"    --val-split-dir data/{args.env}_split --val-split val \\")
        print(f"    --test-split-dir data/{args.env}_split --test-split test; done")
        print(f"  SKILLOPT_OUT=SkillOpt/outputs python tools/decoupled_ship.py --tags {tags} --rules argmax,siggate,lcb@1")
    return max(results.values()) if results else 0


if __name__ == "__main__":
    raise SystemExit(main())
