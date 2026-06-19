#!/usr/bin/env python3
"""Method C — decoupled explore-then-select (post-hoc best-arm selection).

SkillOpt's greedy gate selects step-by-step on a tiny noisy val (D_sel), irreversibly, and
ships the running incumbent (best_skill). C DECOUPLES exploration from selection: keep the
WHOLE explored trajectory (skills/skill_v*.md, including the steps the greedy gate threw
away), then pick the single best skill by a HIGH-POWER held-out val (bigger than the gate's
D_sel, disjoint from test) — best-arm identification — and report its TEST. Anti-oracle:
selection touches only the val, NEVER the test.

For one run it evals every trajectory skill on a VAL split and a TEST split via
scripts/eval_only.py, on the SINGLE model configured in .env (model_config.py — the SAME
model run.py used, so eval is never split across two providers), and reports three TEST nums:
  greedy_shipped : best_skill.md on TEST  (what SkillOpt ships via the greedy gate)
  oracle         : argmax-over-trajectory TEST (the ceiling; selecting on test = oracle)
  C_val_selected : argmax-over-trajectory VAL -> its TEST (realizable; anti-oracle)
C_val_selected > greedy_shipped  =>  decoupled high-power selection beats the greedy gate.

Then tools/decoupled_ship.py applies the inference-aware rules (argmax / siggate / lcb)
across seeds using the per-item results this tool caches (zero new API).

Usage (repo root; the model comes from .env, exactly like run.py):
  python tools/posthoc_select.py --tag run_livemathematicianbench_s1 \
    --env livemathematicianbench \
    --val-split-dir data/lm_sel_v60 --val-split valid_seen \
    --test-split-dir data/lm_sel_v18 --test-split valid_unseen
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLOPT_DIR = REPO_ROOT / "SkillOpt"
ENV_FILE = REPO_ROOT / ".env"
sys.path.insert(0, str(REPO_ROOT))
import model_config  # noqa: E402

_HARD = re.compile(r"hard=([0-9.]+)")


def eval_skill(skill_rel, split, split_dir, env_name, cenv, model, key, endpoint, out_tag) -> float | None:
    # Zero-cost reuse: if this exact eval already ran, read its stored hard score.
    # Makes posthoc idempotent/resumable and the decoupled_ship recompute FREE (no re-eval).
    summ = SKILLOPT_DIR / "outputs" / out_tag / "eval_summary.json"
    if summ.is_file():
        try:
            return float(json.loads(summ.read_text(encoding="utf-8"))["hard"])
        except Exception:  # corrupt/partial -> fall through to a fresh eval
            pass
    cmd = [
        sys.executable, "scripts/eval_only.py",
        "--config", f"configs/{env_name}/default.yaml",
        "--skill", str(skill_rel),
        "--split", split,
        "--split_dir", split_dir,
        "--out_root", f"outputs/{out_tag}",
        # ONE model for both roles (eval-only never calls the optimizer, but we keep it
        # consistent so the config is unambiguous) via the OpenAI-compatible backend.
        "--target_backend", "openai_chat", "--target_model", model,
        "--optimizer_backend", "openai_chat", "--optimizer_model", model,
        "--azure_openai_api_key", key,
        "--azure_openai_endpoint", endpoint,
        "--azure_openai_auth_mode", "openai_compatible",
        "--optimizer_azure_openai_auth_mode", "openai_compatible",
        "--target_azure_openai_auth_mode", "openai_compatible",
        "--reasoning_effort", "",
    ]
    r = subprocess.run(cmd, cwd=str(SKILLOPT_DIR), env=cenv, capture_output=True, text=True)
    m = _HARD.findall(r.stdout)
    if not m:
        print(f"[C]   ! eval FAIL ({out_tag}) rc={r.returncode} {r.stderr[-160:]}", flush=True)
    return float(m[-1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="greedy run whose trajectory we select over")
    ap.add_argument("--env", default="livemathematicianbench")
    ap.add_argument("--val-split-dir", required=True, help="held-out val split dir (high-power, disjoint from test)")
    ap.add_argument("--val-split", default="valid_seen")
    ap.add_argument("--test-split-dir", required=True)
    ap.add_argument("--test-split", default="valid_unseen")
    args = ap.parse_args()

    mc = model_config.resolve(dotenv_path=ENV_FILE)   # the SAME single model run.py used
    key, endpoint, model = mc.api_key, mc.base_url, mc.model
    cenv = dict(os.environ)
    cenv["AZURE_OPENAI_API_KEY"] = key
    cenv["AZURE_OPENAI_ENDPOINT"] = endpoint
    cenv["AZURE_OPENAI_AUTH_MODE"] = "openai_compatible"
    cenv["TARGET_TEMPERATURE"] = "0"   # frozen target -> reproducible eval

    run = SKILLOPT_DIR / "outputs" / args.tag
    traj = sorted((run / "skills").glob("skill_v*.md"))  # full explored trajectory (incl. seed v0000)
    best = run / "best_skill.md"
    if not traj:
        sys.exit(f"[C] no skill_v*.md under {run}/skills")
    to_eval = traj + ([best] if best.is_file() else [])

    print(f"[C] tag={args.tag} env={args.env} model={model} (provider={mc.provider})", flush=True)
    print(f"[C] val={args.val_split}@{args.val_split_dir}  test={args.test_split}@{args.test_split_dir}", flush=True)
    val: dict[str, float | None] = {}
    test: dict[str, float | None] = {}
    for sk in to_eval:
        name = sk.stem
        rel = sk.relative_to(SKILLOPT_DIR)
        v = eval_skill(rel, args.val_split, args.val_split_dir, args.env, cenv, model, key, endpoint,
                       f"posthoc_{args.tag}_{name}_val")
        t = eval_skill(rel, args.test_split, args.test_split_dir, args.env, cenv, model, key, endpoint,
                       f"posthoc_{args.tag}_{name}_test")
        val[name], test[name] = v, t
        print(f"[C]   {name:14s} val={'NA' if v is None else f'{v:.4f}'}  test={'NA' if t is None else f'{t:.4f}'}", flush=True)

    # C selects over the optimizer's EDITS (skill_v0001..vN); greedy ships best_skill.
    # Exclude the hand-written seed skill_v0000 — shipping it would not show the gate "lost"
    # an optimizer-found skill.
    cand = {s.stem: (val[s.stem], test[s.stem]) for s in traj
            if s.stem != "skill_v0000"
            and val.get(s.stem) is not None and test.get(s.stem) is not None}
    shipped = test.get("best_skill")
    print("=== Method C result ===", flush=True)
    if shipped is not None:
        print(f"[C] greedy_shipped(best_skill)  test={shipped:.4f}")
    if not cand:
        print("[C] no scored trajectory skills — cannot select")
        return 0
    oracle_skill = max(cand, key=lambda k: cand[k][1])  # argmax TEST (oracle ceiling)
    c_skill = max(cand, key=lambda k: cand[k][0])        # argmax VAL  (anti-oracle realizable)
    print(f"[C] oracle (argmax-test)        {oracle_skill}: test={cand[oracle_skill][1]:.4f}")
    print(f"[C] C (argmax-val -> its test)  {c_skill}: val={cand[c_skill][0]:.4f} test={cand[c_skill][1]:.4f}")
    if shipped is not None:
        d_c = cand[c_skill][1] - shipped
        d_o = cand[oracle_skill][1] - shipped
        print(f"[C] VERDICT  dTest(C - greedy)={d_c:+.4f}   (oracle headroom={d_o:+.4f})")
        print("[C] " + ("C BEATS greedy on held-out test -> decoupled high-power selection works (anti-oracle)"
                        if d_c > 0 else
                        "C does NOT beat greedy this seed -> decouple gain not realized here"))
        print(f"[C] PICKED c_skill={c_skill}  shipped_skill=best_skill  (for paired McNemar)")

        # Deployable SAFE variant: C+incumbent. argmax-val over edits ∪ {best_skill}
        # (still val-only / anti-oracle), ties broken toward the incumbent. Keeping the
        # incumbent as a candidate makes C never worse than greedy except via a real
        # val/test discordance among edits that out-val it.
        cand_inc = dict(cand)
        if val.get("best_skill") is not None and test.get("best_skill") is not None:
            cand_inc["best_skill"] = (val["best_skill"], test["best_skill"])
        c_inc = max(cand_inc, key=lambda k: (cand_inc[k][0], k == "best_skill"))
        d_c_inc = cand_inc[c_inc][1] - shipped
        print(f"[C] C+inc (argmax-val incl incumbent) {c_inc}: "
              f"val={cand_inc[c_inc][0]:.4f} test={cand_inc[c_inc][1]:.4f}")
        print(f"[C] VERDICT+inc dTest(C+inc - greedy)={d_c_inc:+.4f}  (safe selector, keeps incumbent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
