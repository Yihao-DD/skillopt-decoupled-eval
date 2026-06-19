# Method C — Decoupled Selection · COLD-START HANDOFF

*2026-06-18 · branch `acceleration` · **Read this file alone to resume.** Authoritative entry point for the
decoupled-selection line: the philosophy/method (our contribution), the results, the code, and the EXACT
state + resume commands for the experiments running on the server right now.*

*Companion docs (detail, not required): `PHILOSOPHY-FLAWS-REPORT.md` (deep-research, 25 verified claims +
citations), `SELECTION-FIX-HANDOFF.md` (the diagnosis line this grew out of).*

---

## 0. FIRST THING TO DO (resume the in-flight experiments)

Experiments run on **box2** (`connect.westb.seetacloud.com:20263`). The *greedy training runs are detached
on the server and survive* a session end; the *orchestrators were LOCAL background shells and DIE with the
session* — so you (next agent) must finish the pipeline manually:

1. **Check which greedy finished** (§6 = SSH helper):
   `for t in ssb_selA_v18_s1 .. sq_selA_v18_s1 .. oq_selA_v18_s1 ..; do [ -f SkillOpt/outputs/$t/summary.json ] && echo $t done; done`
2. For each benchmark whose 5 greedy are done but **posthoc not run**, run the 5 posthoc + the 5-seed
   `decoupled_ship` (exact commands in §5).
3. Report per-seed Δ + the **seed-level sign test** (the honest significance test — §3).

LiveMath is already fully done (§3).

---

## 1. THE CONTRIBUTION — philosophy (our "deep thinking")

**Thesis (one sentence, paper-ready):**
> Self-improving LLM skill-optimization is **not a GENERATION problem** (find better edits) but a
> **SELECTION problem** (reliably pick, among edits you already found, under noisy expensive evaluation,
> the one to ship). The greedy accept-gate conflates exploration and selection into one **irreversible
> noisy commitment**; we **decouple** them — explore freely, then commit reliably via best-arm
> identification on a high-power held-out val.

**Why it's philosophy, not a trick:** it *redefines what the problem is*. The field (reflection,
textual-gradient, evolutionary, MCTS) optimizes *how to generate better edits*. Our data **inverts** that:
- **Generation is already good** — the greedy optimizer DOES find a better skill (a trajectory skill is
  +14pt over the initial; oracle headroom exists cross-4-benchmarks × 2-models).
- **Selection wastes it** — the noisy point-estimate gate ships a slow-update-contaminated `best_skill`
  that is **highest-on-val yet worst-on-test** (val-overfit). Generation ≫ Selection.
- Deep-research confirms nobody cleanly beats a *tuned* greedy at equal budget on held-out test — because
  they all attack generation, not selection.

**Three pillars:**
1. **Decouple — selection is a distinct epistemic act.** Greedy fuses explore+select into one greedy/
   irreversible/noisy gate (a category error). Separate *gathering evidence* (the trajectory) from
   *committing* (which skill to ship) = explore-then-commit / best-arm identification.
2. **Irreversible commitment under noise.** Greedy decides irreversibly each step on a noisy point estimate
   (accept = can't undo; reject = skill dies). Principle: under noise, *defer irreversible decisions until
   evidence is sufficient* (optimal stopping / value of information). We defer the commit to the end and
   decide once, at high power.
3. **(Deepest + honest) the binding constraint is EVALUATION COST, not generation cleverness.** Per-edit
   gains (~2–4pt) sit BELOW the selection noise floor (val60 ≈ ±6.3%), so even a correct decouple only
   partially/noisily realizes the headroom. → the true ceiling of test-time self-improvement is *how
   cheaply you can reliably evaluate a candidate*, not *how cleverly you generate one*. Redirects the field
   from "smarter generators" to "cheaper/more-reliable selectors" (an inversion of the bitter lesson).

**Honesty as a philosophical asset:** the literature reports inflated wins (vs human prompts, single-run,
unequal budget). We report **ns, 4/5, McNemar + seed-level sign test**. That rigor + the "bottleneck is
selection, ceiling is evaluation cost" reframe is harder to refute than a fragile big number.

---

## 2. THE METHOD — Method C (explore-then-select)

```
[explore]  greedy training (run_ssb_official.py)  -> saves trajectory skills/skill_v0001..vN (incl. rejected)
[select]   posthoc_select.py  *OUR STEP*          -> eval each trajectory skill on a HIGH-POWER held-out
                                                       val (val60, disjoint from test) -> argmax-val -> ship
[compare]  decoupled_ship.py                       -> greedy(best_skill) vs C(argmax-val), per-seed + sign test
```
- The greedy run does **double duty**: baseline AND our explore phase (same training, same trajectory,
  **equal budget** — only the *ship choice* differs).
- **Why exploration = greedy (not laziness):** proven from code. In strict greedy `current_skill ==
  best_skill` always (`gate.py:177` accepts only `cand>current` → incumbent monotone = rolling best;
  applied at `trainer.py:2526`). So "accept-and-track + rolling-best anchor" is **byte-identical to greedy**
  → relaxing exploration is a NO-OP. Our entire value is the SELECTION step.
- **Rule = argmax-mean-val** (`qd/decoupled_select.py`). Built+tested a Copeland robust-pick: it did NOT
  help (LiveMath argmax +6.5 vs Copeland +4.9) → **argmax is the ship rule (YAGNI)**.
- **Anti-oracle (load-bearing):** selection touches ONLY val; test is read for reporting only (code-review
  verified, no leak).

---

## 3. RESULTS

**LiveMath × DeepSeek — Method C, 5 seeds (DONE):**
| seed | greedy | C | Δ |
|---|---|---|---|
| s1 | 0.4512 | 0.4756 | +2.4 |
| s2 | 0.3902 | 0.4146 | +2.4 |
| s3 | 0.3415 | 0.4878 | **+14.6 (per-seed McNemar p=0.012 SIG)** |
| s4 | 0.5488 | 0.5610 | +1.2 |
| s5 | 0.4024 | 0.3902 | −1.2 |
| **mean** | **0.4268** | **0.4659** | **+3.9** |

→ **4/5 positive, +3.9pt mean. NOT pooled-significant** (pooled-McNemar p=0.117 — anticonservative since the
82 test items are reused across seeds; seed-level sign p=0.1875). **Honest read: directional win, large +
significant on s3, modest elsewhere.** Per the user's bar ("a gain, significant on one/a few seeds, same
single-seed口径 as the paper") this **meets it** (s3 sig). The earlier 3-seed +6.5/p=0.06 was small-sample
optimism (s3 outlier), exposed by the code review's seed-level test.

**Cross-benchmark / cross-model DIAGNOSIS (oracle headroom = gate discards a better skill on test):**
| LiveMath-DS | SearchQA-DS | SSB-DS | Qwen-LiveMath | OfficeQA-DS |
|---|---|---|---|---|
| +3.2/+4.0/−2.4 (2/3) | +2.5 | +5.0 | **+4.0 (cross-model)** | (diag running; candidates ~0.10–0.22 → likely low) |
→ **4/5 cells confirm the gate loses a better skill, across 4 benchmarks + 2 models** — the diagnosis the
literature only *asserts*. (Realizable Method-C gain on these = the §5 in-flight runs.)

---

## 4. CODE BUILT (branch `acceleration`)

| file | purpose | tests |
|---|---|---|
| `qd/decoupled_select.py` | pure selection rules: `argmax_mean`, `copeland_robust`, `select()` | `qd/tests/test_decoupled_select.py` 13 GREEN |
| `tools/posthoc_select.py` | **Method C**: eval trajectory on val+test, argmax-val→test; cross-model `--target-backend qwen_chat` | — |
| `tools/decoupled_ship.py` | offline greedy-vs-C over cached per-item; per-seed + pooled-McNemar + **seed-level sign test** | — |
| `tools/mcnemar_pooled.py` | pooled paired exact McNemar across seeds (reads `test_eval/results.jsonl` OR `results.jsonl`) | — |
| `tools/diag_eval_candidates.py` | oracle-headroom diagnostic; `--target-backend qwen_chat` + `--limit N` (subsample csv/json test) | — |
| `tools/make_experiment_splits.py` | nested-val splits; extended for **CSV** envs (officeqa) | — |

**Deferred cleanups (code review, non-blocking):** shared `qd/stats.py`+`_eval_util.py` (kill McNemar/
load_env/eval-invocation triplication); creds-in-argv; a few edge tests. No CRITICAL issues; anti-oracle clean.

---

## 5. RUNNING ON THE SERVER NOW + how to finish each

**Same split design everywhere:** train35 / val18 (gate) / val60 (C selection) / test82, equal budget, 5
seeds. Tags `<PRE>A_v18_s{1..5}`; splits `data/<PRE>_v18` (gate) + `data/<PRE>_v60` (C val).
**`PRE`/`ENV`:** ssb_sel/spreadsheetbench · sq_sel/searchqa · oq_sel/officeqa · lm_sel/livemathematicianbench.

| benchmark | greedy | state at handoff |
|---|---|---|
| SSB | 5 launched | greedy running (exec-graded ~13s/item, slow) |
| SearchQA | 5 launched | **greedy DONE (5/5)** — needs posthoc + ship |
| OfficeQA | 5 launched | greedy running (24-turn, slowest) |
(local orchestrators `bv3ajsybj` = SSB/SearchQA, `baenoc1bw` = OfficeQA — both die with the session.)

**Greedy run** (`train.train_size=0` REQUIRED for ssb/sq/oq — their configs hardcode 80/400; lm already 0):
```
/root/miniconda3/bin/python repro/official/run_ssb_official.py --env <ENV> --mode full --seed N \
  --key <x2|xx|xnyu> --split-dir data/<PRE>_v18 --tag <PRE>A_v18_sN --cfg-options train.train_size=0
```
**Posthoc (Method C selection) — per seed, once that seed's summary.json exists:**
```
/root/miniconda3/bin/python tools/posthoc_select.py --tag <PRE>A_v18_sN --env <ENV> --key <K> \
  --val-split-dir data/<PRE>_v60 --val-split valid_seen --test-split-dir data/<PRE>_v18 --test-split valid_unseen
```
**5-seed compare (the result table):**
```
SKILLOPT_OUT=/root/skillopt-fullrun-gatesweep/SkillOpt/outputs /root/miniconda3/bin/python \
  tools/decoupled_ship.py --tags <PRE>A_v18_s1,<PRE>A_v18_s2,<PRE>A_v18_s3,<PRE>A_v18_s4,<PRE>A_v18_s5 --rules argmax
```
Launch long jobs **detached** (`setsid nohup … > /root/x.log 2>&1 < /dev/null &`) + poll; never foreground.
⚠️ Check DeepSeek balance after OfficeQA runs (it's the expensive 24-turn one). At handoff: x2 ¥217.86 / xx
¥235.08 / xnyu ¥199.64 = **¥652**.

---

## 6. RUNBOOK (box2)
- `connect.westb.seetacloud.com` port **20263**, root, pw `<redacted>` (⚠️ rotate). Repo
  `/root/skillopt-fullrun-gatesweep`; python `/root/miniconda3/bin/python` (absolute).
- SSH helper (LOCAL from `E:\skillopt`): `AUTODL_PORT=20263 python tools/_autodl_ssh.py '<pw>' exec '<cmd>' [to]`
  / `… put <local> <remote>`. QUIRKS: `put` needs `MSYS_NO_PATHCONV=1`; remote cmd single-quoted locally →
  **no single-quotes inside** (use double / `chr()` for python `-c`); foreground sleep blocked (detach+poll).
- Keys in box2 `.env`: `DEEPSEEK_KEY_{X2,XX,XNYU}` (driver `--key`), `QWEN_KEY` (DashScope, qwen_chat).
- Balance: `curl -s https://api.deepseek.com/user/balance -H "Authorization: Bearer <key>"`.

---

## 7. HONEST CAVEATS
- LiveMath C = +3.9pt, **4/5, NOT pooled-significant** (only s3 individually sig). Pooled-McNemar is
  anticonservative (shared test items) — quote the **seed-level sign test** as the conservative bar.
- `val60` overlaps the gate's `val18` (nested) but **test82 is fully held-out** (anti-oracle intact).
- Realizable gain is **selection-power-limited** (per-edit gains < val noise floor) — that IS the §1.3
  philosophical point, not a bug.
- Exploration relaxation is a proven NO-OP — do NOT rebuild it; width/QD already saturate/fail.
- Diagnosis used some variant runs (SSB-dualev, SearchQA-RG); the §5 NEW 5-seed runs are clean k1 greedy.

---

## 8. NEXT STEPS
1. **Finish §5** — SSB/SearchQA/OfficeQA posthoc + 5-seed `decoupled_ship`; report per-seed Δ + sign test.
   Pool the **seed-level** sign ACROSS benchmarks (different items = real independent power → could reach
   significance + model-general — the prize).
2. Optionally **package** explore+select into one command (`run_ssb_official --selection-mode decoupled`).
3. Write up: §1 thesis + §2 method + §3 results + §7 honesty.
4. **Only after** C is solid: 2nd philosophy lever = **localized credit assignment** (PHILOSOPHY-FLAWS-REPORT
   §2 — reflect on correct+incorrect, edit the relevant skill section, not the whole blob). User: C first.

DO NOT relax anti-oracle. DO NOT rebuild exploration. Beat the *fair* K1 greedy on a *held-out* test; the
lever is SELECTION; the ceiling is EVALUATION COST.
