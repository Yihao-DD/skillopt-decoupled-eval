# Qwen Cross-Model Validation — RESULT (2026-06-19, DONE)

*Branch `acceleration`. The 3-seed Qwen-target (qwen-plus) LiveMath run that addressed the
single-model caveat on the Method-C / decoupled-selection line. Optimizer=DeepSeek, target=Qwen
(**partial** cross-model: DeepSeek authors the skill edits, Qwen executes + drives the gate/selection).
Setup doc: `QWEN-CROSSMODEL-COLDSTART.md`. Raw log: box2 `/root/lmqwenC_verdict.log`. Tags
`lmqwenC_s{1,2,3}`. This is distinct from the root `RESULT_QWEN_CROSSMODEL.md` (older RR-Boost run).*

---

## 0. Verdict (one line) — NOT a failure
**The thesis replicates cross-model and is STRENGTHENED; Method-C's realized gain is shown to be
SNR-gated, not a reliable number.** The greedy gate provably discards a better skill on Qwen
(oracle headroom mean **+6.10**, **3/3 seeds**, *bigger* than DeepSeek's +5.1). Decoupled selection
*can* capture it cross-model (s2 **+6.10** = existence proof) but back-fires when val is noisy
(s1 −3.66, s3 −9.76); 3-seed argmax mean **−2.44**, 1/3. Robust play = **never-lose abstention**
(siggate 0.00 3/3; lcb@1 & deployable C+inc +0.81).

## 1. Setup
3 seeds, LiveMath, **target=qwen-plus** (qwen_chat / DashScope), **optimizer=deepseek-chat**, 4 epochs,
split `data/lm_sel_v18`, posthoc on val60 + test82. Partial cross-model (skills authored by DeepSeek;
executed + gate-selected by Qwen — so the *selection pathology* is Qwen-driven, the *skill generation*
is not). The fuller "Qwen-authors-too" (optimizer=Qwen) run was parked by the user.

## 2. SkillOpt base works strongly on Qwen (sanity)
| seed | no-skill baseline | greedy ship (test) | self-opt gain |
|---|---|---|---|
| s1 | 0.232 | 0.488 | +25.6 |
| s2 | 0.256 | 0.415 | +15.9 |
| s3 | 0.171 | 0.402 | +23.2 |

## 3. The three added components vs greedy (3-seed)
| seed | greedy | **argmax (Method-C)** | siggate (α=0.05) | lcb@1 |
|---|---|---|---|---|
| s1 | 0.500 | v0002 0.463 (**−3.66**) | ABSTAIN 0.00 | −3.66 |
| s2 | 0.390 | v0003 0.451 (**+6.10**) | ABSTAIN 0.00 | +6.10 |
| s3 | 0.463 | v0001 0.366 (**−9.76**) | ABSTAIN 0.00 | ABSTAIN 0.00 |
| **mean** | 0.451 | **−2.44 (1/3, sign_p 0.875 ns)** | **0.00 (never-lose 3/3)** | **+0.81 (1/3)** |

Deployable C+inc (keeps incumbent as candidate): s1 −3.66, s2 +6.10, s3 0.00 → **mean +0.81** (never
catastrophic — on s3 the incumbent out-vals all edits, so it is correctly kept).

## 4. Oracle headroom = the diagnosis (3/3 POSITIVE)
| seed | greedy | C(argmax-val) | oracle(argmax-test) | headroom | unrealized (oracle−C) |
|---|---|---|---|---|---|
| s1 | 0.500 | 0.463 | 0.549 | **+4.88** | +8.54 |
| s2 | 0.390 | 0.451 | 0.451 | **+6.10** | 0.00 |
| s3 | 0.463 | 0.366 | 0.537 | **+7.32** | +17.07 |
| **mean** | 0.451 | — | — | **+6.10** | **+8.54** |

## 5. Read
- **Diagnosis (gate discards a better skill) = MODEL-GENERAL**, 3/3, headroom +6.10 > DeepSeek +5.1.
  The eval-bound / Optimizer's-Curse thesis replicates cross-model. **This is the main win.**
- **Textbook eval-bound:** headroom is *bigger* on Qwen, capture is *worse* (val noisier) → realized
  gain NEGATIVE. realized = headroom × capture, capture SNR-capped — here below zero.
- **s2 = existence proof** decoupled selection captures cross-model (+6.10; val correctly = oracle).
- **Method-C magnitude is model/bench-specific:** DeepSeek-LiveMath +3.9 → Qwen-LiveMath −2.44.
- **Safety net transfers:** siggate never-lose 3/3 (0.00, dodges the −2.44); lcb@1 / C+inc +0.81.
  On Qwen, abstain > naive Method-C.

## 6. What we banked (NOT a failure)
1. Cross-model **confirmation of the core thesis** (diagnosis 3/3 + eval-bound mechanics, predicted
   sign and shape).
2. **Existence proof** Method-C transfers (s2 +6.10).
3. **Validated safety net** (never-lose) cross-model.
4. A clean **scope condition**: realized gain is SNR-gated (fires when val SNR is adequate, not when
   it isn't).

**Reporting guardrail (anti-oracle / anti-cherry-pick):** report the 3-seed picture (mean −2.44, 1/3,
safe-abstain), with s2's +6.10 as an *existence proof*, NOT as the headline. This keeps the result
defensible to "what about the other two seeds?"

## 7. Implication for the OfficeQA +10.2 headline
Method-C's magnitude is now shown to be model-specific → the OfficeQA +10.2 (DeepSeek) single-model
caveat is **NOT lifted — if anything more suspect**. BUT this run tested **LiveMath** (always the
*marginal* bench: DeepSeek C was +2.9..+3.9, never +10.2). OfficeQA-Qwen was **not** tested. To decide
the headline's fate, a fresh **OfficeQA-Qwen** run is required.

## 8. Next steps (user to choose)
- **A. OfficeQA-Qwen** — test whether +10.2 transfers (the real headline bench; 24-turn, costlier Qwen).
- **B. Full-Qwen (optimizer=Qwen) LiveMath** — the parked clean "Qwen-improves-Qwen" test.
- **C. Write it up** on the strengthened thesis (eval-bound + provable gate failure + safe abstention).

## 9. State of the world
box2 (`<server>`) = only live machine, KEEP. Auto-finish watcher
(`/root/lmqwenC_watch.sh`) self-exited at `VERDICT_PIPELINE_DONE`. DeepSeek DION/YH live; QWEN_KEY
spent some (target rollouts + posthoc). Code: committed `qd/decoupled_select.py`,
`tools/{posthoc_select,decoupled_ship}.py`; gitignored `tools/_oracle_headroom.py`.
