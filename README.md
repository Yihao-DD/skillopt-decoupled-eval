# SkillOpt + Decoupled Selection — Gain Pipeline

Pristine **SkillOpt** (the full self-improving-skills pipeline) + our **three decoupled-selection
optimizers** (`argmax` = Method-C / `siggate` / `lcb`) on **four benchmarks** — packaged to run out
of the box. **One model does everything** (optimizer reflection + target rollouts); switch
closed-source APIs by editing **3 lines** in `.env`.

Branch `gain-pipeline` (orphan, clean history). Zero-API tests: **41 passed**.

---

## Quickstart — turnkey, 4 steps
```bash
# 1. install  (Python 3.10+; tested on 3.13).  Installs BOTH the SkillOpt fork and this
#    root package (qd/), plus the materialization/test extras.
python -m venv .venv && .venv/Scripts/python -m pip install -e ./SkillOpt -e . -r requirements-extra.txt

# 2. pick ONE model: copy the template, set 3 lines
cp .env.example .env
#    MODEL_PROVIDER=deepseek     MODEL_API_KEY=sk-...     MODEL_NAME=deepseek-chat

# 3. benchmark data: LiveMath / SearchQA / SpreadsheetBench are ALREADY BUNDLED (offline, no network).
#    ONLY OfficeQA needs a download (HF_TOKEN required; inside CN: HF_ENDPOINT=https://hf-mirror.com):
python tools/materialize_officeqa.py --token "$HF_TOKEN"   # skip entirely if not running OfficeQA

# 4. run the benchmark — 3 seeds (1,2,3) BY DEFAULT — then score them with the 3 optimizers
python run.py --env livemathematicianbench                  # -> run_livemathematicianbench_s1/s2/s3
#   run.py prints the exact NEXT command; it is the 3-seed decoupled-selection verdict:
for s in 1 2 3; do python tools/posthoc_select.py --tag run_livemathematicianbench_s$s --env livemathematicianbench \
  --val-split-dir data/livemathematicianbench_split --val-split val \
  --test-split-dir data/livemathematicianbench_split --test-split test; done
SKILLOPT_OUT=SkillOpt/outputs python tools/decoupled_ship.py \
  --tags run_livemathematicianbench_s1,run_livemathematicianbench_s2,run_livemathematicianbench_s3 --rules argmax,siggate,lcb@1
```
- **Default is 3 seeds** (the experiment unit); `--seeds 1` for one, `--seeds 1 2 3 4 5` for more.
- `python run.py --env <bench> --dry` prints the per-seed commands, spends nothing.
- `python run.py --env <bench> --smoke` does a tiny SINGLE-seed run (checks auth + data + executor).

---

## 🔌 Switching the model API (the point of this package)
Everything reads ONE model from `.env`. To test a different closed-source API, change three lines:
```
MODEL_PROVIDER=openai      # openai|deepseek|qwen|moonshot|zhipu|together|openrouter|siliconflow|custom
MODEL_API_KEY=sk-...
MODEL_NAME=gpt-4o
```
- `MODEL_PROVIDER` fills the base URL (presets in `model_config.py`). For any other OpenAI-compatible
  host: `MODEL_PROVIDER=custom` + `MODEL_BASE_URL=...`.
- **The SAME model does optimizer reflection AND target rollouts** — a run is never split across two
  providers (unlike a mixed optimizer=A / target=B setup). `run.py` (training) and
  `tools/posthoc_select.py` (scoring) both resolve from this one place, so they always use the
  identical model. Check what's configured: `python model_config.py` (prints the resolved model, key masked).

---

## The three components (`qd/decoupled_select.py`)
SkillOpt's greedy gate ships `best_skill` — the running incumbent, overfit to the tiny noisy selection
val. We keep the whole explored trajectory and re-select at *ship time* on a high-power held-out val,
then report TEST. Anti-oracle: selection touches only the val, never the test.

| rule | ships | role |
|---|---|---|
| **argmax** (Method-C) | highest mean-val skill | the gain engine |
| **siggate** | argmax among skills that *significantly* paired-beat the greedy incumbent; else **abstain** | inference-aware, never-significantly-lose |
| **lcb** | highest positive lower-confidence-bound of the paired advantage; else **abstain** | inference-aware (soft shrinkage) |

Across seeds, `tools/decoupled_ship.py --tags t1,t2,t3 --rules argmax,siggate,lcb@1` gives the pooled
3-component verdict (zero new API — reads the per-item results posthoc cached). `tools/_oracle_headroom.py`
reports the diagnosis (did the greedy gate discard a better skill?).

---

## Benchmarks (4, fully wired)
`livemathematicianbench` · `officeqa` · `spreadsheetbench` · `searchqa` — each is one `--env <name>`
away in both `run.py` and `posthoc_select.py`. `tools/materialize_all.py` downloads all of them; after
that, using a benchmark needs no further setup. (Custom held-out val sizes for the selection
experiments: `tools/make_experiment_splits.py`.)

---

## Full per-seed flow (what produced the results in `docs/`)
```bash
python run.py --env livemathematicianbench --tag lm          # 3 seeds by default -> lm_s1/s2/s3
for s in 1 2 3; do python tools/posthoc_select.py --tag lm_s$s --env livemathematicianbench \
  --val-split-dir data/livemathematicianbench_split --val-split val \
  --test-split-dir data/livemathematicianbench_split --test-split test; done
SKILLOPT_OUT=SkillOpt/outputs python tools/decoupled_ship.py --tags lm_s1,lm_s2,lm_s3 --rules argmax,siggate,lcb@1
SKILLOPT_OUT=SkillOpt/outputs python tools/_oracle_headroom.py lm_s1,lm_s2,lm_s3
```

## Tests
```bash
python -m pytest qd/tests -q          # 41 passed (zero-API)
```

## Layout & provenance
```
SkillOpt/        pristine upstream fork @ HEAD 0948d2d via `git archive HEAD` — full pipeline
                 (engine/trainer, evaluation/gate, optimizer, model[OpenAI-compatible], prompts,
                 4 benchmark envs, scripts); 0 experimental hooks, no debug clutter
model_config.py  the single-model API switch (resolved by run.py + posthoc)
run.py           one-command training driver (vanilla SkillOpt greedy, single model end-to-end)
qd/              decoupled_select.py (3 rules) + pairgate.py + tests
tools/           posthoc_select.py · decoupled_ship.py · _oracle_headroom.py · mcnemar_pooled.py
                 · make_experiment_splits.py · materialize_all.py + materialize_{4 benchmarks}.py
repro/official/  mcnemar_compare.py + non-experimental patches
docs/            result writeups
```
**Excluded** from the research repo to keep this clean: the dead experimental lines (avp / kapo / rg /
router / drift / the original QD-over-Skills), their tools & probes, paper/PDF clutter, and all
in-trainer gate experiments (the pristine base has none).

## Results (`docs/`)
- **QWEN-CROSSMODEL-RESULT.md** — 3-seed cross-model verdict (diagnosis replicates 3/3; Method-C gain is SNR-gated).
- **INFERENCE-AWARE-SELECTION-RESULT.md** — siggate/lcb design + OfficeQA Method-C **+10.2**.
- **DECOUPLED-SELECTION-COLD-START.md** — Method-C design + the evaluation-bound thesis.
