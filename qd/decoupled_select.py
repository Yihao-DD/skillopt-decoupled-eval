"""Decoupled selection rules for explore-then-select (Method C, the ship step).

SkillOpt's greedy gate ships ``best_skill`` (the running incumbent, contaminated by the
slow-update — it overfits the tiny noisy D_sel). Method C DECOUPLES selection: keep the
whole explored trajectory and, at ship time, pick the skill by a HIGH-POWER held-out val
(disjoint from test). This module is the pure selection primitive used by that ship step.

Note (mechanistic, proven in trainer.py): with a rolling-best anchor the *exploration*
path is byte-identical to greedy (greedy already keeps current==best). So the realizable
gain is entirely in THIS selection step, not in relaxing exploration.

Four rules over per-item val scores ``{skill_id: {item_id: 0/1}}``:
  - ``argmax``   : ship the highest mean-val skill (simple best-arm). The default ship rule.
  - ``copeland`` : ship the skill that wins the most pairwise half-split duels — robust to
    which val items you happened to sample (winner's-curse-resistant). Aims to push past
    significance. (Same robustness idea as tools/analyze_selection_generalization.py, here
    as a reusable ship-time primitive.)
  - ``siggate``  : INFERENCE-AWARE (deep-research v2 dir ③). Ship argmax-mean only among
    candidates whose paired advantage over the GREEDY INCUMBENT passes a one-sided sign test
    (p < alpha); otherwise abstain to the incumbent. Optimizes for a gain that REPRODUCES on
    held-out, not the point estimate — the Optimizer's-Curse remedy (Smith & Winkler 2006).
  - ``lcb``      : INFERENCE-AWARE (soft). Ship the candidate with the highest LOWER confidence
    bound of its paired advantage over the incumbent (mean_diff - z*SE); abstain if none > 0.

``siggate`` / ``lcb`` need an ``incumbent`` per-item dict (the greedy ship, ``best_skill``) and
may return the ``INCUMBENT`` sentinel = "abstain, ship the incumbent" (a never-significantly-lose
selector). Pure / zero-API / deterministic. Anti-oracle: selects on the VAL only; the caller
reports the shipped skill's TEST score separately.
"""
from __future__ import annotations

import random

from qd.pairgate import compare_paired

DEFAULT_SPLITS = 200
DEFAULT_SEED = 42
INCUMBENT = "__incumbent__"  # sentinel: abstain and ship the greedy incumbent (best_skill)
_EPS = 1e-9


def common_ids(pool: dict[str, dict[str, float]]) -> list[str]:
    """The item ids scored by EVERY skill (so all skills are compared on the same set)."""
    ids: set[str] | None = None
    for scores in pool.values():
        ids = set(scores) if ids is None else ids & set(scores)
    if not ids:
        raise ValueError("no common item ids across skills in the pool")
    return sorted(ids)


def _mean(scores: dict[str, float], ids: list[str]) -> float:
    return sum(scores[i] for i in ids) / len(ids) if ids else 0.0


def argmax_mean(pool: dict[str, dict[str, float]]) -> str:
    """Ship the skill with the highest mean val accuracy.

    Deterministic: iterates skills in sorted-id order and ``max`` keeps the FIRST maximal,
    so ties resolve to the lexically-smallest skill id.
    """
    ids = common_ids(pool)
    return max(sorted(pool), key=lambda k: _mean(pool[k], ids))


def copeland_robust(pool: dict[str, dict[str, float]], *,
                    n_splits: int = DEFAULT_SPLITS, seed: int = DEFAULT_SEED) -> str:
    """Ship the skill that wins the most pairwise half-split duels (Copeland).

    For each of ``n_splits`` random halves of the val items, every pair of skills duels on
    that half (higher half-mean wins; tie = 0.5 each). The skill with the most duel-wins
    ships. This resists the winner's curse: a skill that only looks best on a lucky subset
    rarely wins a majority of duels. Deterministic for a given ``seed``. Ties resolve by
    full-val mean, then skill id.
    """
    ids = common_ids(pool)
    keys = sorted(pool)
    if len(keys) == 1:
        return keys[0]
    half = max(1, len(ids) // 2)
    rng = random.Random(seed)
    wins: dict[str, float] = {k: 0.0 for k in keys}
    for _ in range(n_splits):
        h = rng.sample(ids, half)
        sc = {k: _mean(pool[k], h) for k in keys}
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                if sc[ki] > sc[kj]:
                    wins[ki] += 1
                elif sc[kj] > sc[ki]:
                    wins[kj] += 1
                else:
                    wins[ki] += 0.5
                    wins[kj] += 0.5
    return max(keys, key=lambda k: (wins[k], _mean(pool[k], ids), k))


def significance_gated(pool: dict[str, dict[str, float]],
                       incumbent: dict[str, float], *,
                       alpha: float = 0.05, eps: float = _EPS) -> str:
    """Ship argmax-mean among candidates that SIGNIFICANTLY paired-beat the incumbent.

    A candidate is *eligible* iff its per-item paired advantage over ``incumbent`` (the greedy
    ship) passes a one-sided exact sign test at ``alpha`` (``compare_paired(...).significant``).
    Among the eligible, ship the highest mean-val skill (ties -> lexically-smallest id). If NONE
    are eligible, return :data:`INCUMBENT` — abstain and keep the greedy ship.

    This is the inference-aware accept rule: only commit to a deviation from greedy whose val
    advantage is statistically distinguishable from noise, so the gain is more likely to
    reproduce on held-out test (the Optimizer's-Curse remedy). Deterministic, zero-API.

    Assumes candidates and the incumbent are scored on the SAME items (a fixed val split):
    eligibility uses each candidate's paired intersection with the incumbent; ranking among
    the eligible uses their common item set. These coincide under uniform coverage.
    """
    eligible = [k for k in sorted(pool)
                if compare_paired(pool[k], incumbent, alpha=alpha, eps=eps).significant]
    if not eligible:
        return INCUMBENT
    ids = common_ids({k: pool[k] for k in eligible})
    return max(eligible, key=lambda k: _mean(pool[k], ids))


def paired_lcb(pool: dict[str, dict[str, float]],
               incumbent: dict[str, float], *, z: float = 1.0) -> str:
    """Ship the candidate with the highest positive LOWER confidence bound of its advantage.

    For each candidate, the paired per-item difference ``d_i = cand_i - incumbent_i`` has mean
    ``mean_d`` and standard error ``SE = sd(d)/sqrt(n)`` (sample sd, ddof=1). The lower confidence
    bound of the true advantage is ``LCB = mean_d - z*SE``. Ship the candidate with the largest
    ``LCB`` provided it is strictly > 0; otherwise return :data:`INCUMBENT` (abstain).

    The continuous analogue of :func:`significance_gated`: ``z`` trades reproducibility against
    boldness (higher ``z`` = more skeptical). Penalizing the estimate by its own noise is the
    shrinkage view of the Optimizer's Curse (low-variance criterion, Cawley & Talbot 2010).
    Deterministic, zero-API. Each LCB uses that candidate's paired intersection with the
    incumbent, so LCBs are directly comparable only under uniform item coverage (a fixed val
    split) — the regime this is run in.
    """
    best_key, best_lcb = INCUMBENT, 0.0
    for k in sorted(pool):
        ids = sorted(set(pool[k]) & set(incumbent))
        n = len(ids)
        if n == 0:
            continue
        diffs = [float(pool[k][i]) - float(incumbent[i]) for i in ids]
        mean_d = sum(diffs) / n
        if n >= 2:
            var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
            se = (var / n) ** 0.5
        else:
            se = float("inf")  # cannot estimate noise from one item -> never bold on it
        lcb = mean_d - z * se
        if lcb > best_lcb:
            best_key, best_lcb = k, lcb
    return best_key


def select(pool: dict[str, dict[str, float]], rule: str = "argmax", *,
           incumbent: dict[str, float] | None = None, **kwargs) -> str:
    """Dispatch to a selection rule. ``rule`` in {"argmax", "copeland", "siggate", "lcb"}.

    ``siggate``/``lcb`` are inference-aware and REQUIRE ``incumbent`` (the greedy ship's per-item
    val); they may return :data:`INCUMBENT`. ``argmax``/``copeland`` ignore ``incumbent``.
    """
    if rule == "argmax":
        return argmax_mean(pool)
    if rule == "copeland":
        return copeland_robust(pool, **kwargs)
    if rule in ("siggate", "lcb"):
        if incumbent is None:
            raise ValueError(f"selection rule {rule!r} requires an `incumbent` per-item dict")
        if rule == "siggate":
            return significance_gated(pool, incumbent, **kwargs)
        return paired_lcb(pool, incumbent, **kwargs)
    raise ValueError(
        f"unknown selection rule {rule!r}; expected 'argmax', 'copeland', 'siggate' or 'lcb'")
