"""Paired-difference acceptance gate for SkillOpt selection (Phase-1 SELECTION-FIX).

Background
----------
The official SkillOpt validation gate accepts a candidate skill iff its *mean*
hard accuracy on the fixed selection set ``D_sel`` beats the incumbent's mean
(``skillopt.evaluation.gate.evaluate_gate``: ``cand_score > current_score``).
On a small, noisy ``D_sel`` (n=40) that scalar comparison turns grader noise
into systematic val-overfit (high-val skills that fail on the held-out test).

This module is the *only* change of Phase-1 SELECTION-FIX: a PER-ITEM PAIRED
comparison on the same ``D_sel`` items. Every candidate and the incumbent are
evaluated on the identical, fixed selection set (same items, same target seed),
so their per-item outcomes are *paired*; we compare the paired difference
(``#items the candidate wins`` minus ``#items it loses``) instead of the
difference of means. With positively-correlated per-item noise (measured
ρ≈+0.56 on this benchmark) the paired statistic has far lower variance than the
mean difference — that variance reduction is the lever.

Two modes:

* ``"paired"``     — accept iff ``wins > losses`` (sign of the paired diff).
* ``"paired_sig"`` — accept iff ``wins > losses`` AND a one-sided exact sign
  test on the discordant pairs is significant (``p < alpha``): abstain under
  noise, keep the incumbent.

Pure and dependency-free (exact binomial via :func:`math.comb` — no scipy) so it
is unit-tested zero-API and imported lazily by ``skillopt.engine.trainer``.

Faithfulness
------------
This changes ONLY the accept predicate. Proposal, merge, rank, best-of-K
survivor selection, and the ``D_sel`` set itself are untouched. ``D_sel`` stays
a candidate-level gate — never an edit-level micro-eval signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Mapping

# Gate-action strings mirror skillopt.evaluation.gate.GateAction (kept as plain
# str here so this module has no dependency on the vendored SkillOpt package).
GateActionStr = str  # "accept_new_best" | "accept" | "reject"

_EPS = 1e-9


@dataclass(frozen=True)
class PairedComparison:
    """Outcome of comparing a candidate vs a reference on paired per-item grades."""

    n: int            # paired items compared (intersection of item ids)
    wins: int         # items where the candidate strictly beats the reference
    losses: int       # items where the candidate strictly loses to the reference
    ties: int         # items equal within ``eps``
    diff: int         # wins - losses (net paired difference)
    p_value: float    # one-sided exact sign test: P(X >= wins | X ~ Bin(wins+losses, .5))
    better: bool      # diff > 0
    significant: bool # better and p_value < alpha


def sign_test_p_one_sided(wins: int, losses: int) -> float:
    """One-sided exact sign-test p-value for 'candidate better than reference'.

    Under H0 (no per-item difference) the ``wins + losses`` discordant pairs
    split 50/50, so ``wins ~ Binomial(wins + losses, 0.5)``. Returns the upper
    tail ``P(X >= wins)``. All-ties (``wins + losses == 0``) → ``1.0`` (no
    evidence). This is McNemar's exact test on the discordant pairs.
    """
    n_disc = wins + losses
    if n_disc <= 0:
        return 1.0
    tail = sum(comb(n_disc, k) for k in range(wins, n_disc + 1))
    return tail / (2 ** n_disc)


def compare_paired(
    cand_items: Mapping[str, float],
    ref_items: Mapping[str, float],
    *,
    alpha: float = 0.05,
    eps: float = _EPS,
) -> PairedComparison:
    """Compare candidate vs reference on the intersection of their item ids.

    Only ids present in BOTH maps are paired (defensive — under the fixed
    ``D_sel`` they coincide). An item is a *win* if ``cand - ref > eps``, a
    *loss* if ``ref - cand > eps``, otherwise a *tie*.
    """
    ids = sorted(set(cand_items) & set(ref_items))
    wins = losses = ties = 0
    for i in ids:
        d = float(cand_items[i]) - float(ref_items[i])
        if d > eps:
            wins += 1
        elif d < -eps:
            losses += 1
        else:
            ties += 1
    diff = wins - losses
    p_value = sign_test_p_one_sided(wins, losses)
    better = diff > 0
    return PairedComparison(
        n=len(ids),
        wins=wins,
        losses=losses,
        ties=ties,
        diff=diff,
        p_value=p_value,
        better=better,
        significant=better and p_value < alpha,
    )


def paired_gate_action(
    cand_items: Mapping[str, float],
    current_items: Mapping[str, float],
    best_items: Mapping[str, float],
    *,
    require_significant: bool,
    alpha: float = 0.05,
) -> tuple[GateActionStr, dict]:
    """Paired analogue of ``evaluate_gate``'s accept / accept_new_best / reject.

    Mirrors the scalar gate's two-level structure: accept into the trajectory if
    the candidate paired-beats the *current* skill; mark a new best if it ALSO
    paired-beats the *best* skill. ``require_significant`` switches the predicate
    from ``wins > losses`` to ``wins > losses AND sign-test p < alpha``.

    Returns ``(action, diagnostics)`` where ``action`` is one of
    ``"accept_new_best" | "accept" | "reject"`` and ``diagnostics`` is a
    JSON-able dict for per-step instrumentation (including the significance
    ablation: ``would_accept_plain`` / ``would_accept_significant`` are recorded
    regardless of which predicate is active).
    """
    vs_current = compare_paired(cand_items, current_items, alpha=alpha)
    accept = vs_current.significant if require_significant else vs_current.better

    diag: dict = {
        "require_significant": require_significant,
        "alpha": alpha,
        "vs_current": asdict(vs_current),
        "would_accept_plain": vs_current.better,
        "would_accept_significant": vs_current.significant,
    }

    if not accept:
        diag["vs_best"] = None
        return "reject", diag

    vs_best = compare_paired(cand_items, best_items, alpha=alpha)
    new_best = vs_best.significant if require_significant else vs_best.better
    diag["vs_best"] = asdict(vs_best)
    return ("accept_new_best" if new_best else "accept"), diag
