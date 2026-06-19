"""Zero-API tests for the paired-difference statistics in ``qd.pairgate``.

Covers the pure paired stats the decoupled selector's ``siggate`` rule relies on:
``compare_paired`` (win/loss/tie counts + per-item diff), ``sign_test_p_one_sided``
(exact one-sided binomial p-value), and ``paired_gate_action`` (plain vs
significance accept predicates). All deterministic, zero-API.

(Regression tests for the pristine SkillOpt scalar gate ``evaluate_gate`` live with
the base under ``SkillOpt/tests`` — they are not part of this gain-pipeline overlay.)
"""
from __future__ import annotations

import pytest

from qd.pairgate import (
    compare_paired,
    paired_gate_action,
    sign_test_p_one_sided,
)


# ── helpers ──────────────────────────────────────────────────────────────
def _items(*grades: float) -> dict[str, float]:
    """Build a {id: hard} map with ids '0','1',... from positional grades."""
    return {str(i): float(g) for i, g in enumerate(grades)}


# ── sign_test_p_one_sided: exact binomial upper tail ─────────────────────
def test_sign_test_all_ties_returns_one():
    assert sign_test_p_one_sided(0, 0) == 1.0


def test_sign_test_known_values():
    # P(X >= 5 | Bin(5, .5)) = 1/32
    assert sign_test_p_one_sided(5, 0) == pytest.approx(1 / 32)
    # P(X >= 3 | Bin(3, .5)) = 1/8
    assert sign_test_p_one_sided(3, 0) == pytest.approx(1 / 8)
    # P(X >= 4 | Bin(5, .5)) = (C(5,4)+C(5,5))/32 = 6/32
    assert sign_test_p_one_sided(4, 1) == pytest.approx(6 / 32)
    # symmetric: P(X >= 1 | Bin(2, .5)) = 3/4
    assert sign_test_p_one_sided(1, 1) == pytest.approx(3 / 4)


def test_sign_test_monotone_in_wins():
    # more wins (same discordant total) -> smaller (more significant) p
    assert sign_test_p_one_sided(8, 2) < sign_test_p_one_sided(6, 4)


# ── compare_paired: counts, diff, id intersection ────────────────────────
def test_compare_paired_counts_wins_losses_ties():
    cand = _items(1, 1, 0, 1, 0)
    ref = _items(0, 1, 1, 0, 0)
    # item0 win, item1 tie, item2 loss, item3 win, item4 tie
    c = compare_paired(cand, ref)
    assert (c.wins, c.losses, c.ties, c.n) == (2, 1, 2, 5)
    assert c.diff == 1
    assert c.better is True


def test_compare_paired_only_intersection_of_ids():
    cand = {"a": 1.0, "b": 1.0, "x": 1.0}   # 'x' absent from ref
    ref = {"a": 0.0, "b": 0.0, "y": 0.0}    # 'y' absent from cand
    c = compare_paired(cand, ref)
    assert c.n == 2          # only a, b are paired
    assert c.wins == 2 and c.losses == 0


def test_compare_paired_equal_wins_losses_not_better():
    cand = _items(1, 0, 1, 0)
    ref = _items(0, 1, 0, 1)
    c = compare_paired(cand, ref)   # 2 wins, 2 losses
    assert c.diff == 0
    assert c.better is False
    assert c.significant is False


def test_compare_paired_handles_continuous_grades():
    # soft/continuous hard: win/loss decided by sign of the per-item difference
    cand = {"a": 0.9, "b": 0.2, "c": 0.5}
    ref = {"a": 0.4, "b": 0.6, "c": 0.5}
    c = compare_paired(cand, ref)
    assert (c.wins, c.losses, c.ties) == (1, 1, 1)


# ── paired_gate_action: plain mode (wins > losses) ───────────────────────
def test_paired_plain_accept_new_best_when_beats_current_and_best():
    cand = _items(1, 1, 1, 0)
    cur = _items(0, 0, 0, 0)
    best = _items(0, 0, 0, 0)
    action, diag = paired_gate_action(cand, cur, best, require_significant=False)
    assert action == "accept_new_best"
    assert diag["vs_current"]["diff"] == 3
    assert diag["vs_best"] is not None


def test_paired_plain_accept_not_best_when_beats_current_only():
    cand = _items(1, 1, 0, 0)   # beats current
    cur = _items(0, 0, 0, 0)
    best = _items(1, 1, 1, 1)   # but loses to a stronger best
    action, _ = paired_gate_action(cand, cur, best, require_significant=False)
    assert action == "accept"


def test_paired_plain_reject_when_not_better_than_current():
    cand = _items(1, 0, 0, 0)
    cur = _items(0, 1, 1, 0)   # cand loses 1-2
    best = cur
    action, diag = paired_gate_action(cand, cur, best, require_significant=False)
    assert action == "reject"
    assert diag["vs_best"] is None   # short-circuits before comparing to best


# ── paired_gate_action: significance mode (paired_sig) ───────────────────
def test_paired_sig_rejects_insignificant_even_if_better():
    # 2 wins, 0 losses -> better, but p = 1/4 = 0.25 > 0.05 -> reject under sig
    cand = _items(1, 1, 0, 0)
    cur = _items(0, 0, 0, 0)
    best = cur
    plain, _ = paired_gate_action(cand, cur, best, require_significant=False)
    sig, diag = paired_gate_action(cand, cur, best, require_significant=True, alpha=0.05)
    assert plain == "accept_new_best"
    assert sig == "reject"
    # significance ablation is logged regardless of the active predicate
    assert diag["would_accept_plain"] is True
    assert diag["would_accept_significant"] is False


def test_paired_sig_accepts_when_significant():
    # 5 wins, 0 losses -> p = 1/32 ≈ 0.031 < 0.05 -> accept under sig
    cand = _items(1, 1, 1, 1, 1)
    cur = _items(0, 0, 0, 0, 0)
    best = cur
    sig, diag = paired_gate_action(cand, cur, best, require_significant=True, alpha=0.05)
    assert sig == "accept_new_best"
    assert diag["would_accept_significant"] is True
