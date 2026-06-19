"""Zero-API tests for the inference-aware selection rules (Method C, deep-research v2 direction ③).

Instead of shipping the argmax-mean candidate — which suffers the Optimizer's Curse
(Smith & Winkler 2006: the argmax of noisy estimates is biased high and reproduces poorly) —
these rules ship a deviation from the greedy incumbent ONLY when its held-out-val advantage is
statistically distinguishable from noise. They optimize for REPRODUCIBILITY, not the point
estimate, and abstain to the incumbent (= the greedy ship) otherwise — a never-(significantly)-lose
selector built on the existing paired stats in ``qd.pairgate``.

  - ``significance_gated`` : eligible = candidates passing a one-sided paired sign test vs the
    incumbent (p < alpha); ship argmax-mean among them, else abstain to the incumbent.
  - ``paired_lcb``         : ship the candidate with the highest LOWER confidence bound of its
    paired advantage over the incumbent (mean_diff - z*SE); abstain if none has LCB > 0.
"""
import pytest

from qd.decoupled_select import INCUMBENT, paired_lcb, select, significance_gated


def _pool(**skills):
    """skill_name -> list of 0/1 over items => {skill: {item_id: score}}."""
    return {name: {str(i): float(v) for i, v in enumerate(vals)} for name, vals in skills.items()}


def _items(vals):
    return {str(i): float(v) for i, v in enumerate(vals)}


# ---- significance_gated ----

def test_siggate_selects_significantly_better_candidate():
    # A beats an all-wrong incumbent on all 12 items: one-sided sign p = 2^-12 << 0.05.
    pool = _pool(A=[1] * 12)
    incumbent = _items([0] * 12)
    assert significance_gated(pool, incumbent) == "A"


def test_siggate_abstains_when_advantage_not_significant():
    # A wins in mean (2/4) but only 2 discordant wins -> p = 0.25 > 0.05 -> abstain to incumbent.
    pool = _pool(A=[1, 1, 0, 0])
    incumbent = _items([0, 0, 0, 0])
    assert significance_gated(pool, incumbent) == INCUMBENT


def test_siggate_abstains_when_all_candidates_worse():
    pool = _pool(A=[0] * 8)
    incumbent = _items([1] * 8)
    assert significance_gated(pool, incumbent) == INCUMBENT


def test_siggate_picks_higher_mean_among_significant():
    # Both A and B significantly beat the all-wrong incumbent; ship the higher-mean one.
    pool = _pool(A=[1] * 10 + [0, 0], B=[1] * 12)
    incumbent = _items([0] * 12)
    assert significance_gated(pool, incumbent) == "B"


def test_siggate_candidate_identical_to_incumbent_not_eligible():
    pool = _pool(A=[1, 0, 1, 0, 1, 0])
    incumbent = _items([1, 0, 1, 0, 1, 0])
    assert significance_gated(pool, incumbent) == INCUMBENT


def test_siggate_alpha_controls_strictness():
    # 5 clean wins -> one-sided p = 2^-5 = 0.03125: significant at 0.05, not at 0.01.
    pool = _pool(A=[1, 1, 1, 1, 1])
    incumbent = _items([0, 0, 0, 0, 0])
    assert significance_gated(pool, incumbent, alpha=0.05) == "A"
    assert significance_gated(pool, incumbent, alpha=0.01) == INCUMBENT


# ---- paired_lcb ----

def test_lcb_selects_candidate_with_positive_lower_bound():
    pool = _pool(A=[1] * 16)          # uniform advantage -> SE 0 -> LCB = 1.0 > 0
    incumbent = _items([0] * 16)
    assert paired_lcb(pool, incumbent, z=1.0) == "A"


def test_lcb_abstains_when_lower_bound_not_positive():
    # 2 wins, 1 loss, 9 ties -> mean_diff small vs paired noise -> LCB < 0 at z=1 -> abstain.
    pool = _pool(A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    incumbent = _items([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert paired_lcb(pool, incumbent, z=1.0) == INCUMBENT


def test_lcb_higher_z_more_conservative():
    # 4 wins / 8 ties: LCB > 0 at z=1 (select) but LCB < 0 at z=3 (abstain).
    pool = _pool(A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    incumbent = _items([0] * 12)
    assert paired_lcb(pool, incumbent, z=1.0) == "A"
    assert paired_lcb(pool, incumbent, z=3.0) == INCUMBENT


def test_lcb_picks_highest_lower_bound():
    pool = _pool(A=[1] * 8 + [0] * 8, B=[1] * 16)
    incumbent = _items([0] * 16)
    assert paired_lcb(pool, incumbent, z=1.0) == "B"


def test_lcb_abstains_when_all_worse():
    pool = _pool(A=[0] * 10)
    incumbent = _items([1] * 10)
    assert paired_lcb(pool, incumbent, z=1.0) == INCUMBENT


def test_lcb_deterministic():
    pool = _pool(A=[1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], B=[1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1])
    incumbent = _items([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    assert paired_lcb(pool, incumbent, z=1.0) == paired_lcb(pool, incumbent, z=1.0)


# ---- select dispatch ----

def test_select_siggate():
    assert select(_pool(A=[1] * 12), rule="siggate", incumbent=_items([0] * 12)) == "A"


def test_select_lcb():
    assert select(_pool(A=[1] * 16), rule="lcb", incumbent=_items([0] * 16)) == "A"


def test_select_siggate_requires_incumbent():
    with pytest.raises(ValueError):
        select(_pool(A=[1, 0]), rule="siggate")


def test_select_lcb_requires_incumbent():
    with pytest.raises(ValueError):
        select(_pool(A=[1, 0]), rule="lcb")
