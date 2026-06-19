"""Zero-API tests for qd.decoupled_select (Method C ship-time selection rules)."""
import pytest

from qd.decoupled_select import argmax_mean, common_ids, copeland_robust, select


def _pool(**skills):
    """skill_name -> list of 0/1 over items  =>  {skill: {item_id: score}}."""
    return {name: {str(i): float(v) for i, v in enumerate(vals)} for name, vals in skills.items()}


# ---- argmax_mean ----

def test_argmax_picks_highest_mean():
    pool = _pool(A=[1, 1, 1, 1], B=[1, 1, 0, 0], C=[0, 0, 0, 0])
    assert argmax_mean(pool) == "A"


def test_argmax_tie_breaks_to_lexically_smallest():
    pool = _pool(B=[1, 1, 0, 0], A=[1, 0, 1, 0])  # both mean 0.5
    assert argmax_mean(pool) == "A"


def test_argmax_single_skill():
    assert argmax_mean(_pool(only=[1, 0, 1])) == "only"


# ---- copeland_robust ----

def test_copeland_picks_dominant_skill():
    # A is correct everywhere B is, plus more -> A wins every duel.
    pool = _pool(A=[1, 1, 1, 1, 1, 1], B=[1, 1, 1, 0, 0, 0], C=[0, 0, 0, 0, 0, 0])
    assert copeland_robust(pool, n_splits=100, seed=1) == "A"


def test_copeland_single_skill():
    assert copeland_robust(_pool(only=[1, 0, 1, 0])) == "only"


def test_copeland_deterministic():
    pool = _pool(A=[1, 1, 0, 1, 0, 1], B=[1, 0, 1, 0, 1, 1], C=[0, 1, 1, 1, 0, 0])
    assert copeland_robust(pool, n_splits=50, seed=7) == copeland_robust(pool, n_splits=50, seed=7)


def test_copeland_returns_pool_member():
    pool = _pool(A=[1, 0, 1, 0, 1, 0], B=[0, 1, 0, 1, 0, 1], C=[1, 1, 0, 0, 1, 1])
    assert copeland_robust(pool, n_splits=30, seed=3) in pool


def test_copeland_stable_on_equal_mean():
    # spiky vs steady, SAME mean (0.5) but different distribution; Copeland must be
    # deterministic and pick a valid member (the empirical spike-vs-steady direction is
    # measured on real data, not asserted here).
    pool = _pool(spiky=[1, 1, 1, 1, 0, 0, 0, 0], steady=[1, 0, 1, 0, 1, 0, 1, 0])
    pick = copeland_robust(pool, n_splits=200, seed=42)
    assert pick in {"spiky", "steady"}
    assert pick == copeland_robust(pool, n_splits=200, seed=42)


# ---- select dispatch ----

def test_select_argmax():
    assert select(_pool(A=[1, 1, 1], B=[0, 0, 0]), rule="argmax") == "A"


def test_select_copeland():
    assert select(_pool(A=[1, 1, 1, 1], B=[0, 0, 0, 0]), rule="copeland", n_splits=20, seed=1) == "A"


def test_select_unknown_rule_raises():
    with pytest.raises(ValueError):
        select(_pool(A=[1, 0]), rule="bogus")


# ---- common_ids ----

def test_common_ids_intersects():
    pool = {"A": {"x": 1.0, "y": 0.0}, "B": {"y": 1.0, "z": 0.0}}
    assert common_ids(pool) == ["y"]


def test_common_ids_empty_raises():
    with pytest.raises(ValueError):
        common_ids({"A": {"x": 1.0}, "B": {"y": 0.0}})
