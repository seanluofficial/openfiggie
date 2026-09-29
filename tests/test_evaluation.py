import pytest

from openfiggie.agents import PassiveAgent, RandomAgent
from openfiggie.evaluation import bootstrap_ci, estimate, run_duplicate
from openfiggie.rules import GameConfig

FAST = GameConfig(n_ticks=40)


def test_estimate():
    e = estimate([1.0, 2.0, 3.0, 4.0])
    assert e.mean == 2.5
    lo, hi = e.ci95
    assert lo < 2.5 < hi


def test_bootstrap_ci_contains_mean():
    xs = [float(i % 7) for i in range(200)]
    lo, hi = bootstrap_ci(xs)
    assert lo < sum(xs) / len(xs) < hi


def test_duplicate_passive_agents_score_exactly_zero():
    """Four identical passive agents each play every seat, so card luck cancels exactly."""
    lineup = [("passive", PassiveAgent)] * 4
    result = run_duplicate(lineup, n_deals=5, config=FAST)
    assert result.labels == ("passive#1", "passive#2", "passive#3", "passive#4")
    for row in result.scores:
        assert row == pytest.approx((0, 0, 0, 0))


def test_duplicate_is_zero_sum_per_deal():
    lineup = [("random", RandomAgent), ("passive", PassiveAgent)] * 2
    result = run_duplicate(lineup, n_deals=5, config=FAST)
    for row in result.scores:
        assert sum(row) == pytest.approx(0)
    assert result.n_games == 20
