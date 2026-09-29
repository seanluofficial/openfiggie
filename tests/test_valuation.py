import math
import random

import pytest

from openfiggie.cards import Suit
from openfiggie.inference import deck_posterior, goal_probabilities
from openfiggie.rules import GameConfig
from openfiggie.valuation import card_values, compositions, goal_payoff, sample_values

CONFIG = GameConfig()
NO_KNOWN = [[0] * 4 for _ in range(4)]


def test_compositions_count_and_bounds():
    comps = list(compositions(5, [3, 3, 3]))
    assert all(sum(c) == 5 and max(c) <= 3 for c in comps)
    assert len(comps) == len(set(comps)) == 12


def test_goal_payoff():
    assert goal_payoff(4, [3, 2, 1], 100, 10) == 140
    assert goal_payoff(3, [3, 2, 1], 100, 10) == 30 + 50
    assert goal_payoff(1, [3, 2, 1], 100, 10) == 10


@pytest.mark.parametrize(
    "hand, known",
    [
        ((3, 3, 2, 2), NO_KNOWN),
        ((5, 1, 3, 1), NO_KNOWN),
        ((2, 4, 0, 4), [[0] * 4, [0, 2, 0, 0], [0, 0, 1, 0], [0] * 4]),
    ],
)
def test_exact_matches_monte_carlo(hand, known):
    post = deck_posterior(hand)
    sizes = (10, 10, 10, 10)
    exact = card_values(CONFIG, 0, post, hand, sizes, known)
    mc = sample_values(random.Random(0), CONFIG, 0, post, hand, sizes, known, n_samples=20_000)
    for s in range(4):
        _close(exact.buy[s], mc.buy[s])
        _close(exact.sell[s], mc.sell[s])
        for q in range(1, 4):
            _close(exact.buy_from[q][s], mc.buy_from[q][s])
            _close(exact.sell_to[q][s], mc.sell_to[q][s])


def test_card_worth_at_least_goal_payout_times_probability():
    hand = (3, 3, 2, 2)
    post = deck_posterior(hand)
    v = card_values(CONFIG, 0, post, hand, (10,) * 4, NO_KNOWN)
    probs = goal_probabilities(post)
    for s in range(4):
        assert v.buy[s] >= 10 * probs[s] - 1e-9


def test_cannot_sell_what_you_do_not_hold():
    hand = (5, 5, 0, 0)
    v = card_values(CONFIG, 0, deck_posterior(hand), hand, (10,) * 4, NO_KNOWN)
    assert math.isnan(v.sell[Suit.HEARTS])
    assert all(math.isnan(x) for x in v.buy_from[0])  # no self-trades


def _close(a: float, b: float) -> None:
    if math.isnan(a) or math.isnan(b):
        assert math.isnan(a) and math.isnan(b)
    else:
        assert a == pytest.approx(b, abs=0.6)
