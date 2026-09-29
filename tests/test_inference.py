import random
from collections import Counter

import pytest

from openfiggie.cards import ALL_DECKS, SUITS, deal
from openfiggie.inference import deck_posterior, goal_probabilities


def test_posterior_sums_to_one():
    post = deck_posterior((3, 3, 2, 2))
    assert sum(post.values()) == pytest.approx(1)
    assert len(post) == 12


def test_impossible_decks_excluded():
    # 11 spades rules out every deck where spades is not the 12-card suit.
    post = deck_posterior((10, 0, 0, 0), min_counts=(11, 0, 0, 0))
    assert all(d.long_suit == 0 for d in post)
    assert goal_probabilities(post)[1] == pytest.approx(1)


def test_posterior_matches_empirical_frequencies():
    """Brute force: deal many games and tabulate the deck for one frequent hand shape."""
    rng = random.Random(0)
    target = (3, 3, 2, 2)
    hits: Counter = Counter()
    for _ in range(200_000):
        deck, hands = deal(rng, 4)
        if hands[0] == target:
            hits[deck] += 1
    n = sum(hits.values())
    assert n > 2000
    post = deck_posterior(target)
    for deck in ALL_DECKS:
        assert hits[deck] / n == pytest.approx(post[deck], abs=0.025)


def test_goal_probabilities_are_calibrated():
    """If calibrated, E[p(true goal)] == E[sum_s p_s^2]."""
    rng = random.Random(1)
    p_true, p_sq = 0.0, 0.0
    n = 20_000
    for _ in range(n):
        deck, hands = deal(rng, 4)
        probs = goal_probabilities(deck_posterior(hands[0]))
        p_true += probs[deck.goal_suit]
        p_sq += sum(p * p for p in probs)
    assert p_true / n == pytest.approx(p_sq / n, abs=0.01)
    # A 10-card hand is only weakly informative: ~0.29 on the true goal vs 0.25 uniform.
    assert 0.27 < p_true / n < 0.32


def test_long_suit_signals_partner():
    probs = goal_probabilities(deck_posterior((6, 1, 2, 1)))
    assert max(SUITS, key=lambda s: probs[s]) == 1  # clubs, partner of spades
