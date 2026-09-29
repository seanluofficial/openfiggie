import math
import random

import pytest

from openfiggie import Game
from openfiggie.agents import FundamentalAgent, NaiveAgent, RandomAgent
from openfiggie.agents.bayes import BayesAgent
from openfiggie.cards import deal
from openfiggie.inference import deck_posterior, goal_probabilities
from openfiggie.opponent_model import ParticleBelief
from openfiggie.rules import GameConfig

CONFIG = GameConfig()


def test_mcmc_moves_preserve_the_exact_prior():
    """No events: repeated resample-move must keep the deck posterior equal to the exact one."""
    hand = (4, 1, 3, 2)
    belief = ParticleBelief(0, CONFIG, hand, random.Random(0), n_particles=3000)
    for _ in range(5):
        belief._resample_move()
    exact = goal_probabilities(deck_posterior(hand))
    for got, want in zip(belief.goal_probabilities(), exact, strict=True):
        assert got == pytest.approx(want, abs=0.03)


def test_deck_move_revives_a_dead_deck():
    """Start every particle on one deck; deck moves must spread them back out."""
    hand = (3, 3, 2, 2)
    belief = ParticleBelief(0, CONFIG, hand, random.Random(1), n_particles=2000)
    only = belief.decks[0]
    for j in range(len(belief.decks)):
        belief.decks[j] = only
        belief.hands[j] = belief._deal_opponents(only)
    for _ in range(30):
        belief._resample_move()
    exact = goal_probabilities(deck_posterior(hand))
    for got, want in zip(belief.goal_probabilities(), exact, strict=True):
        assert got == pytest.approx(want, abs=0.05)


def test_cached_likelihoods_match_recomputation():
    agents = [BayesAgent(n_particles=150), NaiveAgent(), FundamentalAgent(), RandomAgent()]
    Game(agents, GameConfig(n_ticks=80), seed=3).play()
    belief = agents[0].belief
    for j in range(len(belief.logw)):
        fresh = belief._particle_loglik(j)
        if fresh == -math.inf:
            assert belief.logw[j] == -math.inf
        else:
            assert belief._ll[j] == pytest.approx(fresh, abs=1e-6)


def test_particles_respect_public_holding_constraints():
    agents = [BayesAgent(n_particles=150), NaiveAgent(), FundamentalAgent(), RandomAgent()]
    Game(agents, GameConfig(n_ticks=80), seed=4).play()
    belief = agents[0].belief
    w = belief.weights()
    for j, hands in enumerate(belief.hands):
        if w[j] > 0:
            for i, hand in enumerate(hands):
                assert belief._feasible(i, hand)


def test_particle_hands_are_consistent_with_their_deck():
    _, hands = deal(random.Random(5), 4)
    belief = ParticleBelief(0, CONFIG, hands[0], random.Random(5), n_particles=200)
    for _ in range(3):
        belief._resample_move()
    for deck, opp in zip(belief.decks, belief.hands, strict=True):
        total = tuple(hands[0][s] + sum(h[s] for h in opp) for s in range(4))
        assert total == deck.counts
