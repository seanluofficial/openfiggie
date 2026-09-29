"""Bayesian inference over the 12 deck configurations.

Given a player's initial hand, the posterior over decks is exact:

    P(deck | hand)  ∝  P(hand | deck)  =  prod_s C(n_s, h_s) / C(40, |hand|)

(the prior over decks is uniform and the denominator is constant, so only the
product of binomials matters).

Trades add hard constraints. Every card currently in some player's hand is a
distinct physical card, so at any moment

    n_s  >=  my_current[s] + sum over others of (cards of suit s known to be in their hand)

where "known" cards are ones a player was seen buying and has not since sold.
Decks violating a bound are ruled out. The *strategic* information in trades and
quotes (why someone bid for hearts) is not modeled here; that is the job of an
opponent model layered on top (see docs/ROADMAP.md, week 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb

from .cards import ALL_DECKS, NUM_SUITS, SUITS, Counts, DeckConfig
from .game import Observation, Trade

Posterior = dict[DeckConfig, float]


def hand_likelihood(deck: DeckConfig, hand: Sequence[int]) -> int:
    """Unnormalized P(hand | deck): the number of ways to draw ``hand`` from ``deck``."""
    counts = deck.counts
    out = 1
    for s in SUITS:
        out *= comb(counts[s], hand[s])
    return out


def deck_posterior(hand: Sequence[int], min_counts: Sequence[int] | None = None) -> Posterior:
    """Exact posterior over decks given a dealt hand and optional lower bounds on suit sizes."""
    weights: Posterior = {}
    for deck in ALL_DECKS:
        if min_counts is not None and any(deck.counts[s] < min_counts[s] for s in SUITS):
            continue
        w = hand_likelihood(deck, hand)
        if w:
            weights[deck] = float(w)
    total = sum(weights.values())
    if total == 0:
        raise ValueError("no deck is consistent with the observations")
    return {d: w / total for d, w in weights.items()}


def goal_probabilities(posterior: Mapping[DeckConfig, float]) -> tuple[float, ...]:
    probs = [0.0] * NUM_SUITS
    for deck, p in posterior.items():
        probs[deck.goal_suit] += p
    return tuple(probs)


class BeliefTracker:
    """Maintains one player's deck posterior and lower bounds on opponents' holdings."""

    def __init__(self, player: int, n_players: int, hand: Counts) -> None:
        self.player = player
        self.initial_hand = hand
        self.hand: Counts = hand
        # known[q][s]: cards of suit s player q is known to hold (always 0 for self).
        self.known = [[0] * NUM_SUITS for _ in range(n_players)]
        self.min_counts = list(hand)
        self._n_trades_seen = 0
        self.posterior = deck_posterior(hand)

    def update(self, obs: Observation) -> bool:
        """Consume new trades from ``obs``. Returns True if beliefs changed."""
        new = obs.trades[self._n_trades_seen :]
        if not new and obs.hand == self.hand:
            return False
        for trade in new:
            self._apply_trade(trade)
        self._n_trades_seen = len(obs.trades)
        self.hand = obs.hand

        changed = False
        for s in SUITS:
            seen = self.hand[s] + sum(k[s] for q, k in enumerate(self.known) if q != self.player)
            if seen > self.min_counts[s]:
                self.min_counts[s] = seen
                changed = True
        if changed:
            self.posterior = deck_posterior(self.initial_hand, self.min_counts)
        return True

    def _apply_trade(self, trade: Trade) -> None:
        if trade.seller != self.player:
            k = self.known[trade.seller]
            k[trade.suit] = max(0, k[trade.suit] - 1)
        if trade.buyer != self.player:
            self.known[trade.buyer][trade.suit] += 1

    def goal_probabilities(self) -> tuple[float, ...]:
        return goal_probabilities(self.posterior)
