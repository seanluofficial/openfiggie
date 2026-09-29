"""A market maker that trades on exact deck inference and exact card values.

Each tick it either takes a mispriced quote (if the edge beats ``take_edge``) or
posts a quote on one (suit, side) slot, cycling through all eight. Quotes sit
``half_spread`` chips around its fair value. Inventory skew comes for free from
the valuation: once it holds enough of a suit to win the majority, the next card
of that suit is worth less, so it bids lower.

``value_noise`` adds a per-game, per-suit bias to every value (a player who
misjudges the deck and stays wrong all game); ``activity`` is the probability of
acting on a given tick. Together they generate a population of imperfect
opponents. Subclasses change the belief model by overriding ``compute_values``.
"""

from __future__ import annotations

import math
import random

from ..actions import Action, Bid, Buy, Offer, Sell
from ..cards import SUITS, Counts
from ..game import Observation
from ..inference import BeliefTracker
from ..rules import GameConfig
from ..valuation import SuitValues, card_values
from .base import Agent


class FundamentalAgent(Agent):
    """Market maker on exact posterior + exact card values; ignores what opponents' quotes imply."""

    name = "fundamental"

    def __init__(
        self,
        half_spread: float = 2.0,
        take_edge: float = 1.0,
        value_noise: float = 0.0,
        activity: float = 1.0,
    ):
        self.half_spread = half_spread
        self.take_edge = take_edge
        self.value_noise = value_noise
        self.activity = activity

    def reset(self, player: int, config: GameConfig, hand: Counts, rng: random.Random) -> None:
        super().reset(player, config, hand, rng)
        self.beliefs = BeliefTracker(player, config.n_players, hand)
        self.values: SuitValues | None = None
        self.bias = [rng.gauss(0, self.value_noise) if self.value_noise else 0.0 for _ in SUITS]
        self._slot = 0

    def compute_values(self, obs: Observation) -> SuitValues | None:
        """Return fresh values if beliefs changed, else None to keep the current ones."""
        if not self.beliefs.update(obs) and self.values is not None:
            return None
        return card_values(
            self.config,
            self.player,
            self.beliefs.posterior,
            obs.hand,
            obs.hand_sizes,
            self.beliefs.known,
        )

    def act(self, obs: Observation) -> Action | None:
        fresh = self.compute_values(obs)
        if fresh is not None:
            self.values = _shift(fresh, self.bias) if self.value_noise else fresh
        if self.activity < 1 and self.rng.random() > self.activity:
            return None
        return self._take(obs, self.values) or self._quote(obs, self.values)

    def _take(self, obs: Observation, v: SuitValues) -> Action | None:
        best: tuple[float, Action] | None = None
        for s in SUITS:
            ask = obs.book.asks[s]
            if ask and ask.player != self.player and ask.price <= obs.chips[self.player]:
                edge = v.buy_from[ask.player][s] - ask.price
                if edge > self.take_edge and (best is None or edge > best[0]):
                    best = (edge, Buy(s))
            bid = obs.book.bids[s]
            if bid and bid.player != self.player and obs.hand[s]:
                edge = bid.price - v.sell_to[bid.player][s]
                if edge > self.take_edge and (best is None or edge > best[0]):
                    best = (edge, Sell(s))
        return best[1] if best else None

    def _quote(self, obs: Observation, v: SuitValues) -> Action | None:
        for _ in range(8):
            slot = self._slot
            self._slot = (self._slot + 1) % 8
            s, is_bid = SUITS[slot // 2], slot % 2 == 0
            if is_bid:
                fair = v.buy[s]
                if math.isnan(fair):
                    continue
                price = math.floor(fair - self.half_spread)
                bid, ask = obs.book.bids[s], obs.book.asks[s]
                if ask and ask.player == self.player and price >= ask.price:
                    continue  # would cross our own offer
                if 1 <= price <= obs.chips[self.player] and (bid is None or price > bid.price):
                    return Bid(s, price)
            else:
                fair = v.sell[s]
                if math.isnan(fair) or not obs.hand[s]:
                    continue
                price = max(1, math.ceil(fair + self.half_spread))
                bid, ask = obs.book.bids[s], obs.book.asks[s]
                if bid and bid.player == self.player and price <= bid.price:
                    continue  # would cross our own bid
                if ask is None or price < ask.price:
                    return Offer(s, price)
        return None


def _shift(v: SuitValues, bias: list[float]) -> SuitValues:
    def row(xs: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(x + bias[s] for s, x in enumerate(xs))

    return SuitValues(
        buy=row(v.buy),
        sell=row(v.sell),
        buy_from=tuple(row(r) for r in v.buy_from),
        sell_to=tuple(row(r) for r in v.sell_to),
    )
