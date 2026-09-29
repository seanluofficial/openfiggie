"""Reference opponents. Every stronger agent should beat all of these."""

from __future__ import annotations

import random

from ..actions import Action, Bid, Buy, Offer, Sell
from ..cards import SUITS, Counts, Suit
from ..game import Observation
from ..rules import GameConfig
from .base import Agent


class PassiveAgent(Agent):
    """Never trades. Its PnL is pure card luck, which makes it a useful zero point."""

    name = "passive"

    def act(self, obs: Observation) -> Action | None:
        return None


class RandomAgent(Agent):
    """Occasionally sends a random order at a random price."""

    name = "random"

    def __init__(self, activity: float = 0.1, max_price: int = 15) -> None:
        self.activity = activity
        self.max_price = max_price

    def act(self, obs: Observation) -> Action | None:
        if self.rng.random() > self.activity:
            return None
        s = self.rng.choice(SUITS)
        kind = self.rng.randrange(4)
        if kind == 0:
            return Bid(s, self.rng.randint(1, self.max_price))
        if kind == 1:
            return Offer(s, self.rng.randint(1, self.max_price)) if obs.hand[s] else None
        if kind == 2:
            return Buy(s)
        return Sell(s) if obs.hand[s] else None


class NaiveAgent(Agent):
    """A reasonable human heuristic with no probability model.

    Guesses the goal suit as the partner of its own longest suit, buys it up to a
    fixed price, and dumps everything else.
    """

    name = "naive"

    def __init__(self, max_buy: int = 8, bid_price: int = 5, min_sell: int = 2, ask_price: int = 4):
        self.max_buy = max_buy
        self.bid_price = bid_price
        self.min_sell = min_sell
        self.ask_price = ask_price

    def reset(self, player: int, config: GameConfig, hand: Counts, rng: random.Random) -> None:
        super().reset(player, config, hand, rng)
        longest = max(hand)
        self.goal = self.rng.choice([s for s in SUITS if hand[s] == longest]).partner

    def act(self, obs: Observation) -> Action | None:
        book, goal = obs.book, self.goal
        ask = book.asks[goal]
        if ask and ask.player != self.player and ask.price <= self.max_buy:
            return Buy(goal)
        for s in SUITS:
            bid = book.bids[s]
            if s != goal and obs.hand[s] and bid and bid.player != self.player:
                if bid.price >= self.min_sell:
                    return Sell(s)

        s = Suit(obs.tick % 4)
        if s == goal:
            bid = book.bids[goal]
            if bid is None or (bid.player != self.player and bid.price < self.bid_price):
                return Bid(goal, self.bid_price)
        elif obs.hand[s]:
            ask = book.asks[s]
            if ask is None or (ask.player != self.player and ask.price > self.ask_price):
                return Offer(s, self.ask_price)
        return None
