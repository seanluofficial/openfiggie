"""Figgie's order book: at most one best bid and one best offer per suit.

Matching and validation live in ``game.Game``; this module only stores quotes.
Per the rules, any trade clears every quote in every suit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import NUM_SUITS


@dataclass(frozen=True)
class Quote:
    player: int
    price: int


@dataclass(frozen=True)
class BookSnapshot:
    bids: tuple[Quote | None, ...]
    asks: tuple[Quote | None, ...]


class OrderBook:
    def __init__(self) -> None:
        self.bids: list[Quote | None] = [None] * NUM_SUITS
        self.asks: list[Quote | None] = [None] * NUM_SUITS

    def clear(self) -> None:
        self.bids = [None] * NUM_SUITS
        self.asks = [None] * NUM_SUITS

    def snapshot(self) -> BookSnapshot:
        return BookSnapshot(tuple(self.bids), tuple(self.asks))
