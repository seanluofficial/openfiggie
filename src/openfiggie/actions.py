"""Actions an agent may return from ``Agent.act``. Returning ``None`` passes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cards import Suit


class Side(Enum):
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True)
class Bid:
    """Post a bid. Must beat the current best bid; trades immediately if it crosses the ask."""

    suit: Suit
    price: int


@dataclass(frozen=True)
class Offer:
    """Post an offer. Must beat the current best offer; trades immediately if it crosses the bid."""

    suit: Suit
    price: int


@dataclass(frozen=True)
class Buy:
    """Lift the best offer in a suit."""

    suit: Suit


@dataclass(frozen=True)
class Sell:
    """Hit the best bid in a suit."""

    suit: Suit


@dataclass(frozen=True)
class Cancel:
    """Withdraw your own quote on one side of a suit."""

    suit: Suit
    side: Side


Action = Bid | Offer | Buy | Sell | Cancel
