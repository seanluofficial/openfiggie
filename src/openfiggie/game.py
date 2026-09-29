"""The Figgie game engine: dealing, the tick loop, order matching, and settlement.

A game is fully determined by ``seed`` (plus the agents' own behavior), so any
result can be replayed exactly. Pass ``deck`` and ``hands`` to fix the deal,
which duplicate evaluation uses to replay the same cards with seats rotated.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .actions import Action, Bid, Buy, Cancel, Offer, Sell, Side
from .cards import SUITS, Counts, DeckConfig, Suit, deal
from .orderbook import BookSnapshot, OrderBook, Quote
from .rules import GameConfig

if TYPE_CHECKING:
    from .agents.base import Agent


@dataclass(frozen=True)
class Trade:
    tick: int
    suit: Suit
    price: int
    buyer: int
    seller: int
    aggressor: int  # the player who crossed the spread (buyer or seller)
    seq: int = 0  # position in the combined quote/trade event stream


@dataclass(frozen=True)
class QuoteEvent:
    """A quote that was posted (``price`` set) or cancelled (``price`` None)."""

    tick: int
    player: int
    suit: Suit
    side: Side
    price: int | None
    seq: int = 0


@dataclass(frozen=True)
class Observation:
    """Everything a player can see when acting. All fields except ``hand`` are public."""

    player: int
    tick: int
    config: GameConfig
    hand: Counts
    chips: tuple[int, ...]
    hand_sizes: tuple[int, ...]
    book: BookSnapshot
    trades: tuple[Trade, ...]
    quotes: tuple[QuoteEvent, ...]


@dataclass(frozen=True)
class GameResult:
    seed: int
    deck: DeckConfig
    initial_hands: tuple[Counts, ...]
    final_hands: tuple[Counts, ...]
    trades: tuple[Trade, ...]
    quotes: tuple[QuoteEvent, ...]
    chips_before_settlement: tuple[int, ...]
    payouts: tuple[float, ...]
    pnl: tuple[float, ...]
    rejected_actions: tuple[int, ...]


def settle(config: GameConfig, deck: DeckConfig, hands: Sequence[Counts]) -> tuple[float, ...]:
    """Pot payouts: 10 per goal card, remainder to the goal-suit majority (split on ties)."""
    goal = deck.goal_suit
    payouts = [float(config.goal_card_payout * h[goal]) for h in hands]
    most = max(h[goal] for h in hands)
    winners = [p for p, h in enumerate(hands) if h[goal] == most]
    bonus = config.remainder(deck.goal_count) / len(winners)
    for p in winners:
        payouts[p] += bonus
    return tuple(payouts)


class Game:
    def __init__(
        self,
        agents: Sequence[Agent],
        config: GameConfig | None = None,
        seed: int = 0,
        deck: DeckConfig | None = None,
        hands: Sequence[Counts] | None = None,
    ) -> None:
        self.config = config or GameConfig()
        if len(agents) != self.config.n_players:
            raise ValueError(f"need {self.config.n_players} agents, got {len(agents)}")
        self.agents = list(agents)
        self.seed = seed

        if hands is None:
            deck, hands = deal(random.Random(f"deal-{seed}"), self.config.n_players, deck)
        elif deck is None:
            raise ValueError("a fixed deal needs its deck")
        if any(sum(h) != self.config.hand_size for h in hands):
            raise ValueError("hand sizes do not match the config")
        self.deck = deck
        self.initial_hands = tuple(tuple(h) for h in hands)

        self._order_rng = random.Random(f"order-{seed}")
        self.hands = [list(h) for h in hands]
        self.chips = [self.config.starting_chips - self.config.ante] * self.config.n_players
        self.book = OrderBook()
        self.trades: list[Trade] = []
        self._trades_view: tuple[Trade, ...] = ()
        self._seq = 0
        self.quotes: list[QuoteEvent] = []
        self._quotes_view: tuple[QuoteEvent, ...] = ()
        self.rejected = [0] * self.config.n_players
        self.tick = 0

    # -- public API -------------------------------------------------------

    def play(self) -> GameResult:
        for p, agent in enumerate(self.agents):
            rng = random.Random(f"agent-{self.seed}-{p}")
            agent.reset(p, self.config, self.initial_hands[p], rng)

        players = list(range(self.config.n_players))
        for tick in range(self.config.n_ticks):
            self.tick = tick
            self._order_rng.shuffle(players)
            for p in players:
                action = self.agents[p].act(self.observe(p))
                if action is not None and not self.apply(p, action):
                    self.rejected[p] += 1

        final_hands = tuple(tuple(h) for h in self.hands)
        payouts = settle(self.config, self.deck, final_hands)
        start = self.config.starting_chips
        result = GameResult(
            seed=self.seed,
            deck=self.deck,
            initial_hands=self.initial_hands,
            final_hands=final_hands,  # type: ignore[arg-type]
            trades=tuple(self.trades),
            quotes=tuple(self.quotes),
            chips_before_settlement=tuple(self.chips),
            payouts=payouts,
            pnl=tuple(c + pay - start for c, pay in zip(self.chips, payouts, strict=True)),
            rejected_actions=tuple(self.rejected),
        )
        for agent in self.agents:
            agent.on_game_end(result)
        return result

    def observe(self, p: int) -> Observation:
        return Observation(
            player=p,
            tick=self.tick,
            config=self.config,
            hand=tuple(self.hands[p]),  # type: ignore[arg-type]
            chips=tuple(self.chips),
            hand_sizes=tuple(sum(h) for h in self.hands),
            book=self.book.snapshot(),
            trades=self._trades_view,
            quotes=self._quotes_view,
        )

    def apply(self, p: int, action: Action) -> bool:
        """Apply one action for player ``p``. Returns False if it was invalid (and ignored)."""
        book = self.book
        if isinstance(action, Bid):
            s, price = action.suit, action.price
            if price < 1 or price > self.chips[p]:
                return False
            ask = book.asks[s]
            if ask is not None and price >= ask.price:
                if ask.player == p:
                    return False
                return self._trade(s, ask.price, buyer=p, seller=ask.player, aggressor=p)
            bid = book.bids[s]
            if bid is not None and price <= bid.price:
                return False
            book.bids[s] = Quote(p, price)
            self._log_quote(p, s, Side.BID, price)
            return True

        if isinstance(action, Offer):
            s, price = action.suit, action.price
            if price < 1 or self.hands[p][s] < 1:
                return False
            bid = book.bids[s]
            if bid is not None and price <= bid.price:
                if bid.player == p:
                    return False
                return self._trade(s, bid.price, buyer=bid.player, seller=p, aggressor=p)
            ask = book.asks[s]
            if ask is not None and price >= ask.price:
                return False
            book.asks[s] = Quote(p, price)
            self._log_quote(p, s, Side.ASK, price)
            return True

        if isinstance(action, Buy):
            ask = book.asks[action.suit]
            if ask is None or ask.player == p:
                return False
            return self._trade(action.suit, ask.price, buyer=p, seller=ask.player, aggressor=p)

        if isinstance(action, Sell):
            bid = book.bids[action.suit]
            if bid is None or bid.player == p:
                return False
            return self._trade(action.suit, bid.price, buyer=bid.player, seller=p, aggressor=p)

        if isinstance(action, Cancel):
            quotes = book.bids if action.side is Side.BID else book.asks
            q = quotes[action.suit]
            if q is None or q.player != p:
                return False
            quotes[action.suit] = None
            self._log_quote(p, action.suit, action.side, None)
            return True

        return False

    # -- internals --------------------------------------------------------

    def _log_quote(self, p: int, suit: Suit, side: Side, price: int | None) -> None:
        self._seq += 1
        self.quotes.append(QuoteEvent(self.tick, p, suit, side, price, self._seq))
        self._quotes_view = tuple(self.quotes)

    def _trade(self, suit: Suit, price: int, buyer: int, seller: int, aggressor: int) -> bool:
        if self.chips[buyer] < price or self.hands[seller][suit] < 1:
            return False
        self.chips[buyer] -= price
        self.chips[seller] += price
        self.hands[buyer][suit] += 1
        self.hands[seller][suit] -= 1
        self._seq += 1
        self.trades.append(Trade(self.tick, suit, price, buyer, seller, aggressor, self._seq))
        self._trades_view = tuple(self.trades)
        self.book.clear()
        return True


def format_result(result: GameResult, names: Sequence[str]) -> str:
    """Human-readable summary of a finished game."""
    lines = [f"seed={result.seed}  {result.deck}  trades={len(result.trades)}"]
    goal = result.deck.goal_suit
    for p, name in enumerate(names):
        hand = " ".join(f"{s.letter}{result.final_hands[p][s]}" for s in SUITS)
        lines.append(
            f"  P{p} {name:<14} final[{hand}] goal={result.final_hands[p][goal]:>2}"
            f"  pnl={result.pnl[p]:+8.2f}"
        )
    return "\n".join(lines)
