"""Suits, the 12 possible deck configurations, and dealing.

Cards within a suit are interchangeable, so hands are represented as per-suit
count tuples indexed by ``Suit``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3

    @property
    def letter(self) -> str:
        return "SCHD"[self]

    @property
    def partner(self) -> Suit:
        """The other suit of the same color (spades<->clubs, hearts<->diamonds)."""
        return Suit(self ^ 1)


SUITS: tuple[Suit, ...] = tuple(Suit)
NUM_SUITS = 4
DECK_SIZE = 40

Counts = tuple[int, int, int, int]


@dataclass(frozen=True)
class DeckConfig:
    """One of the 12 equally likely decks: a 12-card suit, an 8-card suit, two 10-card suits."""

    long_suit: Suit
    short_suit: Suit

    @property
    def counts(self) -> Counts:
        c = [10, 10, 10, 10]
        c[self.long_suit] = 12
        c[self.short_suit] = 8
        return tuple(c)  # type: ignore[return-value]

    @property
    def goal_suit(self) -> Suit:
        """The goal suit is the same color as the 12-card suit."""
        return self.long_suit.partner

    @property
    def goal_count(self) -> int:
        return self.counts[self.goal_suit]

    def __str__(self) -> str:
        long, short, goal = self.long_suit.letter, self.short_suit.letter, self.goal_suit.letter
        return f"Deck(long={long}, short={short}, goal={goal})"


ALL_DECKS: tuple[DeckConfig, ...] = tuple(
    DeckConfig(long, short) for long in SUITS for short in SUITS if short != long
)


def deal(
    rng: random.Random, n_players: int, deck: DeckConfig | None = None
) -> tuple[DeckConfig, list[Counts]]:
    """Choose a deck uniformly (unless given), shuffle, and deal ``40 / n_players`` cards each."""
    if DECK_SIZE % n_players:
        raise ValueError(f"cannot deal {DECK_SIZE} cards evenly to {n_players} players")
    if deck is None:
        deck = rng.choice(ALL_DECKS)
    cards = [s for s in SUITS for _ in range(deck.counts[s])]
    rng.shuffle(cards)
    hand_size = DECK_SIZE // n_players
    hands: list[Counts] = []
    for p in range(n_players):
        c = [0, 0, 0, 0]
        for s in cards[p * hand_size : (p + 1) * hand_size]:
            c[s] += 1
        hands.append(tuple(c))  # type: ignore[arg-type]
    return deck, hands
