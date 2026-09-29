"""Game parameters. Defaults follow the standard Figgie rules (see docs/RULES.md)."""

from __future__ import annotations

from dataclasses import dataclass

from .cards import DECK_SIZE


@dataclass(frozen=True)
class GameConfig:
    n_players: int = 4
    starting_chips: int = 350
    goal_card_payout: int = 10
    pot_size: int = 200
    # Continuous-time trading is modeled as discrete ticks; each tick every
    # player acts once, in a freshly shuffled order.
    n_ticks: int = 240

    def __post_init__(self) -> None:
        if self.n_players not in (4, 5):
            raise ValueError("Figgie is played with 4 or 5 players")
        if self.pot_size % self.n_players:
            raise ValueError("pot must split evenly into antes")

    @property
    def ante(self) -> int:
        return self.pot_size // self.n_players

    @property
    def hand_size(self) -> int:
        return DECK_SIZE // self.n_players

    def remainder(self, goal_count: int) -> int:
        """Pot left for the majority bonus after paying every goal card."""
        return self.pot_size - self.goal_card_payout * goal_count
