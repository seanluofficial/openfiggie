"""The agent interface every bot implements."""

from __future__ import annotations

import random

from ..actions import Action
from ..cards import Counts
from ..game import GameResult, Observation
from ..rules import GameConfig


class Agent:
    """Base class for Figgie agents.

    The engine calls ``reset`` once per game, then ``act`` once per tick. An
    agent must not rely on state from previous games unless it means to learn.
    ``rng`` is seeded from the game seed and seat, so games replay exactly.
    """

    name = "agent"

    def reset(self, player: int, config: GameConfig, hand: Counts, rng: random.Random) -> None:
        self.player = player
        self.config = config
        self.rng = rng

    def act(self, obs: Observation) -> Action | None:
        raise NotImplementedError

    def on_game_end(self, result: GameResult) -> None:
        pass
