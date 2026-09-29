"""The fundamental market maker, with beliefs from the opponent-model particle filter."""

from __future__ import annotations

import random

from ..cards import Counts
from ..game import Observation
from ..opponent_model import ParticleBelief
from ..rules import GameConfig
from ..valuation import SuitValues
from .fundamental import FundamentalAgent


class BayesAgent(FundamentalAgent):
    """Fundamental market maker whose beliefs also read opponents' quotes and trades."""

    name = "bayes"

    def __init__(
        self,
        n_particles: int = 400,
        refresh_ticks: int = 3,
        belief_kwargs: dict | None = None,
        trace_every: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_particles = n_particles
        self.refresh_ticks = refresh_ticks
        self.belief_kwargs = belief_kwargs or {}
        # If > 0, record (tick, particle goal probs, constraint-only goal probs) every N ticks.
        self.trace_every = trace_every

    def reset(self, player: int, config: GameConfig, hand: Counts, rng: random.Random) -> None:
        super().reset(player, config, hand, rng)
        self.belief = ParticleBelief(
            player, config, hand, rng, n_particles=self.n_particles, **self.belief_kwargs
        )
        self._dirty = True
        self._last_refresh = -self.refresh_ticks
        self._n_trades = 0
        self.trace: list[tuple[int, tuple[float, ...], tuple[float, ...]]] = []

    def compute_values(self, obs: Observation) -> SuitValues | None:
        self._dirty |= self.belief.update(obs)
        if self.trace_every and obs.tick % self.trace_every == 0:
            self.beliefs.update(obs)
            self.trace.append(
                (obs.tick, self.belief.goal_probabilities(), self.beliefs.goal_probabilities())
            )
        traded = len(obs.trades) != self._n_trades
        due = obs.tick - self._last_refresh >= self.refresh_ticks
        if self.values is not None and not (self._dirty and (traded or due)):
            return None
        self._dirty = False
        self._last_refresh = obs.tick
        self._n_trades = len(obs.trades)
        return self.belief.values(obs.hand)
