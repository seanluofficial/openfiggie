"""Duplicate evaluation.

Card luck dominates a single Figgie game. Duplicate play removes most of it:
each deal is replayed once per cyclic rotation of the seats, so every agent in
the lineup plays every hand, against the same opponents, with the same turn
order. An agent's score for a deal is its mean PnL over the rotations; we then
report the mean and a CI over deals.

Because every lineup entry is scored on the same deals, differences between two
entries should be compared *paired* (``MatchResult.difference``), which is far
tighter than comparing two independent CIs.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..agents.base import Agent
from ..cards import deal
from ..game import Game, GameResult
from ..rules import GameConfig
from .stats import Estimate, estimate

AgentFactory = Callable[[], Agent]
# Called after every game with the result and seat_of[i] = seat of lineup entry i.
GameObserver = Callable[[GameResult, Sequence[int]], None]


@dataclass(frozen=True)
class MatchResult:
    labels: tuple[str, ...]
    # scores[d][i]: mean PnL of lineup entry i on deal d, averaged over rotations.
    scores: tuple[tuple[float, ...], ...]
    rejected: tuple[int, ...]
    n_games: int

    def estimate(self, i: int) -> Estimate:
        return estimate([row[i] for row in self.scores])

    def difference(self, i: int, j: int) -> Estimate:
        """Paired estimate of (entry i) - (entry j)."""
        return estimate([row[i] - row[j] for row in self.scores])

    def table(self) -> str:
        width = max(len(label) for label in self.labels) + 2
        lines = [
            f"{len(self.scores)} deals x {len(self.labels)} rotations = {self.n_games} games",
            f"{'agent':<{width}}{'mean pnl/game':>16}{'95% CI':>22}{'rejected':>10}",
        ]
        order = sorted(range(len(self.labels)), key=lambda i: -self.estimate(i).mean)
        for i in order:
            e = self.estimate(i)
            lo, hi = e.ci95
            lines.append(
                f"{self.labels[i]:<{width}}{e.mean:>+16.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>22}"
                f"{self.rejected[i]:>10}"
            )
        return "\n".join(lines)


def run_duplicate(
    lineup: Sequence[tuple[str, AgentFactory]],
    n_deals: int,
    config: GameConfig | None = None,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
    observer: GameObserver | None = None,
) -> MatchResult:
    """Play ``n_deals`` deals, each once per seat rotation. ``lineup`` has one entry per seat."""
    config = config or GameConfig()
    n = config.n_players
    if len(lineup) != n:
        raise ValueError(f"lineup needs exactly {n} entries")
    labels = _unique_labels([name for name, _ in lineup])

    scores: list[tuple[float, ...]] = []
    rejected = [0] * n
    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        deck, hands = deal(random.Random(f"deal-{deal_seed}"), n)
        totals = [0.0] * n
        for r in range(n):
            seat_of = [(i + r) % n for i in range(n)]
            agents: list[Agent] = [None] * n  # type: ignore[list-item]
            for i, (_, factory) in enumerate(lineup):
                agents[seat_of[i]] = factory()
            result = Game(agents, config, seed=deal_seed, deck=deck, hands=hands).play()
            if observer:
                observer(result, seat_of)
            for i in range(n):
                totals[i] += result.pnl[seat_of[i]]
                rejected[i] += result.rejected_actions[seat_of[i]]
        scores.append(tuple(t / n for t in totals))
        if progress:
            progress(d + 1)
    return MatchResult(tuple(labels), tuple(scores), tuple(rejected), n_deals * n)


def versus_field(
    pool: Sequence[tuple[str, AgentFactory]],
    n_deals: int,
    config: GameConfig | None = None,
    seed: int = 0,
) -> dict[tuple[str, str], Estimate]:
    """For each (candidate, opponent): candidate's PnL against n-1 copies of opponent."""
    config = config or GameConfig()
    out: dict[tuple[str, str], Estimate] = {}
    for cand_name, cand in pool:
        for opp_name, opp in pool:
            lineup = [(cand_name, cand)] + [(opp_name, opp)] * (config.n_players - 1)
            out[cand_name, opp_name] = run_duplicate(lineup, n_deals, config, seed).estimate(0)
    return out


def _unique_labels(names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    seen: dict[str, int] = {}
    labels = []
    for name in names:
        if counts[name] == 1:
            labels.append(name)
        else:
            seen[name] = seen.get(name, 0) + 1
            labels.append(f"{name}#{seen[name]}")
    return labels
