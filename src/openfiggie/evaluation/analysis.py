"""Where does an agent's PnL come from?

Two decompositions, both exact per game:

* **luck vs. trading.** ``luck`` is what the player's *dealt* hand would have
  paid if nobody traded (minus the ante); ``trading`` is everything else. Under
  duplicate evaluation luck averages out to the same for every lineup entry, so
  trading PnL is the skill signal.
* **maker vs. taker markouts.** Each trade is marked to its hindsight goal-card
  value (10 if the suit was the goal suit, else 0). A maker's quote was hit by
  someone else; a taker crossed the spread. Negative maker markouts mean the
  agent's quotes are being picked off by better-informed players: adverse
  selection. (Markouts ignore the majority bonus, so they do not sum to PnL.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..game import GameResult, settle
from ..rules import GameConfig
from .stats import Estimate, estimate


@dataclass
class AgentStats:
    luck: list[float] = field(default_factory=list)
    trading: list[float] = field(default_factory=list)
    maker: list[float] = field(default_factory=list)
    taker: list[float] = field(default_factory=list)
    trades_per_game: list[int] = field(default_factory=list)


class TradeAnalyzer:
    """Pass as ``observer`` to ``run_duplicate``; then print ``table()``."""

    def __init__(self, config: GameConfig, labels: Sequence[str]) -> None:
        self.config = config
        self.labels = list(labels)
        self.stats = {label: AgentStats() for label in labels}

    def __call__(self, result: GameResult, seat_of: Sequence[int]) -> None:
        dealt = settle(self.config, result.deck, result.initial_hands)
        goal = result.deck.goal_suit
        payout = self.config.goal_card_payout
        for i, label in enumerate(self.labels):
            p = seat_of[i]
            st = self.stats[label]
            luck = dealt[p] - self.config.ante
            st.luck.append(luck)
            st.trading.append(result.pnl[p] - luck)
            n = 0
            for t in result.trades:
                if p not in (t.buyer, t.seller):
                    continue
                n += 1
                value = payout if t.suit == goal else 0
                edge = value - t.price if t.buyer == p else t.price - value
                (st.taker if t.aggressor == p else st.maker).append(edge)
            st.trades_per_game.append(n)

    def summary(self, label: str) -> dict[str, Estimate]:
        st = self.stats[label]
        return {
            "luck": estimate(st.luck),
            "trading": estimate(st.trading),
            "maker": estimate(st.maker),
            "taker": estimate(st.taker),
        }

    def table(self) -> str:
        width = max(len(label) for label in self.labels) + 2
        head = f"{'agent':<{width}}{'trading pnl':>16}{'trades/g':>10}"
        head += f"{'maker edge':>16}{'(n)':>8}{'taker edge':>16}{'(n)':>8}"
        lines = [head]
        for label in self.labels:
            st = self.stats[label]
            s = self.summary(label)
            tpg = sum(st.trades_per_game) / max(1, len(st.trades_per_game))
            lines.append(
                f"{label:<{width}}{_fmt(s['trading']):>16}{tpg:>10.1f}"
                f"{_fmt(s['maker']):>16}{len(st.maker):>8}{_fmt(s['taker']):>16}{len(st.taker):>8}"
            )
        lines.append("edges: chips/trade vs. hindsight goal-card value; ± = 95% CI half-width")
        return "\n".join(lines)


def _fmt(e: Estimate) -> str:
    if e.n == 0:
        return "-"
    if e.n == 1:
        return f"{e.mean:+.2f}"
    return f"{e.mean:+.2f}±{1.96 * e.stderr:.2f}"
