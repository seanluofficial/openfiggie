# OpenFiggie

An open simulator, agent suite, and evaluation harness for **Figgie**, the
card-trading game created at Jane Street. Figgie packs hidden information, a
live order book, a nonlinear payoff, and 4–5 strategic players into a
four-minute game, which makes it a compact benchmark for multi-agent trading.

- **Exact Bayesian inference** over the 12 possible decks, checked against brute
  force and calibrated
- **Exact card valuation:** the marginal value of each card, including its effect
  on the majority bonus, per counterparty, via hypergeometric enumeration
- **A market-making agent** built on both
- **Duplicate evaluation:** every agent plays every hand from every seat, with
  paired confidence intervals
- Deterministic, dependency-free Python; about 20 ms per game

## Results so far

200 deals × 4 seat rotations, mean chips per game, 95% CI:

| agent | PnL / game |
|---|---|
| `fundamental`: exact posterior + exact values, market making | **+29.8** [+26.5, +33.2] |
| `naive`: guesses goal from longest suit, fixed prices | +16.3 [+13.4, +19.2] |
| `passive`: never trades | −14.5 [−16.2, −12.8] |
| `random`: random orders | −31.7 [−34.8, −28.6] |

In progress: `bayes` adds a particle-filter opponent model that reads order
flow. It makes goal-suit beliefs sharper while keeping them calibrated
(mean P(true goal) 0.32 → 0.39 by tick 160), and it is ahead of `fundamental` in
every opponent field tested so far, though not yet significantly. See the
[roadmap](docs/ROADMAP.md).

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run openfiggie play -v                     # one game, every trade printed
uv run openfiggie tournament --deals 200      # duplicate evaluation
uv run openfiggie matrix --deals 50           # each agent vs. a field of each other
uv run openfiggie analyze --deals 100         # + luck/trading split, maker/taker markouts
uv run openfiggie agents                      # list agents
uv run pytest                                 # tests
uv run python experiments/calibration.py      # posterior calibration
uv run python experiments/baselines.py        # headline tables -> results/
uv run python experiments/belief_quality.py   # does reading order flow sharpen beliefs?
uv run python experiments/opponent_modeling.py --field naive,naive,naive
```

Choose a lineup with `--agents fundamental,naive,naive,random`; `--players 5` for
the 5-player game.

## Writing an agent

```python
from openfiggie.agents import Agent
from openfiggie import Bid, Buy, Suit


class HeartsLover(Agent):
    name = "hearts"

    def act(self, obs):
        ask = obs.book.asks[Suit.HEARTS]
        if ask and ask.player != self.player and ask.price <= 6:
            return Buy(Suit.HEARTS)
        return Bid(Suit.HEARTS, 3)
```

Add it to `REGISTRY` in `src/openfiggie/agents/__init__.py` to use it from the CLI.

## Docs

- [Rules and modeling choices](docs/RULES.md)
- [Design: architecture, inference, valuation, evaluation](docs/DESIGN.md)
- [Roadmap](docs/ROADMAP.md)
- [Paper outline](docs/paper/OUTLINE.md)

## License

MIT
