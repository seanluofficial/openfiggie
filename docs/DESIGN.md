# Design

```
src/openfiggie/
  cards.py        Suit, the 12 DeckConfigs, dealing (hands are per-suit count tuples)
  rules.py        GameConfig: players, chips, pot, payout, ticks
  actions.py      Bid, Offer, Buy, Sell, Cancel
  orderbook.py    best bid / best offer per suit
  game.py         Game: tick loop, validation, matching, settlement; Observation, GameResult
  inference.py    exact deck posterior + BeliefTracker (known holdings from trades)
  valuation.py    exact marginal card values (+ Monte Carlo reference)
  agents/         Agent interface, baselines, FundamentalAgent
  evaluation/     duplicate tournaments, versus-field matrix, statistics
  cli.py          openfiggie play | tournament | matrix | agents
experiments/      scripts that produce the paper's tables and figures
tests/            engine, inference, valuation, evaluation
```

## Principles

1. **Everything is reproducible.** A game is a pure function of its seed and its
   agents. Deal, turn order, and each agent's RNG are all derived from the seed,
   so any result in the paper can be replayed exactly (`test_games_replay_exactly`).
2. **The engine never trusts agents.** Every action is validated; invalid ones are
   dropped and counted. Chips and cards are conserved (`test_chips_and_cards_conserved`).
3. **Correctness is tested against brute force.** The posterior is checked against
   empirical deal frequencies; exact valuation against Monte Carlo.
4. **Pure Python, no dependencies.** Easy to install, easy to read.
   About 20 ms per game with the fundamental agent, so 10k-game experiments take minutes.

## Agent interface

```python
class MyAgent(Agent):
    name = "mine"

    def reset(self, player, config, hand, rng): ...  # once per game
    def act(self, obs: Observation) -> Action | None: ...  # once per tick
    def on_game_end(self, result: GameResult): ...  # optional
```

Register it in `agents/__init__.py:REGISTRY` to use it from the CLI.

## Inference (`inference.py`)

The prior over the 12 decks is uniform. Given your initial hand `h`,

```
P(deck | h)  ∝  ∏_s C(n_s, h_s)
```

where `n_s` is the suit's size in that deck. This is exact.

Trades add hard constraints. At any moment the cards in everyone's hands are
distinct, so `n_s ≥ my_current[s] + Σ_opponents known[q][s]`, where `known[q][s]`
counts cards `q` was seen buying and has not since sold. Decks violating a
bound are ruled out.

**Not yet modeled:** the *strategic* information in quotes and trades (someone
bidding hard for hearts probably holds diamonds). That is the week 4 opponent
model, and the main open research question.

**Finding:** from its own 10-card hand, a player puts on average only 0.29
probability on the true goal suit (uniform: 0.25). Most of the edge must come
from reading the market.

## Valuation (`valuation.py`)

A goal card pays 10 and may also swing the 100/120-chip majority bonus, so its
value depends on who else holds the goal suit. Model: given a deck and what is
known, the unseen cards are spread uniformly over opponents' free hand slots.
Only goal cards affect payoffs, so for each deck we enumerate every split
`u` of the `U` unseen goal cards over opponents with free slots `F_q`:

```
P(u) = ∏_q C(F_q, u_q) / C(ΣF, U)
```

and compute, per counterparty `q`,

- `buy_from[q][s]` = E[payoff gain from taking one `s` card from `q` | `q` holds one]
- `sell_to[q][s]`  = E[payoff loss from giving one `s` card to `q`]

These are exact under the model and cached (`_goal_split`), since the inputs are
small integers that recur. `sample_values` computes the same quantities by
dealing out full worlds; the tests check that the two agree.

Buy value can exceed sell value (convexity of the majority bonus), so an agent
can want to bid above its own offer. `FundamentalAgent` never quotes through
itself.

## Evaluation (`evaluation/`)

Card luck dominates any single game. **Duplicate evaluation** replays each deal
once per cyclic seat rotation with the same turn-order seed, so every lineup
entry plays every hand against the same opponents. An agent's score for a deal
is its mean over rotations; we report the mean and 95% CI over deals. Four
identical agents score exactly 0 on every deal (`test_duplicate_passive...`).

Compare two agents with `MatchResult.difference(i, j)`: a paired estimate on
shared deals, much tighter than two independent CIs.

`versus_field` builds the matrix "agent A against n−1 copies of B", which
separates "beats weak players" from "is hard to exploit".
