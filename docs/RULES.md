# Figgie rules, and how OpenFiggie models them

Figgie is a card-trading game created at Jane Street to teach trading intuition.

## Standard rules

- **Deck.** 40 cards in four suits. One suit has 12 cards, one has 8, and two
  have 10. Which suit is which is secret; all 12 arrangements are equally likely.
- **Goal suit.** The suit of the *same color* as the 12-card suit (spades and
  clubs are black, hearts and diamonds red). The goal suit therefore always has
  8 or 10 cards.
- **Deal.** 4 players get 10 cards each (5 players get 8 each).
- **Ante.** Each player pays into a 200-chip pot (50 each with 4 players, 40 each
  with 5). Players start with 350 chips.
- **Trading.** For a fixed period (4 minutes in the real game) anyone may bid for
  or offer a card of any suit, and anyone may take a posted quote. Each suit has
  one best bid and one best offer; a new bid must beat the current best, and a
  new offer must undercut it. **Every trade clears all quotes in all suits.**
- **Settlement.** The goal suit is revealed. Each goal-suit card pays 10 chips
  from the pot. The rest of the pot (100 if the goal suit has 10 cards, 120 if
  8) goes to whoever holds the most goal-suit cards, split evenly on ties.

Figgie is zero-sum: the total PnL across players is always 0.

## Modeling choices in OpenFiggie

| Real game | OpenFiggie | Why |
|---|---|---|
| Continuous time, 4 minutes | `n_ticks` discrete ticks (default 240); each tick every player acts once, in a freshly shuffled order | Reproducible, fair across seats, keeps the "who gets there first" race |
| Humans may send many orders quickly | One action per player per tick | Keeps agents comparable; the random order still models latency races |
| Invalid orders are blocked by the UI | Invalid orders are ignored and counted in `rejected_actions` | Lets you spot buggy agents |
| Chips are integers | Chips integer during trading; the majority bonus may be fractional on a 3-way tie | Exact conservation of the pot |

Public information in each `Observation`: every player's chip count and hand
size, the full order book, and the full trade history (suit, price, buyer,
seller). Private: your own hand.
