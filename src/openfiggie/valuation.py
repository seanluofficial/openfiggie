"""Card valuation: the marginal chip value of buying or selling one card.

A card is worth more than ``P(goal) * 10``: it can also swing the majority
bonus, and that swing depends on who else holds the goal suit. Values are
computed per counterparty, because buying a goal card from the current majority
holder is worth more than buying it from someone who is far behind.

Model: conditional on a deck and on what we know (our hand, plus cards
opponents are known to hold from trades), the unseen cards are spread uniformly
over opponents' remaining slots. Only the goal suit affects payoffs, so for
each deck we enumerate every split of the unseen goal cards across opponents,
weighted by the multivariate hypergeometric probability

    P(u_1..u_k) = prod_q C(F_q, u_q) / C(T, U)

where F_q are opponent q's free slots, T = sum F_q, and U the unseen goal cards.
This gives *exact* expectations under the model (``card_values``).
``sample_values`` estimates the same quantities by Monte Carlo and exists as an
independent check.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from math import comb

from .cards import NUM_SUITS, SUITS, DeckConfig
from .rules import GameConfig

NAN = math.nan


@dataclass(frozen=True)
class SuitValues:
    """Marginal chip values of one card, from one player's perspective.

    ``buy_from[q][s]``: expected gain from acquiring a card of suit ``s`` from
    player ``q``, conditional on ``q`` holding one. ``sell_to[q][s]``: expected
    loss from giving one to ``q``. ``buy[s]`` weights sellers by how many cards
    of ``s`` they hold; ``sell[s]`` averages buyers uniformly. NaN marks trades
    that are impossible (selling a suit you have none of, or self-trades).
    """

    buy: tuple[float, ...]
    sell: tuple[float, ...]
    buy_from: tuple[tuple[float, ...], ...]
    sell_to: tuple[tuple[float, ...], ...]


def goal_payoff(mine: int, others: Sequence[int], remainder: float, payout: int) -> float:
    """Settlement payoff for a player holding ``mine`` goal cards vs. opponents' ``others``."""
    best = max(others)
    value = float(payout * mine)
    if mine > best:
        value += remainder
    elif mine == best:
        value += remainder / (1 + sum(1 for o in others if o == mine))
    return value


def compositions(total: int, caps: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """All tuples ``u`` with ``sum(u) == total`` and ``0 <= u[i] <= caps[i]``."""
    if not caps:
        if total == 0:
            yield ()
        return
    rest = sum(caps[1:])
    for first in range(max(0, total - rest), min(caps[0], total) + 1):
        for tail in compositions(total - first, caps[1:]):
            yield (first, *tail)


def card_values(
    config: GameConfig,
    player: int,
    posterior: Mapping[DeckConfig, float],
    hand: Sequence[int],
    hand_sizes: Sequence[int],
    known: Sequence[Sequence[int]],
) -> SuitValues:
    """Exact marginal card values under the uniform-unseen-cards model (module docstring)."""
    n = config.n_players
    payout = config.goal_card_payout
    opps = [q for q in range(n) if q != player]
    k = len(opps)
    free = [hand_sizes[q] - sum(known[q]) for q in opps]
    total_free = sum(free)

    total_w = 0.0
    buy_num = [0.0] * NUM_SUITS
    bf_num = [[0.0] * NUM_SUITS for _ in range(k)]
    bf_den = [[0.0] * NUM_SUITS for _ in range(k)]
    st_num = [[0.0] * NUM_SUITS for _ in range(k)]

    for deck, w in posterior.items():
        counts = deck.counts
        unseen = [counts[s] - hand[s] - sum(known[q][s] for q in opps) for s in SUITS]
        if min(unseen) < 0 or sum(unseen) != total_free:
            continue
        total_w += w
        g = deck.goal_suit
        remainder = config.remainder(counts[g])

        # Non-goal suits have zero marginal value in this deck, but they still
        # carry weight in buy_from's conditioning on the seller holding the suit.
        for s in SUITS:
            if s == g:
                continue
            for i, q in enumerate(opps):
                bf_den[i][s] += w * _p_holds(known[q][s], free[i], unseen[s], total_free)

        bf, bd, st, b = _goal_split(
            hand[g], tuple(known[q][g] for q in opps), tuple(free), unseen[g], remainder, payout
        )
        for i in range(k):
            bf_num[i][g] += w * bf[i]
            bf_den[i][g] += w * bd[i]
            st_num[i][g] += w * st[i]
        buy_num[g] += w * b

    if total_w == 0:
        raise ValueError("no deck is consistent with the known holdings")

    buy_from = [[NAN] * NUM_SUITS for _ in range(n)]
    sell_to = [[NAN] * NUM_SUITS for _ in range(n)]
    for i, q in enumerate(opps):
        for s in SUITS:
            if bf_den[i][s] > 0:
                buy_from[q][s] = bf_num[i][s] / bf_den[i][s]
            if hand[s]:
                sell_to[q][s] = st_num[i][s] / total_w
    sell = tuple(sum(sell_to[q][s] for q in opps) / k if hand[s] else NAN for s in SUITS)
    return SuitValues(
        buy=tuple(b / total_w for b in buy_num),
        sell=sell,
        buy_from=tuple(map(tuple, buy_from)),
        sell_to=tuple(map(tuple, sell_to)),
    )


@lru_cache(maxsize=65536)
def _goal_split(
    mine: int,
    known_goal: tuple[int, ...],
    free: tuple[int, ...],
    unseen: int,
    remainder: float,
    payout: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    """Expectations over every split of the unseen goal cards, for one deck.

    Returns per-opponent E[delta * 1{holds}], P(holds), E[sell delta], and the
    holding-weighted E[buy delta]. Inputs are small integers that recur across
    decks and games, so this is cached.
    """
    k = len(free)
    bf, bd, st = [0.0] * k, [0.0] * k, [0.0] * k
    buy = 0.0
    denom = comb(sum(free), unseen)
    for u in compositions(unseen, free):
        p = math.prod(comb(free[i], u[i]) for i in range(k)) / denom
        others = [known_goal[i] + u[i] for i in range(k)]
        base = goal_payoff(mine, others, remainder, payout)
        weighted, weight = 0.0, 0
        for i in range(k):
            held = others[i]
            if held:
                others[i] -= 1
                d = goal_payoff(mine + 1, others, remainder, payout) - base
                others[i] += 1
                bf[i] += p * d
                bd[i] += p
                weighted += held * d
                weight += held
            if mine:
                others[i] += 1
                st[i] += p * (base - goal_payoff(mine - 1, others, remainder, payout))
                others[i] -= 1
        if weight:
            buy += p * weighted / weight
    return tuple(bf), tuple(bd), tuple(st), buy


def _p_holds(known: int, free: int, unseen: int, total_free: int) -> float:
    """P(an opponent holds >= 1 card of a suit), given known cards and uniform free slots."""
    if known:
        return 1.0
    return 1.0 - comb(total_free - unseen, free) / comb(total_free, free)


# -- Monte Carlo reference implementation ------------------------------------


def sample_values(
    rng: random.Random,
    config: GameConfig,
    player: int,
    posterior: Mapping[DeckConfig, float],
    hand: Sequence[int],
    hand_sizes: Sequence[int],
    known: Sequence[Sequence[int]],
    n_samples: int = 5000,
) -> SuitValues:
    """Monte Carlo estimate of ``card_values`` by dealing out complete worlds."""
    n = config.n_players
    opps = [q for q in range(n) if q != player]
    decks = list(posterior)
    cum = list(accumulate(posterior[d] for d in decks))
    payout = config.goal_card_payout
    mine = list(hand)

    sums: dict[str, list[list[float]]] = {
        key: [[0.0] * NUM_SUITS for _ in range(n)] for key in ("bf", "bfn", "st")
    }
    buy_sum, buy_n = [0.0] * NUM_SUITS, 0
    used = 0
    for _ in range(n_samples):
        deck = rng.choices(decks, cum_weights=cum)[0]
        world = _sample_world(rng, deck, mine, opps, hand_sizes, known)
        if world is None:
            continue
        used += 1
        g = deck.goal_suit
        remainder = config.remainder(deck.goal_count)
        others = [world[q][g] for q in opps]
        base = goal_payoff(mine[g], others, remainder, payout)
        weighted, weight = 0.0, 0
        for i, q in enumerate(opps):
            for s in SUITS:
                if world[q][s]:
                    sums["bfn"][q][s] += 1
            if others[i]:
                others[i] -= 1
                d = goal_payoff(mine[g] + 1, others, remainder, payout) - base
                others[i] += 1
                sums["bf"][q][g] += d
                weighted += (others[i]) * d
                weight += others[i]
            if mine[g]:
                others[i] += 1
                sums["st"][q][g] += base - goal_payoff(mine[g] - 1, others, remainder, payout)
                others[i] -= 1
        if weight:
            buy_sum[g] += weighted / weight
        buy_n += 1

    buy_from = [[NAN] * NUM_SUITS for _ in range(n)]
    sell_to = [[NAN] * NUM_SUITS for _ in range(n)]
    for q in opps:
        for s in SUITS:
            if sums["bfn"][q][s]:
                buy_from[q][s] = sums["bf"][q][s] / sums["bfn"][q][s]
            if mine[s]:
                sell_to[q][s] = sums["st"][q][s] / used
    sell = tuple(sum(sell_to[q][s] for q in opps) / len(opps) if mine[s] else NAN for s in SUITS)
    return SuitValues(
        buy=tuple(b / buy_n for b in buy_sum),
        sell=sell,
        buy_from=tuple(map(tuple, buy_from)),
        sell_to=tuple(map(tuple, sell_to)),
    )


def _sample_world(
    rng: random.Random,
    deck: DeckConfig,
    mine: Sequence[int],
    opps: Sequence[int],
    hand_sizes: Sequence[int],
    known: Sequence[Sequence[int]],
) -> dict[int, list[int]] | None:
    """Deal unseen cards into opponents' free slots. None if the deck is inconsistent."""
    counts = deck.counts
    unseen: list[int] = []
    for s in range(NUM_SUITS):
        k = counts[s] - mine[s] - sum(known[q][s] for q in opps)
        if k < 0:
            return None
        unseen.extend([s] * k)
    rng.shuffle(unseen)
    world: dict[int, list[int]] = {}
    i = 0
    for q in opps:
        hand = list(known[q])
        free = hand_sizes[q] - sum(hand)
        if free < 0:
            return None
        for s in unseen[i : i + free]:
            hand[s] += 1
        i += free
        world[q] = hand
    if i != len(unseen):
        return None
    return world
