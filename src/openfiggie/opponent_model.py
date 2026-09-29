"""Opponent modeling: a particle filter over the full hidden state.

``inference.BeliefTracker`` only uses hard facts (your hand, cards that change
hands). But quotes carry information: a player bidding 9 for hearts probably
holds a lot of diamonds. This module turns that into a posterior.

**Hidden state (one particle):** the deck, every opponent's *dealt* hand, and a
type per opponent, either INFORMED or NOISE. Opponents' current hands follow
from dealt hands plus public trade flow, so they are not sampled separately.

**Observation model.** For an informed player we assume quotes scatter around
what their *current* hand (dealt hand + public net trades at the time of the
quote) is worth to them: the exact ``card_values`` of that hand shape, with the
deck posterior computed from the same shape. (The holder's true posterior uses
their dealt hand; keying on the current shape alone keeps the cache small, a
few thousand shapes shared across games.) Quote prices are rounded the way
``FundamentalAgent`` rounds them: bids floor, offers ceil.

* bid price  ~ discrete logistic(mu = buy_value - half_spread, scale = tau),
  conditioned on beating the current best bid (a bid that does not improve the
  book is never posted, so only the truncated tail is observable);
* offer price ~ discrete logistic(mu = sell_value + half_spread, tau), conditioned
  on undercutting the best offer;
* taking a quote at price p: sigmoid((value - p) / tau) for a buy,
  sigmoid((p - value) / tau) for a sell.

A noise player quotes uniformly on [1, max_price] and takes with probability 1/2.
Both models are proper distributions over the observed price, so their
likelihood ratio is a fair test of "is this player informative?". An
``epsilon`` mix of the noise model keeps one odd quote from zeroing a hand.

Bots re-post the same quote after every trade, so events are strongly
correlated; the log-likelihood is multiplied by ``temper`` < 1 to avoid counting
the same evidence many times.

**Hard constraints.** Offering or selling requires holding the card, so every
opponent's dealt hand must satisfy ``dealt[s] >= 1 - net_bought[s]`` at the
moment of every offer and sale. Violating particles get zero weight.

**Inference.** Sequential importance sampling. When the effective sample size
drops below half, the filter resamples and then applies MCMC moves (swap one
card between two opponents' dealt hands; flip an opponent's type), accepted by
Metropolis-Hastings on the full-history likelihood. Both proposals leave the
uniform-deal prior invariant, so acceptance is the likelihood ratio alone.
Per-(opponent, hand) log-likelihoods are memoized and caught up lazily, which
makes the moves cheap.

A third move changes the deck: relabel two opponents' cards of suit ``a`` as
suit ``b``, moving to one of the 4 decks whose counts differ by -2/+2. With
``d`` the deck counts and ``h`` our hand, the card-choice and hand-prior terms
cancel, leaving the acceptance ratio

    (d_b+2)(d_b+1) / (d_a(d_a-1))  *  C(d_a-h_a, 2) / C(d_b+2-h_b, 2)  *  L'/L

which is exactly P(h | d') / P(h | d) times the likelihood ratio. Without this
move, a deck whose particles all die could never come back.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from itertools import accumulate
from math import comb

from .actions import Side
from .cards import ALL_DECKS, DECK_SIZE, NUM_SUITS, SUITS, Counts, DeckConfig
from .game import Observation, QuoteEvent, Trade
from .inference import deck_posterior
from .rules import GameConfig
from .valuation import SuitValues, card_values, goal_payoff

INFORMED, NOISE = 0, 1
_BID, _ASK, _TAKE_BUY, _TAKE_SELL = range(4)
Event = tuple[int, int, int, "int | None", tuple[int, ...]]


@lru_cache(maxsize=65536)
def hand_values(config: GameConfig, hand: Counts) -> tuple[tuple[float, ...], ...]:
    """(buy, sell) values of a hand shape, from its holder's point of view."""
    n = config.n_players
    # Spread the remaining cards as evenly as possible over the other players.
    rest, extra = divmod(DECK_SIZE - sum(hand), n - 1)
    sizes = (sum(hand), *(rest + (i < extra) for i in range(n - 1)))
    try:
        v = card_values(config, 0, deck_posterior(hand), hand, sizes, [[0] * 4] * n)
    except ValueError:  # a shape no deck allows (only reachable from infeasible particles)
        flat = (float(config.goal_card_payout),) * NUM_SUITS
        return flat, flat
    # A holder with none of a suit has no sell value; if they later offer one
    # (bought mid-game), treat their reservation price as the buy value.
    sell = tuple(b if math.isnan(x) else x for b, x in zip(v.buy, v.sell, strict=True))
    return v.buy, sell


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class ParticleBelief:
    def __init__(
        self,
        player: int,
        config: GameConfig,
        hand: Counts,
        rng: random.Random,
        n_particles: int = 400,
        tau: float = 2.0,
        half_spread: float = 2.0,
        temper: float = 0.08,
        epsilon: float = 0.05,
        max_price: int = 30,
        n_moves: int = 3,
        p_informed: float = 0.5,
    ) -> None:
        self.player = player
        self.config = config
        self.rng = rng
        self.tau, self.half_spread, self.temper = tau, half_spread, temper
        self.epsilon, self.max_price, self.n_moves = epsilon, max_price, n_moves
        self.initial_hand = hand
        self.opps = [q for q in range(config.n_players) if q != player]
        k = len(self.opps)

        # Public state reconstructed from the event stream.
        self.net = [[0] * NUM_SUITS for _ in range(k)]  # net cards bought, per opponent
        self.req = [[0] * NUM_SUITS for _ in range(k)]  # lower bound on dealt counts
        self._best_bid: list[int | None] = [None] * NUM_SUITS
        self._best_ask: list[int | None] = [None] * NUM_SUITS
        # Per opponent: (kind, suit, price, best price before, net holdings at the time).
        self._events: list[list[Event]] = [[] for _ in range(k)]
        self._noise_ll = [0.0] * k
        self._memo: list[dict[Counts, list[float]]] = [{} for _ in range(k)]
        self._n_quotes = self._n_trades = 0
        self.n_resamples = 0

        # Particles: deck, dealt opponent hands, opponent types, log weight, cached log-lik.
        post = deck_posterior(hand)
        decks = list(post)
        cum = list(accumulate(post[d] for d in decks))
        self.decks: list[DeckConfig] = []
        self.hands: list[list[Counts]] = []
        self.types: list[list[int]] = []
        for _ in range(n_particles):
            deck = rng.choices(decks, cum_weights=cum)[0]
            self.decks.append(deck)
            self.hands.append(self._deal_opponents(deck))
            self.types.append([INFORMED if rng.random() < p_informed else NOISE for _ in range(k)])
        self.logw = [0.0] * n_particles
        self._ll = [0.0] * n_particles

    # -- public API -------------------------------------------------------

    def update(self, obs: Observation) -> bool:
        """Consume new quotes and trades. Returns True if anything new was seen."""
        quotes = obs.quotes[self._n_quotes :]
        trades = obs.trades[self._n_trades :]
        if not quotes and not trades:
            return False
        self._n_quotes, self._n_trades = len(obs.quotes), len(obs.trades)
        events: list[QuoteEvent | Trade] = [*quotes, *trades]
        events.sort(key=lambda e: e.seq)
        for e in events:
            if isinstance(e, Trade):
                self._on_trade(e)
            else:
                self._on_quote(e)
        self._reweight()
        if max(self.logw) == -math.inf:
            self._reinitialize()
        elif self.ess() < len(self.logw) / 2:
            self._resample_move()
        return True

    def weights(self) -> list[float]:
        m = max(self.logw)
        if m == -math.inf:
            return [1.0 / len(self.logw)] * len(self.logw)
        w = [math.exp(x - m) for x in self.logw]
        total = sum(w)
        return [x / total for x in w]

    def ess(self) -> float:
        w = self.weights()
        return 1.0 / sum(x * x for x in w)

    def goal_probabilities(self) -> tuple[float, ...]:
        probs = [0.0] * NUM_SUITS
        for deck, w in zip(self.decks, self.weights(), strict=True):
            probs[deck.goal_suit] += w
        return tuple(probs)

    def type_probabilities(self) -> dict[int, float]:
        """P(opponent is INFORMED), keyed by player id."""
        w = self.weights()
        return {
            q: sum(wi for wi, t in zip(w, self.types, strict=True) if t[i] == INFORMED)
            for i, q in enumerate(self.opps)
        }

    def values(self, hand: Counts) -> SuitValues:
        """Marginal card values averaged over weighted particles (cf. ``valuation``)."""
        n = self.config.n_players
        payout = self.config.goal_card_payout
        k = len(self.opps)
        mine = list(hand)
        bf = [[0.0] * NUM_SUITS for _ in range(k)]
        bfw = [[0.0] * NUM_SUITS for _ in range(k)]
        st = [[0.0] * NUM_SUITS for _ in range(k)]
        buy = [0.0] * NUM_SUITS
        total = 0.0
        for deck, dealt, w in zip(self.decks, self.hands, self.weights(), strict=True):
            if w < 1e-9:
                continue
            total += w
            cur = [[dealt[i][s] + self.net[i][s] for s in SUITS] for i in range(k)]
            g = deck.goal_suit
            remainder = self.config.remainder(deck.goal_count)
            others = [c[g] for c in cur]
            base = goal_payoff(mine[g], others, remainder, payout)
            weighted, weight = 0.0, 0
            for i in range(k):
                for s in SUITS:
                    if cur[i][s] > 0:
                        bfw[i][s] += w
                held = others[i]
                if held > 0:
                    others[i] -= 1
                    d = goal_payoff(mine[g] + 1, others, remainder, payout) - base
                    others[i] += 1
                    bf[i][g] += w * d
                    weighted += held * d
                    weight += held
                if mine[g]:
                    others[i] += 1
                    st[i][g] += w * (base - goal_payoff(mine[g] - 1, others, remainder, payout))
                    others[i] -= 1
            if weight:
                buy[g] += w * weighted / weight

        nan = math.nan
        buy_from = [[nan] * NUM_SUITS for _ in range(n)]
        sell_to = [[nan] * NUM_SUITS for _ in range(n)]
        for i, q in enumerate(self.opps):
            for s in SUITS:
                if bfw[i][s] > 0:
                    buy_from[q][s] = bf[i][s] / bfw[i][s]
                if mine[s]:
                    sell_to[q][s] = st[i][s] / total
        sell = tuple(sum(sell_to[q][s] for q in self.opps) / k if mine[s] else nan for s in SUITS)
        return SuitValues(
            buy=tuple(b / total for b in buy),
            sell=sell,
            buy_from=tuple(map(tuple, buy_from)),
            sell_to=tuple(map(tuple, sell_to)),
        )

    # -- event processing -------------------------------------------------

    def _on_quote(self, e: QuoteEvent) -> None:
        s = e.suit
        book = self._best_bid if e.side is Side.BID else self._best_ask
        if e.price is None:
            book[s] = None
            return
        if e.player != self.player:
            i = self.opps.index(e.player)
            kind = _BID if e.side is Side.BID else _ASK
            if kind == _ASK:
                self.req[i][s] = max(self.req[i][s], 1 - self.net[i][s])
            self._add_event(i, (kind, s, e.price, book[s], tuple(self.net[i])))
        book[s] = e.price

    def _on_trade(self, t: Trade) -> None:
        s = t.suit
        if t.seller != self.player:
            i = self.opps.index(t.seller)
            self.req[i][s] = max(self.req[i][s], 1 - self.net[i][s])
            if t.aggressor == t.seller:
                self._add_event(i, (_TAKE_SELL, s, t.price, None, tuple(self.net[i])))
            self.net[i][s] -= 1
        if t.buyer != self.player:
            i = self.opps.index(t.buyer)
            if t.aggressor == t.buyer:
                self._add_event(i, (_TAKE_BUY, s, t.price, None, tuple(self.net[i])))
            self.net[i][s] += 1
        self._best_bid = [None] * NUM_SUITS
        self._best_ask = [None] * NUM_SUITS

    def _add_event(self, i: int, event: Event) -> None:
        self._events[i].append(event)
        self._noise_ll[i] += self._noise_loglik(event)

    # -- likelihoods ------------------------------------------------------

    def _noise_loglik(self, event: Event) -> float:
        kind, _, _, best, _ = event
        if kind in (_TAKE_BUY, _TAKE_SELL):
            return math.log(0.5)
        # Uniform over prices that would have improved the book.
        if best is None:
            n = self.max_price
        elif kind == _BID:
            n = max(1, self.max_price - best)
        else:
            n = max(1, best - 1)
        return -math.log(n)

    def _informed_loglik(self, event: Event, dealt: Counts) -> float:
        kind, s, price, best, net = event
        current = tuple(max(0, dealt[x] + net[x]) for x in SUITS)
        buy, sell = hand_values(self.config, current)  # type: ignore[arg-type]
        tau = self.tau
        if kind == _TAKE_BUY:
            p = _sigmoid((buy[s] - price) / tau)
        elif kind == _TAKE_SELL:
            p = _sigmoid((price - sell[s]) / tau)
        else:
            if kind == _BID:
                mu = buy[s] - self.half_spread - 0.5  # bids are floored
                # Mass at `price`, conditioned on price > best (i.e. price >= best + 1).
                mass = _sigmoid((price + 0.5 - mu) / tau) - _sigmoid((price - 0.5 - mu) / tau)
                tail = 1.0 - _sigmoid((best + 0.5 - mu) / tau) if best is not None else 1.0
            else:
                mu = sell[s] + self.half_spread + 0.5  # offers are ceiled
                mass = _sigmoid((price + 0.5 - mu) / tau) - _sigmoid((price - 0.5 - mu) / tau)
                tail = _sigmoid((best - 0.5 - mu) / tau) if best is not None else 1.0
            p = mass / max(tail, 1e-12)
        noise = math.exp(self._noise_loglik(event))
        return math.log((1 - self.epsilon) * min(p, 1.0) + self.epsilon * noise)

    def _hand_loglik(self, i: int, hand: Counts, kind: int) -> float:
        if kind == NOISE:
            return self._noise_ll[i]
        entry = self._memo[i].get(hand)
        if entry is None:
            entry = self._memo[i][hand] = [0.0, 0]
        events = self._events[i]
        if entry[1] < len(events):
            ll = entry[0]
            for e in events[int(entry[1]) :]:
                ll += self._informed_loglik(e, hand)
            entry[0], entry[1] = ll, len(events)
        return entry[0]

    def _feasible(self, i: int, hand: Counts) -> bool:
        req = self.req[i]
        return all(hand[s] >= req[s] for s in SUITS)

    def _particle_loglik(self, j: int) -> float:
        total = 0.0
        for i, hand in enumerate(self.hands[j]):
            if not self._feasible(i, hand):
                return -math.inf
            total += self._hand_loglik(i, hand, self.types[j][i])
        return self.temper * total

    def _reweight(self) -> None:
        for j in range(len(self.logw)):
            new = self._particle_loglik(j)
            old = self._ll[j]
            if new == -math.inf:
                self.logw[j] = -math.inf
            elif self.logw[j] != -math.inf:
                self.logw[j] += new - old
            self._ll[j] = new

    # -- resample-move ----------------------------------------------------

    def _resample_move(self) -> None:
        self.n_resamples += 1
        n = len(self.logw)
        w = self.weights()
        cum = list(accumulate(w))
        u0 = self.rng.random() / n
        idx, j = [], 0
        for m in range(n):
            u = u0 + m / n
            while j < n - 1 and cum[j] < u:
                j += 1
            idx.append(j)
        self.decks = [self.decks[j] for j in idx]
        self.hands = [list(self.hands[j]) for j in idx]
        self.types = [list(self.types[j]) for j in idx]
        self._ll = [self._ll[j] for j in idx]
        self.logw = [0.0] * n
        for j in range(n):
            for _ in range(self.n_moves):
                self._move(j)

    def _move(self, j: int) -> None:
        rng = self.rng
        k = len(self.opps)
        hands, types = self.hands[j], self.types[j]
        r = rng.random()
        if r < 0.15:
            self._deck_move(j)
            return
        if r < 0.3:
            i = rng.randrange(k)
            new_t = 1 - types[i]
            delta = self._hand_loglik(i, hands[i], new_t) - self._hand_loglik(i, hands[i], types[i])
            if math.log(rng.random() + 1e-300) < self.temper * delta:
                types[i] = new_t
                self._ll[j] += self.temper * delta
            return
        i1, i2 = rng.sample(range(k), 2)
        a = _random_card(rng, hands[i1])
        b = _random_card(rng, hands[i2])
        if a == b:
            return
        h1, h2 = list(hands[i1]), list(hands[i2])
        h1[a] -= 1
        h1[b] += 1
        h2[b] -= 1
        h2[a] += 1
        n1, n2 = tuple(h1), tuple(h2)
        if not (self._feasible(i1, n1) and self._feasible(i2, n2)):
            return
        old = self._hand_loglik(i1, hands[i1], types[i1]) + self._hand_loglik(
            i2, hands[i2], types[i2]
        )
        new = self._hand_loglik(i1, n1, types[i1]) + self._hand_loglik(i2, n2, types[i2])
        if math.log(rng.random() + 1e-300) < self.temper * (new - old):
            hands[i1], hands[i2] = n1, n2  # type: ignore[assignment]
            self._ll[j] += self.temper * (new - old)

    def _deck_move(self, j: int) -> None:
        """Relabel two opponents' cards of suit a as suit b (see module docstring)."""
        rng = self.rng
        deck, hands, types = self.decks[j], self.hands[j], self.types[j]
        d, h = deck.counts, self.initial_hand
        neighbors = _DECK_NEIGHBORS[deck]
        new_deck, a, b = neighbors[rng.randrange(len(neighbors))]
        if d[a] - h[a] < 2:
            return
        # Pick two distinct opponent cards of suit a, uniformly, and relabel them.
        new = [list(x) for x in hands]
        for _ in range(2):
            r = rng.randrange(sum(x[a] for x in new))
            for x in new:
                r -= x[a]
                if r < 0:
                    x[a] -= 1
                    x[b] += 1
                    break
        new_t: list[Counts] = [tuple(x) for x in new]  # type: ignore[misc]
        changed = [i for i in range(len(hands)) if new_t[i] != hands[i]]
        if not all(self._feasible(i, new_t[i]) for i in changed):
            return
        log_prior = (
            math.log((d[b] + 2) * (d[b] + 1))
            - math.log(d[a] * (d[a] - 1))
            + math.log(comb(d[a] - h[a], 2))
            - math.log(comb(d[b] + 2 - h[b], 2))
        )
        old_ll = sum(self._hand_loglik(i, hands[i], types[i]) for i in changed)
        new_ll = sum(self._hand_loglik(i, new_t[i], types[i]) for i in changed)
        if math.log(rng.random() + 1e-300) < log_prior + self.temper * (new_ll - old_ll):
            self.decks[j] = new_deck
            for i in changed:
                hands[i] = new_t[i]
            self._ll[j] += self.temper * (new_ll - old_ll)

    def _reinitialize(self, max_tries: int = 200) -> None:
        """Every particle was ruled out: redraw from the prior, subject to the hard constraints."""
        self.n_reinits = getattr(self, "n_reinits", 0) + 1
        post = deck_posterior(self.initial_hand)
        decks = list(post)
        cum = list(accumulate(post[d] for d in decks))
        for j in range(len(self.logw)):
            for _ in range(max_tries):
                deck = self.rng.choices(decks, cum_weights=cum)[0]
                hands = self._deal_opponents(deck)
                if all(self._feasible(i, h) for i, h in enumerate(hands)):
                    self.decks[j], self.hands[j] = deck, hands
                    break
            self._ll[j] = self._particle_loglik(j)
            self.logw[j] = self._ll[j]

    def _deal_opponents(self, deck: DeckConfig) -> list[Counts]:
        counts = deck.counts
        unseen = [s for s in SUITS for _ in range(counts[s] - self.initial_hand[s])]
        self.rng.shuffle(unseen)
        size = self.config.hand_size
        out = []
        for i in range(len(self.opps)):
            c = [0] * NUM_SUITS
            for s in unseen[i * size : (i + 1) * size]:
                c[s] += 1
            out.append(tuple(c))
        return out  # type: ignore[return-value]


def _neighbors(deck: DeckConfig) -> list[tuple[DeckConfig, int, int]]:
    """Decks reachable by moving 2 cards from suit a to suit b, as (deck, a, b)."""
    out = []
    for other in ALL_DECKS:
        diff = [o - c for o, c in zip(other.counts, deck.counts, strict=True)]
        if sorted(diff) == [-2, 0, 0, 2]:
            out.append((other, diff.index(-2), diff.index(2)))
    return out


_DECK_NEIGHBORS = {deck: _neighbors(deck) for deck in ALL_DECKS}


def _random_card(rng: random.Random, hand: Counts) -> int:
    r = rng.randrange(sum(hand))
    for s in SUITS:
        r -= hand[s]
        if r < 0:
            return s
    raise AssertionError("unreachable")
