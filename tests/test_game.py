import pytest

from openfiggie import Bid, Buy, Cancel, Game, GameConfig, Offer, Sell, Side, Suit
from openfiggie.agents import FundamentalAgent, NaiveAgent, PassiveAgent, RandomAgent
from openfiggie.cards import DeckConfig
from openfiggie.game import settle

H, S = Suit.HEARTS, Suit.SPADES
# Goal = hearts (long suit diamonds), hearts has 10 cards.
DECK = DeckConfig(long_suit=Suit.DIAMONDS, short_suit=Suit.SPADES)
HANDS = [(2, 3, 3, 2), (2, 2, 3, 3), (2, 3, 2, 3), (2, 2, 2, 4)]


def make_game(**kw) -> Game:
    return Game([PassiveAgent() for _ in range(4)], deck=DECK, hands=HANDS, **kw)


def test_settlement_pays_goal_cards_and_majority():
    config = GameConfig()
    payouts = settle(config, DECK, [(0, 0, 4, 0), (0, 0, 3, 0), (0, 0, 2, 0), (0, 0, 1, 0)])
    assert payouts == (40 + 100, 30, 20, 10)


def test_settlement_splits_ties():
    config = GameConfig()
    payouts = settle(config, DECK, [(0, 0, 3, 0), (0, 0, 3, 0), (0, 0, 3, 0), (0, 0, 1, 0)])
    assert payouts == pytest.approx((30 + 100 / 3,) * 3 + (10,))


def test_bid_must_improve_and_crossing_trades():
    g = make_game()
    assert g.apply(0, Bid(H, 5))
    assert not g.apply(1, Bid(H, 5))
    assert g.apply(1, Bid(H, 6))
    assert g.apply(2, Offer(H, 6))  # crosses player 1's bid -> trade at 6
    assert len(g.trades) == 1
    t = g.trades[0]
    assert (t.buyer, t.seller, t.price, t.suit) == (1, 2, 6, H)
    assert g.hands[1][H] == 4 and g.hands[2][H] == 1


def test_trade_clears_every_quote():
    g = make_game()
    g.apply(0, Bid(S, 3))
    g.apply(1, Offer(H, 9))
    g.apply(2, Buy(H))
    assert all(q is None for q in g.book.bids + g.book.asks)


def test_invalid_actions_rejected():
    g = make_game()
    assert not g.apply(0, Offer(S, 5)) or g.hands[0][S] > 0
    g.hands[0][S] = 0
    assert not g.apply(0, Offer(S, 5))  # no card to sell
    assert not g.apply(0, Buy(H))  # nothing offered
    assert not g.apply(0, Sell(H))  # nothing bid
    assert not g.apply(0, Bid(H, 0))
    assert not g.apply(0, Bid(H, 10_000))  # more than chips
    g.apply(1, Offer(H, 4))
    assert not g.apply(1, Bid(H, 5))  # would self-trade
    assert not g.apply(0, Cancel(H, Side.ASK))  # not our quote
    assert g.apply(1, Cancel(H, Side.ASK))


def test_chips_and_cards_conserved_and_zero_sum():
    agents = [FundamentalAgent(), NaiveAgent(), RandomAgent(activity=0.5), RandomAgent()]
    result = Game(agents, seed=11).play()
    assert len(result.trades) > 0
    total_cards = tuple(sum(h[s] for h in result.final_hands) for s in range(4))
    assert total_cards == result.deck.counts
    assert sum(result.chips_before_settlement) == 4 * (350 - 50)
    assert sum(result.pnl) == pytest.approx(0)


def test_games_replay_exactly():
    def run():
        agents = [FundamentalAgent(), NaiveAgent(), RandomAgent(), RandomAgent()]
        return Game(agents, seed=5).play()

    assert run() == run()


def test_five_player_game():
    config = GameConfig(n_players=5, n_ticks=60)
    agents = [FundamentalAgent(), NaiveAgent(), RandomAgent(), RandomAgent(), PassiveAgent()]
    result = Game(agents, config, seed=2).play()
    assert all(sum(h) == 8 for h in result.initial_hands)
    assert sum(result.pnl) == pytest.approx(0)
