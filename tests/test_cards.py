import random

from openfiggie.cards import ALL_DECKS, DECK_SIZE, SUITS, Suit, deal


def test_twelve_decks_each_forty_cards():
    assert len(ALL_DECKS) == 12
    assert len(set(ALL_DECKS)) == 12
    for deck in ALL_DECKS:
        assert sum(deck.counts) == DECK_SIZE
        assert sorted(deck.counts) == [8, 10, 10, 12]


def test_goal_suit_is_same_color_as_long_suit():
    for deck in ALL_DECKS:
        assert deck.goal_suit == deck.long_suit.partner
        assert deck.goal_suit != deck.long_suit
        assert deck.goal_count in (8, 10)
    assert Suit.SPADES.partner == Suit.CLUBS
    assert Suit.HEARTS.partner == Suit.DIAMONDS


def test_each_suit_is_goal_in_three_decks():
    for s in SUITS:
        assert sum(d.goal_suit == s for d in ALL_DECKS) == 3


def test_deal_uses_every_card():
    for n in (4, 5):
        deck, hands = deal(random.Random(7), n)
        assert len(hands) == n
        assert all(sum(h) == DECK_SIZE // n for h in hands)
        assert tuple(sum(h[s] for h in hands) for s in SUITS) == deck.counts


def test_deal_is_deterministic():
    assert deal(random.Random(3), 4) == deal(random.Random(3), 4)
