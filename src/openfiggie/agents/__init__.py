"""Agents and a name -> factory registry used by the CLI and experiments."""

from collections.abc import Callable
from functools import partial

from .base import Agent
from .baselines import NaiveAgent, PassiveAgent, RandomAgent
from .bayes import BayesAgent
from .fundamental import FundamentalAgent

REGISTRY: dict[str, Callable[[], Agent]] = {}
DESCRIPTIONS: dict[str, str] = {}


def register(name: str, factory: Callable[[], Agent], description: str) -> None:
    REGISTRY[name] = factory
    DESCRIPTIONS[name] = description


register("passive", PassiveAgent, "never trades (zero point for card luck)")
register("random", RandomAgent, "random orders at random prices")
register("naive", NaiveAgent, "goal = partner of longest suit; fixed prices")
register("fundamental", FundamentalAgent, "exact posterior + exact values; ignores order flow")
register(
    "noisy",
    partial(FundamentalAgent, value_noise=4.0, activity=0.5),
    "fundamental with a per-game value bias (sd 4) acting half the time",
)
register(
    "aggressive",
    partial(FundamentalAgent, half_spread=1.0, take_edge=0.0),
    "fundamental with tight quotes that takes any positive edge",
)
register(
    "patient",
    partial(FundamentalAgent, half_spread=4.0, take_edge=3.0, activity=0.5),
    "fundamental with wide quotes, only takes large edges",
)
register("bayes", BayesAgent, "fundamental + particle-filter opponent model over order flow")


def make_agent(name: str) -> Agent:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(REGISTRY)}") from None


__all__ = [
    "Agent",
    "BayesAgent",
    "DESCRIPTIONS",
    "FundamentalAgent",
    "NaiveAgent",
    "PassiveAgent",
    "RandomAgent",
    "REGISTRY",
    "make_agent",
    "register",
]
