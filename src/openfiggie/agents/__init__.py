"""Agents and a name -> factory registry used by the CLI and experiments."""

from collections.abc import Callable

from .base import Agent
from .baselines import NaiveAgent, PassiveAgent, RandomAgent
from .fundamental import FundamentalAgent

REGISTRY: dict[str, Callable[[], Agent]] = {
    "passive": PassiveAgent,
    "random": RandomAgent,
    "naive": NaiveAgent,
    "fundamental": FundamentalAgent,
}


def make_agent(name: str) -> Agent:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(REGISTRY)}") from None


__all__ = [
    "Agent",
    "FundamentalAgent",
    "NaiveAgent",
    "PassiveAgent",
    "RandomAgent",
    "REGISTRY",
    "make_agent",
]
