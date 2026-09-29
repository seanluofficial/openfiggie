"""Experiment 3: does reading order flow improve beliefs about the goal suit?

Plays games with a tracing BayesAgent in seat 0 and compares, at checkpoints
through the game, its particle-filter goal probabilities with the
constraint-only posterior (``inference.BeliefTracker``).

Metrics per checkpoint, averaged over games:
  * log score  E[log p(true goal)]   (higher is better; uniform = log 0.25 = -1.386)
  * mean p(true goal)
  * sharpness  E[sum_s p_s^2]; calibrated beliefs have sharpness == mean p(true)
    (sharpness above mean p(true) means overconfident)

    uv run python experiments/belief_quality.py --games 40 --field fundamental,naive,random
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

from openfiggie import Game
from openfiggie.agents import REGISTRY
from openfiggie.agents.bayes import BayesAgent
from openfiggie.rules import GameConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--field", default="fundamental,naive,random")
    parser.add_argument("--every", type=int, default=40)
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--temper", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    field = args.field.split(",")
    config = GameConfig()
    kwargs = {k: getattr(args, k) for k in ("temper", "tau", "epsilon") if getattr(args, k)}
    rows: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for g in range(args.games):
        bayes = BayesAgent(args.particles, trace_every=args.every, belief_kwargs=kwargs)
        agents = [bayes, *(REGISTRY[name]() for name in field)]
        result = Game(agents, config, seed=args.seed * 100_000 + g).play()
        goal = result.deck.goal_suit
        for tick, particle, baseline in bayes.trace:
            for name, probs in (("pf", particle), ("base", baseline)):
                p = probs[goal]
                rows[tick][f"{name}_log"].append(math.log(max(p, 1e-3)))
                rows[tick][f"{name}_p"].append(p)
                rows[tick][f"{name}_sharp"].append(sum(x * x for x in probs))

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    print(f"{args.games} games, seat 0 = bayes, field = {field}, belief kwargs = {kwargs}")
    print(f"{'tick':>5} | {'log score':>19} | {'mean p(true)':>15} | {'sharpness':>15}")
    print(f"{'':>5} | {'particle':>9} {'base':>9} | {'pf':>7} {'base':>7} | {'pf':>7} {'base':>7}")
    for tick in sorted(rows):
        r = rows[tick]
        print(
            f"{tick:>5} | {mean(r['pf_log']):>9.3f} {mean(r['base_log']):>9.3f} | "
            f"{mean(r['pf_p']):>7.3f} {mean(r['base_p']):>7.3f} | "
            f"{mean(r['pf_sharp']):>7.3f} {mean(r['base_sharp']):>7.3f}"
        )


if __name__ == "__main__":
    main()
