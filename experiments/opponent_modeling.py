"""Experiment 4: does the opponent model turn information into chips?

For each opponent field, plays the same deals twice: once with ``fundamental``
in the candidate seat, once with ``bayes``. Because the deals and seat rotations
are identical, the per-deal difference is a paired comparison.

    uv run python experiments/opponent_modeling.py --deals 60 --field naive,naive,naive
"""

from __future__ import annotations

import argparse
import sys

from openfiggie.agents import REGISTRY
from openfiggie.evaluation import TradeAnalyzer, estimate, run_duplicate
from openfiggie.rules import GameConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=60)
    parser.add_argument("--field", default="naive,naive,naive", help="the n-1 opponents")
    parser.add_argument("--candidates", default="fundamental,bayes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config = GameConfig()
    field = args.field.split(",")
    per_deal: dict[str, list[float]] = {}
    for cand in args.candidates.split(","):
        lineup = [(cand, REGISTRY[cand])] + [(name, REGISTRY[name]) for name in field]
        labels = [cand, *field]
        # Match run_duplicate's label disambiguation for repeated names.
        from openfiggie.evaluation.tournament import _unique_labels

        analyzer = TradeAnalyzer(config, _unique_labels(labels))
        result = run_duplicate(lineup, args.deals, config, args.seed, observer=analyzer)
        per_deal[cand] = [row[0] for row in result.scores]
        print(f"=== {cand} vs field {field}")
        print(result.table())
        print(analyzer.table())
        print()

    names = list(per_deal)
    if len(names) == 2:
        a, b = names
        diff = estimate([x - y for x, y in zip(per_deal[b], per_deal[a], strict=True)])
        print(f"paired difference ({b} - {a}) on identical deals: {diff}  (n={diff.n} deals)")


if __name__ == "__main__":
    main()
