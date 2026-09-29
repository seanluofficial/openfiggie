"""Experiment 2: headline results table and versus-field matrix.

    uv run python experiments/baselines.py --deals 500

Writes results/baselines/{lineup,matrix}.csv. Every number is a duplicate-
evaluation mean PnL per game with a 95% CI over deals.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from openfiggie.agents import REGISTRY
from openfiggie.evaluation import run_duplicate, versus_field
from openfiggie.rules import GameConfig

POOL = ["fundamental", "naive", "random", "passive"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=500)
    parser.add_argument("--matrix-deals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/baselines"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    config = GameConfig()
    pool = [(n, REGISTRY[n]) for n in POOL]

    result = run_duplicate(pool, args.deals, config, args.seed)
    print(result.table())
    with open(args.out / "lineup.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "mean_pnl", "ci_lo", "ci_hi", "n_deals"])
        for i, label in enumerate(result.labels):
            e = result.estimate(i)
            w.writerow([label, e.mean, *e.ci95, e.n])

    print("\npaired differences vs. naive:")
    naive = result.labels.index("naive")
    for i, label in enumerate(result.labels):
        if i != naive:
            print(f"  {label:<12} - naive = {result.difference(i, naive)}")

    matrix = versus_field(pool, args.matrix_deals, config, args.seed)
    with open(args.out / "matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "field", "mean_pnl", "ci_lo", "ci_hi"])
        for (cand, opp), e in matrix.items():
            w.writerow([cand, opp, e.mean, *e.ci95])
    print(f"\nwrote {args.out}/lineup.csv and matrix.csv")


if __name__ == "__main__":
    main()
