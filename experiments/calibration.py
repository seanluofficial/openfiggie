"""Experiment 1: is the deck posterior calibrated?

Deals random games, computes each player's goal-suit probabilities from their
own hand, and bins predictions against outcomes. Writes a CSV and, if
matplotlib is installed (``uv sync --extra plot``), a reliability diagram.

    uv run python experiments/calibration.py --deals 50000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from openfiggie.cards import SUITS, deal
from openfiggie.inference import deck_posterior, goal_probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=50_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/calibration"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    cache: dict = {}
    bins = [[0, 0.0, 0] for _ in range(args.bins)]  # count, sum of p, hits
    brier, p_true_total, n_preds = 0.0, 0.0, 0
    for _ in range(args.deals):
        deck, hands = deal(rng, 4)
        for hand in hands:
            if hand not in cache:
                cache[hand] = goal_probabilities(deck_posterior(hand))
            probs = cache[hand]
            p_true_total += probs[deck.goal_suit]
            for s in SUITS:
                hit = int(s == deck.goal_suit)
                b = bins[min(int(probs[s] * args.bins), args.bins - 1)]
                b[0] += 1
                b[1] += probs[s]
                b[2] += hit
                brier += (probs[s] - hit) ** 2
            n_preds += 1

    with open(args.out / "reliability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "n", "mean_predicted", "observed_freq"])
        for i, (n, psum, hits) in enumerate(bins):
            if n:
                w.writerow([i / args.bins, (i + 1) / args.bins, n, psum / n, hits / n])

    print(f"hands evaluated:          {n_preds}")
    print(f"mean P(true goal suit):   {p_true_total / n_preds:.4f}  (uniform prior: 0.25)")
    print(f"Brier score (4-way):      {brier / n_preds:.4f}  (uniform prior: 0.75)")
    print(f"{'bin':>12} {'n':>9} {'predicted':>10} {'observed':>10}")
    for i, (n, psum, hits) in enumerate(bins):
        if n:
            lo, hi = i / args.bins, (i + 1) / args.bins
            print(f"[{lo:.1f}, {hi:.1f}) {n:>9} {psum / n:>10.3f} {hits / n:>10.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install the 'plot' extra for a reliability diagram)")
        return
    xs = [b[1] / b[0] for b in bins if b[0]]
    ys = [b[2] / b[0] for b in bins if b[0]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("predicted P(goal suit)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Deck posterior calibration")
    fig.tight_layout()
    fig.savefig(args.out / "reliability.png", dpi=200)
    print(f"wrote {args.out / 'reliability.png'}")


if __name__ == "__main__":
    main()
