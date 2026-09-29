"""Command-line entry point: ``openfiggie play | tournament | matrix | agents``."""

from __future__ import annotations

import argparse
import sys
import time

from .agents import REGISTRY, make_agent
from .evaluation import run_duplicate, versus_field
from .game import Game, format_result
from .rules import GameConfig


def _agent_list(value: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()]
    for name in names:
        if name not in REGISTRY:
            choices = ", ".join(sorted(REGISTRY))
            raise argparse.ArgumentTypeError(f"unknown agent {name!r}; choose from {choices}")
    return names


def _config(args: argparse.Namespace) -> GameConfig:
    return GameConfig(n_players=args.players, n_ticks=args.ticks)


def cmd_play(args: argparse.Namespace) -> None:
    config = _config(args)
    _check_lineup(args.agents, config)
    result = Game([make_agent(n) for n in args.agents], config, seed=args.seed).play()
    if args.verbose:
        for t in result.trades:
            print(f"  t={t.tick:>3}  P{t.seller} -> P{t.buyer}  {t.suit.letter} @ {t.price}")
    print(format_result(result, args.agents))


def cmd_tournament(args: argparse.Namespace) -> None:
    config = _config(args)
    _check_lineup(args.agents, config)
    lineup = [(n, REGISTRY[n]) for n in args.agents]
    start = time.perf_counter()

    def progress(done: int) -> None:
        if done % max(1, args.deals // 20) == 0:
            print(f"\r  {done}/{args.deals} deals", end="", file=sys.stderr, flush=True)

    result = run_duplicate(lineup, args.deals, config, args.seed, progress)
    print(f"\r{' ' * 30}\r", end="", file=sys.stderr)
    print(result.table())
    print(f"({time.perf_counter() - start:.1f}s)")


def cmd_matrix(args: argparse.Namespace) -> None:
    config = _config(args)
    pool = [(n, REGISTRY[n]) for n in args.agents]
    table = versus_field(pool, args.deals, config, args.seed)
    width = max(len(n) for n in args.agents) + 2
    print("row agent's mean pnl/game vs. a field of the column agent (95% CI half-width)")
    print(" " * width + "".join(f"{n:>18}" for n in args.agents))
    for cand in args.agents:
        cells = "".join(f"{str(table[cand, opp]):>18}" for opp in args.agents)
        print(f"{cand:<{width}}{cells}")


def cmd_agents(args: argparse.Namespace) -> None:
    for name, factory in sorted(REGISTRY.items()):
        doc = (factory.__doc__ or "").strip().splitlines()
        print(f"{name:<14}{doc[0] if doc else ''}")


def _check_lineup(names: list[str], config: GameConfig) -> None:
    if len(names) != config.n_players:
        sys.exit(f"need {config.n_players} agents (one per seat), got {len(names)}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="openfiggie", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, default_agents: str) -> None:
        p.add_argument("--agents", type=_agent_list, default=_agent_list(default_agents))
        p.add_argument("--players", type=int, default=4, choices=(4, 5))
        p.add_argument("--ticks", type=int, default=GameConfig.n_ticks)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("play", help="play one game and print the result")
    common(p, "fundamental,naive,random,passive")
    p.add_argument("-v", "--verbose", action="store_true", help="print every trade")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("tournament", help="duplicate evaluation of one lineup")
    common(p, "fundamental,naive,random,passive")
    p.add_argument("--deals", type=int, default=200)
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("matrix", help="each agent vs. a field of each other agent")
    common(p, "fundamental,naive,random,passive")
    p.add_argument("--deals", type=int, default=50)
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("agents", help="list available agents")
    p.set_defaults(func=cmd_agents)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
