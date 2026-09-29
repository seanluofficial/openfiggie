# Paper outline

**Working title:** *OpenFiggie: A Benchmark for Multi-Agent Trading Under Imperfect Information*

**Type:** benchmark / environment paper with strong baselines. Target: a workshop on
multi-agent systems, games, or AI in finance; arXiv preprint first.

## Claims (each must be backed by an experiment in this repo)

1. Figgie is a compact but nontrivial testbed. It combines hidden information,
   a real order book with a unique clearing rule, a nonlinear (majority) payoff,
   and 4–5 strategic players. It is small enough to reason about exactly.
2. The deck posterior has a closed form and is calibrated → Exp. 1.
3. Card values under a uniform-unseen model can be computed *exactly* by
   enumerating goal-card splits; the majority bonus makes buy value ≠ sell value
   and makes value depend on the counterparty → Exp. 2, ablation.
4. Duplicate evaluation cuts variance by a factor of X compared with naive
   repeated play → Exp. 3 (measure it: CI width, same number of games).
5. Agents that ignore order-flow information are exploitable, even by a simple
   heuristic; opponent modeling closes the gap → Exp. 4 (the main result).
6. (Optional) LLM agents: where they fall in the agent suite → Exp. 5.

## Sections

1. **Introduction:** why trading games; why Figgie; contributions list
2. **Related work:** imperfect-information games (poker: CFR, Libratus/Pluribus;
   Hanabi), multi-agent market simulators (ABIDES, JAX-LOB), opponent modeling,
   game benchmarks (OpenSpiel, PettingZoo). *To do: find any prior Figgie work.*
3. **The game and the environment:** rules, tick model, observation/action API,
   determinism, performance
4. **Inference and valuation:** posterior derivation; trade constraints;
   exact valuation via multivariate hypergeometric enumeration; complexity
5. **Agents:** baselines; fundamental market maker; opponent-modeling agent
6. **Evaluation methodology:** duplicate play, paired differences, versus-field matrix
7. **Experiments:** Exp. 1–5, ablations, 5-player results
8. **Limitations:** discrete ticks; simple opponent policy model; no equilibrium claims
9. **Conclusion and open problems:** equilibrium approximation, RL, human data

## Tables and figures

- T1: headline lineup results (mean PnL, 95% CI)
- T2: versus-field matrix
- T3: ablations
- F1: reliability diagram (`experiments/calibration.py`)
- F2: CI width vs. number of games, duplicate vs. non-duplicate
- F3: fair value and trades over one annotated game
- F4: adverse-selection curves per agent

## Numbers already in hand

- A 10-card hand puts 0.292 average probability on the true goal suit
  (uniform: 0.25); 4-way Brier score 0.708 (uniform: 0.75). Predicted vs. observed
  within 0.015 in every bin with n > 1,000 (50k deals).
- Fundamental +29.8, naive +16.3, passive −14.5, random −31.7 chips/game
  (200 deals × 4 rotations).
- Naive vs. a field of fundamental: +10.7 ± 6.4 → motivates opponent modeling.
