# Roadmap

Plan written 2026-09-28. Guiding rule: **recruiting comes first.** Applications
go out now, not when the project is done. The project already supports a resume
line; everything below makes it stronger for interviews and for the paper.

## Phase 1: foundation (done)

- [x] Deterministic engine, order book, settlement; 4- and 5-player games
- [x] Baselines: `passive`, `random`, `naive`
- [x] Exact deck posterior, checked against brute force; calibration experiment
- [x] Exact card valuation with a per-counterparty majority bonus; Monte Carlo cross-check
- [x] `FundamentalAgent` market maker
- [x] Duplicate evaluation, paired differences, versus-field matrix, CLI
- [x] Test suite (engine, inference, valuation, evaluation)

**Current results** (200 deals × 4 rotations, 95% CI):

| agent | mean PnL / game |
|---|---|
| fundamental | +29.8 [+26.5, +33.2] |
| naive | +16.3 [+13.4, +19.2] |
| passive | −14.5 [−16.2, −12.8] |
| random | −31.7 [−34.8, −28.6] |

**Open problem this exposes:** `fundamental` ignores the information in other
players' bids, so it gets adversely selected: its maker markout is about −2.7
chips per fill against `naive`. (An early 40-deal matrix suggested `naive` earns
+10.7 against a field of `fundamental`; 60-deal runs did not reproduce a
significant effect. Always trust the larger run.)

## Phase 2: opponent modeling (in progress, branch `phase2-opponent-modeling`)

- [x] Engine: public quote log, trade aggressor, global event sequence numbers
- [x] Opponent population: `noisy`, `aggressive`, `patient` variants of `fundamental`
- [x] PnL decomposition (luck vs. trading) and maker/taker markouts (`openfiggie analyze`)
- [x] **Particle filter** over deck + dealt hands + opponent type (informed / noise),
      with truncated discrete-logistic quote likelihoods, hard holding constraints,
      and resample-move MCMC (card swap, type flip, and a deck-change move with a
      derived MH ratio). Invariant tests: moves preserve the exact prior; dead
      decks revive; cached likelihoods match recomputation.
- [x] Tempering sweep: `temper = 0.08` is calibrated
- [ ] **Significance:** ≥ 900 deals per field (needs the parallel runner below)
- [ ] **Policy, not just beliefs:** widen or skip quotes in suits with informed
      flow; quote one-sided when the posterior is confident; compare against
      `naive`'s zero-loss maker behavior
- [ ] Parallel tournament runner (`multiprocessing`); `bayes` runs about 0.8 s/game
- [ ] Speed up the particle filter (fewer particles plus better moves, or numpy)
- Success criterion: significantly positive against *every* field, including a
  field of itself (ideally about 0).

**Results so far** (60 deals × 4 rotations; ± = 95% CI half-width):

Belief quality at tick 160, field = fundamental, naive, random, 60 games:

| belief | log score | mean P(true goal) | sharpness |
|---|---|---|---|
| constraint-only | −1.216 | 0.318 | 0.300 |
| particle filter (temper 0.08) | **−1.136** | **0.392** | 0.392 |

At `temper = 0.3` the filter reached 0.48 mean P(true) but was overconfident
(sharpness 0.57) and had a worse log score than the baseline.

Candidate seat PnL on identical deals:

| field | fundamental | bayes | paired diff |
|---|---|---|---|
| 3× naive | +1.8 | +2.9 | +1.0 ± 5.9 |
| 3× fundamental | 0.0 | +0.8 | +0.8 ± 4.2 |
| 3× noisy | +9.6 | +12.4 | +2.7 ± 6.8 |
| fundamental + naive + noisy | +0.0 | +1.3 | +1.3 ± 5.9 |

Maker markout vs. 3× naive: fundamental −2.73 ± 0.44, bayes −2.23 ± 0.42;
taker markout −0.11 ± 0.29 → +0.39 ± 0.25. Directionally better everywhere,
not yet significant. Both market makers still lose 2–3 chips per maker fill,
which points at the quoting policy as the next lever.

## Phase 3: evaluation and robustness (week 3)

- [ ] Ablations: no majority term (value = 10·P(goal)), no per-counterparty values,
      no trade constraints, spread and edge sweeps, no opponent model
- [ ] 5-player results; sensitivity to `n_ticks`
- [ ] Bootstrap CIs in the paper tables; 2,000+ deals for headline numbers
- [ ] Parallel tournament runner (`multiprocessing`) if runs get slow
- [ ] Figures: reliability diagram, PnL-by-agent bars, matrix heatmap,
      fair value over time for one example game

## Phase 4: paper (weeks 4–6)

See [`paper/OUTLINE.md`](paper/OUTLINE.md).

- [ ] Literature search (Google Scholar / arXiv: "Figgie", multi-agent trading
      games, imperfect-information market games, opponent modeling in trading)
- [ ] **Find a faculty advisor.** Email 2–3 professors (CS, statistics, or
      operations research: games, multi-agent systems, RL, market microstructure)
      with a one-paragraph summary, the results table, and the repo link.
- [ ] Optional, timely: an LLM-agent adapter (describe the observation in text,
      parse an action) and benchmark a few models against the agent suite
- [ ] Stretch: self-play / RL fine-tuning on top of the structured agent
- [ ] Draft → advisor feedback → arXiv preprint (target: December 2026)
- [ ] Workshop submission, spring 2027 cycle (check each venue's current deadlines)

## Recruiting track (in parallel, starting now)

- [ ] Resume line + repo link on applications this week
- [ ] Clean, pinned GitHub repo with results in the README
- [ ] Be ready to explain in 60 seconds: posterior, why buy value ≠ sell value,
      duplicate evaluation, and the adverse-selection finding
- [ ] Interview prep takes priority over polish once interviews start
