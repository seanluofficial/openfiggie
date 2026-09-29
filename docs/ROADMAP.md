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

**Open problem this exposes:** against a *field* of `fundamental` agents, `naive`
earns about +10.7 ± 6.4 per game. `fundamental` ignores the information in other
players' bids, so it gets adversely selected. Phase 2 is aimed at this.

## Phase 2: opponent modeling (weeks 1–2 from now)

- [ ] **Order-flow likelihood.** Model P(quote or trade | opponent's hand) with a
      simple parametric policy (for example, a player bids for suit s with
      probability rising in their P(goal = s)), and fold it into the posterior over
      decks *and* opponents' holdings. Start with a logit "noisy fundamental" model.
- [ ] Particle filter over full hidden states (deck + every opponent's hand) as the
      general version; compare it with the exact constraint-only posterior.
- [ ] **Adverse-selection metric:** mean change in the agent's own fair value 1, 5
      and 20 trades after each fill. Report it per agent.
- [ ] Use it: widen quotes against informed flow, and stop selling the suit
      everyone is bidding for.
- [ ] **Opponent population:** noisy-fundamental agents at several noise levels,
      aggressive and passive variants, `naive` with different thresholds. Avoid
      tuning against a single opponent.
- Success criterion: the new agent is positive against *every* field in the matrix,
  including a field of itself (ideally about 0).

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
