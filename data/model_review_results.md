# Model review follow-up — November 7, 2026

The seven confirmed bug groups are fixed: owned-player values in rolling simulations,
deadline selection, incomplete double-gameweek caches, discarded solver lineups,
wildcard and terminal free-transfer accounting, the optional coach's headline field,
and default history fetching / empty-data errors. Regression checks run in the weekly
GitHub Actions workflow before refreshing data.

Evaluation now separates training (GW2–6), epoch-selection validation (GW7–8), and a
test (GW9) excluded from epoch selection. Constant points, minutes, and haul baselines
are fitted on training rows only. The deployed model is subsequently refit on all rows.
Solver status, optimality gap, and candidate-pool scope are exposed in squad results.

## Recomputed results

All runs below use the existing local API cache; no fresh API data or LLM calls were used.

| GW3–9 strategy | Previously saved | Corrected, coach enabled | Corrected, coach disabled |
| --- | ---: | ---: | ---: |
| Model rolling manager, after hits | 340 | 345 | 356 |
| Naive rolling manager, after hits | 283 | 270 | 270 |
| Model weekly rebuild | 409 | 409 | 367 |
| Naive weekly rebuild | 352 | 352 | 352 |
| Official FPL average total | 377 | 377 | 377 |
| Model player MAE | 1.687 | 1.687 | 1.692 |
| Weekly rebuild projection bias | +12.1 | +12.1 | +16.8 |

The coach ablation improves rolling-manager totals but worsens weekly rebuilds and
prediction calibration. Seven weeks do not justify selecting a default based on those
totals. The existing coach default is retained. Its behavior remains testable with
`python3 backtest.py --no-coach --no-save`.

The separate GW9 test has 447 player rows: model MAE **1.4868**, naive MAE **1.6706**,
RMSE **2.3330**. Haul Brier is **0.0801**, versus **0.0794** for the training-rate baseline.
This test uses a different training/split protocol from the previously reported GW8–9
validation score, so their MAEs are not a before/after measure of model improvement.

## Remaining modeling work

- Squad projections remain too optimistic. Compare a mean-targeting loss with Huber
  inside chronological validation, then assess captain and selected-XI calibration on
  future weeks; do not tune corrections against the test results above.
- Historical roster, club membership, fixture difficulty and set-piece metadata come
  from the current cache. Fully point-in-time evaluation requires archived snapshots.
- Selling prices still assume current prices; chips beyond wildcard-style transfers
  are not simulated. Candidate pruning and solver time limits prevent any guarantee
  of a full-league optimum.

Validation: `python3 -m unittest -v test_regressions`, training, both GW3–9 backtests,
static page generation, and the squad-report pipeline completed successfully.
