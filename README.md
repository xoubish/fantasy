# FPL AI Squad Optimizer & Transfer Planner

Python + static-web toolkit for Fantasy Premier League. It learns from real per-gameweek
outcomes, estimates minutes and points for every player over the next five gameweeks, and
optimizes squads and transfers within a shortlisted candidate pool using a mixed-integer program.

## What the model does

- **Multi-head neural network** (`neural_model.py`, numpy): one shared trunk with three heads
  trained on actual gameweek data: `P(0 / 1-59 / 60+ minutes)`, points given 60+ minutes, and
  `P(9+ point haul)`. Expected points are minutes-weighted. Match-log features come only from match
  rows **before** the gameweek being predicted, so training, live prediction and backtests share
  one code path. Historical roster and fixture metadata still use the current snapshot.
- **Honest evaluation**: the model is trained on gameweeks `<= K` and scored on the following
  gameweeks against real points, next to two naive baselines (prior play-rate x points per
  appearance, and a constant fitted only on training rows). Early stopping uses chronological
  validation weeks; later test weeks are evaluated once, after epoch selection. The page shows
  test results when available, and explicitly labels validation results otherwise.
- **Availability** (`news_scraper.py`): the learned minutes model combined with the official FPL
  status / chance-of-playing flags and any notes you tag to a player in `my_squad.json`.
  Scraped headlines are shown for information only; they no longer move probabilities.
- **Squad optimisation** (`optimizer.py`, scipy/HiGHS MILP): selects 15 under budget and club
  limits with lineup, captain and bench; six constrained squad archetypes; and a multi-gameweek
  **transfer planner** with free transfers banked to five, `-4` per extra transfer, and a
  friction margin so marginal swaps are skipped. Wildcards preserve banked transfers, and
  terminal value rewards transfers remaining after the final planned week. Results report
  solver status, optimality gap and candidate count; there is no full-league optimality guarantee.
- **Walk-forward backtest** (`backtest.py`): for each gameweek `T` from 3 onward it retrains on
  `< T` and scores real points with auto-subs and captaincy for four strategies: model and naive
  weekly rebuilds (unlimited transfers) and model and naive *realistic* managers (one free
  transfer a week), against the official FPL average. Results feed the page.

## Setup

Requires Python 3.10+ with `numpy`, `scipy` and `pillow`. On a Homebrew Python:

```bash
python3 -m pip install --user --break-system-packages numpy scipy pillow
```

No GPU is needed: training takes about one second on a laptop CPU.

## Quick start

```bash
python3 fpl_assistant.py sync            # live FPL API + per-player match logs
python3 fpl_assistant.py train-nn        # retrain, print separate test report vs baselines
python3 fpl_assistant.py best-squad      # £100m squad for the next editable gameweek
python3 fpl_assistant.py plan --horizon 4   # transfer plan for the squad in my_squad.json
python3 fpl_assistant.py analyze         # full markdown report incl. the plan (data/latest_report.md)
python3 backtest.py                      # walk-forward backtest, writes data/backtest_results.json
python3 web_app.py --serve               # build index.html and serve at http://localhost:8000
python3 -m unittest -v test_regressions   # offline regression checks
```

`my_squad.json` holds your 15 players, bank and free transfers. Add a tagged note under
`custom_news` (for example a manager quote saying a player needs rest) and it is applied to that
player's minutes.

`sync` refreshes every active player's full history, including zero-minute players, so doubles
and corrected bonus points are picked up. `--shallow` explicitly skips histories; `--deep`
remains accepted for compatibility. Recommendations target the next future deadline.

## Known limitations

- Selling prices are assumed equal to current prices (purchase prices are not tracked).
- Chips other than a wildcard-style "unlimited transfers this week" switch are not modelled.
- Historical roster, club membership, fixture difficulty, set-piece duties and last-season
  stats use the current API snapshot. Fully point-in-time backtests require archived snapshots.
- A handful of weeks is a small sample; a single gameweek score has a standard deviation of
  roughly 15 points, so read the backtest margins with their spread.
