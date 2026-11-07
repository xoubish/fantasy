# FPL historical predictor and transfer planner

The model trains on **completed historical seasons**, independently of today's player
roster. Current-season outcomes are excluded from training, model selection and
calibration. Current match logs supply recent form/minutes features and forward-test
outcomes; the live FPL roster determines who can be selected now.

## Data and reproducibility

`historical_data.py` imports four completed seasons (currently 2022–23 through 2025–26)
from [Vaastav Anand's FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League).
The source commit, file URLs, SHA256 hashes and row counts are recorded in
`data/historical_manifest.json`. Raw CSVs and the generated feature cache are ignored
by Git and can be restored with `python3 historical_data.py sync`. `--refresh` explicitly
updates the pinned source commit.

The normalized training set contains **113,260 player-match rows**. Ten identical
duplicates and 322 Assistant Manager chip rows are excluded; `GK` is normalized to
`GKP`. Conflicting duplicates fail validation. Player IDs are scoped to seasons;
only stable FPL codes link prior-season summaries. Historical players need not still
be in the Premier League to contribute training examples.

Match features use earlier gameweeks and kickoffs before the reconstructed deadline.
Club membership comes from the match fixture, not the player's end-of-season club.
The archive's `xP`, end-of-season FDR/strength and set-piece order fields are not model
inputs. Features distinguish missing defensive-contribution data and the post-2025
scoring regime. Earlier seasons retain meaningful training weight (0.65 per year).

## Training, calibration and tests

The numpy network has shared layers and separate minutes, conditional-points and haul
heads. `historical_training.py` compares Huber and squared-error point losses using
validation **RMSE**, then fits regularized point calibration and minutes temperature
on a separate period. The optimizer receives the resulting expected points. Coach
notes are informational; heuristic or LLM adjustments do not alter calibrated scores.

For each historical evaluation season:

1. Fit on all earlier seasons plus GW1–10 of that season.
2. Select loss/epoch count on GW11–14.
3. Fit calibration on GW15–18.
4. Evaluate GW19–38 without any further fitting or tuning.

`historical_backtest.py` tests 2024–25 and 2025–26: **40 historical test weeks**. It
reports player error, selected-player/captain calibration, realistic rolling-manager
scores and four-week block-bootstrap intervals against the same-planner naive
baseline. Official historical human averages are unavailable in this archive; no
historical claim against the human average is made.

After evaluation, deployed weights are refit on all completed seasons with the selected
loss/epoch count and frozen calibration. `data/historical_evaluation_weights.json`
retains the separate evaluation model. The final refit's calibration is monitored on
the current season, never tuned on it. `backtest.py` uses those fixed historical
weights for every current-season test week; it does not learn current-season targets.

## Squad and transfer decisions

The scipy/HiGHS optimizer respects squad composition, budgets, club limits, lineup and
captain constraints. Results preserve the solver's XI and report status, gap and
candidate count. Optimality applies to the shortlisted pool, not the whole league.

Transfers pay their actual hit cost, a fixed friction margin, and a conservative
estimation-risk margin for both outgoing and incoming players. This uses historical
calibration errors and the player's effective recent appearance count; it is a
decision buffer, **not a statistical confidence interval**. Free transfers bank to
five, wildcards preserve the bank, and terminal value counts transfers remaining
after the final planned moves. Owned players keep their price, club and position
even when they are not eligible purchase candidates.

## Setup and commands

Requires Python 3.10+, `numpy`, `scipy` and `pillow`.

```bash
python3 -m pip install numpy scipy pillow
python3 fpl_assistant.py sync          # live roster/form histories, historical archive, news
python3 historical_data.py sync       # restore just the pinned historical archive
python3 fpl_assistant.py train-nn      # historical loss comparison, calibration, test, deployment fit
python3 historical_backtest.py        # 40 historical strategy-test weeks
python3 backtest.py --no-coach         # fixed-model current-season test against FPL averages
python3 fpl_assistant.py analyze       # squad report and transfer plan
python3 web_app.py                    # rebuild index.html
python3 -m unittest -v test_regressions
```

Live `sync` refreshes full histories, including zero-minute players, to pick up doubles
and bonus corrections. `--shallow` skips current histories; `--deep` remains accepted.
Recommendations target the next editable deadline. `my_squad.json` stores the user's
squad, bank, free transfers and tagged availability notes.

Successful live syncs retain compressed, timestamped public-input snapshots in
`data/snapshots/`, including per-player fetch times. Keep/back up that directory for
future deadline-aware evaluation; it is ignored by Git and is not a training source.

## Interpretation and remaining limitations

See `data/model_review_results.md` for measured results. More training data improved
current-season player correlation and squad calibration, but the realistic manager
still trails the current-season human average. Historical margins over the naive
manager remain uncertain; test performance must not be used to retune parameters.

- Scoring-era features cannot fully reconstruct past assist/BPS rules or learn the
  2026–27 rule changes before observing a completed season under them.
- Archived fixture assignments may reflect later postponements. Historical injuries,
  deadline-exact prices and official average scores are not fully archived.
- Current-season backtests still use today's selectable roster and club metadata.
- Selling prices assume current prices; most chips are not simulated. Historical
  rolling managers begin in GW19 with £100m, so they are comparable to each other,
  not directly to a full-season human manager.
- The deployed refit differs from the frozen historical evaluation model. Future
  predictions should continue to be monitored without tuning against held-out results.
