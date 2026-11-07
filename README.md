# FPL historical predictor and transfer planner

The model trains on **completed historical seasons**, independently of today's player
roster. Current-season outcomes are excluded from training, model selection and
calibration. Current match logs supply recent form/minutes features and forward-test
outcomes; the live FPL roster determines who can be selected now.

## Data and reproducibility

`historical_data.py` imports four completed seasons (currently 2022–23 through 2025–26)
from [Vaastav Anand's FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League),
plus the fixtures, teams and season totals of the preceding context season (2021–22),
which supplies previous-season player rates and match results for the team model but
contributes no training rows. The source commit, file URLs, SHA256 hashes and row counts
are recorded in `data/historical_manifest.json`. Raw CSVs and the generated feature cache
are ignored by Git and can be restored with `python3 historical_data.py sync`. `--refresh`
explicitly updates the pinned source commit.

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

**Team ratings** (`team_ratings.py`) come from a time-decayed Poisson attack/defence model
fitted on every archived match before each gameweek deadline, across seasons, keyed by
stable FPL team codes. Each fixture contributes expected goals for and against the
player's club (and a team clean-sheet probability) as features. Newly promoted clubs are
shrunk toward a weaker prior. The 240-day half-life was chosen by one-step-ahead match
log-likelihood on 2022–23 to 2024–25 only (`python3 team_ratings.py select`). Ratings no
longer restart from a flat value each August.

**Per-90 rates** are recency-weighted and shrunk toward the player's previous-season
rates (three pseudo-matches), or toward position averages from the development seasons
when the player had under 450 previous-season minutes (`historical_data.py
position-priors` recomputes those constants).

## Training, calibration and tests

The numpy network has shared layers and separate minutes, conditional-points, haul and
**component** heads (goals, assists, clean sheet, goals conceded, saves, bonus,
defensive-contribution threshold, yellow card; all conditional on 60+ minutes). The
component expected points apply the official scoring rules to those heads, including
the 2025–26 defensive-contribution points; the formula reproduces archived points
exactly on every 60+ minute row. `historical_training.py` compares Huber and
squared-error losses and three points outputs (direct head, component sum, their
average) using validation **RMSE**, then fits regularized point calibration and minutes
temperature on a separate period. The optimizer receives the resulting expected points.
Coach notes are informational; heuristic or LLM adjustments do not alter calibrated scores.

For each historical evaluation season:

1. Fit on all earlier seasons plus GW1–10 of that season.
2. Select loss/epoch count on GW11–14.
3. Fit calibration on GW15–18.
4. Evaluate GW19–38 without any further fitting or tuning.

Evaluation reports overall error, but overall error is dominated by players who do not
play. The decision-relevant metrics are therefore reported separately: error and
correlation of the conditional prediction among rows where the player played 60+
minutes (overall and by position), component calibration, and per test week the
actual points of the eleven highest-predicted players and of the top captain pick,
each against the naive baseline.

`historical_backtest.py` tests 2024–25 and 2025–26: **40 historical test weeks**. It
reports player error, selected-player/captain calibration, realistic rolling-manager
scores and four-week block-bootstrap intervals against the same-planner naive
baseline. Official historical human averages are unavailable in this archive; no
historical claim against the human average is made. `--tune` first sweeps the
planner's transfer friction and estimation-risk margin on GW11–18 of each season
(never on test weeks) and writes the choice to `data/planner_params.json`, which the
live planner and both backtests then use.

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

The **PL Fantasy Predictor** page offers a squad shuffle with an adjustable minimum
projected score (normally 55 points), plus a button to return to the top projected
squad. Each build samples up to 120 distinct starting XIs, within five points of
the initial optimized squad. Random player swaps respect the £100m budget, club
limits and squad composition; the best legal XI and captain are scored using the
unchanged predictions. This is a bounded shortlist, not every possible team.
The page draws from qualifying teams without repetition until the matching pool
has been seen. Projections include the captain, but no bench or automatic-sub bonuses.

Requires Python 3.10+, `numpy`, `scipy` and `pillow`.

```bash
python3 -m pip install numpy scipy pillow
python3 fpl_assistant.py sync          # live roster/form histories, historical archive, news
python3 historical_data.py sync       # restore just the pinned historical archive
python3 fpl_assistant.py train-nn      # historical loss/output comparison, calibration, test, deployment fit
python3 historical_backtest.py --tune # planner sweep on GW11-18, then 40 historical strategy-test weeks
python3 backtest.py --no-coach         # fixed-model current-season test against FPL averages
python3 team_ratings.py select        # half-life check on development seasons; `table` prints current ratings
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

See `data/model_review_results.md` for measured results. The cross-season ratings,
previous-season priors and component heads raised the held-out correlation among
players who actually play from roughly the naive level to 0.30 (2024-25) and 0.18
(2025-26), but the 20-week strategy tests remain within noise of the naive manager and
the realistic manager still trails the current-season human average. Test performance
must not be used to retune parameters.

- Scoring-era features cannot fully reconstruct past assist/BPS rules or learn the
  2026–27 rule changes before observing a completed season under them.
- Archived fixture assignments may reflect later postponements. Historical injuries,
  deadline-exact prices and official average scores are not fully archived.
- Current-season backtests take prices, club membership and registration from each
  player's match rows and team ratings from results before the deadline. Availability
  flags and set-piece roles come from `data/snapshots/` when a snapshot predates the
  deadline; earlier weeks of this season have no snapshot, so everyone is treated as
  available there.
- Selling prices assume current prices; most chips are not simulated. Historical
  rolling managers begin in GW19 with £100m, so they are comparable to each other,
  not directly to a full-season human manager.
- The deployed refit differs from the frozen historical evaluation model. Future
  predictions should continue to be monitored without tuning against held-out results.
