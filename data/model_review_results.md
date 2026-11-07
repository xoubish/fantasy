# Historical model implementation and measured results

Training uses **113,260 player-match rows** from completed seasons 2022-23, 2023-24,
2024-25 and 2025-26, plus the 2021-22 context season (fixtures, teams and season totals
only; no training rows). Current-season outcomes are excluded from all fitting, model
selection and calibration. Current logs supply form/minutes inputs and evaluation
outcomes; the current roster controls live selection only.

## What changed on 2026-11-07

A diagnostic on the previous model showed that its edge over the naive baseline came
almost entirely from predicting minutes: among rows where the player played 60+
minutes, the correlation between predicted and actual points was 0.10 for both the
model and the naive points-per-appearance. Six changes followed.

1. **Cross-season team ratings.** `team_ratings.py` fits a time-decayed Poisson
   attack/defence model on every archived match before each deadline, keyed by stable
   team codes, with a weaker prior for promoted clubs. Features are the expected goals for
   and against the player's club and the team clean-sheet probability. The 240-day
   half-life was chosen on 2022-23 to 2024-25 one-step-ahead match log-likelihood only
   (mean log-likelihood -3.0205 versus -3.0220 at 120 days and -3.0235 at 365 days).
2. **Component heads.** Goals, assists, clean sheet, goals conceded, saves, bonus,
   defensive-contribution threshold and yellow card are predicted given 60+ minutes and
   mapped to points with the official rules. The formula reproduces archived points
   exactly on every 60+ minute row of 2022-23 and 2025-26. The direct head, component
   sum or their average is selected on validation RMSE.
3. **Shrinkage toward previous-season rates** (three pseudo-matches), or toward
   position averages from the development seasons when the previous season is thin.
4. **Decision-relevant metrics**: error and correlation among 60+ minute rows (overall
   and by position), component calibration, and per-week actual points of the eleven
   highest-predicted players and of the top captain pick, against the naive baseline.
5. **Point-in-time current-season test**: prices, club membership and registration come
   from match rows (players without a row for the target week are excluded, 117-121 per
   week this season); team ratings use results before the deadline; availability flags
   use `data/snapshots/` when a snapshot predates the deadline (none exist yet for
   GW3-9, so everyone is treated as available there).
6. **Planner sweep** on GW11-18 of each historical season (`historical_backtest.py
   --tune`): transfer friction 0.75 and risk margin 0.5 were selected (931 to 936 points
   across the top candidates versus 892 for the worst; the differences are within noise).
   The epoch cap was raised from 35 to 60 after the 2024-25 validation curve selected the
   cap itself; selection remains on validation weeks.

## Evaluation protocol

Each historical test fits on earlier seasons plus GW1-10 of its evaluation season,
selects loss/epochs/output on GW11-14, calibrates on GW15-18, and tests GW19-38. There are
**40 historical test weeks**, plus seven current-season evaluation weeks. Deployment
then refits on completed seasons only; it is distinct from the retained test model.

Selected configurations: 2024-25 Huber loss, component sum, 37 epochs; 2025-26 Huber
loss, blend (average of direct head and component sum), 7 epochs. Calibration was not
retuned against test results.

## Held-out prediction results (GW19-38)

| Metric | 2024-25 model | 2024-25 naive | 2025-26 model | 2025-26 naive |
| --- | ---: | ---: | ---: | ---: |
| All rows MAE | 0.947 | 1.027 | 0.946 | 1.031 |
| All rows correlation | 0.569 | 0.511 | 0.580 | 0.518 |
| Minutes log-loss (base rate) | 0.497 (0.927) | | 0.465 (0.890) | |
| Played 60+: MAE | 2.249 | 2.216 | 2.368 | 2.354 |
| Played 60+: correlation | **0.298** | 0.230 | **0.178** | 0.104 |
| Played 60+: bias | -0.06 | | +0.04 | |
| Top-11 picks, actual points per week | 62.1 | 53.5 | 49.9 | 52.4 |
| Top captain pick, actual points per week | 8.1 | 8.2 | 4.2 | 5.3 |
| Top-15 bias (predicted minus actual) | -0.25 | | +0.85 | |

The conditional correlation (the slice that drives selection) roughly doubled relative
to the previous model's 0.10 in 2025-26 and beats the naive baseline in every position in
both seasons. Conditional MAE is level with naive because the naive per-appearance
average underpredicts the skewed target. Component means are close to actual means on
test rows (2025-26: goals 0.124 predicted versus 0.109 actual; clean sheets 0.283 versus
0.278). In 2025-26 the top-of-ranking bias is positive: the model's captain pick
(Haaland in eleven of twenty weeks) averaged 4.2 points over those weeks. Twenty
single-player observations cannot distinguish this from variance.

## Historical rolling-manager results

| Test period | Model | Naive, same planner | Difference | 95% block-bootstrap interval, pts/week |
| --- | ---: | ---: | ---: | --- |
| 2024-25 GW19-38 | 1112 | 1160 | -48 | -8.40 to 4.40 |
| 2025-26 GW19-38 | 1090 | 1087 | +3 | -4.55 to 6.95 |

Weekly-rebuild squads: 1184 versus 1224 (2024-25) and 1157 versus 1201 (2025-26).
Both intervals include zero. Compared with the previous model's runs, every change in
these totals is within one standard error of the paired weekly differences (a 20-week
total has a standard error of roughly 55 to 80 points); the naive manager's totals moved
by up to 108 points from the planner-parameter change alone. These strategy tests
therefore cannot show an edge, and the improved prediction metrics have not yet turned
into measured squad points. Historical official human averages are unavailable, so the
naive baseline must not be described as the human average.

## Current-season results (GW3-9; never used for fitting)

| Metric | Previous model | Current model |
| --- | ---: | ---: |
| Player MAE (model / naive) | 1.660 / 1.842 | 1.881 / 1.842 |
| Player correlation (model / naive) | 0.460 / 0.364 | 0.411 / 0.364 |
| Weekly rebuild overprediction, points/week | +1.2 | +2.4 |
| Weekly rebuild points (model / naive / FPL average) | 388 / 352 / 377 | 411 / 354 / 377 |
| Realistic manager points (model / naive / FPL average) | 328 / 284 / 377 | 274 / 288 / 377 |

The two rows are not on the same population: the current test excludes players who
were not registered in the target week and uses their match-row club and price. Early
weeks now lean on previous-season and position priors instead of shrinking toward zero,
which raises predictions, improves weekly-rebuild points, and worsens MAE and the
realistic manager over these seven weeks. The realistic manager, which starts cold at
GW3, remains well below the human average in both versions; seven weeks with a weekly
standard deviation near 15 points cannot rank the versions.

## Remaining limits

Historical fixtures may include later postponements; deadline-exact injury news is only
available from the snapshots saved since 2026-11-06. Selling-price gains and most chips
are not simulated. Scoring-era features cannot reconstruct past BPS/assist rules or
learn new 2026-27 scoring changes without a completed season under them. Validation
selection still uses overall RMSE, which is dominated by zero-minute rows; moving
selection to the 60+ slice is a protocol change to decide before the next season's test,
not after seeing one. The final refit's calibration is monitored on the current season,
never tuned there.

Validation: archive normalization, chronological partition and current-season exclusion
tests, point-in-time team ratings, component scoring-rule and shrinkage tests,
backtest registration/snapshot blinding tests, transfer-risk and planner-parameter
checks, model training, planner sweep, both 20-week historical strategy tests,
current-season testing, page payload checks, and squad-report generation.
