# Historical model implementation and measured results

Training now uses **113,260 player-match rows** from completed seasons
2022-23, 2023-24, 2024-25, 2025-26. Current-season outcomes are excluded from all fitting,
model selection and calibration. Current logs supply form/minutes inputs and evaluation
outcomes; the current roster controls live selection only.

## Source and data integrity

Source: [Vaastav Anand's FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League).
The archive is pinned by commit and file SHA256 in `historical_manifest.json`.
Ten identical duplicates and 322 Assistant Manager rows were removed; goalkeeper
position names were normalized. Season-local IDs never join across years. Stable
FPL codes supply previous-season summary features. No archive `xP`, end-of-season
FDR/strength or current set-piece orders enter the model.

## Evaluation protocol

Each historical test fits on earlier seasons plus GW1–10 of its evaluation season,
selects loss/epochs on GW11–14, calibrates on GW15–18, and tests GW19–38. There are
**40 historical test weeks**, plus seven current-season evaluation weeks. Deployment
then refits on completed seasons only; it is distinct from the retained test model.

Huber and squared-error losses were compared using validation RMSE. For the latest
historical split, Huber scored 2.0421 and squared
error 2.0425; the difference is small. Huber at
9 epochs was selected without using test scores. Regularized point
calibration and minutes temperature use the separate calibration block. Calibrated
latest-season test RMSE is 1.8559, versus
1.8533 uncalibrated: calibration did not improve
that test's RMSE, although overall bias moved closer to zero. It was not retuned
against these test results.

## Historical rolling-manager results

| Test period | Model | Naive, same planner | Difference | 95% block-bootstrap interval, pts/week |
| --- | ---: | ---: | ---: | --- |
| 2024-25 GW19–38 | 1156 | 1135 | +21 | -3.30 to 5.60 |
| 2025-26 GW19–38 | 1100 | 1014 | +86 | -3.45 to 10.55 |

Combined model advantage: **+107 points over 40 weeks**.
Both seasonal uncertainty intervals include zero. These results do not establish a
reliable edge. Historical official human averages are unavailable, so the naive
baseline must not be described as the human average.

## Current-season results (GW3–9; never used for fitting)

| Metric | Previous corrected short-history model | Historical model |
| --- | ---: | ---: |
| Player MAE | 1.687 | 1.660 |
| Player correlation | 0.400 | 0.460 |
| Weekly rebuild overprediction, points/week | +12.1 | +1.2 |
| Realistic manager points | 345 | 328 |
| Paid transfers | 6 | 0 |
| FPL average total | 377 | 377 |

Prediction calibration improved substantially, but realistic-manager performance
worsened in this short current-season test and remains below average. The new planner
avoids uncertain marginal transfers using historical residual errors and recent
appearance counts. This buffer is a decision heuristic, not a confidence interval.
Neither the buffer nor other parameters were selected to maximize these seven weeks.
Coach prose remains available; unvalidated heuristic/LLM adjustments no longer alter
calibrated expected points.

## Remaining limits

Historical fixtures may include later postponements; deadline-exact prices and injury
news are not fully archived. Current-season tests retain current-snapshot roster/club
metadata. Selling-price gains and most chips are not simulated. Scoring-era features
cannot fully reconstruct past BPS/assist rules or learn new 2026–27 scoring changes
without a completed season under them. The final refit also needs continued calibration
monitoring: its weights differ from the earlier probe on which calibration was fitted.

Successful future live syncs save timestamped public-input snapshots locally in
`data/snapshots/` for later evaluation. They do not repair missing historical snapshots
and are never training inputs.

Validation: historical data normalization, chronological partition and current-season
exclusion tests, transfer-risk regression checks, model training, both 20-week historical
strategy tests, current-season testing, page payload checks, and squad-report generation.
