"""Diagnostic: does a gradient-boosted model find signal the numpy network misses?

Uses the same cached features and the same chronological split as the historical
protocol (train = earlier seasons + GW1-10, validation = GW11-14 for early stopping, test
= GW19-38 of the evaluation season). Two boosted regressors are fitted: one for points
given 60+ minutes (on 60+ rows) and one for P(60+ minutes) (all rows); their product is
compared with the network on the same decision-relevant metrics. Requires scikit-learn,
which is not a runtime dependency of the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from historical_training import EVALUATION_MODEL, load_samples, split_historical
from neural_model import FEATURE_NAMES, FPLNeuralNetwork


def metrics(label, pred_total, pred_full, pts, mins, naive, naive_ppa, gw, pid):
    full = mins >= 60
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    weeks = {}
    for i, g in enumerate(gw):
        weeks.setdefault(g, []).append(i)
    top11, cap = [], []
    for g, idx in weeks.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(-pred_total[idx])]
        top11.append(pts[order[:11]].sum())
        cap.append(pts[order[0]])
    return {"model": label,
            "all_mae": round(float(np.abs(pred_total - pts).mean()), 4), "all_corr": round(corr(pred_total, pts), 4),
            "played60_mae": round(float(np.abs(pred_full[full] - pts[full]).mean()), 4),
            "played60_corr": round(corr(pred_full[full], pts[full]), 4),
            "played60_bias": round(float((pred_full[full] - pts[full]).mean()), 4),
            "top11_actual_per_week": round(float(np.mean(top11)), 2), "captain_actual_per_week": round(float(np.mean(cap)), 2)}


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    args = parser.parse_args()
    samples = load_samples()
    train, val, cal, test = split_historical(samples, args.season)
    dev = train + val
    X = np.asarray([s.features for s in dev]); y = np.asarray([s.points for s in dev]); m = np.asarray([s.minutes for s in dev])
    Xt = np.asarray([s.features for s in test]); yt = np.asarray([s.points for s in test]); mt = np.asarray([s.minutes for s in test])
    naive = np.asarray([s.naive_xp for s in test]); naive_ppa = np.asarray([s.naive_ppa for s in test])
    gw = np.asarray([s.gw for s in test]); pid = np.asarray([s.player_id for s in test])
    is_val = np.asarray([s.gw >= 11 for s in dev])  # validation rows used only for early stopping

    full = m >= 60
    reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.04, max_leaf_nodes=15,
                                        min_samples_leaf=80, l2_regularization=1.0, early_stopping=True,
                                        validation_fraction=None, n_iter_no_change=40, random_state=0)
    # sklearn's early stopping takes a random validation fraction; emulate a chronological one
    # by fitting on train rows and scoring on validation rows manually.
    reg_sq = HistGradientBoostingRegressor(loss="squared_error", max_iter=600, learning_rate=0.04, max_leaf_nodes=15,
                                           min_samples_leaf=80, l2_regularization=1.0, early_stopping=False, random_state=0)
    best = None
    for name, model in (("huber-like (absolute)", reg), ("squared", reg_sq)):
        model.set_params(early_stopping=False)
        model.fit(X[full & ~is_val], y[full & ~is_val])
        # choose iteration count on validation rows (60+)
        val_mask = full & is_val
        scores = [np.sqrt(np.mean((p - y[val_mask]) ** 2)) for p in model.staged_predict(X[val_mask])]
        n_best = int(np.argmin(scores)) + 1
        print(f"{name}: best iterations {n_best}, validation RMSE {min(scores):.4f}")
        if best is None or min(scores) < best[0]:
            best = (min(scores), name, n_best)
    _, name, n_best = best
    loss = "absolute_error" if name.startswith("huber") else "squared_error"
    points_model = HistGradientBoostingRegressor(loss=loss, max_iter=n_best, learning_rate=0.04, max_leaf_nodes=15,
                                                 min_samples_leaf=80, l2_regularization=1.0, early_stopping=False, random_state=0)
    points_model.fit(X[full], y[full])
    minutes_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
                                                   early_stopping=False, random_state=0)
    minutes_model.fit(X, np.where(m <= 0, 0, np.where(m < 60, 1, 2)))
    proba = minutes_model.predict_proba(Xt)
    cameo = float(y[(m > 0) & (m < 60)].mean())
    gbm_full = np.clip(points_model.predict(Xt), 0.5, 14)
    gbm_total = proba[:, 2] * gbm_full + proba[:, 1] * cameo

    net = FPLNeuralNetwork.load(Path(EVALUATION_MODEL))
    r = net.predict_batch(Xt)
    rows = [metrics("network", r["expected_points"], r["xp_full"], yt, mt, naive, naive_ppa, gw, pid),
            metrics("gbm", gbm_total, gbm_full, yt, mt, naive, naive_ppa, gw, pid),
            metrics("average of both", 0.5 * (gbm_total + r["expected_points"]), 0.5 * (gbm_full + r["xp_full"]), yt, mt, naive, naive_ppa, gw, pid),
            metrics("naive", naive, naive_ppa, yt, mt, naive, naive_ppa, gw, pid)]
    print(json.dumps(rows, indent=1))
    importance = sorted(zip(FEATURE_NAMES, points_model.feature_importances_ if hasattr(points_model, "feature_importances_") else [0] * len(FEATURE_NAMES)), key=lambda t: -t[1])[:10]
    print("gbm top features (if available):", importance)


if __name__ == "__main__":
    main()
