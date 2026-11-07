"""Gradient-boosted companion models (boosted_model.py).

Two scikit-learn histogram gradient-boosting models on the same features as the network:
points given 60+ minutes (regression, on 60+ rows) and the minutes class (0 / 1-59 / 60+).
`historical_training.py` chooses their iteration counts on validation weeks and blends
them with the network using weights also chosen on validation weeks; the blend is then
calibrated on the separate calibration weeks like any other output.

Fitted boosters are pickled to data/boosters/ (ignored by Git) keyed by the training
signature; a missing pickle is refit from the cached training rows.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fpl_data import DATA_DIR

BOOSTER_DIR = DATA_DIR / "boosters"
POINTS_PARAMS = dict(learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=0)
MINUTES_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, random_state=0)
ITERATION_GRID = (50, 100, 200, 350, 600)


class Booster:
    def __init__(self, points_iterations: int = 100, minutes_iterations: int = 200, points_loss: str = "squared_error"):
        self.points_iterations = points_iterations
        self.minutes_iterations = minutes_iterations
        self.points_loss = points_loss
        self.points_model: Any = None
        self.minutes_model: Any = None
        self.signature: str = ""

    @staticmethod
    def _arrays(samples: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray([s.features for s in samples], dtype=float)
        y = np.asarray([s.points for s in samples], dtype=float)
        m = np.asarray([s.minutes for s in samples], dtype=float)
        return X, y, m

    @staticmethod
    def minutes_classes(m: np.ndarray) -> np.ndarray:
        return np.where(m <= 0, 0, np.where(m < 60, 1, 2))

    def fit(self, samples: Sequence[Any]) -> "Booster":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        X, y, m = self._arrays(samples)
        full = m >= 60
        self.points_model = HistGradientBoostingRegressor(loss=self.points_loss, max_iter=self.points_iterations,
                                                          early_stopping=False, **POINTS_PARAMS).fit(X[full], y[full])
        self.minutes_model = HistGradientBoostingClassifier(max_iter=self.minutes_iterations, early_stopping=False,
                                                            **MINUTES_PARAMS).fit(X, self.minutes_classes(m))
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        xp_full = self.points_model.predict(X)
        proba = self.minutes_model.predict_proba(X)
        classes = list(self.minutes_model.classes_)
        pm = np.zeros((len(X), 3))
        for j, c in enumerate(classes):
            pm[:, int(c)] = proba[:, j]
        return {"xp_full": xp_full, "pm": pm}

    # ---- selection on validation weeks ----
    @classmethod
    def select(cls, train: Sequence[Any], validation: Sequence[Any], verbose: bool = False) -> Tuple["Booster", Dict[str, Any]]:
        """Iteration counts for both models chosen by validation RMSE (60+ rows) and log-loss."""
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        X, y, m = cls._arrays(train)
        Xv, yv, mv = cls._arrays(validation)
        full, full_v = m >= 60, mv >= 60
        report: Dict[str, Any] = {"points": {}, "minutes": {}}
        best_points = None
        for loss in ("squared_error", "absolute_error"):
            model = HistGradientBoostingRegressor(loss=loss, max_iter=max(ITERATION_GRID), early_stopping=False, **POINTS_PARAMS)
            model.fit(X[full], y[full])
            scores = {}
            for i, pred in enumerate(model.staged_predict(Xv[full_v]), start=1):
                if i in ITERATION_GRID:
                    scores[i] = float(np.sqrt(np.mean((pred - yv[full_v]) ** 2)))
            report["points"][loss] = scores
            n, score = min(scores.items(), key=lambda kv: kv[1])
            if best_points is None or score < best_points[0]:
                best_points = (score, loss, n)
        model = HistGradientBoostingClassifier(max_iter=max(ITERATION_GRID), early_stopping=False, **MINUTES_PARAMS)
        model.fit(X, cls.minutes_classes(m))
        yv_class = cls.minutes_classes(mv)
        scores = {}
        for i, proba in enumerate(model.staged_predict_proba(Xv), start=1):
            if i in ITERATION_GRID:
                p = np.clip(proba[np.arange(len(Xv)), yv_class], 1e-6, 1)
                scores[i] = float(-np.log(p).mean())
        report["minutes"] = scores
        n_minutes = min(scores.items(), key=lambda kv: kv[1])[0]
        booster = cls(points_iterations=best_points[2], minutes_iterations=n_minutes, points_loss=best_points[1]).fit(train)
        report["chosen"] = {"points_loss": best_points[1], "points_iterations": best_points[2], "minutes_iterations": n_minutes}
        if verbose:
            print(f"booster: {report['chosen']}", flush=True)
        return booster, report

    # ---- persistence ----
    def save(self, signature: str) -> Path:
        BOOSTER_DIR.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        path = BOOSTER_DIR / f"{signature[:16]}.pkl"
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, signature: str) -> Optional["Booster"]:
        path = BOOSTER_DIR / f"{signature[:16]}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as handle:
            booster = pickle.load(handle)
        return booster if getattr(booster, "signature", "") == signature else None
