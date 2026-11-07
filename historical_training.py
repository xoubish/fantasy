"""Historical model selection, separate calibration, and season-aware evaluation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fpl_data import DATA_DIR
from historical_data import MANIFEST_PATH, build_historical_samples, current_season_start
from neural_model import FEATURE_NAMES, MODEL_VERSION, FPLNeuralNetwork, Sample, _softmax

FEATURE_CACHE = DATA_DIR / "historical_features.npz"
EVALUATION_MODEL = DATA_DIR / "historical_evaluation_weights.json"


def archive_signature() -> str:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Historical training data is required. Run `python3 historical_data.py sync`.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    provenance = {"commit": manifest["commit"], "seasons": manifest["seasons"],
                  "files": {k: v["sha256"] for k, v in manifest["files"].items()}}
    return hashlib.sha256(json.dumps(provenance, sort_keys=True).encode() + MODEL_VERSION.encode()
                          + json.dumps(FEATURE_NAMES).encode() + str(current_season_start()).encode()).hexdigest()


def load_samples():
    signature = archive_signature()
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE, allow_pickle=False) as data:
            if str(data["signature"]) == signature:
                return [Sample(x.tolist(), int(m), float(p), int(g), int(pid), float(n), str(season), int(fid))
                        for x, m, p, g, pid, n, season, fid in zip(*(data[k] for k in
                            ("features", "minutes", "points", "gw", "player", "naive", "season", "fixture")))]
    samples = build_historical_samples()
    np.savez_compressed(FEATURE_CACHE, signature=signature,
                        features=np.asarray([s.features for s in samples]),
                        minutes=[s.minutes for s in samples], points=[s.points for s in samples],
                        gw=[s.gw for s in samples], player=[s.player_id for s in samples],
                        naive=[s.naive_xp for s in samples], season=[s.season for s in samples],
                        fixture=[s.fixture_id for s in samples])
    return samples


def split_historical(samples, evaluation_season):
    """Predeclare splits; the entire last 20 GWs are excluded from all tuning."""
    train = [s for s in samples if s.season < evaluation_season or (s.season == evaluation_season and s.gw <= 10)]
    validation = [s for s in samples if s.season == evaluation_season and 11 <= s.gw <= 14]
    calibration = [s for s in samples if s.season == evaluation_season and 15 <= s.gw <= 18]
    test = [s for s in samples if s.season == evaluation_season and s.gw >= 19]
    if not all((train, validation, calibration, test)):
        raise ValueError(f"Incomplete chronological split for {evaluation_season}")
    return train, validation, calibration, test


def fit_calibration(net, samples):
    """Fit temperature and regularized conditional-point calibration on separate rows."""
    X = np.asarray([s.features for s in samples])
    actual = np.asarray([s.points for s in samples])
    minutes = np.asarray([s.minutes for s in samples])
    classes = np.where(minutes <= 0, 0, np.where(minutes < 60, 1, 2))
    raw = net._forward(X)
    logits = np.log(np.maximum(raw["pm"], 1e-12))
    temperatures = [0.75, 1.0, 1.25, 1.5, 2.0]
    def logloss(t):
        p = _softmax(logits / t)
        return -np.log(np.maximum(p[np.arange(len(samples)), classes], 1e-12)).mean()
    net.calibration["minutes_temperature"] = min(temperatures, key=logloss)
    coefficients = []
    for pos in range(4):
        mask = (X[:, pos] > 0.5) & (minutes >= 60)
        if mask.sum() < 50:
            coefficients.append([0.0, 1.0])
            continue
        design = np.column_stack([np.ones(mask.sum()), raw["xp"][mask]])
        # Shrink toward identity; calibration cannot freely fit a tiny tail bin.
        penalty = np.diag([40.0, 160.0])
        beta = np.linalg.solve(design.T @ design + penalty,
                               design.T @ actual[mask] + penalty @ np.array([0.0, 1.0]))
        coefficients.append([float(np.clip(beta[0], -3, 3)), float(np.clip(beta[1], 0.1, 1.5))])
    net.calibration["points"] = coefficients
    pred = net.predict_batch(X)["expected_points"]
    net.calibration["residual_std"] = [float(np.sqrt(np.mean((pred[X[:, i] > .5] - actual[X[:, i] > .5]) ** 2)))
                                        if (X[:, i] > .5).any() else 3.0 for i in range(4)]
    net.calibration["fitted"] = True
    net.calibration["periods"] = sorted({f"{s.season} GW{s.gw}" for s in samples})


def selection_metrics(net, samples):
    """Rank and calibration metrics among the players on whom decisions concentrate."""
    r = net.predict_batch([s.features for s in samples])
    xp = r["expected_points"]
    actual = np.asarray([s.points for s in samples])
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault((sample.season, sample.gw, sample.player_id), []).append(i)
    weeks = {}
    for (season, gw, pid), indices in groups.items():
        weeks.setdefault((season, gw), []).append((float(xp[indices].sum()), float(actual[indices].sum())))
    selected, captains = [], []
    for rows in weeks.values():
        rows.sort(reverse=True)
        selected.extend(rows[:15])
        captains.append(rows[0])
    def metrics(rows):
        a = np.asarray(rows)
        return {"n": len(rows), "mean_predicted": round(float(a[:, 0].mean()), 3),
                "mean_actual": round(float(a[:, 1].mean()), 3),
                "bias": round(float((a[:, 0] - a[:, 1]).mean()), 3)}
    return {"top15_per_week": metrics(selected), "top_captain_per_week": metrics(captains)}


def fit_for_season(samples, season, max_epochs=35, verbose=False):
    train, validation, calibration, test = split_historical(samples, season)
    candidates = []
    for loss in ("huber", "mse"):
        net = FPLNeuralNetwork(points_loss=loss)
        best = {"rmse": float("inf"), "epoch": 0, "params": None}
        def checkpoint(epoch, model):
            score = model.evaluate(validation)["rmse_points"]
            if score < best["rmse"]:
                best.update(rmse=score, epoch=epoch, params=model.clone_params())
        net.train(train, epochs=max_epochs, on_epoch_end=checkpoint)
        net.params = best["params"]
        candidates.append((best["rmse"], net, best["epoch"]))
        if verbose:
            print(f"{season}: {loss}, validation RMSE {best['rmse']:.4f}, epoch {best['epoch']}", flush=True)
    _, net, epochs = min(candidates, key=lambda item: item[0])
    uncalibrated = net.evaluate(test)
    fit_calibration(net, calibration)
    report = {"evaluation_season": season, "train_samples": len(train), "validation_samples": len(validation),
              "calibration_samples": len(calibration), "test_samples": len(test),
              "split": {"train": f"earlier seasons + {season} GW1-10", "validation": f"{season} GW11-14",
                        "calibration": f"{season} GW15-18", "test": f"{season} GW19-38"},
              "chosen_loss": net.points_loss, "chosen_epochs": epochs,
              "candidates": [{"loss": n.points_loss, "validation_rmse": score, "epochs": ep} for score, n, ep in candidates],
              "uncalibrated_test": uncalibrated, "test": net.evaluate(test),
              "selection": selection_metrics(net, test), "calibration": net.calibration}
    net.metrics = {"historical_only": True, "coach_adjustments": False, "evaluation_season": season}
    return net, report


def train_historical_model(verbose=False):
    samples = load_samples()
    seasons = sorted({s.season for s in samples})
    probe, report = fit_for_season(samples, seasons[-1], verbose=verbose)
    probe.save(EVALUATION_MODEL)
    # Evaluation is finished before deployment refits on all completed seasons.
    final = FPLNeuralNetwork(points_loss=report["chosen_loss"])
    final.train(samples, epochs=report["chosen_epochs"])
    final.calibration = json.loads(json.dumps(probe.calibration))
    final.metrics = {
        "version": MODEL_VERSION, "architecture": f"{len(FEATURE_NAMES)} -> 48 -> 24 -> [minutes | points | haul]",
        "historical_only": True, "coach_adjustments": False, "training_signature": archive_signature(),
        "training_seasons": seasons, "samples_by_season": dict(Counter(s.season for s in samples)),
        "real_gw_samples": len(samples), "latest_trained_gw": 38,
        "training_end": seasons[-1] + " GW38", "test_gws": report["split"]["test"],
        "test": report["test"], "feature_importance": final.feature_importance(),
        "evaluation": report,
        "evaluation_protocol": "historical train/validation/calibration/test; deployment refit on completed seasons only; current season never fitted",
        "calibration_note": "Calibration is frozen from the earlier probe; final-refit calibration is monitored on the current season, never tuned there.",
    }
    return final, report
