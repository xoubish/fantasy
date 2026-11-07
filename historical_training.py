"""Historical model selection, separate calibration, and season-aware evaluation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fpl_data import DATA_DIR
from historical_data import MANIFEST_PATH, build_historical_samples, current_season_start
from neural_model import COMPONENT_NAMES, FEATURE_NAMES, MODEL_VERSION, POINTS_OUTPUTS, FPLNeuralNetwork, Sample, _softmax

FEATURE_CACHE = DATA_DIR / "historical_features.npz"
EVALUATION_MODEL = DATA_DIR / "historical_evaluation_weights.json"


def archive_signature() -> str:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Historical training data is required. Run `python3 historical_data.py sync`.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    provenance = {"commit": manifest["commit"], "seasons": manifest["seasons"],
                  "files": {k: v["sha256"] for k, v in manifest["files"].items()}}
    from market_odds import ODDS_MANIFEST
    odds = json.loads(ODDS_MANIFEST.read_text()).get("files", {}) if ODDS_MANIFEST.exists() else {}
    # Completed-season odds files are part of the training inputs; the live files are not.
    provenance["odds"] = {k: v["sha256"] for k, v in odds.items() if k.startswith("E0_") and int(k[3:7]) < current_season_start()}
    return hashlib.sha256(json.dumps(provenance, sort_keys=True).encode() + MODEL_VERSION.encode()
                          + json.dumps(FEATURE_NAMES).encode() + str(current_season_start()).encode()).hexdigest()


def load_samples():
    signature = archive_signature()
    if FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE, allow_pickle=False) as data:
            if str(data["signature"]) == signature and "components" in data:
                return [Sample(x.tolist(), int(m), float(p), int(g), int(pid), float(n), str(season), int(fid),
                               tuple(c.tolist()), float(ppa))
                        for x, m, p, g, pid, n, season, fid, c, ppa in zip(*(data[k] for k in
                            ("features", "minutes", "points", "gw", "player", "naive", "season", "fixture",
                             "components", "naive_ppa")))]
    samples = build_historical_samples()
    np.savez_compressed(FEATURE_CACHE, signature=signature,
                        features=np.asarray([s.features for s in samples]),
                        minutes=[s.minutes for s in samples], points=[s.points for s in samples],
                        gw=[s.gw for s in samples], player=[s.player_id for s in samples],
                        naive=[s.naive_xp for s in samples], season=[s.season for s in samples],
                        fixture=[s.fixture_id for s in samples],
                        components=np.asarray([list(s.components) or [0.0] * len(COMPONENT_NAMES) for s in samples]),
                        naive_ppa=[s.naive_ppa for s in samples])
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
    raw = net._outputs(X)
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
    """Rank and calibration metrics among the players on whom decisions concentrate.

    Per week, the eleven highest predicted players (ignoring squad rules) and the top
    predicted captain are scored with their actual points, for the model and for the naive
    baseline, so a prediction change is judged by the points it would have selected.
    """
    r = net.predict_batch([s.features for s in samples])
    xp = r["expected_points"]
    actual = np.asarray([s.points for s in samples])
    naive = np.asarray([s.naive_xp for s in samples])
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault((sample.season, sample.gw, sample.player_id), []).append(i)
    weeks = {}
    for (season, gw, pid), indices in groups.items():
        weeks.setdefault((season, gw), []).append((float(xp[indices].sum()), float(actual[indices].sum()),
                                                   float(naive[indices].sum())))
    selected, captains, top11_model, top11_naive, cap_model, cap_naive = [], [], [], [], [], []
    for rows in weeks.values():
        by_model = sorted(rows, key=lambda t: -t[0])
        by_naive = sorted(rows, key=lambda t: -t[2])
        selected.extend((t[0], t[1]) for t in by_model[:15])
        captains.append((by_model[0][0], by_model[0][1]))
        top11_model.append(sum(t[1] for t in by_model[:11]))
        top11_naive.append(sum(t[1] for t in by_naive[:11]))
        cap_model.append(by_model[0][1])
        cap_naive.append(by_naive[0][1])

    def metrics(rows):
        a = np.asarray(rows)
        return {"n": len(rows), "mean_predicted": round(float(a[:, 0].mean()), 3),
                "mean_actual": round(float(a[:, 1].mean()), 3),
                "bias": round(float((a[:, 0] - a[:, 1]).mean()), 3)}
    return {"top15_per_week": metrics(selected), "top_captain_per_week": metrics(captains),
            "weeks": len(weeks),
            "top11_actual_per_week": {"model": round(float(np.mean(top11_model)), 2),
                                      "naive": round(float(np.mean(top11_naive)), 2)},
            "captain_actual_per_week": {"model": round(float(np.mean(cap_model)), 2),
                                        "naive": round(float(np.mean(cap_naive)), 2)}}


ENSEMBLE_GRID = (0.0, 0.25, 0.5, 0.75)
# Validation criterion for loss/epoch/output/ensemble choices. "rmse_points" (all rows, the
# criterion used for every result reported so far) or "played60" (RMSE among rows where the
# player played 60+ minutes). Changing it is a protocol decision to take before a season's
# test is run, never after seeing test results.
SELECTION_METRIC = "rmse_points"


def selection_score(net, rows):
    scores = net.evaluate(rows)
    if SELECTION_METRIC == "played60" and scores.get("played60"):
        return scores["played60"]["rmse_points"]
    return scores["rmse_points"]


def attach_booster(net, train, validation, signature, verbose=False):
    """Fit boosted companions on train rows, choose their blend weights on validation weeks."""
    from boosted_model import Booster
    booster, report = Booster.select(train, validation, verbose=verbose)
    net.booster = booster
    net.ensemble_weights = {"points": 0.0, "minutes": 0.0}
    scores = {}
    for wm in ENSEMBLE_GRID:
        net.ensemble_weights["minutes"] = wm
        scores[wm] = net.evaluate(validation)["minutes_logloss"]
    net.ensemble_weights["minutes"] = min(scores, key=scores.get)
    report["minutes_weight_logloss"] = {str(k): round(v, 5) for k, v in scores.items()}
    scores = {}
    for wp in ENSEMBLE_GRID:
        net.ensemble_weights["points"] = wp
        scores[wp] = selection_score(net, validation)
    net.ensemble_weights["points"] = min(scores, key=scores.get)
    report["points_weight_rmse"] = {str(k): round(v, 5) for k, v in scores.items()}
    report["weights"] = dict(net.ensemble_weights)
    net.booster_signature = booster.save(signature).stem and signature
    if verbose:
        print(f"ensemble weights {net.ensemble_weights}", flush=True)
    return report


def fit_for_season(samples, season, max_epochs=60, verbose=False):
    """Select loss, epoch count and points output (direct head, component sum or their average)
    on validation RMSE, calibrate on separate weeks, then evaluate once on the test weeks."""
    train, validation, calibration, test = split_historical(samples, season)
    candidates = []
    for loss in ("huber", "mse"):
        net = FPLNeuralNetwork(points_loss=loss)
        best = {output: {"rmse": float("inf"), "epoch": 0, "params": None} for output in POINTS_OUTPUTS}

        def checkpoint(epoch, model):
            for output in POINTS_OUTPUTS:
                model.points_output = output
                score = selection_score(model, validation)
                if score < best[output]["rmse"]:
                    best[output].update(rmse=score, epoch=epoch, params=model.clone_params())
            model.points_output = "direct"
        net.train(train, epochs=max_epochs, on_epoch_end=checkpoint)
        for output in POINTS_OUTPUTS:
            candidate = FPLNeuralNetwork(points_loss=loss, points_output=output)
            candidate.params = best[output]["params"]
            for attr in ("mu", "sigma", "cameo_points", "component_constants", "baselines"):
                setattr(candidate, attr, getattr(net, attr))
            candidates.append((best[output]["rmse"], candidate, best[output]["epoch"]))
            if verbose:
                print(f"{season}: {loss}/{output}, validation RMSE {best[output]['rmse']:.4f}, epoch {best[output]['epoch']}", flush=True)
    _, net, epochs = min(candidates, key=lambda item: item[0])
    ensemble = attach_booster(net, train, validation, f"eval-{season}", verbose=verbose)
    uncalibrated = net.evaluate(test)
    fit_calibration(net, calibration)
    report = {"evaluation_season": season, "train_samples": len(train), "validation_samples": len(validation),
              "calibration_samples": len(calibration), "test_samples": len(test),
              "split": {"train": f"earlier seasons + {season} GW1-10", "validation": f"{season} GW11-14",
                        "calibration": f"{season} GW15-18", "test": f"{season} GW19-38"},
              "chosen_loss": net.points_loss, "chosen_output": net.points_output, "chosen_epochs": epochs,
              "selection_metric": SELECTION_METRIC, "ensemble": ensemble,
              "candidates": [{"loss": n.points_loss, "output": n.points_output, "validation_rmse": round(score, 4), "epochs": ep}
                             for score, n, ep in candidates],
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
    final = FPLNeuralNetwork(points_loss=report["chosen_loss"], points_output=report["chosen_output"])
    final.train(samples, epochs=report["chosen_epochs"])
    final.calibration = json.loads(json.dumps(probe.calibration))
    chosen = report["ensemble"]["chosen"]
    from boosted_model import Booster
    final.booster = Booster(points_iterations=chosen["points_iterations"], minutes_iterations=chosen["minutes_iterations"],
                            points_loss=chosen["points_loss"]).fit(samples)
    final.ensemble_weights = dict(probe.ensemble_weights)
    final.booster.save(archive_signature())
    final.booster_signature = archive_signature()
    final.metrics = {
        "version": MODEL_VERSION, "architecture": f"{len(FEATURE_NAMES)} -> 48 -> 24 -> [minutes | points | haul | components]",
        "points_output": report["chosen_output"], "ensemble_weights": dict(final.ensemble_weights),
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
