"""ML model training, loading, and prediction."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MODEL_DIR
from features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Path to the saved feature medians (used for safe NaN imputation at prediction time)
_MEDIANS_PATH = MODEL_DIR / "feature_medians.joblib"

# Module-level cache so we only load from disk once per process
_feature_medians_cache: Optional[dict] = None


def _load_feature_medians() -> dict:
    """Load saved training medians, caching the result in memory."""
    global _feature_medians_cache
    if _feature_medians_cache is None:
        if _MEDIANS_PATH.exists():
            _feature_medians_cache = joblib.load(_MEDIANS_PATH)
            logger.debug("Loaded feature medians from %s", _MEDIANS_PATH)
        else:
            logger.warning(
                "feature_medians.joblib not found — using 0.0 for NaN imputation. "
                "Re-run train.py to generate it."
            )
            _feature_medians_cache = {}
    return _feature_medians_cache


def tune_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
    """Run Optuna hyperparameter search for XGBoost and LightGBM.

    Uses TimeSeriesSplit(n_splits=3) for speed. Returns best params for each model.
    Silences Optuna's verbose logs.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n_trials: Number of Optuna trials per model (default 50).

    Returns:
        Dict with keys "xgboost" and "lightgbm", each containing best hyperparams.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = TimeSeriesSplit(n_splits=3)

    def xgb_objective(trial):
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score", n_jobs=-1)
        return scores.mean()

    def lgbm_objective(trial):
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_score
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score", n_jobs=-1)
        return scores.mean()

    logger.info("Running Optuna tuning (%d trials per model)...", n_trials)

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
    xgb_best = xgb_study.best_params
    logger.info("XGBoost best params: %s (Brier: %.4f)", xgb_best, -xgb_study.best_value)

    lgbm_study = optuna.create_study(direction="maximize")
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)
    lgbm_best = lgbm_study.best_params
    logger.info("LightGBM best params: %s (Brier: %.4f)", lgbm_best, -lgbm_study.best_value)

    return {"xgboost": xgb_best, "lightgbm": lgbm_best}


def train_models(X: pd.DataFrame, y: pd.Series, tuned_params: dict | None = None) -> dict:
    """Train XGBClassifier, LightGBM, and Logistic Regression, compare, and save all.

    Uses 5-fold temporal CV (TimeSeriesSplit) to evaluate. All models are calibrated
    via CalibratedClassifierCV for well-calibrated probabilities.

    Args:
        X: Feature DataFrame (columns must match FEATURE_COLUMNS).
        y: Binary target Series (1 = home win).
        tuned_params: Optional dict from tune_hyperparameters() with keys "xgboost"
            and "lightgbm". When provided, overrides default hyperparameters.

    Returns:
        Dict with comparison metrics for both models.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    cv = TimeSeriesSplit(n_splits=5)
    logger.info("Training with TimeSeriesSplit CV (5 folds, temporal order preserved)")

    results = {}

    # Save feature medians so predict_win_prob can safely impute NaNs
    medians = X.median().to_dict()
    joblib.dump(medians, _MEDIANS_PATH)
    logger.info("Feature medians saved to %s", _MEDIANS_PATH)

    # Invalidate in-memory cache so the new medians are picked up immediately
    global _feature_medians_cache
    _feature_medians_cache = medians

    # --- XGBoost ---
    logger.info("Training XGBClassifier...")
    xgb_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "eval_metric": "logloss", "verbosity": 0,
    }
    if tuned_params and "xgboost" in tuned_params:
        xgb_params.update(tuned_params["xgboost"])
        logger.info("Using tuned XGBoost params: %s", tuned_params["xgboost"])
    xgb_base = XGBClassifier(**xgb_params)
    xgb_calibrated = CalibratedClassifierCV(xgb_base, cv=5, method="isotonic")
    xgb_calibrated.fit(X, y)

    # CV predictions for evaluation — use the same calibrated wrapper so
    # reported metrics (Brier, log-loss) reflect the model that gets saved.
    xgb_cv_probs = cross_val_predict(
        CalibratedClassifierCV(
            XGBClassifier(**xgb_params),
            cv=5, method="isotonic",
        ),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]

    results["xgboost"] = _evaluate_model("XGBoost", y, xgb_cv_probs)
    joblib.dump(xgb_calibrated, MODEL_DIR / "xgboost.joblib")
    logger.info("XGBoost saved to %s", MODEL_DIR / "xgboost.joblib")

    # --- Logistic Regression ---
    logger.info("Training Logistic Regression...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, solver="lbfgs",
        )),
    ])
    lr_calibrated = CalibratedClassifierCV(lr_pipeline, cv=5, method="sigmoid")
    lr_calibrated.fit(X, y)

    lr_cv_probs = cross_val_predict(
        CalibratedClassifierCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")),
            ]),
            cv=5, method="sigmoid",
        ),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]

    results["logistic_regression"] = _evaluate_model("Logistic Regression", y, lr_cv_probs)
    joblib.dump(lr_calibrated, MODEL_DIR / "logistic_regression.joblib")
    logger.info("Logistic Regression saved to %s", MODEL_DIR / "logistic_regression.joblib")

    # --- LightGBM ---
    logger.info("Training LightGBM...")
    from lightgbm import LGBMClassifier
    lgbm_params = {
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "verbosity": -1,
    }
    if tuned_params and "lightgbm" in tuned_params:
        lgbm_params.update(tuned_params["lightgbm"])
        logger.info("Using tuned LightGBM params: %s", tuned_params["lightgbm"])
    lgbm_base = LGBMClassifier(**lgbm_params)
    lgbm_calibrated = CalibratedClassifierCV(lgbm_base, cv=5, method="isotonic")
    lgbm_calibrated.fit(X, y)

    lgbm_cv_probs = cross_val_predict(
        CalibratedClassifierCV(
            LGBMClassifier(**lgbm_params),
            cv=5, method="isotonic",
        ),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]

    results["lightgbm"] = _evaluate_model("LightGBM", y, lgbm_cv_probs)
    joblib.dump(lgbm_calibrated, MODEL_DIR / "lightgbm.joblib")
    logger.info("LightGBM saved to %s", MODEL_DIR / "lightgbm.joblib")

    # --- Feature importance (XGBoost) ---
    # Refit a plain XGB to get feature importances
    xgb_plain = XGBClassifier(**xgb_params)
    xgb_plain.fit(X, y)
    importances = dict(zip(FEATURE_COLUMNS, xgb_plain.feature_importances_))
    results["feature_importances"] = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # --- Print comparison ---
    _print_comparison(results)

    return results


def _evaluate_model(name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """Compute evaluation metrics for a model."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "name": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
    }
    logger.info("%s — Accuracy: %.4f, Brier: %.4f, LogLoss: %.4f",
                name, metrics["accuracy"], metrics["brier_score"], metrics["log_loss"])
    return metrics


def _print_comparison(results: dict):
    """Print a formatted comparison of both models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)

    for key in ["xgboost", "logistic_regression", "lightgbm"]:
        m = results[key]
        print(f"\n  {m['name']}:")
        print(f"    Accuracy:    {m['accuracy']:.4f}")
        print(f"    Brier Score: {m['brier_score']:.4f}  (lower is better)")
        print(f"    Log Loss:    {m['log_loss']:.4f}  (lower is better)")

    # Determine recommended model
    candidates = {k: results[k]["brier_score"] for k in ["xgboost", "logistic_regression", "lightgbm"]}
    recommended = min(candidates, key=candidates.get)
    print(f"\n  >>> Recommended model: {results[recommended]['name']} "
          f"(Brier Score: {results[recommended]['brier_score']:.4f})")

    print("\n  Top 10 Features (XGBoost importance):")
    for i, (feat, imp) in enumerate(list(results["feature_importances"].items())[:10]):
        print(f"    {i+1:2d}. {feat:<30s} {imp:.4f}")

    print("=" * 60 + "\n")


def load_model(model_name: str = "xgboost"):
    """Load a saved model from disk.

    Args:
        model_name: "xgboost", "logistic_regression", or "lightgbm".

    Returns:
        The loaded sklearn/xgboost model pipeline.
    """
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first."
        )
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def predict_win_prob(model, features: dict) -> float:
    """Predict P(home team wins) for a single game.

    Args:
        model: A fitted sklearn model with predict_proba.
        features: Dict of feature name → value.

    Returns:
        Probability of home team winning (0.0 to 1.0).
    """
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Use training-time medians for NaN imputation — single-row median() is
    # unreliable (returns NaN when the only value is NaN).
    medians = _load_feature_medians()
    if medians:
        X = X.fillna(value=medians)
    else:
        X = X.fillna(0.0)

    prob = model.predict_proba(X)[0][1]  # probability of class 1 (home win)
    return float(prob)
