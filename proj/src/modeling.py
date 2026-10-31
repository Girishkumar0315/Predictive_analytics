"""Model zoo, metrics, tuning and evaluation utilities for both prediction tasks."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import make_preprocessor


# ------------------------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------------------------
class CountryMedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training median of log-salary for the respondent's country.

    A *domain-aware* baseline: it shows how much of the salary signal is plain geography.
    Unseen / missing countries fall back to the global training median.
    """
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.median())
        self.map_ = y.groupby(X["country"].values).median().to_dict()
        return self

    def predict(self, X):
        return X["country"].map(self.map_).fillna(self.global_).to_numpy()


# ------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------
def reg_metrics(y_log, pred_log) -> dict:
    """Metrics on the log scale (model scale) and in USD (business scale)."""
    y_log = np.asarray(y_log); pred_log = np.asarray(pred_log)
    usd, usd_hat = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(usd_hat - usd) / usd
    return dict(R2=r2_score(y_log, pred_log),
                RMSE_log=float(np.sqrt(mean_squared_error(y_log, pred_log))),
                MAE_log=mean_absolute_error(y_log, pred_log),
                MAE_USD=mean_absolute_error(usd, usd_hat),
                MdAPE_pct=float(np.median(ape) * 100),
                Within25_pct=float((ape <= 0.25).mean() * 100))


def clf_metrics(y, proba, thr: float = 0.5) -> dict:
    y = np.asarray(y); pred = (np.asarray(proba) >= thr).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), BalAcc=balanced_accuracy_score(y, pred),
                Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0), ROC_AUC=roc_auc_score(y, proba),
                PR_AUC=average_precision_score(y, proba), Brier=brier_score_loss(y, proba))


# ------------------------------------------------------------------------------------
# Model zoo (name -> (estimator, search space))
# ------------------------------------------------------------------------------------
def regression_zoo(spec):
    lin, tree = make_preprocessor("linear", spec), make_preprocessor("tree", spec, "continuous")
    return {
        "Baseline: global median": (DummyRegressor(strategy="median"), None),
        "Baseline: country median": (CountryMedianBaseline(), None),
        "Ridge regression": (Pipeline([("pre", lin), ("m", Ridge(random_state=RANDOM_STATE))]),
                             {"m__alpha": [0.3, 1, 3, 10, 30, 100, 300]}),
        "Random Forest": (Pipeline([("pre", tree), ("m", RandomForestRegressor(
            n_estimators=150, n_jobs=1, random_state=RANDOM_STATE))]),
                          {"m__max_depth": [8, 12, 16, 24], "m__min_samples_leaf": [3, 5, 10, 20],
                           "m__max_features": [0.3, 0.5, 0.8]}),
        "HistGradientBoosting": (Pipeline([("pre", tree), ("m", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=600))]),
                                 {"m__learning_rate": [0.02, 0.04, 0.08], "m__max_leaf_nodes": [8, 15, 31],
                                  "m__min_samples_leaf": [20, 50, 100], "m__l2_regularization": [0, 1, 10]}),
        "LightGBM": (Pipeline([("pre", tree), ("m", LGBMRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
                     {"m__learning_rate": [0.02, 0.04, 0.08], "m__num_leaves": [7, 15, 31],
                      "m__min_child_samples": [20, 50, 100], "m__subsample": [0.7, 0.9, 1.0],
                      "m__subsample_freq": [1], "m__colsample_bytree": [0.6, 0.8, 1.0],
                      "m__reg_lambda": [0, 1, 10]}),
    }


def classification_zoo(spec):
    lin, tree = make_preprocessor("linear", spec), make_preprocessor("tree", spec, "binary")
    return {
        "Baseline: majority class": (DummyClassifier(strategy="prior"), None),
        "Logistic Regression": (Pipeline([("pre", lin), ("m", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE))]), {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}),
        "Random Forest": (Pipeline([("pre", tree), ("m", RandomForestClassifier(
            n_estimators=150, n_jobs=1, random_state=RANDOM_STATE))]),
                          {"m__max_depth": [8, 12, 16, 24], "m__min_samples_leaf": [3, 5, 10, 20],
                           "m__max_features": [0.3, 0.5, 0.8]}),
        "HistGradientBoosting": (Pipeline([("pre", tree), ("m", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=600))]),
                                 {"m__learning_rate": [0.02, 0.04, 0.08], "m__max_leaf_nodes": [8, 15, 31],
                                  "m__min_samples_leaf": [20, 50, 100], "m__l2_regularization": [0, 1, 10]}),
        "LightGBM": (Pipeline([("pre", tree), ("m", LGBMClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
                     {"m__learning_rate": [0.02, 0.04, 0.08], "m__num_leaves": [7, 15, 31],
                      "m__min_child_samples": [20, 50, 100], "m__subsample": [0.7, 0.9, 1.0],
                      "m__subsample_freq": [1], "m__colsample_bytree": [0.6, 0.8, 1.0],
                      "m__reg_lambda": [0, 1, 10]}),
    }


def tune_and_compare(zoo: dict, X_tr, y_tr, task: str, n_iter: int = 8, tune_cv: int = 3,
                     eval_cv: int = 5, verbose: bool = True):
    """Tune each model on the TRAIN set only, then 5-fold CV the tuned model for a fair comparison.

    Returns (fitted_models, comparison_df, best_params, cv_fold_scores)
    """
    is_reg = task == "regression"
    splitter = (lambda k: KFold(k, shuffle=True, random_state=RANDOM_STATE)) if is_reg else \
               (lambda k: StratifiedKFold(k, shuffle=True, random_state=RANDOM_STATE))
    tune_scoring = "neg_root_mean_squared_error" if is_reg else "roc_auc"
    eval_scoring = ({"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
                    if is_reg else {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"})
    fitted, rows, best_params, fold_scores = {}, [], {}, {}
    for name, (est, space) in zoo.items():
        t0 = time.time()
        if space:
            n_combo = int(np.prod([len(v) for v in space.values()]))
            search = RandomizedSearchCV(est, space, n_iter=min(n_iter, n_combo), cv=splitter(tune_cv),
                                        scoring=tune_scoring, random_state=RANDOM_STATE, n_jobs=1,
                                        refit=True)
            search.fit(X_tr, y_tr)
            model, best_params[name] = search.best_estimator_, {k: (v.item() if hasattr(v, "item") else v)
                                                                for k, v in search.best_params_.items()}
        else:
            model = est.fit(X_tr, y_tr); best_params[name] = {}
        cv = cross_validate(model, X_tr, y_tr, cv=splitter(eval_cv), scoring=eval_scoring, n_jobs=1)
        fitted[name] = model
        row = {"model": name}
        for k in eval_scoring:
            v = cv[f"test_{k}"]; v = -v if k in ("rmse", "mae") else v
            row[f"cv_{k}_mean"], row[f"cv_{k}_std"] = v.mean(), v.std()
            fold_scores.setdefault(name, {})[k] = v.tolist()
        row["seconds"] = time.time() - t0
        rows.append(row)
        if verbose:
            print(f"  {name:28s} " + "  ".join(f"{k}={row[f'cv_{k}_mean']:.4f}" for k in eval_scoring)
                  + f"  [{row['seconds']:.0f}s]", flush=True)
    return fitted, pd.DataFrame(rows), best_params, fold_scores
