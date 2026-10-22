"""Preprocessing pipelines, model zoo, hyperparameter spaces, time-aware CV
and evaluation helpers.

Design choices:
* Temporal split - train <= 2015, test 2016+. A random split would let the
  model see 2020 when predicting 2018 (train-test contamination).
* Expanding-window CV inside the training period (5 validation blocks of 3
  years). Same idea as TimeSeriesSplit, but split on season_year so all
  districts of one season stay in the same fold.
* All preprocessing (imputation, scaling, encoding) lives INSIDE a Pipeline,
  so it is re-fitted on each CV training fold (no preprocessing leakage).
* Scaling matters for Ridge / SVR / KNN; trees ignore it.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from .features import CAT_FEATURES, TARGET

RS = C.RANDOM_STATE


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, train_end: int = C.TRAIN_END):
    train = df[df.season_year <= train_end].reset_index(drop=True)
    test = df[df.season_year > train_end].reset_index(drop=True)
    return train, test


def expanding_window_cv(train: pd.DataFrame, blocks=C.CV_VAL_BLOCKS):
    """List of (train_idx, val_idx): train on everything before the block."""
    folds = []
    for lo, hi in blocks:
        tr = np.where(train.season_year < lo)[0]
        va = np.where(train.season_year.between(lo, hi))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def make_preprocessor(features: list[str], scale: bool = True, onehot: bool = True):
    num = [f for f in features if f not in CAT_FEATURES]
    cat = [f for f in features if f in CAT_FEATURES]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    if onehot:
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_pipe = "passthrough"
    pre = ColumnTransformer([("num", Pipeline(num_steps), num), ("cat", cat_pipe, cat)],
                            verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


# --------------------------------------------------------------------------
# Model zoo: name -> (estimator, param distributions, needs scaling)
# --------------------------------------------------------------------------
def model_zoo(features: list[str]) -> dict:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def pipe(est, scale=True, onehot=True):
        return Pipeline([("pre", make_preprocessor(features, scale, onehot)), ("model", est)])

    def scaled_y(est):   # SVR / KNN work better when y is standardised too
        return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())

    zoo = {
        "Ridge (baseline)": (
            pipe(Ridge()),
            {"model__alpha": loguniform(1e-3, 1e3)}),
        "KNN": (
            pipe(scaled_y(KNeighborsRegressor())),
            {"model__regressor__n_neighbors": randint(3, 40),
             "model__regressor__weights": ["uniform", "distance"]}),
        "SVR (RBF)": (
            pipe(scaled_y(SVR(kernel="rbf"))),
            {"model__regressor__C": loguniform(0.1, 100),
             "model__regressor__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
             "model__regressor__epsilon": uniform(0.02, 0.3)}),
        "Decision Tree": (
            pipe(DecisionTreeRegressor(random_state=RS), scale=False),
            {"model__max_depth": randint(3, 16),
             "model__min_samples_leaf": randint(5, 60),
             "model__ccp_alpha": loguniform(1e-1, 1e4)}),
        "Random Forest": (
            pipe(RandomForestRegressor(random_state=RS, n_jobs=-1), scale=False),
            {"model__n_estimators": randint(200, 700),
             "model__max_depth": [None, 8, 12, 16, 24],
             "model__max_features": ["sqrt", 0.33, 0.5, 0.8],
             "model__min_samples_leaf": randint(1, 12)}),
        "Gradient Boosting": (
            pipe(GradientBoostingRegressor(random_state=RS), scale=False),
            {"model__n_estimators": randint(200, 800),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 6),
             "model__subsample": uniform(0.6, 0.4),
             "model__min_samples_leaf": randint(3, 30)}),
        "XGBoost": (
            pipe(XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist"), scale=False),
            {"model__n_estimators": randint(200, 1000),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 8),
             "model__min_child_weight": randint(1, 15),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5),
             "model__reg_alpha": loguniform(1e-3, 10),
             "model__reg_lambda": loguniform(1e-2, 50)}),
        "LightGBM": (
            pipe(LGBMRegressor(random_state=RS, n_jobs=-1, verbose=-1), scale=False),
            {"model__n_estimators": randint(200, 1000),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__num_leaves": randint(8, 64),
             "model__min_child_samples": randint(5, 50),
             "model__subsample": uniform(0.6, 0.4),
             "model__subsample_freq": [1],
             "model__colsample_bytree": uniform(0.5, 0.5),
             "model__reg_lambda": loguniform(1e-2, 50)}),
    }
    return zoo


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y, p) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "MAPE_%": float(100 * mean_absolute_percentage_error(y, p)),
            "R2": float(r2_score(y, p))}


def shortfall_labels(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """1 if value < threshold x district's previous-5 same-season mean."""
    ref = df["yield_mean5_same"].to_numpy()
    return np.where(np.isnan(ref), np.nan, (values < C.SHORTFALL_THRESHOLD * ref).astype(float))


def shortfall_metrics(df: pd.DataFrame, y_true, y_pred) -> dict:
    a, p = shortfall_labels(df, np.asarray(y_true)), shortfall_labels(df, np.asarray(y_pred))
    m = ~np.isnan(a)
    a, p = a[m].astype(int), p[m].astype(int)
    return {"shortfall_cases": int(a.sum()), "precision": precision_score(a, p, zero_division=0),
            "recall": recall_score(a, p, zero_division=0), "F1": f1_score(a, p, zero_division=0),
            "confusion": confusion_matrix(a, p, labels=[0, 1]).tolist()}


# --------------------------------------------------------------------------
# Tuning loop
# --------------------------------------------------------------------------
def tune_all(train: pd.DataFrame, features: list[str], n_iter: int = 30,
             only: list[str] | None = None, verbose: bool = True) -> dict:
    """RandomizedSearchCV for every model in the zoo with the expanding-window CV.
    Returns {name: {"search": fitted search, "cv_rmse": ..., "fit_s": ...}}."""
    X, y = train[features], train[TARGET]
    folds = expanding_window_cv(train)
    results = {}
    for name, (est, space) in model_zoo(features).items():
        if only and name not in only:
            continue
        t0 = time.time()
        search = RandomizedSearchCV(est, space, n_iter=n_iter, cv=folds,
                                    scoring="neg_root_mean_squared_error",
                                    random_state=RS, n_jobs=-1,
                                    refit=True, error_score="raise")
        search.fit(X, y)
        results[name] = {"search": search, "cv_rmse": -search.best_score_,
                         "cv_rmse_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                         "fit_s": time.time() - t0}
        if verbose:
            print(f"{name:18s} CV RMSE {results[name]['cv_rmse']:7.1f} "
                  f"(+/- {results[name]['cv_rmse_std']:.1f})  [{results[name]['fit_s']:.0f}s]")
    return results


def evaluate(models: dict, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        p = model.predict(test[features])
        rows.append({"model": name, **regression_metrics(test[TARGET], p),
                     **{f"shortfall_{k}": v for k, v in shortfall_metrics(test, test[TARGET], p).items()
                        if k in ("precision", "recall", "F1")}})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def naive_baseline(test: pd.DataFrame) -> np.ndarray:
    """'Same as last same-type season' - the forecast to beat."""
    return test["prev_yield_same"].fillna(test["yield_mean3_same"]).fillna(test[TARGET].median()).to_numpy()
