"""End-to-end training script (the notebooks walk through the same steps).

    python -m src.train                    # 30 random-search draws per model
    python -m src.train --n-iter 10        # faster run

Outputs (artifacts/):
    models/<model>.joblib       tuned on train (<= 2015) - used for test evaluation
    models/final_model.joblib   best model re-fitted on ALL seasons - used by the app
    cv_results.csv, test_metrics.csv, test_predictions.csv, app_metadata.json
"""
from __future__ import annotations

import argparse
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config as C
from .features import FEATURE_DESCRIPTIONS, TARGET, WEATHER_FEATURES, feature_list
from .modeling import (
    evaluate,
    naive_baseline,
    regression_metrics,
    temporal_split,
    tune_all,
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_table() -> pd.DataFrame:
    path = C.PROCESSED / "model_table.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m src.features`")
    return pd.read_csv(path)


def build_app_metadata(df: pd.DataFrame, features: list[str], best_name: str,
                       cv_table: pd.DataFrame, test_table: pd.DataFrame) -> dict:
    defaults = {}
    for (d, s), g in df.sort_values("t").groupby(["district", "season"]):
        g_obs = g[g[TARGET].notna()]
        last = g_obs.iloc[-1]
        base = g[g.season_year.between(*C.CLIMATE_BASE)]
        clim = base[WEATHER_FEATURES].median(numeric_only=True)
        clim_q = {f: [float(base[f].quantile(.05)), float(base[f].quantile(.95))] for f in WEATHER_FEATURES
                  if f in base and base[f].notna().any()}
        harvest_ratio = (g["harvested_ha"] / g["sown_ha"]).median() if "harvested_ha" in g else np.nan
        defaults[f"{d}|{s}"] = {
            "last_season_year": int(last.season_year),
            "prev_yield_same": float(last[TARGET]),
            "yield_mean3_same": float(g_obs[TARGET].tail(3).mean()),
            "yield_mean5_same": float(g_obs[TARGET].tail(5).mean()),
            "sown_ha": float(last.get("sown_ha", np.nan)),
            "harvest_ratio": float(harvest_ratio) if pd.notna(harvest_ratio) else 0.95,
            "climatology": {k: float(v) for k, v in clim.items() if pd.notna(v)},
            "climatology_range": clim_q,
            "history": g_obs[["season_year", TARGET]].tail(15).values.tolist(),
        }
        if "oni_mean" in features:
            defaults[f"{d}|{s}"]["climatology"]["oni_mean"] = 0.0
    last_any = (df[df[TARGET].notna()].sort_values("t").groupby("district").tail(1)
                .set_index("district")[TARGET].to_dict())
    return {
        "features": features,
        "feature_descriptions": {f: FEATURE_DESCRIPTIONS.get(f, f) for f in features},
        "districts": sorted(df.district.unique()),
        "best_model": best_name,
        "defaults": defaults,
        "last_yield_any": {k: float(v) for k, v in last_any.items()},
        "cv_table": cv_table.to_dict(orient="records"),
        "test_table": test_table.to_dict(orient="records"),
        "train_end": C.TRAIN_END,
        "shortfall_threshold": C.SHORTFALL_THRESHOLD,
        "trained_on": f"{int(df.season_year.min())}-{int(df.season_year.max())}",
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }


def summary_metrics(best_name, cv_table, preds, best_slug, n_train, n_test) -> dict:
    """Headline numbers for DVC metrics, the quality gate and MLflow."""
    col = f"pred_{best_slug}"
    crisis = preds.apply(lambda r: (r.season, int(r.season_year)) in C.FERTILIZER_CRISIS, axis=1)
    allm = regression_metrics(preds[TARGET], preds[col])
    norm = regression_metrics(preds.loc[~crisis, TARGET], preds.loc[~crisis, col])
    row = cv_table.set_index("model").loc[best_name]
    return {
        "best_model": best_name,
        "cv_rmse": round(float(row["cv_rmse"]), 2),
        "cv_rmse_std": round(float(row["cv_rmse_std"]), 2),
        "test_rmse": round(allm["RMSE"], 2), "test_mae": round(allm["MAE"], 2),
        "test_mape": round(allm["MAPE_%"], 3), "test_r2": round(allm["R2"], 4),
        "test_rmse_normal": round(norm["RMSE"], 2), "test_mape_normal": round(norm["MAPE_%"], 3),
        "test_r2_normal": round(norm["R2"], 4),
        "n_train": int(n_train), "n_test": int(n_test),
    }


def run(n_iter: int = 30, only: list[str] | None = None) -> dict:
    df = load_table()
    features = feature_list(df)
    train, test = temporal_split(df)
    print(f"Train {len(train)} rows (<= {C.TRAIN_END}), test {len(test)} rows (> {C.TRAIN_END}); "
          f"{len(features)} features")

    results = tune_all(train, features, n_iter=n_iter, only=only)
    fitted = {name: r["search"].best_estimator_ for name, r in results.items()}

    cv_table = pd.DataFrame([{"model": k, "cv_rmse": v["cv_rmse"], "cv_rmse_std": v["cv_rmse_std"],
                              "tuning_seconds": v["fit_s"],
                              "best_params": json.dumps({p.split("__")[-1]: (round(x, 5) if isinstance(x, float) else x)
                                                         for p, x in v["search"].best_params_.items()}, default=str)
                              if v["search"] is not None else "members tuned above"}
                             for k, v in results.items()]).sort_values("cv_rmse").reset_index(drop=True)

    test_table = evaluate(fitted, test, features)
    naive = {"model": "Naive (last same-type season)", **regression_metrics(test[TARGET], naive_baseline(test))}
    test_table = pd.concat([test_table, pd.DataFrame([naive])], ignore_index=True)

    best_name = cv_table.iloc[0]["model"]      
    print(f"\nBest model by CV RMSE: {best_name}")
    print(test_table.round(3).to_string(index=False))

    # Save
    for name, m in fitted.items():
        joblib.dump(m, C.MODELS / f"{slug(name)}.joblib")
    preds = test[["district", "season", "season_year", TARGET, "yield_mean5_same"]].copy()
    for name, m in fitted.items():
        preds[f"pred_{slug(name)}"] = m.predict(test[features])
    preds.to_csv(C.ARTIFACTS / "test_predictions.csv", index=False)
    cv_table.to_csv(C.ARTIFACTS / "cv_results.csv", index=False)
    test_table.to_csv(C.ARTIFACTS / "test_metrics.csv", index=False)

    final = clone(fitted[best_name]).fit(df[features], df[TARGET])
    joblib.dump(final, C.MODELS / "final_model.joblib")
    joblib.dump(fitted[best_name], C.MODELS / "best_model_train_only.joblib")
    train[features + [TARGET]].to_csv(C.ARTIFACTS / "train_sample.csv", index=False)
    meta = build_app_metadata(df, features, best_name, cv_table, test_table)
    (C.ARTIFACTS / "app_metadata.json").write_text(json.dumps(meta, indent=1, default=float))
    metrics = summary_metrics(best_name, cv_table, preds, slug(best_name), len(train), len(test))
    (C.ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models and metadata to {C.ARTIFACTS.relative_to(C.ROOT)}/")
    try:  # MLflow tracking is optional: only runs when MLFLOW_TRACKING_URI / DagsHub env vars are set
        from .tracking import log_training_run
        log_training_run(metrics, cv_table, test_table, final, features, df)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001  tracking must never break training
        print(f"WARNING: MLflow logging failed: {e}")
    return {"df": df, "train": train, "test": test, "features": features, "results": results,
            "fitted": fitted, "cv_table": cv_table, "test_table": test_table, "best_name": best_name}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    a = ap.parse_args()
    run(a.n_iter)
