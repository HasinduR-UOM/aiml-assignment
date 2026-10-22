"""MLflow experiment tracking on DagsHub (optional).

Enabled when either MLFLOW_TRACKING_URI is set, or DAGSHUB_OWNER and
DAGSHUB_TOKEN are set (then the DagsHub MLflow server of DAGSHUB_REPO is used).
Each training run logs parameters, CV/test metrics, the tables, the figures
folder and the final model, and writes artifacts/run_info.json with the run id
so the gate and the registry can find the model later.
"""
from __future__ import annotations

import json
import os

from . import config as C

EXPERIMENT = "paddy-yield"
MODEL_NAME = "paddy-yield"


def configure() -> bool:
    """Point MLflow at DagsHub if the env vars are present. Returns True if tracking is on."""
    owner, token = os.getenv("DAGSHUB_OWNER"), os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO", "paddy-yield-sl")
    if not os.getenv("MLFLOW_TRACKING_URI") and owner and token:
        os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{owner}/{repo}.mlflow"
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", owner)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def log_training_run(metrics: dict, cv_table, test_table, final_model, features, df) -> str | None:
    if not configure():
        print("MLflow tracking off (set DAGSHUB_OWNER / DAGSHUB_TOKEN to enable).")
        return None
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=metrics["best_model"]) as run:
        mlflow.log_params({"best_model": metrics["best_model"], "n_features": len(features),
                           "train_end": C.TRAIN_END, "n_train": metrics["n_train"], "n_test": metrics["n_test"]})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for _, r in cv_table.iterrows():  # every candidate's CV score, for comparison in the UI
            key = "cv_rmse__" + "".join(ch if ch.isalnum() else "_" for ch in r["model"])[:60]
            mlflow.log_metric(key, float(r["cv_rmse"]))
        mlflow.log_dict(json.loads(cv_table.to_json(orient="records")), "cv_results.json")
        mlflow.log_dict(json.loads(test_table.to_json(orient="records")), "test_metrics.json")
        if C.FIGURES.exists():
            mlflow.log_artifacts(str(C.FIGURES), artifact_path="figures")
        X = df[features].head(50)
        mi = mlflow.sklearn.log_model(final_model, artifact_path="model",
                                      signature=infer_signature(X, final_model.predict(X)),
                                      input_example=X.head(3), serialization_format="cloudpickle")
        info = {"run_id": run.info.run_id, "model_uri": mi.model_uri,
                "tracking_uri": mlflow.get_tracking_uri()}
    (C.ARTIFACTS / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Logged MLflow run {info['run_id']}")
    return info["run_id"]
