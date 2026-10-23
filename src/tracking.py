"""MLflow experiment tracking.

Inside an Azure Machine Learning job, MLflow is already pointed at the
workspace (MLFLOW_TRACKING_URI and MLFLOW_RUN_ID are set by Azure ML), so the
parameters, metrics, tables, figures and model land on the job's page in
Azure ML studio. Locally, tracking is on only when MLFLOW_TRACKING_URI is set
(for example `az ml workspace show --query mlflow_tracking_uri -o tsv`).
"""
from __future__ import annotations

import json
import os

from . import config as C

EXPERIMENT = "paddy-yield"


def configure() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def in_azureml_job() -> bool:
    return bool(os.getenv("AZUREML_RUN_ID") or os.getenv("MLFLOW_RUN_ID"))


def log_training_run(metrics: dict, cv_table, test_table, final_model, features, df) -> str | None:
    if not configure():
        print("MLflow tracking off (set MLFLOW_TRACKING_URI, or run as an Azure ML job).")
        return None
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    if not in_azureml_job():  # in a job the run (and its experiment) already exists
        mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({"best_model": metrics["best_model"], "n_features": len(features),
                           "train_end": C.TRAIN_END, "n_train": metrics["n_train"], "n_test": metrics["n_test"]})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for _, r in cv_table.iterrows():  # every candidate's CV score, for comparison in the studio
            key = "cv_rmse__" + "".join(ch if ch.isalnum() else "_" for ch in r["model"])[:60]
            mlflow.log_metric(key, float(r["cv_rmse"]))
        mlflow.log_dict(json.loads(cv_table.to_json(orient="records")), "cv_results.json")
        mlflow.log_dict(json.loads(test_table.to_json(orient="records")), "test_metrics.json")
        if C.FIGURES.exists():
            mlflow.log_artifacts(str(C.FIGURES), artifact_path="figures")
        X = df[features].head(50)
        mlflow.sklearn.log_model(final_model, artifact_path="mlflow_model",
                                 signature=infer_signature(X, final_model.predict(X)),
                                 input_example=X.head(3), serialization_format="cloudpickle",
                                 code_paths=[str(C.ROOT / "src")])
        info = {"run_id": run.info.run_id, "tracking_uri": mlflow.get_tracking_uri()}
    (C.ARTIFACTS / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Logged MLflow run {info['run_id']}")
    return info["run_id"]
