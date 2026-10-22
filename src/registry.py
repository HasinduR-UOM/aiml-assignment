"""Model registry: register the latest run's model and make it the champion.

    python -m src.registry promote     # needs artifacts/run_info.json from a tracked training run
    python -m src.registry show        # print the current champion

Without MLflow tracking, `promote` just copies artifacts/metrics.json to
champion_metrics.json (commit that file) so the gate still has a reference.
"""
from __future__ import annotations

import json
import shutil
import sys

from . import config as C
from .tracking import MODEL_NAME, configure


def promote() -> None:
    if not configure():
        shutil.copy(C.ARTIFACTS / "metrics.json", C.ROOT / "champion_metrics.json")
        print("Tracking off: wrote champion_metrics.json (commit it).")
        return
    import mlflow
    from mlflow import MlflowClient

    info = json.loads((C.ARTIFACTS / "run_info.json").read_text())
    client = MlflowClient()
    try:  # already the champion? then nothing to do (CD calls this on every deploy)
        cur = client.get_model_version_by_alias(MODEL_NAME, "champion")
        if cur.run_id == info["run_id"]:
            print(f"Run {info['run_id']} is already {MODEL_NAME} @champion (version {cur.version}).")
            return
    except Exception:  # noqa: BLE001  no champion yet
        pass
    mv = mlflow.register_model(info["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    metrics = json.loads((C.ARTIFACTS / "metrics.json").read_text())
    client.set_model_version_tag(MODEL_NAME, mv.version, "best_model", metrics["best_model"])
    print(f"Registered {MODEL_NAME} version {mv.version} and set alias @champion")


def show() -> None:
    if not configure():
        print(open(C.ROOT / "champion_metrics.json").read() if (C.ROOT / "champion_metrics.json").exists()
              else "No champion yet.")
        return
    from mlflow import MlflowClient
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"{MODEL_NAME} @champion = version {mv.version} (run {mv.run_id})")


if __name__ == "__main__":
    {"promote": promote, "show": show}.get(sys.argv[1] if len(sys.argv) > 1 else "show", show)()
