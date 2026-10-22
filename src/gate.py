"""Quality gate: does the newly trained model (challenger) beat the current champion?

    python -m src.gate          # exit 0 = pass, 1 = fail; writes gate_report.md

The champion's metrics come from the MLflow model registry
(models:/paddy-yield@champion) when tracking is configured, otherwise from
champion_metrics.json committed in the repo. The comparison metric and the
tolerance are set in params.yaml (default: test RMSE on normal seasons,
lower is better).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from . import config as C
from .tracking import MODEL_NAME, configure

LOCAL_CHAMPION = C.ROOT / "champion_metrics.json"


def champion_metrics() -> dict | None:
    if configure():
        try:
            from mlflow import MlflowClient
            client = MlflowClient()
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            return client.get_run(mv.run_id).data.metrics | {"version": mv.version}
        except Exception as e:  # noqa: BLE001  (no registered model / alias yet)
            print(f"No champion in the MLflow registry yet ({type(e).__name__}).")
            return None
    if LOCAL_CHAMPION.exists():
        return json.loads(LOCAL_CHAMPION.read_text())
    return None


def main() -> int:
    params = yaml.safe_load(Path(C.ROOT / "params.yaml").read_text())["gate"]
    metric, tol = params["metric"], float(params["tolerance"])
    challenger = json.loads((C.ARTIFACTS / "metrics.json").read_text())
    champion = champion_metrics()

    rows = ["test_rmse_normal", "test_mape_normal", "test_r2_normal", "test_rmse", "cv_rmse"]
    lines = ["## Quality gate", "", f"Rule: challenger `{metric}` <= champion + {tol} (lower is better)", "",
             "| Metric | Champion | Challenger |", "|---|---|---|"]
    for m in rows:
        c = champion.get(m) if champion else None
        lines.append(f"| {m} | {'none yet' if c is None else round(float(c), 3)} | {challenger.get(m)} |")

    if champion is None or champion.get(metric) is None:
        passed, why = True, "No champion yet, so the first model passes."
    else:
        passed = float(challenger[metric]) <= float(champion[metric]) + tol
        why = (f"{metric}: challenger {challenger[metric]} vs champion {float(champion[metric]):.2f}. "
               + ("PASS" if passed else "FAIL: the new model is worse, keep the champion."))
    lines += ["", f"**Result: {'PASS' if passed else 'FAIL'}** - {why}",
              "", f"Challenger: {challenger['best_model']}"]
    report = "\n".join(lines)
    (C.ROOT / "gate_report.md").write_text(report + "\n")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
