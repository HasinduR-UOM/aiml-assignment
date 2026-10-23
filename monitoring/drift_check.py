"""Data-drift check: are the inputs the model now sees different from what it trained on?

    python -m monitoring.drift_check                       # latest seasons in the model table
    python -m monitoring.drift_check --current <csv | folder of collected .jsonl> \
        --reference <model_table.csv | folder> --out <folder>

In Azure ML the weekly schedule (aml/monitor-schedule.yml) passes the folder
the endpoint's data collector writes to, and the registered model table.

Reference = training seasons (season_year <= TRAIN_END). For each numeric input
a two-sample Kolmogorov-Smirnov test asks whether the current values could come
from the same distribution. If the share of drifted features reaches the
threshold in params.yaml, the check recommends retraining. If the current rows
also carry the real yield, the model's RMSE on them is reported too.

Outputs drift_report.md and drift_summary.json in --out (default monitoring/),
and logs drift_share and per-feature KS statistics to MLflow when tracking is on.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

from src import config as C
from src.features import TARGET, WEATHER_FEATURES

OUT = C.ROOT / "monitoring"


def read_current(path: str) -> pd.DataFrame:
    """CSV file, or a folder of data-collector JSON lines (each line holds a "data" list of rows)."""
    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    rows = []
    for f in sorted(p.rglob("*.jsonl")):
        if "model_outputs" in f.parts:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            data = rec.get("data", rec)
            rows.extend(data if isinstance(data, list) else [data])
    if not rows:
        raise SystemExit(f"No collected inputs found under {p}")
    return pd.DataFrame(rows)


def read_reference(path: str | None) -> pd.DataFrame:
    p = Path(path) if path else C.PROCESSED / "model_table.csv"
    return pd.read_csv(next(p.rglob("model_table.csv")) if p.is_dir() else p)


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    return head + "\n" + body


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=None, help="CSV or folder of collected inputs; default = latest seasons")
    ap.add_argument("--reference", default=None, help="model_table.csv or a folder containing it")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--last-seasons", type=int, default=4, help="when no CSV: how many recent seasons to test")
    a = ap.parse_args()
    p = yaml.safe_load((C.ROOT / "params.yaml").read_text())["monitor"]

    table = read_reference(a.reference)
    ref = table[table.season_year <= C.TRAIN_END]
    if a.current:
        cur = read_current(a.current)
    else:
        t = table.assign(t=table.season_year + table.season.map({"Maha": .75, "Yala": .25}))
        recent = sorted(t.t.unique())[-a.last_seasons:]
        cur = t[t.t.isin(recent)]

    rows = []
    for f in WEATHER_FEATURES:
        if f in cur and cur[f].notna().sum() >= 10:
            stat, pv = ks_2samp(ref[f].dropna(), cur[f].dropna())
            rows.append({"feature": f, "ks_stat": round(float(stat), 3), "p_value": round(float(pv), 4),
                         "ref_mean": round(float(ref[f].mean()), 1), "cur_mean": round(float(cur[f].mean()), 1),
                         "drift": bool(pv < p["p_value"])})
    rep = pd.DataFrame(rows)
    share = float(rep.drift.mean()) if len(rep) else 0.0
    summary = {"n_reference": int(len(ref)), "n_current": int(len(cur)), "features_tested": int(len(rep)),
               "drift_share": round(share, 3), "threshold": p["drift_share_threshold"],
               "retrain": share >= p["drift_share_threshold"]}

    if TARGET in cur and cur[TARGET].notna().sum() >= 10 and (C.MODELS / "final_model.joblib").exists():
        meta = json.loads((C.ARTIFACTS / "app_metadata.json").read_text())
        model = joblib.load(C.MODELS / "final_model.joblib")
        c = cur[cur[TARGET].notna()]
        err = model.predict(c[meta["features"]]) - c[TARGET]
        summary["current_rmse"] = round(float(np.sqrt((err ** 2).mean())), 1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    md = ["## Drift report", "", f"Reference: {len(ref)} training rows; current: {len(cur)} rows.", "",
          _md_table(rep) if len(rep) else "(no features with enough data)", "",
          f"Drift share {share:.0%} (threshold {p['drift_share_threshold']:.0%}) -> "
          f"**{'RETRAIN recommended' if summary['retrain'] else 'no action'}**"]
    if "current_rmse" in summary:
        md.append(f"\nModel RMSE on current rows with known yield: {summary['current_rmse']} kg/ha")
    (out / "drift_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow
        with mlflow.start_run():
            mlflow.log_metrics({"drift_share": share, "n_current": len(cur), "retrain": int(summary["retrain"])})
            mlflow.log_metrics({f"ks__{r['feature']}": r["ks_stat"] for r in rows})
            mlflow.log_artifact(str(out / "drift_report.md"))
    return summary


if __name__ == "__main__":
    main()
