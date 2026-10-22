"""Data-drift check: are the inputs the model now sees different from what it trained on?

    python -m monitoring.drift_check                         # latest season(s) in the model table
    python -m monitoring.drift_check --current monitoring/current.csv   # rows exported from the API logs

Reference = training seasons (season_year <= TRAIN_END). For each numeric input
a two-sample Kolmogorov-Smirnov test asks whether the current values could come
from the same distribution. If the share of drifted features reaches the
threshold in params.yaml, the check recommends retraining. If the current rows
also carry the real yield, the model's RMSE on them is reported too.

Outputs monitoring/drift_report.md and monitoring/drift_summary.json, and sets
the GitHub Actions output `retrain=true|false` when run inside Actions.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

from src import config as C
from src.features import TARGET, WEATHER_FEATURES

OUT = C.ROOT / "monitoring"


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    return head + "\n" + body


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=None, help="CSV of current inputs; default = seasons after the last 2")
    ap.add_argument("--last-seasons", type=int, default=4, help="when no CSV: how many recent seasons to test")
    a = ap.parse_args()
    p = yaml.safe_load((C.ROOT / "params.yaml").read_text())["monitor"]

    table = pd.read_csv(C.PROCESSED / "model_table.csv")
    ref = table[table.season_year <= C.TRAIN_END]
    if a.current:
        cur = pd.read_csv(a.current)
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

    if TARGET in cur and cur[TARGET].notna().sum() >= 10:
        meta = json.loads((C.ARTIFACTS / "app_metadata.json").read_text())
        model = joblib.load(C.MODELS / "final_model.joblib")
        c = cur[cur[TARGET].notna()]
        err = model.predict(c[meta["features"]]) - c[TARGET]
        summary["current_rmse"] = round(float(np.sqrt((err ** 2).mean())), 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    md = ["## Drift report", "", f"Reference: {len(ref)} training rows; current: {len(cur)} rows.", "",
          _md_table(rep) if len(rep) else "(no features with enough data)", "",
          f"Drift share {share:.0%} (threshold {p['drift_share_threshold']:.0%}) -> "
          f"**{'RETRAIN recommended' if summary['retrain'] else 'no action'}**"]
    if "current_rmse" in summary:
        md.append(f"\nModel RMSE on current rows with known yield: {summary['current_rmse']} kg/ha")
    (OUT / "drift_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"retrain={'true' if summary['retrain'] else 'false'}\n")
    return summary


if __name__ == "__main__":
    main()
