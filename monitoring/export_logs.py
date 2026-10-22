"""Export prediction logs from Azure Log Analytics into a CSV of model inputs.

    az login        # or run inside GitHub Actions after azure/login
    python -m monitoring.export_logs --hours 168 --out monitoring/current.csv

Needs the workspace id (LOG_ANALYTICS_WORKSPACE_ID env var). The API prints one
JSON line per prediction; Container Apps stores console output in the
ContainerAppConsoleLogs_CL table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import pandas as pd

QUERY = """ContainerAppConsoleLogs_CL
| where TimeGenerated > ago({hours}h)
| where Log_s has '"event": "prediction"'
| project TimeGenerated, Log_s"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=168)
    ap.add_argument("--out", default="monitoring/current.csv")
    a = ap.parse_args()
    ws = os.environ["LOG_ANALYTICS_WORKSPACE_ID"]
    res = subprocess.run(["az", "monitor", "log-analytics", "query", "-w", ws, "--analytics-query",
                          QUERY.format(hours=a.hours), "-o", "json"], capture_output=True, text=True, check=True)
    rows = []
    for r in json.loads(res.stdout):
        try:
            ev = json.loads(r["Log_s"])
            rows.append({"timestamp": r["TimeGenerated"], "request_id": ev["request_id"],
                         "prediction": ev["prediction"], **ev["features"]})
        except (KeyError, json.JSONDecodeError):
            continue
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"Exported {len(rows)} predictions to {a.out}")


if __name__ == "__main__":
    main()
