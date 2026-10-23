"""Send demo traffic to the deployed model so the drift monitor has data to check.

Azure ML endpoint:
    URI=$(az ml online-endpoint show -n paddy-yield-ep --query scoring_uri -o tsv)
    KEY=$(az ml online-endpoint get-credentials -n paddy-yield-ep --query primaryKey -o tsv)
    python -m monitoring.send_traffic "$URI" --key "$KEY" --n 300 --mode normal
    python -m monitoring.send_traffic "$URI" --key "$KEY" --n 300 --mode drought

Local FastAPI service:
    python -m monitoring.send_traffic http://localhost:8000 --n 50

normal  = rainfall scattered around the long-term normal
drought = rainfall 30-60 % of normal, the kind of shift the drift check should catch
"""
from __future__ import annotations

import argparse
import random

import requests

from src import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Azure ML scoring URI (ends in /score) or base URL of the FastAPI service")
    ap.add_argument("--key", default=None, help="endpoint key (Azure ML)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=["normal", "drought"], default="normal")
    a = ap.parse_args()
    url = a.url if a.url.rstrip("/").endswith("/score") else a.url.rstrip("/") + "/predict"
    headers = {"Content-Type": "application/json"}
    if a.key:
        headers["Authorization"] = f"Bearer {a.key}"
    rng = random.Random(42)
    ok = 0
    for _ in range(a.n):
        rain = rng.uniform(30, 60) if a.mode == "drought" else rng.gauss(100, 20)
        body = {"district": rng.choice(list(C.DISTRICTS)), "season": rng.choice(["Maha", "Yala"]),
                "rain_pct_of_normal": max(0.0, min(300.0, rain))}
        r = requests.post(url, json=body, headers=headers, timeout=60)
        ok += r.status_code == 200
    print(f"sent {a.n} requests to {url}, {ok} OK")


if __name__ == "__main__":
    main()
