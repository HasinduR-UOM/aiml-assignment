"""Send demo traffic to the deployed API so the monitor has prediction logs to check.

    python -m monitoring.send_traffic https://<app-fqdn> --n 300 --mode normal
    python -m monitoring.send_traffic https://<app-fqdn> --n 300 --mode drought   # simulated dry year

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
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", choices=["normal", "drought"], default="normal")
    a = ap.parse_args()
    rng = random.Random(42)
    ok = 0
    for _ in range(a.n):
        rain = rng.uniform(30, 60) if a.mode == "drought" else rng.gauss(100, 20)
        body = {"district": rng.choice(list(C.DISTRICTS)), "season": rng.choice(["Maha", "Yala"]),
                "rain_pct_of_normal": max(0.0, min(300.0, rain))}
        r = requests.post(a.url.rstrip("/") + "/predict", json=body, timeout=30)
        ok += r.status_code == 200
    print(f"sent {a.n} requests, {ok} OK")


if __name__ == "__main__":
    main()
