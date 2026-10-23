"""Pipeline step 1 (Azure ML): raw data -> validated model table.

    python -m aml.steps.prep --raw <raw folder> --out <output folder>

<raw folder> has the layout of data/raw/: dcs/*.xlsx|pdf, weather/<district>.csv, oni/oni.csv
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src import config as C
from src import contract, dcs, features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = Path(a.raw)
    for sub, dest in (("dcs", C.RAW_DCS), ("weather", C.RAW_WEATHER), ("oni", C.RAW_ONI)):
        if (raw / sub).exists():
            shutil.copytree(raw / sub, dest, dirs_exist_ok=True)
    print(f"Raw files: dcs={len(list(C.RAW_DCS.glob('*')))}, weather={len(list(C.RAW_WEATHER.glob('*.csv')))}")

    dcs.parse_all()
    table = features.main()
    contract.validate(pd.read_csv(C.PROCESSED / "model_table.csv"))
    print("Data contract passed")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(C.PROCESSED / "model_table.csv", out / "model_table.csv")
    print(f"Model table: {len(table)} rows -> {out / 'model_table.csv'}")


if __name__ == "__main__":
    main()
