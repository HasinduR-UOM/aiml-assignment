"""Data contract for the model table (pandera). Bad data fails here, before training.

    python -m src.contract          # validates data/processed/model_table.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

from . import config as C

SCHEMA = pa.DataFrameSchema(
    {
        "district": Column(str, Check.isin(list(C.DISTRICTS))),
        "season": Column(str, Check.isin(["Maha", "Yala"])),
        "season_year": Column(int, Check.in_range(1975, 2035)),
        # 0 means "not reported" in the DCS tables and must be cleaned to NaN first
        "yield_kg_ha": Column(float, Check.in_range(500, 9000), nullable=True),
        "sown_ha": Column(float, Check.ge(0), nullable=True),
        "rain_total": Column(float, Check.in_range(0, 6000), nullable=True),
        "longest_dry_spell": Column(float, Check.in_range(0, 184), nullable=True),
        "tmax_mean": Column(float, Check.in_range(10, 40), nullable=True),
        "fertilizer_crisis": Column(int, Check.isin([0, 1])),
    },
    unique=["district", "season", "season_year"],
    strict=False,
    coerce=True,
)

MAX_MISSING_WEATHER = float(os.getenv("CONTRACT_MAX_MISSING_WEATHER", "0.05"))  # share of rows allowed without weather


def validate(df: pd.DataFrame) -> pd.DataFrame:
    out = SCHEMA.validate(df, lazy=True)
    miss = df["rain_total"].isna().mean()
    if miss > MAX_MISSING_WEATHER:
        bad = df[df.rain_total.isna()].district.value_counts().head(8).to_dict()
        raise ValueError(f"{miss:.0%} of rows have no weather (limit {MAX_MISSING_WEATHER:.0%}). "
                         f"Rerun `python -m src.weather`. Most affected: {bad}")
    return out


if __name__ == "__main__":
    path = C.PROCESSED / "model_table.csv"
    try:
        validate(pd.read_csv(path))
    except (pa.errors.SchemaErrors, ValueError) as e:
        print(f"DATA CONTRACT FAILED for {path.name}:\n{getattr(e, 'failure_cases', e)}")
        sys.exit(1)
    print(f"Data contract passed: {path.name}")
