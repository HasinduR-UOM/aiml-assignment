"""Turn raw paddy statistics + daily weather into one modelling table.

One row = one district in one season (Maha or Yala). Every feature is
something known BEFORE harvest, so the model can be used as a pre-harvest
forecast. Harvested extent and production are kept only for EDA / the app,
never as model inputs (they would leak the outcome).

    python -m src.features             # -> data/processed/model_table.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

TARGET = "yield_kg_ha"
CAT_FEATURES = ["district", "season"]
WEATHER_FEATURES = [
    "rain_total", "rain_establish", "rain_late", "rain_days", "rain_max5d",
    "longest_dry_spell", "tmax_mean", "tmin_mean", "hot_days", "et0_total",
    "water_balance", "rain_anom_pct", "antecedent_rain_6m",
]
HISTORY_FEATURES = ["prev_yield_same", "prev_yield_any", "yield_mean3_same",
                    "sown_ha", "sown_change_pct"]
OTHER_FEATURES = ["season_year", "fertilizer_crisis"]
OPTIONAL_FEATURES = ["oni_mean", "minor_share", "rainfed_share"]

FEATURE_DESCRIPTIONS = {
    "district": "District (categorical)",
    "season": "Maha (Oct-Mar, NE monsoon) or Yala (Apr-Sep, SW monsoon)",
    "season_year": "Year the season starts - captures the long-run technology trend",
    "fertilizer_crisis": "1 for Maha 2021/22 and Yala 2022 (fertilizer import ban / shortage)",
    "rain_total": "Total rainfall in the 6-month season window (mm)",
    "rain_establish": "Rainfall in the first 60 days - land preparation and crop establishment (mm)",
    "rain_late": "Rainfall in the last 45 days - ripening / harvest damage (mm)",
    "rain_days": "Days with >= 1 mm rain",
    "rain_max5d": "Largest 5-day rainfall total - flood proxy (mm)",
    "longest_dry_spell": "Longest run of consecutive days < 1 mm - drought stress proxy",
    "tmax_mean": "Mean daily maximum temperature (C)",
    "tmin_mean": "Mean daily minimum temperature (C)",
    "hot_days": "Days with Tmax >= 35 C (heat stress at flowering)",
    "et0_total": "FAO-56 reference evapotranspiration over the season (mm)",
    "water_balance": "rain_total - et0_total (mm); negative = crop water deficit",
    "rain_anom_pct": "Season rainfall as % departure from the 1981-2010 district normal",
    "antecedent_rain_6m": "Rainfall in the 6 months before the season - reservoir storage proxy",
    "prev_yield_same": "District yield in the previous season of the same type (kg/ha)",
    "prev_yield_any": "District yield in the immediately preceding season (kg/ha)",
    "yield_mean3_same": "Mean district yield over the last 3 seasons of the same type (kg/ha)",
    "sown_ha": "Gross extent sown this season (ha) - known at planting",
    "sown_change_pct": "% change in sown extent vs previous same-type season",
    "oni_mean": "Mean Oceanic Nino Index over the season (ENSO state)",
    "minor_share": "Share of sown extent under minor irrigation (2004+ only)",
    "rainfed_share": "Share of sown extent that is rainfed (2004+ only)",
}


# --------------------------------------------------------------------------
# Weather -> seasonal features
# --------------------------------------------------------------------------
def _season_window(season: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = C.SEASONS[season]
    start = pd.Timestamp(year=year, month=s["start_month"], day=1)
    end = start + pd.DateOffset(months=s["n_months"]) - pd.Timedelta(days=1)
    return start, end


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def season_weather(daily: pd.DataFrame, season: str, year: int) -> dict:
    start, end = _season_window(season, year)
    w = daily[(daily.date >= start) & (daily.date <= end)]
    if len(w) < 150:           # incomplete season (e.g. still in progress)
        return {}
    pre = daily[(daily.date >= start - pd.DateOffset(months=6)) & (daily.date < start)]
    rain = w["rain_mm"].fillna(0).to_numpy()
    est = w[w.date < start + pd.Timedelta(days=60)]["rain_mm"].sum()
    late = w[w.date > end - pd.Timedelta(days=45)]["rain_mm"].sum()
    return {
        "rain_total": rain.sum(),
        "rain_establish": est,
        "rain_late": late,
        "rain_days": int((rain >= 1).sum()),
        "rain_max5d": pd.Series(rain).rolling(5).sum().max(),
        "longest_dry_spell": _longest_run(rain < 1),
        "tmax_mean": w["tmax_c"].mean(),
        "tmin_mean": w["tmin_c"].mean(),
        "hot_days": int((w["tmax_c"] >= 35).sum()),
        "et0_total": w["et0_mm"].sum(),
        "antecedent_rain_6m": pre["rain_mm"].sum() if len(pre) > 150 else np.nan,
    }


def build_weather_table(daily_by_district: dict[str, pd.DataFrame],
                        years: range) -> pd.DataFrame:
    rows = []
    for d, daily in daily_by_district.items():
        for season in C.SEASONS:
            for y in years:
                f = season_weather(daily, season, y)
                if f:
                    rows.append({"district": d, "season": season, "season_year": y, **f})
    wt = pd.DataFrame(rows)
    wt["water_balance"] = wt["rain_total"] - wt["et0_total"]
    base = wt[wt.season_year.between(*C.CLIMATE_BASE)]
    normal = base.groupby(["district", "season"])["rain_total"].mean().rename("rain_normal")
    wt = wt.merge(normal, on=["district", "season"], how="left")
    wt["rain_anom_pct"] = 100 * (wt["rain_total"] / wt["rain_normal"] - 1)
    return wt.drop(columns="rain_normal")


def add_oni(table: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    oni = oni.assign(ym=oni.year * 12 + oni.month - 1).set_index("ym")["oni"]
    vals = []
    for s, y in zip(table.season, table.season_year):
        m0 = y * 12 + C.SEASONS[s]["start_month"] - 1
        vals.append(oni.reindex(range(m0, m0 + 6)).mean())
    return table.assign(oni_mean=vals)


# --------------------------------------------------------------------------
# History features (strictly from earlier seasons)
# --------------------------------------------------------------------------
def add_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t"] = df.season_year + df.season.map({k: v["t_offset"] for k, v in C.SEASONS.items()})
    df = df.sort_values(["district", "t"])
    g_same = df.groupby(["district", "season"])
    df["prev_yield_same"] = g_same[TARGET].shift(1)
    df["yield_mean3_same"] = g_same[TARGET].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    df["yield_mean5_same"] = g_same[TARGET].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df["prev_sown_same"] = g_same["sown_ha"].shift(1)
    df["sown_change_pct"] = 100 * (df["sown_ha"] / df["prev_sown_same"].replace(0, np.nan) - 1)
    df["prev_yield_any"] = df.groupby("district")[TARGET].shift(1)
    df["fertilizer_crisis"] = [int((s, y) in C.FERTILIZER_CRISIS)
                               for s, y in zip(df.season, df.season_year)]
    return df.drop(columns="prev_sown_same")


def build_model_table(paddy: pd.DataFrame, weather: pd.DataFrame,
                      oni: pd.DataFrame | None = None,
                      irrigation: pd.DataFrame | None = None) -> pd.DataFrame:
    # DCS tables use 0 for "not reported" (e.g. conflict years) and for seasons
    # that are not finished yet. Treat those as missing BEFORE computing lags.
    paddy = paddy.copy()
    for c in ("yield_kg_ha", "production_mt"):
        if c in paddy:
            paddy.loc[paddy[c] <= 0, c] = np.nan
    for c in ("sown_ha", "harvested_ha", "production_mt"):
        if c not in paddy:
            paddy = paddy.assign(**{c: np.nan})
    df = paddy.merge(weather, on=["district", "season", "season_year"], how="left")
    df = add_history(df)
    if oni is not None:
        df = add_oni(df, oni)
    if irrigation is not None and len(irrigation):
        irr = irrigation.assign(
            minor_share=irrigation.sown_minor_ha / irrigation.sown_all_ha,
            rainfed_share=irrigation.sown_rainfed_ha / irrigation.sown_all_ha,
        )[["district", "season", "season_year", "minor_share", "rainfed_share"]]
        df = df.merge(irr, on=["district", "season", "season_year"], how="left")
    df = df[df[TARGET].notna()].sort_values(["t", "district"]).reset_index(drop=True)
    return df


def feature_list(df: pd.DataFrame) -> list[str]:
    """Features used by the models: core set + optional ones that are mostly present."""
    feats = CAT_FEATURES + OTHER_FEATURES + HISTORY_FEATURES + WEATHER_FEATURES
    for f in OPTIONAL_FEATURES:
        if f in df and df[f].notna().mean() > 0.9:
            feats.append(f)
    return feats


# --------------------------------------------------------------------------
def main() -> pd.DataFrame:
    paddy = pd.read_csv(C.INTERIM / "paddy_district_long.csv")
    paddy = paddy[paddy.district.isin(C.DISTRICTS)]
    daily = {}
    for name in C.DISTRICTS:
        p = C.RAW_WEATHER / f"{name.replace(' ', '_')}.csv"
        if p.exists():
            daily[name] = pd.read_csv(p, parse_dates=["date"])
    if not daily:
        raise SystemExit("No weather files found - run `python -m src.weather` first.")
    years = range(int(paddy.season_year.min()), int(paddy.season_year.max()) + 1)
    weather = build_weather_table(daily, years)
    oni_path = C.RAW_ONI / "oni.csv"
    oni = pd.read_csv(oni_path) if oni_path.exists() else None
    irr_path = C.INTERIM / "paddy_irrigation_split.csv"
    irr = pd.read_csv(irr_path) if irr_path.exists() else None
    table = build_model_table(paddy, weather, oni, irr)
    out = C.PROCESSED / "model_table.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(C.ROOT)}: {len(table)} rows x {table.shape[1]} cols, "
          f"{table.district.nunique()} districts, seasons {table.season_year.min()}-{table.season_year.max()}")
    print("Model features:", ", ".join(feature_list(table)))
    return table


if __name__ == "__main__":
    main()
