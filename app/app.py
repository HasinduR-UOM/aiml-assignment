"""Paddy Yield Early-Warning - Streamlit app.

    streamlit run app/app.py

Loads artifacts/models/final_model.joblib and artifacts/app_metadata.json
(written by `python -m src.train`), takes a district / season and a weather
scenario, predicts yield, flags a shortfall and explains the prediction with
SHAP and LIME.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import explain as E  # noqa: E402

ART = ROOT / "artifacts"
st.set_page_config(page_title="Paddy Yield Early Warning - Sri Lanka", page_icon="🌾", layout="wide")


@st.cache_resource
def load():
    meta = json.loads((ART / "app_metadata.json").read_text())
    model = joblib.load(ART / "models" / "final_model.joblib")
    train = pd.read_csv(ART / "train_sample.csv")
    return meta, model, train


@st.cache_resource
def lime_bundle(_model, _train, features):
    return E.lime_explainer(_model, _train[features], features)


try:
    meta, model, train = load()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python -m src.train` first.")
    st.stop()

F = meta["features"]
DESC = meta["feature_descriptions"]
WEATHER_RAIN = ["rain_total", "rain_establish", "rain_late", "rain_max5d"]


st.title("🌾 Paddy Yield Early Warning — Sri Lanka")
st.caption(f"District-level pre-harvest yield forecast · model: **{meta['best_model']}** · "
           f"trained on seasons {meta['trained_on']} · DCS paddy statistics + ERA5 weather")

tab_fc, tab_models, tab_about = st.tabs(["Forecast", "Model comparison", "About"])

# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Season")
    district = st.selectbox("District", meta["districts"],
                            index=meta["districts"].index("Polonnaruwa") if "Polonnaruwa" in meta["districts"] else 0)
    season = st.radio("Season", ["Maha", "Yala"], horizontal=True,
                      help="Maha: Oct–Mar (north-east monsoon). Yala: Apr–Sep (south-west monsoon).")
    key = f"{district}|{season}"
    if key not in meta["defaults"]:
        st.error("No history for this district/season.")
        st.stop()
    dflt = meta["defaults"][key]
    year = st.number_input("Season start year", 1980, 2035, value=dflt["last_season_year"] + 1)

    st.header("Weather scenario")
    preset = st.radio("Rainfall vs the 1981–2010 normal",
                      ["Normal", "Dry (−30 %)", "Severe drought (−50 %)", "Wet (+30 %)", "Custom"], index=0)
    scale = {"Normal": 1.0, "Dry (−30 %)": 0.7, "Severe drought (−50 %)": 0.5, "Wet (+30 %)": 1.3}.get(preset, 1.0)
    fert = st.checkbox("Fertilizer supply crisis (as in 2021/22)", value=False)

clim = dflt["climatology"]
rng = dflt["climatology_range"]


def scenario_values(scale: float) -> dict:
    v = dict(clim)
    for f in WEATHER_RAIN:
        if f in v:
            v[f] = clim[f] * scale
    if "rain_days" in v:
        v["rain_days"] = clim["rain_days"] * (0.6 + 0.4 * scale)
    if "longest_dry_spell" in v:
        v["longest_dry_spell"] = clim["longest_dry_spell"] / max(scale, 0.3) ** 0.8
    v["rain_anom_pct"] = 100 * (scale - 1)
    if "et0_total" in v:
        v["et0_total"] = clim["et0_total"] * (1 + 0.08 * (1 - scale))
    if {"rain_total", "et0_total"} <= set(v):
        v["water_balance"] = v["rain_total"] - v["et0_total"]
    return v


with tab_fc:
    vals = scenario_values(scale)
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("Recent yield history and planting (known before harvest)", expanded=True):
            prev_same = st.number_input("Previous same-season yield (kg/ha)", 500.0, 9000.0,
                                        float(dflt["prev_yield_same"]), 50.0)
            mean3 = st.number_input("Mean of last 3 same-season yields (kg/ha)", 500.0, 9000.0,
                                    float(dflt["yield_mean3_same"]), 50.0)
            prev_any = st.number_input("Immediately preceding season yield (kg/ha)", 500.0, 9000.0,
                                       float(meta["last_yield_any"].get(district, prev_same)), 50.0)
            sown = st.number_input("Extent sown this season (ha)", 0.0, 500000.0,
                                   float(np.nan_to_num(dflt["sown_ha"], nan=10000.0)), 500.0)
    with c2:
        with st.expander("Seasonal weather (pre-filled from the scenario)", expanded=preset == "Custom"):
            for f in ["rain_total", "rain_anom_pct", "longest_dry_spell", "antecedent_rain_6m", "tmax_mean", "rain_max5d"]:
                if f in F and f in vals:
                    lo, hi = rng.get(f, [vals[f] * 0.3, vals[f] * 1.7])
                    lo, hi = min(lo, vals[f]) , max(hi, vals[f])
                    if f == "rain_anom_pct":
                        lo, hi = -90.0, 150.0
                    span = hi - lo if hi > lo else abs(hi) + 1
                    vals[f] = st.slider(DESC.get(f, f), float(lo - 0.2 * span), float(hi + 0.2 * span),
                                        float(vals[f]), help=f, disabled=preset != "Custom")
            if "oni_mean" in F:
                vals["oni_mean"] = st.slider("ENSO (ONI): negative = La Niña, positive = El Niño", -2.5, 2.5, 0.0, 0.1)

    row = {"district": district, "season": season, "season_year": year,
           "fertilizer_crisis": int(fert), "prev_yield_same": prev_same, "prev_yield_any": prev_any,
           "yield_mean3_same": mean3, "sown_ha": sown,
           "sown_change_pct": 100 * (sown / dflt["sown_ha"] - 1) if dflt["sown_ha"] else 0.0}
    for f in F:
        if f not in row:
            row[f] = vals.get(f, clim.get(f, np.nan))
    X = pd.DataFrame([row])[F]

    pred = float(model.predict(X)[0])
    ref = dflt["yield_mean5_same"]
    ratio = pred / ref
    production = pred * sown * dflt["harvest_ratio"] / 1000
    shortfall = ratio < meta["shortfall_threshold"]

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecast yield", f"{pred:,.0f} kg/ha", f"{100 * (ratio - 1):+.1f} % vs 5-season mean")
    m2.metric("Estimated production", f"{production:,.0f} t", help="yield × sown extent × typical harvested/sown ratio")
    m3.metric("5-season mean", f"{ref:,.0f} kg/ha")
    m4.metric("Status", "⚠️ SHORTFALL" if shortfall else "✅ Normal",
              help=f"Shortfall = forecast below {int(meta['shortfall_threshold'] * 100)} % of the 5-season mean")

    hist = pd.DataFrame(dflt["history"], columns=["season_year", "yield_kg_ha"])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(hist.season_year, hist.yield_kg_ha, "o-", color="#2a7ab9", label="observed")
    ax.plot([year], [pred], "D", ms=10, color="#c0504d" if shortfall else "#3a9a5b", label="forecast")
    ax.axhline(ref * meta["shortfall_threshold"], ls="--", c="grey", lw=1, label="shortfall threshold")
    ax.set_ylabel("kg/ha"); ax.set_title(f"{district} — {season}"); ax.legend(loc="lower left", fontsize=8)
    st.pyplot(fig, width="stretch")

    st.subheader("Why this forecast?")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**SHAP** — contribution of each feature (kg/ha) from the average forecast")
        with st.spinner("Computing SHAP ..."):
            ex = E.shap_explanation(model, X, F, background=train.sample(min(200, len(train)), random_state=0),
                                    max_evals=2 * len(F) + 1 if not E.is_tree_pipeline(model) else 800)
        plt.figure()
        shap.plots.waterfall(ex[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), width="stretch")
        plt.close("all")
    with e2:
        st.markdown("**LIME** — local linear surrogate around this input")
        with st.spinner("Computing LIME ..."):
            exp = E.lime_explain_row(model, train[F], X, F, num_features=10,
                                     explainer_bundle=lime_bundle(model, train, F))
        lt = E.lime_table(exp)
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        ax2.barh(lt.condition[::-1], lt.weight_kg_ha[::-1],
                 color=np.where(lt.weight_kg_ha[::-1] > 0, "#3a9a5b", "#c0504d"))
        ax2.set_xlabel("kg/ha"); ax2.axvline(0, c="k", lw=.7)
        st.pyplot(fig2, width="stretch")
        st.caption(f"LIME surrogate prediction {exp.local_pred[0]:,.0f} kg/ha (surrogate R² {exp.score:.2f})")

    top = pd.Series(ex.values[0], index=F).sort_values()
    neg, pos = top.head(2), top.tail(2)[::-1]
    st.info(
        f"**Plain-language summary.** The model forecasts **{pred:,.0f} kg/ha** for {district} {season} {year}. "
        f"Biggest upward drivers: {', '.join(f'{k} (+{v:,.0f})' for k, v in pos.items() if v > 0) or 'none'}. "
        f"Biggest downward drivers: {', '.join(f'{k} ({v:,.0f})' for k, v in neg.items() if v < 0) or 'none'}.")

    with st.expander("Input row sent to the model"):
        st.dataframe(pd.DataFrame({"feature": F, "value": [str(round(v, 2)) if isinstance(v, float) else str(v) for v in X.iloc[0]],
                                  "description": [DESC.get(f, "") for f in F]}), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_models:
    st.subheader("Cross-validation (model selection, seasons ≤ %d)" % meta["train_end"])
    st.dataframe(pd.DataFrame(meta["cv_table"]).round(1), width="stretch")
    st.subheader("Held-out test seasons (> %d)" % meta["train_end"])
    st.dataframe(pd.DataFrame(meta["test_table"]).round(3), width="stretch")
    for f, cap in [("cv_rmse.png", "CV RMSE by model"), ("test_pred_vs_actual.png", "Test: predicted vs actual"),
                   ("shap_beeswarm.png", "Global SHAP summary")]:
        p = ART / "figures" / f
        if p.exists():
            st.image(str(p), caption=cap)

with tab_about:
    st.markdown("""
**Problem.** Paddy is Sri Lanka's staple crop. Planners need to know before harvest whether a district's
season will fall short, so stocks, imports and support can be arranged early.

**Data.** Department of Census and Statistics district paddy statistics (Maha and Yala, 1979–2025);
ERA5 daily weather via the Open-Meteo historical API (one paddy-area point per district);
NOAA Oceanic Niño Index.

**Model.** Several model families were tuned with expanding-window cross-validation on seasons up to the
training cut-off and tested on later seasons. The selected model was then re-fitted on all seasons for this app.

**Limits.** One weather point per district; shocks outside the training history (e.g. the 2021/22 fertilizer
ban) cannot be learned from weather; forecasts are for planning support, not official statistics.
""")
    st.dataframe(pd.DataFrame({"feature": F, "description": [DESC.get(f, "") for f in F]}), width="stretch")
