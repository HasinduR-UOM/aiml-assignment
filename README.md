# Paddy Yield Early Warning for Sri Lanka

Pre-harvest forecasting of district paddy yield (kg/ha) for the Maha and Yala seasons, with a shortfall warning, SHAP and LIME explanations, a Streamlit app, a FastAPI service and an MLOps pipeline deployed on Azure.

IT 5504 Applied Machine Learning, Assignment 2 · MSc in Artificial Intelligence, University of Moratuwa · Hasindu Ramanayake

## Problem

Paddy is Sri Lanka's staple crop, and district yields swing with the monsoon, droughts and policy shocks such as the 2021/22 fertilizer import ban. Planners need to know before harvest whether a district's season will fall short, so that stocks, imports and support can be arranged in time.

The task is regression: given a district and season, predict yield in kg/ha from information available before harvest. A season is flagged as a **shortfall** when the forecast is below 85% of the district's previous five same-season yields.

## Data

| Source | Content | Coverage |
|---|---|---|
| [Department of Census and Statistics: Paddy Statistics](https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics) | District average yield, production, sown and harvested extent | Maha and Yala, 1978-2025 |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 reanalysis) | Daily rainfall, max/min temperature and FAO-56 reference evapotranspiration at one paddy-area point per district | 1979-present |
| [NOAA CPC Oceanic Niño Index](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | ENSO state | 1950-present |

One row is one district in one season. The 23 model inputs fall into three groups:

- **Seasonal weather:** totals and anomalies against the 1981-2010 normal, dry spells, 5-day maxima, temperature, evapotranspiration, and rainfall in the six months before the season.
- **Yield history and planting:** lagged yields and sown extent.
- **Context:** district, season, year, ENSO and a fertilizer-crisis flag.

Harvested extent and production are only known after harvest, so they are excluded from the inputs to avoid target leakage.

## Method

1. **Split by time.** Train on seasons up to 2015, test on 2016-2025. Inside the training period, hyperparameters are tuned with expanding-window cross-validation (five 3-year validation blocks).
2. **Preprocess inside a pipeline.** Median imputation with missing indicators, scaling for the distance- and margin-based models, and one-hot encoding.
3. **Compare models.** Ridge, KNN, SVR (RBF), decision tree, random forest, gradient boosting, XGBoost, and LightGBM. Each one is tuned with `RandomizedSearchCV`. A naive "same as last season" forecast is the baseline.
4. **Select and evaluate.** The model is selected on CV RMSE and evaluated once on the test seasons with RMSE, MAE, MAPE and R², plus precision, recall and F1 for the shortfall flag.
5. **Explain.** SHAP (TreeExplainer and permutation) gives global and local explanations, and LIME gives local explanations. Their rankings and stability are compared.
6. **Deploy.** A Streamlit app and a FastAPI service run the model, which is refitted on all seasons.

## Repository layout

```
src/
  config.py       districts and coordinates, seasons, split years, data URLs
  dcs.py          download and parse DCS tables (PDF or Excel)
  weather.py      Open-Meteo daily weather per district, NOAA ONI
  features.py     seasonal weather features and lagged history -> model table
  contract.py     pandera data contract for the model table
  modeling.py     pipelines, model zoo, search spaces, time-aware CV, metrics
  train.py        tuning, test evaluation, artifacts, MLflow logging
  explain.py      SHAP and LIME helpers
  inference.py    builds a model input row from a short request
  tracking.py     MLflow tracking on DagsHub
  gate.py         quality gate against the registered champion
  registry.py     MLflow model registry (@champion alias)
notebooks/        01 EDA, 02 modelling, 03 explainability
app/app.py        Streamlit app: inputs -> forecast -> shortfall flag -> SHAP + LIME
serve/api.py      FastAPI service: /health, /version, /predict, /feedback
monitoring/       drift check, Log Analytics export, demo traffic
infra/            Azure Container Apps bootstrap script
tests/            unit, data-contract and API tests
dvc.yaml          parse -> features -> validate -> train pipeline
params.yaml       pipeline, gate and monitoring settings
.github/workflows CI, CD, Monitor, Retrain
```

## Quick start

```bash
# Python 3.11 or 3.12
uv sync --all-extras          # or: pip install -r requirements.txt

python run_pipeline.py        # download -> parse -> weather -> features -> validate -> train
jupyter notebook notebooks    # 01 -> 02 -> 03
streamlit run app/app.py      # interactive forecast with explanations
uvicorn serve.api:app --port 8000   # REST API, docs at /docs
pytest -q
```

On Windows, activate the environment with `.venv\Scripts\activate` before running these commands.

## Results

Test seasons 2016-2025 (filled in from `artifacts/metrics.json` after the final run):

| Model | CV RMSE | Test RMSE | Test RMSE, normal seasons | MAPE, normal seasons | R², normal seasons |
|---|---|---|---|---|---|
| Selected model | | | | | |
| Naive baseline | | | | | |

"Normal seasons" excludes the 2021/22 fertilizer-crisis seasons, which have no precedent in the training period.

## MLOps

| Stage | Implementation |
|---|---|
| Environment | `uv` with a locked `uv.lock`; GitHub Codespaces devcontainer |
| Data and model versioning | DVC pipeline (`dvc.yaml`), remote storage on DagsHub |
| Experiment tracking and registry | MLflow on DagsHub, model `paddy-yield` with alias `@champion` |
| Data quality | pandera data contract (`src/contract.py`) as a pipeline stage |
| Quality gate | a new model must not have a higher normal-season test RMSE than the champion (`src/gate.py`) |
| Serving | FastAPI in Docker, image on GitHub Container Registry |
| Hosting | Azure Container Apps (scale to zero), logs in Log Analytics, OIDC login from GitHub Actions |
| CI | lint, tests, `dvc.lock` check, API tests, quality gate; gate report posted on the pull request |
| CD | build and push image, manual approval, deploy, smoke test, promote to `@champion` |
| Monitoring | weekly Kolmogorov-Smirnov drift test on live inputs against the training data |
| Retraining | fresh data, rebuilt pipeline, gate, and a pull request for review |

Setup and deployment steps are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Limitations

- Weather comes from one ERA5 grid point per district rather than an average over the paddy area.
- Shocks outside the training history, such as the 2021/22 fertilizer ban, cannot be learned from weather and history alone.
- Tree models cannot extrapolate the long-run yield trend; the lagged-yield features carry it instead.
- The Major, Minor and Rainfed split of sown extent is only published from 2004, so it is not part of the main model.
