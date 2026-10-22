# Deployment and MLOps guide

This guide sets up the full pipeline: GitHub, a Codespace for development, DVC and MLflow on DagsHub, CI/CD with GitHub Actions, and hosting on Azure Container Apps. Commands run in the Codespace terminal unless marked Azure Cloud Shell.

**Prerequisites:** a GitHub account (Codespaces), an Azure subscription (Azure for Students works), and a DagsHub account linked to GitHub.

**Flow:**

1. A change to code or data goes in through a pull request.
2. **CI** runs lint and tests, checks `dvc.lock` and runs the quality gate.
3. After the merge, **CD** builds the image and waits for approval. It then deploys to Azure, smoke-tests the service and promotes the model to `@champion`.
4. **Monitor** checks drift every week.
5. **Retrain** opens a new pull request when retraining is needed.

# Step 1. Create the GitHub repository

1. On github.com create a **public** repository named `paddy-yield-sl` (public = free Actions minutes; no README, no .gitignore).
2. In the project folder:

```powershell
git init -b main
git add .
git status          # check: no .venv, no data\raw, no artifacts\models, no *.joblib
git commit -m "Initial commit"
git remote add origin https://github.com/<owner>/paddy-yield-sl.git
git push -u origin main
```

If `git status` lists `.venv` or large data files, check `.gitignore` before committing.

# Step 2. Connect DagsHub and add secrets

1. dagshub.com → **Create** → **Connect a repository** → pick `paddy-yield-sl`.
2. DagsHub → profile → **User Settings** → **Tokens** → create a token and copy it.
3. GitHub repo → **Settings** → **Secrets and variables** → **Actions**:
   - **Secrets** tab: `DAGSHUB_TOKEN` = the token.
   - **Variables** tab: `DAGSHUB_OWNER` = your DagsHub username; `DAGSHUB_REPO` = `paddy-yield-sl`.
4. github.com/settings/codespaces → **Codespaces secrets**: add `DAGSHUB_TOKEN`, `DAGSHUB_OWNER`, `DAGSHUB_REPO` with the same values, and give each one access to `paddy-yield-sl`.

Workflows read the owner from **variables**, and Codespaces cannot see Actions secrets. That is why the owner appears in both places.

# Step 3. Open a Codespace and check the tools

Repo page → **Code** → **Codespaces** → **Create codespace on main**. The devcontainer installs Python 3.12, Docker, Azure CLI and `uv`, then runs `uv sync --all-extras`. Wait until the terminal is idle, then:

```bash
make lint        # ruff: all checks passed
make test        # 10 passed (the API tests skip until a model exists)
make doctor      # tool versions + "MLflow on DagsHub OK"
```

If `uv: command not found`: `pip install --user uv` and open a new terminal.
If `DAGSHUB_OWNER / DAGSHUB_TOKEN missing`: the Codespaces secrets were added after the Codespace started. Use **Reload to apply** in the notification, or stop and start the Codespace.

# Step 4. Version the raw data with DVC

Download the raw data inside the Codespace, then track it with DVC.

```bash
uv run python -m src.dcs download
uv run python -m src.weather          # 25 districts; rerun if any are skipped
uv run python -m src.weather oni

uv run dvc init
make dvc-auth                          # DagsHub as the DVC remote; credentials stay local
uv run dvc add data/raw/dcs data/raw/weather data/raw/oni
uv run dvc push
git add .dvc .dvcignore data/raw/*.dvc
git commit -m "Track raw data with DVC" && git push
```

**Check:** the DagsHub repo page shows the `data/raw` folders under its DVC tab.

# Step 5. Run the pipeline and log to MLflow

`dvc.yaml` has four stages: `parse` → `features` → `validate` (the pandera data contract) → `train`. Because the DagsHub variables are set, training also logs to MLflow.

```bash
make repro
```

What to watch for:

- `validate` fails with *"…% of rows have no weather"*: a district's weather is missing. Rerun `uv run python -m src.weather`, `uv run dvc add data/raw/weather`, then `make repro`.
- `validate` fails on `yield_kg_ha`: a value outside 500-9,000 kg/ha that is not a zero. Look at the row it names; zeros are already turned into missing values in `src/features.py`.
- Training too slow: set `n_iter: 15` in `params.yaml` (it is a DVC parameter, so `make repro` notices the change).

When it finishes:

```bash
cat artifacts/metrics.json             # headline metrics
uv run dvc push                        # model and data to DagsHub
git add -A && git status   # expect dvc.lock, artifacts/metrics.json, cv_results.csv, run_info.json; no .joblib or big CSVs
git commit -m "Pipeline run on real data" && git push
```

**Check on DagsHub → Experiments:** one run named after the best model, with `cv_rmse`, `test_rmse_normal` and the other metrics, the figures and a `model` artifact.

# Step 6. Quality gate and model registry

```bash
make gate        # first time: "No champion yet, so the first model passes"
make promote     # registers paddy-yield version 1 and sets alias @champion
```

The gate rule lives in `params.yaml`: the new model's RMSE on normal test seasons must be less than or equal to the champion's. From now on, a worse model cannot be promoted.

**Check on DagsHub → Models:** `paddy-yield`, version 1, alias `champion`.

# Step 7. Run the API locally, then in Docker

```bash
make serve                 # open the Ports tab → port 8000 → add /docs
```

In the `/docs` page try **POST /predict** with:

```json
{"district": "Kurunegala", "season": "Yala", "rain_pct_of_normal": 50}
```

The response contains the forecast in kg/ha, the 5-season mean, the % change and a shortfall flag. Stop with Ctrl+C, then check the image Azure will run:

```bash
make docker-build
make docker-run            # in a second terminal:
curl localhost:8000/health
```

# Step 8. CI, CD and Azure

The order matters: Azure can only pull an image that already exists.

1. **CI.** In the Codespace:

   ```bash
   git switch -c mlops-ci
   git commit --allow-empty -m "Trigger CI" && git push -u origin mlops-ci
   ```

   Open a pull request into `main`. CI runs lint, tests, `dvc pull`, checks that `dvc.lock` matches, runs the API tests on the real model and the gate, then comments the gate table on the PR. Merge when green.

2. **CD build.** The merge starts CD. The `build` job pushes `ghcr.io/<owner>/paddy-yield-sl:latest`. The `deploy` job is skipped because the Azure variables do not exist yet.

3. **Make the image public.** GitHub → your profile → **Packages** → `paddy-yield-sl` → **Package settings** → **Change visibility** → Public.

4. **Create the Azure resources.** portal.azure.com → Cloud Shell (Bash):

   ```bash
   git clone https://github.com/<owner>/paddy-yield-sl.git && cd paddy-yield-sl
   bash infra/bootstrap.sh <owner>/paddy-yield-sl ghcr.io/<owner>/paddy-yield-sl:latest
   ```

   It picks a region your student subscription allows, creates the resource group `rg-paddy-mlops`, Log Analytics, the Container Apps environment, the app `ca-paddy-yield` and a managed identity that trusts GitHub through OIDC (no password is stored anywhere). At the end it prints the app URL and six values.

5. **Add the six printed values** as Actions **variables**: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `LOG_ANALYTICS_WORKSPACE_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_CONTAINER_APP`.

6. **Create the approval step.** Repo → **Settings** → **Environments** → **New environment** `production` → Required reviewers: the repository owner; Deployment branches: `main` only.

7. **Deploy.** Actions → **CD** → **Run workflow**. It builds, then pauses on `deploy` until a reviewer clicks **Review deployments → Approve**. It then updates the container app, waits for the new revision, smoke-tests `/health` and `/predict`, and promotes the model to `@champion`.

**Check:** open `https://<app-url>/docs` in a browser and call `/predict`. The `model_version` in the answer is the 7-character commit id.

**Rollback** if a release goes wrong (Cloud Shell):

```bash
az containerapp revision list -n ca-paddy-yield -g rg-paddy-mlops -o table
az containerapp update -n ca-paddy-yield -g rg-paddy-mlops --image ghcr.io/<owner>/paddy-yield-sl:<old-tag>
```

# Step 9. Monitoring

1. Send traffic to the live API from the Codespace:

   ```bash
   uv run python -m monitoring.send_traffic https://<app-url> --n 300 --mode normal
   uv run python -m monitoring.send_traffic https://<app-url> --n 300 --mode drought
   ```

   Every prediction is written as one JSON line to the container log, which goes to Log Analytics. Allow about 5 minutes for the logs to arrive.

2. Actions → **Monitor** → **Run workflow**. It exports the last 7 days of predictions, runs a Kolmogorov-Smirnov test on each weather input against the training seasons and writes the result to the run summary. If 30% or more of the inputs drifted (`params.yaml`), it starts **Retrain** by itself. Monitor also runs every Monday.

The same check runs locally on the newest seasons in the data with `make drift`.

# Step 10. Retraining

Actions → **Retrain** → **Run workflow** → reason, for example *"2025/26 Maha statistics released"*. It downloads fresh DCS tables and weather, rebuilds the pipeline, runs the gate and, only if the new model is at least as good, pushes data and model to DagsHub and opens a pull request with the gate table. Merging that PR runs CI and then CD, so nothing reaches the live API without passing the gate and a review.

# Step 11. Clean up

Delete all Azure resources when the deployment is no longer needed (Cloud Shell):

```bash
az group delete -n rg-paddy-mlops --yes --no-wait
```

# Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `output ... is already tracked by SCM` | A DVC output (e.g. `artifacts/models`) is also in git | `git rm -r --cached <path>`, commit, rerun `make repro` |
| `make dvc-auth` then `dvc push` gives 401 | Wrong token or owner | Recreate the DagsHub token; update the Codespaces and Actions secrets |
| DagsHub 404 / MLflow cannot connect | `DAGSHUB_REPO` differs from the DagsHub repo name | Set the variable to the exact name |
| CI fails at "Pipeline is up to date" | Code or data changed but `dvc.lock` was not updated | `make repro && uv run dvc push`, commit `dvc.lock` |
| CI job skipped / no Actions run | Actions not enabled, or default branch not `main` | Actions tab → enable; Settings → default branch `main` |
| Bootstrap: `DENIED: requested access to the resource is denied` | Image not on ghcr.io yet, or still private | Run CD build first, then make the package Public |
| Bootstrap: region not allowed | Student policy | `LOCATION=<allowed region> bash infra/bootstrap.sh ...` |
| Deploy step: `AADSTS70021` / no federated credential | Repo name in bootstrap differs from the real repo, or environment not called `production` | Rerun bootstrap with the exact `<owner>/<repo>`; check the environment name |
| Smoke test times out | The app scaled to zero and is cold-starting | Rerun the job; first start can take a minute |
| `/predict` returns 404 for a district | That district had no data in the trained model | Fix its data, retrain |
| Monitor exports 0 rows | Logs not arrived yet, or no traffic | Wait 5-10 minutes after `send_traffic`, rerun |

