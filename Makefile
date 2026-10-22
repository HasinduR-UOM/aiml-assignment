# Run these in a GitHub Codespace (Linux). On Windows use the commands inside each target.
IMAGE ?= paddy-yield:local
DAGSHUB_REPO ?= paddy-yield-sl

.PHONY: setup lint format test doctor dvc-auth repro gate promote serve docker-build docker-run drift ui

setup:            ## install everything into .venv
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

test:
	uv run pytest -q

doctor:           ## check tools and DagsHub access
	@echo "=== tools ===" ; uv --version ; uv run python --version ; uv run dvc --version ; docker --version || true ; az version -o tsv 2>/dev/null | head -1 || echo "az not installed (fine in Codespace; use Azure Cloud Shell)"
	@echo "=== DagsHub ===" ; test -n "$$DAGSHUB_OWNER" && test -n "$$DAGSHUB_TOKEN" && echo "DAGSHUB_OWNER=$$DAGSHUB_OWNER token set" || (echo "DAGSHUB_OWNER / DAGSHUB_TOKEN missing" && exit 1)
	@uv run python -c "from src.tracking import configure; import mlflow; configure(); mlflow.search_experiments(); print('MLflow on DagsHub OK')"

dvc-auth:         ## store DagsHub credentials for DVC (never committed)
	uv run dvc remote add -f -d origin https://dagshub.com/$$DAGSHUB_OWNER/$${DAGSHUB_REPO:-$(DAGSHUB_REPO)}.dvc
	uv run dvc remote modify origin --local auth basic
	uv run dvc remote modify origin --local user $$DAGSHUB_OWNER
	uv run dvc remote modify origin --local password $$DAGSHUB_TOKEN

repro:            ## parse -> features -> validate -> train (only changed stages)
	uv run dvc repro

gate:
	uv run python -m src.gate

promote:
	uv run python -m src.registry promote

serve:            ## API with auto-reload on :8000 (open /docs)
	uv run uvicorn serve.api:app --reload --port 8000

ui:               ## Streamlit app on :8501
	uv run streamlit run app/app.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -e MODEL_VERSION=local $(IMAGE)

drift:
	uv run python -m monitoring.drift_check
