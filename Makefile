# Developer commands. Run in a GitHub Codespace (Linux) or any shell with make.
# Azure targets need `az login` and `az configure --defaults group=<rg> workspace=<ws>` (done by aml/setup.sh).
N_ITER ?= 30
EP ?= paddy-yield-ep

.PHONY: setup lint format test pipeline serve ui drift aml-validate aml-setup aml-data aml-train aml-invoke aml-monitor

setup:            ## install everything into .venv
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

test:
	uv run pytest -q

pipeline:         ## local run: download -> parse -> weather -> features -> validate -> train
	uv run python run_pipeline.py

serve:            ## local REST API on :8000 (open /docs)
	uv run uvicorn serve.api:app --reload --port 8000

ui:               ## Streamlit app on :8501
	uv run streamlit run app/app.py

drift:            ## local drift check on the newest seasons
	uv run python -m monitoring.drift_check

aml-validate:     ## offline schema check of aml/*.yml
	uv run python -m aml.validate

aml-setup:        ## one-time Azure ML resources (pass REPO=<owner>/<repo>)
	bash aml/setup.sh $(REPO)

aml-data:         ## upload data/raw as a new version of the paddy-raw data asset
	az ml data create --name paddy-raw --version $$(date +%Y%m%d%H%M) --type uri_folder --path data/raw

aml-train:        ## submit the training pipeline and stream its logs
	az ml job create -f aml/pipeline.yml --set inputs.n_iter=$(N_ITER) --stream

aml-invoke:       ## call the live endpoint with aml/sample-request.json
	az ml online-endpoint invoke -n $(EP) --request-file aml/sample-request.json

aml-monitor:      ## create the weekly drift schedule (after the endpoint has received traffic)
	az ml schedule create -f aml/monitor-schedule.yml
