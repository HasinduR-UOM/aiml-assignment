# Prediction API image. Build after `dvc pull` (or `dvc repro`) so the model exists.
# Library versions come from uv.lock, so they match the ones the model was trained with.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /srv

# libgomp is needed by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra serve

COPY src/ src/
COPY serve/ serve/
COPY artifacts/app_metadata.json artifacts/app_metadata.json
COPY artifacts/models/final_model.joblib artifacts/models/final_model.joblib

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION} PATH="/srv/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
