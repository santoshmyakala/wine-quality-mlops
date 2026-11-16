# syntax=docker/dockerfile:1
# Serving image: contains the API and a model produced by `dvc repro`.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/santoshmyakala/wine-quality-mlops" \
      org.opencontainers.image.description="Wine quality prediction API" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib
RUN useradd --create-home --uid 10001 app
COPY --from=build /install /usr/local
WORKDIR /app
COPY --chown=app:app models/model.joblib ./models/model.joblib
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz'); sys.exit(0)"
CMD ["uvicorn", "wine_quality.api:app", "--host", "0.0.0.0", "--port", "8000"]
