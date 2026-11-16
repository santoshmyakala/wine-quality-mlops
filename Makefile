.PHONY: install test lint pipeline serve image compose mlflow-ui
install:   ; pip install -e ".[train,dev]"
lint:      ; ruff check src tests
test:      ; pytest -q
pipeline:  ; dvc repro && dvc metrics show
serve:     ; uvicorn wine_quality.api:app --reload --port 8000
image:     ; docker build -t wine-quality-api:local .
compose:   ; docker compose up --build
mlflow-ui: ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
