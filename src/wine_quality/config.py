"""Paths and parameter loading shared by every pipeline stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("WINE_QUALITY_ROOT", Path.cwd()))
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "model.joblib"
RUN_ID_PATH = MODELS_DIR / "run_id.txt"


def load_params(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "params.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def tracking_uri() -> str:
    """MLflow tracking URI; defaults to a local SQLite store so runs work offline."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
