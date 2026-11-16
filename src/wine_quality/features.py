"""Feature definitions and the preprocessing/model pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Column names exactly as published in the UCI dataset, plus a wine-type flag.
PHYSICOCHEMICAL = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
FEATURES = [*PHYSICOCHEMICAL, "is_red"]
TARGET = "quality"


def build_pipeline(params: dict[str, Any], seed: int) -> Pipeline:
    name = params["model"]
    cfg = params.get(name, {}) or {}
    if name == "random_forest":
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **cfg)
    elif name == "gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=seed, **cfg)
    elif name == "ridge":
        model = Ridge(**cfg)
    else:
        raise ValueError(f"Unknown model '{name}'")
    return Pipeline([("scale", StandardScaler()), ("model", model)])
