"""FastAPI model-serving service with health probes and Prometheus metrics."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from wine_quality import __version__
from wine_quality.features import FEATURES

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

PREDICTIONS = Counter("wine_predictions_total", "Number of samples scored")
LATENCY = Histogram("wine_predict_latency_seconds", "Latency of /predict requests")
PREDICTED_QUALITY = Histogram(
    "wine_predicted_quality", "Distribution of predicted quality scores",
    buckets=(3, 4, 5, 6, 7, 8, 9),
)


class WineSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fixed_acidity: float = Field(ge=0, alias="fixed acidity")
    volatile_acidity: float = Field(ge=0, alias="volatile acidity")
    citric_acid: float = Field(ge=0, alias="citric acid")
    residual_sugar: float = Field(ge=0, alias="residual sugar")
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(ge=0, alias="total sulfur dioxide")
    density: float = Field(gt=0)
    pH: float = Field(gt=0, lt=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0, le=100)
    is_red: bool


class PredictRequest(BaseModel):
    samples: list[WineSample] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Wine Quality API", version=__version__, lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    start = time.perf_counter()
    rows = [s.model_dump(by_alias=True) for s in req.samples]
    frame = pd.DataFrame(rows)[FEATURES].astype(float)
    preds = [round(float(p), 3) for p in state["model"].predict(frame)]
    for p in preds:
        PREDICTED_QUALITY.observe(p)
    PREDICTIONS.inc(len(preds))
    LATENCY.observe(time.perf_counter() - start)
    return PredictResponse(predictions=preds, model_version=__version__)
