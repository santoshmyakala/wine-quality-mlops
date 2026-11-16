"""Stage 3: evaluate the trained model on the held-out test set against a baseline."""

from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from wine_quality.config import MODEL_PATH, REPORTS_DIR, RUN_ID_PATH, tracking_uri
from wine_quality.features import FEATURES, TARGET


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.rint(y_pred), 0, 10)
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "exact_grade_accuracy": float(np.mean(rounded == y_true)),
        "within_one_grade_accuracy": float(np.mean(np.abs(rounded - y_true) <= 1)),
    }


def main() -> None:
    test = pd.read_csv(REPORTS_DIR.parent / "data" / "processed" / "test.csv")
    train = pd.read_csv(REPORTS_DIR.parent / "data" / "processed" / "train.csv")
    model = joblib.load(MODEL_PATH)
    y = test[TARGET].to_numpy()

    model_metrics = regression_metrics(y, model.predict(test[FEATURES]))
    baseline = regression_metrics(y, np.full_like(y, train[TARGET].mean(), dtype=float))
    report = {"test_rows": len(test), "model": model_metrics, "baseline_mean_predictor": baseline}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=RUN_ID_PATH.read_text().strip()):
        mlflow.log_metrics({f"test_{k}": v for k, v in model_metrics.items()})
        mlflow.log_artifact(str(REPORTS_DIR / "metrics.json"))
    print("evaluate:", json.dumps(report["model"]))


if __name__ == "__main__":
    main()
