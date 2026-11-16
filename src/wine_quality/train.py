"""Stage 2: train the model, track the run in MLflow and register the model."""

from __future__ import annotations

import json
import subprocess

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import KFold, cross_validate

from wine_quality.config import MODEL_PATH, MODELS_DIR, PROCESSED_DIR, RUN_ID_PATH, load_params, tracking_uri
from wine_quality.features import FEATURES, TARGET, build_pipeline


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    params = load_params()
    seed, tparams = params["seed"], params["train"]
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X, y = train[FEATURES], train[TARGET]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=tparams["model"]) as run:
        mlflow.set_tags({"git_sha": _git_sha(), "stage": "train"})
        mlflow.log_params({"model": tparams["model"], "seed": seed, **(tparams.get(tparams["model"]) or {})})

        pipeline = build_pipeline(tparams, seed)
        cv = cross_validate(
            pipeline, X, y,
            cv=KFold(5, shuffle=True, random_state=seed),
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        cv_metrics = {
            "cv_rmse": float(-np.mean(cv["test_rmse"])),
            "cv_mae": float(-np.mean(cv["test_mae"])),
            "cv_r2": float(np.mean(cv["test_r2"])),
        }
        mlflow.log_metrics(cv_metrics)

        pipeline.fit(X, y)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X.head(), pipeline.predict(X.head())),
            registered_model_name=params["mlflow"]["registered_model"],
            # Model is produced by this run, so its tree structures are trusted for skops serialisation.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        RUN_ID_PATH.write_text(run.info.run_id)
    print("train:", json.dumps(cv_metrics))


if __name__ == "__main__":
    main()
