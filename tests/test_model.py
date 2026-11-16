import numpy as np

from wine_quality.evaluate import regression_metrics
from wine_quality.features import FEATURES, TARGET, build_pipeline
from wine_quality.prepare import load_raw, split


def test_model_beats_mean_baseline():
    train, test = split(load_raw(), 0.2, seed=0)
    pipe = build_pipeline({"model": "random_forest", "random_forest": {"n_estimators": 50}}, seed=0)
    pipe.fit(train[FEATURES], train[TARGET])
    y = test[TARGET].to_numpy()
    model = regression_metrics(y, pipe.predict(test[FEATURES]))
    baseline = regression_metrics(y, np.full_like(y, train[TARGET].mean(), dtype=float))
    assert model["rmse"] < baseline["rmse"]
    assert model["r2"] > 0.3


def test_regression_metrics_perfect_prediction():
    y = np.array([5, 6, 7])
    m = regression_metrics(y, y.astype(float))
    assert m["rmse"] == 0 and m["exact_grade_accuracy"] == 1


def test_unknown_model_rejected():
    import pytest

    with pytest.raises(ValueError):
        build_pipeline({"model": "nope"}, seed=0)
