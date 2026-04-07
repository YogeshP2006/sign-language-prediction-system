"""Unit tests for ML model classes."""

import numpy as np
import pytest

from src.models.classical_models import RandomForestModel, SVMModel


def make_data(n_samples=100, n_features=63, n_classes=5):
    rng = np.random.default_rng(0)
    X = rng.random((n_samples, n_features)).astype(np.float32)
    y = (np.arange(n_samples) % n_classes).astype(int)
    return X, y


def test_random_forest_build_and_train():
    X, y = make_data()
    model = RandomForestModel(config={"n_estimators": 10, "random_state": 0})
    model.build(X.shape[1:], 5)
    result = model.train(X, y)
    assert "train_accuracy" in result
    assert result["train_accuracy"] > 0


def test_random_forest_predict():
    X, y = make_data()
    model = RandomForestModel(config={"n_estimators": 10, "random_state": 0})
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 5)


def test_svm_build_and_train():
    X, y = make_data(n_samples=50)
    model = SVMModel(config={"kernel": "linear", "C": 1.0, "probability": True, "random_state": 0})
    model.build(X.shape[1:], 5)
    result = model.train(X, y)
    assert "train_accuracy" in result


def test_svm_predict():
    X, y = make_data(n_samples=50)
    model = SVMModel(config={"kernel": "linear", "C": 1.0, "probability": True, "random_state": 0})
    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_model_repr():
    model = RandomForestModel()
    assert "RandomForest" in repr(model)
    assert "trained=False" in repr(model)
