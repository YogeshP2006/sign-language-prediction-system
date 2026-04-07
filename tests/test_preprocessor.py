"""Unit tests for the DataPreprocessor."""

import numpy as np
import pytest

from src.data.preprocessor import DataPreprocessor


def make_dummy_data(n_samples=100, n_features=63, n_classes=5):
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features)).astype(np.float32)
    y = np.array([chr(ord("A") + i % n_classes) for i in range(n_samples)])
    return X, y


def test_fit_transform():
    X, y = make_dummy_data()
    pp = DataPreprocessor()
    X_scaled, y_enc = pp.fit_transform(X, y)
    assert X_scaled.shape == X.shape
    assert y_enc.shape == y.shape
    assert y_enc.dtype in (np.int32, np.int64, int)


def test_inverse_transform():
    X, y = make_dummy_data()
    pp = DataPreprocessor()
    _, y_enc = pp.fit_transform(X, y)
    y_decoded = pp.inverse_transform_labels(y_enc)
    np.testing.assert_array_equal(y_decoded, y)


def test_label_mapping():
    X, y = make_dummy_data(n_classes=3)
    pp = DataPreprocessor()
    pp.fit(X, y)
    mapping = pp.get_label_mapping()
    assert len(mapping) == 3
    for k, v in mapping.items():
        assert isinstance(k, int)
        assert isinstance(v, str)


def test_split_data():
    X, y = make_dummy_data(n_samples=200, n_classes=4)
    pp = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = pp.split_data(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(X_train) > len(X_val)


def test_augment_data():
    X, y = make_dummy_data(n_samples=50)
    pp = DataPreprocessor()
    X_aug, y_aug = pp.augment_data(X, y, factor=3)
    assert len(X_aug) == 3 * len(X)
    assert len(y_aug) == 3 * len(y)


def test_transform_not_fitted():
    pp = DataPreprocessor()
    X, _ = make_dummy_data(n_samples=10)
    with pytest.raises(RuntimeError):
        pp.transform(X)
