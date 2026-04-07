"""Data preprocessing module for sign language gesture data."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import PREPROCESSING_CONFIG, PROCESSED_DATA_DIR
from src.utils import ensure_dir, save_pickle, load_pickle, setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Preprocesses hand landmark data for model training."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or PREPROCESSING_CONFIG
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataPreprocessor":
        """Fit scaler and label encoder on training data."""
        self.scaler.fit(X)
        self.label_encoder.fit(y)
        self._is_fitted = True
        logger.info(
            f"Fitted preprocessor: {len(X)} samples, "
            f"{len(self.label_encoder.classes_)} classes"
        )
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Transform features (and optionally labels)."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming")
        X_scaled = self.scaler.transform(X)
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        return X_scaled

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode integer labels back to class names."""
        return self.label_encoder.inverse_transform(y_encoded)

    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from integer index to class name."""
        return {i: cls for i, cls in enumerate(self.label_encoder.classes_)}

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets with stratification.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        train_ratio = self.config["train_ratio"]
        val_ratio = self.config["val_ratio"]
        test_ratio = self.config["test_ratio"]
        random_state = self.config["random_state"]

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}"

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y,
        )

        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=y_train_val,
        )

        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def augment_data(
        self, X: np.ndarray, y: np.ndarray, factor: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data by adding Gaussian noise.

        Args:
            X: Feature matrix.
            y: Label array.
            factor: Augmentation factor (multiplier for dataset size).

        Returns:
            Augmented (X, y) tuple.
        """
        factor = factor or self.config.get("augmentation_factor", 2)
        noise_std = self.config.get("noise_std", 0.01)

        X_aug = [X]
        y_aug = [y]

        for _ in range(factor - 1):
            noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
            X_aug.append(X + noise)
            y_aug.append(y)

        return np.vstack(X_aug), np.concatenate(y_aug)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save preprocessor state to disk."""
        path = Path(path) if path else PROCESSED_DATA_DIR / "preprocessor.pkl"
        ensure_dir(path.parent)
        save_pickle(
            {"scaler": self.scaler, "label_encoder": self.label_encoder, "config": self.config},
            path,
        )
        logger.info(f"Preprocessor saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> "DataPreprocessor":
        """Load preprocessor state from disk."""
        path = Path(path) if path else PROCESSED_DATA_DIR / "preprocessor.pkl"
        state = load_pickle(path)
        self.scaler = state["scaler"]
        self.label_encoder = state["label_encoder"]
        self.config = state.get("config", self.config)
        self._is_fitted = True
        logger.info(f"Preprocessor loaded from {path}")
        return self

    def prepare_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = True,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Full preprocessing pipeline: split, augment, scale.

        Returns:
            Dictionary with train/val/test splits (scaled features, encoded labels).
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        if augment:
            X_train, y_train = self.augment_data(X_train, y_train)

        self.fit(X_train, y_train)
        X_train_s, y_train_e = self.transform(X_train, y_train)
        X_val_s, y_val_e = self.transform(X_val, y_val)
        X_test_s, y_test_e = self.transform(X_test, y_test)

        dataset = {
            "X_train": X_train_s,
            "X_val": X_val_s,
            "X_test": X_test_s,
            "y_train": y_train_e,
            "y_val": y_val_e,
            "y_test": y_test_e,
            "label_mapping": self.get_label_mapping(),
            "num_classes": len(self.label_encoder.classes_),
            "feature_size": X.shape[1],
        }

        if save_dir:
            save_dir = Path(save_dir)
            ensure_dir(save_dir)
            save_pickle(dataset, save_dir / "dataset.pkl")
            with open(save_dir / "label_mapping.json", "w") as f:
                json.dump({str(k): v for k, v in dataset["label_mapping"].items()}, f, indent=2)
            self.save(save_dir / "preprocessor.pkl")

        return dataset
