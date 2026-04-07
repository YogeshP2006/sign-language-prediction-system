"""Classical ML models (Random Forest, SVM) for sign language classification."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import TRAINED_MODELS_DIR, TRAINING_CONFIG
from src.models.base_model import BaseModel
from src.utils import save_model, load_model, setup_logger

logger = setup_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest classifier for sign language gesture recognition."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("RandomForest", config or TRAINING_CONFIG["random_forest"])

    def build(self, input_shape: Tuple, num_classes: int) -> None:
        self.model = RandomForestClassifier(**self.config)
        logger.info(f"Built RandomForest with config: {self.config}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        if self.model is None:
            self.build(X_train.shape, len(np.unique(y_train)))

        logger.info(f"Training RandomForest on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self._is_trained = True

        train_acc = self.model.score(X_train, y_train)
        result = {"train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            result["val_accuracy"] = val_acc
            logger.info(f"Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")
        else:
            logger.info(f"Train accuracy: {train_acc:.4f}")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else TRAINED_MODELS_DIR / "random_forest_model.pkl"
        save_model(self.model, path)
        logger.info(f"RandomForest saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path else TRAINED_MODELS_DIR / "random_forest_model.pkl"
        self.model = load_model(path)
        self._is_trained = True
        logger.info(f"RandomForest loaded from {path}")


class SVMModel(BaseModel):
    """Support Vector Machine classifier for sign language gesture recognition."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("SVM", config or TRAINING_CONFIG["svm"])

    def build(self, input_shape: Tuple, num_classes: int) -> None:
        self.model = SVC(**self.config)
        logger.info(f"Built SVM with config: {self.config}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        if self.model is None:
            self.build(X_train.shape, len(np.unique(y_train)))

        logger.info(f"Training SVM on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self._is_trained = True

        train_acc = self.model.score(X_train, y_train)
        result = {"train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            result["val_accuracy"] = val_acc
            logger.info(f"Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}")
        else:
            logger.info(f"Train accuracy: {train_acc:.4f}")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else TRAINED_MODELS_DIR / "svm_model.pkl"
        save_model(self.model, path)
        logger.info(f"SVM saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path else TRAINED_MODELS_DIR / "svm_model.pkl"
        self.model = load_model(path)
        self._is_trained = True
        logger.info(f"SVM loaded from {path}")
