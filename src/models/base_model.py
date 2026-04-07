"""Abstract base model class for sign language classification."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all sign language classification models."""

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.model: Any = None
        self._is_trained = False

    @abstractmethod
    def build(self, input_shape: Tuple, num_classes: int) -> None:
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train the model and return training history/metrics."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model and return accuracy."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"{self.name} accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, trained={self._is_trained})"
