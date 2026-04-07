"""Real-time prediction module for sign language gestures."""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import INFERENCE_CONFIG, TRAINED_MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import setup_logger

logger = setup_logger(__name__)


class GesturePredictor:
    """Performs real-time inference on hand landmark features."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        label_mapping_path: Optional[Path] = None,
        config: Optional[Dict] = None,
    ):
        self.config = config or INFERENCE_CONFIG
        self.model = None
        self.preprocessor = None
        self.label_mapping: Dict[int, str] = {}
        self.prediction_history: deque = deque(
            maxlen=self.config["prediction_buffer_size"]
        )
        self._smoothed_proba: Optional[np.ndarray] = None

        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
        if label_mapping_path:
            self.load_label_mapping(label_mapping_path)

    def load_model(self, path: Path) -> None:
        """Load a trained model from disk (supports pkl and h5/keras formats)."""
        path = Path(path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".pkl":
                import joblib
                self.model = joblib.load(path)
            elif suffix in (".h5", ".keras"):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(path))
            else:
                raise ValueError(f"Unsupported model format: {suffix}")
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise

    def load_preprocessor(self, path: Path) -> None:
        """Load a DataPreprocessor from disk."""
        from src.data.preprocessor import DataPreprocessor
        self.preprocessor = DataPreprocessor().load(path)
        self.label_mapping = self.preprocessor.get_label_mapping()
        logger.info(f"Preprocessor loaded from {path}")

    def load_label_mapping(self, path: Path) -> None:
        """Load label mapping from JSON file."""
        with open(path, "r") as f:
            raw = json.load(f)
        self.label_mapping = {int(k): v for k, v in raw.items()}
        logger.info(f"Label mapping loaded: {len(self.label_mapping)} classes")

    def predict_frame(
        self, features: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """
        Predict gesture from a single frame's landmark features.

        Args:
            features: Landmark feature array of shape (63,).

        Returns:
            Tuple of (predicted_label, confidence, class_probabilities).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        x = features.reshape(1, -1)
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)

        # Get class probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(x)
            proba = raw[0] if raw.ndim > 1 else raw
        else:
            raise RuntimeError("Model does not support predict_proba or predict")

        # Exponential smoothing
        alpha = self.config["smoothing_alpha"]
        if self._smoothed_proba is None or len(self._smoothed_proba) != len(proba):
            self._smoothed_proba = proba.copy()
        else:
            self._smoothed_proba = alpha * proba + (1 - alpha) * self._smoothed_proba

        pred_idx = int(np.argmax(self._smoothed_proba))
        confidence = float(self._smoothed_proba[pred_idx])
        predicted_label = self.label_mapping.get(pred_idx, str(pred_idx))

        self.prediction_history.append(pred_idx)
        return predicted_label, confidence, self._smoothed_proba

    def get_top_predictions(
        self, proba: np.ndarray, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top-k predictions with confidence scores."""
        top_indices = np.argsort(proba)[::-1][:top_k]
        return [
            (self.label_mapping.get(int(i), str(i)), float(proba[i]))
            for i in top_indices
        ]

    def is_confident(self, confidence: float) -> bool:
        """Check if prediction meets confidence threshold."""
        return confidence >= self.config["confidence_threshold"]

    def reset(self) -> None:
        """Reset prediction state."""
        self.prediction_history.clear()
        self._smoothed_proba = None
