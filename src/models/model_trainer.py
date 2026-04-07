"""Model training pipeline for sign language classification."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import TRAINED_MODELS_DIR, TRAINING_CONFIG
from src.models.classical_models import RandomForestModel, SVMModel
from src.models.deep_models import CNNModel, LSTMModel
from src.utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "cnn": CNNModel,
    "lstm": LSTMModel,
}


class ModelTrainer:
    """Orchestrates training of multiple sign language classification models."""

    def __init__(self, models_dir: Path = TRAINED_MODELS_DIR):
        self.models_dir = Path(models_dir)
        ensure_dir(self.models_dir)
        self.results: Dict = {}

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Train a single model by name."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}"
            )

        logger.info(f"Training {model_name}...")
        start_time = time.time()

        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(config=config)
        model.build(X_train.shape[1:], len(np.unique(y_train)))

        history = model.train(X_train, y_train, X_val, y_val)
        elapsed = time.time() - start_time

        save_path = self.models_dir / f"{model_name}_model"
        if model_name in ("cnn", "lstm"):
            save_path = save_path.with_suffix(".h5")
        else:
            save_path = save_path.with_suffix(".pkl")
        model.save(save_path)

        result = {**history, "training_time_seconds": elapsed, "model_path": str(save_path)}
        self.results[model_name] = result
        logger.info(f"{model_name} training complete in {elapsed:.1f}s: {result}")
        return result

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_names: Optional[List[str]] = None,
    ) -> Dict:
        """Train all (or specified) models."""
        model_names = model_names or list(MODEL_REGISTRY.keys())
        for name in model_names:
            try:
                self.train_model(name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                self.results[name] = {"error": str(e)}
        return self.results

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate all saved models on test data."""
        model_names = model_names or list(MODEL_REGISTRY.keys())
        eval_results = {}

        for name in model_names:
            try:
                model_cls = MODEL_REGISTRY[name]
                model = model_cls()

                model_path = self.models_dir / f"{name}_model"
                if name in ("cnn", "lstm"):
                    model_path = model_path.with_suffix(".h5")
                else:
                    model_path = model_path.with_suffix(".pkl")

                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue

                model.load(model_path)
                metrics = model.evaluate(X_test, y_test)
                eval_results[name] = metrics
                logger.info(f"{name} test metrics: {metrics}")
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                eval_results[name] = {"error": str(e)}

        return eval_results

    def save_results(self, path: Optional[Path] = None) -> None:
        """Save training results to JSON."""
        path = path or self.models_dir / "training_results.json"
        serializable_results = {}
        for k, v in self.results.items():
            serializable_results[k] = {
                key: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                for key, val in v.items()
            }
        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Training results saved to {path}")
