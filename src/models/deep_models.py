"""Deep learning models (CNN, LSTM) for sign language classification."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import TRAINED_MODELS_DIR, TRAINING_CONFIG
from src.models.base_model import BaseModel
from src.utils import setup_logger

logger = setup_logger(__name__)


class CNNModel(BaseModel):
    """1D CNN model for sign language gesture recognition from landmarks."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("CNN", config or TRAINING_CONFIG["cnn"])
        self.history: Optional[Dict] = None

    def build(self, input_shape: Tuple, num_classes: int) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            n_features = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
            dropout_rate = self.config.get("dropout_rate", 0.3)

            inp = layers.Input(shape=(n_features,))
            x = layers.Reshape((n_features, 1))(inp)
            x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(256, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(dropout_rate)(x)
            out = layers.Dense(num_classes, activation="softmax")(x)

            self.model = models.Model(inp, out)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.get("learning_rate", 0.001)
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.num_classes = num_classes
            logger.info(f"Built CNN model: {self.model.count_params()} parameters")
        except ImportError:
            logger.error("TensorFlow not installed. Cannot build CNN model.")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        try:
            import tensorflow as tf

            if self.model is None:
                self.build(X_train.shape[1:], len(np.unique(y_train)))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if X_val is not None else "loss",
                    patience=self.config.get("patience", 10),
                    restore_best_weights=True,
                )
            ]

            validation_data = (X_val, y_val) if X_val is not None else None
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.get("epochs", 50),
                batch_size=self.config.get("batch_size", 32),
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1,
            )
            self._is_trained = True
            self.history = history.history

            result = {"train_accuracy": history.history["accuracy"][-1]}
            if "val_accuracy" in history.history:
                result["val_accuracy"] = history.history["val_accuracy"][-1]
            return result
        except ImportError:
            logger.error("TensorFlow not installed")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else TRAINED_MODELS_DIR / "cnn_model.h5"
        from src.utils import ensure_dir
        ensure_dir(path.parent)
        self.model.save(str(path))
        logger.info(f"CNN model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        try:
            import tensorflow as tf
            path = Path(path) if path else TRAINED_MODELS_DIR / "cnn_model.h5"
            self.model = tf.keras.models.load_model(str(path))
            self._is_trained = True
            logger.info(f"CNN model loaded from {path}")
        except ImportError:
            logger.error("TensorFlow not installed")
            raise


class LSTMModel(BaseModel):
    """LSTM model for sign language gesture recognition from landmark sequences."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("LSTM", config or TRAINING_CONFIG["lstm"])
        self.history: Optional[Dict] = None
        self.sequence_length: int = 30

    def build(self, input_shape: Tuple, num_classes: int) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            seq_len = self.sequence_length
            n_features = input_shape[-1] if len(input_shape) > 1 else input_shape[0] // seq_len
            units = self.config.get("units", 128)
            dropout_rate = self.config.get("dropout_rate", 0.3)

            inp = layers.Input(shape=(seq_len, n_features))
            x = layers.LSTM(units, return_sequences=True)(inp)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.LSTM(units // 2)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(128, activation="relu")(x)
            out = layers.Dense(num_classes, activation="softmax")(x)

            self.model = models.Model(inp, out)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.get("learning_rate", 0.001)
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.num_classes = num_classes
            logger.info(f"Built LSTM model: {self.model.count_params()} parameters")
        except ImportError:
            logger.error("TensorFlow not installed")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        try:
            import tensorflow as tf

            # Reshape for LSTM: (samples, seq_len, features)
            n_features = X_train.shape[1] // self.sequence_length
            X_train_r = X_train.reshape(-1, self.sequence_length, n_features)
            X_val_r = None
            if X_val is not None:
                X_val_r = X_val.reshape(-1, self.sequence_length, n_features)

            if self.model is None:
                self.build((self.sequence_length, n_features), len(np.unique(y_train)))

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if X_val_r is not None else "loss",
                    patience=self.config.get("patience", 10),
                    restore_best_weights=True,
                )
            ]

            validation_data = (X_val_r, y_val) if X_val_r is not None else None
            history = self.model.fit(
                X_train_r, y_train,
                epochs=self.config.get("epochs", 50),
                batch_size=self.config.get("batch_size", 32),
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1,
            )
            self._is_trained = True
            self.history = history.history

            result = {"train_accuracy": history.history["accuracy"][-1]}
            if "val_accuracy" in history.history:
                result["val_accuracy"] = history.history["val_accuracy"][-1]
            return result
        except ImportError:
            logger.error("TensorFlow not installed")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1] // self.sequence_length
        X_r = X.reshape(-1, self.sequence_length, n_features)
        proba = self.model.predict(X_r, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1] // self.sequence_length
        X_r = X.reshape(-1, self.sequence_length, n_features)
        return self.model.predict(X_r, verbose=0)

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else TRAINED_MODELS_DIR / "lstm_model.h5"
        from src.utils import ensure_dir
        ensure_dir(path.parent)
        self.model.save(str(path))
        logger.info(f"LSTM model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        try:
            import tensorflow as tf
            path = Path(path) if path else TRAINED_MODELS_DIR / "lstm_model.h5"
            self.model = tf.keras.models.load_model(str(path))
            self._is_trained = True
            logger.info(f"LSTM model loaded from {path}")
        except ImportError:
            logger.error("TensorFlow not installed")
            raise
