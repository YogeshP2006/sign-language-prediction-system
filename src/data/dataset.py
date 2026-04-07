"""Dataset utilities for TensorFlow and PyTorch compatibility."""

from typing import Dict, Optional, Tuple

import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


class SignLanguageDataset:
    """Dataset class for sign language gesture data."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            sequence_length: If set, reshape X for sequence models (LSTM).
        """
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int64)
        self.sequence_length = sequence_length

        if sequence_length is not None:
            n_features = self.X.shape[1] // sequence_length
            self.X = self.X.reshape(-1, sequence_length, n_features)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.X[idx], self.y[idx]

    @property
    def shape(self) -> Tuple:
        return self.X.shape

    def to_tf_dataset(self, batch_size: int = 32, shuffle: bool = True):
        """Convert to TensorFlow dataset."""
        try:
            import tensorflow as tf
            ds = tf.data.Dataset.from_tensor_slices((self.X, self.y))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(self.X))
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        except ImportError:
            logger.error("TensorFlow not installed")
            raise


def create_datasets(
    data: Dict, sequence_length: Optional[int] = None
) -> Tuple[SignLanguageDataset, SignLanguageDataset, SignLanguageDataset]:
    """Create train, val, test datasets from preprocessed data dictionary."""
    train_ds = SignLanguageDataset(data["X_train"], data["y_train"], sequence_length)
    val_ds = SignLanguageDataset(data["X_val"], data["y_val"], sequence_length)
    test_ds = SignLanguageDataset(data["X_test"], data["y_test"], sequence_length)
    return train_ds, val_ds, test_ds
