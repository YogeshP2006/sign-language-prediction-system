"""Utility functions for the Sign Language Prediction System."""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

from src.config import LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """Set up a logger with the specified name and level."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Path) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Save object to pickle file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model: Any, path: Path) -> None:
    """Save a scikit-learn model using joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Load a scikit-learn model using joblib."""
    return joblib.load(path)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks relative to the wrist (landmark 0).

    Args:
        landmarks: Array of shape (21, 3) with x, y, z coordinates.

    Returns:
        Normalized landmarks as a flat array of shape (63,).
    """
    if landmarks.shape != (21, 3):
        landmarks = landmarks.reshape(21, 3)

    # Translate so wrist is at origin
    wrist = landmarks[0].copy()
    normalized = landmarks - wrist

    # Scale by the distance from wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(normalized[9])
    if scale > 0:
        normalized = normalized / scale

    return normalized.flatten()


def extract_landmark_features(hand_landmarks) -> np.ndarray:
    """
    Extract landmark features from MediaPipe hand landmarks.

    Args:
        hand_landmarks: MediaPipe hand landmarks object.

    Returns:
        Normalized feature array of shape (63,).
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    return normalize_landmarks(coords)


def compute_hand_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute joint angles from hand landmarks.

    Args:
        landmarks: Array of shape (21, 3) or (63,) with landmark coordinates.

    Returns:
        Array of joint angles.
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    # Finger joint triplets for angle computation
    finger_joints = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),      # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20), # Pinky
    ]

    angles = []
    for a, b, c in finger_joints:
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(angle)

    return np.array(angles, dtype=np.float32)


def compute_finger_distances(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute distances between fingertips and palm center.

    Args:
        landmarks: Array of shape (21, 3) or (63,) with landmark coordinates.

    Returns:
        Array of fingertip distances.
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    fingertip_indices = [4, 8, 12, 16, 20]
    palm_center = landmarks[[0, 5, 9, 13, 17]].mean(axis=0)

    distances = []
    for idx in fingertip_indices:
        dist = np.linalg.norm(landmarks[idx] - palm_center)
        distances.append(dist)

    return np.array(distances, dtype=np.float32)


def smooth_predictions(
    history: List[int], weights: Optional[List[float]] = None
) -> int:
    """
    Smooth predictions using weighted voting.

    Args:
        history: List of recent prediction class indices.
        weights: Optional weights for each prediction (most recent first).

    Returns:
        Smoothed prediction.
    """
    if not history:
        return -1

    if weights is None:
        # Exponentially increase weight for more recent predictions
        weights = [2 ** i for i in range(len(history))]

    counts: Dict[int, float] = {}
    for pred, w in zip(reversed(history), weights):
        counts[pred] = counts.get(pred, 0) + w

    return max(counts, key=lambda k: counts[k])


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage string."""
    return f"{confidence * 100:.1f}%"


def get_available_models(models_dir: Path) -> List[str]:
    """Get list of available trained model files."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    extensions = {".pkl", ".h5", ".keras"}
    return [f.name for f in models_dir.iterdir() if f.suffix in extensions]
