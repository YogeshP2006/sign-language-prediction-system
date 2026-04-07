"""Configuration module for the Sign Language Prediction System."""

import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GESTURES_DIR = RAW_DATA_DIR / "gestures"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
MODEL_CONFIGS_DIR = MODELS_DIR / "model_configs"

# Supported gesture classes (A-Z)
GESTURE_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# MediaPipe hand detection config
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}

# Hand landmark count
NUM_LANDMARKS = 21
NUM_COORDINATES = 3  # x, y, z
FEATURE_SIZE = NUM_LANDMARKS * NUM_COORDINATES  # 63 features

# Data collection config
COLLECTION_CONFIG = {
    "samples_per_class": 200,
    "sequence_length": 30,
    "fps": 30,
    "countdown_seconds": 3,
}

# Preprocessing config
PREPROCESSING_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
    "random_state": 42,
    "augmentation_factor": 2,
    "noise_std": 0.01,
}

# Model training config
TRAINING_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "svm": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "probability": True,
        "random_state": 42,
    },
    "cnn": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "patience": 10,
    },
    "lstm": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "units": 128,
        "dropout_rate": 0.3,
        "patience": 10,
    },
}

# Inference config
INFERENCE_CONFIG = {
    "confidence_threshold": 0.7,
    "smoothing_alpha": 0.3,
    "prediction_buffer_size": 10,
    "confirmation_threshold": 0.8,
}

# Streamlit UI config
UI_CONFIG = {
    "page_title": "Sign Language Prediction System",
    "page_icon": "🤟",
    "layout": "wide",
    "video_width": 640,
    "video_height": 480,
}

# Logging config
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
