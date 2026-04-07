"""
Sign Language Prediction System - Configuration Module
Central configuration management for the entire system
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import json

@dataclass
class HandDetectionConfig:
    """Configuration for hand detection module"""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2
    static_image_mode: bool = False
    model_complexity: int = 1

@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    num_samples_per_gesture: int = 100
    gesture_duration_seconds: int = 2
    class_labels: List[str] = None
    
    def __post_init__(self):
        if self.class_labels is None:
            self.class_labels = [chr(i) for i in range(65, 91)]

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    normalization_method: str = "minmax"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    augmentation_enabled: bool = True
    augmentation_noise_std: float = 0.01

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    cnn_learning_rate: float = 0.001
    cnn_epochs: int = 50
    cnn_batch_size: int = 32
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 50
    lstm_batch_size: int = 32

@dataclass
class InferenceConfig:
    """Configuration for real-time inference"""
    confidence_threshold: float = 0.7
    smoothing_window: int = 5
    smoothing_method: str = "exponential"
    enable_tts: bool = True
    tts_engine: str = "pyttsx3"

@dataclass
class PathConfig:
    """Configuration for file paths"""
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw: str = None
    data_processed: str = None
    models_dir: str = None
    logs_dir: str = None
    
    def __post_init__(self):
        self.data_raw = os.path.join(self.project_root, "data", "raw")
        self.data_processed = os.path.join(self.project_root, "data", "processed")
        self.models_dir = os.path.join(self.project_root, "models", "trained_models")
        self.logs_dir = os.path.join(self.project_root, "logs")
        
        for path in [self.data_raw, self.data_processed, self.models_dir, self.logs_dir]:
            os.makedirs(path, exist_ok=True)

@dataclass
class SystemConfig:
    """Master configuration class"""
    hand_detection: HandDetectionConfig = None
    data_collection: DataCollectionConfig = None
    preprocessing: PreprocessingConfig = None
    model: ModelConfig = None
    inference: InferenceConfig = None
    paths: PathConfig = None
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.hand_detection is None:
            self.hand_detection = HandDetectionConfig()
        if self.data_collection is None:
            self.data_collection = DataCollectionConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.paths is None:
            self.paths = PathConfig()

CONFIG = SystemConfig()