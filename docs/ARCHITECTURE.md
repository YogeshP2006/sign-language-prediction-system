# Architecture

## Overview

The Sign Language Prediction System is a modular end-to-end pipeline:

```
Webcam → MediaPipe Hand Detection → Landmark Extraction → ML Model → Gesture Label → TTS
```

## Components

### Data Layer (`src/data/`)
- **`collector.py`** – Captures hand landmarks from webcam via MediaPipe, saves to CSV.
- **`preprocessor.py`** – Scales features, encodes labels, splits train/val/test, augments data.
- **`dataset.py`** – Wraps numpy arrays as dataset objects; supports TensorFlow conversion.

### Model Layer (`src/models/`)
- **`base_model.py`** – Abstract base class defining the model interface.
- **`classical_models.py`** – Random Forest and SVM implementations (scikit-learn).
- **`deep_models.py`** – 1D CNN and LSTM implementations (TensorFlow/Keras).
- **`model_trainer.py`** – Orchestrates multi-model training, evaluation, and result persistence.

### Inference Layer (`src/inference/`)
- **`hand_detector.py`** – Real-time hand landmark detection using MediaPipe.
- **`predictor.py`** – Loads trained model, applies smoothing, returns predictions with confidence.
- **`post_processor.py`** – Confirms gestures by majority voting over a sliding window buffer.

### Utilities (`src/utils/`)
- **`metrics.py`** – Accuracy, precision, recall, F1, confusion matrix.
- **`visualizer.py`** – Matplotlib/Seaborn plots for training history and metrics.
- **`tts_engine.py`** – Text-to-speech output via pyttsx3.

### Application (`app/`)
- Streamlit multi-page app for data collection, training, live prediction, and analytics.

## Feature Representation

Each hand is represented as **63 normalized floats** (21 landmarks × 3 coordinates: x, y, z). Landmarks are translated to place the wrist at the origin and scaled by the wrist-to-MCP distance, making features invariant to hand position and scale.
