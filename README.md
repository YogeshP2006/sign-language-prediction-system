# 🤟 Sign Language Prediction System

A complete end-to-end Sign Language Prediction System using computer vision and machine learning. It detects hand landmarks in real time via **MediaPipe** and classifies them into A–Z gestures using **Random Forest**, **SVM**, **CNN**, or **LSTM** models.

---

## ✨ Features

- 📷 **Real-time webcam** hand landmark detection (MediaPipe)
- 🧠 **4 ML models**: Random Forest, SVM, 1D CNN, LSTM
- 🔄 **Prediction smoothing** with exponential filtering and majority-vote confirmation
- 🔊 **Text-to-speech** output via pyttsx3
- 📊 **Streamlit dashboard** for data collection, training, live prediction, and analytics
- 🧪 **Unit tests** for all core components

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect gesture data (requires webcam)

```bash
python scripts/collect_data.py --gesture A --samples 200
# Or collect all A-Z at once:
python scripts/collect_data.py --all
```

### 3. Train models

```bash
python scripts/train_models.py --model all
```

### 4. Run live demo

```bash
python scripts/demo_inference.py --model random_forest
```

### 5. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
sign-language-prediction-system/
├── app/
│   ├── streamlit_app.py          # Main Streamlit entry point
│   └── pages/                    # Multi-page app (Home, Collection, Training, etc.)
├── data/
│   ├── raw/gestures/             # Per-class CSV files from webcam
│   └── processed/                # Scaled features, label mappings, preprocessor
├── docs/                         # Architecture, data format, training, inference, deployment docs
├── models/
│   ├── model_configs/config.json # Default hyperparameters
│   └── trained_models/           # Saved .pkl and .h5 model files
├── notebooks/                    # Jupyter notebooks for exploration
├── scripts/
│   ├── collect_data.py           # CLI: webcam data collection
│   ├── train_models.py           # CLI: model training
│   ├── evaluate_models.py        # CLI: model evaluation
│   └── demo_inference.py         # CLI: real-time inference demo
├── src/
│   ├── config.py                 # Centralised configuration
│   ├── utils.py                  # Shared utilities (logging, normalization, etc.)
│   ├── data/
│   │   ├── collector.py          # GestureCollector
│   │   ├── preprocessor.py       # DataPreprocessor
│   │   └── dataset.py            # SignLanguageDataset
│   ├── models/
│   │   ├── base_model.py         # Abstract BaseModel
│   │   ├── classical_models.py   # RandomForestModel, SVMModel
│   │   ├── deep_models.py        # CNNModel, LSTMModel
│   │   └── model_trainer.py      # ModelTrainer orchestrator
│   ├── inference/
│   │   ├── hand_detector.py      # HandDetector (MediaPipe)
│   │   ├── predictor.py          # GesturePredictor
│   │   └── post_processor.py     # PostProcessor (smoothing/confirmation)
│   └── utils/
│       ├── metrics.py            # compute_metrics, print_metrics
│       ├── visualizer.py         # Plotting utilities
│       └── tts_engine.py         # TTSEngine
├── tests/
│   ├── test_preprocessor.py
│   ├── test_models.py
│   └── test_inference.py
├── requirements.txt
└── setup.py
```

---

## 🔧 Configuration

All tuneable parameters live in `src/config.py`:

| Section | Key settings |
|---------|-------------|
| `MEDIAPIPE_CONFIG` | Detection / tracking confidence |
| `COLLECTION_CONFIG` | Samples per class, countdown |
| `PREPROCESSING_CONFIG` | Train/val/test ratio, augmentation factor |
| `TRAINING_CONFIG` | Per-model hyperparameters |
| `INFERENCE_CONFIG` | Confidence threshold, smoothing alpha, buffer size |

---

## 🧪 Running Tests

```bash
pip install pytest numpy scikit-learn scipy joblib
python -m pytest tests/ -v
```

---

## 📖 Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data Format](docs/DATA_FORMAT.md)
- [Model Training](docs/MODEL_TRAINING.md)
- [Inference](docs/INFERENCE.md)
- [Deployment](docs/DEPLOYMENT.md)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.