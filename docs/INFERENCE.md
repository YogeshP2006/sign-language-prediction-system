# Inference

## Real-Time Demo

```bash
python scripts/demo_inference.py --model random_forest --threshold 0.7
```

Press **Q** to quit.

## Inference Pipeline

```
Frame (BGR) → HandDetector.detect_single_hand()
           → features (63 floats)
           → GesturePredictor.predict_frame()
              ├── StandardScaler.transform()
              ├── model.predict_proba()
              └── exponential smoothing
           → (label, confidence, proba_array)
           → PostProcessor.update()
           → confirmed gesture label
```

## Key Classes

### `HandDetector`

Wraps MediaPipe Hands. Converts BGR frames to RGB, runs hand detection, draws landmarks, returns normalized feature arrays.

### `GesturePredictor`

- Loads `.pkl` (RF/SVM) or `.h5` (CNN/LSTM) model files.
- Applies optional `DataPreprocessor` scaling.
- Applies **exponential smoothing** (`alpha=0.3`) over consecutive frame probabilities to reduce jitter.

### `PostProcessor`

- Maintains a sliding window buffer (10 frames by default).
- Returns a confirmed gesture when ≥80% of buffered predictions agree.

## Confidence Threshold

Predictions below `confidence_threshold` (default `0.7`) are shown in orange instead of green in the demo. Adjust with `--threshold`:

```bash
python scripts/demo_inference.py --model svm --threshold 0.85
```

## Programmatic Usage

```python
from src.inference.hand_detector import HandDetector
from src.inference.predictor import GesturePredictor

detector = HandDetector()
predictor = GesturePredictor(model_path="models/trained_models/random_forest_model.pkl")

annotated, features = detector.detect_single_hand(frame)
if features is not None:
    label, confidence, proba = predictor.predict_frame(features)
    print(f"{label}: {confidence:.2%}")
```
