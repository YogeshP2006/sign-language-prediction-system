# Model Training

## Quick Start

```bash
# 1. Collect gesture data (requires webcam)
python scripts/collect_data.py --gesture A --samples 200

# 2. Train all models
python scripts/train_models.py --model all

# 3. Evaluate on test set
python scripts/evaluate_models.py --model all
```

## Available Models

| Model | Type | File |
|-------|------|------|
| `random_forest` | Random Forest (scikit-learn) | `random_forest_model.pkl` |
| `svm` | Support Vector Machine (scikit-learn) | `svm_model.pkl` |
| `cnn` | 1D Convolutional Neural Network (TensorFlow) | `cnn_model.h5` |
| `lstm` | Long Short-Term Memory (TensorFlow) | `lstm_model.h5` |

## Configuration

Default hyperparameters are defined in `src/config.py` under `TRAINING_CONFIG`. Override per model:

```python
from src.models.classical_models import RandomForestModel
model = RandomForestModel(config={"n_estimators": 500, "random_state": 42})
```

Model configs are also stored in `models/model_configs/config.json`.

## Training Pipeline

1. **Load** – `GestureCollector.load_existing_data()` reads all CSV files.
2. **Split** – 70% train / 10% val / 20% test (stratified).
3. **Augment** – Gaussian noise is added to the training set (2× by default).
4. **Scale** – `StandardScaler` fitted on train set, applied to all sets.
5. **Train** – Each model is trained and saved to `models/trained_models/`.
6. **Results** – Metrics written to `models/trained_models/training_results.json`.

## Tips

- Collect at least **100–200 samples per gesture** for reliable results.
- Random Forest and SVM train in seconds; CNN/LSTM may take several minutes.
- Use `--no-augment` flag to skip data augmentation.
