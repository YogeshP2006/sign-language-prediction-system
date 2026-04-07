# Data Format

## Raw Data (`data/raw/gestures/`)

Collected samples are stored as CSV files, one file per collection session:

```
data/raw/gestures/
├── A/
│   ├── A_1700000000.csv
│   └── A_1700001000.csv
├── B/
│   └── B_1700002000.csv
...
```

### CSV Schema

Each row represents one frame of detected hand landmarks:

| Column | Type | Description |
|--------|------|-------------|
| `lm_0_x` … `lm_20_x` | float | X coordinate for each of 21 landmarks |
| `lm_0_y` … `lm_20_y` | float | Y coordinate |
| `lm_0_z` … `lm_20_z` | float | Z coordinate (depth estimate) |
| `label` | str | Gesture class (e.g., `"A"`) |

Total feature columns: **63** (21 landmarks × 3 axes).

## Processed Data (`data/processed/`)

After running the preprocessing pipeline:

| File | Description |
|------|-------------|
| `dataset.pkl` | Dict with `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` |
| `preprocessor.pkl` | Fitted `StandardScaler` + `LabelEncoder` |
| `label_mapping.json` | Integer index → gesture class name mapping |

### Label Mapping Example

```json
{
  "0": "A",
  "1": "B",
  ...
  "25": "Z"
}
```

## Normalization

Before model input, each sample is normalized:
1. **Translation** – subtract wrist (landmark 0) position so wrist is at origin.
2. **Scaling** – divide by distance from wrist to middle finger MCP (landmark 9).

This makes features invariant to absolute hand position and size.
