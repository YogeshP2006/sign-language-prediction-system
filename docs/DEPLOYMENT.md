# Deployment

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Requirements

- Python 3.8+
- Webcam (for data collection and live prediction)
- ~2 GB disk space (for dependencies)

## Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t sign-language-app .
docker run -p 8501:8501 sign-language-app
```

> **Note:** Webcam access inside Docker requires additional configuration (e.g., `--device /dev/video0`). For data collection and live prediction, running locally is recommended.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## CI/CD

The `.github/workflows/ci.yml` workflow runs `pytest` on every push and pull request to `main`. It installs only the lightweight dependencies (numpy, scikit-learn, scipy, joblib) needed to run the unit tests without requiring TensorFlow or MediaPipe.
