"""FastAPI service for sliding-window HMM anomaly detection."""
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from detector import detect_stream

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hmm_anomaly_model.joblib"

app = FastAPI(
    title="HMM Sensor Anomaly Detection API",
    version="1.0.0",
    description="Detect unusual machine sensor patterns using a Gaussian Hidden Markov Model.",
)

class SensorReading(BaseModel):
    temperature: float = Field(..., description="Machine temperature reading")
    vibration: float = Field(..., description="Machine vibration reading")

class SensorBatch(BaseModel):
    readings: list[SensorReading]

@app.get("/")
def home():
    return {
        "message": "HMM Anomaly Detection API is running.",
        "docs": "/docs",
        "endpoint": "POST /detect",
    }

@app.post("/detect")
def detect(data: SensorBatch):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run python train.py first."
        )
    if len(data.readings) < 40:
        raise HTTPException(
            status_code=400,
            detail="Send at least 40 readings because the sliding window size is 40."
        )

    bundle = joblib.load(MODEL_PATH)
    values = np.array([
        [item.temperature, item.vibration] for item in data.readings
    ], dtype=float)

    scores, starts, flags, point_flags = detect_stream(
        bundle["model"],
        bundle["scaler"],
        values,
        bundle["threshold"],
        bundle["window_size"],
        bundle["step_size"],
    )

    return {
        "selected_hidden_states": int(bundle["hidden_states"]),
        "threshold": float(bundle["threshold"]),
        "windows_checked": int(len(scores)),
        "anomalous_windows": int(flags.sum()),
        "anomaly_detected": bool(point_flags.any()),
        "anomalous_window_start_indices": starts[flags == 1].tolist(),
        "point_anomaly_flags": point_flags.tolist(),
        "window_log_likelihoods": np.round(scores, 4).tolist(),
    }
