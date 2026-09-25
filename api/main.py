from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
import json

try:
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

app = FastAPI(title="Fraud Detection API", version="3.0.0")

# Path to the web UI file (lives next to this file: api/index.html)
HTML_PATH = Path(__file__).resolve().parent / "index.html"

ml_model = None
scaler = None
optimal_threshold = 0.5
model_name_loaded = "Unknown"

@app.on_event("startup")
def load_artifacts():
    global ml_model, scaler, optimal_threshold, model_name_loaded

    # Option 1: MLflow Model Registry (local development)
    if HAS_MLFLOW:
        try:
            print("Loading model from MLflow Registry...")
            ml_model = mlflow.sklearn.load_model("models:/FraudDetectionModel@Production")
            scaler = joblib.load("models/scaler.joblib")
            with open("reports/best_thresholds.json", "r") as f:
                thresholds = json.load(f)
            best_model = max(thresholds, key=lambda k: thresholds[k]['best_f1'])
            optimal_threshold = thresholds[best_model]['best_threshold']
            model_name_loaded = best_model
            print(f"Loaded from registry: {model_name_loaded} (threshold {optimal_threshold})")
            return
        except Exception as e:
            print(f"Registry not available ({e}). Trying committed artifacts...")

    # Option 2: committed artifacts (cloud deployment)
    try:
        print("Loading committed artifacts...")
        ml_model = joblib.load("artifacts/champion.joblib")
        scaler = joblib.load("artifacts/scaler.joblib")
        with open("artifacts/thresholds.json", "r") as f:
            thresholds = json.load(f)
        best_model = max(thresholds, key=lambda k: thresholds[k]['best_f1'])
        optimal_threshold = thresholds[best_model]['best_threshold']
        model_name_loaded = best_model + " (deployed)"
        print(f"Loaded deployed artifacts: {model_name_loaded} (threshold {optimal_threshold})")
    except Exception as e:
        print(f"Artifacts not available ({e}). /predict will return 503.")

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the modern web UI from api/index.html"""
    try:
        return HTML_PATH.read_text(encoding="utf-8")
    except Exception:
        return "<h1 style='font-family:sans-serif'>api/index.html not found</h1>"

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": ml_model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    if ml_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    data = transaction.dict()
    df = pd.DataFrame([data])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    proba = ml_model.predict_proba(df)[0][1]
    is_fraud = proba >= optimal_threshold

    return {
        "fraud_probability": round(float(proba), 4),
        "prediction": "FRAUD" if is_fraud else "LEGIT",
        "threshold_used": optimal_threshold,
        "model_used": model_name_loaded,
    }