from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import joblib
import pandas as pd
import json

app = FastAPI(title="Fraud Detection API", version="1.0.0")

ml_model = None
scaler = None
optimal_threshold = 0.5
model_name_loaded = "Unknown"

@app.on_event("startup")
def load_artifacts():
    global ml_model, scaler, optimal_threshold, model_name_loaded
    try:
        print("Loading production artifacts...")
        ml_model = mlflow.sklearn.load_model("models:/FraudDetectionModel@Production")
        scaler = joblib.load("models/scaler.joblib")
        with open("reports/best_thresholds.json", "r") as f:
            thresholds = json.load(f)
            best_model = max(thresholds, key=lambda k: thresholds[k]['best_f1'])
            optimal_threshold = thresholds[best_model]['best_threshold']
            model_name_loaded = best_model
        print(f"✅ Loaded Production Model: {model_name_loaded} (threshold {optimal_threshold})")
    except Exception as e:
        print(f"⚠️ Model artifacts not available ({e}). /predict will return 503.")

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

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