from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import json

try:
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

app = FastAPI(title="Fraud Detection API", version="2.1.0")

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

WEB_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fraud Detection System</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:40px; }
  .card { max-width: 950px; margin:auto; background:#1e293b; border-radius:12px; padding:30px; box-shadow:0 10px 30px rgba(0,0,0,.4); }
  h1 { text-align:center; margin-bottom:5px; }
  .sub { text-align:center; color:#94a3b8; margin-top:0; }
  .grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:20px 0; }
  label { font-size:12px; color:#94a3b8; }
  input { width:100%; padding:6px; border-radius:6px; border:1px solid #334155; background:#0f172a; color:#e2e8f0; box-sizing:border-box; }
  .buttons { text-align:center; }
  button { padding:10px 18px; margin:5px; border:none; border-radius:8px; cursor:pointer; font-weight:600; }
  .predict { background:#2563eb; color:white; }
  .sample { background:#334155; color:#e2e8f0; }
  #result { margin-top:25px; padding:20px; border-radius:10px; display:none; text-align:center; font-size:20px; }
  .fraud { background:#7f1d1d; color:#fecaca; }
  .legit { background:#14532d; color:#bbf7d0; }
</style>
</head>
<body>
<div class="card">
  <h1>Credit Card Fraud Detection</h1>
  <p class="sub">Enter transaction values or load a sample, then click Predict.</p>
  <div class="grid" id="form"></div>
  <div class="buttons">
    <button class="predict" onclick="predict()">Predict</button>
    <button class="sample" onclick="loadSample(fraudSample)">Load Fraud Sample</button>
    <button class="sample" onclick="loadSample(legitSample)">Load Legit Sample</button>
  </div>
  <div id="result"></div>
</div>
<script>
const fields = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"];
const form = document.getElementById("form");
fields.forEach(f => {
  const div = document.createElement("div");
  div.innerHTML = '<label>' + f + '</label><input id="in_' + f + '" type="number" step="any" value="0">';
  form.appendChild(div);
});
const fraudSample = {"Time":406.0,"V1":-2.3122265423263,"V2":1.95199201064158,"V3":-1.60985073229769,"V4":3.9979055875468,"V5":-0.522187864667764,"V6":-1.42654531920595,"V7":-2.53738730624579,"V8":1.39165724829804,"V9":-2.77008927719433,"V10":-2.77227214465915,"V11":3.20203320709635,"V12":-2.89990738849473,"V13":-0.595221881324605,"V14":-4.28925378244217,"V15":0.389724120274487,"V16":-1.14074717980657,"V17":-2.83005567450437,"V18":-0.0168224681808257,"V19":0.416955705037907,"V20":0.126910559061474,"V21":0.517232370861764,"V22":-0.0350493686052974,"V23":-0.465211076182388,"V24":0.320198198514526,"V25":0.0445191674731724,"V26":0.177839798284401,"V27":0.261145002567677,"V28":-0.143275874698919,"Amount":0.0};
const legitSample = {"Time":1453.0,"V1":-1.359,"V2":-0.072,"V3":2.536,"V4":1.378,"V5":-0.338,"V6":0.462,"V7":0.239,"V8":0.098,"V9":0.363,"V10":0.155,"V11":-0.213,"V12":-0.035,"V13":-0.143,"V14":-0.112,"V15":-0.214,"V16":0.141,"V17":-0.069,"V18":0.059,"V19":-0.042,"V20":-0.015,"V21":-0.053,"V22":-0.132,"V23":-0.028,"V24":0.112,"V25":0.045,"V26":0.088,"V27":0.012,"V28":0.005,"Amount":149.50};
function loadSample(s){ fields.forEach(f => document.getElementById("in_" + f).value = s[f]); }
async function predict(){
  const payload = {};
  fields.forEach(f => payload[f] = parseFloat(document.getElementById("in_" + f).value || 0));
  const res = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
  const box = document.getElementById("result");
  box.style.display = "block";
  if (!res.ok){ box.className = "fraud"; box.innerHTML = "Error: model not loaded. Start server from project root."; return; }
  const d = await res.json();
  box.className = d.prediction === "FRAUD" ? "fraud" : "legit";
  box.innerHTML = "<b>" + d.prediction + "</b><br>Fraud probability: " + d.fraud_probability +
                  "<br>Threshold: " + d.threshold_used + " | Model: " + d.model_used;
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return WEB_PAGE

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