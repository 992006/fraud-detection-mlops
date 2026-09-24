import joblib
import shutil
from pathlib import Path
import mlflow.sklearn

Path("artifacts").mkdir(exist_ok=True)

print("Loading champion model from MLflow Registry...")
model = mlflow.sklearn.load_model("models:/FraudDetectionModel@Production")

joblib.dump(model, "artifacts/champion.joblib")
shutil.copy("models/scaler.joblib", "artifacts/scaler.joblib")
shutil.copy("reports/best_thresholds.json", "artifacts/thresholds.json")

print("✅ Exported to artifacts/:")
for p in Path("artifacts").iterdir():
    print(f"   {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")