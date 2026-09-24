import subprocess
import sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def simulate_current_data():
    """In production this would be new live transactions. Here we simulate drifted data."""
    curr = pd.read_csv("data/processed/X_test.csv")
    curr['Amount'] = curr['Amount'] + np.random.normal(2.0, 1.0, size=len(curr))
    curr['V1'] = curr['V1'] * 1.5
    return curr

def detect_drift(ref, curr, p_thresh=0.05):
    drifted = []
    for col in ref.columns:
        if pd.api.types.is_numeric_dtype(ref[col]):
            _, p = ks_2samp(ref[col], curr[col])
            if p < p_thresh:
                drifted.append(col)
    return drifted

def run(cmd):
    print(f"\n>>> RUNNING: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"❌ Pipeline stage failed: {cmd}")

def main():
    print("========== AUTOMATED RETRAINING WORKFLOW ==========")

    ref = pd.read_csv("data/processed/X_train.csv")
    curr = simulate_current_data()

    drifted = detect_drift(ref, curr)
    print(f"Drifted features detected: {drifted}")

    if not drifted:
        print("✅ No drift detected. Production model stays unchanged.")
        return

    print("⚠️ DRIFT DETECTED! Triggering full retraining pipeline...\n")

    # Stage 1: retrain all models + log to MLflow
    run("python src/models/train.py")

    # Stage 2: re-optimize thresholds
    run("python src/models/threshold_optimization.py")

    # Stage 3: register best model and promote to Production
    run("python src/models/register_model.py")

    print("\n✅ RETRAINING COMPLETE!")
    print("A new model version was registered and promoted to Production.")
    print("👉 Check the MLflow Model Registry for the new version.")

if __name__ == "__main__":
    main()