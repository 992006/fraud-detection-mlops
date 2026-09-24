# 🚀 End-to-End MLOps Pipeline: Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![CI/CD](https://github.com/992006/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Deployed](https://img.shields.io/badge/Deployed-Render-success)

An automated, production-grade MLOps pipeline for detecting highly imbalanced credit card fraud (0.17% positive class). This project moves beyond Jupyter Notebooks to demonstrate **data versioning, automated training, model registry, drift monitoring, closed-loop retraining, and cloud deployment**.

🔗 **Live Web App & API:** [https://fraud-detection-mlops-2svf.onrender.com/](https://fraud-detection-mlops-2svf.onrender.com/) *(Note: Free tier may take 30s to wake up on first request)*

---

## 📌 The Business Problem
In fraud detection, **accuracy is a trap**. A naive model that guesses "Legitimate" every time achieves 99.8% accuracy but catches zero fraud. 
This project solves the imbalance problem by:
1. Optimizing for **F1-Score** and **PR-AUC** instead of accuracy.
2. Implementing **Automated Threshold Sweeping** (e.g., lowering the decision boundary from 0.50 to 0.30 to catch more fraud without overwhelming investigators with false alarms).
3. Handling imbalance natively via `class_weight='balanced'` and XGBoost's `scale_pos_weight`.

## 🏗️ MLOps Architecture

```text
[Raw Data] 
   ↓ (DVC Versioning)
[Data Quality Gate] (Pandera Schema Validation)
   ↓
[Preprocessing] (Stratified Split + Leakage-safe Scaling)
   ↓
[Training] (LR, Random Forest, XGBoost)
   ↓ (MLflow Tracking)
[Threshold Optimization] (Per-model F1 Sweep)
   ↓
[Model Registry] (Auto-promotes highest F1 to "Production")
   ↓
[Cloud Deployment] (FastAPI + Web UI hosted on Render)
   ↓
[Monitoring] (KS-Test Data Drift Detection)
   ↓ (Triggers Auto-Retrain)
[Closed Loop] (New model v2 trained and promoted)]

 Quickstart (Local Execution)
To run the entire pipeline from raw data to a registered production model with a single command:
1. Clone and setup environment:

   git clone https://github.com/992006/fraud-detection-mlops.git
   cd fraud-detection-mlops
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt

2.Run the Orchestrator:

   # Executes validation, preprocessing, training, threshold tuning, and registry promotion
   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1

3.Launch the API & MLflow:

   # Terminal 1: Start API
   uvicorn api.main:app --reload
   
   # Terminal 2: Start MLflow UI
   mlflow ui