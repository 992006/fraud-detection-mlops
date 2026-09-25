@echo off
title Fraud Detection Launcher
cd /d "D:\fraud-detection-mlops"

echo [1/4] Running full MLOps pipeline (validate - train - optimize - register - test)...
start "MLOps Pipeline" cmd /k "powershell -ExecutionPolicy Bypass -File run_pipeline.ps1"

echo [2/4] Starting Fraud Detection API (port 8000)...
start "FraudAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn api.main:app --reload"

echo [3/4] Starting MLflow UI (port 5000)...
start "MLflowUI" cmd /k "call .venv\Scripts\activate.bat && mlflow ui"

echo [4/4] Opening browsers: local app, MLflow dashboard, live Render app...
timeout /t 10 /nobreak >nul
start "" http://127.0.0.1:8000
start "" http://127.0.0.1:5000
start "" https://fraud-detection-mlops-2svf.onrender.com

exit