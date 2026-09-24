# ============================================================
# Full MLOps Training Pipeline (fail-fast, stage by stage)
# validate -> EDA -> preprocess -> train -> thresholds -> register -> test
# ============================================================

# Auto-activate virtual environment if not active
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        . .\.venv\Scripts\Activate.ps1
    } else {
        Write-Host "ERROR: No .venv found. Run: python -m venv .venv" -ForegroundColor Red
        exit 1
    }
}

function Run-Stage {
    param([string]$Title, [string]$Command)
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host " STAGE: $Title" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PIPELINE FAILED at stage: $Title" -ForegroundColor Red
        exit 1
    }
    Write-Host "Stage complete: $Title" -ForegroundColor Green
}

Run-Stage "1/7 Data Validation (Pandera)"     "python src/data/validation.py"
Run-Stage "2/7 EDA + Imbalance Plot"          "python scripts/day1_eda.py"
Run-Stage "3/7 Preprocessing (split+scale)"   "python src/features/preprocessing.py"
Run-Stage "4/7 Training + MLflow Tracking"    "python src/models/train.py"
Run-Stage "5/7 Threshold Optimization"        "python src/models/threshold_optimization.py"
Run-Stage "6/7 Model Registry + Promotion"    "python src/models/register_model.py"
Run-Stage "7/7 API Tests (pytest)"            "pytest tests/test_api.py -v"

Write-Host ""
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host ""
Write-Host "Next (separate terminals):" -ForegroundColor Yellow
Write-Host "  API:        uvicorn api.main:app --reload  ->  http://127.0.0.1:8000/docs"
Write-Host "  MLflow UI:  mlflow ui                       ->  http://127.0.0.1:5000"
Write-Host "  Monitoring: python src/monitoring/drift.py"
Write-Host "  Retraining: python src/models/retrain.py"