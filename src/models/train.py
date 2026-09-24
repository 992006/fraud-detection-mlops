import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import mlflow
import mlflow.sklearn

def get_metrics(model, X_test, y_test):
    """Evaluate a model and return metrics dictionary."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }

def main():
    print("Loading preprocessed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    
    # Calculate scale_pos_weight for XGBoost to handle severe imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    
    # Set MLflow Experiment Name
    mlflow.set_experiment("fraud-detection-baseline")
    
    models_to_train = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1, 
            scale_pos_weight=spw,  # XGBoost's way of handling imbalance
            random_state=42, 
            eval_metric='logloss'
        )
    }
    
    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")
        
        # Start an MLflow run for this specific model
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            
            # 1. Log Parameters
            params = model.get_params()
            mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})
            mlflow.log_param("model_name", model_name)
            
            # Log the specific imbalance handling technique used
            if hasattr(model, 'class_weight'):
                mlflow.log_param("imbalance_handling", "class_weight=balanced")
            elif hasattr(model, 'scale_pos_weight'):
                mlflow.log_param("imbalance_handling", f"scale_pos_weight={spw:.2f}")
            
            # 2. Log Metrics
            metrics = get_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            
            # 3. Log the model itself using cloudpickle
            mlflow.sklearn.log_model(
                model, 
                name=f"{model_name}_model", 
                serialization_format="cloudpickle"
            )
            
            print(f"  F1: {metrics['f1_score']} | Recall: {metrics['recall']} | Precision: {metrics['precision']}")
            print(f"  ✅ Run logged to MLflow!")

    print("\n✅ All training complete!")

if __name__ == "__main__":
    main()