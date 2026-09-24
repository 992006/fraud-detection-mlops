import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support

def sweep_thresholds(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
        })

    table = pd.DataFrame(rows)
    best_row = table.loc[table["f1"].idxmax()]
    return table, best_row

def main():
    print("Loading data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    spw = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                 scale_pos_weight=spw, random_state=42, eval_metric='logloss'),
    }

    Path("reports").mkdir(exist_ok=True)
    summary = {}

    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        print(f"\nSweeping thresholds for {name}...")
        table, best = sweep_thresholds(model, X_train, X_test, y_train, y_test)

        # Save per-model threshold table (report artifact)
        safe_name = name.replace(" ", "_").lower()
        table.to_csv(f"reports/threshold_table_{safe_name}.csv", index=False)

        summary[name] = {
            "best_threshold": float(best["threshold"]),
            "precision_at_best": float(best["precision"]),
            "recall_at_best": float(best["recall"]),
            "best_f1": float(best["f1"]),
        }

        plt.plot(table["threshold"], table["f1"], marker='o', label=name)
        print(f"  Best threshold: {best['threshold']:.2f} | F1: {best['f1']:.4f}")

    # Save summary JSON (the registry and API will use this next)
    with open("reports/best_thresholds.json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.xlabel("Probability Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs Threshold (All Models)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("reports/threshold_comparison_all_models.png")

    print("\n=== BEST THRESHOLD PER MODEL ===")
    print(pd.DataFrame(summary).T.to_string())
    print("\n✅ Saved reports/best_thresholds.json")
    print("✅ Saved reports/threshold_comparison_all_models.png")

if __name__ == "__main__":
    main()