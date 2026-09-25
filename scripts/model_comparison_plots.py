import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

out = Path("reports")
out.mkdir(exist_ok=True)

def getm(m, keys):
    for k in keys:
        if k in m:
            return m[k]
    return np.nan

# ---------- Load latest results of each model from MLflow ----------
rows = []
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp = client.search_experiments()[0]
    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                              order_by=["attributes.start_time DESC"])
    latest = {}
    for r in runs:
        name = r.info.run_name or "unknown"
        if name not in latest:
            latest[name] = r
    for name, r in latest.items():
        m = r.data.metrics
        rows.append({
            "Model": name,
            "Precision": getm(m, ["precision", "Precision", "test_precision"]),
            "Recall":    getm(m, ["recall", "Recall", "test_recall"]),
            "F1":        getm(m, ["f1_score", "f1", "F1"]),
            "ROC-AUC":   getm(m, ["roc_auc", "roc_auc_score", "auc"]),
            "PR-AUC":    getm(m, ["pr_auc", "pr_auc_score", "average_precision"]),
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} models from MLflow.")
except Exception as e:
    print("MLflow query failed:", e)
    df = pd.read_csv(out / "model_comparison.csv")
    print("Loaded comparison from model_comparison.csv instead.")

if df is None or len(df) == 0:
    raise SystemExit("No results found. Run: python src/models/train.py first.")

print("\n===== MODEL COMPARISON =====")
print(df.to_string(index=False))

# ---------- 1. Precision / Recall / F1 grouped bars ----------
cols = [c for c in ["Precision", "Recall", "F1"] if c in df.columns and df[c].notna().any()]
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(df)); w = 0.8 / len(cols)
palette = ["#4f8cff", "#f59e0b", "#22c55e"]
for i, c in enumerate(cols):
    ax.bar(x + i*w - (len(cols)-1)*w/2, df[c], width=w, label=c, color=palette[i % 3])
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], fontsize=12)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Precision / Recall / F1", fontsize=16, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out / "model_comparison_metrics.png", dpi=150)
plt.close(fig)
print("✓ Saved model_comparison_metrics.png")

# ---------- 2. ROC-AUC / PR-AUC grouped bars ----------
auc_cols = [c for c in ["ROC-AUC", "PR-AUC"] if c in df.columns and df[c].notna().any()]
if auc_cols:
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(df)); w = 0.8 / len(auc_cols)
    for i, c in enumerate(auc_cols):
        ax.bar(x + i*w - (len(auc_cols)-1)*w/2, df[c], width=w, label=c, color=palette[i % 3])
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: ROC-AUC vs PR-AUC", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "model_comparison_auc.png", dpi=150)
    plt.close(fig)
    print("✓ Saved model_comparison_auc.png")

# ---------- 3. F1 Leaderboard (champion highlighted) ----------
if "F1" in df.columns and df["F1"].notna().any():
    d = df.dropna(subset=["F1"]).sort_values("F1")
    colors = ["#94a3b8"] * len(d)
    colors[-1] = "#f59e0b"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["Model"], d["F1"], color=colors)
    for i, v in enumerate(d["F1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title("F1-Score Leaderboard  (gold = champion → Production)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("F1-Score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "model_comparison_winner.png", dpi=150)
    plt.close(fig)
    print("✓ Saved model_comparison_winner.png")
    print(f"\n🏆 Champion by F1: {d['Model'].iloc[-1]}")

print("\nAll comparison graphs saved in reports/ folder.")