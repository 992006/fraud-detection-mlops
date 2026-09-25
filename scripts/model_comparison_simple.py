import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
out = Path("reports")

print("Loading model results...")
# Read the CSV that train.py generated
df = pd.read_csv(out / "model_comparison.csv")

# Make sure column names are clean
df.columns = [c.strip().title() for c in df.columns]

print("\n===== MODEL RESULTS =====")
print(df.to_string(index=False))

# ---------- 1. Simple Grouped Bar Chart (Precision, Recall, F1) ----------
fig, ax = plt.subplots(figsize=(10, 6))

# Pandas makes grouped bar charts super easy!
df.plot(x="Model", y=["Precision", "Recall", "F1"], kind="bar", 
        ax=ax, color=["#4f8cff", "#f59e0b", "#22c55e"], width=0.7)

ax.set_title("Model Comparison: Precision vs Recall vs F1", fontsize=18, fontweight='bold')
ax.set_ylabel("Score (0 to 1)", fontsize=14)
ax.set_xlabel("Machine Learning Model", fontsize=14)
ax.set_xticklabels(df["Model"], rotation=0, fontsize=12)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=12)

# Add exact numbers on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9, fontweight='bold')

fig.tight_layout()
fig.savefig(out / "model_comparison_simple.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved model_comparison_simple.png")

# ---------- 2. The "Winner" Chart (F1-Score Leaderboard) ----------
fig, ax = plt.subplots(figsize=(9, 5))

# Sort by F1 score so the winner is at the top
df_sorted = df.sort_values(by="F1")

# Color the winner gold, others grey
colors = ["#cbd5e1"] * (len(df_sorted) - 1) + ["#f59e0b"]

sns.barplot(data=df_sorted, x="F1", y="Model", hue="Model", legend=False,
            palette=colors, ax=ax)

ax.set_title("🏆 F1-Score Leaderboard (Gold = Champion Model)", fontsize=18, fontweight='bold')
ax.set_xlabel("F1-Score (Higher is Better)", fontsize=14)
ax.set_ylabel("Model", fontsize=14)
ax.set_xlim(0, 1.0)

# Add the exact score next to the bar
for index, value in enumerate(df_sorted["F1"]):
    ax.text(value + 0.02, index, f"{value:.3f}", va='center', fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig(out / "model_comparison_winner.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved model_comparison_winner.png")

print("\n" + "="*50)
print("Simple comparison graphs saved in reports/!")
print("="*50)