import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
out = Path("reports")
out.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv("data/raw/creditcard.csv")

# ---------- 1. Simple Class Distribution (Pie Chart) ----------
fig, ax = plt.subplots(figsize=(8, 6))
counts = df["Class"].value_counts()
ax.pie(counts.values, labels=["Legit (0)", "Fraud (1)"], 
       colors=["#2ca02c", "#d62728"], autopct='%1.2f%%', startangle=90)
ax.set_title("Transaction Distribution", fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(out / "eda_simple_class.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved eda_simple_class.png")

# ---------- 2. Amount Comparison (Simple Boxplot) ----------
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x="Class", y="Amount", hue="Class", legend=False,
            palette={0: "#2ca02c", 1: "#d62728"}, ax=ax)
ax.set_xlabel("Transaction Type", fontsize=14)
ax.set_ylabel("Amount ($)", fontsize=14)
ax.set_title("Transaction Amount: Legit vs Fraud", fontsize=16, fontweight='bold')
ax.set_xticklabels(["Legitimate", "Fraud"])
fig.tight_layout()
fig.savefig(out / "eda_simple_amount.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved eda_simple_amount.png")

# ---------- 3. Time Pattern (Simple Histogram) ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df[df["Class"]==0]["Time"], bins=50, alpha=0.5, label="Legit", color="#2ca02c")
ax.hist(df[df["Class"]==1]["Time"], bins=50, alpha=0.7, label="Fraud", color="#d62728")
ax.set_xlabel("Time (seconds)", fontsize=14)
ax.set_ylabel("Number of Transactions", fontsize=14)
ax.set_title("When Do Frauds Happen?", fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(out / "eda_simple_time.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved eda_simple_time.png")

# ---------- 4. Key Features (2x2 Grid - Simple) ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ["V14", "V17", "V12", "V10"]
for ax, feat in zip(axes.flat, features):
    sns.boxplot(data=df, x="Class", y=feat, hue="Class", legend=False,
                palette={0: "#2ca02c", 1: "#d62728"}, ax=ax)
    ax.set_title(f"{feat} Pattern", fontsize=14, fontweight='bold')
    ax.set_xlabel("Legit vs Fraud", fontsize=12)
    ax.set_xticklabels(["Legit", "Fraud"])
fig.suptitle("Important Features for Fraud Detection", fontsize=18, fontweight='bold', y=0.995)
fig.tight_layout()
fig.savefig(out / "eda_simple_features.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved eda_simple_features.png")

# ---------- 5. Top 5 Correlations (Simple Bar) ----------
corr = df.corr(numeric_only=True)["Class"].drop("Class").sort_values(ascending=False)
top5 = corr.head(5)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#d62728" if v > 0 else "#2ca02c" for v in top5.values]
ax.bar(top5.index, top5.values, color=colors, alpha=0.8)
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Correlation with Fraud", fontsize=14)
ax.set_title("Which Features Matter Most?", fontsize=16, fontweight='bold')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for i, (feat, val) in enumerate(zip(top5.index, top5.values)):
    ax.text(i, val, f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
fig.tight_layout()
fig.savefig(out / "eda_simple_correlation.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("✓ Saved eda_simple_correlation.png")

print("\n" + "="*50)
print("All 5 simple graphs saved in reports/ folder!")
print("="*50)