import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. Load Data
    print("Loading data from DVC...")
    df = pd.read_csv("data/raw/creditcard.csv")
    print(f"Dataset shape: {df.shape}")
    
    # 2. Check Class Imbalance
    class_counts = df['Class'].value_counts()
    print("\nClass Distribution (Raw Counts):")
    print(class_counts)
    
    legit_pct = class_counts[0] / len(df) * 100
    fraud_pct = class_counts[1] / len(df) * 100
    print(f"\nLegitimate: {legit_pct:.4f}%")
    print(f"Fraud: {fraud_pct:.4f}%")
    
    # 3. Save a quick plot for the report
    Path("reports").mkdir(exist_ok=True)
    sns.countplot(x='Class', data=df)
    plt.title("Class Distribution (Severe Imbalance)")
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.savefig("reports/class_imbalance.png")
    print("\nSaved plot to reports/class_imbalance.png")

if __name__ == "__main__":
    main()