import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main():
    # 1. Load the raw data
    print("Loading raw data...")
    df = pd.read_csv("data/raw/creditcard.csv")
    
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # 2. Stratified Train/Test Split
    # stratify=y ensures the 0.17% fraud rate is kept exactly the same in both train and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Feature Scaling (Preventing Data Leakage)
    # We ONLY fit the scaler on the training data. 
    print("Scaling features...")
    scaler = StandardScaler()
    
    # Fit on Train, Transform Train
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    
    # Transform Test (DO NOT FIT on test data!)
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])
    
    # 4. Save the processed data and the scaler
    Path("data/processed").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    # We save the scaler because our FastAPI will need it later to transform live incoming transactions!
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("✅ Preprocessing complete!")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

if __name__ == "__main__":
    main()