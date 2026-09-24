import pandas as pd, json

# Read the dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Find the first actual fraud transaction and drop the 'Class' column
fraud_row = df[df["Class"] == 1].iloc[0].drop("Class")

# Print it as formatted JSON
print(json.dumps(fraud_row.to_dict(), indent=2))