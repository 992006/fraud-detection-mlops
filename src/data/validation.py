import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import sys

def get_schema():
    # Define the rules our data MUST follow
    schema = DataFrameSchema({
        "Time": Column(float, Check.ge(0), nullable=False), # Must be >= 0
        "Amount": Column(float, Check.ge(0), nullable=False), # Money can't be negative
        "Class": Column(int, Check.isin([0, 1]), nullable=False) # Must be exactly 0 or 1
    })
    return schema

def validate_data():
    print("Loading data...")
    df = pd.read_csv("data/raw/creditcard.csv")
    
    print("Validating schema...")
    schema = get_schema()
    
    try:
        # lazy=True means it checks ALL columns before reporting errors
        schema.validate(df, lazy=True)
        print("✅ Data validation passed! The dataset is clean and matches our schema.")
        return True
    except pa.errors.SchemaErrors as err:
        print("❌ Data validation failed! Found bad data:")
        print(err.failure_cases)
        return False

if __name__ == "__main__":
    success = validate_data()
    # Exit with code 0 if successful, 1 if failed (important for CI/CD later)
    sys.exit(0 if success else 1)