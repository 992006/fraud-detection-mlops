from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Fraud Detection API",
    description="Production-style fraud detection inference API",
    version="0.1.0",
)

class Transaction(BaseModel):
    amount: float
    time: float
    features: dict = {}

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Placeholder: later this will load the actual ML model
    fraud_probability = 0.5 
    prediction = "FRAUD" if fraud_probability >= 0.5 else "LEGIT"

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction,
    }