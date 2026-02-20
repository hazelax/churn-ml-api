from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class Customer(BaseModel):
    age: int
    salary: float
    tenure: int

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    data = np.array([[customer.age, customer.salary, customer.tenure]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }