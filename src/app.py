from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import random

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_a = joblib.load(os.path.join(BASE_DIR, "models/model_a.pkl"))
model_b = joblib.load(os.path.join(BASE_DIR, "models/model_b.pkl"))

# Define input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "A/B Deployment API Running"}

@app.post("/predict")
def predict(data: InputData):
    features = data.features

    if random.random() > 0.5:
        prediction = model_a.predict([features])
        model_used = "Model A"
    else:
        prediction = model_b.predict([features])
        model_used = "Model B"

    return {
        "prediction": int(prediction[0]),
        "model_used": model_used
    }