from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import random
import logging

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Setup logging
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load models
model_a = joblib.load(os.path.join(BASE_DIR, "models/model_a.pkl"))
model_b = joblib.load(os.path.join(BASE_DIR, "models/model_b.pkl"))

# Input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Monitoring API Running"}

@app.post("/predict")
def predict(data: InputData):
    features = data.features

    # A/B testing
    if random.random() > 0.5:
        prediction = model_a.predict([features])
        model_used = "Model A"
    else:
        prediction = model_b.predict([features])
        model_used = "Model B"

    pred_value = int(prediction[0])

    # 🔥 LOGGING
    logging.info(f"Input: {features}, Prediction: {pred_value}, Model: {model_used}")

    # 🚨 ALERTING (example condition)
    if pred_value == 2:
        logging.warning("ALERT: Class 2 detected (potential anomaly)")

    return {
        "prediction": pred_value,
        "model_used": model_used
    }