from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn

# Load the saved model
model_path = "../model/wine_rmodel.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI
app = FastAPI()

# Define input data model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def read_root():
    return {"Project": "Wine Prediction Project."}

@app.post("/predict/")
def predict_wine_quality(features: WineFeatures):
    from fastapi.encoders import jsonable_encoder
    print("Received raw request:", jsonable_encoder(features))  # Debugging raw request
    
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)
    
    return {"predicted_quality": int(prediction[0])}
