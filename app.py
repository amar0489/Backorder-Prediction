# FastAPI which takes input from the web app and sends the predictions using the trained model

import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.backorderproject.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI()

# Define the input data schema using Pydantic
class PredictionInput(BaseModel):
    national_inv: float  # Numerical feature
    lead_time: float  # Numerical feature
    in_transit_qty: float  # Numerical feature
    forecast_3_month: float  # Numerical feature
    sales_1_month: float  # Numerical feature
    min_bank: float  # Numerical feature
    potential_issue: str  # Categorical feature (e.g., 'yes' or 'no')
    pieces_past_due: float   #Numerical feature
    perf_6_month_avg: float  # Numerical feature
    local_bo_qty: float  # Numerical feature
    deck_risk: str  # Categorical feature (e.g., 'yes' or 'no')
    oe_constraint: str  # Categorical feature (e.g., 'yes' or 'no')
    ppap_risk: str  # Categorical feature (e.g., 'yes' or 'no')
    stop_auto_buy: str  # Categorical feature (e.g., 'yes' or 'no')
    rev_stop: str  # Categorical feature (e.g., 'yes' or 'no')

# Loading the model and preprocessor
model_path = os.path.join("artifacts", "model.pkl.gz")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

# Initialize the prediction pipeline
prediction_pipeline = PredictionPipeline(model_path, preprocessor_path)

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Converting the dictionary into a DataFrame, this is necessary for preprocessing
        input_df = pd.DataFrame([input_data.dict(exclude_unset=True)])  # Convert to DataFrame with a single row

         # Start measuring time
        start_time = time.time()

        # Get prediction
        prediction = prediction_pipeline.predict(input_df)

        # End measuring time
        end_time = time.time()
        latency_seconds = end_time - start_time  # Calculate latency

        # Get prediction from the prediction pipeline
        prediction = prediction_pipeline.predict(input_df)
        
        return {"prediction": prediction, 'latency_seconds':latency_seconds}

    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(ex)}")
