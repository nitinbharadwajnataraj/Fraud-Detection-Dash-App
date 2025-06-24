from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any
import joblib
import pandas as pd
from pathlib import Path
import json
import traceback

app = FastAPI()

class PredictRequest(BaseModel):
    model_name: str
    columns: List[str]
    rows: List[List[Any]]

# Load expected raw column names (before encoding)
with open("results/encoder_columns.json") as f:
    expected_columns = json.load(f)

# Load encoder
encoder = joblib.load("results/encoder.pkl")

# Load all trained models
model_dir = Path("results")
models = {}
for model_file in model_dir.glob("*.pkl"):
    if model_file.name != "encoder.pkl":
        model_name = model_file.stem
        models[model_name] = joblib.load(model_file)

@app.get("/")
def root():
    return {"status": "API is up", "available_models": list(models.keys())}

@app.post("/batch_predict")
def batch_predict(req: PredictRequest):
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    if not all(len(row) == len(req.columns) for row in req.rows):
        raise HTTPException(status_code=400, detail="Row length mismatch with columns")

    try:
        # Create DataFrame from request
        df = pd.DataFrame(req.rows, columns=req.columns)

        # Check for missing columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        # Align DataFrame to match expected column order exactly
        df = df[expected_columns]

        # Transform input using saved encoder
        transformed = encoder.transform(df)

        # Predict using selected model
        preds = models[req.model_name].predict(transformed)

        return {"predictions": preds.tolist()}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
