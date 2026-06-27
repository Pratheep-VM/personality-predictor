from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional

# disable the default docs/redoc paths so we can serve our custom styled reference
app = FastAPI(
    title="Personality Predictor API", 
    description="Production-grade classification engine utilizing a tuned machine learning pipeline.",
    docs_url=None, 
    redoc_url=None
)

# Load the saved model pipeline
try:
    model = joblib.load('model/model_pipeline.pkl')
except FileNotFoundError:
    model = None

# Match our dataset columns exactly
class PredictionInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: Optional[float] = None
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time_spent_Alone": 9.0,
                "Stage_fear": "Yes",
                "Social_event_attendance": 0.0,
                "Going_outside": 0.0,
                "Drained_after_socializing": "Yes",
                "Friends_circle_size": 0.0,
                "Post_frequency": 3.0
            }
        }
    }

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Personality Predictor API is operational. Access developer documentation at /docs"
    }

# CUSTOM PRODUCTION DOCUMENTATION ROUTE (Scalar UI)
@app.get("/docs", include_in_schema=False)
def custom_docs():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Personality Predictor API Docs</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
          body {
            margin: 0;
            padding: 0;
          }
        </style>
      </head>
      <body>
        <!-- Serve Scalar API Reference from CDN -->
        <script
          id="api-reference"
          data-url="/openapi.json"
          data-configuration='{
            "theme": "purple",
            "showSidebar": true,
            "hideModels": false,
            "searchHotKey": "k"
          }'></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
      </body>
    </html>
    """)

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model pipeline file 'model_pipeline.pkl' is missing on the server."
        )
    
    input_dict = {
        'Time_spent_Alone': [input_data.Time_spent_Alone],
        'Stage_fear': [input_data.Stage_fear],
        'Social_event_attendance': [input_data.Social_event_attend],
        'Going_outside': [input_data.Going_outside],
        'Drained_after_socializing': [input_data.Drained_after_social],
        'Friends_circle_size': [input_data.Friends_circle_size],
        'Post_frequency': [input_data.Post_frequency]
    }
    
    df_input = pd.DataFrame(input_dict)
    
    try:
        prediction_encoded = model.predict(df_input)[0]
        prediction_label = "Extrovert" if prediction_encoded == 1 else "Introvert"
        
        probabilities = model.predict_proba(df_input)[0]
        confidence = float(probabilities[prediction_encoded])
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")