from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Optional

app = FastAPI(
    title="Personality Predictor API", 
    description="Predicts if a person is an Introvert or Extrovert based on behavioral metrics."
)

# Load the saved model pipeline
try:
    model = joblib.load('model/model_pipeline.pkl')
except FileNotFoundError:
    model = None

# Define input schema matching our dataset columns
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
        "message": "Personality Predictor API is operational. Send a POST request to /predict"
    }

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model pipeline file 'model_pipeline.pkl' is missing on the server.")
    
    # Convert incoming request data to a DataFrame matching the model's feature names
    input_dict = {
        'Time_spent_Alone': [input_data.Time_spent_Alone],
        'Stage_fear': [input_data.Stage_fear],
        'Social_event_attendance': [input_data.Social_event_attendance],
        'Going_outside': [input_data.Going_outside],
        'Drained_after_socializing': [input_data.Drained_after_socializing],
        'Friends_circle_size': [input_data.Friends_circle_size],
        'Post_frequency': [input_data.Post_frequency]
    }
    
    df_input = pd.DataFrame(input_dict)
    
    try:
        # Predict class (0: Introvert, 1: Extrovert)
        prediction_encoded = model.predict(df_input)[0]
        prediction_label = "Extrovert" if prediction_encoded == 1 else "Introvert"
        
        # Calculate confidence
        probabilities = model.predict_proba(df_input)[0]
        confidence = float(probabilities[prediction_encoded])
        
        return {
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")