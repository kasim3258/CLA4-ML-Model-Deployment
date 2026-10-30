
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Titanic ML Prediction API",
    description="Model Serving using FastAPI",
    version="1.0"
)

# Load the previously trained model
model = joblib.load("model.pkl")


# Pydantic data validation
class Passenger(BaseModel):

    Pclass: int = Field(ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None


# GET endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "message": "Titanic ML API is running"
    }


# POST endpoint
@app.post("/predict")
def predict_survival(passenger: Passenger):

    data = pd.DataFrame([passenger.model_dump()])

    prediction = model.predict(data)[0]

    result = "Survived" if prediction == 1 else "Not Survived"

    return {
        "prediction": int(prediction),
        "result": result
    }
