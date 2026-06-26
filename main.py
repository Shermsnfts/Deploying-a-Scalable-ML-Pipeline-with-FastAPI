import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.model import inference, load_model
from ml.data import process_data

# Define the input schema with Field aliases to match the hyphenated census column names
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        populate_by_name = True

app = FastAPI()

# Load model, encoder, and label binarizer at startup
# Ensure these files exist in your 'model/' directory after running train_model.py
model = load_model("model/model.pkl")
encoder = load_model("model/encoder.pkl")
lb = load_model("model/lb.pkl")

@app.get("/")
async def root():
    """Welcome message endpoint."""
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/predict")
async def predict(data: CensusData):
    """
    Takes census data as JSON, processes it using the encoder/lb, 
    and returns the model's income prediction.
    """
    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    
    # List of categorical features used during training
    cat_features = [
        "workclass", "education", "marital-status", "occupation", 
        "relationship", "race", "sex", "native-country"
    ]
    
    # Process the data (training=False because we are using the existing encoder/lb)
    X, _, _, _ = process_data(
        df, 
        categorical_features=cat_features, 
        training=False, 
        encoder=encoder, 
        lb=lb
    )
    
    # Run inference
    pred = inference(model, X)
    
    # Inverse transform the prediction back to label ('<=50K' or '>50K')
    result = lb.inverse_transform(pred)[0]
    
    return {"result": result}
