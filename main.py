import os
import gzip
import pickle
from re import T
import pandas as pd
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.modeling.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.modeling.constants import DATAFOLDER,MODELFOLDER

MODELNAME = 'rf_model.p'
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Instantiate the app.
app = FastAPI()

class FeaturesIn(BaseModel):
    workclass: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlgt: int
    education_num: int
    hours_per_week: int
    capital_diff: int

class PredictionsOut(BaseModel):
    prediction: str
    probability: float

# load model
with gzip.open(os.path.join(MODELFOLDER, MODELNAME),'rb') as fin:
    model = pickle.load(fin)
with gzip.open(os.path.join(DATAFOLDER, 'X_test.pkl'),'rb') as fin:
    X_test = pickle.load(fin)

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# Define a POST method
@app.post('/predict_holdout')
def predict_holdout(row: int):
    assert (row < 3229) & (row >= 0), 'i must be between 0 and 3228 rows'

    input_data = pd.DataFrame(X_test.reset_index().iloc[row],index=FEATURES).T
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {'prediction':prediction, 'probability':probability}
    
@app.post('/predict_new_data', response_description=FeaturesIn)
def predict_new_data(data:FeaturesIn):
    
    input_data = pd.DataFrame(data.dict(),index=[0])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return  {
        'prediction': prediction, 'probability': probability
    }
    