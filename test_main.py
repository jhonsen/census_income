import os
import gzip
import pickle
from fastapi.testclient import TestClient
from src.modeling.constants import DATAFOLDER, MODELFOLDER
# Import our app from main.py.
from main import app

MODELNAME = 'rf_model.p'
with gzip.open(os.path.join(MODELFOLDER, MODELNAME),'rb') as fin:
    model = pickle.load(fin)
with gzip.open(os.path.join(DATAFOLDER, 'X_test.pkl'),'rb') as fin:
    X_test = pickle.load(fin)

# Instantiate the testing client with our app.
client = TestClient(app)

def test_api_locally_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}

def test_predict_holdout():
    response = client.post(
        "/predict_holdout?row={}".format(0),
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": ">50K",
        "probability": 0.94
    }


def test_predict_new_data():
    response = client.post(
        "/predict_new_data",
        json={
            "age": 34,
            "fnlgt": 2321,
            "education_num": 13,
            "hours_per_week": 40,
            "capital_diff": 2000,
            "workclass": "State-gov",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "native_country": "United-States"
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": "<=50K",
        "probability": 0.13
    }