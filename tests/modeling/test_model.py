import sys
import os
import pytest
import gzip
import pickle
import pandas as pd
CWD = os.getcwd()
sys.path.append('{}/src'.format(CWD))
from src.modeling.model import feature_engineer, inference
from src.modeling.constants import MODELFOLDER, DATAFOLDER

@pytest.fixture(scope="session")
def clean_data():

    local_path = 'data/clean_census.csv'
    df = pd.read_csv(local_path, low_memory=False)

    return df

@pytest.fixture(scope="session")
def holdout_data():

    with gzip.open(os.path.join(DATAFOLDER, 'X_test.pkl'),'rb') as fin:
        X_test = pickle.load(fin)
    
    return X_test

@pytest.fixture(scope="session")
def model():

    with gzip.open(os.path.join(MODELFOLDER, 'rf_model.p'),'rb') as fin:
        model = pickle.load(fin)

    return model

def test_feature_engineer_output(clean_data):

    cleaned_data = feature_engineer(clean_data)
    assert 'capital_diff' in cleaned_data.columns

def test_inference(model, holdout_data):

    pred_array = inference(model, holdout_data)
    prediction = pred_array[0]
    assert prediction in (['>50K','<=50K'])

