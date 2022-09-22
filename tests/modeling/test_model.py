import sys
import os
import pytest
import pandas as pd
CWD = os.getcwd()
sys.path.append('{}/src'.format(CWD))
from src.modeling.model import feature_engineer

@pytest.fixture(scope="session")
def clean_data():

    local_path = 'data/clean_census.csv'
    df = pd.read_csv(local_path, low_memory=False)

    return df

def test_feature_engineer_output(clean_data):

    cleaned_data = feature_engineer(clean_data)
    assert 'capital_diff' in cleaned_data.columns
