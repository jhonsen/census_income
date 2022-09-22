import os
import sys
import pytest
import pandas as pd
CWD = os.getcwd()
sys.path.append('{}/src'.format(CWD))
from src.modeling.constants import DATAFOLDER, MODELFOLDER
from src.preprocessing.data_processing import preprocess_data


@pytest.fixture(scope="session")
def raw_data():

    local_path = 'data/census.csv'
    df = pd.read_csv(local_path, low_memory=False)

    return df

@pytest.fixture(scope="session")
def clean_data():

    local_path = 'data/clean_census.csv'
    df = pd.read_csv(local_path, low_memory=False)

    return df

@pytest.fixture(scope="session")
def model_path():
    return MODELFOLDER

@pytest.fixture(scope="session")
def data_path():
    return DATAFOLDER

def test_loaded_data(clean_data):
    
    try:
        assert clean_data.shape[0] > 0
        assert clean_data.shape[1] > 0
    except FileNotFoundError as err:
        raise err

def test_dir_paths(data_path,model_path):
    
    try:
        assert os.path.isdir(model_path)
    except AssertionError as err:
        raise err

    try:
        assert os.path.isdir(data_path)
    except AssertionError as err:
        raise err


def test_preprocess_data(raw_data):

    processed_data = preprocess_data(raw_data)

    try:
        num_dashes = len([c for c in processed_data.columns if '-' in c])
        assert num_dashes == 0
    except NameError as err:
        raise err