import os
import pytest
import pandas as pd

@pytest.fixture(scope="session")
def clean_data():

    local_path = os.path.join('./data/clean_census.csv')
    df = pd.read_csv(local_path, low_memory=False)

    return df

def test_column_presence_and_type(clean_data):
    
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education_num": pd.api.types.is_integer_dtype,
        "capital_gain": pd.api.types.is_integer_dtype,
        "capital_loss": pd.api.types.is_integer_dtype,
        "hours_per_week": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_object_dtype,
        "education": pd.api.types.is_object_dtype,
        "marital_status": pd.api.types.is_object_dtype,
        "occupation": pd.api.types.is_object_dtype,
        "relationship": pd.api.types.is_object_dtype,
        "race": pd.api.types.is_object_dtype,
        "sex": pd.api.types.is_object_dtype,
        "native_country": pd.api.types.is_object_dtype,
        "salary": pd.api.types.is_object_dtype,
    }

    # Check column presence
    assert set(clean_data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(clean_data[col_name]), f"Column {col_name} failed test {format_verification_funct}"

def test_categories(clean_data):

    categorical_features = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country',
        'salary'
    ]
    for cat in categorical_features:
        assert (clean_data[cat].str.contains('?', regex=False)==False).all()