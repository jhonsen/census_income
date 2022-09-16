import os
import pandas as pd
import numpy as np

DATAFOLDER = 'data'

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def fix_column_names(data):

    # Fix colnames for pandas
    colnames = [c.replace('-','_').strip() for c in data.columns]
    data.columns = colnames

    return data

def preprocess_data(data):
    
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
    numerical_features = [
        'age',
        'fnlgt',
        'education_num',
        'capital_gain',
        'capital_loss',
        'hours_per_week'
    ]
    
    data = fix_column_names(data)

    # replace nans with mean
    for cat in numerical_features:
        data[cat].fillna(data[cat].mean(), inplace=True)

    # replace nans with mode
    for cat in categorical_features:
        data[cat] = data[cat].apply(lambda s: s.strip())
        data.loc[data[cat]=="?", cat] = np.nan
        data[cat].fillna(data[cat].mode().values[0], inplace=True)        

    return data

if __name__ == "__main__":

    census_data = load_data(os.path.join(DATAFOLDER, 'census.csv'))
    clean_data = preprocess_data(census_data)
    clean_data.to_csv(os.path.join(DATAFOLDER, 'clean_census.csv'))