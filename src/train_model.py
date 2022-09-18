#Import libraries
import os
import gzip
import pickle
import logging
# Sklearn
from sklearn.model_selection import train_test_split
# src code
from preprocessing.data_processing import preprocess_data
from modeling.model import train_random_forest, feature_engineer
from preprocessing.data_processing import load_data
from preprocessing.data_processing import DATAFOLDER
from evaluate_model import evaluate_test_by_slice, load_model, load_holdout

MODELFOLDER = 'model'
DATAFOLDER = 'data'

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def main():
    
    logger.info("Loading data file")
    # Add code to load in the data.
    data = load_data('census.csv')

    logger.info("Preprocessing data file")
    # Process the test data with the process_data function.
    clean_data = preprocess_data(data)

    numeric_features = [
        'age',
        'fnlgt',
        'education_num',
        'hours_per_week',
        'capital_diff'
    ]

    categorical_features = [
        'workclass',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country'
    ]

    feature_cols = categorical_features + numeric_features
    target_col = ['salary']

    # Define feature space and target
    X = feature_engineer(clean_data)
    X = clean_data[feature_cols]
    y = clean_data[target_col]
    
    logger.info("Splitting train and test file")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    # Train and save a model.
    logger.info("Training random forest model")
    clf = train_random_forest(X_train, y_train)

    return clf, X_test, y_test

def save_model(model, pklname='rf_model.p'):
    
    logger.info("Saving pickled model file")
    with gzip.open(os.path.join(MODELFOLDER, pklname),'wb') as fout:
        pickle.dump(model, fout)

def save_holdout(X_test, y_test):
    logger.info("Saving holdout set")
    with gzip.open(os.path.join(DATAFOLDER, 'X_test.pkl'),'wb') as fout:
        pickle.dump(X_test, fout)
    with gzip.open(os.path.join(DATAFOLDER, 'y_test.pkl'),'wb') as fout:
        pickle.dump(y_test, fout)

if __name__ == '__main__':
    
    clf, X_test, y_test = main()
    save_model(clf)
    save_holdout(X_test, y_test)


