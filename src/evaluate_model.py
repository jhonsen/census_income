import os
import gzip
import pickle
import logging
import pandas as pd
from modeling.model import compute_model_metrics, inference
from preprocessing.data_processing import DATAFOLDER
from train_model import MODELFOLDER

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def load_holdout():

    logger.info("Loading holdout set")
    with gzip.open(os.path.join(DATAFOLDER, 'X_test.pkl'),'rb') as fin:
        X_test = pickle.load(fin)
    with gzip.open(os.path.join(DATAFOLDER, 'y_test.pkl'),'rb') as fin:
        y_test = pickle.load(fin)
    return X_test, y_test

def load_model(pklname='rf_model.p'):
    
    logger.info("Loading pickled model file")
    with gzip.open(os.path.join(MODELFOLDER, pklname),'rb') as fin:
        clf = pickle.load(fin)
    return clf

def evaluate_test_by_slice(clf, y_preds, colname):

    X_test, y_test = load_holdout()
    
    # get a slice by colname
    logger.info(f'Evaluating by slice:{colname}')
    unique_slices = X_test[colname].unique().tolist()
    
    output = 'METRICS by feature slice: workplace'
    for unq_slice in unique_slices:
        X_subset = X_test[X_test.workclass==unq_slice]
        X_subset_idx = X_subset.index.tolist()
        
        y_subset = y_test[y_test.index.isin(X_subset_idx)]
        
        y_preds = inference(clf, X_subset)
        precision, recall, fbeta = compute_model_metrics(y_subset, y_preds)
        output += f'\n{unq_slice}, precision: {precision:.2f}, recall: {recall:2f}, fbeta: {fbeta:.2f}'
    
    return output

def save_metrics_by_slice(metrics_output):

    logger.info("Saving metrics by slice")
    with open(os.path.join(DATAFOLDER,'slice_output.txt'),'w') as fout:
        fout.write(metrics_output)
    
    