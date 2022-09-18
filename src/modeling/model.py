from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.combine import SMOTEENN


def feature_engineer(X_train):

    # combine feature
    X_train['capital_diff'] = (X_train['capital_gain'] - X_train['capital_loss'])

    return X_train

# Optional: implement hyperparameter tuning.
def train_random_forest(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # select features
    capital_diff_feature = ['capital_diff']
    numeric_features = ['age','fnlgt','education_num','hours_per_week']
    categorical_features = ['workclass', 'marital_status', 'occupation', 'relationship',
        'race', 'sex', 'native_country']


    # Setup transformers
    numeric_transformer = make_pipeline(SimpleImputer())
    categorical_transformer = make_pipeline(OneHotEncoder(drop='first',handle_unknown='ignore'))

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features+capital_diff_feature),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # This drops the columns that we do not transform
    )

    classifier = RandomForestClassifier(n_jobs=-1)

    train_cols = numeric_features + categorical_features + ['capital_diff']

    model = imbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("resampler",  SMOTEENN(n_jobs=-1, random_state=42)),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train[train_cols], y_train.values.ravel())

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    
    return predictions