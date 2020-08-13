#!/usr/bin/env python
import argparse
import pandas as pd
import os
import sklearn as sl
import scipy as sp

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb
from numpy import load

from sklearn import tree
#import sklearn.external.joblib as extjoblib
import joblib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--ftype', type=str, default="fft")

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    for file in input_files:
        if args.ftype == "agg":
            if "feat" in file:
                train_X = pd.read_csv(file)
            if "tgt" in file:
                train_y = pd.read_csv(file)
        elif args.ftype == "fft":
            if "fft" in file:
                train_X = load(file)
            if "tgt" in file:
                train_y = pd.read_csv(file)

    # Now use scikit-learn's decision tree classifier to train the model.
    base_model = GradientBoostingRegressor(loss='quantile',
                                      n_estimators=100,
                                      criterion='mae',
                                      random_state=2021,
                                      max_features='sqrt',
                                      n_iter_no_change=2)
    reg = MultiOutputRegressor(base_model)
    reg.fit(train_X,train_y.drop(['id'],axis=1))
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(reg, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf