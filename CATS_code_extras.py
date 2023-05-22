import argparse
import sys

import pandas as pd
import numpy as np
from numpy import mean, std
import math
import random

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, RocCurveDisplay, roc_auc_score, roc_curve
from sklearn.feature_selection import RFE, RFECV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

svc_params = {'C': [0.001, 0.1, 1, 10], 'gamma': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'sigmoid', 'rbf']}
rf_params = {'n_estimators': [50, 100, 250, 500], 'max_depth': [5, 10, 30], 'min_samples_split': [2, 5, 10], 'max_features': [53, 100, 500, 1000]}    # 53 bcs it is the sqrt of the length of the dataset. It was adviced in a paper that the sqrt of the size of the dataset will be good value for max_features choice.


"""
Run this file with the following command: 

python3 CATS_code_extras.py Train_call.txt Train_clinical.txt

"""


def get_filename():
    try:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
    except:
        sys.exit('\nERROR: No files given\n')

    return f1, f2


def proc_and_split(data_filename, label_filename):
    """
    * This function prepares and orients the dataset into the correct dimentions and reomving unneccessary 
        columns. Then separate the data and labels columns. 
    * Followed by an 80-20% train-test split and returns the output_.
    """

    input_data = pd.read_csv(data_filename, sep='\t')
    labels = pd.read_csv(label_filename, sep='\t')

    dataset_transposed = input_data.T
    dataset_transposed = dataset_transposed.drop(['Start', 'End', 'Nclone', 'Chromosome'])
    dataset_transposed = dataset_transposed.reset_index()

    trainset = pd.concat([dataset_transposed, labels], axis=1)
    trainset = trainset.set_index('Sample')
    trainset = trainset.drop(columns = ['index'])

    X = trainset.iloc[:, :-1].values
    y = trainset.iloc[:, -1].values

    x , y, z, w = train_test_split(X, y, test_size=0.2, random_state=0)
    return x, y, z, w


def nested_cv_param_selection(X_train, y_train):
    """
    * This function performs a Nested-Cross-Validation procedure and produces the best set of hyperparameters
        for the SVC and RF models.
    * Returns the chosen set of hyper_params_.
    """

    outer_cv = KFold(n_splits=10, shuffle=True)
    inner_cv = KFold(n_splits=5, shuffle=True)
    
    # For SVC.
    #
    svc_grid = GridSearchCV(estimator=SVC(), param_grid=svc_params, cv=inner_cv)
    svc_grid.fit(X_train, y_train)
    svc_best_params = svc_grid.best_params_
    svc_scores = cross_val_score(estimator=svc_grid, X=X_train, y=y_train, cv=outer_cv)

    print("SVC best params:", svc_best_params)
    print("SVC: Mean accuracy=%.3f, std=%.3f" % (svc_scores.mean(), svc_scores.std()))

    # For RF.
    #
    rf_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_params, cv=inner_cv)
    rf_grid.fit(X_train, y_train)
    rf_best_params = rf_grid.best_params_
    rf_scores = cross_val_score(estimator=rf_grid, X=X_train, y=y_train, cv=outer_cv)

    print("\nRandom Forest best params:", rf_best_params)
    print("Random Forest: Mean accuracy=%.3f, std=%.3f" % (rf_scores.mean(), rf_scores.std()))

    return svc_best_params, rf_best_params


def rfe_dim_reduction(X_train, y_train, svc_params, rf_params):
    """
    This function uses the RFE and RFECV modules to find the optimal number of features as well as
        the best contributing set of features, and calculates the accuracies for the SVC and RF models. 
    """

    # RFE with SVC.
    #
    # estimator = SVC(C = 0.001, gamma = 0.1, kernel='linear')
    estimator = SVC(**svc_params)
    rfe = RFE(estimator, n_features_to_select=X_train.shape[1], verbose=1)
    pipeline = Pipeline(steps=[('s',rfe),('e',estimator)])
    rfe.fit(X_train, y_train)
    
    support = np.where(rfe.support_)[0]
    ranking = np.where(rfe.ranking_)[0]

    for i in range(X_train.shape[1]):
        if rfe.support_[i] == True and rfe.ranking_[i] != 1:
            print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

    # RFECV with Random Forest.
    #
    # rf = RandomForestClassifier(n_estimators= 150, max_depth = 5, max_features = 1000, min_samples_split = 10)
    rf = RandomForestClassifier(**rf_params)
    rfecv = RFECV(estimator=rf, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X_train, y_train)

    selected_indices = np.where(rfecv.support_ == True)[0]
    print('# selected features = %.3f' % (rfecv.n_features_))
    print('List the selected features = ', selected_indices)


# Main function
def main():
    
    data_filename, label_filename = get_filename()   # Get filenames from user.

    # Process and split dataset.
    #
    X_train, X_test, y_train, y_test = proc_and_split(data_filename, label_filename)    

    # Perform Nested-CV.
    #
    svc_params, rf_params = nested_cv_param_selection(X_train, y_train)

    print("Best set of hyperparameters for SVC: ", svc_params)
    print("Best set of hyperparameters for RF: ", rf_params)

    # Dim reduction using RFE & RFECV.
    #
    rfe_dim_reduction(X_train, y_train, svc_params, rf_params)


if __name__ == '__main__':
    main()