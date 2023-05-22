#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
import pickle

# Start your coding

import pandas as pd
import numpy as np
import csv
from numpy import mean, std
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import math
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import matplotlib.pyplot as plt

# import the library you need here

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # suggested steps
    # Step 1: load the model from the model file
    # Step 2: apply the model to the input file to do the prediction
    # Step 3: write the prediction into the desinated output file
    
    # Data Reconfigure.
    #
    input_data = pd.read_csv(args.input_file, sep='\t')
    dataset_transposed = input_data.T
    unlabelled_set = dataset_transposed.drop(['Start', 'End', 'Nclone', 'Chromosome'])
    unlabelled_set = unlabelled_set.reset_index()
    unlabelled_set = unlabelled_set.set_index('index')

    # Loading Model
    #
    RF_model = pickle.load(open(args.model_file, 'rb'))

    # Generating Predictions.
    #
    y_pred = RF_model.predict(unlabelled_set)

    # Save predicted values.
    #
    sample_names = unlabelled_set.index

    predictions = pd.DataFrame({'\"Sample\"': sample_names, '\"Subgroup\"': y_pred})
    predictions.to_csv(args.output_file, sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)

    # End your coding


if __name__ == '__main__':
    main()
