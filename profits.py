#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script maximises a campaign's revenue.
The context and data are from the KDD Cup 98 Competition.
'''
import numpy as np
import pandas as pd

from pydoc import help
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression

# Reads project's classes
from lib.importer import Importer
from lib.preprocessor import Preprocessor
from lib.analyser import Analyser
from lib.utils import Performance

if __name__ == '__main__':

    #### Importation ####

    # Loads configuration
    cfg = Importer.get_cfg()
    cfg['target'] = 'TARGET_D'

    # Loads raw data
    raw_dat = Importer.get_raw_dat(cfg)

    # Creates a reduced version of raw_dat
    # Otherwise I can't test my buggy solution
    # TODO: optimise such that this workaround is not necessary
    pos = raw_dat[raw_dat.TARGET_B == 1]
    neg = raw_dat[raw_dat.TARGET_B == 0][1:pos.shape[0]]
    y_train_bal = [1] * pos.shape[0]
    y_train_bal.extend([0] * neg.shape[0])
    raw_dat = pos.append(neg, ignore_index = True)

    #### Exploratory Analysis ####

    # !!! It's already done at donors.py !!! #

    # Correlation between TARGET_D and the predictors
    TARGET_D_corr = raw_dat.corr()["TARGET_D"].copy()
    TARGET_D_corr.sort(ascending = False)
    TARGET_D_corr

    # TODO: see how donations are distributed among age groups
    # TODO: see how donations are distributed per gender

    # The majority of the donations are smaller than 20 dollars.

    #### Preprocessing ####

    # Gets some redundant variables based on variance, sparsity & common sense
    redundant_vars = Analyser.get_redundant_vars(cfg, raw_dat)

    # Drops redundant cols
    dat = raw_dat.drop(redundant_vars, axis = 1)

    # Imputes the data and fills in the missing values
    dat = Preprocessor.fill_nans(raw_dat)

    # Shuffles observations
    dat = dat.apply(np.random.permutation)

    #### Feature Selection ####

    # Gets important variables
    important_vars = Analyser.get_important_vars(cfg, dat)
    important_vars.extend(['TARGET_B'])

    # Changes categorical vars to a numerical form
    # TODO: find a faster alternative, or clever cleaning to the vars before
    feats = pd.get_dummies(dat)

    # Drops the non-important variables
    feats = feats[important_vars]

    # Does train/test datasets, 70% and 30% respectively
    cut = int(feats.shape[0] * .5)

    train = feats[1:cut].drop(['TARGET_B', 'TARGET_D'], axis = 1)
    y_train = feats.TARGET_B[1:cut]

    test = feats[(cut + 1):-1].drop(['TARGET_B', 'TARGET_D'], axis = 1)
    y_test = feats.TARGET_B[(cut + 1):-1]

    #### Model Selection ####

    # Do cross-validation Grid Search to find the optimal parameters
    # TODO

    #### Get Estimated Donors ####

    # Linear Regression Model to predict who are the donors

    # Training
    # TODO: do cross validation training
    clf = LogisticRegression(verbose = 1, max_iter = 200)
    clf = clf.fit(train.values, y_train.values)

    # Testing
    y_test_pred = clf.predict(test.values)

    # Confusion Matrix
    print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

    # Gets performance
    perf_model = Performance.get_perf(y_test, y_test_pred)

    # Extracts donors from dat

    sub_feats = feats[(cut + 1):-1].drop(['TARGET_B'], axis = 1)
    sub_feats = sub_feats[y_test_pred == 1]

    # Divide sub_test into train and test
    # TODO: do cross validation here
    cut = int(sub_feats.shape[0] * .5)
    train = sub_feats[1:cut]
    test = sub_feats[(cut + 1):-1]

    #### For the estimated donors predict how much they will donate ####
    # TODO: cross validation

    # Training
    clf = LinearRegression(n_jobs = -1)
    clf = clf.fit(train.drop('TARGET_D', axis = 1).values,
                  train.TARGET_D.values)

    # Testing
    y_test_pred = clf.predict(test.drop('TARGET_D', axis = 1).values)

    # Evaluates result
    print pearsonr(y_test_pred, test.TARGET_D.values)
