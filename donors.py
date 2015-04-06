#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script computes how likely
somebody will contributes for charity.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Reads project's classes
from lib.importer import Importer
from lib.preprocessor import DataFrameImputer
from lib.preprocessor import Preprocessor
from lib.analyser import Analyser
from lib.utils import Performance

if __name__ == '__main__':

    #### Importation ####

    # Loads configuration
    cfg = Importer.get_cfg()

    # Loads raw data
    raw_dat = Importer.get_raw_dat(cfg)

    #### Exploratory Analysis ####

    raw_dat.shape
    raw_dat.count() # checks how many missing values are in the dataset
    raw_dat.head()
    raw_dat.columns

    # Distribution of the target variables
    plt.plot(raw_dat.TARGET_B)
    plt.plot(raw_dat[raw_dat.TARGET_D > 0].TARGET_D)

    # Correlation between TARGET_B and the predictors
    TARGET_B_corr = raw_dat.corr()["TARGET_B"].copy()
    TARGET_B_corr.sort(ascending = False)
    TARGET_B_corr

    # Correlation between TARGET_D and the predictors
    TARGET_D_corr = raw_dat.corr()["TARGET_D"].copy()
    TARGET_D_corr.sort(ascending = False)
    TARGET_D_corr

    # Some statistics about raw_dat's variables
    raw_dat.describe()

    # Variable distribution

    #[1:28] # demographics
    #[29:42] # response to other types of mail orders
    #[43:55] # overlay data
    #[56:74] # donor interests
    #[75] # PEP star RFA status
    #[76:361] # characteristics of donor neighborhood
    #[362:407] # promotion history
    #[408:412] # summary variables of promotion history
    #[413:456] # giving history
    #[457:469] # summary variables of giving history
    #[470:473] ## ID & TARGETS
    #[474:479] # RFA (recency-frequency-donation amount)
    #[480:481] # cluster & geocode

    # % of donors
    print 'Percentage of donors: %s' % (
        100.0 * sum(raw_dat.TARGET_B) / raw_dat.shape[0])

    # TODO: see how donations are distributed among age groups
    # TODO: see how donations are distributed per gender

    # This data is quite noisy, high dimensional, with lots of missing values
    # and just with 5% of positive cases. Feature selection and preprocessing
    # will be vital for good modelling.

    #### Preprocessing ####

    # Gets some redundant variables based on variance, sparsity & common sense
    redundant_vars = Analyser.get_redundant_vars(cfg, raw_dat)

    # Drops redundant cols
    dat = raw_dat.drop(redundant_vars, axis = 1)

    # Imputes the data and fills in the missing values
    dat = Preprocessor.fill_nans(dat)

    # Shuffles observations
    dat.apply(np.random.permutation)

    #### Feature Selection ####

    # Gets important variables
    important_vars = Analyser.get_important_vars(cfg, dat)

    # Drops the non-important variables
    feats = dat[important_vars]

    # Changes categorical vars to a numerical form
    feats = pd.get_dummies(dat)

    # Does train/test datasets, 70% and 30% respectively
    cut = int(feats.shape[0] * .7)

    train = feats[1:cut].drop(['TARGET_B', 'TARGET_D'], axis = 1)
    y_train = feats.TARGET_B[1:cut]

    test = feats[(cut + 1):-1].drop(['TARGET_B', 'TARGET_D'], axis = 1)
    y_test = feats.TARGET_B[(cut + 1):-1]

    # Creates a balanced trainset
    # In classification, some methods perform better with bal datasets,
    # particularly tree-based methods like decision trees and random forests.
    pos = train[y_train == 1]
    neg = train[y_train == 0][1:pos.shape[0]]
    y_train_bal = [1] * pos.shape[0]
    y_train_bal.extend([0] * neg.shape[0])
    train_bal = pos.append(neg, ignore_index = True)

    # Build Validation Set
    # TODO

    #### Model Selection ####

    # Do Grid Search for Optimal parameters, use validation set for that
    # TODO

    #### Training ####

    #### Model 1 | Decision Tree Model ####

    print "Model 1 executing..."

    # Training
    clf = DecisionTreeClassifier(max_depth = 10) # TODO: should let the tree fully grow
    # and then prune it automatically according to an optimal depth
    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_all_models = y_test_pred.copy()

    # Confusion Matrix
    print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

    # Gets performance
    perf_model1 = Performance.get_perf(y_test.values, y_test_pred)

    #### Model 2 | Random Forest Model ####

    print "Model 2 executing..."

    # Training
    clf = ExtraTreesClassifier(n_estimators = 500, verbose = 1,
        bootstrap = True, max_depth = 10, oob_score = True, n_jobs = -1)

    #clf = RandomForestClassifier(
    #    n_estimators = 500, max_depth = 10, verbose = 1, n_jobs = -1)

    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_all_models += y_test_pred

    # Confusion Matrix
    print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

    # Gets performance
    perf_model2 = Performance.get_perf(y_test, y_test_pred)

    #### Model 3 | Linear Regression Model ####

    print "Model 3 executing..."

    # Training
    clf = LogisticRegression(max_iter = 200, verbose = 1)
    clf = clf.fit(train_bal.values, y_train_bal)

    # Testing
    y_test_pred = clf.predict(test.values)
    y_all_models += y_test_pred

    # Confusion Matrix
    print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

    # Gets performance
    perf_model3 = Performance.get_perf(y_test, y_test_pred)

    #### Model 4 | Ensemble Model (majority vote for model 1, 2 and 3) ####

    print "Model 4 executing..."

    # Gets performance for an ensemble of all 3 models
    y_test_pred = np.array([0] * len(y_all_models))
    y_test_pred[y_all_models > 1] = 1
    perf_model_ensemble = Performance.get_perf(y_test, y_test_pred)

    # Confusion Matrix
    print pd.crosstab(
        y_test, y_test_pred, rownames = ['actual'], colnames = ['preds'])

    # Model comparison
    all_models = {'Decision Trees Model': perf_model1,
                  'Random Forest Model': perf_model2,
                  'Logistic Regression Model': perf_model3,
                  'Ensemble Model': perf_model_ensemble}

    perf_all_models = pd.DataFrame([[col1, col2, col3 * 100] for col1, d in
        all_models.items() for col2, col3 in d.items()], index = None,
        columns = ['Model Name', 'Performance Metric', 'Value'])

    print perf_all_models
