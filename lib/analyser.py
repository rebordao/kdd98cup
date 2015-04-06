'''
Contains all methods to do analysis in this project.
'''

import numpy as np
import pandas as pd
import operator
from sklearn import feature_selection
from sklearn import ensemble

from preprocessor import DataFrameImputer
from preprocessor import Preprocessor

class Analyser:

    @staticmethod
    def get_corr_vars(dat, corr_val):
        '''
        Outputs a list of redundant vars that are correlated with others.
        '''

        # Computes correlation
        dat_cor = dat.corr()

        # Cherry picks the lower triangular, excludes diagonal
        dat_cor.loc[:, :] = np.tril(dat_cor, k = -1)

        # Stacks the data.frame
        dat_cor = dat_cor.stack()

        # Get list of correlated vars
        corr_pairs = dat_cor[dat_cor > corr_val].to_dict().keys()
        chosen_vars = [i[0] for i in corr_pairs]
        chosen_vars.extend([i[1] for i in corr_pairs if i[1] not in chosen_vars])

        redundant_vars = [var for var in [
            x for t in corr_pairs for x in t] if var not in chosen_vars]

        return redundant_vars

    @staticmethod
    def get_redundant_vars(cfg, dat):
        '''
        This method outputs a set of redundant variables.
        '''

        # Some vars that don't seem of good value
        redundant_vars = ['CONTROLN', 'ZIP']

        # Identifies numerical variables with variance zero < 0.1%
        #sel = feature_selection.VarianceThreshold(threshold = 0.001)
        #sel.fit_transform(dat)
        dat_var = dat.var()
        redundant_vars.extend(dat_var.index[dat_var < 0.001])

        # Identifies variables that are too sparse (less than 1%)
        idxs = dat.count() < int(dat.shape[0] * .01)
        redundant_vars.extend(dat.columns[idxs])

        # Identifies variables that are strongly correlated with others
        #redundant_vars.extend(Analyser.get_corr_vars(dat, corr_val = 0.9))

        return redundant_vars

    @staticmethod
    def get_important_vars(cfg, dat):
        '''
        This method does Feature Selection.
        '''

        # Balances the dataset
        idxs_pos = dat.TARGET_B == 1
        pos = dat[idxs_pos]
        neg = dat[dat.TARGET_B == 0][1:sum(idxs_pos)]

        # Concatenates pos and neg, it's already shuffled
        sub_dat = pos.append(neg, ignore_index = True)

        # Imputes the data and fills in the missing values
        sub_dat = Preprocessor.fill_nans(sub_dat)

        X = sub_dat.drop("TARGET_B", axis = 1)
        y = sub_dat.TARGET_B

        # Changes categorical vars to a numerical form
        X = pd.get_dummies(X)

        # Unfortunately all these methods remove column names...

        # Variance-based Feature Selection
        #sel = feature_selection.VarianceThreshold(threshold = 0.005)
        #X = sel.fit_transform(X)

        # Univariate Feature Selection
        #X_new = feature_selection.SelectKBest(
        #    feature_selection.chi2, k = 10).fit_transform(X.values, y.values)

        # Tree-based Feature Selection
        clf = ensemble.ExtraTreesClassifier()
        X_new = clf.fit(X.values, y.values).transform(X.values)

        aux = dict(zip(X.columns, clf.feature_importances_))
        important_vars = [i[0] for i in sorted(
            aux.items(), key = operator.itemgetter(0))]

        important_vars = ["AGE", "AVGGIFT", "CARDGIFT", "CARDPM12",
                          "CARDPROM", "CLUSTER2", "DOMAIN", "GENDER",
                          "GEOCODE2", "HIT", "HOMEOWNR", "HPHONE_D",
                          "INCOME", "LASTGIFT", "MAXRAMNT", "MDMAUD_F",
                          "MDMAUD_R", "MINRAMNT", "NGIFTALL", "NUMPRM12",
                          "PCOWNERS", "PEPSTRFL", "PETS", "RAMNTALL",
                          "RECINHSE", "RFA_2A", "RFA_2F", "STATE",
                          "TIMELAG", "TARGET_B"]

        import_vars = []

        return important_vars
