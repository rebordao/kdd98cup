'''
Contains all methods to do preprocessing in this project.
'''

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class Preprocessor:

    @staticmethod
    def fill_nans(dat):
        '''
        Fills in NaNs with either the mean or the most common value.
        '''

        return DataFrameImputer().fit_transform(dat)

class DataFrameImputer(TransformerMixin):
    '''
    This class came from http://stackoverflow.com/questions/25239958/
    impute-categorical-missing-values-in-scikit-learn
    '''

    def __init__(self):
        '''
        Impute missing values.
        Columns of dtype object are imputed with the most frequent value in col.
        Columns of other types are imputed with mean of column.
        '''

    def fit(self, X, y = None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index = X.columns)

        return self

    def transform(self, X, y = None):
        return X.fillna(self.fill)
