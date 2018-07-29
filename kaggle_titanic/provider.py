# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:10:52 2018

@author: Gorkem Polat
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]
    
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index = X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)