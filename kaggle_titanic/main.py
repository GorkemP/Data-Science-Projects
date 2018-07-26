# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:59:14 2018

@author: Gorkem Polat
"""
from os.path import dirname, abspath
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

def load_titanic_data(fileName):
    fileName = dirname(dirname(abspath(__file__)))+'\datasets\\'+fileName
    return pd.read_csv(fileName)

trainData = load_titanic_data('train.csv')
testData = load_titanic_data('test.csv')

