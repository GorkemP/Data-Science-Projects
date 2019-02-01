#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 18:05:37 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import numpy as np

# data => pandas dataframe
def getSequencedData(data, n_inputs, n_steps):
    X_data = np.zeros((len(data)-n_steps, n_inputs*n_steps))
    y_data = np.zeros((len(data)-n_steps, 1))
    
    for i in range(len(data)-n_steps):
        all_sequence = data.iloc[i:i+n_steps, 1:5].values
        X_data[i,:] = np.reshape(all_sequence, (1, n_inputs*n_steps))
        
        if (data.iloc[i+n_steps, 4] > data.iloc[i+n_steps-1, 4] ):
            y_data[i] = 1
        else:
            y_data[i] = 0
    
    return X_data, y_data

# data => numpy ndarray
def getSequencedDataWithColumns(data, n_inputs, n_steps, columnStart, columnEnd, valueColumn):
    X_data = np.zeros((len(data)-n_steps, n_inputs*n_steps))
    y_data = np.zeros((len(data)-n_steps, 1))
    
    for i in range(len(data)-n_steps):
        all_sequence = data[i:i+n_steps, columnStart:columnEnd]
        X_data[i,:] = np.reshape(all_sequence, (1, n_inputs*n_steps))
        
        if (data[i+n_steps, valueColumn] > data[i+n_steps-1, valueColumn]):
            y_data[i] = 1
        else:
            y_data[i] = 0
    
    return X_data, y_data

def fetchTrainingData(X_data, y_data, batchSize):
    randomIDs = np.random.permutation(len(X_data))
    randomIDs = randomIDs[0:batchSize]
    return X_data[randomIDs], y_data[randomIDs]

def convertWeeklyToDaily(data):
    dailyTrends = np.zeros((len(data)*7))
    for i in range(len(data)):
        dailyTrends[i*7:(i*7+7)] = data.iloc[i, 1]   
    return dailyTrends
    