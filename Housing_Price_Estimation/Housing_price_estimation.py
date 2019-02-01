#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:09:05 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

randomState = 0

# fetch the data 
# By looking to the dataset['DESCR'], we can get information on the dataset
dataset = fetch_california_housing()

x = dataset["data"]
y = dataset["target"]

# divide the data into train and test sets
# 80% of the data kept for training, the rest 20% is for test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=randomState)

# Scaling the features in order to get better results
# Scaling parameters will be adjusted only for training set, otherwise it would be cheating
# We are assuming that we do not know the test set.
minMaxScaler = MinMaxScaler()
x_train = minMaxScaler.fit_transform(x_train)

# Use grid search to try different values and find best parameters
# degree = Degree of the polynomial kernel function ('poly')
# C: Penalty parameter C of the error term
regressor = SVR()

parameters = {"degree": [1, 2, 3], "C":[0.5, 1, 2]}
grid_search_CV = GridSearchCV(regressor, parameters, cv=4, scoring="neg_mean_absolute_error", verbose=3)

grid_search_CV.fit(x_train, y_train)

# Best estimator is found when C=2 and degre=1
print("Best estimator: " + str(grid_search_CV.best_params_))

# See the training set performance with the best estimator
y_train_predicted = grid_search_CV.best_estimator_.predict(x_train)
meanAbsoluteError = mean_absolute_error(y_train, y_train_predicted)
print("Mean Absolute Error (Training set): "+ str(meanAbsoluteError))

# Rescale the test set according to the training set scaling values
x_test = minMaxScaler.transform(x_test)

# See the test set performance with the best estimator
y_test_predicted = grid_search_CV.best_estimator_.predict(x_test)
meanAbsoluteError = mean_absolute_error(y_test, y_test_predicted)
print("Mean Absolute Error (Test set): "+ str(meanAbsoluteError))

""" 
I used mean absolute error because it gives the magnitude of the expected 
error when we make prediction. In this problem, expected error is around 0.51

I wanted to use GridSearch functionality of the scikit-learn; therefore, I used
their SVR implementation.

Total search is completed about 7 minutes because in total, there will be 36
training process:
---> 9 different combinations and 4 cross-validation for each of them

According to the best parameters, here are results:
    
Mean Absolute Error (Training set): 0.5091558962209776
Mean Absolute Error (Test set): 0.5114956018135732

"""
