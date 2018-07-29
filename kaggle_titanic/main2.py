#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 20:15:09 2018

@author: Görkem Polat
"""
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

def load_titanic_data(fileName):
    return pd.read_csv('datasets/'+fileName)

train_data = load_titanic_data('train.csv')
test_data = load_titanic_data('test.csv')

train_data_y = train_data["Survived"].values

# Handle missing values

train_data_numeric = train_data[["Age", "SibSp", "Parch", "Fare"]]
train_data_categoric = train_data[["Pclass", "Sex", "Embarked"]]

imputer = Imputer(strategy="median")
train_data_numeric_filled = imputer.fit_transform(train_data_numeric)

train_data_categoric_filled = train_data_categoric.fillna(value={"Pclass":"3", "Sex": "male", "Embarked":"S"})

encoder = LabelBinarizer()
train_data_categoric_encoded_sex = encoder.fit_transform(train_data_categoric_filled["Sex"])
train_data_categoric_encoded_pclass = encoder.fit_transform(train_data_categoric_filled["Pclass"])
train_data_categoric_encoded_embarked = encoder.fit_transform(train_data_categoric_filled["Embarked"])

#Normalize features
scaler = MinMaxScaler()
train_data_numeric_scaled = scaler.fit_transform(train_data_numeric_filled)

# Final Data
train_data_final = np.concatenate((train_data_numeric_scaled,
                                  train_data_categoric_encoded_pclass,
                                  train_data_categoric_encoded_sex,
                                  train_data_categoric_encoded_embarked), axis=1)

# Classifiers
randomForest_classifier = RandomForestClassifier(random_state=42)
randomForest_scores = cross_val_score(randomForest_classifier, train_data_final, train_data_y, cv=8)

SVM_classifier = SVC()
SVC_scores = cross_val_score(SVM_classifier, train_data_final, train_data_y, cv=8)
