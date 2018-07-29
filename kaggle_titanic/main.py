# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:59:14 2018

@author: Gorkem Polat
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from future_encoders import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from provider import DataFrameSelector, MostFrequentImputer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_titanic_data(fileName):
    return pd.read_csv('datasets/'+fileName)

trainData = load_titanic_data('train.csv')
testData = load_titanic_data('test.csv')

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", Imputer(strategy="median"))
        ])
    
num_pipeline.fit_transform(trainData)    

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
    
cat_pipeline.fit_transform(trainData)    

preprocessed_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
        ])

train_data_preprocessed = preprocessed_pipeline.fit_transform(trainData)
train_data_labels = trainData["Survived"]

svm_classifier = SVC()
svm_classifier.fit(train_data_preprocessed, train_data_labels)

test_data = preprocessed_pipeline.transform(testData)
y_pred = svm_classifier.predict(test_data)

svm_scores = cross_val_score(svm_classifier, train_data_preprocessed, train_data_labels, cv=8)

randomForest_classifier = RandomForestClassifier(random_state=42)
randomForest_scores = cross_val_score(randomForest_classifier, train_data_preprocessed, train_data_labels, cv=8)

plt.figure(figsize=(10,8))
plt.plot([1]*8, svm_scores, ".")
plt.plot([2]*8, randomForest_scores, ".")
plt.boxplot([svm_scores, randomForest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Models", fontsize=14)
plt.show()








