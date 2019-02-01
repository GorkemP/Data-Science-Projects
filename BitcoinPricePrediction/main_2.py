#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:18:15 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

"""
Tensorboard command: tensorboard --logdir="TF_Logs"
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from provider import getSequencedDataWithColumns, fetchTrainingData, convertWeeklyToDaily
from datetime import datetime
import time
now = datetime.now()

tf.reset_default_graph()

n_iteration = 3000
n_steps  = 100
n_inputs = 5
n_neurons= 5
n_outputs= 1
n_layers = 2

learningRate = 0.001

# Prepare the data
data_historicalPrice = pd.read_csv('datasets/BTC-USD.csv')
data_webTrends = pd.read_csv('datasets/multiTimeline.csv')

data_webTrends = convertWeeklyToDaily(data_webTrends)

data = np.column_stack((data_historicalPrice.iloc[:,1].values, data_historicalPrice.iloc[:,2].values, \
                 data_historicalPrice.iloc[:,3].values, data_historicalPrice.iloc[:,4].values, data_webTrends))

##TEST: In order to test whether model is learning
#data = data[0:105]
##TEST

X_data, y_data = getSequencedDataWithColumns(data, n_inputs, n_steps, 0, 5, 3)    
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

batchSize = len(X_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test = X_test.reshape((-1, n_steps, n_inputs))

# Build TF model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None, 1])

lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons) for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, state = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

state_h = state[-1][1]
logit = tf.layers.dense(state_h, n_outputs, activation=tf.nn.sigmoid)

loss = tf.losses.log_loss(y, logit)
tf.summary.scalar("Loss", loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
training_operation = optimizer.minimize(loss)

prediction = tf.cast(logit >= 0.5, tf.int32)
correct_predictions = tf.equal(prediction, y)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
training_summary = tf.summary.scalar("train_accuracy", accuracy)
test_summary = tf.summary.scalar("Test_Accuracy", accuracy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Summary Writer
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./TF_Logs/'+now.strftime("%Y_%m_%d-%H:%M:%S")+"_"+str(learningRate)+"/", sess.graph)

def trainModel(n_iteration):
    for i in range(n_iteration):
        X_batch, y_batch = fetchTrainingData(X_train, y_train, batchSize)    
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))
        sess.run(training_operation, feed_dict={X: X_batch, y: y_batch})
            
        if (i%20 == 0):
            accuracy_train, train_summ = sess.run([accuracy, training_summary], feed_dict={X: X_batch, y: y_batch})
            writer.add_summary(train_summ, i)      
            
            accuracy_test, test_summ = sess.run([accuracy, test_summary], feed_dict={X: X_test, y: y_test})
            writer.add_summary(test_summ, i)
            
            print(str(i)+": Train Accuracy: "+ str(accuracy_train)+" : Test Accuracy: "+str(accuracy_test))
       
startTime = time.time() 
trainModel(n_iteration)
endTime = time.time()

print("Training duration: " +str((endTime-startTime)//60)+ " minutes")

predictions_test = sess.run(prediction, feed_dict={X:X_test})

confusion_matrix = confusion_matrix(y_test, predictions_test)




