#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:18:15 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from provider import getSequencedData, fetchTrainingData

tf.reset_default_graph()

n_steps  = 5
n_inputs = 4
n_neurons= 50
n_outputs= 1
n_layers = 2

learningRate = 0.0001
batchSize = 200

# Prepare data
data = pd.read_csv('datasets/BTC-USD.csv')

#TEST: BEGIN
data = data[0:105]
#TEST: END

X_data, y_data = getSequencedData(data, n_inputs, n_steps)    
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# Build TF model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None, 1])

lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons) for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, state = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

state_h = state[-1][1]
logit = tf.layers.dense(state_h, n_outputs, activation=tf.nn.sigmoid)

loss = tf.losses.log_loss(y, logit)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
training_operation = optimizer.minimize(loss)

prediction = tf.cast(logit >= 0.5, tf.int32)
correct_predictions = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(500):
    X_batch, y_batch = fetchTrainingData(X_train, y_train, batchSize)    
    X_batch = X_batch.reshape((-1, n_steps, n_inputs))
    sess.run(training_operation, feed_dict={X: X_batch, y: y_batch})
    
#    #TEST: BEGIN
#    logit_D = sess.run(logit, feed_dict={X: X_batch, y: y_batch})
#    loss_D = sess.run(loss, feed_dict={X: X_batch, y: y_batch})
#    prediction_D = sess.run(prediction, feed_dict={X: X_batch, y: y_batch})
#    #TEST: END
    accuracy_train = accuracy.eval(session=sess, feed_dict={X: X_batch, y: y_batch})
    
    X_test = X_test.reshape((-1, n_steps, n_inputs))
    accuracy_test = accuracy.eval(session=sess, feed_dict={X: X_test, y: y_test})
    print(str(i)+": Train Accuracy: "+ str(accuracy_train)+" : Test Accuracy: "+str(accuracy_test))
    
















