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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from provider import getSequencedData, fetchTrainingData
from datetime import datetime
now = datetime.now()

tf.reset_default_graph()

n_steps  = 20
n_inputs = 4
n_neurons= 50
n_outputs= 1
n_layers = 2

learningRate = 0.0001
batchSize = 200

# Prepare the data
data = pd.read_csv('datasets/BTC-USD.csv')

##TEST: BEGIN
#data = data[0:105]
##TEST: END

X_data, y_data = getSequencedData(data, n_inputs, n_steps)    
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

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
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
training_operation = optimizer.minimize(loss)

prediction = tf.cast(logit >= 0.5, tf.int32)
correct_predictions = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
tf.summary.scalar("Accuracy", accuracy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Summary Writer
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./TF_Logs/'+now.strftime("%Y%m%d-%H:%M:%S")+"_"+str(learningRate)+"/", sess.graph)

for i in range(500):
    X_batch, y_batch = fetchTrainingData(X_train, y_train, batchSize)    
    X_batch = X_batch.reshape((-1, n_steps, n_inputs))
    sess.run(training_operation, feed_dict={X: X_batch, y: y_batch})
        
    if (i%10 == 0):
        accuracy_train = accuracy.eval(session=sess, feed_dict={X: X_batch, y: y_batch})
                
        accuracy_test = accuracy.eval(session=sess, feed_dict={X: X_test, y: y_test})
        print(str(i)+": Train Accuracy: "+ str(accuracy_train)+" : Test Accuracy: "+str(accuracy_test))
        summary = sess.run(merged, feed_dict={X: X_batch, y: y_batch})
        writer.add_summary(summary, i)
















