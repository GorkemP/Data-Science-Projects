import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import Provider

# Read from csv fie, skip the first line get first 1000 rows
trainData = genfromtxt('Data/train.csv', dtype='float',delimiter=',',skip_header=1,max_rows=1000)

#First column is output values
trainOutputData =np.array(trainData[:,0])

trainInputData = np.array(trainData[:,1:784])
# Make binary image
trainInputData[trainInputData<101] = 0
trainInputData[trainInputData>100] = 1

# TEST
#print(trainInputData[0,131])

x = tf.placeholder(tf.float32,[784, 1])
W = tf.Variable(tf.zeros([10, 784]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder(tf.float32,[1,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    xData = trainInputData[i,0:783]
    yData = Provider.numToArray(trainOutputData[i])
    sess.run(train_step, feed_dict={x:xData, y_:yData})


testData = genfromtxt('Data/train.csv', dtype='float',delimiter=',',skip_header=10000,max_rows=1000)

testOutputData = np.array(testData[:,0])

testInputData = np.array(testData[:,1:784])
# Make binary image
testInputData[testInputData<101] = 0
testInputData[testInputData>100] = 1

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(sess.run(accuracy,feed_dict={x:testInputData, y_:Provider.numArrayToBinaryArray(testOutputData)}))