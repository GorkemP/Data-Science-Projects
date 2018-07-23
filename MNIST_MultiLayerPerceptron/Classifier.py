import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import Provider

TRAIN_SIZE = 40000
TEST_SIZE = 1000
HIDDEN_LAYER_SIZE = 200
EPOCH = 1

# Read from csv fie, skip the first line get first TRAIN_SIZE rows
trainData = np.array(genfromtxt('Data/train.csv', dtype='float', delimiter=',', skip_header=1, max_rows=TRAIN_SIZE))

# First column is output values
trainOutputData = np.array(trainData[:, 0])

# Rest is pixel values
trainInputData = np.array(trainData[:, 1:])

# Make binary image
# trainInputData[trainInputData < 101] = 0
# trainInputData[trainInputData > 100] = 1
trainInputData = np.multiply(trainInputData, 1.0/255.0)


x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.truncated_normal([784, HIDDEN_LAYER_SIZE], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE],stddev=0.1))

W2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER_SIZE, 10], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

h1 = tf.matmul(x, W1) + b1

# y = tf.nn.softmax(tf.matmul(h1, W2)+b2)
y = tf.nn.softmax(tf.matmul(h1, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)),reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#cross_entropy = tf.reduce_sum(tf.pow(0.5*(y_ - y),2))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

pW1, pW2 = sess.run([W1, W2]);

for epoch in range(EPOCH):
    print('Epoch: ' + str(epoch))
    for i in range(TRAIN_SIZE):
        xData = np.array(trainInputData[i, :]).reshape(1, 784)
        yData = Provider.numToArray(trainOutputData[i]).reshape(1, 10)

        pW1, pW2 = sess.run([W1, W2]);
        ph1 = sess.run(h1, feed_dict={x: xData})
        ptrain, predicted, cost, ph1, pw2 = sess.run([train_step, y, cross_entropy, h1, W2],
                                                     feed_dict={x: xData, y_: yData})
        #   print('Error:'+str(sess.run(cross_entropy, feed_dict={x:xData})))
        #  print(sess.run(h1,feed_dict={x:xData}))
        #   weight1 = sess.run(W1, feed_dict={x: xData})
        #   weight2 = sess.run(W2, feed_dict={x: xData})
        #  print(sess.run(y,feed_dict={x:xData}))
        if i%500 == 0:
            print("EPOCH: "+str(epoch)+" *** TRAIN: "+str(i))
            print("Predicted Output: "+str(predicted))
            print("Real Output: " + str(yData))
            print("Cost: "+str(cost))
           #input("Enter to Continue...")

testData = np.array(
    genfromtxt('Data/train.csv', dtype='float', delimiter=',', skip_header=(1 + TRAIN_SIZE), max_rows=TEST_SIZE))

testOutputData = np.array(testData[:, 0]).reshape(1, TEST_SIZE)

testInputData = np.array(testData[:, 1:]).reshape(TEST_SIZE, 784)
# Make binary image
#testInputData[testInputData < 101] = 0
#testInputData[testInputData > 100] = 1

testInputData = np.multiply(testInputData,1.0/255.0)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testInputData, y_: Provider.numArrayToBinaryArray(testOutputData)}))
