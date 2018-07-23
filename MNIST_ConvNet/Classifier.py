import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import Provider

TRAIN_SIZE = 20000
TEST_SIZE = 1000
HIDDEN_LAYER_SIZE = 200
EPOCH = 1

# Read from csv fie, skip the first line get first TRAIN_SIZE rows
trainData = np.array(genfromtxt('Data/train.csv', dtype='float', delimiter=',', skip_header=1, max_rows=TRAIN_SIZE))

# First column is output values
trainOutputData = np.array(trainData[:, 0])

# Rest is pixel values
trainInputData = np.array(trainData[:, 1:])

# Normalize input
trainInputData = np.multiply(trainInputData, 1.0/255.0)

x = tf.placeholder(tf.float32, [None, 784], name='X-Input')

y_ = tf.placeholder(tf.float32, [None, 10], name='Y-Output')

W_conv1 = Provider.weight_variable([5,5,1,32])
b_conv1= Provider.bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

with tf.name_scope("Layer-1") as scope:
    h_conv1 = tf.nn.relu(Provider.conv2d(x_image, W_conv1)+b_conv1)
    h_pool1 = Provider.max_pool_2x2(h_conv1)

W_conv2 = Provider.weight_variable([5,5,32,64])
b_conv2 = Provider.bias_variable([64])

with tf.name_scope("Layer-2") as scope:
    h_conv2 = tf.nn.relu(Provider.conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = Provider.max_pool_2x2(h_conv2)

W_fc1 = Provider.weight_variable([7*7*64, 1024])
b_fc1 = Provider.bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = Provider.weight_variable([1024,10])
b_fc2 = Provider.bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

with tf.name_scope("Training") as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv,y_))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

tf.summary.scalar("Cross Entropy",cross_entropy)
tf.summary.histogram("Weights 1", W_fc1)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='Accuracy')
tf.summary.scalar("Accuracy",accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Sumamry Writer
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Logs', sess.graph)


for i in range(TRAIN_SIZE):
    xData = np.array(trainInputData[i,:]).reshape(1,784)
    yData = Provider.numToArray(trainOutputData[i]).reshape(1,10)

    sess.run(train_step, feed_dict={x: xData, y_: yData, keep_prob:0.5})
    if ((i%100) == 0):
        summary = sess.run(merged, feed_dict={x: xData, y_: yData, keep_prob: 0.5})
        writer.add_summary(summary, i)
        print("Epoch: "+str(i))
        print(sess.run(cross_entropy, feed_dict={x:xData, y_:yData, keep_prob:1.0}))

#TEST
testData = np.array(
    genfromtxt('Data/train.csv', dtype='float', delimiter=',', skip_header=(1 + TRAIN_SIZE), max_rows=TEST_SIZE))

testOutputData = np.array(testData[:, 0]).reshape(1, TEST_SIZE)

testInputData = np.array(testData[:, 1:]).reshape(TEST_SIZE, 784)

testInputData = np.multiply(testInputData,1.0/255.0)

print("Accuracy:")
print(sess.run(accuracy, feed_dict={x:testInputData, y_:Provider.numArrayToBinaryArray(testOutputData),keep_prob:1.0}))

