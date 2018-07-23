import numpy as np
import Provider
import tensorflow as tf

x= np.array([[0,1,2,3,4,5,6],[7,8,9,10,11,12,13]])
#x = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sess = tf.Session()
print(sess.run(tf.argmax(x,0)))

m = np.array(x[0:, :])
print(m)

m=np.array(x[0:2,:]).reshape(2,7)
print(m)

x = ([1,0,0,0,],[0,1,0,0],[0,1,0,0],[0,1,0,0]);
y = ([0.87,0.65,0,0,],[0,0.8,0.3,0],[0,0.6,1,0],[0,0,0,1]);
print(x)

prediction = tf.equal(tf.argmax(x,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

sess = tf.Session()
print(sess.run(prediction))


anArray=np.array([1,2,3,4]).reshape(1,4)

dizi = Provider.numToArray(anArray)

a= np.array([1,1,1,2,2,2])
print("dimension a: ")
print(a.shape)

b=np.array([[1,1,1],[1,1,1],[2,2,2],[1,1,1],[1,1,1],[1,2,3]])
print("dimension b: ")
print(b.shape)

c= np.matmul(a,b)

print(c)

print(Provider.numToArray(a[0]))
print(Provider.numToArray(1))
print(Provider.numToArray(2))
print(Provider.numToArray(3))
print(Provider.numToArray(4))
print(Provider.numToArray(5))
print(Provider.numToArray(6))
print(Provider.numToArray(7))
print(Provider.numToArray(8))
print(Provider.numToArray(9))

testData = np.genfromtxt('Data/train.csv', dtype='float',delimiter=',',skip_header=10000,max_rows=1000)
testOutputData = np.array(testData[:,0])

v = np.zeros((1000,10))

for i in range(1000):
    v[i,:] = Provider.numToArray(testOutputData[i])

print(v[0:9,:])