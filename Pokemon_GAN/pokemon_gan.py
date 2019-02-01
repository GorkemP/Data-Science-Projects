#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 18:48:52 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

tf.reset_default_graph()

folderPath = "pokemons"
IMG_WIDTH, IMG_HEIGHT, CHANNEL = 128, 128, 3

num_iteration = 50000
learning_step_D = 0.0002
learning_step_G = 0.0002
batch_size = 32

n_discriminatorOutput = 1
n_noise = 100
 
filelist = glob.glob(folderPath+"/*")
X_data = np.zeros((len(filelist), IMG_WIDTH, IMG_HEIGHT, CHANNEL))

# Read all images into a numpy array
for i in range(len(filelist)):
    X_data[i,:,:,:] = mpimg.imread(filelist[i])/255

random_pokemon = X_data[np.random.randint(0, X_data.shape[0]),:,:,:]
plt.imshow(random_pokemon)
plt.show()

# Define placeholders, since we are using batch normalization, we have to define
# an extra parameter that indicates training and testing
X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNEL])
z = tf.placeholder(tf.float32, [None, n_noise])
is_training = tf.placeholder(dtype=tf.bool)

# Construct the graphs
# Discriminator
def discriminator(input_image, is_training=is_training, reuse=None):
    c1, c1_f, c2, c2_f, c3, c3_f, n_hidden = 5, 64, 5, 64, 5, 64, 128
    with tf.variable_scope("discriminator", reuse=reuse):
            
        x = tf.layers.conv2d(input_image, kernel_size=[c1, c1], filters=c1_f, strides=[2, 2], padding="SAME")        
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)
        
        x = tf.layers.conv2d(input_image, kernel_size=[c2, c2], filters=c2_f, strides=[2, 2], padding="SAME")        
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)

        x = tf.layers.conv2d(input_image, kernel_size=[c3, c3], filters=c3_f, strides=[2, 2], padding="SAME")        
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)        

        x = tf.layers.dense(x, units=n_hidden, activation=tf.nn.elu)
        x = tf.layers.dense(x, units=n_discriminatorOutput, activation=tf.nn.sigmoid)
        
        return x
       
# Generator
def generator(input_noise, is_training=is_training):
    n_hidden, n_hidden_dim, c1, c1_f, c2, c2_f, c3, c3_f, c4, c4_f = 8, 5, 5, 64, 5, 64, 5, 64, 5, 3   
    with tf.variable_scope("generator", reuse=None):
        
        x = tf.layers.dense(input_noise, units=n_hidden*n_hidden*n_hidden_dim, activation=tf.nn.elu)
        x = tf.reshape(x, shape=[-1, n_hidden, n_hidden, n_hidden_dim])
        
        x = tf.layers.conv2d_transpose(x, c1_f, kernel_size=[c1, c1], strides=[2, 2], padding="SAME")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)        
        
        x = tf.layers.conv2d_transpose(x, c2_f, kernel_size=[c2, c2], strides=[2, 2], padding="SAME")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)        

        x = tf.layers.conv2d_transpose(x, c3_f, kernel_size=[c3, c3], strides=[2, 2], padding="SAME")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.elu(x)        

        x = tf.layers.conv2d_transpose(x, c4_f, kernel_size=[c4, c4], strides=[2, 2], padding="SAME")
        x = tf.nn.sigmoid(x)        

        return x   #128*128*3
    
generated_samples = generator(z, is_training)
D_real = discriminator(X, is_training)
D_fake = discriminator(generated_samples, is_training, reuse=True)

discriminator_variables = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
generator_variables     = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

loss_discriminator_D = -tf.math.log(D_real)
loss_discriminator_G = -tf.math.log(1-D_fake)
loss_discriminator = tf.reduce_mean(loss_discriminator_D+loss_discriminator_G)

loss_generator = tf.reduce_mean(-tf.math.log(D_fake))

# Optimizers

optimizer_D = tf.train.AdamOptimizer(learning_rate=learning_step_D).minimize(loss_discriminator, var_list=discriminator_variables)
optimizer_G = tf.train.AdamOptimizer(learning_rate=learning_step_G).minimize(loss_generator, var_list=generator_variables)

session = tf.Session()
session.run(tf.global_variables_initializer())

log_step = 2
static_noise_size = 5

loss_log_discriminator = np.array(np.zeros(int(num_iteration/log_step)))
loss_log_generator = np.array(np.zeros(int(num_iteration/log_step)))

satic_noise = np.random.normal(size=(static_noise_size, n_noise))               

def train(num_iterations):
    counter = 0
    k_D = 1
    k_G = 1
    for i in range(num_iterations):
        
        # Train Discriminator
        for i_D in range(k_D):   
            randomIDs = np.random.permutation(X_data.shape[0])
            x_batch = X_data[randomIDs[0:batch_size], :, :, :]
            noise_batch = np.random.normal(size=(batch_size, n_noise))

            feed_dict_D = {X: x_batch, z:noise_batch, is_training:True}
            session.run(optimizer_D, feed_dict=feed_dict_D)
        
        # Train Generator
        for i_G in range(k_G):
            noise_batch = np.random.normal(size=(batch_size, n_noise))
            feed_dict_G = {z: noise_batch, is_training:True}
            session.run(optimizer_G, feed_dict=feed_dict_G)
        
        
        if (i%log_step == 0):                
            loss_log_discriminator[counter] = session.run(loss_discriminator, feed_dict=feed_dict_D)                   
            
            feed_dict_G = {z: satic_noise, is_training:False}
            loss_log_generator[counter] = session.run(loss_generator, feed_dict=feed_dict_G)                        
            
            if i%4==0:
                print("iteration: "+str(i) + " | discriminator loss: "+str(loss_log_discriminator[counter])+ " | generator loss: "+str(loss_log_generator[counter]))
                
                if i%4000==0:                    
                    generated_images = session.run(generated_samples, feed_dict=feed_dict_G)
                    #reshaped_generated_images = np.reshape(generated_images, newshape=(static_noise_size,IMG_HEIGHT, IMG_WIDTH, CHANNEL))
                    reshaped_generated_images = generated_images
                    
                    f, axes = plt.subplots(1, static_noise_size, figsize=(static_noise_size, 2))
                    for img_index in range(static_noise_size):
                        axes[img_index].grid(False)
                        axes[img_index].imshow(reshaped_generated_images[img_index])
                    plt.show()
            
            counter = counter + 1
            
startTime = time.time()
train(num_iterations=num_iteration)
endTime = time.time()            

print("Training duration: " +str(endTime-startTime)+ " seconds")
plt.plot(loss_log_discriminator, 'b')
plt.plot(loss_log_generator, 'r')
plt.show()

feed_dict_G = {z: satic_noise}
generated_images = session.run(generated_samples, feed_dict=feed_dict_G)
#reshaped_generated_images = np.reshape(generated_images, newshape=(static_noise_size,IMG_HEIGHT, IMG_WIDTH, CHANNEL))
reshaped_generated_images = generated_images

f, axes = plt.subplots(1, static_noise_size, figsize=(10,2))
for img_index in range(static_noise_size):
    axes[img_index].grid(False)
    axes[img_index].imshow(reshaped_generated_images[img_index])
plt.show()

session.close()








