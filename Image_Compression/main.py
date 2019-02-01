#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:59:33 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from kMeansClassifier import k_means_classifier

c=16
k=8

image = imread("images/image_256_256.jpg")

# ------------------------------  Encoding  -----------------------------------
# NOTE: image is assumed to have equal size in width and height
strideNumber = int(image.shape[0] / c)
numberOfDataPoints = (strideNumber**2)
dimensionOfData = c**2

dataPoints_R = np.zeros((numberOfDataPoints, dimensionOfData)) 
dataPoints_G = np.zeros((numberOfDataPoints, dimensionOfData)) 
dataPoints_B = np.zeros((numberOfDataPoints, dimensionOfData)) 

# For red channel
counter=0
for i in range(strideNumber):
    for j in range(strideNumber):
        dataPoints_R[counter] = np.reshape(image[i*c:(i+1)*c, j*c:(j+1)*c, 0], c**2)
        counter=counter+1

# For green channel
counter=0
for i in range(strideNumber):
    for j in range(strideNumber):
        dataPoints_G[counter] = np.reshape(image[i*c:(i+1)*c, j*c:(j+1)*c, 1], c**2)
        counter=counter+1

# For blue channel
counter=0
for i in range(strideNumber):
    for j in range(strideNumber):
        dataPoints_B[counter] = np.reshape(image[i*c:(i+1)*c, j*c:(j+1)*c, 2], c**2)
        counter=counter+1        
        
m_R, labels_R = k_means_classifier(k, dataPoints_R)       
m_G, labels_G = k_means_classifier(k, dataPoints_G)
m_B, labels_B = k_means_classifier(k, dataPoints_B)

# -----------------------------------------------------------------------------

# ----------------------------  Transmission  ---------------------------------
transmittedIndexes_R = labels_R 
transmittedVectors_R = m_R.astype(int)

transmittedIndexes_G = labels_G 
transmittedVectors_G = m_G.astype(int)

transmittedIndexes_B = labels_B 
transmittedVectors_B = m_B.astype(int)
# -----------------------------------------------------------------------------

# ------------------------------  Decoding  -----------------------------------

# Find the reconstructed image size
size = int(transmittedIndexes_R.shape[0]**(0.5))*c

reconstructedImage_R = np.zeros((size, size))
reconstructedImage_G = np.zeros((size, size))
reconstructedImage_B = np.zeros((size, size))

reconstructedStrideNumber = int(transmittedIndexes_R.shape[0]**(0.5))

# Reconstrcut red, green, and blue channels
for i in range(transmittedIndexes_R.shape[0]):
    columnNumber = (i % reconstructedStrideNumber)*c
    rowNumber = int(i / reconstructedStrideNumber)*c
    
    vector = transmittedVectors_R[transmittedIndexes_R[i]]
    reconstructedImage_R[rowNumber:rowNumber+c, columnNumber:columnNumber+c] = \
    (np.reshape(vector, (c, c)))
    
    vector = transmittedVectors_G[transmittedIndexes_G[i]]
    reconstructedImage_G[rowNumber:rowNumber+c, columnNumber:columnNumber+c] = \
    (np.reshape(vector, (c, c)))

    vector = transmittedVectors_B[transmittedIndexes_B[i]]
    reconstructedImage_B[rowNumber:rowNumber+c, columnNumber:columnNumber+c] = \
    (np.reshape(vector, (c, c)))    

# Form the final image
reconstructedImage = np.zeros((size, size, 3))
reconstructedImage[:,:,0] = reconstructedImage_R
reconstructedImage[:,:,1] = reconstructedImage_G
reconstructedImage[:,:,2] = reconstructedImage_B
reconstructedImage = reconstructedImage.astype(int)

# -----------------------------------------------------------------------------
# Reconstruction error
reconstructedPoints_R = np.zeros((transmittedIndexes_R.shape[0], c*c))
reconstructedPoints_G = np.zeros((transmittedIndexes_G.shape[0], c*c))
reconstructedPoints_B = np.zeros((transmittedIndexes_B.shape[0], c*c))
for i in range(transmittedIndexes_R.shape[0]):    
    reconstructedPoints_R[i] = transmittedVectors_R[transmittedIndexes_R[i]]
    reconstructedPoints_G[i] = transmittedVectors_G[transmittedIndexes_G[i]]
    reconstructedPoints_B[i] = transmittedVectors_B[transmittedIndexes_B[i]]
    
reconstructionError = np.sum(np.linalg.norm((dataPoints_R-reconstructedPoints_R), axis=1)**2) + \
                      np.sum(np.linalg.norm((dataPoints_G-reconstructedPoints_G), axis=1)**2) + \
                      np.sum(np.linalg.norm((dataPoints_B-reconstructedPoints_B), axis=1)**2)

print("Reconstruction error: " + str(reconstructionError))

# Compression rate
# It is assumed that each pixel is represented by 8 bis
originalData = image.shape[0]*image.shape[0]*8*3                      
compressedData = (((image.shape[0]/c)**2)*np.log2(k) + k*c*c*8)*3
compression = int(originalData/compressedData)

print("Compression rate: "+str(compression))

# Show the original and reconstructed image
f, axes = plt.subplots(1, 2)
f.suptitle("Compression result for k= "+str(k)+" c= "+str(c))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[1].imshow(reconstructedImage)
axes[1].set_title("Reconstructed Image")

    
    
    