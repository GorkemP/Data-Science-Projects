#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:24:36 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import numpy as np

def k_means_classifier(clusterNumber, dataPoints):
    stoppingThreshold = 1
    
    clusterDimension = dataPoints.shape[1]
    clusters = np.random.randint(0, 255, (clusterNumber, clusterDimension))
    updatedClusters = np.array(clusters)
    indexes = np.zeros((dataPoints.shape[0]))
    
    iterationNumber=0
    while (iterationNumber<300):
        # 1 - Assign each data point to a cluster
        distances= np.zeros((clusterNumber, dataPoints.shape[0]))
        for k in range(clusterNumber):
            distances[k, :] = np.linalg.norm((dataPoints-clusters[k, :]), axis=1)
        
        indexes = np.argmin(distances, axis=0)
        
        # 2 - Update cluster centers
        for k in range(clusterNumber):
            updatedClusters[k, :] = np.average(dataPoints[indexes==k], axis=0)
        
        # Check update change size
        if (np.linalg.norm(updatedClusters-clusters) < stoppingThreshold):
            break
        else:
            clusters=np.array(updatedClusters)
        
        iterationNumber = iterationNumber + 1
    
    #print("K-means completed in "+str(iterationNumber)+" iteration")
    return clusters, indexes
    