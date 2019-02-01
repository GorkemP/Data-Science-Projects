#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:33:02 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt

image = imread("images/image_256_256.jpg")

imgplot_R = plt.imshow(image[:,:,0])

