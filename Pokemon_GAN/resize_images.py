#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:55:52 2018

Author: Gorkem Polat
e-mail: polatgorkem@gmail.com

Written with Spyder IDE
"""

import cv2
import glob

size = (64, 64)

original_folder = "pokemons"
resized_folder = "pokemons_64"

filelist = glob.glob(original_folder+"/*")

for i in range(len(filelist)):
    original_img = cv2.imread(filelist[i])
    new_img = cv2.resize(original_img, dsize=size, interpolation=cv2.INTER_CUBIC)
    new_file_path = resized_folder+filelist[i][8:]
    cv2.imwrite(new_file_path, new_img)