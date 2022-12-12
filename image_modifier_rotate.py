#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:36:58 2022

@author: michael
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


dsize = (100, 100)

def rotate(image, value):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), value, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w,h))
    return rotated_img

#%%

for i in range(2, 2 + 1):
#for i in range(99, 100):
    if i < 10:
        dir_path = "0000" + str(i)
    else:
        dir_path = "000" + str(i)
        
    obj = os.scandir(dir_path)
    print("Files in {}".format(dir_path))
    for entry in obj:
        if entry.is_file() and (".ppm" in entry.name) and not(("CW.ppm" in entry.name) or ("CCW.ppm" in entry.name)) and not(("cloudy.ppm" in entry.name) or ("sunny.ppm" in entry.name)):
            print(entry.name)
            image_path = dir_path + "/" + entry.name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize)
            
            image_CW = rotate(image, -15)
            image_CCW = rotate(image, 15)
    
            CW_name = image_path[:-4] + "_CW.ppm"
            CCW_name = image_path[:-4] + "_CCW.ppm"
            
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image_CW = cv2.cvtColor(image_CW, cv2.COLOR_BGR2RGB)
            #image_CCW = cv2.cvtColor(image_CCW, cv2.COLOR_BGR2RGB)
            
            #cv2.imwrite(CW_name, image_CW)
            #cv2.imwrite(CCW_name, image_CCW)
    
            plt.figure()
            plt.subplot(1,3,1)
            plt.title("Original")
            plt.axis('off')
            plt.imshow(image)
            
            plt.subplot(1,3,2)
            plt.title("Clockwise")
            plt.axis('off')
            plt.imshow(image_CW)
            
            plt.subplot(1,3,3)
            plt.title("Counter-Clockwise")
            plt.axis('off')
            plt.imshow(image_CCW)
            plt.show()



#%% Purge files

for i in range(42 + 1):
#for i in range(99, 100):
    if i < 10:
        dir_path = "0000" + str(i)
    else:
        dir_path = "000" + str(i)

    print("Files in {}".format(dir_path))
    obj = os.scandir(dir_path)
    for entry in obj:
        if entry.is_file() and ( ("CW.ppm" in entry.name) or ("CCW.ppm" in entry.name) ):
            print(entry.name)
            image_path = dir_path + "/" + entry.name
            os.remove(image_path)
