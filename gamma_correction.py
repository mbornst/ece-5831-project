#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:11:31 2022

@author: michael
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

dark_path = "GTSRB_Final_Training_Images/00099/too_dark2.ppm" 
bright_path = "GTSRB_Final_Training_Images/00099/too_bright3.ppm" 
normal_path = "GTSRB_Final_Training_Images/00099/normal.ppm" 

image_d = cv2.imread(dark_path)
image_b = cv2.imread(bright_path)
image_n = cv2.imread(normal_path)


def GammaCorrection(img):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s = (s * 1.4)
    s[s >= 255] = 255
    s = s.astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid*255)/math.log(mean) * 1.8
    
    
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        img_corrected = cv2.LUT(img2, lookUpTable)

    img_corrected = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB)


    return img_corrected


#%%

img_corrected = GammaCorrection(image_d)

image_d = cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.axis('off')
plt.imshow(image_d)

plt.subplot(1,2,2)
plt.title("Gamma/Color Corrected")
plt.axis('off')
plt.imshow(img_corrected)

#%%
img_corrected = GammaCorrection(image_b)

image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.axis('off')
plt.imshow(image_b)

plt.subplot(1,2,2)
plt.title("Gamma/Color Corrected")
plt.axis('off')
plt.imshow(img_corrected)

#%%
img_corrected = GammaCorrection(image_n)

image_n = cv2.cvtColor(image_n, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(1,2,1)
plt.title("Original")
plt.axis('off')
plt.imshow(image_n)

plt.subplot(1,2,2)
plt.title("Gamma/Color Corrected")
plt.axis('off')
plt.imshow(img_corrected)