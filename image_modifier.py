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
cloudy_filt = np.full((dsize[0],dsize[1],3),(149,170,245), np.uint8)
sunny_filt = np.full((dsize[0],dsize[1],3),(232,237,126), np.uint8)

def change_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

#%%

for i in range(42 + 1):
    if i < 10:
        dir_path = "0000" + str(i)
    else:
        dir_path = "000" + str(i)
        
    obj = os.scandir(dir_path)
    print("Files in {}".format(dir_path))
    for entry in obj:
        if entry.is_file() and (".ppm" in entry.name) and not(("cloudy.ppm" in entry.name) or ("sunny.ppm" in entry.name)):
            print(entry.name)
            image_path = dir_path + "/" + entry.name
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize)
            
            image_cloudy = cv2.addWeighted(image, 0.8, cloudy_filt, 0.2,0)
            image_cloudy = change_brightness(image_cloudy, -30)
            image_sunny = cv2.addWeighted(image, 0.85, sunny_filt, 0.15,0)
            image_sunny = change_brightness(image_sunny, 50)
    
            sun_name = image_path[:-4] + "_sunny.ppm"
            cloud_name = image_path[:-4] + "_cloudy.ppm"
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_sunny = cv2.cvtColor(image_sunny, cv2.COLOR_BGR2RGB)
            image_cloudy = cv2.cvtColor(image_cloudy, cv2.COLOR_BGR2RGB)
            
            cv2.imwrite(image_path, image)
            cv2.imwrite(sun_name, image_sunny)
            cv2.imwrite(cloud_name, image_cloudy)
    
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.title("Original")
            # plt.axis('off')
            # plt.imshow(image)
            
            # plt.subplot(1,3,2)
            # plt.title("Cloudy")
            # plt.axis('off')
            # plt.imshow(image_cloudy)
            
            # plt.subplot(1,3,3)
            # plt.title("Sunny")
            # plt.axis('off')
            # plt.imshow(image_sunny)
            # plt.show()



#%% Purge files

# print("Files in {}".format(dir_path))
# for entry in obj:
#     if entry.is_file() and ( ("cloudy.ppm" in entry.name) or ("sunny.ppm" in entry.name) ):
#         print(entry.name)
#         image_path = dir_path + "/" + entry.name
#         os.remove(image_path)
