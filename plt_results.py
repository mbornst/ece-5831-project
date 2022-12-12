#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:05:24 2022

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Data

labels = ["GTSRB", "Sunny/Couldy", "Rotation"]

Base_means = [87.36, 90.82, 91.28]
Base_stds = [02.85, 00.61, 01.36]

GaC_means = [81.12, 84.07, 83.73]
GaC_stds = [01.25, 00.73, 00.99]

HSV_means = [78.45, 86.03, 82.15]
HSV_stds = [01.94, 02.76, 02.20]

#%% Plots

x = np.arange(len(labels))
width = 0.3
width2 = 0.1

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, Base_means, width, yerr = Base_stds, label="None")
rects2 = ax.bar(x, GaC_means, width, yerr = GaC_stds, label="GaC")
rects3 = ax.bar(x + width, HSV_means, width, yerr = HSV_stds, label="HSV")


ax.set_ylabel('Accuracy %')
ax.set_xlabel('Training Set')
ax.set_title('Accuracies of each Neural Network')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

#fig.tight_layout()
plt.ylim([70, 95])
plt.yticks(np.arange(70, 96, 5))
plt.xlim([-0.5, 2.8])

plt.show()

#%% ANOVA2
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

dataset = np.concatenate([np.repeat("GTSRB", 10), 
                         np.repeat("weather", 10), 
                         np.repeat("rotate", 10)], axis=None)
dataset = np.concatenate([dataset, dataset, dataset])

image_aug = np.concatenate([np.repeat("Control", 30), 
                         np.repeat("GaC", 30), 
                         np.repeat("HSV", 30)], axis=None)

acc = [88.58, 90.57, 85.56, 89.03, 87.95, 89.57, 88.78, 85.04, 80.82, 87.69, #Base
       91.29, 91.60, 90.83, 90.64, 89.56, 91.23, 90.47, 90.25, 91.28, 91.09, #Base weather
       90.66, 92.66, 92.34, 89.46, 90.93, 91.99, 92.68, 92.45, 90.67, 88.91, #Base rotate
       81.62, 80.13, 82.41, 80.40, 80.97, 78.37, 81.35, 81.72, 81.54, 82.68, #GaC
       84.31, 85.00, 83.36, 84.34, 83.31, 85.00, 84.40, 84.54, 82.94, 83.52, #GaC weather
       83.22, 83.91, 82.63, 84.77, 83.57, 83.84, 81.71, 84.65, 84.57, 84.47, #GaC rotate
       79.36, 79.03, 80.71, 78.76, 78.49, 77.91, 76.16, 76.33, 81.78, 75.92, #HSV
       86.71, 85.65, 87.24, 81.66, 88.32, 81.43, 88.18, 89.79, 84.66, 86.61, #HSV weather
       80.56, 83.37, 79.78, 78.40, 80.95, 83.45, 85.00, 84.02, 84.37, 81.63 #HSV rotate
       ]


dataframe = pd.DataFrame({'Training_Dataset': dataset,
                          'Image_Aug': image_aug,
                          'Accuracy': acc
                          })

model = ols(
    'Accuracy ~ C(Training_Dataset) + C(Image_Aug) +\
        C(Training_Dataset):C(Image_Aug)',
        data=dataframe).fit()
results = sm.stats.anova_lm(model, type=2)
    
print(results)
