#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:06:55 2022

@author: michael
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt

#%%

files = ["traffic_signs_NN", "traffic_signs_SunnyCloudy_NN", "traffic_signs_NN_rotate",
         "traffic_signs_HSV_NN", "traffic_signs_HSV_SunnyCloudy_NN", "traffic_signs_HSV_Rotate_NN",
         "traffic_signs_GaC_NN", "traffic_signs_GaC_SunnyCloudy_NN", "traffic_signs_GaC_rotate_NN"
         ]
acc = []
val_acc = []
loss_values = []
val_loss_values = []

for names in files:

    with open(names, 'rb') as f:
                history = pickle.load(f)
                
                
    history_dict = history.history
    acc.append( history_dict["accuracy"])
    val_acc.append( history_dict["val_accuracy"])
    loss_values.append( history_dict["loss"])
    val_loss_values.append( history_dict["val_loss"])

acc = np.array(acc)
val_acc = np.array(val_acc)
loss = np.array(loss_values)
val_loss = np.array(val_loss_values)

acc *= 100
val_acc *= 100

#%%

epochs = range(1,11)

fig = plt.figure(1)
plt.subplots_adjust(top=1.6, bottom=0, hspace=0.4)
#plt.suptitle("Training and validation accuracy")


plt.subplot(3,1,1)
plt.title("Training and validation accuracy\n\nNo Image Preprocessing")

plt.plot(epochs, acc[0], "b", label="GTSRB Training acc")
plt.plot(epochs, val_acc[0], "b--", label="GTSRB Validation acc")
plt.plot(epochs, acc[1], "k", label="Weather Training acc")
plt.plot(epochs, val_acc[1], "k--", label="Weather Validation acc")
plt.plot(epochs, acc[2], "g", label="Rotate Training acc")
plt.plot(epochs, val_acc[2], "g--", label="Rotate Validation acc")
plt.legend()
plt.xticks(range(1,11))
plt.ylim([20, 100])

plt.subplot(3,1,2)
plt.ylabel("Accuracy %")
plt.title("HSV")
plt.plot(epochs, acc[3], "b", label="GTSRB Training acc")
plt.plot(epochs, val_acc[3], "b--", label="GTSRB Validation acc")
plt.plot(epochs, acc[4], "k", label="Weather Training acc")
plt.plot(epochs, val_acc[4], "k--", label="Weather Validation acc")
plt.plot(epochs, acc[5], "g", label="Rotate Training acc")
plt.plot(epochs, val_acc[5], "g--", label="Rotate Validation acc")
#plt.legend()
plt.xticks(range(1,11))
plt.ylim([20, 100])
    
plt.subplot(3,1,3)
plt.title("Gamma and Color")
plt.plot(epochs, acc[6], "b", label="GTSRB Training acc")
plt.plot(epochs, val_acc[6], "b--", label="GTSRB Validation acc")
plt.plot(epochs, acc[7], "k", label="Weather Training acc")
plt.plot(epochs, val_acc[7], "k--", label="Weather Validation acc")
plt.plot(epochs, acc[8], "g", label="Rotate Training acc")
plt.plot(epochs, val_acc[8], "g--", label="Rotate Validation acc")
#plt.legend()
    

plt.xlabel("Epochs")
plt.xticks(range(1,11))
plt.ylim([20, 100])

plt.show()


#%%

fig = plt.figure(2)
plt.subplots_adjust(top=1.6, bottom=0, hspace=0.4)
#plt.suptitle("Training and validation accuracy")


plt.subplot(3,1,1)
plt.title("Training and validation loss\n\nNo Image Preprocessing")

plt.plot(epochs, loss[0], "b", label="GTSRB Training loss")
plt.plot(epochs, val_loss[0], "b--", label="GTSRB Validation loss")
plt.plot(epochs, loss[1], "k", label="Weather Training loss")
plt.plot(epochs, val_loss[1], "k--", label="Weather Validation loss")
plt.plot(epochs, loss[2], "g", label="Rotate Training loss")
plt.plot(epochs, val_loss[2], "g--", label="Rotate Validation loss")
plt.legend()
plt.xticks(range(1,11))
plt.ylim([0, 9])

plt.subplot(3,1,2)
plt.ylabel("Loss ")
plt.title("HSV")
plt.plot(epochs, loss[3], "b", label="GTSRB Training loss")
plt.plot(epochs, val_loss[3], "b--", label="GTSRB Validation loss")
plt.plot(epochs, loss[4], "k", label="Weather Training loss")
plt.plot(epochs, val_loss[4], "k--", label="Weather Validation loss")
plt.plot(epochs, loss[5], "g", label="Rotate Training loss")
plt.plot(epochs, val_loss[5], "g--", label="Rotate Validation loss")
#plt.legend()
plt.xticks(range(1,11))
plt.ylim([0, 9])
    
plt.subplot(3,1,3)
plt.title("Gamma and Color")
plt.plot(epochs, loss[6], "b", label="GTSRB Training loss")
plt.plot(epochs, val_loss[6], "b--", label="GTSRB Validation loss")
plt.plot(epochs, loss[7], "k", label="Weather Training loss")
plt.plot(epochs, val_loss[7], "k--", label="Weather Validation loss")
plt.plot(epochs, loss[8], "g", label="Rotate Training loss")
plt.plot(epochs, val_loss[8], "g--", label="Rotate Validation loss")
#plt.legend()
    

plt.xlabel("Epochs")
plt.xticks(range(1,11))
plt.ylim([0, 9])

plt.show()