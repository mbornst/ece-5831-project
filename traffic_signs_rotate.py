#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:30:32 2022

@author: michael
"""

#%%
# Import packages
import os
import csv
import numpy as np
import cv2
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)


DSIZE = (50, 50)
BATCH_SIZE = 256


#%%

class TrafficSigns:
    def __init__(self):
        self.x_train, self.y_train  = self._load_training()
        self.x_test, self.y_test = self._load_testing()
        self.model = self._model()
        
        print("Done initializing")

    def _load_training(self):
        x_train = []
        y_train = []
        for i in range(42 + 1):
            if i < 10:
                dir_path = "GTSRB_Final_Training_Images/" + "0000" + str(i)
            else:
                dir_path = "GTSRB_Final_Training_Images/" +"000" + str(i)
                
            obj = os.scandir(dir_path)
            print("Files in {}".format(dir_path))
            for entry in obj:
                if entry.is_file() and (".ppm" in entry.name) and not(("cloudy.ppm" in entry.name) or ("sunny.ppm" in entry.name)):
                    #print(entry.name)
                    image_path = dir_path + "/" + entry.name
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, DSIZE)
                    x_train.append(image)
                    y_train.append(i)

        x_train = np.array(x_train).astype(np.float32)
        x_train /= 255.0
        y_train = np.array(y_train)
        y_train = to_categorical(y_train)
        return x_train, y_train
    
    def _load_testing(self):
        x_test = []
        y_test = []

        dir_path = "GTSRB_Final_Test_Images/Images/"
                
        list_dir = os.listdir(dir_path)
        list_dir = sorted(list_dir)

        print("Files in {}".format(dir_path))
        for entry in list_dir:
            if (".ppm" in entry):
                #print(entry.name)
                image_path = dir_path + "/" + entry
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, DSIZE)
                x_test.append(image)

        print("Opening csv...")
        with open("GTSRB_Final_Test_Images/GT-final_test_reduced.csv") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONE) 
            for row in reader: # each row is a list
                y_test.append(row)
        
        x_test = np.array(x_test).astype(np.float32)
        x_test /= 255.0
        y_test = np.array(y_test)
        y_test = to_categorical(y_test)
        return x_test, y_test

    def _model(self):
        model = Sequential()
        
        model.add(Conv2D(8, (5,5), padding="same", input_shape=self.x_train.shape[1:]))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        
        model.add(Dense(43))
        model.add(Activation("softmax"))

        model.compile(optimizer="rmsprop",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"])
        return model
    
    def train(self, epochs=10):
        if self.model is None: 
            print('[INFO] model is not defined.')
            return
        
        x_train_split, x_test_val, y_train_split, y_test_val = train_test_split(self.x_train, self.y_train, train_size=0.8)
        
        classTotals = y_train_split.sum(axis=0)
        self.classWeight = dict()
        for i in range(0, len(classTotals)):
            self.classWeight[i] = classTotals.max() / classTotals[i]
        
        history = self.model.fit(x_train_split, y_train_split,
                                 epochs = epochs, validation_split = 0,
                                 batch_size = BATCH_SIZE, validation_data=(x_test_val, y_test_val),
                                 class_weight=self.classWeight)
        return history
    
    
    def plot_loss(self, history):
        # Plotting the training and validation loss
        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.figure(1)
        plt.plot(epochs, loss_values, "r", label="Training loss")
        plt.plot(epochs, val_loss_values, "r--", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


    def plot_accuracy(self, history):
        history_dict = history.history
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        epochs = range(1, len(acc) + 1)
        plt.figure(2)
        plt.plot(epochs, acc, "b", label="Training acc")
        plt.plot(epochs, val_acc, "b--", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test)
        print(f'[INFO] Test loss: {score[0]}')
        print(f'[INFO] Test accuracy: {score[1]}')
        return score
    
#%%
import time

losses = []
accuracies = []
times = []

for i in range (0,4):
    start_time = time.time()
    
    ts = TrafficSigns()
    x_train = ts.x_train
    y_train = ts.y_train
    
    x_test = ts.x_test
    y_test = ts.y_test
    
    # train NN
    history = ts.train(epochs=10)
    
    # plot accuracy and loss
    ts.plot_accuracy(history)
    ts.plot_loss(history)
    
    score = ts.evaluate()
    
    run_time = time.time() - start_time
    losses.append(score[0])
    accuracies.append(score[1])
    times.append(run_time)
    
    print("--- %s seconds ---" % (time.time() - start_time))

#%%
with open("traffic_signs_NN_rotate", 'wb') as f:
    pickle.dump(history, f, -1)
    
with open("traffic_signs_NN_rotate", 'rb') as f:
            history = pickle.load(f)
            
            
#%%          
dir_path = "GTSRB_Final_Test_Images/Images/"
        
n = 0
list_dir = os.listdir(dir_path)

list_dir = sorted(list_dir)

print("Files in {}".format(dir_path))
for entry in list_dir:
    if n > 10:
        break
    if  (".ppm" in entry):
        plt.figure()
        print(entry)
        image_path = dir_path + "/" + entry
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()
        n = n + 1
        