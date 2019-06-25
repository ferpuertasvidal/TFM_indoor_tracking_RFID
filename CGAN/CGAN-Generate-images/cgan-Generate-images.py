#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:00:36 2019

@author: Fer
"""
from keras.models import load_model
import numpy as np


from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd

import matplotlib.pyplot as plt

model = load_model('model_CGAN.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

num_imgs = 767
noise = np.random.normal(0, 1, (num_imgs, 20))

# Generate a half batch of new images
sampled_labels = np.random.randint(0, 20, 128).reshape(-1, 1)

pred = []
pred.append(model.predict([noise]))
pred = np.asarray(pred)
pred = np.squeeze(pred)

pred = np.reshape(pred, (28, num_imgs, 4))

# Save Data Augmentaation
np.save('dataset_cgan',pred)   

# Load Data Augmentation
arr = np.load('dataset_cgan.npy')
