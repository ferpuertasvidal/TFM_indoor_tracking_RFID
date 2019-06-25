# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:13:08 2019

@author: Fer
"""

from __future__ import print_function, division
import keras
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import numpy as np
#import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd

import matplotlib.pyplot as plt
#
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
#config = tf.ConfigProto( device_count = {'GPU': 0} ) 
#sess = tf.Session(config=config) 


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 4
        self.img_shape = (self.img_rows, self.img_cols)
        self.num_classes = 28
        self.latent_dim = 20

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(20,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes,self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])

        img = model(model_input)

        # Save CGAN model
        model.save('model_CGAN.h5')

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes,np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        X_train = []
        # Load the dataset
        dataset_fp = pd.read_csv('./Dataset/dataset_fingerprinting.csv')
        for i in range(0,28):
            for index in range(0,len(dataset_fp.iloc[0])-1):
               #row.tolist()
               #print(row)
               arr = dataset_fp.iloc[i]
               fila = []
               #for i in range(0,len(arr)-1):
               fila.append(list(arr[index+1].strip('[').strip(']').split(',')))
               X_train.append(fila)
               
        X_train = np.array(X_train)
        X_train = (X_train.astype('float32')+99)/99
        X_train = (X_train.reshape(len(dataset_fp.iloc[0])-1,28,4))
            
        xls = pd.ExcelFile('./Dataset/Etiquetas-ID.xlsx')
        df_cell = xls.parse(xls.sheet_names[0])

        y_train = df_cell['ID_Etiqueta']
        y_train = y_train.tolist()
        y_train = np.asarray(y_train)
        
        y_train = y_train.reshape(-1, 1)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            jdx = np.random.randint(0, y_train.shape[0], half_batch)
            imgs = X_train[idx]
            labels = y_train[jdx]
            #imgs = imgs.reshape(imgs.shape[0],1,imgs.shape[3])

            noise = np.random.normal(0, 1, (half_batch, 20))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])
            print(gen_imgs.shape)
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 20))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # digits
            sampled_labels = np.random.randint(0, 20, batch_size).reshape(-1, 1)
          
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0],100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,X_train, sampled_labels,noise)

    def sample_images(self, epoch, X_reals, sampled_labels1, noise):

         gen_imgs = self.generator.predict([noise, sampled_labels1])

         for j in range(0,4):
           R =  X_reals[sampled_labels1[-1],:,j].reshape(4,7)
           plt.imshow(R,cmap='coolwarm')

           plt.savefig("images_real/img_ant%s_epoch%s.png" % ( str(j),str(epoch) ))
           Z =  gen_imgs[-1,:,j].reshape(7,4)
           plt.imshow(Z,cmap='coolwarm')

           plt.savefig("images/img_ant%s_epoch%s.png" % ( str(j),str(epoch) ))

#%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=30000, batch_size=1290, sample_interval=100)