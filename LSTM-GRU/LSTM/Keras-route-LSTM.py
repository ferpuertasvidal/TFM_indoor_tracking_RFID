# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:50:50 2019

@author: Fer
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd


print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

nb_classes = 28
xls = pd.ExcelFile('./Dataset/ETIQUETAS-TRAIN.xlsx')
labels = xls.parse(xls.sheet_names[0])

X_train = []

# Load the dataset
dataset_train = pd.read_csv('./Dataset/dataset_routes_train.csv')

for index, row in dataset_train.iterrows():
   row.tolist()
   #print(index)
   fila = []
   for i in range(0,400):
       fila.append(list(row[i+1].strip('[').strip(']').split(',')))
   X_train.append(fila)
X_train = np.array(X_train)
X_train = (X_train.astype('float32')+99)/99
X_train = (X_train.reshape(140,20,4))

y_train = labels

y_train = np.asarray(y_train)

y_train = y_train.astype('uint8')
Y_train = y_train.astype('float32')
Y_train = Y_train.reshape((140,1))
Y_train = np_utils.to_categorical(y_train, nb_classes)

# TEST DATA 
xls = pd.ExcelFile('./Dataset/ETIQUETAS-TEST.xlsx')
labels_test = xls.parse(xls.sheet_names[0])

X_test = []
# Load the dataset
dataset_test = pd.read_csv('./Dataset/dataset_routes_test.csv')
for index, row in dataset_test.iterrows():
   row.tolist()
   #print(index)
   fila = []
   for i in range(0,400):
       fila.append(list(row[i+1].strip('[').strip(']').split(',')))
   X_test.append(fila)
X_test = np.array(X_test)
X_test = (X_test.astype('float32')+99)/99
X_test = (X_test.reshape(20,20,4))

y_test = labels_test
#y_test = y_test.tolist()
y_test = np.asarray(y_test)
y_test = y_test.astype('uint8')

Y_test = y_test.astype('float32')
Y_test = y_test.reshape((20,1))
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Number of hidden units to use:
nb_units = 512

model = Sequential()

# Recurrent layers supported: SimpleRNN, LSTM, GRU:
model.add(LSTM(nb_units,
                    input_shape=(20,4)))

model.add(Dense(units=nb_classes))
model.add(Activation('softmax'))
Adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam,
              metrics=['accuracy'])

print(model.summary())

## Visualizar el modelo gráficamente mediante SVG (Scalable Vector Graphics)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

epochs = 1000
history = model.fit(X_train, 
                    Y_train, 
                    epochs=epochs, 
                    batch_size=None,
                    verbose=2)


plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');

#Save Model
model.save('model_LSTM.h5')

scores = model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X_test, verbose=1)

pred = np.zeros((20,28))

for i in range(0,20):
    for j in range(0,28):
        pred[i][j] = np.round(np.argmax(predictions[i]))

predic = []
predic.append(pred[:,0])
