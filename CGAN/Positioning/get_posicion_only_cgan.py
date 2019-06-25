#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:18:10 2019

@author: Fer
"""

import pandas as pd
import numpy as np


xls = pd.ExcelFile('./Dataset/Etiquetas-ID.xlsx')
df_cell = xls.parse(xls.sheet_names[0])

x = df_cell['x']
y =df_cell['y']
pos = y =df_cell['x,y']

fingerprint = np.load('./Dataset/dataset_cgan.npy')


X_test = []
# Load the dataset
dataset_pos = pd.read_csv('./Dataset/dataset_position.csv')

# Number of points to predict
t=62

# cell = 0 (true_value = 13),1 (true_value = 10),2 (true_value = 22)
celda=1
arr = np.zeros((1,len(dataset_pos.iloc[0])))
arr = dataset_pos.iloc[celda]


#%%
X_test = []
for i in range(0,62):
   #row.tolist()
   #print(index)
   fila = []
   for i in range(0,61):
       fila.append(list(arr[i+1].strip('[').strip(']').split(',')))
   X_test.append(fila)
   
X_test = np.array(X_test)
X_test = (X_test.astype('float32')+99)/99

## Obtener los pesos
w = []
K = 1/2*(np.exp(-abs(X_test)))
h = 1
w = np.asarray(w)
w = np.zeros((t,28,4))


for j in range(0,61):
    for i in range(0,28):        
        print((fingerprint[i][:]))
        print(i)
        RSS_vec = np.mean(fingerprint[i][:],axis=0)
        print(RSS_vec.shape)
        #w[j][i]= (K[j]*((X_test[j]-RSS_vec)/h))       
        w[j][i]= 1/2*(np.exp(-np.abs(((X_test[0][j]-RSS_vec)/h))))       


w.shape    
c = np.sum(w)        

l = []
l = np.asarray(l)
l = np.zeros((t,28,4))
norm = (-93+99)/99
for k in range(0,len(X_test[0])):
    for i in range(0,28):
        RSS_vec = np.mean(fingerprint[i][:],axis=0)
        print(RSS_vec)
        if  np.mean(X_test[0][k]) >= norm:
            l[k][i] =  1/2*(np.exp(-np.abs(((X_test[0][k]-RSS_vec)/h))))/c    
        else:
            continue

ll = np.zeros((t,1))
for k in range(0,len(X_test[0])):
    print(l[k].shape)
    A= np.sum(l[k], axis=1)
    print(np.argmax(A))
    ll[k] = np.argmax(A)