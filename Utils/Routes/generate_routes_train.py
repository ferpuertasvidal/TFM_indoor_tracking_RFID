# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:03:55 2019

@author: Fer
"""

import pandas as pd
import numpy as np

df = pd.read_csv('TRAIN.csv')

xls = pd.ExcelFile('./Etiquetas-ID.xlsx')
df_cell = xls.parse(xls.sheet_names[0])
print(df_cell.to_dict())


t_init = float(df.Timestamp[0])
t_final = float(max(df['Timestamp']))

interval = 1000000
cell = 1
feature = 4

data = [[[0 for k in range(feature)] for j in range(int(t_init),int(t_final-19000000),interval)] for i in range(cell)]


for k in range(0,28):
    for i in range(0,cell):
        for j in range(0,len(data[0])):
        
            value = df.loc[(df['EPC'] == df_cell['EPCID'][k]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 1)].RSSI
            if len(value) != 0 :
               data[i][j][0] = np.round(value.mean())
            else:
               data[i][j][0] = -99 
               
            value = df.loc[(df['EPC'] == df_cell['EPCID'][k]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 2)].RSSI
            if len(value) != 0 :
               data[i][j][1] = np.round(value.mean())
            else:
               data[i][j][1] = -99 
       
            value = df.loc[(df['EPC'] == df_cell['EPCID'][k]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 3)].RSSI
            if len(value) != 0 :
               data[i][j][2] = np.round(value.mean())
            else:
               data[i][j][2] = -99 
         
            value = df.loc[(df['EPC'] == df_cell['EPCID'][k]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 4)].RSSI
            if len(value) != 0 :
               data[i][j][3] = np.round(value.mean())
            else:
               data[i][j][3] = -99      
     
  
df_final = pd.DataFrame(data)
df_final.to_csv('dataset_routes_train.csv')

df_read = pd.read_csv('dataset_routes_train.csv')           
           