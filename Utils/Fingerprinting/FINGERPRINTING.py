# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:59:45 2019

@author: Fer
"""

import pandas as pd 
import numpy as np
from datetime import datetime

df = pd.read_csv('FINGERPRINTING.csv')

xls = pd.ExcelFile('./Etiquetas-ID.xlsx')
df_cell = xls.parse(xls.sheet_names[0])
  
t_init = df['Timestamp'][0]
t_final = np.max(df['Timestamp'])
interval1 = t_final - t_init

interval = 6000000
cell = 28
feature = 4

data = [[[0 for k in range(feature)] for j in range(int(t_init),int(t_final),interval)] for i in range(cell)]

for i in range(0,28):
    for j in range(0,len(data[0])):
        
        value = df.loc[(df['EPC'] == df_cell['EPCID'][i]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 1)].RSSI

        if len(value) != 0 :
           data[i][j][0] = np.round(value.mean())
        else:
           data[i][j][0] = -99 
           
        value = df.loc[(df['EPC'] == df_cell['EPCID'][i]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 2)].RSSI

        if len(value) != 0 :
           data[i][j][1] = np.round(value.mean())
        else:
           data[i][j][1] = -99 
   
        value = df.loc[(df['EPC'] == df_cell['EPCID'][i]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 3)].RSSI

        if len(value) != 0 :
           data[i][j][2] = np.round(value.mean())
        else:
           data[i][j][2] = -99 
     
        value = df.loc[(df['EPC'] == df_cell['EPCID'][i]) & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 4)].RSSI

        if len(value) != 0 :
           data[i][j][3] = np.round(value.mean())
        else:
           data[i][j][3] = -99     
        print(j)
    print(i)      

# Save Fingerprinting
df_final = pd.DataFrame(data)
df_final.to_csv('dataset_fingerprinting.csv')

df_read = pd.read_csv('dataset_fingerprinting.csv')             