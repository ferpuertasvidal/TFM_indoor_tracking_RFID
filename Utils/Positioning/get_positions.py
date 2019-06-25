# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 20:59:45 2019

@author: Fer
"""

import pandas as pd 
import numpy as np

#%%

df = pd.read_csv('Positions.csv')

xls = pd.ExcelFile('./Etiquetas-ID.xlsx')
df_cell = xls.parse(xls.sheet_names[0])

t_init = df['Timestamp'][0]
t_final = np.max(df['Timestamp'])
interval1 = t_final - t_init

interval = 2000000
cell = 3
feature = 4

data = [[[0 for k in range(feature)] for j in range(int(t_init),int(t_final),interval)] for i in range(cell)]

# Cell 13
i=0
for j in range(0,len(data[0])):
        print(df['EPC'])
        value = df.loc[(df['EPC'] == 'E2000019510802091320B910') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 1)].RSSI

        if len(value) != 0 :
           data[i][j][0] = np.round(value.mean())
        else:
           data[i][j][0] = -99 
           
        value = df.loc[(df['EPC'] == 'E2000019510802091320B910') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 2)].RSSI
        if len(value) != 0 :
           data[i][j][1] = np.round(value.mean())
        else:
           data[i][j][1] = -99 
   
        value = df.loc[(df['EPC'] == 'E2000019510802091320B910') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 3)].RSSI
        if len(value) != 0 :
           data[i][j][2] = np.round(value.mean())
        else:
           data[i][j][2] = -99 
     
        value = df.loc[(df['EPC'] == 'E2000019510802091320B910') & (j*1000000+ t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 4)].RSSI
        if len(value) != 0 :
           data[i][j][3] = np.round(value.mean())
        else:
           data[i][j][3] = -99      
# Cell 10
i=1
for j in range(0,len(data[0])):
        
        value = df.loc[(df['EPC'] == 'E2000019510802051320B90E') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 1)].RSSI
        if len(value) != 0 :
           data[i][j][0] = np.round(value.mean())
        else:
           data[i][j][0] = -99 
           
        value = df.loc[(df['EPC'] == 'E2000019510802051320B90E') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 2)].RSSI
        if len(value) != 0 :
           data[i][j][1] = np.round(value.mean())
        else:
           data[i][j][1] = -99 
   
        value = df.loc[(df['EPC'] == 'E2000019510802051320B90E') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 3)].RSSI
        if len(value) != 0 :
           data[i][j][2] = np.round(value.mean())
        else:
           data[i][j][2] = -99 
     
        value = df.loc[(df['EPC'] == 'E2000019510802051320B90E') & (j*1000000+ t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 4)].RSSI
        if len(value) != 0 :
           data[i][j][3] = np.round(value.mean())
        else:
           data[i][j][3] = -99      
# Cell 22
i=2
for j in range(0,len(data[0])):
        
        value = df.loc[(df['EPC'] == 'E2000019510802011320B0CC') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<= t_final) & (df['Antenna'] == 1)].RSSI
        if len(value) != 0 :
           data[i][j][0] = np.round(value.mean())
        else:
           data[i][j][0] = -99 
           
        value = df.loc[(df['EPC'] == 'E2000019510802011320B0CC') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 2)].RSSI
        if len(value) != 0 :
           data[i][j][1] = np.round(value.mean())
        else:
           data[i][j][1] = -99 
   
        value = df.loc[(df['EPC'] == 'E2000019510802011320B0CC') & (j*1000000 + t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 3)].RSSI
        if len(value) != 0 :
           data[i][j][2] = np.round(value.mean())
        else:
           data[i][j][2] = -99 
     
        value = df.loc[(df['EPC'] == 'E2000019510802011320B0CC') & (j*1000000+ t_init <= df['Timestamp']) & (df['Timestamp'] <= j*1000000 + t_init + interval) & (df['Timestamp']<=t_final) & (df['Antenna'] == 4)].RSSI
        if len(value) != 0 :
           data[i][j][3] = np.round(value.mean())
        else:
           data[i][j][3] = -99      

#Save position data
df_final = pd.DataFrame(data)
df_final.to_csv('dataset_position.csv')

df_read = pd.read_csv('dataset_position.csv') 