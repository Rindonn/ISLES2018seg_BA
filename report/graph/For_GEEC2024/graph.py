#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-29 14:19:32
@ LastEditors: Rindon
@ LastEditTime: 2024-05-26 14:55:23
@ Description: draw graph
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

directory = r'D:\ISLES2018seg_BA\report\graph\For_GEEC2024' 

files = [f for f in os.listdir(directory) if f.endswith('.csv')]

plt.figure(figsize=(14, 10))

for file in files:
    filepath = os.path.join(directory, file)
    data = pd.read_csv(filepath)
    Accuracy = data.iloc[:, 4]
    Loss = data.iloc[:, 2]
    Epoch = range(1, 301)
    #plt.plot(Epoch, Loss, label=file[:-4])  # 去掉文件名的'.csv'后缀
    plt.plot(Epoch, Accuracy, label=file[:-4])  # 去掉文件名的'.csv'后缀

plt.legend()
#plt.title('Dice',fontsize = 20)
#plt.title('Training Loss',fontsize = 20)
plt.xlabel('Epoch',fontsize = 15)
#plt.ylabel('Loss',fontsize = 15)
plt.ylabel('Accuracy',fontsize = 15)
plt.xlim((0,300))
plt.ylim((0,0.75))
#plt.ylim((0.2,1))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.02)) 
plt.grid()
plt.show()
