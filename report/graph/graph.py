#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-29 14:19:32
@ LastEditors: Rindon
@ LastEditTime: 2024-04-29 14:52:35
@ Description: draw graph
'''
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = r'D:\ISLES2018seg_BA\report\graph' 

files = [f for f in os.listdir(directory) if f.endswith('.csv')]

plt.figure(figsize=(14, 10))

for file in files:
    filepath = os.path.join(directory, file)
    data = pd.read_csv(filepath)
    Dice = data.iloc[:, 4]
    Epoch = range(1, 301)
    plt.plot(Epoch, Dice, label=file[:-4])  # 去掉文件名的'.csv'后缀

plt.legend()
plt.title('Training dice',fontsize = 20)
plt.xlabel('Epoch',fontsize = 15)
plt.ylabel('Dice',fontsize = 15)
plt.xlim((0,300))
plt.ylim((0,0.72))
plt.grid()
plt.show()
