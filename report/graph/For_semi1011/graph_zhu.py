#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-29 14:19:32
@ LastEditors: Rindon
@ LastEditTime: 2024-10-06 13:48:00
@ Description: draw graph
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

categories = ['swinTAUNet_2', 'swinTAUNet', 'swinTXUNet','swinTWUNet','swinTUNet', 'DeepTransUnet', 'TransUnet', 'UNet', 'FCN8s']
values = [0.789482, 0.733083, 0.683273, 0.590775 ,0.622567, 0.635591, 0.587543, 0.622802, 0.621229]

plt.figure(figsize=(12, 5))
plt.bar(categories, values, color='skyblue')
bars = plt.bar(categories, values, color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')  # Center alignment

plt.title('Test result')
plt.xlabel('Models')
plt.ylabel('Mean dice')
plt.ylim((0,0.85))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05)) 
plt.show()