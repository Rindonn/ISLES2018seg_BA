#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-29 14:19:32
@ LastEditors: Rindon
@ LastEditTime: 2024-06-07 08:32:14
@ Description: draw graph
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

categories = ['allskip', 'skip1', 'skip2', 'skip3', 'skip4', 'skip5', 'skip6', 'skip7']
values = [0.733900, 0.688695, 0.690085, 0.674521, 0.61668, 0.613893, 0.569196, 0.488571]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color='skyblue')
bars = plt.bar(categories, values, color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')  # Center alignment

plt.title('Dice')
plt.xlabel('Categories')
plt.ylabel('Best Dice')
plt.ylim((0,0.8))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05)) 
plt.show()