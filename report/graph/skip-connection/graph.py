#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-29 14:19:32
@ LastEditors: Rindon
@ LastEditTime: 2024-05-24 09:31:16
@ Description: draw graph
'''
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

categories = ['allskip', 'skip1', 'skip2', 'skip3']
values = [0.733900, 0.688695, 0.690085, 0.674521]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color='skyblue')
bars = plt.bar(categories, values, color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')  # Center alignment

plt.title('Dice Values')
plt.xlabel('Categories')
plt.ylabel('Best Dice')
plt.ylim((0,0.8))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.05)) 
plt.show()