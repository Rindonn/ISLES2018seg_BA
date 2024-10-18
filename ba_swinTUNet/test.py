#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-10-14 10:27:48
@ Description: 
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data


from torchvision import transforms
from datetime import datetime

from loss_function import FocalLoss, TverskyLoss, FocalTverskyLoss,BinaryDiceLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataloader import ISLES2018Dataset
from sklearn.model_selection import KFold
from lion_pytorch import Lion
from torch.optim import AdamW

from ba_swinTUNet import swinTUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)
#dataset_m = ISLES2018Dataset(r'D:\ba_ISLES2018\ISLES2018_Testing\TESTING', modalities=modalities)
training_part, testing_part = torch.utils.data.random_split(dataset_m, (420,82), generator=torch.Generator().manual_seed(41)) # random_split(数据集，长度)随机将一个数据集分割成给定长度的不重叠的新数据集
testset = testing_part
testloader = DataLoader(testset)

model = swinTUNet()
model.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\swinTAUNet\8.15(76.6\ba_swinTAUNet_300-fold-1.pth'))

model.to(device)

binarydice_loss = BinaryDiceLoss()

 ### 评估 ###
model.eval()

# 创建csv文件
df = pd.DataFrame(columns=['Time', 'Img', 'Test loss', 'Test dice'])# 列名
df.to_csv(f'swinTUNet_test.csv',index=False)  

count = 0
with torch.no_grad():

    Time = "%s"%datetime.now()

    for i, data in enumerate(testloader, 0):

        modalities, OT = data[0].to(device), data[1].to(device)
        
        pred = model(modalities)
        
        loss, dice = binarydice_loss(pred, OT)

        count += 1
        Img ="%d"%(count)

        print('Test Loss: {:.3f}'.format(loss))
        print('-'*30)
        print('Accuracy for test : %.3f %%' % (100.0 * (dice)))
        print('-'*30)

        Test_loss = loss
        Test_loss = "%f"%loss
        Test_dice = dice
        Test_dice = "%f"%dice


        #将数据保存为一维列表
        list = [Time, Img, Test_loss, Test_dice]
        file = pd.DataFrame([list])
        file.to_csv(f'swinTUNet_test.csv', mode='a', header=False, index=False)
    #计算统计数据
    file = pd.read_csv(f'D:\ISLES2018seg_BA\swinTUNet_test.csv',header=None)
    #file = file[:,2].astype(float)
    file.iloc[:, 2] = pd.to_numeric(file.iloc[:, 2], errors='coerce')  # 第3列
    file.iloc[:, 3] = pd.to_numeric(file.iloc[:, 3], errors='coerce')  # 第4列
    filtered_df1 = file[file.iloc[:, 2]<1]
    mean_value = filtered_df1.iloc[:,2].mean()        # 平均值
    max_value = filtered_df1.iloc[:,2].max()          # 最大值
    min_value = filtered_df1.iloc[:,2].min()          # 最小值
    variance_value = filtered_df1.iloc[:,2].var()     # 方差
    std_dev_value = filtered_df1.iloc[:,2].std()      # 标准差
    median_value = filtered_df1.iloc[:,2].median()    # 中位数
    loss_row = {'mean_loss': mean_value, 'max_loss': max_value, 'min_loss': min_value, 
        'variance_loss': variance_value,'std_dev_loss': std_dev_value,'median_loss': median_value} 
    print(loss_row)
    #print(file.dtypes)
    # 计算所需的统计量
    filtered_df = file[file.iloc[:, 3]>0]
    mean_value = filtered_df.iloc[:,3].mean()        # 平均值
    max_value = filtered_df.iloc[:,3].max()          # 最大值
    min_value = filtered_df.iloc[:,3].min()          # 最小值
    variance_value = filtered_df.iloc[:,3].var()     # 方差
    std_dev_value = filtered_df.iloc[:,3].std()      # 标准差
    median_value = filtered_df.iloc[:,3].median()    # 中位数
    dice_row = {'mean_dice': mean_value, 'max_dice': max_value, 'min_dice': min_value, 
        'variance_dice': variance_value,'std_dev_dice': std_dev_value,'median_dice': median_value} 
    print(dice_row)