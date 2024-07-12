#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-05-02 17:23:47
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
'''
from m_transunet import TransUnet
from m_fcn import FCN8s
from m_deeplab import DeepLabV3
from m_unet import Unet
from m_DeepTransUnet import DeepTransUnet
'''
from ba_swinTAAUNet import swinTAUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)
#dataset_m = ISLES2018Dataset(r'D:\ba_ISLES2018\ISLES2018_Testing\TESTING', modalities=modalities)
training_part, testing_part = torch.utils.data.random_split(dataset_m, (480,22), generator=torch.Generator().manual_seed(42)) # random_split(数据集，长度)随机将一个数据集分割成给定长度的不重叠的新数据集
testset = testing_part
testloader = DataLoader(testset)

model = swinTAUNet()
model.load_state_dict(torch.load(r'D:\ba_ISLES2018\record\swinTAUNet\4.21(68%)\ba_swinTAUNet_400-fold-0.pth'))

model.to(device)

binarydice_loss = BinaryDiceLoss()

 ### 评估 ###
model.eval()

# 创建csv文件
df = pd.DataFrame(columns=['Time', 'Img', 'Test loss', 'Test dice'])# 列名
df.to_csv(f'swinTAUNet_test.csv',index=False)  

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
        file.to_csv(f'swinTAUNet_test.csv', mode='a', header=False, index=False)
