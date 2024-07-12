#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-05-02 17:23:44
@ Description: 
'''
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt

'''
from m_transunet import TransUnet
from m_fcn import FCN8s
from m_deeplab import DeepLabV3
from m_unet import Unet
from m_DeepTransUnet import DeepTransUnet
'''
from ba_swinTAAUNet import swinTAUNet

from torch.utils.data import DataLoader, ConcatDataset
from dataloader import ISLES2018Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)
training_part, testing_part = torch.utils.data.random_split(dataset_m, (480,22), generator=torch.Generator().manual_seed(42)) # random_split(数据集，长度)随机将一个数据集分割成给定长度的不重叠的新数据集
testset = testing_part
testloader = DataLoader(testset)

model = swinTAUNet()
model.load_state_dict(torch.load(r'D:\ba_ISLES2018\record\swinTAUNet\4.21(68%)\ba_swinTAUNet_400-fold-0.pth'))
model.to(device)

model.eval()
count = 0
with torch.no_grad():
    
    fig, axes = plt.subplots(22, 7)
    for i, data in enumerate(testloader, 0):
        if count == 22:
            break
        modalities, OT = data[0].to(device), data[1].to(device)
        pred = model(modalities)
        
        list = []
        origin = torch.split(modalities,1,dim=1)
        for a in origin:
            list.append(a)
        CT = list[0].cpu().numpy()
        CT_CBV = list[1].cpu().numpy()
        CT_CBF = list[2].cpu().numpy()
        CT_Tmax = list[3].cpu().numpy()
        CT_MTT = list[4].cpu().numpy()
        
        masks = OT.cpu().numpy()
        outputs = pred.cpu().numpy()

        for ct in CT[0]:
            axes[i][0].axis('off')
            axes[i][0].imshow(ct, cmap = 'gray')
            for ct_cbv in CT_CBV[0]:
                axes[i][1].axis('off')
                axes[i][1].imshow(ct_cbv, cmap = 'gray')
                for ct_cbf in CT_CBF[0]:
                    axes[i][2].axis('off')
                    axes[i][2].imshow(ct_cbf, cmap = 'gray')
                    for ct_tmax in CT_Tmax[0]:
                        axes[i][3].axis('off')
                        axes[i][3].imshow(ct_tmax, cmap = 'gray')
                        for ct_mtt in CT_MTT[0]:
                            axes[i][4].axis('off')
                            axes[i][4].imshow(ct_mtt, cmap = 'gray')
                            for mask in masks[0]:
                                axes[i][5].axis('off')
                                axes[i][5].imshow(mask, cmap = 'gray')
                                for output in outputs[0]:
                                    axes[i][6].axis('off')
                                    axes[i][6].imshow(output, cmap = 'gray')
                                    count += 1
    plt.show()
    