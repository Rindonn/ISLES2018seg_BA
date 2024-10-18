#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-10-18 11:25:54
@ Description: 所有预测图全部输出
'''
import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from m_transunet import TransUnet
from m_fcn import FCN8s
from m_deeplab import DeepLabV3
from m_unet import Unet
from m_DeepTransUnet import DeepTransUnet

from ba_TAU_module import TAU_module

from torch.utils.data import DataLoader, ConcatDataset
from dataloader import ISLES2018Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax', 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)

# 设置测试集为整个数据集
testset = dataset_m

# 设置批处理大小为4
testloader = DataLoader(testset, batch_size=4)

model = TAU_module()
# model = TransUnet()
model.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\swinTAUNet\8.15(76.6\ba_swinTAUNet_300-fold-1.pth'))
# model.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\Others\ba_TransUnet_300-fold-3.pth'))

model.to(device)

model.eval()
with torch.no_grad():

    for batch_idx, data in enumerate(testloader, 0):
        modalities_batch, OT_batch = data[0].to(device), data[1].to(device)
        pred_batch = model(modalities_batch)

        # 将数据移动到CPU并转换为numpy数组
        modalities_np = modalities_batch.cpu().numpy()
        OT_np = OT_batch.cpu().numpy()
        pred_np = pred_batch.cpu().numpy()

        # 创建图形和轴
        fig, axes = plt.subplots(4, 7, figsize=(15, 10))

        for idx in range(modalities_np.shape[0]):
            # 获取每个模态的图像
            CT = modalities_np[idx, 0, :, :]
            CT_CBV = modalities_np[idx, 1, :, :]
            CT_CBF = modalities_np[idx, 2, :, :]
            CT_Tmax = modalities_np[idx, 3, :, :]
            CT_MTT = modalities_np[idx, 4, :, :]

            mask = OT_np[idx, 0, :, :]
            output = pred_np[idx, 0, :, :]

            # 绘制图像
            axes[idx][0].imshow(CT, cmap='gray')
            axes[idx][0].axis('off')
            axes[idx][0].set_title('CT')

            axes[idx][1].imshow(CT_CBV, cmap='gray')
            axes[idx][1].axis('off')
            axes[idx][1].set_title('CT_CBV')

            axes[idx][2].imshow(CT_CBF, cmap='gray')
            axes[idx][2].axis('off')
            axes[idx][2].set_title('CT_CBF')

            axes[idx][3].imshow(CT_Tmax, cmap='gray')
            axes[idx][3].axis('off')
            axes[idx][3].set_title('CT_Tmax')

            axes[idx][4].imshow(CT_MTT, cmap='gray')
            axes[idx][4].axis('off')
            axes[idx][4].set_title('CT_MTT')

            axes[idx][5].imshow(mask, cmap='gray')
            axes[idx][5].axis('off')
            axes[idx][5].set_title('Mask')

            axes[idx][6].imshow(output, cmap='jet')
            axes[idx][6].axis('off')
            axes[idx][6].set_title('Output')

        plt.tight_layout()
        # 保存图像到 img 文件夹
        plt.savefig(f'ba_swinTAUNet_v2/img/result_batch_{batch_idx}.png',dpi=300)
        plt.close(fig)