#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-10-21 15:50:14
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
from torchvision import transforms
from datetime import datetime

from loss_function import FocalLoss, TverskyLoss, FocalTverskyLoss,BinaryDiceLoss
from torch.utils.data import DataLoader, ConcatDataset
from dataloader import ISLES2018Dataset
from sklearn.model_selection import KFold
from lion_pytorch import Lion
from torch.optim import AdamW

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax', 'CT_MTT']
dataset_m = ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=modalities)

# 设置测试集为整个数据集
testset = dataset_m

# 设置批处理大小为4
testloader = DataLoader(testset, batch_size=4)
binarydice_loss = BinaryDiceLoss()
model = TAU_module()
model1 = TransUnet()
model2 = Unet()
model3 = DeepTransUnet()
model4 = FCN8s()
#model.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\swinTAUNet\8.15(76.6\ba_swinTAUNet_300-fold-1.pth'))
model.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\swinTAUNet\8.15(76.6\ba_swinTAUNet_300-fold-1.pth'))
model1.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\Others\ba_TransUnet_300-fold-3.pth'))
model2.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\Others\ba_Unet_300-fold-0.pth'))
model3.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\Others\ba_DeepTransUnet_300-fold-3.pth'))
model4.load_state_dict(torch.load(r'D:\ISLES2018seg_BA\record\Others\ba_FCN8s_300-fold-3.pth'))
df = pd.DataFrame(columns=['Time', 'Img', 'Test loss', 'Test dice', 'Test_loss1', 'Test_dice1',
                           'Test_loss2', 'Test_dice2', 'Test_loss3', 'Test_dice3', 'Test_loss4', 'Test_dice4'])# 列名
df.to_csv(f'ba_swinTAUNet_v2/DTSUnet_img/test.csv',index=False)  
model.to(device)
model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

model.eval()
model1.eval()
model2.eval()
model3.eval()
model4.eval()
with torch.no_grad():
    Time = "%s"%datetime.now()
    for batch_idx, data in enumerate(testloader, 0):
        modalities_batch, OT_batch = data[0].to(device), data[1].to(device)
        pred_batch = model(modalities_batch)
        pred_batch1 = model1(modalities_batch)
        pred_batch2 = model2(modalities_batch)
        pred_batch3 = model3(modalities_batch)
        pred_batch4 = model4(modalities_batch)
        loss, dice = binarydice_loss(pred_batch, OT_batch)
        loss1, dice1 = binarydice_loss(pred_batch1, OT_batch)
        loss2, dice2 = binarydice_loss(pred_batch2, OT_batch)
        loss3, dice3 = binarydice_loss(pred_batch3, OT_batch)
        loss4, dice4 = binarydice_loss(pred_batch4, OT_batch)

        # 将数据移动到CPU并转换为numpy数组
        modalities_np = modalities_batch.cpu().numpy()
        OT_np = OT_batch.cpu().numpy()
        pred_np = pred_batch.cpu().numpy()
        pred_np1 = pred_batch1.cpu().numpy()
        pred_np2 = pred_batch2.cpu().numpy()
        pred_np3 = pred_batch3.cpu().numpy()
        pred_np4 = pred_batch4.cpu().numpy()

        Test_loss = loss
        Test_loss = "%f"%loss
        Test_dice = dice
        Test_dice = "%f"%dice

        Test_loss1 = loss1
        Test_loss1 = "%f"%loss1
        Test_dice1 = dice1
        Test_dice1 = "%f"%dice1

        Test_loss2 = loss2
        Test_loss2 = "%f"%loss2
        Test_dice2 = dice2
        Test_dice2 = "%f"%dice2

        Test_loss3 = loss3
        Test_loss3 = "%f"%loss3
        Test_dice3 = dice3
        Test_dice3 = "%f"%dice3

        Test_loss4 = loss4
        Test_loss4 = "%f"%loss4
        Test_dice4 = dice4
        Test_dice4 = "%f"%dice4

        list = [Time, batch_idx, Test_loss, Test_dice, Test_loss1, Test_dice1, Test_loss2, Test_dice2, Test_loss3, Test_dice3, Test_loss4, Test_dice4]
        file = pd.DataFrame([list])
        file.to_csv(f'ba_swinTAUNet_v2/DTSUnet_img/test.csv', mode='a', header=False, index=False)

        # 创建图形和轴
        fig, axes = plt.subplots(4, 11, figsize=(15, 10))

        for idx in range(modalities_np.shape[0]):
            # 获取每个模态的图像
            CT = modalities_np[idx, 0, :, :]
            CT_CBV = modalities_np[idx, 1, :, :]
            CT_CBF = modalities_np[idx, 2, :, :]
            CT_Tmax = modalities_np[idx, 3, :, :]
            CT_MTT = modalities_np[idx, 4, :, :]

            mask = OT_np[idx, 0, :, :]
            output = pred_np[idx, 0, :, :]
            output1 = pred_np1[idx, 0, :, :]
            output2 = pred_np2[idx, 0, :, :]
            output3 = pred_np3[idx, 0, :, :]
            output4 = pred_np4[idx, 0, :, :]

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
            axes[idx][6].set_title('swinTAUNet')

            axes[idx][7].imshow(output3, cmap='jet')
            axes[idx][7].axis('off')
            axes[idx][7].set_title('DeepTransUNet')

            axes[idx][8].imshow(output1, cmap='jet')
            axes[idx][8].axis('off')
            axes[idx][8].set_title('TransUNet')

            axes[idx][9].imshow(output2, cmap='jet')
            axes[idx][9].axis('off')
            axes[idx][9].set_title('UNet')

            axes[idx][10].imshow(output4, cmap='jet')
            axes[idx][10].axis('off')
            axes[idx][10].set_title('FCN8s')

        plt.tight_layout()
        # 保存图像到 img 文件夹
        plt.savefig(f'ba_swinTAUNet_v2/DTSUnet_img/result_batch_{batch_idx}.png',dpi=300)
        plt.close(fig)