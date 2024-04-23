#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-19 09:28:06
@ LastEditors: Rindon
@ LastEditTime: 2024-04-23 10:07:03
@ Description: 
'''
from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nibabel.viewers import OrthoSlicer3D

class ISLES2018Dataset(Dataset):
    def __init__(self, folder, modalities=None):
        self.samples = []
        
        ### 读取数据

        for case_name in os.listdir(folder): #os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表
            case_path = os.path.join(folder, case_name) #..join(path1,path2,path3)->会输出路径"case_path = path1/path2/path3"
            case = {}
            
            for file_path in os.listdir(case_path):
                modality = re.search(r'XX.O.(\w+).\d+',file_path).group(1) # group(1)列出第一个括号匹配部分
                if modality != 'CT_4DPWI': #除了‘4DPWI’这个文件不要，其他的都要。change to modality in modalities but das slow
                    nii_path_name = os.path.join(case_path,file_path,file_path+'.nii') #把文件夹最里面的.nii文件的名字赋给nii_path_name
                    img = nib.load(nii_path_name)#用nib.load，把.nii文件的数据存储到img里面，.nii数据是一堆矩阵
                    #print(img.shape)# 输出了每张图的维度
                    #OrthoSlicer3D(img.dataobj).show()#可以查看3d图像
                    case[modality] = img

            #print("img is :",case['CT'].shape[2])

        ### 数据预处理
        
            for i in range(case['CT'].shape[2]): #case['CT'].shape[2]是CT图的第3维度
                arr = []
                for modality in modalities:
                    if modality != 'OT':
                        arr.append(torch.from_numpy(case[modality].get_fdata()[:,:,i]).float().unsqueeze(0))
                        ### torch.from_numpy用来将数组array转换为张量Tensor,
                        ### get_fdata返回浮点数,然后.float()将该tensor投射为float类型，
                        ### .unsqueeze对数据维度进行扩充。需要通过dim指定位置，给指定位置加上维数为1的维度,unsqueeze(0)是在第0维加上1。
                        #最后得到[1,256,256]的张量矩阵
                
                new = torch.cat(tuple(arr), dim=0)# tuple()暂且称tuple为限定长度的list, new的维度是[5,256,256]
                self.samples.append((new, torch.from_numpy(case['OT'].get_fdata()[:,:,i]).float().unsqueeze(0) ))
                # .unsqueeze对数据维度进行扩充。需要通过dim指定位置，给指定位置加上维数为1的维度,unsqueeze(0)是在第0维加上1。
                # 采用 unsqueeze(0) 来扩充一个假的 batch 维度，即从 3 维变为 4 维。   
                
                
        
        print("Cases loaded: ", str(len(self.samples)))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    ISLES2018Dataset(r'D:\dataset\ISLES_Dataset\ISLES2018_Training', modalities=['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT'])
