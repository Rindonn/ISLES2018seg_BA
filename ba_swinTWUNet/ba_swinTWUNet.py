#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-18 14:55:42
@ LastEditors: Rindon
@ LastEditTime: 2024-04-29 09:09:32
@ Description: unet With swinT
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import swinT
from odconv import ODConv2d

class swinTWUNet(nn.Module):
    def __init__(self):
        super(swinTWUNet, self).__init__()

        # DOWN BLOCK 1 #5*256*256
        self.conv1 = ODConv2d(5, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        #gelu
        
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        #gelu
        self.conv3 = ODConv2d(32, 32, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        #gelu
        self.swint1 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint2 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #64*128*128

        #DOWN BLOCK 2 #64*128*128
        self.conv4 = ODConv2d(64, 64, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        #gelu
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(64)
        #gelu
        self.conv6 = ODConv2d(64, 64, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(64)
        #gelu
        self.swint3 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=4, 
                    window_size=16, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint4 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=4, 
                    window_size=16, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #128*64*64

        #DOWN BLOCK 3 #128*64*64
        self.conv7 = ODConv2d(128, 128, 3, padding=1)
        self.norm7 = nn.BatchNorm2d(128)
        #gelu
        self.conv8 = nn.Conv2d(128,128, 3, padding=1)
        self.norm8 = nn.BatchNorm2d(128)
        #gelu
        self.conv9 = ODConv2d(128,128, 3, padding=1)
        self.norm9 = nn.BatchNorm2d(128)
        #gelu
        self.swint5 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=8, 
                    window_size=16, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint6 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=8, 
                    window_size=16, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #256*32*32
        
        #############
        
        #Bottleneck #
        self.convB1 = ODConv2d(256, 256, 3, padding=1)
        self.normB1 = nn.BatchNorm2d(256)
        #gelu
        self.convB2 = ODConv2d(256, 256, 3, padding=1)
        self.normB2 = nn.BatchNorm2d(256)
        #gelu
        self.swintB1 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=8, 
                    window_size=32, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swintB2 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=16, 
                    window_size=32, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        
        #############
        
        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.norm10 = nn.BatchNorm2d(128)
        #gelu
        self.conv11 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm11 = nn.BatchNorm2d(128)
        #gelu
        
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv12 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.norm12 = nn.BatchNorm2d(64)
        #gelu
        self.conv13 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.norm13 = nn.BatchNorm2d(64)
        #gelu
        
        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv14 = nn.Conv2d(64,32,3,padding=1, bias=False)
        self.norm14 = nn.BatchNorm2d(32)
        #gelu
        self.conv15 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.norm15 = nn.BatchNorm2d(32)
        #gelu
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        #### ENCODER ####
        
        #BLOCK 1
        x = self.conv1(x)
        #x = F.gelu(x)
        x = F.gelu(self.norm1(x))
        x = self.swint1(x)
        #x = self.conv2(x)
        #x = F.gelu(self.norm2(x))
        #x = F.gelu(x)
        x = self.conv3(x)
        x = F.gelu(self.norm3(x))
        #x = self.swint1(x)
        enc1 = self.swint1(x)
        x = self.swint2(x)
        
        #BLOCK 2
        x = self.conv4(x)
        x = F.gelu(self.norm4(x))
        x = self.swint3(x)
        #x = F.gelu(x)
        #x = self.conv5(x)
        #x = F.gelu(self.norm5(x))
        #x = F.gelu(x)
        x = self.conv6(x)
        x = F.gelu(self.norm6(x))
        enc2 = self.swint3(x)
        #x = self.swint3(x)
        x = self.swint4(x)
        
        #BLOCK 3
        x = self.conv7(x)
        x = F.gelu(self.norm7(x))
        x = self.swint5(x)
        #x = F.gelu(x)
        #x = self.conv8(x)
        #x = F.gelu(x)
       # x = F.gelu(self.norm8(x))
        x = self.conv9(x)
        x = F.gelu(self.norm9(x))
        #x = self.swint5(x)
        x = self.swint5(x)
        x = self.swint5(x)
        enc3 = self.swint5(x)
        x = self.swint6(x)
        #x = self.pool3(enc3)

        #BOTTLENECK
        x = self.convB1(x)
        x = F.gelu(self.normB1(x))
        #x = F.gelu(x)
        x = self.convB2(x)
        x = F.gelu(self.normB2(x))
        #x = self.swintB1(x)
        #x = self.swintB1(x)
        #x = self.swintB2(x)
        #x = self.swintB2(x)
        
        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x)
        #skip1
        #x = torch.cat((x, enc3), dim=1)
        #x = self.conv10(x)
        #x = F.gelu(self.norm10(x))
        #x = F.gelu(x)
        x = self.conv11(x)
        x = F.gelu(self.norm11(x))
        #x = self.swint7(x)
        #x = self.swint8(x)
        
        #BLOCK 2
        x = self.upconv2(x)
        #skip2
        x = torch.cat((x, enc2), dim=1)
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        #x = F.gelu(x)
        x = self.conv13(x)
        x = F.gelu(self.norm13(x))
        #x = self.swint9(x)
        #x = self.swint10(x)
        
        #BLOCK 3
        x = self.upconv3(x)
        #skip3
        x = torch.cat((x, enc1), dim=1)
        x = self.conv14(x)
        x = F.gelu(self.norm14(x))
        #x = F.gelu(x)
        x = self.conv15(x)
        x = F.gelu(self.norm15(x))
        #x = self.swint11(x)
        #x = self.swint12(x)
        
        return torch.sigmoid(self.convEND(x))   
    


if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = swinTWUNet().cuda()
  out = net(x1)
  print(out.shape)
  
