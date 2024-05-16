#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-04-25 16:37:12
@ Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import swinT
from odconv import ODConv2d

class swinTUNet(nn.Module):
    def __init__(self):
        super(swinTUNet, self).__init__()
        # DOWN BLOCK 1 #5*256*256
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        #gelu
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        #gelu
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        #gelu
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #32*128*128
        #DOWN BLOCK 2 #32*128*128
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        #gelu
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(64)
        #gelu
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(64)
        #gelu
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #64*64*64
        #DOWN BLOCK 3 #64*64*64
        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm7 = nn.BatchNorm2d(128)
        #gelu
        self.conv8 = nn.Conv2d(128,128, 3, padding=1)
        self.norm8 = nn.BatchNorm2d(128)
        #gelu
        self.conv9 = nn.Conv2d(128,128, 3, padding=1)
        self.norm9 = nn.BatchNorm2d(128)
        #gelu
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #128*32*32
        #DOWN BLOCK 4 #128*32*32
        self.swint1 = swinT.SwinT(in_channels=128, input_resolution=(32,32), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint2 = swinT.SwinT(in_channels=128, input_resolution=(32,32), num_heads=8, 
                    window_size=16, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint3 = swinT.SwinT(in_channels=128, input_resolution=(32,32), num_heads=16, 
                    window_size=32, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        #############
        
        #Bottleneck #
        self.convB1 = nn.Conv2d(128, 256, 3, padding=1)
        self.normB1 = nn.BatchNorm2d(256)
        #gelu
        self.convB2 = nn.Conv2d(256, 256, 3, padding=1)
        self.normB2 = nn.BatchNorm2d(256)
        #gelu
        self.convB3 = nn.Conv2d(256, 256, 3, padding=1)
        self.normB3 = nn.BatchNorm2d(256)
        
        
        #############
        
        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm10 = nn.BatchNorm2d(128)
        #gelu
        self.conv11 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm11 = nn.BatchNorm2d(128)
        #gelu
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm12 = nn.BatchNorm2d(128)
        
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1)
        self.norm13 = nn.BatchNorm2d(64)
        #gelu
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm14 = nn.BatchNorm2d(64)
        #gelu
        self.conv15 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm15 = nn.BatchNorm2d(64)
        
        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv16 = nn.Conv2d(64,32,3,padding=1)
        self.norm16 = nn.BatchNorm2d(32)
        #gelu
        self.conv17 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm17 = nn.BatchNorm2d(32)
        #gelu
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm18 = nn.BatchNorm2d(32)
        
        '''
        #UP BLOCK 4#
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(64,32,3,padding=1)
        self.norm15 = nn.BatchNorm2d(32)
        #gelu
        self.conv16 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm16 = nn.BatchNorm2d(32)
        #gelu
        '''
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        #### ENCODER ####
        
        #BLOCK 1
        x = self.conv1(x)
        x = F.gelu(self.norm1(x))
        #x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(self.norm2(x))
        x = self.conv3(x)
        enc1 = F.gelu(self.norm3(x))
        x = self.pool1(enc1)
        
        #BLOCK 2
        x = self.conv4(x)
        x = F.gelu(self.norm4(x))
        #x = F.gelu(x)
        x = self.conv5(x)
        x = F.gelu(self.norm5(x))
        x = self.conv6(x)
        enc2 = F.gelu(self.norm6(x))
        x = self.pool2(enc2)
        
        #BLOCK 3
        x = self.conv7(x)
        x = F.gelu(self.norm7(x))
        #x = F.gelu(x)
        x = self.conv8(x)
        x = F.gelu(self.norm8(x))
        x = self.conv9(x)
        enc3 = F.gelu(self.norm9(x))
        x = self.pool3(enc3)
        
        #BLOCK 4(swin-Transformer)
        x = self.swint1(x)
        x = self.swint1(x)
        
        x = self.swint2(x)
        x = self.swint2(x)
        
        x = self.swint3(x)
        x = self.swint3(x)
        #x = self.swint3(x)
        #x = self.swint3(x)
        x = self.swint3(x)
        x = self.swint3(x)
        
        x = self.swint2(x)
        x = self.swint2(x)
        
        #x = self.swint(x)
        #x = self.swint(x)
        #x = self.swint(x)
        

        #BOTTLENECK
        x = self.convB1(x)
        x = F.gelu(self.normB1(x))
        #x = F.gelu(x)
        x = self.convB2(x)
        x = F.gelu(self.normB2(x))
        x = self.convB3(x)
        x = F.gelu(self.normB3(x))

        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x)
        #x = torch.cat((x, enc3), dim=1)
        #x = self.conv10(x)
        #x = F.gelu(self.norm10(x))
        #x = F.gelu(x)
        x = self.conv11(x)
        x = F.gelu(self.norm11(x))
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        
        #BLOCK 2
        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.conv13(x)
        x = F.gelu(self.norm13(x))
        #x = F.gelu(x)
        x = self.conv14(x)
        x = F.gelu(self.norm14(x))
        x = self.conv15(x)
        x = F.gelu(self.norm15(x))
        
        #BLOCK 3
        x = self.upconv3(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.conv16(x)
        x = F.gelu(self.norm16(x))
        #x = F.gelu(x)
        x = self.conv17(x)
        x = F.gelu(self.norm17(x))
        x = self.conv18(x)
        x = F.gelu(self.norm18(x))
        
        return torch.sigmoid(self.convEND(x))   

if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = swinTUNet().cuda()
  out = net(x1)
  print(out.shape)
  
'''
net = Unet()
from torchsummary import summary
summary(net.cuda(),(5, 256, 256))
'''