#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-05-24 10:48:31
@ Description: swinT and unet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import swinT
from ScConv import ScConv
from HWD import Down_wt

class TXN_module(nn.Module):
    def __init__(self):
        super(TXN_module, self).__init__()
        self.convC0 = nn.Conv2d(5, 32, 1, padding=0, bias=False)
        self.normC0 = nn.GroupNorm(4, 32)

        self.convC1 = nn.Conv2d(128, 64, 1, padding=0, bias=False)
        self.convC2 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
        self.convC3 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
        # DOWN BLOCK 1-1 
        #32*256*256
        self.conv1 = ScConv(32)
        self.norm1 = nn.GroupNorm(4, 32)
        #gelu
        self.conv2 = ScConv(32)
        self.norm2 = nn.GroupNorm(4, 32)
        #gelu
        self.pool1 = Down_wt(32,64)
        #64*128*128

        #DOWN BLOCK 2-1 
        #64*128*128
        self.conv3 = ScConv(64)
        self.norm3 = nn.GroupNorm(4, 64)
        #gelu
        self.conv4 = ScConv(64)
        self.norm4 = nn.GroupNorm(4, 64)
        #gelu
        self.pool2 = Down_wt(64,128)
        #128*64*64

        #DOWN BLOCK 3-1 
        #128*64*64
        self.conv5 = ScConv(128)
        self.norm5 = nn.GroupNorm(4, 128)
        #gelu
        self.conv6 = ScConv(128)
        self.norm6 = nn.GroupNorm(4, 128)
        #gelu
        self.pool3 = Down_wt(128,256)
        #256*32*32

        # DOWN BLOCK 1-2
        #32*256*256
        self.swint1 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint2 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint3 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #128*128*128

        #DOWN BLOCK 2-2 
        #128*128*128      
        self.swint4 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=8, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint5 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=8, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint6 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=8, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #256*64*64

        #DOWN BLOCK 3-2 
        #256*64*64
        self.swint7 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=16, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint8 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=16, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint9 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=16, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #512*32*32

        #############
        
        #Bottleneck #
        #self.conv8 = nn.Conv2d(1024, 512, 1, padding=0, bias=False)
        #self.norm8 = nn.GroupNorm(4, 512)
        #gelu
        #self.conv9 = nn.Conv2d(512, 256, 1, padding=0, bias=False)
        self.norm9 = nn.GroupNorm(4, 256)

        self.convB1= ScConv(256)
        self.convB2 = ScConv(256)
        self.swintB1 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=16, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swintB2 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=16, 
                    window_size=8, qkv_bias=True, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        
        #############
        
        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
        self.norm10 = nn.GroupNorm(4, 128)
        #gelu
        self.conv12 = ScConv(128)
        self.norm12 = nn.GroupNorm(4, 128)
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 1, padding=0, bias=False)
        self.norm13 = nn.GroupNorm(4, 64)
        #gelu
        self.conv15 = ScConv(64)
        self.norm15 = nn.GroupNorm(4, 64)

        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv16 = nn.Conv2d(64,32,1,padding=0, bias=False)
        self.norm16 = nn.GroupNorm(4, 32)
        #gelu
        self.conv18 = ScConv(32)
        self.norm18 = nn.GroupNorm(4, 32)
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)
        
    def forward(self, x):
        #### ENCODER ####
        #5*256*256
        x = self.convC0(x) #32*256*256
        x = F.gelu(self.normC0(x))
        x1 = x
        enc1 = x #32*256*256

        #BLOCK 1-1 #32*256*256
        x = self.conv1(x)
        x = F.gelu(self.norm2(x)) #32*256*256
        x = self.conv2(x)
        x = F.gelu(self.norm2(x)) #32*256*256
        x = self.conv2(x)
        x = F.gelu(self.norm2(x)) #32*256*256
        x = self.pool1(x) 
        #64*128*128

        #BLOCK 1-2 #32*256*256
        #x1 = self.swint1(x1)
        x1 = self.swint2(x1) #32*256*256
        x1 = self.swint3(x1) 
        #64*128*128

        #concat1 
        x = torch.concat((x,x1),dim=1) #128*128*128
        x = self.convC1(x)#64*128*128
        x1 = x
        enc2 = x #64*128*128

        #BLOCK 2-1 #64*128*128
        x = self.conv3(x) 
        x = F.gelu(self.norm4(x)) #64*128*128
        x = self.conv4(x)
        x = F.gelu(self.norm4(x)) #64*128*128
        x = self.conv4(x)
        x = F.gelu(self.norm4(x)) #64*128*128
        x = self.pool2(x)
        #128*64*64

        #BLOCK 2-2 #64*128*128
        #x1 = self.swint4(x1)
        x1 = self.swint5(x1) #64*128*128
        x1 = self.swint6(x1) 
        #128*64*64

        #concat2 
        x = torch.concat((x,x1),dim=1) #256*64*64
        x = self.convC2(x)#128*64*64
        x1 = x
        enc3 = x #128*64*64

        #BLOCK 3-1 #128*64*64
        x = self.conv5(x)
        x = F.gelu(self.norm6(x)) #128*64*64
        x = self.conv6(x)
        x = F.gelu(self.norm6(x)) #128*64*64
        x = self.conv6(x)
        x = F.gelu(self.norm6(x)) #128*64*64
        x = self.pool3(x)
        #256*32*32

        #BLOCK 3-2 #128*64*64
        x1 = self.swint7(x1)
        x1 = self.swint8(x1) #128*64*64
        x1 = self.swint8(x1)
        #x1 = self.swint8(x1)
        #x1 = self.swint8(x1)
        x1 = self.swint9(x1)
        #256*32*32

        #concat3
        x = torch.cat((x, x1), dim=1) #512*32*32
        x = self.convC3(x)#256*32*32
        x = F.gelu(self.norm9(x))
        #BottleNeck
        x = self.convB1(x)
        x = F.gelu(self.norm9(x))
        x = self.convB2(x)
        x = F.gelu(self.norm9(x))
        x = self.swintB1(x)
        x = self.swintB2(x)
        #256*32*32

        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x) #128*64*64
        x = torch.cat((x,enc3),dim=1)#256*64*64
        x = self.conv10(x) #128*64*64
        x = F.gelu(self.norm12(x))
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        #x = self.swint8(x)
        #x = self.swint8(x)
        #128*64*64
        
        
        #BLOCK 2
        x = self.upconv2(x) #64*128*128
        x = torch.cat((x, enc2), dim=1) #128*128*128
        x = self.conv13(x) #64*128*128
        x = F.gelu(self.norm15(x))
        x = self.conv15(x)
        x = F.gelu(self.norm15(x))
        x = self.conv15(x)
        x = F.gelu(self.norm15(x))
        #x = self.swint4(x)
        #x = self.swint4(x)
        #64*128*128

        #BLOCK 3 
        x = self.upconv3(x) #32*256*256
        x = torch.cat((x, enc1), dim=1) #64*256*256
        x = self.conv16(x) #32*256*256
        x = F.gelu(self.norm18(x))
        x = self.conv18(x) 
        x = F.gelu(self.norm18(x))
        x = self.conv18(x)
        x = F.gelu(self.norm18(x))
        #x = self.swint1(x)
        #x = self.swint1(x)
        #32*256*256
        
        x = self.convEND(x)
        return torch.sigmoid(x)   
        #return torch.softmax(x,dim=1)


if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = TXN_module().cuda()
  out = net(x1)
  print(out.shape)
  