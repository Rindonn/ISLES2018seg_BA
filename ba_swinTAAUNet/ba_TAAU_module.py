#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-05-02 18:09:05
@ Description: swinT and unet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import swinT
from ScConv import ScConv
from HWD import Down_wt

class TAAU_module(nn.Module):
    def __init__(self):
        super(TAAU_module, self).__init__()
        # DOWN BLOCK 1-1 #5*256*256
        self.conv111 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        self.norm111 = nn.GroupNorm(4, 32)
        #gelu
        self.conv112 = ScConv(32)
        self.norm112 = nn.GroupNorm(4, 32)
        #gelu
        self.pool111 = nn.MaxPool2d(kernel_size=2, stride=2)
        #32*128*128

        # DOWN BLOCK 1-2 #5*256*256
        self.conv121 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        self.norm121 = nn.GroupNorm(4, 32)
        #gelu
        self.conv122 = ScConv(32)
        self.norm122 = nn.GroupNorm(4, 32)
        #gelu
        self.pool122 = nn.MaxPool2d(kernel_size=2, stride=2)
        #32*128*128

        # DOWN BLOCK 1-3 #5*256*256
        self.conv131 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        self.norm131 = nn.GroupNorm(4, 32)
        #32*256*256
        self.swint131 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint132 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #64*128*128
        self.conv132 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.norm132 = nn.GroupNorm(4, 32)

        # DOWN BLOCK 1-4 #5*256*256
        self.conv141 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        #self.norm7 = nn.BatchNorm2d(32)
        self.norm141 = nn.GroupNorm(4, 32)
        #32*256*256
        self.swint141 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint142 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #64*128*128
        self.conv142 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.norm142 = nn.GroupNorm(4, 32)

        #DOWN BLOCK 2-1 #32*128*128
        self.conv211 = ScConv(64)
        self.norm211 = nn.GroupNorm(4, 64)
        #gelu
        self.conv212 = ScConv(64)
        self.norm212 = nn.GroupNorm(4, 64)
        #gelu
        self.pool211 = nn.MaxPool2d(kernel_size=2, stride=2)
        #64*64*64

        #DOWN BLOCK 2-2 #64*128*128      
        self.swint221 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint222 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        self.conv221 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.norm221 = nn.GroupNorm(4, 64)
        #128*64*64

        #DOWN BLOCK 3 #64*64*64
        self.conv31 = ScConv(128)
        self.norm31 = nn.GroupNorm(4, 128)
        #gelu
        self.conv32 = ScConv(128)
        self.norm32 = nn.GroupNorm(4, 128)
        #gelu
        self.swint31 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=2, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint32 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=2, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #128*32*32

        #############
        
        #Bottleneck #
        self.convB1 = ScConv(256)
        self.normB1 = nn.GroupNorm(4, 256)
        #gelu
        self.convB2 = ScConv(256)
        self.normB2 = nn.GroupNorm(4, 256)
        self.swintB1 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swintB2 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        
        #############
        
        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.norm10 = nn.GroupNorm(4, 128)
        #gelu
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm12 = nn.GroupNorm(4, 128)
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, 3, padding=1, bias=False)
        self.norm13 = nn.GroupNorm(4, 64)
        #gelu
        self.conv15 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.norm15 = nn.GroupNorm(4, 64)

        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv16 = nn.Conv2d(64,32,3,padding=1, bias=False)
        self.norm16 = nn.GroupNorm(4, 32)
        #gelu
        self.conv18 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.norm18 = nn.GroupNorm(4, 32)
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)
        
        
        
        
        
        
    def forward(self, x):
        #### ENCODER ####
        #5*256*256
        x11 = x
        x12 = x
        x13 = x
        x14 = x
        #BLOCK 1-1
        x11 = self.conv111(x11)
        #x = F.gelu(self.norm1(x))
        x11 = self.conv112(x11)
        enc11 = F.gelu(self.norm112(x11)) #32*256*256
        x11 = self.pool111(enc11) 
        #32*128*128

        #BLOCK 1-2
        x12 = self.conv121(x12)
        #x = F.gelu(self.norm1(x))
        x12 = self.conv122(x12)
        enc12 = F.gelu(self.norm122(x12)) #32*256*256
        x12 = self.pool122(enc12) 
        #32*128*128

        #5*256*256
        #BLOCK 1-3
        x13 = self.conv131(x13) #32*256*256
        x13 = F.gelu(self.norm131(x13))
        #x1 = self.swint1(x1)
        x13 = self.swint131(x13) #32*256*256
        enc13 = self.swint131(x13) #32*256*256
        x13 = self.swint132(x13)
        x13 = self.conv132(x13) #64*128*128
        x13 = F.gelu(self.norm132(x13)) 
        #32*128*128

        #5*256*256
        #BLOCK 1-4
        x14 = self.conv141(x14) #32*256*256
        x14 = F.gelu(self.norm141(x14))
        x14 = self.swint141(x14)
        enc14 = self.swint141(x14) #32*256*256
        x14 = self.swint142(x14) 
        x14 = self.conv142(x14) #64*128*128
        x14 = F.gelu(self.norm142(x14))
        #32*128*128


        #BLOCK 2-1
        x21 = torch.concat((x11,x13),dim = 1)
        x21 = self.conv211(x21) 
        #x = F.gelu(self.norm3(x))
        x21 = self.conv212(x21)
        enc21 = F.gelu(self.norm212(x21)) #64*128*128
        x21 = self.pool211(x21)
        #64*64*64

        #BLOCK 2-2
        x22 = torch.concat((x12,x14),dim = 1)
        x22 = self.conv212(x22)
        x22 = F.gelu(self.norm212(x22)) #64*128*128
        x22 = self.swint221(x22) #64*128*128
        enc22 = self.swint221(x22) #64*128*128
        x22 = self.swint222(x22) 
        x22 = self.conv221(x22) #32*256*256
        x22 = F.gelu(self.norm221(x22))
        #128*64*64

                
        #BLOCK 3
        x = torch.concat((x21,x22),dim = 1)
        x = self.conv31(x)
        #x = F.gelu(self.norm5(x))
        x = self.conv32(x)
        x = F.gelu(self.norm32(x)) #128*64*64
        x = self.swint31(x) #128*64*64
        #x = self.swint31(x)
        '''x1 = self.swint8(x1)
        x1 = self.swint8(x1)'''
        enc3 = self.swint31(x)  #128*64*64
        x = self.swint32(enc3)
        #128*32*32

        #BOTTLENECK
        x = self.convB1(x)
        x = F.gelu(self.normB1(x))
        x = self.convB2(x)
        x = F.gelu(self.normB2(x))
        #x = self.swintB1(x)
        #x = self.swintB2(x)
        #256*32*32

        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x) #128*64*64
        x = torch.cat((x, enc3), dim=1) #256*64*64
        x = self.conv10(x) #128*64*64
        x = F.gelu(x)
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        #128*64*64
        
        
        #BLOCK 2
        x = self.upconv2(x) #64*128*128
        x1 = torch.cat((enc21, enc22), dim=1) #128*128*128
        x1 = self.conv13(x1) #64*128*128
        x1 = F.gelu(self.norm13(x1))
        x = torch.cat((x, x1), dim=1) #128*128*128
        x = self.conv13(x) #64*128*128
        x = F.gelu(x)
        x = self.conv15(x)
        x = F.gelu(self.norm15(x))
        #64*128*128

        #BLOCK 3 
        x = self.upconv3(x) #32*256*256
        x1 = torch.cat((enc11, enc13), dim=1) #64*256*256
        x1 = self.conv16(x1) #32*256*256
        x1 = F.gelu(self.norm16(x1))
        x2 = torch.cat((enc12, enc14), dim=1) #64*256*256
        x2 = self.conv16(x2) #32*256*256
        x2 = F.gelu(self.norm16(x2))
        x3 = torch.cat((x1, x2), dim=1)#64*256*256
        x3 = self.conv16(x3) #32*256*256
        x3 = F.gelu(self.norm16(x3))
        x = torch.cat((x, x3), dim=1)#64*256*256
        x = self.conv16(x) #32*256*256
        x = F.gelu(x)
        x = self.conv18(x)
        x = F.gelu(self.norm18(x))
        #32*256*256

        return torch.sigmoid(self.convEND(x))   
    


if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = TAAU_module().cuda()
  out = net(x1)
  print(out.shape)
  