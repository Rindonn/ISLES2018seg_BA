#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-06-27 10:12:11
@ LastEditors: Rindon
@ LastEditTime: 2024-06-27 10:12:11
@ Description: 
'''

def test():
	'''
	@ description: 
	@ param {type} 
	@ return: 
	'''
	pass

if __name__ == "__main__":
	test()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-16 11:35:28
@ LastEditors: Rindon
@ LastEditTime: 2024-05-04 17:22:36
@ Description: swinT and unet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import swinT
from ScConv import ScConv
from HWD import Down_wt

class TAU_module(nn.Module):
    def __init__(self):
        super(TAU_module, self).__init__()
        # DOWN BLOCK 1-1 #5*256*256
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(4, 32)
        #gelu
        self.conv2 = ScConv(32)
        self.norm2 = nn.GroupNorm(4, 32)
        #gelu
        self.pool1 = Down_wt(32,64)
        #32*128*128

        #DOWN BLOCK 2-1 #32*128*128
        self.conv3 = ScConv(64)
        self.norm3 = nn.GroupNorm(4, 64)
        #gelu
        self.conv4 = ScConv(64)
        self.norm4 = nn.GroupNorm(4, 64)
        #gelu
        self.pool2 = Down_wt(64,128)
        #64*64*64

        #DOWN BLOCK 3-1 #64*64*64
        self.conv5 = ScConv(128)
        self.norm5 = nn.GroupNorm(4, 128)
        #gelu
        self.conv6 = ScConv(128)
        self.norm6 = nn.GroupNorm(4, 128)
        #gelu
        self.pool3 = Down_wt(128,128)
        #128*32*32

        # DOWN BLOCK 1-2 #5*256*256
        self.conv7 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        #self.norm7 = nn.BatchNorm2d(32)
        self.norm7 = nn.GroupNorm(4, 32)
        #32*256*256
        '''self.swint1 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=4, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)'''
        self.swint2 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint3 = swinT.SwinT(in_channels=32, input_resolution=(256,256), num_heads=4, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #64*128*128

        #DOWN BLOCK 2-2 #64*128*128      
        '''self.swint4 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=4, 
                    window_size=4, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)'''
        self.swint5 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=8, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint6 = swinT.SwinT(in_channels=64, input_resolution=(128,128), num_heads=8, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #128*64*64

        #DOWN BLOCK 3-2 #128*64*64
        '''self.swint7 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=4, 
                    window_size=4, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)'''
        self.swint8 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=16, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swint9 = swinT.SwinT(in_channels=128, input_resolution=(64,64), num_heads=16, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=True)
        #128*32*32
        self.conv8 = nn.Conv2d(256, 128, 1, padding=0, bias=False)
        self.norm8 = nn.GroupNorm(4, 128)

        #############
        
        #Bottleneck #
        self.convB1 = ScConv(256)
        self.normB1 = nn.GroupNorm(4, 256)
        #gelu
        self.convB2 = ScConv(256)
        self.normB2 = nn.GroupNorm(4, 256)
        self.swintB1 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=16, 
                    window_size=8, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
        self.swintB2 = swinT.SwinT(in_channels=256, input_resolution=(32,32), num_heads=16, 
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
        x1 = x
        #BLOCK 1-1
        x = self.conv1(x)
        #x = F.gelu(self.norm1(x))
        x = self.conv2(x)
        enc1 = F.gelu(self.norm2(x)) #32*256*256
        x = self.pool1(enc1) 
        #32*128*128


        #BLOCK 2-1
        x = self.conv3(x) 
        #x = F.gelu(self.norm3(x))
        x = self.conv4(x)
        enc3 = F.gelu(self.norm4(x)) #64*128*128
        x = self.pool2(enc3)
        #64*64*64

                
        #BLOCK 3-1
        x = self.conv5(x)
        #x = F.gelu(self.norm5(x))
        x = self.conv6(x)
        enc5 = F.gelu(self.norm6(x)) #128*64*64
        x = self.pool3(enc5)
        #128*32*32

        #5*256*256
        #BLOCK 1-2
        x1 = self.conv7(x1) #32*256*256
        x1 = F.gelu(self.norm7(x1))
        #x1 = self.swint1(x1)
        enc2 = self.swint2(x1) #32*256*256
        x1 = self.swint3(enc2) 
        #64*128*128

        #BLOCK 2-2
        #x1 = self.swint4(x1)
        enc4 = self.swint5(x1) #64*128*128
        x1 = self.swint6(x1) 
        #128*64*64
        
        #BLOCK 3-2
        #x1 = self.swint7(x1)
        x1 = self.swint8(x1) #128*64*64
        x1 = self.swint8(x1)
        '''x1 = self.swint8(x1)
        x1 = self.swint8(x1)'''
        enc6 = self.swint8(x1) 
        x1 = self.swint9(enc6)
        x1 = self.conv8(x1) #256*32*32
        x1 = F.gelu(self.norm8(x1))
        #128*32*32

        #BOTTLENECK
        x = torch.cat((x, x1), dim=1) #256*32*32
        
        x = self.convB1(x)
        x = F.gelu(self.normB1(x))
        x = self.convB2(x)
        x = F.gelu(self.normB2(x))
        x = self.swintB1(x)
        x = self.swintB2(x)
        #256*32*32

        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x) #128*64*64
        x1 = torch.cat((enc5, enc6), dim=1) #256*64*64
        x1 = self.conv10(x1) #128*64*64
        x1 = F.gelu(self.norm10(x1))
        x = torch.cat((x, x1), dim=1) #256*64*64
        x = self.conv10(x) #128*64*64
        x = F.gelu(x)
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        x = self.conv12(x)
        x = F.gelu(self.norm12(x))
        #128*64*64
        
        
        #BLOCK 2
        x = self.upconv2(x) #64*128*128
        x1 = torch.cat((enc3, enc4), dim=1) #128*128*128
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
        x1 = torch.cat((enc1, enc2), dim=1) #64*256*256
        x1 = self.conv16(x1) #32*256*256
        x = F.gelu(self.norm16(x))
        x = torch.cat((x, x1), dim=1)#64*256*256
        x = self.conv16(x) #32*256*256
        x = F.gelu(x)
        x = self.conv18(x)
        x = F.gelu(self.norm18(x))
        #32*256*256

        return torch.sigmoid(self.convEND(x))   
    


if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = TAU_module().cuda()
  out = net(x1)
  print(out.shape)
  