import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat
from einops.layers.torch import Rearrange



##### xception 模块 #####
bn_mom = 0.0003

class SeparableConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
          dilation=1, bias=False, activate_first=True, inplace=True):
    super(SeparableConv2d, self).__init__()
    self.relu0 = nn.ReLU(inplace=inplace)
    self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                    padding, dilation, groups=in_channels, bias=bias)
    self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
    self.relu1 = nn.ReLU(inplace=True)
    self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
    self.relu2 = nn.ReLU(inplace=True)
    self.activate_first = activate_first

  def forward(self, x):
    if self.activate_first:
        x = self.relu0(x)
    x = self.depthwise(x)
    x = self.bn1(x)
    if not self.activate_first:
        x = self.relu1(x)
    x = self.pointwise(x)
    x = self.bn2(x)
    if not self.activate_first:
        x = self.relu2(x)
    return x


class Block(nn.Module):
  def __init__(self, in_filters, out_filters, strides=1, atrous=None, grow_first=True,
          activate_first=True, inplace=True):
    super(Block, self).__init__()
    if atrous == None:
        atrous = [1] * 3
    elif isinstance(atrous, int):
        atrous_list = [atrous] * 3
        atrous = atrous_list
    idx = 0
    self.head_relu = True
    if out_filters != in_filters or strides != 1:
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
        self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
        self.head_relu = False
    else:
        self.skip = None

    self.hook_layer = None
    if grow_first:
        filters = out_filters
    else:
        filters = in_filters
    self.sepconv1 = SeparableConv2d(in_filters, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0],
                      bias=False, activate_first=activate_first, inplace=self.head_relu)
    self.sepconv2 = SeparableConv2d(filters, out_filters, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1],
                      bias=False, activate_first=activate_first)
    self.sepconv3 = SeparableConv2d(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],
                      dilation=atrous[2], bias=False, activate_first=activate_first, inplace=inplace)

  def forward(self, inp):
    if self.skip is not None:
        skip = self.skip(inp)
        skip = self.skipbn(skip)
    else:
        skip = inp

    x = self.sepconv1(inp)
    x = self.sepconv2(x)
    self.hook_layer = x
    x = self.sepconv3(x)

    x += skip
    return x


class Xception(nn.Module):
  def __init__(self, downsample_factor):
    super(Xception, self).__init__()

    stride_list = None
    if downsample_factor == 8:
      stride_list = [2, 1, 1]
    elif downsample_factor == 16:
      stride_list = [2, 2, 1]
    else:
      raise ValueError('xception.py: output stride=%d is not supported.' % os)
    self.conv1 = nn.Conv2d(5, 16, 3, 2, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(16,momentum=bn_mom)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(32, momentum=bn_mom)
    # do relu here

    self.block1 = Block(32, 64, 2)
    self.block2 = Block(64, 128, stride_list[0], inplace=False)
    self.block3 = Block(128, 256, stride_list[1])

    rate = 16 // downsample_factor
    self.block4 = Block(256, 256, 1, atrous=rate)
    self.block5 = Block(256, 256, 1, atrous=rate)
    self.block6 = Block(256, 256, 1, atrous=rate)
    self.block7 = Block(256, 256, 1, atrous=rate)

    self.block8 = Block(256, 256, 1, atrous=rate)
    self.block9 = Block(256, 256, 1, atrous=rate)
    self.block10 = Block(256, 256, 1, atrous=rate)
    self.block11 = Block(256, 256, 1, atrous=rate)

    self.block12 = Block(256, 256, 1, atrous=rate)
    self.block13 = Block(256, 256, 1, atrous=rate)
    self.block14 = Block(256, 256, 1, atrous=rate)
    self.block15 = Block(256, 256, 1, atrous=rate)

    self.block16 = Block(256, 256, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
    self.block17 = Block(256, 256, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
    self.block18 = Block(256, 256, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
    self.block19 = Block(256, 256, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

    self.block20 = Block(256, 512, stride_list[2], atrous=rate, grow_first=False)
    self.conv3 = SeparableConv2d(512, 768, 3, 1, 1 * rate, dilation=rate, activate_first=False)

    self.conv4 = SeparableConv2d(768, 768, 3, 1, 1 * rate, dilation=rate, activate_first=False)

    self.conv5 = SeparableConv2d(768, 1024, 3, 1, 1 * rate, dilation=rate, activate_first=False)
    self.layers = []

  def forward(self, input):
    self.layers = []
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    low_featrue_1 = self.block1.hook_layer
    x = self.block2(x)
    low_featrue_2 = self.block2.hook_layer
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)
    x = self.block13(x)
    x = self.block14(x)
    x = self.block15(x)
    x = self.block16(x)
    x = self.block17(x)
    x = self.block18(x)
    x = self.block19(x)
    x = self.block20(x)

    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)

    return low_featrue_1, low_featrue_2, x


def xception(downsample_factor=8,pretrained=False):
  model = Xception(downsample_factor=downsample_factor)
  return model


##### aspp 模块 #####

class asppConv(nn.Sequential):
  def __init__(self, in_channels, out_channels, dilation):
    modules = [
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    super(asppConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class asppPooling(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(asppPooling, self).__init__(
      nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
      nn.Conv2d(in_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU())

  def forward(self, x):
    size = x.shape[-2:]
    for mod in self:
        x = mod(x)
    # 上采样

    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# 整个 ASPP 架构
class aspp(nn.Module):
  def __init__(self, in_channels, atrous_rates, out_channels=256):
    super(aspp, self).__init__()
    modules = []
    # 1*1 卷积
    modules.append(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    )

    # 多尺度空洞卷积
    rates = tuple(atrous_rates)
    for rate in rates:
      modules.append(asppConv(in_channels, out_channels, rate))

    # 池化
    modules.append(asppPooling(in_channels, out_channels))
    self.convs = nn.ModuleList(modules)

    # 拼接后的卷积
    self.project = nn.Sequential(
        nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(0.5))

  def forward(self, x):
    res = []
    for conv in self.convs:
      res.append(conv(x))
    res = torch.cat(res, dim=1)
    return self.project(res)


##### ViT 模块 #####

def pair(t):
  return t if isinstance(t, tuple) else (t,t)


class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim,hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim,dim),
        nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)


class Attention(nn.Module):
  def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
    super().__init__()
    inner_dim = dim_head * heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(dropout)

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=self.heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)
    out = rearrange(out,'b h n d -> b n (h d)')

    return self.to_out(out)


class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
    super().__init__()
    self.layers = nn.ModuleList([])

    for _ in range(depth):
      self.layers.append(nn.ModuleList([
          PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
          PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
      ]))

  def forward(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return x


class ViT(nn.Module):
  def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels, dim_head=64, dropout=0., emb_dropout=0.):
    super().__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0,'Image dimensions must be divisible by the patch size.'

    num_patches = (image_height // patch_height) * (image_width // patch_width)
    patch_dim = channels * patch_height * patch_width

    self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        nn.Linear(patch_dim, dim)
    )
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    self.dropout = nn.Dropout(emb_dropout)

    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    self.out = Rearrange('b (h w) c -> b c h w', h=image_height//patch_height, w=image_width//patch_width)

    # 为了保持和论文中的图的feature size一样，上采样倍数为8倍
    self.upsample = nn.UpsamplingBilinear2d(scale_factor=patch_size//2)
    self.conv = nn.Sequential(
        nn.Conv2d(dim, dim, 3, padding=1),
        nn.BatchNorm2d(dim),
        nn.ReLU()
    )

  def forward(self, img):
    # 为了对应论文中的Linear Projection，是将图片分块嵌入，成为一个序列
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape
    # 为图像切片序列加上索引
    cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d',b=b)
    x = torch.cat((cls_tokens, x),dim=1)
    x += self.pos_embedding[:, :(n+1)]
    x = self.dropout(x)
    # 输入到Transformer中处理
    x = self.transformer(x)

    # delete cls_tokens 输出前将索引删除
    output = x[:,1:,:]
    output = self.out(output)

    # Transformer输出后，上采样到原始尺寸
    output = self.upsample(output)
    output = self.conv(output)

    return output


### Encoder ###

class Encoder(nn.Module):
  def __init__(self, pretrained=False, downsample_factor=8, **kwargs):
    super(Encoder, self).__init__()
    self.backbone = xception(downsample_factor=downsample_factor,pretrained=pretrained)
    in_channels = 1024
    #low_level_channels = 256

    # -----------------------------------------#
    #   ASPP特征提取模块
    #   利用不同膨胀率的膨胀卷积进行特征提取
    # -----------------------------------------#

    self.aspp = aspp(in_channels=in_channels, 
                     out_channels=256, 
                     atrous_rates=[16 // downsample_factor, 16 // downsample_factor])
    
    self.vit = ViT(image_size=(32,32), 
                   patch_size=8, 
                   channels=256,
                   dim=512, 
                   depth=2, 
                   heads=5, 
                   mlp_dim=1024, 
                   dropout=0.4, emb_dropout=0.4)

  def forward(self,x):
    x1, x2, x3 = self.backbone(x)
    x4 = self.aspp(x3)
    x5 = self.vit(x4)

    return [x1, x2, x3, x4, x5]


### Decoder ###

class Decoder(nn.Module):
  def __init__(self, **kwargs):
    super(Decoder, self).__init__()
    
    ## up1 ##
    
    self.upsample4 = nn.Sequential(
        nn.Conv2d(512, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.decoder4 = nn.Sequential(
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    '''
    ## up2 ##
    self.upsample3 = nn.Sequential(
        nn.Conv2d(256, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    self.decoder3 = nn.Sequential(
        nn.Conv2d(256, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    
    ## up3 ##
    self.upsample2 = nn.Sequential(
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.decoder2 = nn.Sequential(
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    '''
    ## up4 ##
    self.upsample1 = nn.Sequential(
        nn.Conv2d(64, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU()
    )
    self.decoder1 = nn.Sequential(
        nn.Conv2d(8, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU()
    )
    

  def forward(self, inputs):
    x1, x2, x3, x4, x5 = inputs
    # x3 unuse 
    x5 = self.upsample4(nn.functional.interpolate(input=x5, scale_factor=8, mode='nearest'))
    x = self.decoder4(torch.cat([x5,x1],dim=1))
    '''
    x = self.upsample3(nn.functional.interpolate(input=x, scale_factor=2, mode='nearest'))
    x = self.decoder3(torch.cat([x,x2],dim=1))
    
    x = self.upsample2(nn.functional.interpolate(input=x, scale_factor=2, mode='nearest'),)
    x = self.decoder2(torch.cat([x,x1],dim=1))
    '''
    x = self.upsample1(nn.functional.interpolate(input=x, scale_factor=2, mode='nearest'),)
    x = self.decoder1(x)
    

    return x


### DeepTransUnet ###

class DeepTransUnet(nn.Module):
  # 修改num_classes参数
  def __init__(self, num_classes=1):
    super(DeepTransUnet, self).__init__()
    self.Encoder = Encoder()
    self.Decoder = Decoder()
    self.cls_head = nn.Conv2d(8,num_classes,1)

  def forward(self, x):
    x = self.Encoder(x)
    x = self.Decoder(x)
    x = self.cls_head(x)

    return torch.sigmoid(x)
  



if __name__ == "__main__":
  x1 = torch.randn([2,5,256,256])
  net = DeepTransUnet()
  out = net(x1)
  print(out.shape)