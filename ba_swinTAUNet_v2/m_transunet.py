### Encoder ###

import torch
import torch.nn as nn
import torch.functional as F

from einops import rearrange,repeat
from einops.layers.torch import Rearrange


class BasicBlock(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=32, dilation=1, norm_layer=None):
    super(BasicBlock, self).__init__()

    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 32:
      raise ValueError("BasicBlock only supports groups=1 and base_width=32")
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    
    # 先将输入下采样，当stride != 1时, input 走 self.conv1 和 self.downsample
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True) # inplace:表示是否覆盖之前的值
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x
    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=32, dilation=1, norm_layer=None):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width/32.0)) * groups
    # 当stride != 1时, input 走 self.conv1 和 self.downsample
    self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
    self.bn1 = norm_layer(width)
    self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = nn.Conv2d(width, planes*self.expansion, kernel_size=1, stride=1, bias=False)
    self.bn3 = norm_layer(planes*self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1, zero_init_residual=False, groups=1, width_per_group=32, replace_stride_with_dilation=None, norm_layer=None):
    super(ResNet, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d

    self._norm_layer = norm_layer
    self.inplanes = 32
    self.dilation = 2

    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False, False]

    if len(replace_stride_with_dilation) != 4:
      raise ValueError(
          "replace_stride_with_dilation should be None"
          f"or a 4-element tuple, got{replace_stride_with_dilation}"
      )
    
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) #更改: 3 -> 5
    self.bn1 = norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 32//4, layers[0], stride=2)
    self.layer2 = self._make_layer(block, 64//4, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 128//4, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 256//4, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(256 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d,nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


    # 零初始化BN, zero-initialize the last BN in each residual branch,
    # the residual branch starts with zeros, and each residual block behaves like an identity.
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m,Bottleneck):
          nn.init.constant_(m.bn3.weight,0)
        elif isinstance(m,BasicBlock):
          nn.init.constant_(m.bn2.weight,0)
  
  def _make_layer(
      self,
      block,
      planes,
      blocks,
      stride = 1,
      dilate = False
      ):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = stride

    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
          norm_layer(planes * block.expansion)
      )

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
    )
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer
          )
      )
    return nn.Sequential(*layers)

  def _forward_impl(self, x):
    out = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.layer1(x)
    out.append(x)
    x = self.layer2(x)
    out.append(x)
    x = self.layer3(x)
    out.append(x)
    #最后一层 self.layer4 不输出

    return out

  def forward(self, x):
    return self._forward_impl(x)

  def _resnet(block, layers, pretrained_path=None, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained_path is not None:
      model.load_state_dict(torch.load(pretrained_path), strict=False)
    return model

  def resnet50(pretrained_path=None, **kwargs):
    return ResNet._resnet(Bottleneck,[3,4,6,3],pretrained_path, **kwargs)
  
  def resnet101(pretrained_path=None, **kwargs):
    return ResNet._resnet(Bottleneck,[3,4,23,3],pretrained_path, **kwargs)
  


### ViT ###

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
  def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
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
          PreNorm(dim, Attention(dim,heads=heads,dim_head=dim_head,dropout=dropout)),
          PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout))
      ]))
    
  def forward(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return x


class ViT(nn.Module):
  def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels, dim_head=32, dropout=0., emb_dropout=0.):
    super().__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0,'Image dimensions must be divisible by the patch size.'

    num_patches = (image_height // patch_height) * (image_width // patch_width)
    patch_dim = channels * patch_height * patch_width

    self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height,p2=patch_width),
        nn.Linear(patch_dim,dim)     
    )
    self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,dim))
    self.cls_token = nn.Parameter(torch.randn(1,1,dim))
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
    # 输出前将索引删除
    output = x[:,1:,:]
    output = self.out(output)
    # Transformer输出后，上采样到原始尺寸
    output = self.upsample(output)
    output = self.conv(output)

    return output


### TransUnet_Encoder ###

class TransUnetEncoder(nn.Module):
  def __init__(self, **kwargs):
    super(TransUnetEncoder, self).__init__()
    self.R50 = ResNet.resnet50()
    self.Vit = ViT(image_size=(32,32), patch_size=8, channels=128, dim=256, depth=12, heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1)
    
  def forward(self,x):
    x1, x2, x3 = self.R50(x)
    x4 = self.Vit(x3)

    return [x1, x2, x3, x4]
  

### Decoder ###

class TransUnetDecoder(nn.Module):
  def __init__(self,out_channels=32, **kwargs):
    super(TransUnetDecoder, self).__init__()
    self.decoder1 = nn.Sequential(
        nn.Conv2d(out_channels//4, out_channels//4, 3, padding=1),
        nn.BatchNorm2d(out_channels//4),
        nn.ReLU()
    )
    self.upsample1 = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(out_channels, out_channels//4, 3, padding=1),
        nn.BatchNorm2d(out_channels//4),
        nn.ReLU()
    )

    self.decoder2 = nn.Sequential(
        nn.Conv2d(out_channels*2, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.upsample2 = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(out_channels*2, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.decoder3 = nn.Sequential(
        nn.Conv2d(out_channels*4, out_channels*2, 3, padding=1),
        nn.BatchNorm2d(out_channels*2),
        nn.ReLU()
    )
    self.upsample3 = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(out_channels*4, out_channels*2, 3, padding=1),
        nn.BatchNorm2d(out_channels*2),
        nn.ReLU()
    )
    self.decoder4 = nn.Sequential(
        nn.Conv2d(out_channels*8, out_channels*4, 3, padding=1),
        nn.BatchNorm2d(out_channels*4),
        nn.ReLU()
    )
    self.upsample4 = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(out_channels*8, out_channels*4, 3, padding=1),
        nn.BatchNorm2d(out_channels*4),
        nn.ReLU()
    )

  def forward(self, inputs):
    x1, x2, x3, x4 = inputs
    # 维度[b, 512, H/8, W/8]

    x4 = self.upsample4(x4)
    x = self.decoder4(torch.cat([x4,x3],dim=1))

    x = self.upsample3(x)
    x = self.decoder3(torch.cat([x,x2],dim=1))

    x = self.upsample2(x)
    x = self.decoder2(torch.cat([x,x1],dim=1))

    x = self.upsample1(x)
    x = self.decoder1(x)

    return x
  

### TransUnet ###

class TransUnet(nn.Module):
  # 修改num_classes参数
  def __init__(self, num_classes=1, **kwargs):
    super(TransUnet, self).__init__()
    self.TransUnetEncoder = TransUnetEncoder()
    self.TransUnetDecoder = TransUnetDecoder()
    self.cls_head = nn.Conv2d(8,num_classes,1)

  def forward(self, x):
    x = self.TransUnetEncoder(x)
    x = self.TransUnetDecoder(x)
    x = self.cls_head(x)

    return torch.sigmoid(x)
  

if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256])
  net = TransUnet()
  out = net(x1)
  print(out.shape)
