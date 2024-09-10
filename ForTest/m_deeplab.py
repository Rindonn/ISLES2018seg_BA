import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

# ResNet101
def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=dilation, bias=False, dilation=dilation)
  
def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
      super(Bottleneck, self).__init__()
      self.conv1 = conv1x1(inplanes, planes)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = conv3x3(planes, planes, stride, dilation)
      self.bn2 = nn.BatchNorm2d(planes)
      self.conv3 = conv1x1(planes, planes * self.expansion)
      self.bn3 = nn.BatchNorm2d(planes * self.expansion)
      self.relu = nn.ReLU(inplace=True)
      self.downsample = downsample

  def forward(self, x):
      residual = x

      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)
      out = self.relu(out)

      out = self.conv3(out)
      out = self.bn3(out)

      if self.downsample is not None:
          residual = self.downsample(x)

      out += residual
      out = self.relu(out)

      return out


class ResNet(nn.Module):

  def __init__(self, block, layers, output_stride, pretrained=True):
      self.inplanes = 64
      super(ResNet, self).__init__()
      if output_stride == 16:
          strides = [2, 2, 1]
          dilations = [1, 1, 2]
      elif output_stride == 8:
          strides = [2, 1, 1]
          dilations = [1, 2, 4]
      else:
          raise NotImplementedError

      # Modules
      self.conv1 = nn.Conv2d(5, self.inplanes, kernel_size=7, stride=2, padding=3,
                   bias=False)
      self.bn1 = nn.BatchNorm2d(self.inplanes)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

      self.layer1 = self._make_layer(block, 64, layers[0])
      self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[0], dilation=dilations[0])
      self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[1], dilation=dilations[1])
      self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[2], dilation=dilations[2])

      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
          nn.init.constant_(m.weight,1)
          nn.init.constant_(m.bias,0)

      if pretrained:
          self._load_pretrained_model()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
              conv1x1(self.inplanes, planes * block.expansion, stride),
              nn.BatchNorm2d(planes * block.expansion),
          )

      layers = []
      layers.append(block(self.inplanes, planes, stride, dilation, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
          layers.append(block(self.inplanes, planes, dilation=dilation))

      return nn.Sequential(*layers) 

  def forward(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.layer4(x)
      
      return x

  def _load_pretrained_model(self):
      pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
      model_dict = {}
      state_dict = self.state_dict()
      for k, v in pretrain_dict.items():
          if k in state_dict:
              model_dict[k] = v
      state_dict.update(model_dict)
      self.load_state_dict(state_dict)

# 对于ResNet101，四个模块对应的bottleneck数量为[3,4,23,3]
def ResNet101(output_stride=16, pretrained=True):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, pretrained=pretrained)
  return model


# ASPP
class ASPPConv(nn.Sequential):
  def __init__(self, in_channels, out_channels, dilation):
    modules = [
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
  def __init__(self, in_channels, out_channels):
    super(ASPPPooling,self).__init__(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels,out_channels,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def forward(self,x):
    size = x.shape[-2:]
    for mod in self:
      x = mod(x)
    return F.interpolate(x, size=size, mode='bilinear',align_corners=False)


class ASPP(nn.Module):
  def __init__(self, in_channels, atrous_rates, out_channels=256):
    super(ASPP, self).__init__()
    modules = []
    modules.append(nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ))

    rates = tuple(atrous_rates)
    for rate in rates:
      modules.append(ASPPConv(in_channels, out_channels, rate))

    modules.append(ASPPPooling(in_channels, out_channels))
    self.convs = nn.ModuleList(modules)
    self.project = nn.Sequential(
        nn.Conv2d(len(self.convs)*out_channels, out_channels,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(0.5)
    )

  def forward(self,x):
    res = []
    for conv in self.convs:
      res.append(conv(x))
    res = torch.cat(res, dim=1)
    return self.project(res)
  

#DeeplabV3
class DeepLabV3(nn.Module):
  def __init__(self, num_classes=1, output_stride=16, pretrained=False):
      super(DeepLabV3, self).__init__()
      if output_stride == 16:
          atrous_rates = [6, 12, 18]
      elif output_stride == 8:
          atrous_rates = [12, 24, 36]
      else:
          raise NotImplementedError

      self.backbone = ResNet101(output_stride, pretrained)
      self.aspp = ASPP(2048, atrous_rates)
      self.out = nn.Sequential(
          nn.Conv2d(256,256,3,padding=1,bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256,num_classes,1)
      )

  def forward(self, x):
      input_shape = x.size()[-2:]
      x = self.backbone(x)
      x = self.aspp(x)
      x = self.out(x)
      x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

      return torch.sigmoid(x) 
    
if __name__ == "__main__":
  x1 = torch.randn([4,5,256,256]).cuda()
  net = DeepLabV3().cuda()
  out = net(x1)
  print(out.shape)