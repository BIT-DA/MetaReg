import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from meta_layers import *

class BasicBlockMeta(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockMeta, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        #     )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out


class BottleneckMeta(MetaModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleneckMeta, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        #     )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)
        return out


class ResNetMeta(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10, bottleneck_dim=256):
        super(ResNetMeta, self).__init__()
        self.inplanes = 64

        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)

        self.bottleneck = nn.Sequential(
            MetaLinear(512*block.expansion, bottleneck_dim),
            MetaBatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.head = MetaLinear(bottleneck_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes*block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes*block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = F.avg_pool2d(x, 7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        y = self.head(x)

        return x, y

def resnet_meta18(num_class): return ResNetMeta(BasicBlockMeta, [2,2,2,2], num_class)
#def preact_resnet_meta2332(): return PreActResNetMeta(PreActBlockMeta, [2,3,3,2])
#def preact_resnet_meta3333(): return PreActResNetMeta(PreActBlockMeta, [3,3,3,3])
def resnet_meta34(num_class): return ResNetMeta(BasicBlockMeta, [3,4,6,3], num_class)
def resnet_meta50(num_class): return ResNetMeta(BottleneckMeta, [3,4,6,3], num_class)
def resnet_meta101(num_class): return ResNetMeta(BottleneckMeta, [3,4,23,3], num_class)
def resnet_meta152(num_class): return ResNetMeta(BottleneckMeta, [3,8,36,3])

meta_net_dict={'resnet18':resnet_meta18,
    'resnet34':resnet_meta34,
    'resnet50':resnet_meta50,
    'resnet101':resnet_meta101,
    'resnet152':resnet_meta152,}
