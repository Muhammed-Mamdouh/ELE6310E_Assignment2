"""
Modified from https://github.com/chenyaofo/pytorch-cifar-models
"""
import sys
import torch.nn as nn
import torch
import torch.nn.functional as F



class Conv2dWithChannelAttention(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, multiplier=0.5):
        super(Conv2dWithChannelAttention, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.attention = nn.Parameter(torch.rand(out_channels)/10+multiplier)
        # self.attention.data.fill_(multiplier)
        self.attention.requires_grad = True

    def forward(self, input):
        # Multiply weights by the attention
        modified_weights = self.weight * self.attention.view(-1, 1, 1, 1)

        # Compute output using F.conv2d
        output = F.conv2d(input, modified_weights, None, self.stride, self.padding, self.dilation, self.groups)

        # Add bias if it's not None
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output




try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

cifar10_pretrained_weight_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2dWithChannelAttention(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2dWithChannelAttention(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
        arch: str,
        layers: List[int],
        model_url: str,
        save_path: str = './save/',
        progress: bool = True,
        pretrained: bool = False,
        device=torch.device('cpu'),
        **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url, save_path,
                                              map_location=device, progress=progress)
        network_kvpair = model.state_dict()
        for key in state_dict.keys():
            network_kvpair[key] = state_dict[key]
        model.load_state_dict(network_kvpair)
    return model


def resnet32(pretrained=False, progress=True, device=torch.device('cpu'), **kwargs):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      device:
      progress:
      pretrained:
    """
    save_path = kwargs.pop('save_path')
    return _resnet("resnet32", [5, 5, 5], cifar10_pretrained_weight_url, save_path, progress, pretrained, device,
                   **kwargs)

