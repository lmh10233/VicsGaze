import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from orepa.dbb_transforms import transI_fusebn
from timm.models.layers import trunc_normal_, to_2tuple, DropPath, PatchEmbed
import math
import numpy as np
from orepa.blocks import OREPA, OREPA_1x1
import torch
import copy
from einops import rearrange


DEPLOY_FLAG = False


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type =  OREPA
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = OREPA
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_bn(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1,
                                    stride=stride, assign_type=ConvBN)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = conv_bn_relu(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv_bn(in_channels=planes, out_channels=self.expansion * planes, kernel_size=3, stride=1,
                             padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
#         out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
       
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv_bn(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = conv_bn_relu(in_planes, planes, kernel_size=1)
        self.conv2 = conv_bn_relu(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = conv_bn(planes, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1):
        super(ResNet, self).__init__()
        self.maps = 32
        self.in_planes = int(64 * width_multiplier)
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1',
                               conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2,
                                            padding=3, assign_type=ConvBN))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2)

        output_channels = int(512 * block.expansion * width_multiplier)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)        
        self.linear = nn.Linear(int(512*block.expansion*width_multiplier), num_classes)   

    def _make_stage(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def reparameterize_model(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def create_Res():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000, width_multiplier=1)
    # return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000, width_multiplier=1)


if __name__ == '__main__':
    model = create_Res()
    model.cuda()
    model = reparameterize_model(model)
    x = torch.rand(1, 3, 224, 224).cuda()
    from torchsummary import summary
    summary(model, (3, 224, 224))
    print(model)
    from thop import profile
    flops, params = profile(model, (x,))
    print('flops: %.2f B, params: %.2f M' % (flops / 1e9, params / 1e6))