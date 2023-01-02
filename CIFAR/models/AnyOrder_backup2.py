import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron
import onnx
from thop import profile
from torchsummary import summary

from onnx import shape_inference
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from typing import Dict, Callable
import torchextractor as tx



global num_cla
num_cla = 10


class SamBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, stepsize, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
        super(SamBlock, self).__init__()
        self.steps = steps
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.pre_planes_b = pre_planes
        # self.stepsize = stepsize
        self.fix_coe = fix_coe
        # self.coesA = coesA
        # self.coesB = coesB
        self.start=1
        self.has_ini_block = False
        for pre_pl in self.pre_planes_b:
            if pre_pl <=0:
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block #wrong
        if not (self.has_ini_block):
            if self.in_planes != self.planes:
                # self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
                self.downsample_x_Fix = Downsample_Fix(self.in_planes, self.planes)
            start_DS = 1        
            for i in range(self.steps-1):
                if self.pre_planes_b[i] != self.planes:
                    if start_DS:
                        self.FixDS = nn.ModuleList([])
                        start_DS = 0
                    self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )
        # self.bn3 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn4 = nn.BatchNorm2d(planes)
        # self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
    def forward(self, x, coesA, coesB, stepsize):  

        # ## residual = pre_features[0]
        
        # # ori >
        # residual = x
        # F_x_n = self.bn1(x)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv1(F_x_n)
        # F_x_n = self.bn2(F_x_n)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv2(F_x_n)
        # # < ori
        
        # added >
        residual = x
        F_x_n = self.bn1(x)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)
        F_x_n = self.bn2(F_x_n)

        # > added relu
        
            
        if self.in_planes != self.planes:
            residual = self.downsample_x_Fix(residual)  

        if self.stride == 1:
            # SAM
            #print('SAM')
            k2 = residual - F_x_n
            k2 = self.bn1(k2)
            k2 = self.relu(k2)
            k2 = self.conv1(k2)
            k2 = self.relu(k2)
            k2 = self.conv2(k2)
            k2 = self.bn2(k2)
            #            
            x = residual + 1/2*F_x_n+1/2*k2
        else:
            x = residual + F_x_n

        return x #, pre_features, pre_acts, coesA, coesB
 
    
class ZeroSBlock(nn.Module): 
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, coe_ini=1, fix_coe=False, stepsize=1, given_coe=[1.0/3, 5.0/9, 1.0/9, 16.0/9], downsample=None):
        super(ZeroSBlock,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_coe = fix_coe
        if self.fix_coe:
            self.coe = coe_ini
            self.a_0 = float(given_coe[0])
            self.a_1 = float(given_coe[1])
            self.a_2 = float(given_coe[2])
            self.b_0 = float(given_coe[3])
        else:
            self.coe =nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))
        if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
            if self.in_planes !=  self.planes:
                self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
            if self.last_res_planes != self.planes:
                self.downsample_l = Downsample_clean(self.last_res_planes, self.planes, 2)
            if self.l_last_res_planes != self.planes:
                self.downsample_ll = Downsample_clean(self.l_last_res_planes, self.planes, 2)
                        
    def forward(self, x, last_res, l_last_res): 
        residual = x
        F_x_n=self.bn1(x)
        F_x_n=self.relu(F_x_n)
        F_x_n=self.conv1(F_x_n)
        F_x_n=self.bn2(F_x_n)
        F_x_n=self.relu(F_x_n)
        F_x_n=self.conv2(F_x_n)
        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
            if self.in_planes !=  self.planes:
                residual = self.downsample_x(residual)
            if self.last_res_planes != self.planes:
                last_res = self.downsample_l(last_res)
            if self.l_last_res_planes != self.planes:
                l_last_res = self.downsample_ll(l_last_res)



            if not self.fix_coe:
                self.b_0 = (3 * self.coe - 1) / (self.coe * 2)
                self.a_0 = (3 * self.coe + 3) / (self.coe * 4)
                self.a_1 = -1 / (self.coe)
                self.a_2 = (self.coe + 1) / (4 * self.coe)
            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
        else:
            x = F_x_n
        l_last_res = last_res
        last_res = residual 
        return x, last_res, l_last_res, self.coe
    
class ZeroSBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, coe_ini=1, fix_coe=False, stepsize=1, given_coe=[1.0/3, 5.0/9, 1.0/9, 16.0/9]):
        super(ZeroSBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = 4
        self.in_planes = in_planes
        self.planes = planes*self.expansion
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_coe = fix_coe
        if self.fix_coe:
            self.coe = coe_ini
            self.a_0 = float(given_coe[0])
            self.a_1 = float(given_coe[1])
            self.a_2 = float(given_coe[2])
            self.b_0 = float(given_coe[3])
        else:
            self.coe = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))
        if not (last_res_planes == -1 or l_last_res_planes == -1):
            
            if self.in_planes !=  self.planes:
                self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
            if self.last_res_planes != self.planes:
                self.downsample_l = Downsample_clean(self.last_res_planes, self.planes, 2)
            if self.l_last_res_planes != self.planes:
                self.downsample_ll = Downsample_clean(self.l_last_res_planes, self.planes, 2)

            if self.planes == 64:
                if self.in_planes ==16:
                    self.downsample_x = Downsample_clean(16, 64, 1)
                if self.last_res_planes ==16:
                    self.downsample_l = Downsample_clean(16, 64, 1)
                if self.l_last_res_planes == 16:
                    self.downsample_ll = Downsample_clean(16, 64, 1)

                    
    def forward(self, x, last_res, l_last_res):
        residual = x
        F_x_n = self.bn1(x)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)

        F_x_n = self.bn2(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)

        F_x_n = self.bn3(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv3(F_x_n)

        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
            if self.in_planes !=  self.planes:
                residual = self.downsample_x(residual)
            if self.last_res_planes != self.planes:
                last_res = self.downsample_l(last_res)
            if self.l_last_res_planes != self.planes:
                l_last_res = self.downsample_ll(l_last_res)
            if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
                if not self.fix_coe:
                    self.b_0 = (3 * self.coe - 1) / (self.coe * 2)
                    self.a_0 = (3 * self.coe + 3) / (self.coe * 4)
                    self.a_1 = -1 / (self.coe)
                    self.a_2 = (self.coe + 1) / (4 * self.coe)
                    
                x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)

        else:
            x = F_x_n
        l_last_res = last_res
        last_res = residual 
        return x, last_res, l_last_res, self.coe


class Downsample(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample,self).__init__()
        self.downsample=nn.Sequential(
                        nn.BatchNorm2d(in_planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample(x)
        return x

class Downsample_clean(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=2):
        super(Downsample_clean,self).__init__()
        self.downsample_=nn.Sequential(
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample_(x)
        return x
# class Downsample_Fix_AllEle00(nn.Module): 
#     def __init__(self,in_planes,out_planes,stride=2):
#         super(Downsample_Fix_AllEle00,self).__init__()
#         self.downsample_=nn.Sequential(
#                         nn.AvgPool2d(kernel_size=1, stride=stride, padding=(0,0,0,0), bias=False)
#                         )
#     def forward(self,x):
#         x=self.downsample_(x)
#         return x
# class Downsample_Fix_AllEle01(nn.Module): 
#     def __init__(self,in_planes,out_planes,stride=2):
#         super(Downsample_Fix_AllEle00,self).__init__()
#         self.downsample_=nn.Sequential(
#                         nn.AvgPool2d(kernel_size=1, stride=stride, padding=(-1,0,1,0), bias=False)
#                         )
#     def forward(self,x):
#         x=self.downsample_(x)
#         return x    
class Downsample_Fix(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=1):#stride=2):
        super(Downsample_Fix,self).__init__()
        self.downsample_=nn.Sequential(
                    nn.AvgPool2d(2),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample_(x)
        return x

class MaskConv(nn.Module): 
    def __init__(self,in_planes, out_planes, kernel_size, stride, padding, bias):
        super(MaskConv,self).__init__()
        self.mask_conv_=nn.Sequential(
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                        )        
        # self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
        # pre_features[i] = self.FixDS[-1](pre_features[i])
        # nn.init.dirac_(self.FixDS[-1].weight, 2)
        # self.FixDS[-1].weight.requires_grad = False
    def forward(self,x):
        x=self.mask_conv_(x)
        return x
    
def _downsample_All_Ele(x):
    out00 = F.pad(x, pad=(0,0,0,0), mode='constant', value=0)
    out01 = F.pad(x, pad=(-1,1,0,0), mode='constant', value=0)
    out10 = F.pad(x, pad=(0,0,-1,1), mode='constant', value=0)
    out11 = F.pad(x, pad=(-1,1,-1,1), mode='constant', value=0)

    out00 = F.avg_pool2d(out00, kernel_size=1, stride=2, padding=0)
    out01 = F.avg_pool2d(out01, kernel_size=1, stride=2, padding=0)
    out10 = F.avg_pool2d(out10, kernel_size=1, stride=2, padding=0)
    out11 = F.avg_pool2d(out11, kernel_size=1, stride=2, padding=0)
    x = torch.cat((out00, out01, out10, out11), dim=1)
    return x 
class Mask_(nn.Module): 
    def __init__(self,in_planes, out_planes, kernel_size, stride, padding, bias):
        super(Mask_,self).__init__()
        self.mask_=nn.Sequential(
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                        )        
        # self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
        # pre_features[i] = self.FixDS[-1](pre_features[i])
        # nn.init.dirac_(self.FixDS[-1].weight, 2)
        # self.FixDS[-1].weight.requires_grad = False
    def forward(self,x):
        x=self.mask_(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    # print('m_name', m)
    if classname.find('Fix') != -1:
        # print('classname', classname)
        # print('m', m)

        # p = m.downsample_
        nn.init.dirac_(m.downsample_[1].weight.data, 4)
    # if classname.find('MaskConv') != -1:
    #     # print('classname', classname)
    #     # print('m', m)
    #     nn.init.sparse_(m.mask_conv_[0].weight.data, sparsity=0.1)
        # nn.init.dirac_(m.downsample_[0].weight.data, 2)
        # nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Mask_') != -1:
        # print('classname', classname)
        # print('m', m)
        nn.init.sparse_(m.mask_[0].weight.data, sparsity=0.1)

def weights_init_plot(m):
    classname = m.__class__.__name__
    # print('m_name', m)
    if classname.find('Conv2d') != -1:

        nn.init.constant_(m.weight.data, 0.0001)

        # nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
         torch.nn.init.constant_(m.weight, 0.0001)
         torch.nn.init.constant_(m.bias, 0.0)

class ZeroSNet_Tra(nn.Module):

    def __init__(self, block, layers, coe_ini=1, share_coe=False, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(ZeroSNet_Tra, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.coe_ini= coe_ini
        # if self.share_coe == True:
            
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.coes = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i], coe_ini=self.coe_ini))
                if l ==0 or l==1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, coe_ini=self.coe_ini))
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1

        else:
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,self.strides[i], coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0]]))
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.coes = []
        x = self.conv1(x)
        last_res=-1
        l_last_res=-1

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        x += residual
        residual = x
        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        x += residual

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                continue
            residual = x 
            if self.pretrain:  #
                x = b(x) + residual

            else:  
                x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                self.coes += k.data
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.coes, 1
import math
import warnings

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter    
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
class LinearInv(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearInv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.div(1, self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
class Conv2dInv(_ConvNd):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2dInv, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)#, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, torch.div(1, self.weight), self.bias)

class ZeroSBlockAnyAblation(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, stepsize, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], ablation='', Layer_idx=0, Decay=1):        
        super(ZeroSBlockAnyAblation, self).__init__()
        self.ablation = ablation 
        self.drop_rate = None      
        if 'drop' in self.ablation:
            self.drop_rate = 0.2
      
        self.Layer_idx = Layer_idx
        if 'LearnBal' in self.ablation:
            self.Balance = nn.Parameter(torch.zeros(1))
        else:
            self.Balance = nn.Parameter(torch.zeros(1), requires_grad = False)
        self.Decay = Decay
        self.steps = steps
        if 'DSfirst' in self.ablation:    
            # print('DSfirst')   
            self.bn1 = nn.BatchNorm2d(planes)
            if 'InverseW' in self.ablation:
                self.conv1 = Conv2dInv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            else:                
                self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            if 'InverseW' in self.ablation:
                self.conv1 = Conv2dInv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            else:                
                self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(planes)
        if 'InverseW' in self.ablation:
            self.conv2 = Conv2dInv(planes, planes, kernel_size=3, padding=1, bias=False)
        else:        
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        if 'Swish' in self.ablation:
            self.relu = nn.SiLU(inplace=False)
        elif 'Mish' in self.ablation:
            self.relu = nn.Mish(inplace=False)            
        else:
            self.relu = nn.ReLU(inplace=False)
            
        self.identity = nn.Identity()

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.pre_planes_b = pre_planes
        # self.stepsize = stepsize
        self.fix_coe = fix_coe
        # self.coesA = coesA
        # self.coesB = coesB
        self.start=1
        self.has_ini_block = False
        for pre_pl in self.pre_planes_b:
            if pre_pl <=0:
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block #wrong
        if 'ConvStride2Fix' in self.ablation:
            print('ConvStride2Fix')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Fix = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.FixDS = nn.ModuleList([])
                            start_DS = 0
                        self.FixDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )            
        elif 'ConvStride2Learn' in self.ablation:
            # print('ConvStride2Learn')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.LearnDS = nn.ModuleList([])
                            start_DS = 0
                        self.LearnDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )
                        self.LearnDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )  
                        
        elif 'ConvStride2ResLikeShare' in self.ablation:
            # print('ConvStride2ResLikeShare')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample(self.in_planes, self.planes, 2)
                start_DS = 1        
                if self.pre_planes_b[0] != self.planes:
                    if start_DS:
                        self.LearnDS = nn.ModuleList([])
                        start_DS = 0
                    self.LearnDS.append( Downsample(self.pre_planes_b[0], self.planes) )
        elif 'ConvStride2ResLike' in self.ablation:        
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.LearnDS = nn.ModuleList([])
                            start_DS = 0
                        self.LearnDS.append( Downsample(self.pre_planes_b[i], self.planes) )
                        self.LearnDS.append( Downsample(self.pre_planes_b[i], self.planes) )                                                                            
        elif 'ConvStride2Share' in self.ablation:
            # print('ConvStride2Share')

            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Learn = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                # for i in range(self.steps-1):
                if self.pre_planes_b[0] != self.planes:
                    if start_DS:
                        self.LearnDS = nn.ModuleList([])
                        start_DS = 0
                    self.LearnDS.append( Downsample_clean(self.pre_planes_b[0], self.planes) )                         
        elif 'AllEle' in self.ablation:
            # print('AllEle')
            AllEle = 1
        else:    
            # print('DiracConvFix')
                    
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Fix = Downsample_Fix(self.in_planes, self.planes)
                start_DS = 1        
                # for i in range(self.steps-1):
                #     if self.pre_planes_b[i] != self.planes:
                #         if start_DS:
                #             self.FixDS = nn.ModuleList([])
                #             start_DS = 0
                #         self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )

    def forward(self, x, pre_features, pre_acts, coesA, coesB, stepsize):  
        
             
        # ## residual = pre_features[0]
        residual = x

        #             if 'ConvStride2Share' in self.ablation or 'ConvStride2ResLikeShare' in self.ablation:
        #                 if self.in_planes != self.planes:
        #                     residual = self.downsample_x_Learn(residual)  
        #                 for i in range(self.steps-1):
        #                     if pre_features[i].shape[1] != F_x_n.shape[1]:
        #                         pre_features[i] = self.LearnDS[0](pre_features[i])
        #                     if pre_acts[i].shape[1] != F_x_n.shape[1]:
        #                         pre_acts[i] = self.LearnDS[0](pre_acts[i])
        #             elif 'ConvStride2Learn' in self.ablation or 'ConvStride2ResLike' in self.ablation:
        #                 # print('ConvStride2Learn')
        #                 if self.in_planes != self.planes:
        #                     residual = self.downsample_x_Learn(residual)  
        #                 for i in range(self.steps-1):
        #                     if pre_features[i].shape[1] != F_x_n.shape[1]:
        #                         pre_features[i] = self.LearnDS[i](pre_features[i])
        #                     if pre_acts[i].shape[1] != F_x_n.shape[1]:
        #                         pre_acts[i] = self.LearnDS[self.steps-1+i](pre_acts[i])                                                
        #             elif 'AllEle' in self.ablation:    
        #                 # print('AllEle')
        #                 if self.in_planes != self.planes:
        #                     residual = _downsample_All_Ele(residual)  
        #                 for i in range(self.steps-1):
        #                     if pre_features[i].shape[1] != F_x_n.shape[1]:
        #                         pre_features[i] =  _downsample_All_Ele(pre_features[i])
        #                     if pre_acts[i].shape[1] != F_x_n.shape[1]:
        #                         pre_acts[i] =  _downsample_All_Ele(pre_acts[i])                 
        #             else:
        #                 if self.in_planes != self.planes:
        #                     residual = self.downsample_x_Fix(residual)  
        #                 for i in range(self.steps-1):
        #                     if pre_features[i].shape[1] != F_x_n.shape[1]:
        #                         pre_features[i] = self.downsample_x_Fix(pre_features[i])
        #                     if pre_acts[i].shape[1] != F_x_n.shape[1]:
        #                         pre_acts[i] = self.downsample_x_Fix(pre_acts[i])    
        # else:
        # # ori >
        # if 'InverseW' in self.ablation:
        #     self.conv1.weight.data = torch.div(1, self.conv1.weight.data)
        #     self.conv2.weight.data = torch.div(1, self.conv2.weight.data)
                    
        if not 'DSfirst' in self.ablation:        
            if 'BnReluConv' in self.ablation:
                F_x_n = self.bn1(x)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv1(F_x_n)
                F_x_n = self.bn2(F_x_n)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv2(F_x_n)
            elif 'BnReluConvBn' in self.ablation:
                F_x_n = self.bn1(x)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv1(F_x_n)
                F_x_n = self.bn2(F_x_n)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv2(F_x_n)
                F_x_n = self.bn3(F_x_n)            
            else:
                # added >
                F_x_n = self.bn1(x)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv1(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)               
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv2(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)              
                F_x_n = self.bn2(F_x_n)
                # > added relu

            # residual = x
            # F_x_n = self.bn1(x)
            # F_x_n = self.relu(F_x_n)
            # F_x_n = self.conv1(F_x_n)
            # F_x_n = self.bn2(F_x_n)
            # F_x_n = self.relu(F_x_n)
            # F_x_n = self.conv2(F_x_n)
            # # < ori
        
        self.has_ini_block = False
        for pre_fea in pre_features:
            if isinstance(pre_fea, int):
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block
            
        if not (self.has_ini_block):

            if 'ConvStride2Share' in self.ablation or 'ConvStride2ResLikeShare' in self.ablation:
                if self.in_planes != self.planes:
                    residual = self.downsample_x_Learn(residual)  
                    for i in range(self.steps-1):
                    # if self.in_planes != self.planes:                           
                    # if pre_features[i].shape[1] != F_x_n.shape[1]:
                        pre_features[i] = self.LearnDS[0](pre_features[i])
                    # if pre_acts[i].shape[1] != F_x_n.shape[1]:
                        pre_acts[i] = self.LearnDS[0](pre_acts[i])
            elif 'ConvStride2Learn' in self.ablation or 'ConvStride2ResLike' in self.ablation:
                # print('ConvStride2Learn')
                if self.in_planes != self.planes:
                    residual = self.downsample_x_Learn(residual)  
                    for i in range(self.steps-1):
                        # if pre_features[i].shape[1] != F_x_n.shape[1]:
                        pre_features[i] = self.LearnDS[i](pre_features[i])
                        # if pre_acts[i].shape[1] != F_x_n.shape[1]:
                        pre_acts[i] = self.LearnDS[self.steps-1+i](pre_acts[i])                                                
            elif 'AllEle' in self.ablation:    
                # print('AllEle')
                if self.in_planes != self.planes:
                    residual = _downsample_All_Ele(residual)  
                    for i in range(self.steps-1):
                        # if pre_features[i].shape[1] != F_x_n.shape[1]:
                        # if self.in_planes != self.planes:                         
                        pre_features[i] =  _downsample_All_Ele(pre_features[i])
                        # if pre_acts[i].shape[1] != F_x_n.shape[1]:
                        # if self.in_planes != self.planes:  
                        pre_acts[i] =  _downsample_All_Ele(pre_acts[i])                 
            else:
                if self.in_planes != self.planes:
                    residual = self.downsample_x_Fix(residual)  
                    for i in range(self.steps-1):
                    # if pre_features[i].shape[1] != F_x_n.shape[1]:
                        pre_features[i] = self.downsample_x_Fix(pre_features[i])
                    # if pre_acts[i].shape[1] != F_x_n.shape[1]:
                        pre_acts[i] = self.downsample_x_Fix(pre_acts[i])
                        
            if 'DSfirst' in self.ablation:      
                # print("==>> type(DSfirst): ")                 
                F_x_n = self.bn1(residual)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv1(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)               
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv2(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)              
                F_x_n = self.bn2(F_x_n)

                # for i in range(self.steps-1):
                #     if pre_features[i].shape[1] != F_x_n.shape[1]:
                #         pre_features[i] = self.FixDS[-1](pre_features[i])
                #     if pre_acts[i].shape[1] != F_x_n.shape[1]:
                #         pre_acts[i] = self.FixDS[-1](pre_acts[i])                        
            # if self.in_planes != self.planes:
            #     kernel = torch.empty(self.planes, self.in_planes, 1, 1).cuda()
            #     # print('self.planes/ self.in_planes', self.planes/self.in_planes)
            #     nn.init.dirac_(kernel, int(self.planes/self.in_planes)) #self.out_chs/self.in_chs
            #     residual = F.avg_pool2d(residual, kernel_size=self.stride )
            #     # print('shortcut', shortcut.shape)
            #     # print('kernel', kernel)
            #     residual = F.conv2d(residual, kernel, stride=1, padding='same')
            # for i in range(self.steps-1):
            #     if pre_features[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_features[i].shape[1], 1, 1).cuda()
            #         # print('F_x_n.shape[1]/pre_features[i].shape[1]', F_x_n.shape[1]/pre_features[i].shape[1])
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_features[i].shape[1])) #self.out_chs/self.in_chs
            #         pre_features[i] = F.avg_pool2d( pre_features[i], kernel_size=self.stride )
            #         # print('shortcut', shortcut.shape)
            #         # print('kernel', kernel)
            #         pre_features[i] = F.conv2d( pre_features[i], kernel, stride=1, padding='same')
            #     if pre_acts[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_acts[i].shape[1], 1, 1).cuda()
                    
            #         # print('int(F_x_n.shape[1]/pre_acts[i].shape[1])', int(F_x_n.shape[1]/pre_acts[i].shape[1]))
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_acts[i].shape[1])) #self.out_chs/self.in_chs
            #         pre_acts[i] = F.avg_pool2d(pre_acts[i], kernel_size=self.stride )
            #         # print('shortcut', shortcut.shape)
            #         # print('kernel', kernel)
            #         pre_acts[i] = F.conv2d(pre_acts[i], kernel, stride=1, padding='same')      

            if 'ExpDecay' in self.ablation:
                # print('self.Decay',self.Decay)
                F_x_n =  torch.mul((1-torch.sigmoid(self.Balance)), torch.exp(-self.Decay*self.Layer_idx*stepsize) * (F_x_n-self.Decay*residual) ) +torch.mul(torch.sigmoid(self.Balance), F_x_n)
                # print('self.Decay, stepsize, torch.sigmoid(self.Balance)', self.Decay, stepsize, torch.sigmoid(self.Balance)  )        
                # F_x_n = torch.mul((1-self.Decay), torch.exp(-stepsize*self.Layer_idx) * (F_x_n-residual) )+torch.mul(self.Decay,F_x_n)
                
                
                # F_x_n = torch.exp(-self.Decay*self.Layer_idx) * (F_x_n-residual)
                # print('self.Decay, self.Layer_idx', self.Decay.data, self.Layer_idx)            
            sum_features = coesA[0].expand_as(residual)*residual
            
            sum_acts = coesB[0].expand_as(F_x_n)*F_x_n
            # print('-coesB[0]', -coesB[0].data)
            
            for i in range(self.steps-1):
                sum_features = torch.add( sum_features, coesA[i+1].expand_as(pre_features[i])*pre_features[i] )
                
                sum_acts = torch.add( sum_acts, coesB[i+1].expand_as(pre_acts[i])*pre_acts[i] )  
            # if stepsize == -1:
            #     stepsize = 1      
            # x =  torch.add( sum_features, torch.mul(stepsize, sum_acts ) )
            x =  torch.add( sum_features, torch.mul(stepsize, -sum_acts ) )
        
            # x =  torch.add( sum_features, torch.mul(self.stepsize, self.coesA[-1].expand_as(F_x_n)*F_x_n) )

        else:
            if 'DSfirst' in self.ablation:       
                F_x_n = self.bn1(residual)
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv1(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)               
                F_x_n = self.relu(F_x_n)
                F_x_n = self.conv2(F_x_n)
                if self.drop_rate:
                    F_x_n = F.dropout(F_x_n, p=float(self.drop_rate), training=self.training)              
                F_x_n = self.bn2(F_x_n)            
            # #### x = F_x_n
            if 'ExpDecay' in self.ablation:
                F_x_n =  torch.mul((1-torch.sigmoid(self.Balance)), torch.exp(-self.Decay*self.Layer_idx*stepsize) * (F_x_n-self.Decay*residual) ) +torch.mul(torch.sigmoid(self.Balance), F_x_n)            
            x = F_x_n + residual
            
        for i in range(self.steps-2, 0, -1): #steps-2, steps-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1]

        pre_features[0] = residual
        pre_acts[0] = F_x_n
        x = self.identity(x)
        
        # print("self.coesA, self.coesA", self.coesA, self.coesB)
        # coes = self.coesA.extend(self.coesB)
        # print("self.coes", self.coes)

        # self.coes = torch.cat([self.coesA,self.coesB],1)
        return x, pre_features, pre_acts, coesA, coesB


class ZeroSAnyAblation(nn.Module):

    def __init__(self, block, layers, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False, ablation=''):
        self.ablation = ablation
        self.drop_rate  = None
        if 'dropout' in self.ablation:
            self.drop_rate = 0.2
        if 'Start8' in self.ablation or '8,16,32' in self.ablation:
            # print('Start8')
            self.in_planes = 8
        else:
            self.in_planes = 16
        if '2Chs' in self.ablation:
            self.planes = [16, 32, 64]
        elif 'Start8' in self.ablation:
            self.planes = [8, 32, 128]
        elif '8,16,32' in self.ablation:
            self.planes = [8, 16, 32]
        else:
            self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        steps = len(givenA)
        self.test = []
        for i in range(steps-1):
            self.pre_planes += [-i]
        for i in range(steps-1):
            self.test += [-i]

        self.strides = [1, 2, 2]
        super(ZeroSAnyAblation, self).__init__()
        self.block = block
        if 'mnist' in self.ablation:
            print('mnist')
            if 'InverseW' in self.ablation:
                self.conv1 = Conv2dInv(1, self.in_planes, kernel_size=3, padding=1, bias=False)
            else:            
                self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=3, padding=1, bias=False)
        else:
            if 'InverseW' in self.ablation:
                self.conv1 = Conv2dInv(3, self.in_planes, kernel_size=3, padding=1, bias=False)   
                # self.conv1 = Conv2dInv(3, self.in_planes, kernel_size=3, padding='same', bias=False)            
                self.convDS = Conv2dInv(3, self.in_planes, kernel_size=1, padding=1, bias=False)  
            else:             
                self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False)   
                # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding='same', bias=False)            
                self.convDS = nn.Conv2d(3, self.in_planes, kernel_size=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(16)        
        # if 'mnist' in self.ablation:
        #     print('mnist')
        #     self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding='same', bias=False)
        # else:
        #     self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)            
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.share_coe = share_coe
        self.noise_level = noise_level
        self.PL = PL
        self.noise = noise
        self.steps = steps
        self.coe_ini= coe_ini
        # self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB
        # self.givenA = [1.0/3, 5.0/9, 1.0/9] 
        # self.givenB = [16.0/9, 0, 0] #ConverOrder=3
        # self.givenB = [8.0/9, 8.0/9, 0] #ConverOrder=2
        # self.givenB = [16.0/27, 16.0/27, 16.0/27] #ConverOrder=2
        # self.givenB = [16.0/27, 16.0/27, -16.0/27] #ConverOrder=1

        
        ### AB 3
        # self.givenA = [1, 0, 0]
        # self.givenB = [23/12, -16/12, 5/12]
        
        # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
        # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
        # self.given = [-0.380903884996629, 1.16875181181267, 0.415624094093667, -0.203472020909704, 2.38958393727089]
        # self.given = [0.403523597279250, 0.475777922837489, 0.112412848357367, 0.00828563152589346, 1.72546051412990]
        # self.given = [0.380905880882445, -1.16875507780764, -0.415622461096181, 0.203471658021374, -2.38958502593588]
        # self.given = [0.165137619606463, -0.914615956838281, -0.375261502764746, 0.0828822102925340, 0.0418576297040286, -2.24906181267406]
        # self.given = [-0.165137619606463, 0.914615956838281, 0.375261502764746, -0.0828822102925340, -0.0418576297040286, 2.24906181267406]
        # self.given = [0, 1, 2]
        # self.givenA = [0, 1]
        # COnverOrder 2
        # self.givenB = [2, 0]
        # self.givenB = [1, 1]


        # self.given = [1, 0, 0, 1.0/2]

        # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.steps+1)]) 
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 

        # if self.share_coe == True:
        if 'Step0p1' in self.ablation:
            ini_stepsize = 0.1
        if 'LearnStepSize'in self.ablation:
            self.ini_stepsize =nn.Parameter(torch.Tensor(1).uniform_(ini_stepsize, ini_stepsize))
        else:
            self.ini_stepsize = ini_stepsize
        if 'ExpDecay' in self.ablation and 'LearnDecay' in self.ablation:            
            self.Decay =nn.Parameter(torch.ones(1)*1)
        else:
            self.Decay = nn.Parameter(torch.ones(1)*1, requires_grad=False)
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                Layer_idx = nn.Parameter(torch.ones(1)*l, requires_grad=False)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini, ablation=self.ablation,Layer_idx=Layer_idx, Decay=self.Decay))

                if l < steps-1:
                    for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(steps-2,0,-1): # steps-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):
                    Layer_idx = nn.Parameter(torch.ones(1)*l, requires_grad=False)

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, steps=self.steps, coe_ini=self.coe_ini, ablation=self.ablation,Layer_idx=Layer_idx, Decay=self.Decay))

                    if l < steps-1:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.planes[i] * block.expansion
                        self.pre_planes[0] = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        self.blocks = nn.ModuleList(blocks)
        # self.downsample1 = Downsample(16, 64, stride=1)
        self.downsample1 = Downsample(self.in_planes, 64, stride=1)
        
        if '2Chs' in self.ablation:
            self.bn = nn.BatchNorm2d(64 * block.expansion)
        elif 'Start8' in self.ablation:
            self.bn = nn.BatchNorm2d(128 * block.expansion)
        elif '8,16,32' in self.ablation:
            self.bn = nn.BatchNorm2d(32 * block.expansion)
        else:
            self.bn = nn.BatchNorm2d(256 * block.expansion)

        self.avgpool = nn.AvgPool2d(8)
        if 'InverseW' in self.ablation:
            if '2Chs' in self.ablation:
                self.fc = LinearInv(64 * block.expansion, num_classes)
            elif 'Start8' in self.ablation:
                self.fc = LinearInv(128 * block.expansion, num_classes)
            elif '8,16,32' in self.ablation:
                self.fc = LinearInv(32 * block.expansion, num_classes)
            else:        
                self.fc = LinearInv(256 * block.expansion, num_classes)
        else:
            if '2Chs' in self.ablation:
                self.fc = nn.Linear(64 * block.expansion, num_classes)
            elif 'Start8' in self.ablation:
                self.fc = nn.Linear(128 * block.expansion, num_classes)
            elif '8,16,32' in self.ablation:
                self.fc = nn.Linear(32 * block.expansion, num_classes)
            else:        
                self.fc = nn.Linear(256 * block.expansion, num_classes)            
        self.identityStart0 = nn.Identity()
        self.identityStart1 = nn.Identity()
        self.identityEndM1 = nn.Identity()
        self.identityEnd = nn.Identity()

        for m in self.modules():  # initialization
            
            # print('m', torch.nn.Module.load_state_dict( m ) )
            # print('Name', name)
            # if 'Fix' in m:
            #     m.requires_grad = False
            #     print('FIX')
            if 'PlotTraj' in self.ablation:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.fill_(0.01)
                    
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(0.01)
                    m.bias.data.zero_()
            else:              
                if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dInv):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            if 'InverseW' in self.ablation and (isinstance(m, Conv2dInv) or isinstance(m, LinearInv)):
                m.weight.data = torch.div(1, m.weight.data)
                           
    def forward(self, x):
        # print('x.shape', x.shape)
        if 'mnist' in self.ablation:
            # print('mnist_f')
            x = F.pad(x, pad=(2,2,2,2), mode='constant', value=0)
        # print('x.shape', x.shape)
        x = self.identityStart0(x)
        if not 'mnist' in self.ablation:
            residual = self.convDS(x)
            # if 'InverseW' in self.ablation:
            #     self.convDS.weight.data = torch.div(1, self.convDS.weight.data)
        else:
            residual = x
        # if 'InverseW' in self.ablation:
            # self.conv1.weight.data = torch.div(1, self.conv1.weight.data)
            # print('self.conv1.weight.data', self.conv1.weight.data.mean())            
        x = self.conv1(x)
        if 'startRes' in self.ablation:
            # print('startRes')
            x += residual
        x = self.identityStart1(x)
        pre_features = []
        pre_acts = []
        for i in range(self.steps-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        # traj = [float(x.mean().data)]
        for j in range(self.steps-1):
            x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
            # x += residual
            # residual = x
            # print('x.shape:', x.shape)
            # traj = traj+[float(x.mean().data)]

        for i, b in enumerate(self.blocks):  # index and content
            if i < self.steps-1:
                continue
            residual = x 
            # if self.pretrain:  #
            #     x = b(x) + residual

            # else:  
            x, pre_features, pre_acts, coesA, coesB = b(x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            # traj = traj+[float(x.mean().data)]
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.identityEndM1(x)
        # if 'InverseW' in self.ablation:
        #     self.fc.weight.data = torch.div(1, self.fc.weight.data)        
        x = self.fc(x)
        x = self.identityEnd(x)
        # save_path = '/media/bdc/clm/OverThreeOrders/CIFAR/runs/mnist/trajectory/'
        # df = pd.DataFrame()

        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # file_name = self.ablation+'.csv'
        # df.to_csv(save_path+file_name)
        
        # print('traj', traj)
        return x#, coesA, 1

class ZeroSBlockAny(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, stepsize, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], ablation=''):        
        super(ZeroSBlockAny, self).__init__()
        self.ablation = ablation 
        self.steps = steps
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.pre_planes_b = pre_planes
        # self.stepsize = stepsize
        self.fix_coe = fix_coe
        # self.coesA = coesA
        # self.coesB = coesB
        self.start=1
        self.has_ini_block = False
        for pre_pl in self.pre_planes_b:
            if pre_pl <=0:
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block #wrong
        if 'ConvStride2Learn' in self.ablation:
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.FixDS = nn.ModuleList([])
                            start_DS = 0
                        self.FixDS.append( Downsample_clean(self.pre_planes_b[i], self.planes) )            

        else:            
            if not (self.has_ini_block):
                if self.in_planes != self.planes:
                    self.downsample_x_Fix = Downsample_Fix(self.in_planes, self.planes)
                start_DS = 1        
                for i in range(self.steps-1):
                    if self.pre_planes_b[i] != self.planes:
                        if start_DS:
                            self.FixDS = nn.ModuleList([])
                            start_DS = 0
                        self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )

    def forward(self, x, pre_features, pre_acts, coesA, coesB, stepsize):  

        # ## residual = pre_features[0]
        residual = x
        
        # # ori >
        if 'BnReluConv' in self.ablation:
            F_x_n = self.bn1(x)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv1(F_x_n)
            F_x_n = self.bn2(F_x_n)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv2(F_x_n)
        elif 'BnReluConvBn' in self.ablation:
            F_x_n = self.bn1(x)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv1(F_x_n)
            F_x_n = self.bn2(F_x_n)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv2(F_x_n)
            F_x_n = self.bn3(F_x_n)
        else:
            # added >
            F_x_n = self.bn1(x)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv1(F_x_n)
            F_x_n = self.relu(F_x_n)
            F_x_n = self.conv2(F_x_n)
            F_x_n = self.bn2(F_x_n)
            # > added relu
        # residual = x
        # F_x_n = self.bn1(x)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv1(F_x_n)
        # F_x_n = self.bn2(F_x_n)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv2(F_x_n)
        # # < ori
        

        
        self.has_ini_block = False
        for pre_fea in pre_features:
            if isinstance(pre_fea, int):
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block
            
        if not (self.has_ini_block):
            if self.in_planes != self.planes:
                residual = self.downsample_x_Fix(residual)  
            for i in range(self.steps-1):
                if pre_features[i].shape[1] != F_x_n.shape[1]:
                    pre_features[i] = self.FixDS[-1](pre_features[i])
                if pre_acts[i].shape[1] != F_x_n.shape[1]:
                    pre_acts[i] = self.FixDS[-1](pre_acts[i])

            # if self.in_planes != self.planes:
            #     kernel = torch.empty(self.planes, self.in_planes, 1, 1).cuda()
            #     # print('self.planes/ self.in_planes', self.planes/self.in_planes)
            #     nn.init.dirac_(kernel, int(self.planes/self.in_planes)) #self.out_chs/self.in_chs
            #     residual = F.avg_pool2d(residual, kernel_size=self.stride )
            #     # print('shortcut', shortcut.shape)
            #     # print('kernel', kernel)
            #     residual = F.conv2d(residual, kernel, stride=1, padding='same')
            # for i in range(self.steps-1):
            #     if pre_features[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_features[i].shape[1], 1, 1).cuda()
            #         # print('F_x_n.shape[1]/pre_features[i].shape[1]', F_x_n.shape[1]/pre_features[i].shape[1])
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_features[i].shape[1])) #self.out_chs/self.in_chs
            #         pre_features[i] = F.avg_pool2d( pre_features[i], kernel_size=self.stride )
            #         # print('shortcut', shortcut.shape)
            #         # print('kernel', kernel)
            #         pre_features[i] = F.conv2d( pre_features[i], kernel, stride=1, padding='same')
            #     if pre_acts[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_acts[i].shape[1], 1, 1).cuda()
                    
            #         # print('int(F_x_n.shape[1]/pre_acts[i].shape[1])', int(F_x_n.shape[1]/pre_acts[i].shape[1]))
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_acts[i].shape[1])) #self.out_chs/self.in_chs
            #         pre_acts[i] = F.avg_pool2d(pre_acts[i], kernel_size=self.stride )
            #         # print('shortcut', shortcut.shape)
            #         # print('kernel', kernel)
            #         pre_acts[i] = F.conv2d(pre_acts[i], kernel, stride=1, padding='same')      
                    
                                 
            sum_features = torch.mul(coesA[0], residual )
            
            sum_acts = torch.mul(coesB[0], F_x_n) #coesB[0].expand_as(F_x_n)*F_x_n
            
            for i in range(self.steps-1):
                sum_features = torch.add( sum_features, torch.mul(coesA[i+1], pre_features[i]) )# coesA[i+1].expand_as(pre_features[i])*pre_features[i] )
                
                sum_acts = torch.add( sum_acts, torch.mul(coesB[i+1], pre_acts[i]))#coesB[i+1].expand_as(pre_acts[i])*pre_acts[i] )  
                  
            x =  torch.add( sum_features, torch.mul(stepsize, -sum_acts ) )
        
            # x =  torch.add( sum_features, torch.mul(self.stepsize, self.coesA[-1].expand_as(F_x_n)*F_x_n) )

        else:
            x = F_x_n + residual
        for i in range(self.steps-2, 0, -1): #steps-2, steps-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1] 

        pre_features[0] = residual
        pre_acts[0] = F_x_n
        # print("self.coesA, self.coesA", self.coesA, self.coesB)
        # coes = self.coesA.extend(self.coesB)
        # print("self.coes", self.coes)

        # self.coes = torch.cat([self.coesA,self.coesB],1)
        return x, pre_features, pre_acts, coesA, coesB

class ZeroSAny_Tra(nn.Module):

    def __init__(self, block, layers, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False, ablation=''):
        self.in_planes = 16
        self.ablation = ablation
        if '2Chs' in self.ablation:
            self.planes = [16, 32, 64]
        else:
            self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        steps = len(givenA)
        self.test = []
        for i in range(steps-1):
            self.pre_planes += [-i]
        for i in range(steps-1):
            self.test += [-i]

        self.strides = [1, 2, 2]
        super(ZeroSAny_Tra, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.share_coe = share_coe
        self.noise_level = noise_level
        self.PL = PL
        self.noise = noise
        self.steps = steps
        self.coe_ini= coe_ini
        self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB
        # self.givenA = [1.0/3, 5.0/9, 1.0/9] 
        # self.givenB = [16.0/9, 0, 0] #ConverOrder=3
        # self.givenB = [8.0/9, 8.0/9, 0] #ConverOrder=2
        # self.givenB = [16.0/27, 16.0/27, 16.0/27] #ConverOrder=2
        # self.givenB = [16.0/27, 16.0/27, -16.0/27] #ConverOrder=1

        
        ### AB 3
        # self.givenA = [1, 0, 0]
        # self.givenB = [23/12, -16/12, 5/12]
        
        # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
        # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
        # self.given = [-0.380903884996629, 1.16875181181267, 0.415624094093667, -0.203472020909704, 2.38958393727089]
        # self.given = [0.403523597279250, 0.475777922837489, 0.112412848357367, 0.00828563152589346, 1.72546051412990]
        # self.given = [0.380905880882445, -1.16875507780764, -0.415622461096181, 0.203471658021374, -2.38958502593588]
        # self.given = [0.165137619606463, -0.914615956838281, -0.375261502764746, 0.0828822102925340, 0.0418576297040286, -2.24906181267406]
        # self.given = [-0.165137619606463, 0.914615956838281, 0.375261502764746, -0.0828822102925340, -0.0418576297040286, 2.24906181267406]
        # self.given = [0, 1, 2]
        # self.givenA = [0, 1]
        # COnverOrder 2
        # self.givenB = [2, 0]
        # self.givenB = [1, 1]


        # self.given = [1, 0, 0, 1.0/2]

        # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.steps+1)]) 
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 

        # if self.share_coe == True:
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini, ablation=self.ablation))

                if l < steps-1:
                    for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(steps-2,0,-1): # steps-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, steps=self.steps, coe_ini=self.coe_ini, ablation=self.ablation))

                    if l < steps-1:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.planes[i] * block.expansion
                        self.pre_planes[0] = self.planes[i] * block.expansion
                    l += 1

        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        if '2Chs' in self.ablation:
            self.bn = nn.BatchNorm2d(64 * block.expansion)
        else:
            self.bn = nn.BatchNorm2d(256 * block.expansion)

        self.avgpool = nn.AvgPool2d(8)
        if '2Chs' in self.ablation:
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:        
            self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            
            # print('m', torch.nn.Module.load_state_dict( m ) )
            # print('Name', name)
            # if 'Fix' in m:
            #     m.requires_grad = False
            #     print('FIX')
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
                           
    def forward(self, x):
                                
        x = self.conv1(x)

        pre_features = []
        pre_acts = []
        for i in range(self.steps-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x


        for j in range(self.steps-1):
            x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
            # x += residual
            # residual = x
            # print('x.shape:', x.shape)


        for i, b in enumerate(self.blocks):  # index and content
            if i < self.steps-1:
                continue
            residual = x 
            # if self.pretrain:  #
            #     x = b(x) + residual

            # else:  
            x, pre_features, pre_acts, coesA, coesB = b(x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
        # print('self.coesB', self.coesB[0].data)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # coesA.extend(coesB)
    
        return x#, coesA, 1

class SamNet(nn.Module):

    def __init__(self, block, layers, steps=3, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        # self.planes = [16, 32, 64]
        self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        self.test = []
        for i in range(steps-1):
            self.pre_planes += [-i]
        for i in range(steps-1):
            self.test += [-i]

        self.strides = [1, 2, 2]
        super(SamNet, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.share_coe = share_coe
        self.noise_level = noise_level
        self.PL = PL
        self.noise = noise
        self.steps = steps
        self.coe_ini= coe_ini
        self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB

        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 

        # if self.share_coe == True:
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini))

                if l < steps-1:
                    for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(steps-2,0,-1): # steps-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, steps=self.steps, coe_ini=self.coe_ini))

                    if l < steps-1:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.planes[i] * block.expansion
                        self.pre_planes[0] = self.planes[i] * block.expansion
                    l += 1

        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.bn = nn.BatchNorm2d(256 * block.expansion)

        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
                           
    def forward(self, x):
                                
        x = self.conv1(x)

        pre_features = []
        pre_acts = []
        for i in range(self.steps-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x


        # for j in range(self.steps-1):
        #     x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
        #     x += residual
        #     residual = x
        #     # print('x.shape:', x.shape)


        for i, b in enumerate(self.blocks):  # index and content
            # if i < self.steps-1:
            #     continue
            # residual = x 
            # if self.pretrain:  #
            #     x = b(x) + residual

            # else:  
            x = b(x, self.coesA, self.coesB, self.ini_stepsize)
        # print('self.coesB', self.coesB[0].data)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # coesA.extend(coesB)
    
        return x# , coesA, 1

class MaskBlock(nn.Module):
    expansion = 1
    def __init__(self, input_size, in_planes, planes, pre_planes, coesA, coesB, stepsize, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
        super(MaskBlock, self).__init__()
        self.steps = steps
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.pre_planes_b = pre_planes
        # self.stepsize = stepsize
        self.fix_coe = fix_coe
        # self.coesA = coesA
        # self.coesB = coesB
        self.start=1
        self.has_ini_block = False
        for pre_pl in self.pre_planes_b:
            if pre_pl <=0:
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block #wrong
        if not (self.has_ini_block):
            if self.in_planes != self.planes:
                # self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
                self.downsample_x_Fix = Downsample_Fix(self.in_planes, self.planes)
            start_DS = 1        
            for i in range(self.steps-1):
                if self.pre_planes_b[i] != self.planes:
                    if start_DS:
                        self.FixDS = nn.ModuleList([])
                        start_DS = 0
                    self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )

        # self.mask = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.conv_mask = MaskConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_mask = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask = nn.Parameter(torch.randn(input_size*stride, input_size*stride), requires_grad=True)

    def forward(self, x, pre_features, pre_acts, coesA, coesB, stepsize):  

        # ## residual = pre_features[0]
        
        # # ori >
        # residual = x
        # F_x_n = self.bn1(x)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv1(F_x_n)
        # F_x_n = self.bn2(F_x_n)
        # F_x_n = self.relu(F_x_n)
        # F_x_n = self.conv2(F_x_n)
        # # < ori
        
        # added >
        # print('x.shape1 = ', x.shape)
        
        x = self.conv_mask(x)#.to(torch.bool)
        self.mask.data = self.relu(self.mask)
        # print('mask portion  =', self.mask.data.to(torch.bool).sum()/self.mask.data.size()[0]**2 )
        # print('self.mask.shape', self.mask.shape)
        x= torch.mul(x, self.mask.to(torch.bool))     
        # print('x.shape2 = ', x.shape)
   
        residual = x
        x = self.bn1(x)
        F_x_n = self.relu(x)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)
        F_x_n = self.bn2(F_x_n)

        # print(' Mask' ,  Mask.sum())
        # F_x_n = torch.mul(F_x_n, Mask)
        # > added relu
        
        self.has_ini_block = False
        for pre_fea in pre_features:
            if isinstance(pre_fea, int):
                self.is_ini = True
            else:
                self.is_ini = False
            self.has_ini_block = self.is_ini or self.has_ini_block
            
        if not (self.has_ini_block):
            if self.in_planes != self.planes:
                residual = self.downsample_x_Fix(residual)  
            for i in range(self.steps-1):
                if pre_features[i].shape[1] != F_x_n.shape[1]:
                    pre_features[i] = self.FixDS[-1](pre_features[i])
                if pre_acts[i].shape[1] != F_x_n.shape[1]:
                    pre_acts[i] = self.FixDS[-1](pre_acts[i])
                  
                                 
            sum_features = coesA[0].expand_as(residual)*residual
            
            sum_acts = coesB[0].expand_as(F_x_n)*F_x_n
            
            for i in range(self.steps-1):
                sum_features = torch.add( sum_features, coesA[i+1].expand_as(pre_features[i])*pre_features[i] )
                
                sum_acts = torch.add( sum_acts, coesB[i+1].expand_as(pre_acts[i])*pre_acts[i] )  
                  
            x =  torch.add( sum_features, torch.mul(stepsize, -sum_acts ) )
        
            # x =  torch.add( sum_features, torch.mul(self.stepsize, self.coesA[-1].expand_as(F_x_n)*F_x_n) )

        else:
            x = F_x_n
        for i in range(self.steps-2, 0, -1): #steps-2, steps-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1] 

        pre_features[0] = residual
        pre_acts[0] = F_x_n
        # print("self.coesA, self.coesA", self.coesA, self.coesB)
        # coes = self.coesA.extend(self.coesB)
        # print("self.coes", self.coes)

        # self.coes = torch.cat([self.coesA,self.coesB],1)
        return x, pre_features, pre_acts, coesA, coesB

class MaskNet(nn.Module):

    def __init__(self, block, layers, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        # self.planes = [16, 32, 64]
        self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        steps = len(givenA)
        self.test = []
        for i in range(steps-1):
            self.pre_planes += [-i]
        for i in range(steps-1):
            self.test += [-i]

        self.strides = [1, 2, 2]
        super(MaskNet, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.share_coe = share_coe
        self.noise_level = noise_level
        self.PL = PL
        self.noise = noise
        self.steps = steps
        self.coe_ini= coe_ini
        self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB

        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.steps)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.steps)]) 

        # if self.share_coe == True:
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0

        self.input_size = [32,16,8]
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                blocks.append(block(self.input_size[i], self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini))
                if l < steps-1:
                    for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(steps-2,0,-1): # steps-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):

                    blocks.append(block(self.input_size[i], self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, steps=self.steps, coe_ini=self.coe_ini))

                    if l < steps-1:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
                            self.pre_planes[j] =  self.planes[i] * block.expansion
                        self.pre_planes[0] = self.planes[i] * block.expansion
                    l += 1

        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.bn = nn.BatchNorm2d(256 * block.expansion)

        self.avgpool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            
            if isinstance(m, nn.Conv2d):
                # print('m', m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
                           
    def forward(self, x):
                                
        x = self.conv1(x)

        pre_features = []
        pre_acts = []
        for i in range(self.steps-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x


        for j in range(self.steps-1):
            x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
            x += residual
            residual = x

        for i, b in enumerate(self.blocks):  # index and content
            if i < self.steps-1:
                continue
            residual = x 

            x, pre_features, pre_acts, coesA, coesB = b(x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # coesA.extend(coesB)
    
        return x#, coesA, 1



class ZeroSNet_Opt(nn.Module):

    def __init__(self, block, layers, coe_ini=-9.0 / 5, fix_coe=True, given_coe=[1.0/3, 5.0/9, 1.0/9, 16.0/9], pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001, noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(ZeroSNet_Opt, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.coe_ini = coe_ini
        self.fix_coe = fix_coe
        self.given_coe = given_coe
        # self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        self.stepsize = 1
        blocks = []
        self.coes = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          coe_ini=self.coe_ini, stepsize=self.stepsize, fix_coe=self.fix_coe, given_coe=self.given_coe))

                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for j in range(1, layers[
                    i]):  
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        coe_ini=self.coe_ini, stepsize=self.stepsize, fix_coe=self.fix_coe, given_coe=self.given_coe)) 
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
        else:  
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          coe_ini=self.coe_ini, fix_coe=self.fix_coe, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]], given_coe=self.given_coe))  
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        coe_ini=self.coe_ini, fix_coe=self.fix_coe, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j], given_coe=self.given_coe))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  
        self.downsample1 = Downsample(16, 64, stride=1)  
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  
            if isinstance(m, nn.Conv2d):  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.coes = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1

        if self.block.expansion == 4:  
            residual = self.downsample1(x)
        else:
            residual = x

        x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        x += residual
        residual = x

        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        x += residual

        for i, b in enumerate(self.blocks):  
            if i == 0 or i == 1:
                continue
            residual = x 
            if self.pretrain:  #
                x = b(x) + residual
            else:  
                x, last_res, l_last_res, k = b(x, last_res, l_last_res)
                if isinstance(k, float):
                    self.coes = k
                else:
                    self.coes += k.data
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x, self.coes, self.stepsize

def ZeroSAny20Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [3,3,3], **kwargs)
def ZeroSAny32Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [5,5,5], **kwargs)
def ZeroSAny44Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation,  [7,7,7], **kwargs)
def ZeroSAny56Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [9,9,9], **kwargs)
def ZeroSAny68Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [11,11,11], **kwargs)
def ZeroSAny80Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [13,13,13], **kwargs)
def ZeroSAny92Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [15,15,15], **kwargs)
def ZeroSAny104Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [17,17,17], **kwargs)
def ZeroSAny110Ablation(**kwargs):
    return ZeroSAnyAblation(ZeroSBlockAnyAblation, [18,18,18], **kwargs)

def MaskNet20(**kwargs):
    return MaskNet(MaskBlock, [3, 3, 3], **kwargs)
def MaskNet32(**kwargs):
    return MaskNet(MaskBlock, [5,5,5], **kwargs)

def SamNet20_Tra(**kwargs):
    return SamNet(SamBlock, [3, 3, 3], **kwargs)
def SamNet32_Tra(**kwargs):
    return SamNet(SamBlock, [5,5,5], **kwargs)
def SamNet44_Tra(**kwargs):
    return SamNet(SamBlock, [7,7,7], **kwargs)
def SamNet56_Tra(**kwargs):
    return SamNet(SamBlock, [9,9,9], **kwargs)
def SamNet68_Tra(**kwargs):
    return SamNet(SamBlock, [11,11,11], **kwargs)
def SamNet80_Tra(**kwargs):
    return SamNet(SamBlock, [13,13,13], **kwargs)
def SamNet92_Tra(**kwargs):
    return SamNet(SamBlock, [15,15,15], **kwargs)
def SamNet104_Tra(**kwargs):
    return SamNet(SamBlock, [17,17,17], **kwargs)
def SamNet110_Tra(**kwargs):
    return SamNet(SamBlock, [18,18,18], **kwargs)

def ZeroSAny20_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [3, 3, 3], **kwargs)
def ZeroSAny32_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [5,5,5], **kwargs)
def ZeroSAny44_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [7,7,7], **kwargs)
def ZeroSAny56_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [9,9,9], **kwargs)
def ZeroSAny68_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [11,11,11], **kwargs)
def ZeroSAny80_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [13,13,13], **kwargs)
def ZeroSAny92_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [15,15,15], **kwargs)
def ZeroSAny104_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [17,17,17], **kwargs)
def ZeroSAny110_Tra(**kwargs):
    return ZeroSAny_Tra(ZeroSBlockAny, [18,18,18], **kwargs)
    
# def ZeroSAny164_Tra(**kwargs):
#     return ZeroSNet_Tra(ZeroSBottleneckAny, [18,18,18], **kwargs)
# def ZeroSAny326_Tra(**kwargs):
#     return ZeroSNet_Tra(ZeroSBottleneckAny, [36,36,36], **kwargs) #36*3*3+2
# def ZeroSAny650_Tra(**kwargs):
#     return ZeroSNet_Tra(ZeroSBottleneckAny, [72,72,72], **kwargs) #72*3*3+2
# def ZeroSAny1298_Tra(**kwargs):
#     return ZeroSNet_Tra(ZeroSBottleneckAny, [144,144,144], **kwargs)

def ZeroSNet20_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [3, 3, 3], **kwargs)
def ZeroSNet32_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [5,5,5], **kwargs)
def ZeroSNet44_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [7,7,7], **kwargs)
def ZeroSNet56_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [9,9,9], **kwargs)
def ZeroSNet110_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [18,18,18], **kwargs)
def ZeroSNet164_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [18,18,18], **kwargs)
def ZeroSNet326_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [36,36,36], **kwargs) #36*3*3+2
def ZeroSNet650_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [72,72,72], **kwargs) #72*3*3+2
def ZeroSNet1298_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [144,144,144], **kwargs) #144*3*3+2

def ZeroSNet20_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [3, 3, 3], **kwargs)
def ZeroSNet32_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [5,5,5], **kwargs)
def ZeroSNet44_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [7,7,7], **kwargs)
def ZeroSNet56_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [9,9,9], **kwargs)
def ZeroSNet110_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBlock, [18,18,18], **kwargs)
def ZeroSNet164_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [18,18,18], **kwargs)
def ZeroSNet326_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [36,36,36], **kwargs) #36*3*3+2
def ZeroSNet650_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [72,72,72], **kwargs) #72*3*3+2
def ZeroSNet1298_Tra(**kwargs):
    return ZeroSNet_Tra(ZeroSBottleneck, [144,144,144], **kwargs) #144*3*3+2

def ZeroSNet20_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBlock, [3, 3, 3], **kwargs)
def ZeroSNet32_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBlock, [5,5,5], **kwargs)
def ZeroSNet44_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBlock, [7,7,7],  **kwargs)
def ZeroSNet56_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBlock, [9,9,9], **kwargs)
def ZeroSNet110_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBlock, [18,18,18],  **kwargs)
def ZeroSNet164_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBottleneck, [18,18,18], **kwargs)
def ZeroSNet326_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBottleneck, [36,36,36], **kwargs) #36*3*3+2
def ZeroSNet650_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBottleneck, [72,72,72], **kwargs) #72*3*3+2
def ZeroSNet1298_Opt(**kwargs):
    return ZeroSNet_Opt(ZeroSBottleneck, [144,144,144], **kwargs) #144*3*3+2

# class ZeroSAny_Tra(nn.Module):

#     def __init__(self, block, layers, steps=3, coe_ini=1, share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
#                  noise=False):
#         self.in_planes = 16
#         self.planes = [16, 32, 64]
#         # self.last_res_planes = -1
#         # self.l_last_res_planes = -1
#         self.pre_planes = [] # the smaller, the closer to the current layer
#         self.test = []
#         for i in range(steps-1):
#             self.pre_planes += [-i]
#         for i in range(steps-1):
#             self.test += [-i]

#         self.strides = [1, 2, 2]
#         super(ZeroSAny_Tra, self).__init__()
#         self.block = block
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.pretrain = pretrain
#         self.stochastic_depth = stochastic_depth
#         self.share_coe = share_coe
#         self.noise_level = noise_level
#         self.PL = PL
#         self.noise = noise
#         self.steps = steps
#         self.coe_ini= coe_ini
#         self.ini_stepsize = ini_stepsize
#         self.given = [1.0/3, 5.0/9, 1.0/9, 16.0/9]
#         # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
#         # self.given = [-3.0/2, 3.0, -1.0/2, 3.0]
#         # self.given = [-0.380903884996629, 1.16875181181267, 0.415624094093667, -0.203472020909704, 2.38958393727089]
#         # self.given = [0.403523597279250, 0.475777922837489, 0.112412848357367, 0.00828563152589346, 1.72546051412990]
#         # self.given = [0.380905880882445, -1.16875507780764, -0.415622461096181, 0.203471658021374, -2.38958502593588]
#         # self.given = [0.165137619606463, -0.914615956838281, -0.375261502764746, 0.0828822102925340, 0.0418576297040286, -2.24906181267406]
#         # self.given = [-0.165137619606463, 0.914615956838281, 0.375261502764746, -0.0828822102925340, -0.0418576297040286, 2.24906181267406]
#         # self.given = [0, 1, 2]
#         # self.given = [1, 0, 1]

#         # self.given = [1, 0, 0, 1.0/2]

#         # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.steps+1)]) 
#         self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=False) for i in range(self.steps+1)]) 
#         # if self.share_coe == True:
#         # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
#         blocks = []
#         n = layers[0] + layers[1] + layers[2]
#         # print('n', n)
#         l = 0
#         if not self.stochastic_depth:
#             for i in range(3):
#                 # print('self.pre_planes in net1', self.pre_planes)
#                 blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coes=self.coes, steps=self.steps, stride=self.strides[i], coe_ini=self.coe_ini))
#                 # if l ==0 or l==1:
#                 #     self.l_last_res_planes = self.last_res_planes
#                 #     self.last_res_planes = self.in_planes
#                 if l < steps-1:
#                     for j in range(steps-2,0,-1): # steps-1, steps-2, ..., 1
#                         self.pre_planes[j] =  self.pre_planes[j-1]
#                         # self.pre_planes[j-2] =  self.pre_planes[j-1]
                        
#                     self.pre_planes[0] = self.in_planes
#                 else:
#                     for j in range(steps-2,0,-1): # steps-2, ..., 1
#                         self.pre_planes[j] =  self.planes[i] * block.expansion
#                     self.pre_planes[0] = self.planes[i] * block.expansion

#                     # self.l_last_res_planes = self.planes[i] * block.expansion
#                     # self.last_res_planes = self.planes[i] * block.expansion
#                 self.in_planes = self.planes[i] * block.expansion
#                 l += 1
#                 for k in range(1, layers[i]):
#                     # print('self.pre_planes in net2', self.pre_planes)
#                     blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coes=self.coes, steps=self.steps, coe_ini=self.coe_ini))
#                     # if l == 0 or l == 1:
#                     #     self.l_last_res_planes = self.last_res_planes
#                     #     self.last_res_planes = self.in_planes
#                     if l < steps-1:
#                         for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
#                             self.pre_planes[j] =  self.pre_planes[j-1]
#                         self.pre_planes[0] = self.in_planes
#                     else:
#                         for j in range(steps-2,0,-1): # steps-2, steps-3, ..., 1
#                             self.pre_planes[j] =  self.planes[i] * block.expansion
#                         self.pre_planes[0] = self.planes[i] * block.expansion
#                         # self.l_last_res_planes = self.planes[i] * block.expansion
#                         # self.last_res_planes = self.planes[i] * block.expansion
#                     l += 1
#                 # print('l, self.pre_planes', l, self.pre_planes)

#         # else:
#         #     death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]
#         #     for i in range(3):
#         #         blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,self.strides[i], coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0]]))
#         #         self.l_last_res_planes = self.last_res_planes
#         #         self.last_res_planes = self.in_planes
#         #         self.in_planes = self.planes[i] * block.expansion
#         #         for j in range(1, layers[i]):
#         #             blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0] + j]))
#         #             self.l_last_res_planes = self.last_res_planes
#         #             self.last_res_planes = self.in_planes
#         self.blocks = nn.ModuleList(blocks)
#         self.downsample1 = Downsample(16, 64, stride=1)
#         self.bn = nn.BatchNorm2d(64 * block.expansion)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64 * block.expansion, num_classes)

#         for m in self.modules():  # initialization
            
#             # print('m', torch.nn.Module.load_state_dict( m ) )
#             # print('Name', name)
#             # if 'Fix' in m:
#             #     m.requires_grad = False
#             #     print('FIX')
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
                
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


#         # # for name, param in net.named_parameters():
#         # #     if 'Fix' in name:
#         # #         param.requires_grad = False
                
#         # # coes = nn.Parameter(torch.Tensor(1))#.cuda()
#         # # self.coes = nn.Parameter(torch.Tensor(1).uniform_(self.coe_ini, self.coe_ini), requires_grad = True)#.cuda()
#         # # self.coes[0].data.uniform_(coe_ini, coe_ini)
#         # self.coes = torch.empty(1)
#         # torch.nn.init.normal_(self.coes, mean=self.coe_ini, std=0)
        
#         # for i in range(1,self.steps+1,1):
#         #     # coe_tem = nn.Parameter(torch.Tensor(1))#.cuda()
#         #     # coe_tem =nn.Parameter(torch.Tensor(1).uniform_(self.coe_ini, self.coe_ini), requires_grad = True)#.cuda()
#         #     coe_tem = torch.empty(1)
#         #     torch.nn.init.normal_(coe_tem, mean=self.coe_ini, std=0)
#         #     self.coes = torch.cat( (self.coes, coe_tem), 0)
#         #     # self.coes.append( nn.Parameter(torch.Tensor(1).uniform_(self.coe_ini, self.coe_ini), requires_grad = True).cuda() )
#         # # self.coes = nn.ModuleList(self.coes)
#         # # self.register_parameter("Coes",self.coes)
        
                           
#     def forward(self, x):
        
#         # stepsize =nn.Parameter(torch.Tensor(1).uniform_(self.coe_ini, self.coe_ini))
#         # torch.nn.init.constant_(stepsize, self.coe_ini)
#         # coes = torch.cat( (coes, stepsize), 0)
                        
#         x = self.conv1(x)
#         # last_res=-1
#         # l_last_res=-1
#         pre_features = []
#         for i in range(self.steps-1):
#             pre_features.append(-i)
#         if self.block.expansion == 4:
#             residual = self.downsample1(x)
#         else:
#             residual = x


#         # x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
#         # x += residual
#         # residual = x
#         # x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
#         # x += residual
#         for j in range(self.steps-1):
#             x, pre_features, coes = self.blocks[j](x, pre_features, self.coes, self.ini_stepsize)
#             x += residual
#             residual = x
#             # print('x.shape:', x.shape)


#         # x, self.pre_features, self.coes = self.blocks[i](x, self.pre_features)
#         # x += residual
#         # residual = x
#         for i, b in enumerate(self.blocks):  # index and content
#             if i < self.steps-1:
#                 continue
#             residual = x 
#             if self.pretrain:  #
#                 x = b(x) + residual

#             else:  
#                 x, pre_features, coes = b(x, pre_features, self.coes, self.ini_stepsize)
#                 # print('i th block, coes', i, coes[2].data)
#             # print('x.shape:', x.shape)
   
#                 # x, last_res, l_last_res, k = b(x, last_res, l_last_res)
#         # for i, b in enumerate(self.blocks):  # index and content
#         #     if i == 0 or i == 1:
#         #         continue
#         #     residual = x 
#         #     if self.pretrain:  #
#         #         x = b(x) + residual

#         #     else:  
#         #         x, last_res, l_last_res, k = b(x, last_res, l_last_res)

#                 # self.coes += [k.data]
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         # print('type(x):', type(x))
#         return x, coes, 1

   
# class ZeroSBlockAny(nn.Module):
#     expansion = 1

#     # def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, l_l_last_res_planes, steps=3, stride=1, coe_ini=1, fix_coe=False,
#     #              stepsize=1, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], downsample=None):
#     def __init__(self, in_planes, planes, pre_planes, coes, steps=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
#         super(ZeroSBlockAny, self).__init__()
#         self.steps = steps
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#         self.in_planes = in_planes
#         self.planes = planes
#         self.pre_planes_b = pre_planes
#         # self.stepsize = ini_stepsize
#         self.fix_coe = fix_coe
#         self.coes = coes
#         # self.coes = []
#         self.start=1
#         # self.pre_downsamples = nn.ModuleList([])
        
#         # if self.fix_coe:
#         #     self.coe = coe_ini
#         #     for i in range(self.steps+1):
#         #         self.coes.append(float(given_coe[i]))
#         #     # self.a_0 = float(given_coe[0])
#         #     # self.a_1 = float(given_coe[1])
#         #     # self.a_2 = float(given_coe[2])
#         #     # self.b_0 = float(given_coe[3])
#         # else:
#         #     self.coes = nn.Parameter(torch.Tensor(1)).cuda()
#         #     # self.coes[0].data.uniform_(coe_ini, coe_ini)
#         #     torch.nn.init.constant_(self.coes[0], coe_ini)
#         #     for i in range(1,self.steps+1,1):
#         #         coe_tem = nn.Parameter(torch.Tensor(1)).cuda()
#         #         torch.nn.init.constant_(coe_tem, coe_ini*i)
#         #         self.coes = torch.cat( (self.coes,  coe_tem), 0)
#         #         # print('self.coes', self.coes)
#         #         # self.coes.append(nn.Parameter(torch.Tensor(1)).cuda() )
#         #         # self.coes[i].data.uniform_(coe_ini, coe_ini)

#             # print('self.pre_planes_b, self.planes:', self.pre_planes_b, self.planes)

#             # for i in range(self.steps-1):
#             #     print("self.start1, i ", self.start, i)
#             #     self.start *= (self.pre_planes_b[i] <=0) # maybe can be replaced by < steps
#             #     print("self.start2, i ", self.start, i)

#             # self.coes.append(nn.Parameter(torch.Tensor(1)))
#             # self.coes.append(nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini) ))
            
   
            
#             # if (not self.start) and self.pre_planes_b[i] != self.planes:
#         # if (self.start==0):
#         # print('self.pre_planes_b', self.pre_planes_b)
#         self.has_ini_block = False
#         for pre_pl in self.pre_planes_b:
#             # print('pre_pl', pre_pl)
#             if pre_pl <=0:
#                 self.is_ini = True
#             else:
#                 self.is_ini = False
#             self.has_ini_block = self.is_ini or self.has_ini_block #wrong
#         # print('end block')  
#         # print('self.has_ini_block', self.has_ini_block)  
#         if not (self.has_ini_block):
#         # if all(value > 0 for value in self.pre_planes_b):
#         #     self.is_ini = True
#         # if -1 in :
#             if self.in_planes != self.planes:
#                 self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
#             start_DS = 1        
#             for i in range(self.steps-1):
#                 if self.pre_planes_b[i] != self.planes:
#                     if start_DS:
#                         self.FixDS = nn.ModuleList([])
#                         start_DS = 0
#                     # self.FixDS.append( nn.Conv2d(self.pre_planes_b[i].shape[1], self.planes, kernel_size=1, stride=2, padding=0, bias=False) )
#                     self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes, 2) )
#             # print('not (self.has_ini_block)')
#             # for i in range(self.steps-1): 
#             #     if self.pre_planes_b[i] != self.planes:
#             #         self.pre_downsamples.append(Downsample_clean(self.pre_planes_b[i], self.planes, 2) )
            
#         # self.downsample_l = [Downsample_clean(32, self.planes, 2)]
#         # if not (self.has_ini_block):
#         #     self.pre_downsamples.append(Downsample_clean(self.pre_planes_b[0], self.planes, 2) )
#         #     self.pre_downsamples.append(Downsample_clean(self.pre_planes_b[1], self.planes, 2) )

#         # print('self.pre_downsamples', self.pre_downsamples)
            
#         # print('self.pre_downsamples', self.pre_downsamples)
#         # if not (self.last_res_planes == -1 or self.l_last_res_planes == -1 or self.l_l_last_res_planes == -1):
#         #     if self.in_planes != self.planes:
#         #         self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
#         #     if self.last_res_planes != self.planes:
#         #         self.downsample_l = Downsample_clean(self.last_res_planes, self.planes, 2)
#         #     if self.l_last_res_planes != self.planes:
#         #         self.downsample_ll = Downsample_clean(self.l_last_res_planes, self.planes, 2)
#         #     if self.l_l_last_res_planes != self.planes:
#         #         self.downsample_lll = Downsample_clean(self.l_l_last_res_planes, self.planes, 2)

            
#             # self.a_0 = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini) )
#             # self.a_1 = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))
#             # self.a_2 = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))
#             # self.a_3 = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))
#             # self.b_0 = nn.Parameter(torch.Tensor(1).uniform_(coe_ini, coe_ini))

#         # if not (self.last_res_planes == -1 or self.l_last_res_planes == -1 or self.l_l_last_res_planes == -1):
#         #     if self.in_planes != self.planes:
#         #         self.downsample_x = Downsample_clean(self.in_planes, self.planes, 2)
#         #     if self.last_res_planes != self.planes:
#         #         self.downsample_l = Downsample_clean(self.last_res_planes, self.planes, 2)
#         #     if self.l_last_res_planes != self.planes:
#         #         self.downsample_ll = Downsample_clean(self.l_last_res_planes, self.planes, 2)
#         #     if self.l_l_last_res_planes != self.planes:
#         #         self.downsample_lll = Downsample_clean(self.l_l_last_res_planes, self.planes, 2)

    
#     # def forward(self, x, last_res, l_last_res, l_l_last_res):  # Pre-ResNet
#     def forward(self, x, pre_features, coes, stepsize):  

#         # residual = pre_features[0]
#         residual = x
#         F_x_n = self.bn1(x)
#         F_x_n = self.relu(F_x_n)
#         F_x_n = self.conv1(F_x_n)
#         F_x_n = self.bn2(F_x_n)
#         F_x_n = self.relu(F_x_n)
#         F_x_n = self.conv2(F_x_n)
#         # print('F_x_n', F_x_n.shape)
#         # if not (isinstance(pre_features[1], int) or isinstance(l_last_res, int) or isinstance(l_l_last_res, int)):
#         # self.start = 1
#         # for i in range(self.steps):
#         #     self.start *= (self.pre_planes_b[i] == -1)
#         # if not self.start:
#         # print('self.pre_planes_b', self.pre_planes_b)
#         # print('self.planes', self.planes)


#         # print('self.pre_downsamples', self.pre_downsamples)
#         # print('self.downsample_l', self.downsample_l)
#         # print('pre_features', pre_features)
#         # if not all( isinstance(pre_fea, int) for pre_fea in pre_features):    
#         self.has_ini_block = False
#         for pre_fea in pre_features:
#             if isinstance(pre_fea, int):
#                 self.is_ini = True
#             else:
#                 self.is_ini = False
#             self.has_ini_block = self.is_ini or self.has_ini_block
            
#         if not (self.has_ini_block):
#             if self.in_planes != self.planes:
#                 residual = self.downsample_x(residual)
#             # self.FixDS = []
#             # self.FixDS = nn.ModuleList([])
#             # for i in range(self.steps-1):
#             #     if self.pre_planes_b[i] != self.planes:
#             #         pre_features[i] = self.pre_downsamples[i](pre_features[i])
#             #     # print('pre_features[i].shape[1], F_x_n.shape[1]', pre_features[i].shape[1], F_x_n.shape[1])
#             #     # if pre_features[i].shape[1] != F_x_n.shape[1]:
#             #         # w = torch.empty(F_x_n.shape[1], pre_features[i].shape[1], 1, 1).cuda()
#             #         # nn.init.dirac_(w, 2)
#             #         # w = nn.Parameter(w, requires_grad=False)
#             #         # pre_features[i] = functional.conv2d(pre_features[i], w, bias=None, stride=2, padding=0, dilation=1, groups=1)
                    
#             #         # self.FixDS.add_module( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
#             #         # self.FixDS.cuda() 
#             #         self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
#             #         # self.FixDS.cuda()    
                                  
#             #         # self.FixDS = self.FixDS + [nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() ]
#             #         pre_features[i] = self.FixDS[-1](pre_features[i])
#             #         nn.init.dirac_(self.FixDS[-1].weight, 2)
#             # start_DS = 1        
#             for i in range(self.steps-1):
#                 if pre_features[i].shape[1] != F_x_n.shape[1]:
#                     # if start_DS:
#                     #     self.FixDS = nn.ModuleList([])
#                     #     start_DS = 0
#                     # self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
#                     pre_features[i] = self.FixDS[-1](pre_features[i])
#                     # nn.init.dirac_(self.FixDS[-1].weight, 2)
#                     # self.FixDS[-1].weight.requires_grad = False
#             # sum_features = torch.mul(self.coes[0].data, residual )
#             sum_features = self.coes[0].expand_as(residual)*residual

#             # print('self.coes[0]', self.coes[0])
#             for i in range(self.steps-1):
#                 # sum_features = torch.add( sum_features, torch.mul(self.coes[i+1].data, pre_features[i]) )
#                 sum_features = torch.add( sum_features, self.coes[i+1].expand_as(pre_features[i])*pre_features[i] )
                
#             # x =  torch.add( sum_features, torch.mul(stepsize, torch.mul(self.coes[-1].data, F_x_n) ) )
#             x =  torch.add( sum_features, torch.mul(stepsize, self.coes[-1].expand_as(F_x_n)*F_x_n) )

#             # pre_features[0] =  torch.add( sum_features, torch.mul(self.stepsize, torch.mul(self.coes[-1], F_x_n) ) )


#                 # x = torch.mul(self.a_0, residual) + torch.mul(
#                 # self.a_1, last_res) + torch.mul(self.a_2, l_last_res)+ torch.mul(self.a_3, l_l_last_res)+torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n))
                            
#         # if not (isinstance(pre_features[1], int) or isinstance(l_last_res, int) or isinstance(l_l_last_res, int)):
#         #     if self.in_planes != self.planes:
#         #         residual = self.downsample_x(residual)
#         #     if self.last_res_planes != self.planes:
#         #         last_res = self.downsample_l(last_res)
#         #     if self.l_last_res_planes != self.planes:
#         #         l_last_res = self.downsample_ll(l_last_res)
#         #     if self.l_l_last_res_planes != self.planes:
#         #         l_l_last_res = self.downsample_lll(l_l_last_res)

#             # if not self.fix_coe:
#                 # self.b_0 = (3 * self.coe - 1) / (self.coe * 2)
#                 # self.a_0 = (3 * self.coe + 3) / (self.coe * 4)
#                 # self.a_1 = -1 / (self.coe)
#                 # self.a_2 = (self.coe + 1) / (4 * self.coe)
#             # x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(
#             #     self.a_1, last_res) + torch.mul(self.a_2, l_last_res)+ torch.mul(self.a_3, l_l_last_res)
#         else:
#             x = F_x_n
#         for i in range(self.steps-2, 0, -1): #steps-2, steps-1, ..., 0 #1, 0 
#             # print('i', i)
#             # print('pre_features', pre_features)
#             pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
#         pre_features[0] = residual
#         # pre_features[0] = residual 
#         # x = residual 
#         # for i in range(self.steps, 0, -1): #steps, steps-1, ..., 1
#         #     pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
        
#         return x, pre_features, self.coes

#             # l_last_res = last_res # [1]=[0]
#             # last_res = residual # [0]=residual
#         # self.coes = [[self.a_0]+[self.a_1]+[self.a_2]+[self.a_3]+[self.b_0]]
#         # return x, last_res, l_last_res, l_l_last_res, self.coes
# def forward_hook(module, inp, outp):
#     feature_map['blocks'] = outp
    
output_list=[]
input_list=[]    
def forward_hook(module,data_input,data_output):
    input_list.append(data_input)
    output_list.append(data_output)
        
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True    
    

                
if __name__ == '__main__':
    # w = torch.empty(6, 3, 1, 1)
    # nn.init.dirac_(w,2)
    # # w = torch.empty(3, 24, 5, 5)
    # # nn.init.dirac_(w, 3)
    # print('w', w)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_torch()

    # d = torch.rand(1, 3, 32, 32).to(device)
    # d = torch.rand(1, 1, 28, 28).to(device)

#     net = ZeroSAny32Ablation(givenA=[1.49323762497707, -0.574370781405754, 0.0855838379295368, -0.00445068150085398,
# ], givenB=[0.400601121454727, -1.68484817541425, 1.80393876806197, -2.10313656405320], ablation='mnistAllElePlotTraj')#S4O0
    
#     net = ZeroSAny32Ablation(givenA=[1.49323762497707, -0.574370781405754, 0.0855838379295368, -0.00445068150085398,
# ], givenB=[-2.10313656405320, 5.60787753612393, -7.29272571153819, 3.20453988951669
# ], ablation='mnistAllElePlotTraj')#S4O2 ExpDecay
    
    net = ZeroSAny56Ablation(givenA=[1.49323762497707, -0.574370781405754, 0.0855838379295368, -0.00445068150085398,
], givenB=[-2.10313656405320, 2.80393876806197, -1.68484817541425, 0.400601121454727
], ablation='mnist2ChsLearnCoeAllEle')#S4O4
    # net = ZeroSAny32Ablation(givenA=[1, 0, 0, 0, 0, 0], givenB=[-2.9701388888889, 5.5020833333333,  -6.9319444444444, 5.0680555555556, -1.9979166666667, 0.32986111111111 ], ablation='mnistAllElePlotTraj')
    # net = ZeroSAny32Ablation(givenA=[1, 0], givenB=[-1, 0 ], ablation='mnistAllElePlotTraj')    
    # d = torch.randn(1, 3, 32, 32).to(device)
    # net = SamNet20_Tra()
    # net = MaskNet20()
    
    d = 0.5*torch.ones(1, 1, 28, 28).to(device)

    # net.apply(weights_init) 
    # net.apply(weights_init_plot)
    # net = net.to(device)
    # module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity)
    # net = tx.Extractor(net, module_filter_fn=module_filter_fn)
    # out, features = net(d)          
    # fmap = []
    # feature_shapes = {name: f.shape for name, f in features.items()}
    # for name, f in features.items():
    #     fmap += [float(f.data.mean())]
    # print(fmap)
    # flops, params = profile(net, inputs=(d, ))
    # print('flops, params', flops, params)
    # total_trainable_params2 = 0
    # for name, p in net.named_parameters():
    #     if p.requires_grad: 
    #         if name:
    #             if 'Fix' not in name: # and 'bn3' not in name:
    #                 total_trainable_params2 +=p.numel()
    #             else:
    #                 print('name', name)
    #         else:
    #             total_trainable_params2 +=p.numel()

    # print(f'{total_trainable_params2:,} training parameters(w/o Fix).')
    
    # total_params2 = 0
    # for p in net.parameters():
    #     total_params2 +=p.numel()
    # print(f'{total_params2:,} total parameters.')
    # total_trainable_params2 = 0
    # for name, p in net.named_parameters():
    #     if p.requires_grad: 
    #         total_trainable_params2 +=p.numel()

    # print(f'{total_trainable_params2:,} training parameters(w/ Fix).')
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if (p.requires_grad))# and 'Fix' not in p[0]))
    # print(f'{total_trainable_params:,} training parameters.')

    # summary(net, input_size=(3, 32, 32), batch_size=-1)
    # summary(net, input_size=(1, 28, 28), batch_size=-1)

    # macs, params = profile(net, inputs=(d, ))
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True
    # net = ZeroSNet20_Tra()
    # o = net(d)
    # print('net:', net)
    # net.apply(weights_init) 
    # net.apply(weights_init_plot)
    net = net.to(device)
    df = pd.DataFrame()
    for mag in range(0,10,1):

        d = d + mag/1000.0

        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity)
        net = tx.Extractor(net, module_filter_fn=module_filter_fn)
        out, features = net(d)          
        fmap = []
        feature_shapes = {name: f.shape for name, f in features.items()}
        for name, f in features.items():
            fmap += [float(f.data.mean())]
        # print(fmap)
        # feature_map = {}
        # features = []
        # blocks = list(net.children())[5]
        
        # for i in range(len(blocks)):
        #     hook_layer = blocks[i].identity
        #     hook_layer.register_forward_hook(forward_hook)   
        #     net.eval()
        #     # with torch.no_grad():
        #     out = net(d)          
        #     # print("==>> type(feature_map): ", feature_map)
        #     print('output_list', output_list[0].shape)
        #     features += [ float(output_list[0].data.mean() ) ]
        #     # features += [ float(feature_map['blocks'].data.mean() ) ]
            
        #     # features += [ feature_map['blocks'].data  ]
            
  
        # with torch.no_grad():
        #     out = net(d)    
        # print('net:', net)    
        
        # for i in range(len(blocks)):
        # print('features', fmap)
        df_row = pd.DataFrame([fmap])

        df = df.append(df_row)
        # torch.cuda.empty_cache()

    print('df', df)
    save_path = './OverThreeOrders/CIFAR/OverThreeOrders/CIFAR/runs/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = 'features_test_S4O4ExpDecay.csv'
    df.to_csv(save_path+file_name)
    # hook_layer = blocks[10].identity
    # print('hook_layer',hook_layer)
    # hook_layer.register_forward_hook(forward_hook)

    # with torch.no_grad():
    #     score = net(d)    
    # print(feature_map['blocks10'].data.mean())
    
    # for name, param in net.named_parameters():
    #     if 'Fix' in name:
    #         print(name, param.data.mean())
    # #     if 'mask_conv_' in name:
    # #         print(name, param.data)
    # for i in net.named_parameters():
    #     print(i[0], i[1].shape)
    # for name in net.state_dict():
        # print('name', name)
    
    # onnx_path = "onnx_model_name.onnx"
    # torch.onnx.export(net, d, onnx_path)
    # netron.start(onnx_path)