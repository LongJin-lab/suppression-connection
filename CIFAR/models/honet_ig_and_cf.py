import torch.nn as nn

import torch
import torch.nn.functional as functional
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron
# from init import *
from random import random
import argparse



# __all__ = ['pre_resnet18', 'pre_resnet34', 'pre_resnet50', 'pre_resnet101',
#            'pre_resnet152']
__all__ = [ 'honet18_cifar', 'honet50_cifar', 'honet34_cifar', 'honet18_in', 'honet34_in',  'honet50_in']
# __all__ = ['honet18_in', 'honet34_in', 'honet50_in', 'pre_act_resnet18_in', 'pre_act_resnet34_in', 'pre_act_resnet50_in', 'honet18_cifar', 'honet34_cifar', 'honet50_cifar']
# __all__ = ['HONet34_IN', 'HONet18_IN']
# parser = argparse.ArgumentParser(description='PyTorch Training')
# args = parser.parse_args()

global num_cla
num_cla = 10

class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, death_rate=0., downsample=None):
        super(BasicBlockWithDeathRate, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.death_rate = death_rate

    def forward(self, x):

        if not self.training or torch.rand(1)[
            0] >= self.death_rate:  # 2nd condition: death_rate is below the upper bound
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            # ^ the same with Pre-ResNet
            if self.training:
                out /= (1. - self.death_rate)  # out = out/(1. - death_rate) ? maybe it is mutiplied by the rate before
        else:
            if self.stride == 1:
                out = Variable(torch.FloatTensor(x.size()).cuda().zero_(), requires_grad=False)
            else:

                size = list(x.size())
                size[-1] //= 2  # Maybe it is the Height (interger, devide)
                size[-2] //= 2  # Maybe it is the Width
                size[-3] *= 2  # Maybe Channel
                size = torch.Size(size)
                out = Variable(torch.FloatTensor(size).cuda().zero_(), requires_grad=False)  # all zero tensor
        return out


class BasicBlock_cifar(nn.Module):  # actually, this is the preact block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):  # Pre-ResNet
        out = self.bn1(x)  # wo BN
        # out = x # wo BN
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class HOBlock(nn.Module):  # actually, this is the preact block
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, stride=1, k_ini=-9.0 / 5, fix_k=False,
                 stepsize=1, given_ks=[10, 10, 10, 10], downsample=None):
        super(HOBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm2d(planes)# 20210803
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_k = fix_k
        if self.fix_k:
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        else:
            self.k = nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks = nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # print('l_last_res_planes, last_res_planes, in_planes, planes', l_last_res_planes, last_res_planes, in_planes, planes)

        if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
            # if 1:
            if self.planes == 32:
                if in_planes == 16:
                    self.downsample_16_32_x = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_x')
                if self.last_res_planes == 16:
                    self.downsample_16_32_l = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_l')
                if self.l_last_res_planes == 16:
                    self.downsample_16_32_ll = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_ll')
            if self.planes == 64:
                if self.in_planes == 32:
                    self.downsample_32_64_x = Downsample_clean(32, 64, 2)
                if self.last_res_planes == 32:
                    self.downsample_32_64_l = Downsample_clean(32, 64, 2)
                if self.l_last_res_planes == 32:
                    self.downsample_32_64_ll = Downsample_clean(32, 64, 2)
            if self.planes == 128:
                if self.in_planes == 64:
                    self.downsample_64_128_x = Downsample_clean(64, 128, 2)
                if self.last_res_planes == 64:
                    self.downsample_64_128_l = Downsample_clean(64, 128, 2)
                if self.l_last_res_planes == 64:
                    self.downsample_64_128_ll = Downsample_clean(64, 128, 2)
            if self.planes == 256:
                if self.in_planes == 128:
                    self.downsample_128_256_x = Downsample_clean(128, 256, 2)
                if self.last_res_planes == 128:
                    self.downsample_128_256_l = Downsample_clean(128, 256, 2)
                if self.l_last_res_planes == 128:
                    self.downsample_128_256_ll = Downsample_clean(128, 256, 2)

    def forward(self, x, last_res, l_last_res):  # Pre-ResNet
        residual = x
        F_x_n = self.bn1(x)  # wo BN
        # F_x_n=x
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)
        # if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
        # print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res.size()[1],l_last_res.size()[1])
        # print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
            # print('HO')
            # if 1:
            if self.planes == 32:
                if self.in_planes == 16:
                    residual = self.downsample_16_32_x(residual)
                    # print('residual.size()', residual.size())
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_32_l(last_res)
                # print('last_res.size()', last_res.size())
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_32_ll(l_last_res)
                    # print('l_last_res.size()', l_last_res.size())
            if self.planes == 64:
                if self.in_planes == 32:
                    residual = self.downsample_32_64_x(residual)
                if self.last_res_planes == 32:
                    last_res = self.downsample_32_64_l(last_res)
                if self.l_last_res_planes == 32:
                    l_last_res = self.downsample_32_64_ll(l_last_res)
            if self.planes == 128:
                if self.in_planes == 64:
                    residual = self.downsample_64_128_x(residual)
                if self.last_res_planes == 64:
                    last_res = self.downsample_64_128_l(last_res)
                if self.l_last_res_planes == 64:
                    l_last_res = self.downsample_64_128_ll(l_last_res)
            if self.planes == 256:
                if self.in_planes == 128:
                    residual = self.downsample_128_256_x(residual)
                if self.last_res_planes == 128:
                    last_res = self.downsample_128_256_l(last_res)
                if self.l_last_res_planes == 128:
                    l_last_res = self.downsample_128_256_ll(l_last_res)
            if not self.fix_k:
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)
                # print("trainable")
            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(
                self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
            # print('x', x[0][0][0][0])
            # print("self.a_0, self.a_1, self.a_2, self.b_0", self.a_0, self.a_1, self.a_2, self.b_0)
        else:
            # print('res')
            x = F_x_n
        # x = self.bn3(x)
        l_last_res = last_res
        last_res = residual  # x means the residual
        # residual = x
        return x, last_res, l_last_res, self.k


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(torch.randn(x.size()).cuda() * self.stddev, requires_grad=False)
        return x


class Bottleneck_cifar(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_cifar, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class HoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, stride=1, k_ini=-9.0 / 5, fix_k=False,
                 stepsize=1, given_ks=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):
        super(HoBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.expansion = 4
        self.in_planes = in_planes
        self.planes = planes * self.expansion
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.fix_k = fix_k
        if self.fix_k:
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        else:
            self.k = nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks=nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # self.downsample_16_64_res = Downsample_clean(16, 64, 1)
        # if not (last_res_planes == -1 and l_last_res_planes == -1):
        # if 1:
        if not (last_res_planes == -1 or l_last_res_planes == -1):
            if self.planes == 32:
                if in_planes == 16:
                    self.downsample_16_32_x = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_x')
                if last_res_planes == 16:
                    self.downsample_16_32_l = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_l')
                if l_last_res_planes == 16:
                    self.downsample_16_32_ll = Downsample_clean(16, 32, 2)
                    # print('downsample_16_32_ll')
            if self.planes == 64:
                if self.in_planes == 16:
                    self.downsample_16_64_x = Downsample_clean(16, 64, 1)
                    # print('downsample_16_32_x')
                if self.last_res_planes == 16:
                    self.downsample_16_64_l = Downsample_clean(16, 64, 1)
                    # print('downsample_16_32_l')
                if self.l_last_res_planes == 16:
                    self.downsample_16_64_ll = Downsample_clean(16, 64, 1)
                if self.in_planes == 32:
                    self.downsample_32_64_x = Downsample_clean(32, 64, 2)
                if self.last_res_planes == 32:
                    self.downsample_32_64_l = Downsample_clean(32, 64, 2)
                if self.l_last_res_planes == 32:
                    self.downsample_32_64_ll = Downsample_clean(32, 64, 2)
            if self.planes == 128:
                if self.in_planes == 64:
                    self.downsample_64_128_x = Downsample_clean(64, 128, 2)
                if self.last_res_planes == 64:
                    self.downsample_64_128_l = Downsample_clean(64, 128, 2)
                if self.l_last_res_planes == 64:
                    self.downsample_64_128_ll = Downsample_clean(64, 128, 2)
            if self.planes == 256:
                if self.in_planes == 128:
                    self.downsample_128_256_x = Downsample_clean(128, 256, 2)
                if self.last_res_planes == 128:
                    self.downsample_128_256_l = Downsample_clean(128, 256, 2)
                if self.l_last_res_planes == 128:
                    self.downsample_128_256_ll = Downsample_clean(128, 256, 2)

    def forward(self, x, last_res, l_last_res):
        # if self.expansion==4:
        #     residual = self.downsample_16_64_res(x)
        # elif self.expansion==1:
        #     residual = x
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

        # self.planes = self.planes*self.expansion

        # if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res.size()[1],l_last_res.size()[1])
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        # elif not (isinstance(last_res,int)):
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()', F_x_n.size()[
        #     1], residual.size()[1], last_res.size()[1], l_last_res)
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        # else:
        #     print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res,l_last_res)
        #     print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)
        if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
            # print('HO')
            # if 1:
            if self.planes == 32:
                if self.in_planes == 16:
                    residual = self.downsample_16_32_x(residual)
                    # print('residual.size()', residual.size())
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_32_l(last_res)
                # print('last_res.size()', last_res.size())
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_32_ll(l_last_res)
                    # print('l_last_res.size()', l_last_res.size())
            if self.planes == 64:
                if self.in_planes == 16:
                    residual = self.downsample_16_64_x(residual)
                if self.last_res_planes == 16:
                    last_res = self.downsample_16_64_l(last_res)
                if self.l_last_res_planes == 16:
                    l_last_res = self.downsample_16_64_ll(l_last_res)
                if self.in_planes == 32:
                    residual = self.downsample_32_64_x(residual)
                if self.last_res_planes == 32:
                    last_res = self.downsample_32_64_l(last_res)
                if self.l_last_res_planes == 32:
                    l_last_res = self.downsample_32_64_ll(l_last_res)

            if self.planes == 128:
                if self.in_planes == 64:
                    residual = self.downsample_64_128_x(residual)
                if self.last_res_planes == 64:
                    last_res = self.downsample_64_128_l(last_res)
                if self.l_last_res_planes == 64:
                    l_last_res = self.downsample_64_128_ll(l_last_res)
            if self.planes == 256:
                if self.in_planes == 128:
                    residual = self.downsample_128_256_x(residual)
                if self.last_res_planes == 128:
                    last_res = self.downsample_128_256_l(last_res)
                if self.l_last_res_planes == 128:
                    l_last_res = self.downsample_128_256_ll(l_last_res)
            if not (isinstance(last_res, int) or isinstance(l_last_res, int)):
                if not self.fix_k:
                    self.b_0 = (3 * self.k - 1) / (self.k * 2)
                    self.a_0 = (3 * self.k + 3) / (self.k * 4)
                    self.a_1 = -1 / (self.k)
                    self.a_2 = (self.k + 1) / (4 * self.k)
                # x = torch.mul(b_0, F_x_n) + torch.mul(a_0, residual) + torch.mul(a_1, last_res) + torch.mul(a_2, l_last_res)

                x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(
                    self.a_1, last_res) + torch.mul(self.a_2, l_last_res)

        else:
            # print('res')
            x = F_x_n
        l_last_res = last_res
        last_res = residual  # x means the residual
        # residual = x
        # print('x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        return x, last_res, l_last_res, self.k


class Downsample(nn.Module):  # ReLU and BN are involved in this downsample
    def __init__(self, in_planes, out_planes, stride=2):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class Downsample_clean(nn.Module):  # ReLU and BN are involved in this downsample
    def __init__(self, in_planes, out_planes, stride=2):
        super(Downsample_clean, self).__init__()
        self.downsample_ = nn.Sequential(
            # nn.BatchNorm2d(in_planes),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample_(x)
        return x


class Downsample_real(nn.Module):  # ReLU and BN are not involved in this downsample
    def __init__(self, in_shape, out_shape):
        super(Downsample_real, self).__init__()
        # in_shape = x.shape()
        self.in_planes = in_shape[1]
        self.out_planes = out_shape[1]
        self.stride = int(in_shape[2] / out_shape[2])
        # [256, 64, 32, 32]->[256, 128, 16, 16]
        self.downsample_real = nn.Sequential(
            # nn.BatchNorm2d(in_planes),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, self.out_planes,
                      kernel_size=1, stride=self.stride, bias=False)
        )

    def forward(self, x):
        x = self.downsample_real(x)
        return x


class MResNet(nn.Module):

    # def __init__(self,block,layers,pretrain=True,num_classes=num_cla,stochastic_depth=False,PL=0.5,noise_level=0.001,noise=False):
    def __init__(self, block, layers, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0,
                 noise_level=0.001, noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(MResNet, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.ks = nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.1)) for i in
                                    range(layers[0] + layers[1] + layers[2])])  # each layer has a trainable $k_n$
        self.stochastic_depth = stochastic_depth
        blocks = []
        n = layers[0] + layers[1] + layers[2]

        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 0 to 2
                    blocks.append(block(self.in_planes, self.planes[i]))  # three (Basic) Blocks
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i],
                                    death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], death_rate=death_rates[i * layers[0] + j]))
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride=2):
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion,
                                       32 * block.expansion)  # "expansion" is 1 for BasicBlocks and is 4 for the Bottleneck
        # self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        # self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample

            residual = self.downsample1(x)  # residual.size()[1]: 16->64
        else:
            residual = x
        x = self.blocks[0](x) + residual  # x.size()[1]: 16->64
        last_res = residual
        for i, b in enumerate(self.blocks):  # index and content
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
                if b.planes == 32:
                    residual = self.downsample21(x)
                    # if not self.pretrain:
                    # last_res=self.downsample22(last_res)
                elif b.planes == 64:
                    residual = self.downsample31(x)
                    # if not self.pretrain:
                    # last_res=self.downsample32(last_res)
                x = b(x)
                # print(x.size())
                # print(residual.size())
                x += residual

            elif self.pretrain:  #
                x = b(x) + residual
            else:  # in.channel = out.channel and not pretrain
                x = b(x) + self.ks[i].expand_as(residual) * residual + (1 - self.ks[i]).expand_as(
                    last_res) * last_res  # "B.expand_as (A)": expand B in A's shape

            last_res = residual

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.ks




class HONet_v2(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_v2, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini))
                # ###
                # if
                #
                # ###
                # self.l_last_res_planes = self.last_res_planes
                # self.last_res_planes = self.in_planes
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 1 to 2
                    # if l == 0:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    #
                    # elif l==1:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    # else:
                    # self.l_last_res_planes = self.planes[i]*block.expansion
                    # self.last_res_planes = self.planes[i]*block.expansion
                    # self.plane = self.planes[i]*block.expansion
                    # print('j', j)
                    # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini))  # three (Basic) Blocks
                    # self.l_last_res_planes = self.last_res_planes
                    # self.last_res_planes = self.in_planes
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
            # print('downsample1')
        else:
            residual = x

        x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        # print('v2: x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        x += residual
        # l_last_res = residual
        residual = x

        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        # x = self.blocks[1](x)[0] + residual
        x += residual
        # last_res = residual
        # residual = x # moved from below. Flag:318
        ### \end

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318
            ####
            # if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
            #     if b.planes == 32:
            #         residual = self.downsample21(x)
            #         # if not self.pretrain:
            #         # last_res=self.downsample22(last_res)
            #     elif b.planes == 64:
            #         residual = self.downsample31(x)
            #
            #     x = b(x)
            #     # print(x.size())
            #     # print(residual.size())
            #     x += residual
            ####
            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                # \begin HONet core

                x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                self.ks += k.data
                # print('i, ks', i, self.ks)

                # \end HONet core
            # print('cnt', cnt1, cnt2, cnt3, cnt4)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('out')
        return x#, self.ks


class HONet_stepsize(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False, dataset='CIFAR'):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_stepsize, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize))
                # ###
                # if
                #
                # ###
                # self.l_last_res_planes = self.last_res_planes
                # self.last_res_planes = self.in_planes
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 1 to 2
                    # if l == 0:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    #
                    # elif l==1:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    # else:
                    # self.l_last_res_planes = self.planes[i]*block.expansion
                    # self.last_res_planes = self.planes[i]*block.expansion
                    # self.plane = self.planes[i]*block.expansion
                    # print('j', j)
                    # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize))  # three (Basic) Blocks
                    # self.l_last_res_planes = self.last_res_planes
                    # self.last_res_planes = self.in_planes
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
            # print('downsample1')
        else:
            residual = x

        x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        # print('v2: x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        x += residual
        # l_last_res = residual
        residual = x

        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        # x = self.blocks[1](x)[0] + residual
        x += residual
        # last_res = residual
        # residual = x # moved from below. Flag:318
        ### \end

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318
            ####
            # if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
            #     if b.planes == 32:
            #         residual = self.downsample21(x)
            #         # if not self.pretrain:
            #         # last_res=self.downsample22(last_res)
            #     elif b.planes == 64:
            #         residual = self.downsample31(x)
            #
            #     x = b(x)
            #     # print(x.size())
            #     # print(residual.size())
            #     x += residual
            ####
            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                # \begin HONet core

                x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                self.ks += k.data
                # print('i, ks', i, self.ks)

                # \end HONet core
            # print('cnt', cnt1, cnt2, cnt3, cnt4)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('out')
        return x, self.ks, self.stepsize


class HONet_given(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, fix_k=True, given_ks=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9],
                 pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False, dataset="CIFAR"):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_given, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.fix_k = fix_k
        self.given_ks = given_ks
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize, fix_k=self.fix_k, given_ks=self.given_ks))
                # ###
                # if
                #
                # ###
                # self.l_last_res_planes = self.last_res_planes
                # self.last_res_planes = self.in_planes
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 1 to 2
                    # if l == 0:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    #
                    # elif l==1:
                    #     self.l_last_res_planes = self.last_res_planes
                    #     self.last_res_planes = self.in_planes
                    # else:
                    # self.l_last_res_planes = self.planes[i]*block.expansion
                    # self.last_res_planes = self.planes[i]*block.expansion
                    # self.plane = self.planes[i]*block.expansion
                    # print('j', j)
                    # print('v2: self.planes[i],  self.in_planes, self.last_res_planes, self.l_last_res_planes', self.planes[i]* block.expansion, self.in_planes, self.last_res_planes, self.l_last_res_planes)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize, fix_k=self.fix_k,
                                        given_ks=self.given_ks))  # three (Basic) Blocks
                    # self.l_last_res_planes = self.last_res_planes
                    # self.last_res_planes = self.in_planes
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, fix_k=self.fix_k, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]],
                          given_ks=self.given_ks))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, fix_k=self.fix_k, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j], given_ks=self.given_ks))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
            # print('downsample1')
        else:
            residual = x

        out = self.blocks[0](x, last_res, l_last_res)
        # print('out', type(out))
        x = out[0]
        last_res = out[1]
        l_last_res = out[2]
        k = out[3]
        # x, last_res, l_last_res, k = self.blocks[0](x, last_res, l_last_res)
        # print('v2: x.sixe()[1], residual.size()[1]', x.size()[1], residual.size()[1])
        x += residual
        # l_last_res = residual
        residual = x

        out = self.blocks[1](x, last_res, l_last_res)
        x = out[0]
        last_res = out[1]
        l_last_res = out[2]
        k = out[3]
        # x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        # x = self.blocks[1](x)[0] + residual
        x += residual
        # last_res = residual
        # residual = x # moved from below. Flag:318
        ### \end

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318
            ####
            # if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
            #     if b.planes == 32:
            #         residual = self.downsample21(x)
            #         # if not self.pretrain:
            #         # last_res=self.downsample22(last_res)
            #     elif b.planes == 64:
            #         residual = self.downsample31(x)
            #
            #     x = b(x)
            #     # print(x.size())
            #     # print(residual.size())
            #     x += residual
            ####
            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                # \begin HONet core
                out = b(x, last_res, l_last_res)
                x = out[0]
                last_res = out[1]
                l_last_res = out[2]
                k = out[3]
                # x, last_res, l_last_res, k = b(x, last_res, l_last_res)
                if isinstance(k, float):
                    self.ks = k
                else:
                    self.ks += k.data
                # print('i, ks', i, self.ks)

                # \end HONet core
            # print('cnt', cnt1, cnt2, cnt3, cnt4)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('out')
        return x#, self.ks, self.stepsize


class HONet_fix(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, fix_k=True, pretrain=False, num_classes=num_cla,
                 stochastic_depth=False, PL=0.5, noise_level=0.001, noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_fix, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.fix_k = fix_k
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, stepsize=self.stepsize, fix_k=self.fix_k))
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[i]):  # Recalling "MResNet(BasicBlock,
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, fix_k=self.fix_k,
                                        stepsize=self.stepsize))  # three (Basic) Blocks
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, fix_k=self.fix_k, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
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
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318

            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                x, last_res, l_last_res, k = b(x, last_res, l_last_res)
                if isinstance(k, float):
                    self.ks = k
                else:
                    self.ks += k.data

        x = self.bn(x)  # wo BN
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.ks


class HONet_rec(nn.Module):

    def __init__(self, block, layers, k_ini=-9.0 / 5, fix_k=True, pretrain=False, num_classes=num_cla,
                 stochastic_depth=False, PL=1.0, noise_level=0.001, noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.strides = [1, 2, 2]
        super(HONet_rec, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.fix_k = fix_k
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.ks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):  # there are 3 elements in the list like [7,7,7]
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, fix_k=self.fix_k, stepsize=self.stepsize))
                if l == 0 or l == 1:
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
                else:
                    self.l_last_res_planes = self.planes[i] * block.expansion
                    self.last_res_planes = self.planes[i] * block.expansion
                self.in_planes = self.planes[i] * block.expansion
                l += 1
                # print('l', l)
                # print('i', i)
                for j in range(1, layers[i]):  # Recalling "MResNet(BasicBlock,
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, fix_k=self.fix_k,
                                        stepsize=self.stepsize))  # three (Basic) Blocks
                    if l == 0 or l == 1:
                        self.l_last_res_planes = self.last_res_planes
                        self.last_res_planes = self.in_planes
                    else:
                        self.l_last_res_planes = self.planes[i] * block.expansion
                        self.last_res_planes = self.planes[i] * block.expansion
                    l += 1
                    # print('l', l)
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(
                    block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, self.strides[i],
                          k_ini=self.k_ini, fix_k=self.fix_k, stepsize=self.stepsize,
                          death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.l_last_res_planes = self.last_res_planes
                self.last_res_planes = self.in_planes
                self.in_planes = self.planes[i] * block.expansion
                # print('i', i)
                for j in range(1, layers[i]):
                    # print('j', j)
                    blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,
                                        k_ini=self.k_ini, stepsize=self.stepsize,
                                        death_rate=death_rates[i * layers[0] + j]))
                    self.l_last_res_planes = self.last_res_planes
                    self.last_res_planes = self.in_planes
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride):

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        self.ks = []
        x = self.conv1(x)
        last_res = -1
        l_last_res = -1
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
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
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318

            if self.pretrain:  #
                x = b(x) + residual

            else:  # in.channel = out.channel and not pretrain
                ## rec
                x_size_pre = x.size()
                last_res_size_pre = last_res.size()
                l_last_res_size = l_last_res.size()
                x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                if x.size() == last_res.size() and x.size() == l_last_res.size() and x_size_pre == x.size() and last_res_size_pre == last_res.size() and l_last_res_size == l_last_res.size():
                    # print('i:', i)
                    for rec in range(2):
                        x, last_res, l_last_res, k = b(x, last_res, l_last_res)

                ## rec
                # x, last_res, l_last_res, k = b(x, last_res, l_last_res)
                if isinstance(k, float):
                    self.ks = k
                else:
                    self.ks += k.data

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.ks, self.stepsize


class HONet_single(nn.Module):

    def __init__(self, block, layers, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0,
                 noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(HONet_single, self).__init__()
        self.noise = noise  # what for?
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        # self.ks = nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(1, 1.1)) for i in
        #                             range(layers[0] + layers[1] + layers[2])])  # each layer has a trainable $k_n$
        self.ks = nn.Parameter(torch.Tensor(1).uniform_(-9.0 / 5, -9.0 / 5))  # all layer share a trainable $k$
        self.stochastic_depth = stochastic_depth
        blocks = []
        n = layers[0] + layers[1] + layers[2]

        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[
                    i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 0 to 2
                    blocks.append(block(self.in_planes, self.planes[i]))  # three (Basic) Blocks
        else:  # with death_rates
            death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
            # print(death_rates)
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.strides[i],
                                    death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
                self.in_planes = self.planes[i] * block.expansion
                for j in range(1, layers[i]):
                    blocks.append(block(self.in_planes, self.planes[i], death_rate=death_rates[i * layers[0] + j]))
        self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
        self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride=2):
        self.downsample_16_32 = Downsample_clean(16 * block.expansion, 32 * block.expansion, 2)
        self.downsample_32_64 = Downsample_clean(32 * block.expansion, 64 * block.expansion)
        self.downsample_64_128 = Downsample_clean(64, 128)
        self.downsample_128_256 = Downsample_clean(128, 256)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):  # if m is a conv
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def Downsample_ins(self, ):
        Downsample_real()

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
        else:
            residual = x
        x = self.blocks[0](x) + residual
        l_last_res = residual

        ### \begin
        '''
        if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
            residual = self.downsample1(x)
        else:
            residual = x
            '''
        residual = x
        x = self.blocks[1](x) + residual
        last_res = residual
        # residual = x # moved from below. Flag:318
        ### \end

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                # print('i', i)
                continue
            residual = x  # moved up. Flag:318

            if self.pretrain:  #
                x = b(x) + residual
            else:  # in.channel = out.channel and not pretrain
                # \begin HONet core
                # print('ks[i], i', self.ks[i].data, i)
                F_x_n = b(x)
                # b_0 = (3*self.ks-1)/(self.ks*2)
                b_0 = (3 * self.ks[i] - 1) / (self.ks[i] * 2)
                # b_0 = b_0.expand_as(F_x_n)
                # a_0 = (3*self.ks+3)/(self.ks*4)
                a_0 = (3 * self.ks[i] + 3) / (self.ks[i] * 4)
                # a_0 = b_0.expand_as(residual)
                # a_1 = -1/ (self.ks )
                a_1 = -1 / (self.ks[i])
                # a_1 = a_1.expand_as(last_res)
                # a_2 = (self.ks +1)/(4*self.ks)
                a_2 = (self.ks[i] + 1) / (4 * self.ks[i])
                # a_2 = a_2.expand_as(l_last_res)
                # print('a_0, a_1, a_2, b_0. init', a_0.size(), a_1.size(), a_2.size(), b_0.size())
                if 1:

                    if F_x_n.size()[1] == 32:
                        if residual.size()[1] == 16:
                            residual = self.downsample_16_32(residual)
                        if last_res.size()[1] == 16:
                            last_res = self.downsample_16_32(last_res)
                        if l_last_res.size()[1] == 16:
                            l_last_res = self.downsample_16_32(l_last_res)

                    if F_x_n.size()[1] == 64:
                        if residual.size()[1] == 32:
                            residual = self.downsample_32_64(residual)
                        if last_res.size()[1] == 32:
                            last_res = self.downsample_32_64(last_res)
                        if l_last_res.size()[1] == 32:
                            l_last_res = self.downsample_32_64(l_last_res)

                    if F_x_n.size()[1] == 128:
                        if residual.size()[1] == 64:
                            residual = self.downsample_64_128(residual)
                        if last_res.size()[1] == 64:
                            last_res = self.downsample_64_128(last_res)
                        if l_last_res.size()[1] == 64:
                            l_last_res = self.downsample_64_128(l_last_res)

                    if F_x_n.size()[1] == 256:
                        if residual.size()[1] == 128:
                            residual = self.downsample_128_256(residual)
                        if last_res.size()[1] == 128:
                            last_res = self.downsample_128_256(last_res)
                        if l_last_res.size()[1] == 128:
                            l_last_res = self.downsample_128_256(l_last_res)

                # if l_last_res.size() != residual.size():  # sizes of the input and output are not the same
                #     # print('a_2.size(1)', a_2.size(1))
                #     # print('a_2.size(1)', a_2.size[1])
                #
                #     if residual.size(1) == 32:
                #         # print("32, a_2", a_2.size())
                #         # a_2 = self.downsample21(a_2)  # [256, 16, 32, 32]->[256, 32, 16, 16]
                #         l_last_res = self.downsample21(l_last_res)
                #         # print("32, a_2", a_2.size())
                #         # if not self.pretrain:
                #         # last_res=self.downsample22(last_res)
                #     elif residual.size(1) == 64:
                #         # print("64, a_2", a_2.size())
                #         # a_2 = self.downsample31(a_2)  # [256, 32, 16, 16]->[256, 64, 8, 8]
                #         l_last_res = self.downsample31(l_last_res)
                #         # print("64, a_2", a_2.size())
                #     elif residual.size(1) == 128:
                #         # print("128, a_2", a_2.size())
                #         # a_2 = self.downsample41(a_2)  # [256, 64, 32, 32]->[256, 128, 16, 16]
                #         l_last_res = self.downsample41(l_last_res)
                #         # print("128, a_2", a_2.size())
                #     elif residual.size(1) == 256:
                #         # print("256, a_2", a_2.size())
                #         # a_2 = self.downsample51(a_2)  # [256, 64, 32, 32]->[256, 128, 16, 16]
                #         l_last_res = self.downsample51(l_last_res)
                #         # print("256, a_2", a_2.size())
                # # print('a_0, a_1, a_2, b_0. updated', a_0.size(), a_1.size(), a_2.size(), b_0.size())

                # x = b_0*F_x_n + a_0*residual + a_1*last_res + a_2*l_last_res
                x = torch.mul(b_0, F_x_n) + torch.mul(a_0, residual) + torch.mul(a_1, last_res) + torch.mul(a_2,
                                                                                                            l_last_res)

                # \end HONet core
            l_last_res = last_res
            last_res = residual

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.ks


# class HONet_fix(nn.Module):
#
#     def __init__(self, block, layers, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=0.5, noise_level=0.001,
#                  noise=False, a_0=1.0/3, a_1=5.0/9, a_2=1.0/9, b_0=16.0/9):
#     # def __init__(self, block, layers, pretrain=True, num_classes=num_cla, stochastic_depth=False, PL=0.5,
#     #                  noise_level=0.001, noise=False, a_0=1.0 / 3, a_1=5.0 / 9, a_2=1.0 / 9, b_0=16.0 / 9):
#         self.in_planes = 16
#         self.planes = [16, 32, 64]
#         self.strides = [1, 2, 2]
#         super(HONet_fix, self).__init__()
#         self.noise = noise  # what for?
#         self.block = block
#         self.a_0 = a_0
#         self.a_1 = a_1
#         self.a_2 = a_2
#         self.b_0 = b_0
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.pretrain = pretrain
#         # self.ks = nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.1)) for i in
#         #                             range(layers[0] + layers[1] + layers[2])])  # each layer has a trainable $k_n$
#         self.stochastic_depth = stochastic_depth
#         blocks = []
#         n = layers[0] + layers[1] + layers[2]
#
#         if not self.stochastic_depth:
#             for i in range(3):
#                 blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
#                 self.in_planes = self.planes[i] * block.expansion
#                 for j in range(1, layers[
#                     i]):  # Recalling "MResNet(BasicBlock,[3,3,3],**kwargs)", and "layers" is assigned as "[3,3,3]"; then j is 0 to 2
#                     blocks.append(block(self.in_planes, self.planes[i]))  # three (Basic) Blocks
#         else:  # with death_rates
#             death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]  # n is the sum of elements of "[3,3,3]"
#             # print(death_rates)
#             for i in range(3):
#                 blocks.append(block(self.in_planes, self.planes[i], self.strides[i],
#                                     death_rate=death_rates[i * layers[0]]))  # note that layers[k] == layers[j]
#                 self.in_planes = self.planes[i] * block.expansion
#                 for j in range(1, layers[i]):
#                     blocks.append(block(self.in_planes, self.planes[i], death_rate=death_rates[i * layers[0] + j]))
#         self.blocks = nn.ModuleList(blocks)  # ModuleList cannot determine the sequence of layers
#         self.downsample1 = Downsample(16, 64, stride=1)  # Downsample: (in_planes,out_planes,stride=2):
#         # self.downsample1=nn.Conv2d(16, 64,
#         #                    kernel_size=1, stride=1, bias=False)
#         self.downsample21 = Downsample(16 * block.expansion,
#                                        32 * block.expansion)  # "expansion" is 1 for BasicBlocks and is 4 for the Bottleneck
#         # self.downsample22=Downsample(16*block.expansion,32*block.expansion)
#         self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
#         # self.downsample32=Downsample(32*block.expansion,64*block.expansion)
#
#         self.downsample41 = Downsample(64, 128)
#         self.downsample51 = Downsample(128, 256)
#
#         self.bn = nn.BatchNorm2d(64 * block.expansion)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64 * block.expansion, num_classes)
#
#         for m in self.modules():  # initialization
#             if isinstance(m, nn.Conv2d):  # if m is a conv
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # element num of the kernel
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def change_state(self):
#         self.pretrain = not self.pretrain
#
#     def forward(self, x):
#
#
#         x = self.conv1(x)
#         # x=self.bn1(x)
#         # x=self.relu(x)
#
#         if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
#             residual = self.downsample1(x)
#         else:
#             residual = x
#         x = self.blocks[0](x) + residual
#         l_last_res = residual
#
#         ### \begin
#         '''
#         if self.block.expansion == 4:  # 4 is the "expansion" of the "Bottleneck". If "Bottleneck" is used, we need to downsample
#             residual = self.downsample1(x)
#         else:
#             residual = x
#             '''
#         residual = x
#         x = self.blocks[1](x) + residual
#         last_res = residual
#         # residual = x # moved from below. Flag:318
#         ### \end
#
#
#         for i, b in enumerate(self.blocks):  # index and content
#             if i == 0 or i==1:
#                 continue
#             residual = x # moved up. Flag:318
#
#             if b.in_planes != b.planes * b.expansion:  # sizes of the input and output are not the same
#                 # print("1")
#                 # print('b.in_planes != b.planes * b.expansion', b.in_planes, b.planes, b.expansion)
#                 if b.planes == 32:
#                     # print("32, x", x.size())
#                     residual = self.downsample21(x)#[256, 16, 32, 32]->[256, 32, 16, 16]
#                     # print("32, residual", residual.size())
#                     # if not self.pretrain:
#                     # last_res=self.downsample22(last_res)
#                 elif b.planes == 64:
#                     # print("64, x", x.size())
#                     residual = self.downsample31(x)#[256, 32, 16, 16]->[256, 64, 8, 8]
#                     # print("64, residual", residual.size())
#                     # if not self.pretrain:
#                     # last_res=self.downsample32(last_res)
#                 elif b.planes == 128:
#                     # print("128, x", x.size())
#                     residual = self.downsample41(x)  # [256, 32, 16, 16]->[256, 64, 8, 8]
#                     # print("64, residual", residual.size())
#                     # if not self.pretrain:
#                     # last_res=self.downsample32(last_res)
#                 x = b(x)
#                 # print(x.size())
#                 # print(residual.size())
#                 x += residual
#
#             elif self.pretrain:  #
#                 x = b(x) + residual
#                 # print("2")
#             else:  # in.channel = out.channel and not pretrain
#                 # print("3")
#                 # \begin HONet core
#                 # print('a_0, a_1, a_2, b_0. init', self.a_0, self.a_1, self.a_2, self.b_0)
#                 F_x_n = b(x)
#                 #b_0 = 16.0/9
#                 # b_0 = self.b_0.expand_as(F_x_n)#.cuda()
#
#                 #a_0 = 1.0/3
#                 # a_0 = self.a_0.expand_as(residual)#.cuda()
#
#                 #a_1 = 5.0/9
#                 # a_1 = self.a_1.expand_as(last_res)#.cuda()
#
#                 #a_2 = 1.0/9
#                 # a_2 = self.a_2.expand_as(l_last_res)#.cuda()
#                 # print('a_0, a_1, a_2, b_0. init', a_0.size(), a_1.size(), a_2.size(), b_0.size())
#                 if l_last_res.size() != residual.size():  # sizes of the input and output are not the same
#                     # print('a_2.size(1)', a_2.size(1))
#                     # print('a_2.size(1)', a_2.size[1])
#
#                     if residual.size(1) == 32:
#                         # print("32, a_2", a_2.size())
#                         # print('a_2', l_last_res)
#                         # a_2 = self.downsample21(a_2)  # [256, 16, 32, 32]->[256, 32, 16, 16]
#                         # print('a_2_down', l_last_res)
#                         l_last_res = self.downsample21(l_last_res)
#                         # print("32, a_2", a_2.size())
#                         # if not self.pretrain:
#                         # last_res=self.downsample22(last_res)
#                     elif residual.size(1) == 64:
#                         # print("64, a_2", a_2.size())
#                         # a_2 = self.downsample31(a_2)  # [256, 32, 16, 16]->[256, 64, 8, 8]
#                         l_last_res = self.downsample31(l_last_res)
#                         # print("64, a_2", a_2.size())
#                     elif residual.size(1) == 128:
#                         # print("128, a_2", a_2.size())
#                         # a_2 = self.downsample41(a_2)  # [256, 64, 32, 32]->[256, 128, 16, 16]
#                         l_last_res = self.downsample41(l_last_res)
#                         # print("128, a_2", a_2.size())
#                     elif residual.size(1) == 256:
#                         # print("256, a_2", a_2.size())
#                         # a_2 = self.downsample51(a_2)  # [256, 64, 32, 32]->[256, 128, 16, 16]
#                         l_last_res = self.downsample51(l_last_res)
#                         # print("256, a_2", a_2.size())
#                 # print('a_0, a_1, a_2, b_0. updated', a_0.size(), a_1.size(), a_2.size(), b_0.size())
#
#                 # x = b_0*F_x_n + a_0*residual + a_1*last_res + a_2*l_last_res
#                 x = torch.mul(self.b_0,F_x_n) + torch.mul(self.a_0,residual) + torch.mul(self.a_1,last_res) + torch.mul(self.a_2,l_last_res)
#                 # \end HONet core
#
#                 # # ori
#                 # x = b(x) + self.ks[i].expand_as(residual) * residual + (1 - self.ks[i]).expand_as(
#                 #     last_res) * last_res  # "B.expand_as (A)": expand B in A's shape
#             ### \begin
#             l_last_res = last_res
#             ### \end
#             last_res = residual
#
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x, 9999

class MResNetC(nn.Module):

    def __init__(self, block, layers, pretrain=False, num_classes=num_cla):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(MResNetC, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        ks = []
        for i in range(3):
            ks += [nn.Parameter(torch.Tensor(self.planes[i] * block.expansion, 1, 1).uniform_(-0.1, -0.0)) for j in
                   range(layers[i])]
        self.ks = nn.ParameterList(ks)

        blocks = []
        for i in range(3):
            blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
            self.in_planes = self.planes[i] * block.expansion
            for j in range(1, layers[i]):
                blocks.append(block(self.in_planes, self.planes[i]))
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion, 32 * block.expansion)
        # self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        # self.downsample32=Downsample(32*block.expansion,64*block.expansion)

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

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x
        x = self.blocks[0](x) + residual
        last_res = residual
        if self.training and self.noise:
            x += Variable(torch.FloatTensor(x.size()).cuda().uniform_(0, self.noise_level), requires_grad=False)
        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:
                if b.planes == 32:
                    residual = self.downsample21(x)
                    # if not self.pretrain:
                    # last_res=self.downsample22(last_res)
                elif b.planes == 64:
                    residual = self.downsample31(x)
                    # if not self.pretrain:
                    # last_res=self.downsample32(last_res)
                x = b(x) + residual
            elif self.pretrain:
                x = b(x) + residual
            else:

                x = b(x) + self.ks[i].expand_as(residual) * residual + (1 - self.ks[i]).expand_as(last_res) * last_res
            last_res = residual
            if self.training and self.noise:
                x += Variable(torch.FloatTensor(x.size()).cuda().uniform_(0, self.noise_level), requires_grad=False)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, block, layers, in_planes, planes, stride=2, pretrain=False):
        super(DenseBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.layers = layers
        blocks = []
        blocks.append(block(self.in_planes, self.planes, self.stride))
        for j in range(1, layers):
            blocks.append(block(self.planes, self.planes))

        self.downsample = None
        if in_planes != planes * block.expansion or stride != 1:
            self.downsample = Downsample(in_planes, planes, stride=stride)

        self.ks = (
            nn.ParameterList([nn.Parameter(torch.Tensor(1).uniform_(-0.1, -0.0)) for i in range(layers * layers)]))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residuals = []
        for i, b in enumerate(self.blocks):
            if i == 0 and self.downsample != None:
                residuals.append(self.downsample(x))
            else:
                residuals.append(x)

            residual = (self.ks[i * self.layers + i]).expand_as(residuals[i]) * residuals[i]
            sumk = self.ks[i * self.layers + i].clone()
            for j in range(i):
                residual += (self.ks[i * self.layers + j]).expand_as(residuals[j]) * residuals[j]
                sumk += self.ks[i * self.layers + j]
            x = residual / sumk.expand_as(residual) + b(x)
        return x


class DenseResNet(nn.Module):

    def __init__(self, block, layers, pretrain=False, num_classes=num_cla):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(DenseResNet, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain

        self.denseblock1 = DenseBlock(self.block, layers[0], 16, self.planes[0], 1)

        self.denseblock2 = DenseBlock(self.block, layers[1], self.planes[0], self.planes[1], 2)

        self.denseblock3 = DenseBlock(self.block, layers[2], self.planes[1], self.planes[2], 2)

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

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        x = self.denseblock1(x)
        x = self.denseblock2(x)
        x = self.denseblock3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet_N(nn.Module):

    def __init__(self, block, layers, noise_level=0.001, pretrain=True, num_classes=num_cla):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(ResNet_N, self).__init__()
        self.noise_level = noise_level
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain

        blocks = []
        for i in range(3):
            blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
            self.in_planes = self.planes[i] * block.expansion
            for j in range(1, layers[i]):
                blocks.append(block(self.in_planes, self.planes[i]))
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample22 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        self.downsample32 = Downsample(32 * block.expansion, 64 * block.expansion)

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

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        x = self.blocks[0](x) + residual
        if self.training:
            x += Variable(torch.FloatTensor(x.size()).cuda().normal_(0, self.noise_level), requires_grad=False)
        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:
                if b.planes == 32:
                    residual = self.downsample21(x)

                elif b.planes == 64:
                    residual = self.downsample31(x)

            x = b(x) + residual

            if self.training:
                x += Variable(torch.FloatTensor(x.size()).cuda().uniform_(0, self.noise_level), requires_grad=False)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet_cifar(nn.Module):

    def __init__(self, block, layers, noise_level=0.001, pretrain=True, num_classes=num_cla):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(ResNet_cifar, self).__init__()
        self.noise_level = noise_level
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain

        blocks = []
        for i in range(3):
            blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
            self.in_planes = self.planes[
                                 i] * block.expansion  # the first short cut is with downsample(16->64), then 64 all the way
            for j in range(1, layers[i]):
                blocks.append(block(self.in_planes, self.planes[i]))
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample22 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        self.downsample32 = Downsample(32 * block.expansion, 64 * block.expansion)

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

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        x = self.blocks[0](x) + residual
        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:
                if b.planes == 32:
                    residual = self.downsample21(x)

                elif b.planes == 64:
                    residual = self.downsample31(x)

            x = b(x) + residual

        x = self.bn(x)  # wo BN
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x#, 9999, 999

#
# ### \begin
# def HONet20(**kwargs):
#     return HONet(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def HONet32(**kwargs):
#     return HONet(BasicBlock_cifar, [5, 5, 5], **kwargs)
#
#
# def HONet44(**kwargs):
#     return HONet(BasicBlock_cifar, [7, 7, 7], **kwargs)
#
#
# # def HONet56(**kwargs):
# #     return HONet(BasicBlock_cifar, [9,9,9], **kwargs)
# def HONet56(**kwargs):
#     return HONet(BasicBlock_cifar, [9, 9, 9], **kwargs)
#
#
# def HONet110(**kwargs):
#     return HONet(BasicBlock_cifar, [18, 18, 18], **kwargs)
#
#
# def HONet164(**kwargs):
#     return HONet(Bottleneck_cifar, [18, 18, 18], **kwargs)
#
#
# def HONet328(**kwargs):
#     return HONet(Bottleneck_cifar, [36, 36, 36], **kwargs)
#
#
# def HONet656(**kwargs):
#     return HONet(Bottleneck_cifar, [72, 72, 72], **kwargs)
#
#
# def HONet1312(**kwargs):
#     return HONet(Bottleneck_cifar, [144, 144, 144], **kwargs)
#
#
# def HONet20_fix(**kwargs):
#     return HONet_fix(HOBlock, [3, 3, 3], **kwargs)
#
#
# def HONet32_fix(**kwargs):
#     return HONet_fix(HOBlock, [5, 5, 5], **kwargs)
#
#
# def HONet44_fix(**kwargs):
#     return HONet_fix(HOBlock, [7, 7, 7], **kwargs)
#
#
# def HONet56_fix(**kwargs):
#     return HONet_fix(HOBlock, [9, 9, 9], **kwargs)
#
#
# def HONet110_fix(**kwargs):
#     return HONet_fix(HOBlock, [18, 18, 18], **kwargs)
#
#
# def HONet164_fix(**kwargs):
#     return HONet_fix(HoBottleneck, [18, 18, 18], **kwargs)
#
#
# def HONet326_fix(**kwargs):
#     return HONet_fix(HoBottleneck, [36, 36, 36], **kwargs)
#
#
# def HONet650_fix(**kwargs):
#     return HONet_fix(HoBottleneck, [72, 72, 72], **kwargs)
#
#
# def HONet1298_fix(**kwargs):
#     return HONet_fix(HoBottleneck, [144, 144, 144], **kwargs)
#

def HONet20_v2(**kwargs):
    return HONet_v2(HOBlock, [3, 3, 3], **kwargs)
#
#
def HONet32_v2(**kwargs):
    return HONet_v2(HOBlock, [5, 5, 5], **kwargs)


def HONet44_v2(**kwargs):
    return HONet_v2(HOBlock, [7, 7, 7], **kwargs)
#

def HONet56_v2(**kwargs):
    return HONet_v2(HOBlock, [9, 9, 9], **kwargs)
#
#
def HONet110_v2(**kwargs):
    return HONet_v2(HOBlock, [18, 18, 18], **kwargs)


def HONet164_v2(**kwargs):
    return HONet_v2(HoBottleneck, [18, 18, 18], **kwargs)


def HONet326_v2(**kwargs):
    return HONet_v2(HoBottleneck, [36, 36, 36], **kwargs)  # 36*3*3+2


def HONet650_v2(**kwargs):
    return HONet_v2(HoBottleneck, [72, 72, 72], **kwargs)  # 72*3*3+2


def HONet1298_v2(**kwargs):
    return HONet_v2(HoBottleneck, [144, 144, 144], **kwargs)  # 144*3*3+2

#
# def HONet20_stepsize(**kwargs):
#     return HONet_stepsize(HOBlock, [3, 3, 3], **kwargs)
#
#
# def HONet32_stepsize(**kwargs):
#     return HONet_stepsize(HOBlock, [5, 5, 5], **kwargs)
#
#
# def HONet44_stepsize(**kwargs):
#     return HONet_stepsize(HOBlock, [7, 7, 7], **kwargs)
#
#
# def HONet56_stepsize(**kwargs):
#     return HONet_stepsize(HOBlock, [9, 9, 9], **kwargs)
#
#
# def HONet110_stepsize(**kwargs):
#     return HONet_stepsize(HOBlock, [18, 18, 18], **kwargs)
#
#
# def HONet164_stepsize(**kwargs):
#     return HONet_stepsize(HoBottleneck, [18, 18, 18], **kwargs)
#
#
# def HONet326_stepsize(**kwargs):
#     return HONet_stepsize(HoBottleneck, [36, 36, 36], **kwargs)  # 36*3*3+2
#
#
# def HONet650_stepsize(**kwargs):
#     return HONet_stepsize(HoBottleneck, [72, 72, 72], **kwargs)  # 72*3*3+2
#
#
# def HONet1298_stepsize(**kwargs):
#     return HONet_stepsize(HoBottleneck, [144, 144, 144], **kwargs)  # 144*3*3+2
#
#
# def HONet20_given(**kwargs):
#     return HONet_given(HOBlock, [3, 3, 3], **kwargs)
#
#
# def HONet32_given(**kwargs):
#     return HONet_given(HOBlock, [5, 5, 5], **kwargs)
#
#
# def HONet44_given(**kwargs):
#     return HONet_given(HOBlock, [7, 7, 7], **kwargs)
#
#
# def HONet56_given(**kwargs):
#     return HONet_given(HOBlock, [9, 9, 9], **kwargs)
#
#
# def HONet110_given(**kwargs):
#     return HONet_given(HOBlock, [18, 18, 18], **kwargs)
#
#
# def HONet164_given(**kwargs):
#     return HONet_given(HoBottleneck, [18, 18, 18], **kwargs)
#
#
# def HONet326_given(**kwargs):
#     return HONet_given(HoBottleneck, [36, 36, 36], **kwargs)  # 36*3*3+2
#
#
# def HONet650_given(**kwargs):
#     return HONet_given(HoBottleneck, [72, 72, 72], **kwargs)  # 72*3*3+2
#
#
# def HONet1298_given(**kwargs):
#     return HONet_given(HoBottleneck, [144, 144, 144], **kwargs)  # 144*3*3+2
#
#
# def HONet20_rec(**kwargs):
#     return HONet_rec(HOBlock, [3, 3, 3], **kwargs)
#
#
# def HONet32_rec(**kwargs):
#     return HONet_rec(HOBlock, [5, 5, 5], **kwargs)
#
#
# def HONet44_rec(**kwargs):
#     return HONet_rec(HOBlock, [7, 7, 7], **kwargs)
#
#
# def HONet56_rec(**kwargs):
#     return HONet_rec(HOBlock, [9, 9, 9], **kwargs)
#
#
# def HONet110_rec(**kwargs):
#     return HONet_rec(HOBlock, [18, 18, 18], **kwargs)
#
#
# def HONet164_rec(**kwargs):
#     return HONet_rec(HoBottleneck, [18, 18, 18], **kwargs)
#
#
# def HONet326_rec(**kwargs):
#     return HONet_rec(HoBottleneck, [36, 36, 36], **kwargs)
#
#
# def HONet650_rec(**kwargs):
#     return HONet_rec(HoBottleneck, [72, 72, 72], **kwargs)
#
#
# def HONet1298_rec(**kwargs):
#     return HONet_rec(HoBottleneck, [144, 144, 144], **kwargs)
#
#
# def MResNet20(**kwargs):
#     return MResNet(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def MResNet32(**kwargs):
#     return MResNet(BasicBlock_cifar, [5, 5, 5], **kwargs)
#
#
# def MResNet44(**kwargs):
#     return MResNet(BasicBlock_cifar, [7, 7, 7], **kwargs)
#
#
# def MResNet56(**kwargs):
#     return MResNet(BasicBlock_cifar, [9, 9, 9], **kwargs)
#
#
# def MResNet110(**kwargs):
#     return MResNet(BasicBlock_cifar, [18, 18, 18], **kwargs)
#
#
# def MResNet164(**kwargs):
#     return MResNet(Bottleneck_cifar, [18, 18, 18], **kwargs)
#
#
# def MResNet326(**kwargs):
#     return MResNet(Bottleneck_cifar, [36, 36, 36], **kwargs)
#
#
# def MResNet650(**kwargs):
#     return MResNet(Bottleneck_cifar, [72, 72, 72], **kwargs)
#
#
# def MResNet1298(**kwargs):
#     return MResNet(Bottleneck_cifar, [144, 144, 144], **kwargs)
#
#
# def ResNet_20_cifar(**kwargs):
#     return ResNet_cifar(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def ResNet_32_cifar(**kwargs):
#     return ResNet_cifar(BasicBlock_cifar, [5, 5, 5], **kwargs)
#
#
# def ResNet_44_cifar(**kwargs):
#     return ResNet_cifar(BasicBlock_cifar, [7, 7, 7], **kwargs)
#
#
# def ResNet_56_cifar(**kwargs):
#     return ResNet_cifar(BasicBlock_cifar, [9, 9, 9], **kwargs)  # 9*3*2+2
#
#
# def ResNet_110_cifar(**kwargs):
#     return ResNet_cifar(BasicBlock_cifar, [18, 18, 18], **kwargs)  # 18*3*2+2
#
#
# def ResNet_164_cifar(**kwargs):
#     return ResNet_cifar(Bottleneck_cifar, [18, 18, 18], **kwargs)  # 18*3*3+2
#
#
# def ResNet_326(**kwargs):
#     return ResNet(Bottleneck_cifar, [36, 36, 36], **kwargs)  # 36*3*3+2
#
#
# def ResNet_650_cifar(**kwargs):
#     return ResNet_cifar(Bottleneck_cifar, [72, 72, 72], **kwargs)  # 72*3*3+2
#
#
# def ResNet_1298_cifar(**kwargs):
#     return ResNet_cifar(Bottleneck_cifar, [144, 144, 144], **kwargs)  # 144*3*3+2
#
#
# def ResNet_N20(**kwargs):
#     return ResNet_N(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def ResNet_N110(**kwargs):
#     return ResNet_N(BasicBlock_cifar, [18, 18, 18], **kwargs)
#
#
# def MResNetSD20(**kwargs):
#     return MResNet(BasicBlockWithDeathRate, [3, 3, 3], stochastic_depth=True, **kwargs)
#
#
# def MResNetSD110(**kwargs):
#     return MResNet(BasicBlockWithDeathRate, [18, 18, 18], stochastic_depth=True, **kwargs)
#
#
# def MResNetC20(**kwargs):
#     return MResNetC(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def MResNetC32(**kwargs):
#     return MResNetC(BasicBlock_cifar, [5, 5, 5], **kwargs)
#
#
# def MResNetC44(**kwargs):
#     return MResNetC(BasicBlock_cifar, [7, 7, 7], **kwargs)
#
#
# def MResNetC56(**kwargs):
#     return MResNetC(BasicBlock_cifar, [9, 9, 9], **kwargs)
#
#
# def DenseResNet20(**kwargs):
#     return DenseResNet(BasicBlock_cifar, [3, 3, 3], **kwargs)
#
#
# def DenseResNet110(**kwargs):
#     return DenseResNet(BasicBlock_cifar, [18, 18, 18], **kwargs)





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class Bottleneck_cifar(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_cifar, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class PreResNet(nn.Module):

    def __init__(self, block, layers, noise_level=0.001, pretrain=True, num_classes=num_cla):
        self.in_planes = 64
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super(PreResNet, self).__init__()
        self.noise_level = noise_level
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain

        blocks = []
        for i in range(3):
            blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
            self.in_planes = self.planes[
                                 i] * block.expansion  # the first short cut is with downsample(16->64), then 64 all the way
            for j in range(1, layers[i]):
                blocks.append(block(self.in_planes, self.planes[i]))
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample22 = Downsample(16 * block.expansion, 32 * block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        self.downsample32 = Downsample(32 * block.expansion, 64 * block.expansion)

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

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x

        x = self.blocks[0](x) + residual
        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:
                if b.planes == 32:
                    residual = self.downsample21(x)

                elif b.planes == 64:
                    residual = self.downsample31(x)

            x = b(x) + residual

        x = self.bn(x)  # wo BN
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x#, 9999, 999

# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=num_cla00, zero_init_residual=False):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
#
# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     return model
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model
#
#
# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model


# def demo():
#     net = resnet50(num_classes=num_cla0)
#     y = net(torch.randn(1, 3, 224,224))
#     print(y.size())

# demo()

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion*planes: # "expansion*planes" is the real output channel number
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = functional.relu(self.bn1(x))

        if hasattr(self, 'shortcut'): #if self include "shortcut", return True. It means downsample
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)
        out = out + shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = functional.relu(self.bn1(x))
        input_out = out

        out = self.conv1(out)
        out = self.bn2(out)
        out = functional.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        out = functional.relu(out)
        out = self.conv3(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
        else:
            shortcut = x

        out = out + shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=num_cla, dataset="CIFAR"):
        super(PreActResNet, self).__init__()

        self.in_planes = 64
        self.dataset = dataset

        if dataset == "CIFAR":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_classes = num_cla
        elif dataset == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000


        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if dataset == "CIFAR":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "ImageNet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # print('self.in_planes', self.in_planes)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.dataset == "ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR":
            out = self.linear(out)
        elif self.dataset == "ImageNet":
            out = self.fc(out)
        return out#, 9999, 999

class HOBlock_IN(nn.Module): #actually, this is the preact block
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, k_ini=-9.0/5,  share_k=False, stepsize=1, given_ks=None, downsample=None,                  pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):
        super(HOBlock_IN,self).__init__()
        self.bn1=nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm2d(planes)# 20210803
        self.stride=stride
        self.in_planes=in_planes
        self.planes=planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.share_k = share_k
        self.given_ks = given_ks
        print('self.share_k, self.given_ks', self.share_k, self.given_ks)
        if self.given_ks is not None:
            print('given_ks')
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        elif self.share_k is True:
            self.k = k_ini
            print('share_k')
        else:
            print('no_share_k')
            self.k =nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks = nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # print('l_last_res_planes, last_res_planes, in_planes, planes', l_last_res_planes, last_res_planes, in_planes, planes)


        if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
        # if 1:
            if self.in_planes != self.expansion*planes:
                self.shortcut_x = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
            if self.last_res_planes != self.expansion*planes:
                self.shortcut_l = nn.Sequential(
                    nn.Conv2d(last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
            if self.l_last_res_planes != self.expansion*planes:
                self.shortcut_ll = nn.Sequential(
                    nn.Conv2d(l_last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )

            # if self.planes == 32:
            #     if self.in_planes ==16:
            #         self.downsample_16_32_x = Downsample_clean(16, 32, 2)
            #         # print('downsample_16_32_x')
            #     if self.last_res_planes ==16:
            #         self.downsample_16_32_l = Downsample_clean(16, 32, 2)
            #         # print('downsample_16_32_l')
            #     if self.l_last_res_planes == 16:
            #         self.downsample_16_32_ll = Downsample_clean(16, 32, 2)
            #         # print('downsample_16_32_ll')
            # if self.planes == 64:
            #     if self.in_planes ==32:
            #         self.downsample_32_64_x = Downsample_clean(32, 64, 2)
            #     if self.last_res_planes ==32:
            #         self.downsample_32_64_l = Downsample_clean(32, 64, 2)
            #     if self.l_last_res_planes == 32:
            #         self.downsample_32_64_ll = Downsample_clean(32, 64, 2)
            # if self.planes == 128:
            #     if self.in_planes ==64:
            #         self.downsample_64_128_x = Downsample_clean(64, 128, 2)
            #     if self.last_res_planes ==64:
            #         self.downsample_64_128_l = Downsample_clean(64, 128, 2)
            #     if self.l_last_res_planes == 64:
            #         self.downsample_64_128_ll = Downsample_clean(64, 128, 2)
            # if self.planes == 256:
            #     if self.in_planes ==128:
            #         self.downsample_128_256_x = Downsample_clean(128, 256, 2)
            #     if self.last_res_planes ==128:
            #         self.downsample_128_256_l = Downsample_clean(128, 256, 2)
            #     if self.l_last_res_planes == 128:
            #         self.downsample_128_256_ll = Downsample_clean(128, 256, 2)
    def forward(self, inp): # Pre-ResNet

        x = inp[0]
        last_res = inp[1]
        l_last_res = inp[2]

        # residual = x
        # F_x_n=self.bn1(x)# wo BN
        # # F_x_n=x
        # F_x_n=self.relu(F_x_n)
        # F_x_n=self.conv1(F_x_n)
        # F_x_n=self.bn2(F_x_n)
        # F_x_n=self.relu(F_x_n)
        # F_x_n=self.conv2(F_x_n)

        F_x_n = functional.relu(self.bn1(x))
        if hasattr(self, 'shortcut_x'): #if self include "shortcut", return True. It means downsample
            residual = self.shortcut_x(F_x_n)
        else:
            residual = x
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = functional.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)

        # if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
            # print('F_x_n.size(), residual.size(),last_res.size(),l_last_res.size()',  F_x_n.size()[1], residual.size()[1],last_res.size()[1],l_last_res.size()[1])
            # print('planes, in_planes, last_res_planes, l_last_res_planes', self.planes, self.in_planes, self.last_res_planes, self.l_last_res_planes)

        # if hasattr(self, 'shortcut_x'):
        #     residual = self.shortcut_x(residual)
        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
            # print('HO')
        # if 1:
            #TODO: check if the dowsample for the first 2 layers is missing

            if hasattr(self, 'shortcut_l'):
                last_res = self.shortcut_l(last_res)
            if hasattr(self, 'shortcut_ll'):
                l_last_res = self.shortcut_ll(l_last_res)

            #
            #
            # if self.planes == 32:
            #     if self.in_planes ==16:
            #         residual = self.downsample_16_32_x(residual)
            #         # print('residual.size()', residual.size())
            #     if self.last_res_planes ==16:
            #         last_res = self.downsample_16_32_l(last_res)
            #     # print('last_res.size()', last_res.size())
            #     if self.l_last_res_planes == 16:
            #         l_last_res = self.downsample_16_32_ll(l_last_res)
            #         # print('l_last_res.size()', l_last_res.size())
            # if self.planes == 64:
            #     if self.in_planes ==32:
            #         residual = self.downsample_32_64_x(residual)
            #     if self.last_res_planes ==32:
            #         last_res = self.downsample_32_64_l(last_res)
            #     if self.l_last_res_planes == 32:
            #         l_last_res = self.downsample_32_64_ll(l_last_res)
            # if self.planes == 128:
            #     if self.in_planes ==64:
            #         residual = self.downsample_64_128_x(residual)
            #     if self.last_res_planes ==64:
            #         last_res = self.downsample_64_128_l(last_res)
            #     if self.l_last_res_planes == 64:
            #         l_last_res = self.downsample_64_128_ll(l_last_res)
            # if self.planes == 256:
            #     if self.in_planes ==128:
            #         residual = self.downsample_128_256_x(residual)
            #     if self.last_res_planes ==128:
            #         last_res = self.downsample_128_256_l(last_res)
            #     if self.l_last_res_planes == 128:
            #         l_last_res = self.downsample_128_256_ll(l_last_res)
            if self.given_ks is None: # trainable k
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)
                # print("trainable")
            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
            # print('x', x[0][0][0][0])
            # print("self.a_0, self.a_1, self.a_2, self.b_0", self.a_0, self.a_1, self.a_2, self.b_0)
        else:
            # print('res')
            x = F_x_n+residual
        # x = self.bn3(x)
        l_last_res = last_res
        last_res = residual # x means the residual
        # residual = x
        # print('self.k', self.k)
        # self.k = self.k*torch.ones([1,1])
        # print('self.k', self.k)

        out = [x]+ [last_res]+ [l_last_res]+ [self.k]

        # tem = []
        # return x, last_res, l_last_res, self.k
        # print("out", out )
        return out

class HOBottleneck_IN(nn.Module): #actually, this is the preact block
    expansion = 4
    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, k_ini=-9.0/5, share_k=False, stepsize=1, given_ks=None, downsample=None):
        super(HOBottleneck_IN,self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # self.bn3 = nn.BatchNorm2d(planes)# 20210803
        self.stride=stride
        self.expansion = 4
        self.in_planes=in_planes
        self.planes=planes
        self.last_res_planes = last_res_planes
        self.l_last_res_planes = l_last_res_planes
        self.stepsize = stepsize
        self.share_k = share_k
        self.given_ks = given_ks
        if self.given_ks is not None:
            self.k = k_ini
            self.a_0 = float(given_ks[0])
            self.a_1 = float(given_ks[1])
            self.a_2 = float(given_ks[2])
            self.b_0 = float(given_ks[3])
        elif self.share_k:
            self.k = k_ini
        else:
            self.k =nn.Parameter(torch.Tensor(1).uniform_(k_ini, k_ini))
        # self.ks = nn.ParameterList(torch.Tensor(1).uniform_(1.0, 1.1))
        # print('l_last_res_planes, last_res_planes, in_planes, planes*self.expansion', l_last_res_planes, last_res_planes, in_planes, planes*self.expansion)
        # print('self.in_planes, planes, self.expansion*planes', self.in_planes, planes, self.expansion*planes)
        if self.in_planes != self.expansion*planes:
            self.shortcut_x = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        # if not (self.last_res_planes == -1 or self.l_last_res_planes == -1):
        # if 1:

                #TODO: make sure the planes are correct
            # elif self.in_planes != planes:
            #     self.shortcut_x = nn.Sequential(
            #         nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            #     )
        if not self.last_res_planes == -1:
            if self.last_res_planes != self.expansion*planes:
                self.shortcut_l = nn.Sequential(
                    nn.Conv2d(last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )
        if not self.l_last_res_planes == -1:
            if self.l_last_res_planes != self.expansion*planes:
                self.shortcut_ll = nn.Sequential(
                    nn.Conv2d(l_last_res_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                )


    def forward(self, inp): # Pre-ResNet

        x = inp[0]
        last_res = inp[1]
        l_last_res = inp[2]
        # residual = x
        # F_x_n=self.bn1(x)# wo BN
        # # F_x_n=x
        # F_x_n=self.relu(F_x_n)
        # F_x_n=self.conv1(F_x_n)
        # F_x_n=self.bn2(F_x_n)
        # F_x_n=self.relu(F_x_n)
        # F_x_n=self.conv2(F_x_n)

        F_x_n = functional.relu(self.bn1(x))
        residual = F_x_n

        F_x_n = self.conv1(F_x_n)
        F_x_n = self.bn2(F_x_n)
        F_x_n = functional.relu(F_x_n)

        F_x_n = self.conv2(F_x_n)
        F_x_n = self.bn3(F_x_n)

        F_x_n = functional.relu(F_x_n)
        F_x_n = self.conv3(F_x_n)
        if hasattr(self, 'shortcut_x'):
            residual = self.shortcut_x(residual)
            # print('shortcut_x')
        else:
            residual = x

        # if hasattr(self, 'shortcut_x'):
        #     residual = self.shortcut_x(residual)
        if hasattr(self, 'shortcut_l'):
            last_res = self.shortcut_l(last_res)
        if hasattr(self, 'shortcut_ll'):
            l_last_res = self.shortcut_ll(l_last_res)
        if not (isinstance(last_res,int) or isinstance(l_last_res,int)):
            # print('HO')
        # if 1:
            #TODO: check if the dowsample for the first 2 layers is missing

            # print('l_last_res.size()[1], last_res.size()[1], residual.size()[1], F_x_n.size()[1]: ', l_last_res.size()[1], last_res.size()[1], residual.size()[1], F_x_n.size()[1])
            # if hasattr(self, 'shortcut_l'):
            #     last_res = self.shortcut_l(last_res)
            # if hasattr(self, 'shortcut_ll'):
            #     l_last_res = self.shortcut_ll(l_last_res)

            if self.given_ks is None: # trainable k
                self.b_0 = (3 * self.k - 1) / (self.k * 2)
                self.a_0 = (3 * self.k + 3) / (self.k * 4)
                self.a_1 = -1 / (self.k)
                self.a_2 = (self.k + 1) / (4 * self.k)
                # print("trainable")

            x = torch.mul(self.stepsize, torch.mul(self.b_0, F_x_n)) + torch.mul(self.a_0, residual) + torch.mul(self.a_1, last_res) + torch.mul(self.a_2, l_last_res)
            # print('x', x[0][0][0][0])
            # print("self.a_0, self.a_1, self.a_2, self.b_0", self.a_0, self.a_1, self.a_2, self.b_0)
        else:
            # print('res')
            # print('F_x_n.size(), residual.size(): ', F_x_n.size(), residual.size())
            x = F_x_n+residual
        # x = self.bn3(x)
        l_last_res = last_res
        last_res = residual # x means the residual
        # residual = x
        # print('self.k', self.k)
        # self.k = self.k*torch.ones([1,1])
        # print('self.k', self.k)

        out = [x]+ [last_res]+ [l_last_res]+ [self.k]

        # tem = []
        # return x, last_res, l_last_res, self.k
        # print("out", out )
        return out

class HONet_IN(nn.Module):
    def __init__(self, block, num_blocks, dataset="CIFAR", k_ini=-9.0 / 5, share_k=False, given_ks=None,
                 pretrain=False, num_classes=num_cla, stochastic_depth=False,
                 PL=1.0, noise_level=0.001,
                 noise=False):
    # def __init__(self, block, num_blocks, dataset="CIFAR", k_ini=-9.0 / 5, fix_k=True,
    #                  given_ks=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9],
    #                  pretrain=False, num_classes=num_cla, stochastic_depth=False,
    #                  PL=1.0, noise_level=0.001,
    #                  noise=False):
        super(HONet_IN, self).__init__()
        self.last_res_planes = -1
        self.l_last_res_planes = -1
        self.noise = noise  # what for?
        self.block = block
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        self.stochastic_depth = stochastic_depth
        self.k_ini = k_ini
        self.share_k = share_k
        self.given_ks = given_ks
        self.stepsize = nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        # blocks = []
        self.ks = []
        self.l = 0

        self.in_planes = 64
        self.dataset = dataset
        if self.share_k:
            self.k_ini = nn.Parameter(torch.Tensor(1).uniform_(-9.0 / 5, -9.0 / 5))  # all layer share a trainable $k$
        if dataset == "CIFAR":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_classes = num_cla
        elif dataset == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if dataset == "CIFAR":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "ImageNet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []



        # self.last_res_planes = self.in_planes

        for stride in strides:
            # print('self.l: ', self.l)
            layers.append(block(self.in_planes, planes, self.last_res_planes, self.l_last_res_planes, stride,
                          k_ini=self.k_ini, stepsize=self.stepsize, share_k=self.share_k, given_ks=self.given_ks))

            # if self.l == 0: # or self.l == 1:
            #     self.l_last_res_planes = self.last_res_planes
            #     self.last_res_planes = planes * block.expansion
            # elif self.l == 1:
            #     self.l_last_res_planes = self.last_res_planes
            #     self.last_res_planes = planes* block.expansion
            # else:
            self.l_last_res_planes = planes * block.expansion
            self.last_res_planes = planes * block.expansion
            self.in_planes = planes * block.expansion
            # self.in_planes = planes * block.expansion
            self.l += 1

            # layers.append(block(self.in_planes, planes, stride))
            # self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.ks = []
        last_res = -1
        l_last_res = -1

        out = self.conv1(x)
        if self.dataset == "ImageNet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
        out = [out]+[last_res]+[l_last_res]
        out = self.layer1(out)
        # print('out[3].size()1: ', out[3].size() )
        self.ks = self.ks+[out[3]]

        out = self.layer2(out)
        # print('out[3].size()2: ', out[3].size() )
        self.ks = self.ks+[out[3]]

        out = self.layer3(out)
        # print('out[3].size()3: ', out[3].size() )
        self.ks = self.ks+[out[3]]

        out = self.layer4(out)
        # print('out[3].size()4: ', out[3].size() )
        self.ks = self.ks+[out[3]]
        # print('self.ks', self.ks)

        out = out[0]
        # self.stepsize = out[2]

        # k = out[3]
        #TODO: extract k from the layers
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR":
            out = self.linear(out)
        elif self.dataset == "ImageNet":
            out = self.fc(out)
        return out#, self.ks, self.stepsize

def honet18_cifar(dataset="CIFAR", **kwargs):
    return HONet_IN(HOBlock_IN, [2,2,2,2], dataset=dataset, **kwargs)
def honet18_in(dataset="ImageNet", **kwargs):
    return HONet_IN(HOBlock_IN, [2,2,2,2], dataset=dataset,**kwargs)
def honet34_cifar(dataset="CIFAR", **kwargs):
    return HONet_IN(HOBlock_IN, [3,4,6,3], dataset=dataset, **kwargs)
def honet34_in(dataset="ImageNet", **kwargs):
    return HONet_IN(HOBlock_IN, [3,4,6,3], dataset=dataset, **kwargs)
def honet50_cifar(dataset = "CIFAR", **kwargs):
    return HONet_IN(HOBottleneck_IN, [3,4,6,3], dataset = dataset, **kwargs)
def honet50_in(dataset = "ImageNet", **kwargs):
    return HONet_IN(HOBottleneck_IN, [3,4,6,3], dataset = dataset, **kwargs)


# def honet101_in(dataset = "ImageNet", **kwargs):
#     return HONet_IN(HOBottleneck_IN, [3,4,23,3], dataset=dataset, PL=args.PL, k_ini=args.k_ini, given_ks=args.given_ks, **kwargs)
# def honet152_in(dataset = "ImageNet", **kwargs):
#     return HONet_IN(HOBottleneck_IN, [3,8,36,3], dataset=dataset, PL=args.PL, k_ini=args.k_ini, given_ks=args.given_ks, **kwargs)

# def se_resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model

# def HONet50_cifar(dataset = "CIFAR"):
#     return HONet_IN(HOBottleneck_IN, [3,4,6,3], dataset = dataset)
# def HONet50_IN(dataset = "ImageNet"):
#     return HONet_IN(HOBottleneck_IN, [3,4,6,3], dataset = dataset)
# def HONet101():
#     return HONet_IN(HOBottleneck_IN, [3,4,23,3])
# def HONet152():
#     return HONet_IN(HOBottleneck_IN, [3,8,36,3])


def PreActResNet18_cifar(dataset="CIFAR"):
    return PreActResNet(PreActBlock, [2,2,2,2], dataset=dataset)

def pre_act_resnet18_in(dataset="ImageNet"):
    return PreActResNet(PreActBlock, [2,2,2,2], dataset=dataset)

def PreActResNet34_cifar(dataset="CIFAR"):
    return PreActResNet(PreActBlock, [3,4,6,3], dataset=dataset)

def pre_act_resnet34_in(dataset="ImageNet"):
    return PreActResNet(PreActBlock, [3,4,6,3], dataset=dataset)

def PreActResNet50_cifar(dataset = "CIFAR"):
    return PreActResNet(PreActBottleneck, [3,4,6,3], dataset = dataset)

def pre_act_resnet50_in(dataset = "ImageNet"):
    return PreActResNet(PreActBottleneck, [3,4,6,3], dataset = dataset)

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])



if __name__ == '__main__':
# # ##
# #     # d = torch.rand(2, 3, 32, 32)
# #     d = torch.randn(2, 3, 224, 224)
# #     # net = HONet20_rec()
# #     # net = HONet164_v2()
# #     # net = HONet650_v2()
# #     # net=ResNet_20()
# #     # net = MResNet164()
# #     # net = ResNet_164()
# #     # net = ResNet_650()
# #     net = resnet18()
# #     o = net(d)
# #     probs = functional.softmax(o).detach().numpy()[0]
# #     pred = np.argmax(probs)
# #     # print('pred, probs', pred, probs)
# #
# #     total_params = sum(p.numel() for p in net.parameters())
# #     print(f'{total_params:,} total parameters.')
# #     total_trainable_params = sum(
# #         p.numel() for p in net.parameters() if p.requires_grad)
# #     print(f'{total_trainable_params:,} training parameters.')
# #     for name, parameters in net.named_parameters():
# #         print(name, ':', parameters.size())
# #     onnx_path = "onnx_model_name_res18_img.onnx"
# #     torch.onnx.export(net, d, onnx_path)
# #     netron.start(onnx_path)
#
# ##
#     # # d = torch.rand(2, 3, 32, 32)
#     # net = ResNet_20_cifar()
#     # o = net(d)
#     # probs = functional.softmax(o).detach().numpy()[0]
#     # pred = np.argmax(probs)
#     # # print('pred, probs', pred, probs)
#     #
#     # total_params = sum(p.numel() for p in net.parameters())
#     # print(f'{total_params:,} total parameters.')
#     # total_trainable_params = sum(
#     #     p.numel() for p in net.parameters() if p.requires_grad)
#     # print(f'{total_trainable_params:,} training parameters.')
#     # for name, parameters in net.named_parameters():
#     #     print(name, ':', parameters.size())
#     # onnx_path = "onnx_model_name_res20_c10.onnx"
#     # torch.onnx.export(net, d, onnx_path)
#     # netron.start(onnx_path)
#
# # ##
# #     d = torch.randn(2, 3, 32, 32)
# #     # net = HONet20_rec()
# #     # net = HONet164_v2()
# #     # net = HONet650_v2()
# #     # net=ResNet_20()
# #     # net = MResNet164()
# #     # net = ResNet_164()
# #     # net = ResNet_650()
# #     net = PreActResNet34_cifar()
# #     o = net(d)
# #     probs = functional.softmax(o).detach().numpy()[0]
# #     pred = np.argmax(probs)
# #     # print('pred, probs', pred, probs)
# #
# #     total_params = sum(p.numel() for p in net.parameters())
# #     print(f'{total_params:,} total parameters.')
# #     total_trainable_params = sum(
# #         p.numel() for p in net.parameters() if p.requires_grad)
# #     print(f'{total_trainable_params:,} training parameters.')
# #     for name, parameters in net.named_parameters():
# #         print(name, ':', parameters.size())
# #     onnx_path = "onnx_model_name_PreActResNet18_cifar.onnx"
# #     torch.onnx.export(net, d, onnx_path)
# #     netron.start(onnx_path)
#
##
    d = torch.randn(2, 3, 224, 224)
    # d = torch.randn(2, 3, 32, 32)

    # net = HONet20_rec()
    # net = HONet20_v2()
    # net = HONet650_v2()
    # net=ResNet_20()
    # net = MResNet164()
    # net = ResNet_164()
    # net = ResNet_650()
    # net = PreActResNet34_IN()
    net = honet50_in()
    # net = pre_act_resnet50_in()
    o = net(d)
    probs = functional.softmax(o).detach().numpy()[0]
    pred = np.argmax(probs)
    # print('pred, probs', pred, probs)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # for name, parameters in net.named_parameters():
    #     print(name, ':', parameters.size())
    # onnx_path = "onnx_model_name_honet152_in.onnx"
    onnx_path = "onnx_model_name_honet34.onnx"

    torch.onnx.export(net, d, onnx_path)
    netron.start(onnx_path)
