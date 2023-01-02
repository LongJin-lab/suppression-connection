import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron

global num_cla
num_cla = 10


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
    

class ZeroSBlock_fixDS(nn.Module): 
    expansion = 1

    def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes,  stride=1, coe_ini=1, fix_coe=False, stepsize=1, given_coe=[1.0/3, 5.0/9, 1.0/9, 16.0/9], downsample=None):
        super(ZeroSBlock_fixDS,self).__init__()
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
                self.downsample_x = Downsample_fixDS(self.in_planes, self.planes, 2)
            if self.last_res_planes != self.planes:
                self.downsample_l = Downsample_fixDS(self.last_res_planes, self.planes, 2)
            if self.l_last_res_planes != self.planes:
                self.downsample_ll = Downsample_fixDS(self.l_last_res_planes, self.planes, 2)
                        
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

class Downsample_fixDS(nn.Module): 
    def __init__(self, input, in_planes,out_planes, stride=2):
        super(Downsample_fixDS,self).__init__()
        w = torch.empty(out_planes, in_planes, 1, 1)
        nn.init.ones_(w)
        self.downsample_=nn.Sequential(
                        nn.functional.conv2d(input, w, stride=stride, bias=False)
                        )
    def forward(self,x):
        x=self.downsample_(x)
        return x

class ZeroSNet_Tra(nn.Module):

    def __init__(self, block, layers, coe_ini=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
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
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        self.coes = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,self.strides[i], coe_ini=self.coe_ini))
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

        # else:
        #     death_rates = [i / (n - 1) * (1 - PL) for i in range(n)]
        #     for i in range(3):
        #         blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes,self.strides[i], coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0]]))
        #         self.l_last_res_planes = self.last_res_planes
        #         self.last_res_planes = self.in_planes
        #         self.in_planes = self.planes[i] * block.expansion
        #         for j in range(1, layers[i]):
        #             blocks.append(block(self.in_planes, self.planes[i], self.last_res_planes, self.l_last_res_planes, coe_ini = self.coe_ini, death_rate=death_rates[i * layers[0] + j]))
        #             self.l_last_res_planes = self.last_res_planes
        #             self.last_res_planes = self.in_planes
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
        print('x.shape:', x.shape)

        residual = x
        x, last_res, l_last_res, k = self.blocks[1](x, last_res, l_last_res)
        x += residual
        print('x.shape:', x.shape)

        for i, b in enumerate(self.blocks):  # index and content
            if i == 0 or i == 1:
                continue
            residual = x 
            if self.pretrain:  #
                x = b(x) + residual

            else:  
                x, last_res, l_last_res, k = b(x, last_res, l_last_res)
                self.coes += k.data
            print('x.shape:', x.shape)
            
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print('x.shape:', x.shape)
        return x#, self.coes, 1

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

if __name__ == '__main__':
    d = torch.rand(2, 3, 32, 32)
    net = ZeroSNet20_Tra()
    o = net(d)
    print('net', net)
    onnx_path = "onnx_model_name.onnx"
    torch.onnx.export(net, d, onnx_path)
    netron.start(onnx_path)
