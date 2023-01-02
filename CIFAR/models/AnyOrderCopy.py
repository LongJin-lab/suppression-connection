import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
from torch.autograd import Variable
import numpy as np

import torch.onnx
import netron
import onnx
from thop import profile
from torchsummary import summary

from onnx import shape_inference



global num_cla
num_cla = 10


class ZeroSBlockAny(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, stepsize, order=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
        super(ZeroSBlockAny, self).__init__()
        self.order = order
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
            for i in range(self.order-1):
                if self.pre_planes_b[i] != self.planes:
                    if start_DS:
                        self.FixDS = nn.ModuleList([])
                        start_DS = 0
                    self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )

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
        residual = x
        F_x_n = self.bn1(x)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv1(F_x_n)
        F_x_n = self.relu(F_x_n)
        F_x_n = self.conv2(F_x_n)
        F_x_n = self.bn2(F_x_n)

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
                # Fx_n_1 = self.downsample_x(Fx_n_1)
            for i in range(self.order-1):
                if pre_features[i].shape[1] != F_x_n.shape[1]:
                    pre_features[i] = self.FixDS[-1](pre_features[i])
                
                if pre_acts[i].shape[1] != F_x_n.shape[1]:
                    pre_acts[i] = self.FixDS[-1](pre_acts[i])
            sum_features = coesA[0].expand_as(residual)*residual
            
            sum_acts = coesB[0].expand_as(F_x_n)*F_x_n
            
            for i in range(self.order-1):
                sum_features = torch.add( sum_features, coesA[i+1].expand_as(pre_features[i])*pre_features[i] )
                
                sum_acts = torch.add( sum_acts, coesB[i+1].expand_as(pre_acts[i])*pre_acts[i] )  
                  
            x =  torch.add( sum_features, torch.mul(stepsize, -sum_acts ) )
        
            # x =  torch.add( sum_features, torch.mul(self.stepsize, self.coesA[-1].expand_as(F_x_n)*F_x_n) )

        else:
            x = F_x_n
        for i in range(self.order-2, 0, -1): #order-2, order-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1] 

        pre_features[0] = residual
        pre_acts[0] = F_x_n
        # print("self.coesA, self.coesA", self.coesA, self.coesB)
        # coes = self.coesA.extend(self.coesB)
        # print("self.coes", self.coes)

        # self.coes = torch.cat([self.coesA,self.coesB],1)
        return x, pre_features, pre_acts, coesA, coesB

class SamBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, pre_planes, coesA, coesB, stepsize, order=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
        super(SamBlock, self).__init__()
        self.order = order
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
            for i in range(self.order-1):
                if self.pre_planes_b[i] != self.planes:
                    if start_DS:
                        self.FixDS = nn.ModuleList([])
                        start_DS = 0
                    self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes) )

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
    
# class ZeroSBlockAny(nn.Module):
#     expansion = 1

#     # def __init__(self, in_planes, planes, last_res_planes, l_last_res_planes, l_l_last_res_planes, order=3, stride=1, coe_ini=1, fix_coe=False,
#     #              stepsize=1, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], downsample=None):
#     def __init__(self, in_planes, planes, pre_planes, coes, order=3, stride=1, coe_ini=1, fix_coe=False, given_coe=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9]):        
#         super(ZeroSBlockAny, self).__init__()
#         self.order = order
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
#         #     for i in range(self.order+1):
#         #         self.coes.append(float(given_coe[i]))
#         #     # self.a_0 = float(given_coe[0])
#         #     # self.a_1 = float(given_coe[1])
#         #     # self.a_2 = float(given_coe[2])
#         #     # self.b_0 = float(given_coe[3])
#         # else:
#         #     self.coes = nn.Parameter(torch.Tensor(1)).cuda()
#         #     # self.coes[0].data.uniform_(coe_ini, coe_ini)
#         #     torch.nn.init.constant_(self.coes[0], coe_ini)
#         #     for i in range(1,self.order+1,1):
#         #         coe_tem = nn.Parameter(torch.Tensor(1)).cuda()
#         #         torch.nn.init.constant_(coe_tem, coe_ini*i)
#         #         self.coes = torch.cat( (self.coes,  coe_tem), 0)
#         #         # print('self.coes', self.coes)
#         #         # self.coes.append(nn.Parameter(torch.Tensor(1)).cuda() )
#         #         # self.coes[i].data.uniform_(coe_ini, coe_ini)

#             # print('self.pre_planes_b, self.planes:', self.pre_planes_b, self.planes)

#             # for i in range(self.order-1):
#             #     print("self.start1, i ", self.start, i)
#             #     self.start *= (self.pre_planes_b[i] <=0) # maybe can be replaced by < order
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
#             for i in range(self.order-1):
#                 if self.pre_planes_b[i] != self.planes:
#                     if start_DS:
#                         self.FixDS = nn.ModuleList([])
#                         start_DS = 0
#                     # self.FixDS.append( nn.Conv2d(self.pre_planes_b[i].shape[1], self.planes, kernel_size=1, stride=2, padding=0, bias=False) )
#                     self.FixDS.append( Downsample_Fix(self.pre_planes_b[i], self.planes, 2) )
#             # print('not (self.has_ini_block)')
#             # for i in range(self.order-1): 
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
#         # for i in range(self.order):
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
#             # for i in range(self.order-1):
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
#             for i in range(self.order-1):
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
#             for i in range(self.order-1):
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
#         for i in range(self.order-2, 0, -1): #order-2, order-1, ..., 0 #1, 0 
#             # print('i', i)
#             # print('pre_features', pre_features)
#             pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
#         pre_features[0] = residual
#         # pre_features[0] = residual 
#         # x = residual 
#         # for i in range(self.order, 0, -1): #order, order-1, ..., 1
#         #     pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
        
#         return x, pre_features, self.coes

#             # l_last_res = last_res # [1]=[0]
#             # last_res = residual # [0]=residual
#         # self.coes = [[self.a_0]+[self.a_1]+[self.a_2]+[self.a_3]+[self.b_0]]
#         # return x, last_res, l_last_res, l_l_last_res, self.coes

    
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

class Downsample_Fix(nn.Module): 
    def __init__(self,in_planes,out_planes,stride=1):#stride=2):
        super(Downsample_Fix,self).__init__()
        self.downsample_=nn.Sequential(
                    nn.AvgPool2d(2),
                        nn.Conv2d(in_planes, out_planes,
                                kernel_size=1, stride=stride, bias=False)
                        )
        
        # self.FixDS.append( nn.Conv2d(pre_features[i].shape[1], F_x_n.shape[1], kernel_size=1, stride=2, padding=0, bias=False).cuda() )
        # pre_features[i] = self.FixDS[-1](pre_features[i])
        # nn.init.dirac_(self.FixDS[-1].weight, 2)
        # self.FixDS[-1].weight.requires_grad = False
    def forward(self,x):
        x=self.downsample_(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Fix') != -1:
        # p = m.downsample_
        nn.init.dirac_(m.downsample_[1].weight.data, 2)
        # nn.init.dirac_(m.downsample_[0].weight.data, 2)
        # nn.init.constant_(m.bias.data, 0.0)

        
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


class ZeroSAny_Tra(nn.Module):

    def __init__(self, block, layers, order=3, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        # self.planes = [16, 32, 64]
        self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        self.test = []
        for i in range(order-1):
            self.pre_planes += [-i]
        for i in range(order-1):
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
        self.order = order
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

        # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.order+1)]) 
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.order)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.order)]) 

        # if self.share_coe == True:
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, order=self.order, stride=self.strides[i], coe_ini=self.coe_ini))

                if l < order-1:
                    for j in range(order-2,0,-1): # order-1, order-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(order-2,0,-1): # order-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, order=self.order, coe_ini=self.coe_ini))

                    if l < order-1:
                        for j in range(order-2,0,-1): # order-2, order-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(order-2,0,-1): # order-2, order-3, ..., 1
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
        for i in range(self.order-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x


        for j in range(self.order-1):
            x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
            x += residual
            residual = x
            # print('x.shape:', x.shape)


        for i, b in enumerate(self.blocks):  # index and content
            if i < self.order-1:
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
    
        return x, coesA, 1

class SamNet(nn.Module):

    def __init__(self, block, layers, order=3, coe_ini=1, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
                 noise=False):
        self.in_planes = 16
        # self.planes = [16, 32, 64]
        self.planes = [16, 64, 256]
        
        self.pre_planes = [] # the smaller, the closer to the current layer
        self.test = []
        for i in range(order-1):
            self.pre_planes += [-i]
        for i in range(order-1):
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
        self.order = order
        self.coe_ini= coe_ini
        self.ini_stepsize = ini_stepsize
        self.givenA = givenA 
        self.givenB = givenB

        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenA[i], requires_grad=False) for i in range(self.order)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*self.givenB[i], requires_grad=False) for i in range(self.order)]) 

        # if self.share_coe == True:
        # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
        blocks = []
        n = layers[0] + layers[1] + layers[2]
        l = 0
        if not self.stochastic_depth:
            for i in range(3):
                # print('self.pre_planes in net1', self.pre_planes)
                blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB,stepsize=self.ini_stepsize, order=self.order, stride=self.strides[i], coe_ini=self.coe_ini))

                if l < order-1:
                    for j in range(order-2,0,-1): # order-1, order-2, ..., 1
                        self.pre_planes[j] =  self.pre_planes[j-1]
                        
                    self.pre_planes[0] = self.in_planes
                else:
                    for j in range(order-2,0,-1): # order-2, ..., 1
                        self.pre_planes[j] =  self.planes[i] * block.expansion
                    self.pre_planes[0] = self.planes[i] * block.expansion


                self.in_planes = self.planes[i] * block.expansion
                l += 1
                for k in range(1, layers[i]):

                    blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coesA=self.coesA, coesB=self.coesB, stepsize=self.ini_stepsize, order=self.order, coe_ini=self.coe_ini))

                    if l < order-1:
                        for j in range(order-2,0,-1): # order-2, order-3, ..., 1
                            self.pre_planes[j] =  self.pre_planes[j-1]
                        self.pre_planes[0] = self.in_planes
                    else:
                        for j in range(order-2,0,-1): # order-2, order-3, ..., 1
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
        for i in range(self.order-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x


        # for j in range(self.order-1):
        #     x, pre_features, pre_acts, coesA, coesB = self.blocks[j](x, pre_features, pre_acts, self.coesA, self.coesB, self.ini_stepsize)
            
        #     x += residual
        #     residual = x
        #     # print('x.shape:', x.shape)


        for i, b in enumerate(self.blocks):  # index and content
            # if i < self.order-1:
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
    
# class ZeroSAny_Tra(nn.Module):

#     def __init__(self, block, layers, order=3, coe_ini=1, share_coe=False, ini_stepsize=1, pretrain=False, num_classes=num_cla, stochastic_depth=False, PL=1.0, noise_level=0.001,
#                  noise=False):
#         self.in_planes = 16
#         self.planes = [16, 32, 64]
#         # self.last_res_planes = -1
#         # self.l_last_res_planes = -1
#         self.pre_planes = [] # the smaller, the closer to the current layer
#         self.test = []
#         for i in range(order-1):
#             self.pre_planes += [-i]
#         for i in range(order-1):
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
#         self.order = order
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

#         # self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=True) for i in range(self.order+1)]) 
#         self.coes = nn.ParameterList([nn.Parameter(torch.ones(1)*self.given[i], requires_grad=False) for i in range(self.order+1)]) 
#         # if self.share_coe == True:
#         # self.stepsize =nn.Parameter(torch.Tensor(1).uniform_(1, 1))
#         blocks = []
#         n = layers[0] + layers[1] + layers[2]
#         # print('n', n)
#         l = 0
#         if not self.stochastic_depth:
#             for i in range(3):
#                 # print('self.pre_planes in net1', self.pre_planes)
#                 blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coes=self.coes, order=self.order, stride=self.strides[i], coe_ini=self.coe_ini))
#                 # if l ==0 or l==1:
#                 #     self.l_last_res_planes = self.last_res_planes
#                 #     self.last_res_planes = self.in_planes
#                 if l < order-1:
#                     for j in range(order-2,0,-1): # order-1, order-2, ..., 1
#                         self.pre_planes[j] =  self.pre_planes[j-1]
#                         # self.pre_planes[j-2] =  self.pre_planes[j-1]
                        
#                     self.pre_planes[0] = self.in_planes
#                 else:
#                     for j in range(order-2,0,-1): # order-2, ..., 1
#                         self.pre_planes[j] =  self.planes[i] * block.expansion
#                     self.pre_planes[0] = self.planes[i] * block.expansion

#                     # self.l_last_res_planes = self.planes[i] * block.expansion
#                     # self.last_res_planes = self.planes[i] * block.expansion
#                 self.in_planes = self.planes[i] * block.expansion
#                 l += 1
#                 for k in range(1, layers[i]):
#                     # print('self.pre_planes in net2', self.pre_planes)
#                     blocks.append(block(self.in_planes, self.planes[i], self.pre_planes, coes=self.coes, order=self.order, coe_ini=self.coe_ini))
#                     # if l == 0 or l == 1:
#                     #     self.l_last_res_planes = self.last_res_planes
#                     #     self.last_res_planes = self.in_planes
#                     if l < order-1:
#                         for j in range(order-2,0,-1): # order-2, order-3, ..., 1
#                             self.pre_planes[j] =  self.pre_planes[j-1]
#                         self.pre_planes[0] = self.in_planes
#                     else:
#                         for j in range(order-2,0,-1): # order-2, order-3, ..., 1
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
        
#         # for i in range(1,self.order+1,1):
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
#         for i in range(self.order-1):
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
#         for j in range(self.order-1):
#             x, pre_features, coes = self.blocks[j](x, pre_features, self.coes, self.ini_stepsize)
#             x += residual
#             residual = x
#             # print('x.shape:', x.shape)


#         # x, self.pre_features, self.coes = self.blocks[i](x, self.pre_features)
#         # x += residual
#         # residual = x
#         for i, b in enumerate(self.blocks):  # index and content
#             if i < self.order-1:
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

if __name__ == '__main__':
    # w = torch.empty(6, 3, 1, 1)
    # nn.init.dirac_(w,2)
    # # w = torch.empty(3, 24, 5, 5)
    # # nn.init.dirac_(w, 3)
    # print('w', w)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = torch.rand(2, 3, 32, 32).to(device)
    # net = ZeroSAny20_Tra()
    net = SamNet20_Tra()
    
    net.apply(weights_init) 
    net = net.to(device)
    # summary(net, input_size=(3, 32, 32), batch_size=-1)

    # macs, params = profile(net, inputs=(d, ))
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True
    # net = ZeroSNet20_Tra()
    o = net(d)
    # print('net:', net)
    for i in net.named_parameters():
        print(i[0], i[1].shape)
    # for name in net.state_dict():
        # print('name', name)
    
    # onnx_path = "onnx_model_name.onnx"
    # torch.onnx.export(net, d, onnx_path)
    # netron.start(onnx_path)