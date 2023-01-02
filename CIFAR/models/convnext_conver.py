""" ConvNeXtConver

Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

Original code and weights from https://github.com/facebookresearch/ConvNeXtConver, original copyright below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# import sys,os
# sys.path.append(os.getcwd())
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from fx_features import register_notrace_module
# from helpers import named_apply, build_model_with_cfg, checkpoint_seq
# from layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp
# from registry import register_model


# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .fx_features import register_notrace_module
from .helpers import named_apply, build_model_with_cfg, checkpoint_seq
from .layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp
from .registry import register_model


__all__ = ['ConvNeXtConver']  # model_registry will add each entrypoint fn to this
# dataset = 'cifar10'
dataset = 'imagenet'

def MeanStd(**kwargs):
    pool_size =  (3, 3)
    if dataset == 'cifar10':
        IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10

    elif dataset == 'cifar100':
        IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
    elif dataset == 'mnist':
        IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD = (0.1307,), (0.3081,)
        num_classes = 10
    elif dataset == 'svhn':
        IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)       
        num_classes = 10
    elif dataset == 'imagenet':
        IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        num_classes = 1000
        pool_size =  (7, 7)
    return IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD,num_classes,pool_size

IMAGE_DEFAULT_MEAN,IMAGE_DEFAULT_STD,num_classes,pool_size = MeanStd()

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': num_classes, 'input_size': (3, 224, 224), 'pool_size': pool_size,
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGE_DEFAULT_MEAN, 'std': IMAGE_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    convnext_conver_tiny=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_tiny_1k_224_ema.pth"),
    convnext_conver_small_narrow=_cfg(url=""),
    convnext_conver_small=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_small_1k_224_ema.pth"),
    convnext_conver_base=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_base_1k_224_ema.pth"),
    convnext_conver_large=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_large_1k_224_ema.pth"),

    convnext_conver_nano_hnf=_cfg(url=''),
    convnext_conver_tiny_hnf=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_conver_tiny_hnf_a2h-ab7e9df2.pth',
        crop_pct=0.95),

    convnext_conver_tiny_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_tiny_22k_1k_224.pth'),
    convnext_conver_small_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_small_22k_1k_224.pth'),
    convnext_conver_base_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_base_22k_1k_224.pth'),
    convnext_conver_large_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_large_22k_1k_224.pth'),
    convnext_conver_xlarge_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_xlarge_22k_1k_224_ema.pth'),

    convnext_conver_tiny_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_tiny_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_conver_small_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_small_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_conver_base_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_base_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_conver_large_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_large_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_conver_xlarge_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_conver_xlarge_22k_1k_384_ema.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    convnext_conver_tiny_in22k=_cfg(
        url="http://dl.fbaipublicfiles.com/convnext/convnext_conver_tiny_22k_224.pth", num_classes=21841),
    convnext_conver_small_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_small_22k_224.pth", num_classes=21841),
    convnext_conver_base_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_base_22k_224.pth", num_classes=21841),
    convnext_conver_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_large_22k_224.pth", num_classes=21841),
    convnext_conver_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_conver_xlarge_22k_224.pth", num_classes=21841),
)


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp

        if 'imagenet' in dataset:
            self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)     
        else:
            self.conv_dw = nn.Conv2d(dim, dim, kernel_size=3, padding='same', groups=dim)  # depthwise conv                
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.end_block_vis = nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        # x = self.end_block_vis(x)
        return x


class ConvNeXtStage(nn.Module):

    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, ls_init_value=1.0, conv_mlp=False,
            norm_layer=None, cl_norm_layer=None, cross_stage=False):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[ConvNeXtBlock(
            dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
            norm_layer=norm_layer if conv_mlp else cl_norm_layer)
            for j in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

# class ConvNeXtBlock_ini(nn.Module):

#     def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None, coesA=[-1,0], coesB=[0,1], coes_stepsize=1.0, IniBlockNum=9999, settings=''):
#         super().__init__()
#         if not norm_layer:
#             norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
#         mlp_layer = ConvMlp if conv_mlp else Mlp
#         self.use_conv_mlp = conv_mlp
#         self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = norm_layer(dim)
#         self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
#         self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.begin_block = nn.Identity()
#         self.end_block = nn.Identity()
        
#     def forward(self, x):
#         x = self.begin_block(x)
#         shortcut = x
#         x = self.conv_dw(x)
#         if self.use_conv_mlp:
#             x = self.norm(x)
#             x = self.mlp(x)
#         else:
#             x = x.permute(0, 2, 3, 1)
#             x = self.norm(x)
#             x = self.mlp(x)
#             x = x.permute(0, 3, 1, 2)
#         if self.gamma is not None:
#             x = x.mul(self.gamma.reshape(1, -1, 1, 1))
#         x = self.drop_path(x) + shortcut
#         x = self.end_block(x)
#         return x

class ConvNeXtConverBlock(nn.Module):

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None, coesA=[-1,0], coesB=[0,1], coes_stepsize=1.0, IsIni=9999, settings='',Layer_idx=None, coesDecay=None):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.settings = settings
        self.use_conv_mlp = conv_mlp
        if 'imagenet' in dataset:
            self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)     
        else:
            print('not imagenet')
            self.conv_dw = nn.Conv2d(dim, dim, kernel_size=3, padding='same', groups=dim)  # depthwise conv          

        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.start_block_vis = nn.Identity()
        self.end_block_vis = nn.Identity()
        self.steps = len(coesA)
        self.coesA = coesA
        # self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*coesA[i], requires_grad=True) for i in range(self.steps)])
        if 'learnCoesB' in settings:
            print('learnCoesB')
            self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*coesB[i], requires_grad=True) for i in range(self.steps)]) 
        else:     
            self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*coesB[i], requires_grad=False) for i in range(self.steps)])  
        
        self.coes_stepsize = coes_stepsize
        self.IsIni = IsIni
        
        self.order = len(self.coesA)
        if 'LearnBal' in self.settings:
            self.coesBalance = nn.Parameter(torch.zeros(1), requires_grad = True)
        else:
            self.coesBalance = nn.Parameter(torch.zeros(1), requires_grad = False)
        if 'ShareExpDecay' in self.settings:
            self.coesDecay_ = coesDecay#nn.Parameter(torch.ones(1)*(coesDecay))
        else:
            self.coesDecay_ = nn.Parameter(torch.ones(1)*coesDecay)
        self.Layer_idx = Layer_idx

        self.coes_stepsize = 1.0
        if 'BiasExp' in self.settings:
            self.bias_inner = nn.Parameter(torch.zeros(1), requires_grad = True)
            self.bias_outer = nn.Parameter(torch.zeros(1), requires_grad = True)
        else:
            self.bias_inner = nn.Parameter(torch.zeros(1), requires_grad = False)
            self.bias_outer = nn.Parameter(torch.zeros(1), requires_grad = False)
            
    def forward(self, all):
        # print('all',all)
        # if self.IsIni:
        #     x = all
        # else:
        
        x_k = all[0]
        pre_features = all[1]
        pre_acts = all[2]
        if isinstance(pre_features,int):
            pre_features = [pre_features]
            pre_acts = [pre_acts]
        # print('pre_features',isinstance(pre_features,int)) 
        x_k = self.start_block_vis(x_k)
        shortcut = x_k
        if 'RLExp' in self.settings:
            # print('RLExp')
            self.coesDecay = torch.relu(self.coesDecay_)
            # print('self.coesDecay',self.coesDecay)
        elif 'GLExp' in self.settings:
            self.coesDecay = torch.nn.functional.gelu(self.coesDecay_)  
        elif 'AbsExp'  in self.settings:
            self.coesDecay = torch.abs(self.coesDecay_) 
        elif 'PowExp'  in self.settings:
            self.coesDecay = (self.coesDecay_)**2                       
        elif 'SigExp' in self.settings:
            self.coesDecay = torch.sigmoid(self.coesDecay_)
        else:
            self.coesDecay = self.coesDecay_      
        if 'ExpDecay' in self.settings:  
            # print('self.Layer_idx',self.Layer_idx)    
            if 'InvExp' in self.settings:
                x_k_exp = torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.bias_inner)
            else:
                x_k_exp = torch.exp(self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.bias_inner)         
            fx = self.conv_dw(x_k_exp)
        else:
            fx = self.conv_dw(x_k)
            
        if self.use_conv_mlp:
            fx = self.norm(fx)
            fx = self.mlp(fx)
        else:
            fx = fx.permute(0, 2, 3, 1)
            fx = self.norm(fx)
            fx = self.mlp(fx)
            fx = fx.permute(0, 3, 1, 2)
        if self.gamma is not None:
            fx = fx.mul(self.gamma.reshape(1, -1, 1, 1))


        if 'ExpDecay' in self.settings:
            # print('ExpDecay')
            # print('self.coesDecay',self.coesDecay)
            # fx =  torch.mul((1-torch.sigmoid(self.coesBalance)), torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize) * (fx-self.coesDecay*shortcut) ) +torch.mul(torch.sigmoid(self.coesBalance), fx)
            if 'InvExp' in self.settings:
                fx = torch.exp(self.coesDecay*self.Layer_idx*self.coes_stepsize) * fx -self.coesDecay*shortcut+self.bias_outer
                
            else:
                fx = torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize) * fx -self.coesDecay*shortcut+self.bias_outer
        # print('self.IsIni',self.IsIni)
        # print('fx',float(fx.max().data))
        if self.IsIni == True:
            x = self.drop_path(fx) + shortcut
            # pre_features = pre_features
        else:
            sum_features = torch.mul(self.coesA[0],shortcut)
            
            sum_acts = torch.mul(self.coesB[0],fx)
            
            for i in range(self.order-1):
                # print('pre_features[i]',pre_features[i])
                # if isinstance(pre_features[i],torch.Tensor):
                #     print('sum_features.shape,pre_features[i].shape',sum_features.shape,pre_features[i].shape)
                sum_features = torch.add( sum_features, torch.mul(self.coesA[i+1],pre_features[i] ))
                sum_acts = torch.add( sum_acts, torch.mul(self.coesB[i+1],pre_acts[i]) )
                    
            x =  torch.add( sum_features, torch.mul(self.coes_stepsize, -sum_acts ) )  
        # print('pre_features',pre_features) 
              
        for i in range(self.order-2, 0, -1): #order-2, order-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1] 
            # if not isinstance( pre_acts[i], int):
                # print('pre_acts.mean(),pre_features.mean()',pre_acts[i].mean(),pre_features[i].mean())
        # print('pre_features',pre_features) 
        pre_features[0] = shortcut
        pre_acts[0] = fx            
        # all = torch.stack( (x, pre_features, pre_acts), dim=0 )
        all = [x] + [pre_features] + [pre_acts]            
        # x = self.drop_path(x) + shortcut
        if self.IsIni == False:
            x = self.end_block_vis(x)
        return all

    
class ConvNeXtConverStage(nn.Module):

    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, ls_init_value=1.0, conv_mlp=False,
            norm_layer=None, cl_norm_layer=None, cross_stage=False, coesA=[-1,0], coesB=[0,1], coes_stepsize=1.0, IniBlockNum=9999, settings='', coesDecay=None):
        super().__init__()
        self.grad_checkpointing = False
        self.downsample_fea = nn.ModuleList([])
        self.downsample_act = nn.ModuleList([])
        self.settings = settings
        # print('stride',stride)
        if in_chs != out_chs or stride > 1:
            self.downsample_x = nn.Sequential(
                    norm_layer(in_chs),
                    nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
                )            
            for i in range(len(coesA)):
                self.downsample_fea.append(nn.Sequential(
                    norm_layer(in_chs),
                    nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
                ) )
                self.downsample_act.append(nn.Sequential(
                    norm_layer(in_chs),
                    nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
                ) )             
        else:
            self.downsample_x = nn.Identity()        
            for i in range(len(coesA)):
                self.downsample_fea.append(nn.Identity() )
                self.downsample_act.append(nn.Identity() )              
            
        dp_rates = dp_rates or [0.] * depth
        # self.blocks = nn.Sequential() 
        # self.blocks = nn.Sequential(*[ConvNeXtConverBlock(
        #     dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
        #     norm_layer=norm_layer if conv_mlp else cl_norm_layer, coesA=coesA, coesB=coesB, coes_stepsize=coes_stepsize, IsIni=IsIni, settings=settings)
        #     for j in range(depth)]
        # )
        self.blocks = []
        
  
        self.has_ini_block = False
        self.has_non_ini_block = False
        for block_idx in range(0,depth): 
            l_ = block_idx
            if 'RestaLayerIdx' in self.settings:
                if self.settings.split('RestaLayerIdx')[1]:
                    Split = int(self.settings.split('RestaLayerIdx')[1])
                    l_ = block_idx%Split
                    # print('Split',Split)    
            print('l_',l_)            
            # print('block_idx, IniBlockNum',block_idx, IniBlockNum)
            if block_idx < IniBlockNum:#-1:
                self.IsIni = True
                self.has_ini_block = True
            else:
                self.IsIni = False
                self.has_non_ini_block = True
            # print('IsIni,len(coesA)',self.IsIni,len(coesA))
            self.blocks.append(ConvNeXtConverBlock(dim=out_chs,drop_path=dp_rates[block_idx], ls_init_value=ls_init_value, conv_mlp=conv_mlp,norm_layer=norm_layer if conv_mlp else cl_norm_layer, coesA=coesA, coesB=coesB, coes_stepsize=coes_stepsize, IsIni=self.IsIni, settings=settings,Layer_idx=l_, coesDecay=coesDecay))
        self.order = len(coesA)    
        self.blocks = nn.Sequential(*self.blocks)
    def forward(self, x):
        #TODO: put downsample in the first block
        pre_features = []
        pre_acts = []        
        for i in range(self.order-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        all = [x, pre_features, pre_acts]        
        # print('x[0].shape()',x[0].shape)
        
        # if self.IsIni is True:
        #     all = self.downsample(all)   
        # else:
        # if self.has_ini_block==True and self.has_non_ini_block==False:
        #     print(',has_ini_block')    
        #     x = self.downsample(all[0]) 
        #     pre_features = all[1]
        #     pre_acts = all[2]    
        # elif:
                         
        # else:
        pre_features = []
        pre_acts = []
        x = self.downsample_x(all[0])
        i = 0
        for pre_f in all[1]:
            
            if isinstance(pre_f,torch.Tensor):
                pre_features += [self.downsample_fea[i](pre_f)]
             
            else:
                pre_features += [pre_f]
            i += 1
        i = 0 
        for pre_a in all[2]:
            if isinstance(pre_a,torch.Tensor):
               
                pre_acts += [self.downsample_act[i](pre_a)]
            else:
                pre_acts += [pre_a]
            i += 1
        all = [x,pre_features,pre_acts]
            
        if self.grad_checkpointing and not torch.jit.is_scripting():
            all = checkpoint_seq(self.blocks, all)
        else:
            all = self.blocks(all)
        # return all
        x = all[0]
        return x 


class ConvNeXtConver(nn.Module):
    r""" 
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, patch_size=4,
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),  ls_init_value=1e-6, conv_mlp=False, stem_type='patch',
            head_init_scale=1., head_norm_first=False, norm_layer=None, drop_rate=0., drop_path_rate=0., givenA=[1,0], givenB=[1, 0], ini_stepsize=1, settings='',PL=1, ini_block_shift=None, IniDecay=None
    ):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer
        self.order = len(givenA)
        self.settings = settings
        if 'mnist' in self.settings:
            in_chans = 1
            
        if 'LearnCoe' in self.settings:   
            # print('LearnCoe')  
            requires_grad = True
        else:
            requires_grad = False
        if 'ExpDecay' in self.settings and 'LearnDecay' in self.settings:            
            self.coesDecay =nn.Parameter(torch.ones(1)*IniDecay, requires_grad=True)
        else:
            self.coesDecay = nn.Parameter(torch.ones(1)*IniDecay, requires_grad=False)
                    
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*givenA[i], requires_grad=requires_grad) for i in range(self.order)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*givenB[i], requires_grad=requires_grad) for i in range(self.order)])
        if 'LearnStepSize'in self.settings: 
            self.coes_stepsize = self.ini_stepsize =nn.Parameter(torch.Tensor(1).uniform_(ini_stepsize, ini_stepsize))
        else:
            self.coes_stepsize = ini_stepsize        
        
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []
        if 'imagenet' in dataset:
            patch_size = patch_size
            stride = patch_size
        else:
            patch_size = 3
            stride = 1
        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        if stem_type == 'patch':
            if 'imagenet' in dataset:
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
                    norm_layer(dims[0])
                )    
            else:
                self.stem = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=stride,padding='same'),
                    norm_layer(dims[0])
                )
            curr_stride = stride
                
            prev_chs = dims[0]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                norm_layer(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
            )
            curr_stride = 2
            prev_chs = 64

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        sum_depth = 0
        
        self.ini_block_shift = ini_block_shift
        if 'imagenet' in dataset:
            sta_num = 4
        else:
            sta_num = 3
        for i in range(sta_num):
            
            stride = 2 if curr_stride == 2 or i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            # TODO: there is no shortcut between two stages in the ori convnext
            if self.ini_block_shift is not None:
                IniBlockNum = self.ini_block_shift - sum_depth
            else:
                IniBlockNum = self.order - sum_depth
            # print('stage ',i)
            if i == 2:
                
                if self.ini_block_shift is not None:
                    IniBlockNum = self.ini_block_shift
                else:
                    IniBlockNum = self.order     
                                
                stages.append(ConvNeXtConverStage(
                    prev_chs, out_chs, stride=stride,
                    depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                    norm_layer=norm_layer, cl_norm_layer=cl_norm_layer, coesA=self.coesA, coesB=self.coesB, coes_stepsize=self.coes_stepsize, IniBlockNum=IniBlockNum, settings=settings, coesDecay=self.coesDecay)
                )
            else:
                stages.append(ConvNeXtStage(
                    prev_chs, out_chs, stride=stride,
                    depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                    norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
                )                
            sum_depth += depths[i]
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXtConver ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # pre_features = []
        # pre_acts = []
        x = self.stem(x)
        # for st in range(4):
        #     if st == 2:
        #         for i in range(self.order-1):
        #             pre_features.append(-i)
        #             pre_acts.append(-i)
        #         x = [x, pre_features, pre_acts]
                
        x = self.stages(x)
        # x = x[0]        
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

class Downsample_Fix(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None,
            preact=True, conv_layer=None, norm_layer=None):

        super(Downsample_Fix, self).__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = None #nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        # self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        if self.pool is not None:
            x = self.conv(self.pool(x))
        else:
            x = self.conv(x)
        return x 

class DownsampleAvg(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None,
            preact=True, conv_layer=None, norm_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvg, self).__init__()
        self.in_chs = in_chs
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = None #nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = None if preact else norm_layer(out_chs, apply_act=False)
        # self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        if self.pool is None:        
            if self.norm is None:
                x = self.conv(x)
            else:
                x = self.norm(self.conv(self.pool(x)))     
        else:        
            if self.norm is None:
                x = self.conv(x)
            else:
                x = self.norm(self.conv(self.pool(x)))
        return x          
        # return self.norm(self.conv(self.pool(x)))
        
def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict


def _create_convnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        ConvNeXtConver, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


@register_model
def convnext_conver_nano_hnf(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        dims = (80, 160, 320, 640)
        depths = (2, 2, 8,2)
    else:
        dims = (40, 80, 160)
        depths = (2, 2, 8)
    model_args = dict(depths=depths, dims=dims, head_norm_first=True, conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_conver_nano_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny_hnf(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, **kwargs)
    model = _create_convnext('convnext_conver_tiny_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny_hnfd(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, stem_type='dual', **kwargs)
    model = _create_convnext('convnext_conver_tiny_hnf', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    else:
        model_args = dict(depths=(3, 3, 9), dims=(96, 192, 384), **kwargs)
    model = _create_convnext('convnext_conver_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_small(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    else:
        model_args = dict(depths=[3, 3, 27], dims=[96, 192, 384], **kwargs)
    model = _create_convnext('convnext_conver_small', pretrained=pretrained, **model_args)
    return model

@register_model
def convnext_conver_small_narrow(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        model_args = dict(depths=[3, 3, 27, 3], dims=[48, 96, 192, 384, ], **kwargs)
    else:
        model_args = dict(depths=[3, 3, 27], dims=[48, 96, 192], **kwargs)
    model = _create_convnext('convnext_conver_small_narrow', pretrained=pretrained, **model_args)
    return model

@register_model
def convnext_conver_base(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    else:
        model_args = dict(depths=[3, 3, 27], dims=[128, 256, 512], **kwargs)
    model = _create_convnext('convnext_conver_base', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_large(pretrained=False, **kwargs):
    if 'imagenet' in dataset:
        model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    else:
        model_args = dict(depths=[3, 3, 27], dims=[192, 384, 768], **kwargs)
    model = _create_convnext('convnext_conver_large', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_tiny_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_small_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_small_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_base_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_base_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_large_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_conver_large_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_xlarge_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_conver_xlarge_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_tiny_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_small_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_small_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_base_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_base_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_large_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_conver_large_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_xlarge_384_in22ft1k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_conver_xlarge_384_in22ft1k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_tiny_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_tiny_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_small_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_small_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_base_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_conver_base_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_large_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = _create_convnext('convnext_conver_large_in22k', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_conver_xlarge_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model = _create_convnext('convnext_conver_xlarge_in22k', pretrained=pretrained, **model_args)
    return model



