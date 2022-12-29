"""Pre-Activation ResNet v2 with GroupNorm and Weight Standardization.

A PyTorch implementation of ResNetV2 adapted from the Google Big-Transfoer (BiT) source code
at https://github.com/google-research/big_transfer to match timm interfaces. The BiT weights have
been included here as pretrained models from their original .NPZ checkpoints.

Additionally, supports non pre-activation bottleneck for use as a backbone for Vision Transfomers (ViT) and
extra padding support to allow porting of official Hybrid ResNet pretrained weights from
https://github.com/google-research/vision_transformer

Thanks to the Google team for the above two repositories and associated papers:
* Big Transfer (BiT): General Visual Representation Learning - https://arxiv.org/abs/1912.11370
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - https://arxiv.org/abs/2010.11929
* Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237

Original copyright of Google code below, modifications by Ross Wightman, Copyright 2020.
"""
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import sys,os
sys.path.append(os.getcwd())


from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
# from .helpers import build_model_with_cfg, named_apply, adapt_input_conv
# from .registry import register_model
# from .layers import GroupNormAct, BatchNormAct2d, EvoNormBatch2d, EvoNormSample2d,\
#     ClassifierHead, DropPath, AvgPool2dSame, create_pool2d, StdConv2d, create_conv2d
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.registry import register_model
from timm.models.layers import GroupNormAct, BatchNormAct2d, EvoNormBatch2d, EvoNormSample2d,\
    ClassifierHead, DropPath, AvgPool2dSame, create_pool2d, StdConv2d, create_conv2d

import torch.onnx
import netron
# import onnx
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 10, 'input_size': (3, 32, 32), 'pool_size': (3, 3),
#         'crop_pct': 0.875, 'interpolation': 'bilinear',
#         'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
#         'first_conv': 'stem.conv', 'classifier': 'head.fc',
#         **kwargs
#     }

default_cfgs = {
    # pretrained on imagenet21k, finetuned on imagenet1k
    'resnetv2_50x1_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0),
    'resnetv2_50x3_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R50x3-ILSVRC2012.npz',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0),
    'resnetv2_101x1_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R101x1-ILSVRC2012.npz',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0),
    'resnetv2_101x3_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R101x3-ILSVRC2012.npz',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0),
    'resnetv2_152x2_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R152x2-ILSVRC2012.npz',
        input_size=(3, 448, 448), pool_size=(14, 14), crop_pct=1.0),
    'resnetv2_152x4_bitm': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R152x4-ILSVRC2012.npz',
        input_size=(3, 480, 480), pool_size=(15, 15), crop_pct=1.0),  # only one at 480x480?

    # trained on imagenet-21k
    'resnetv2_50x1_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz',
        num_classes=21843),
    'resnetv2_50x3_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R50x3.npz',
        num_classes=21843),
    'resnetv2_101x1_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz',
        num_classes=21843),
    'resnetv2_101x3_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R101x3.npz',
        num_classes=21843),
    'resnetv2_152x2_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz',
        num_classes=21843),
    'resnetv2_152x4_bitm_in21k': _cfg(
        url='https://storage.googleapis.com/bit_models/BiT-M-R152x4.npz',
        num_classes=21843),

    'resnetv2_50x1_bit_distilled': _cfg(
        url='https://storage.googleapis.com/bit_models/distill/R50x1_224.npz',
        interpolation='bicubic'),
    'resnetv2_152x2_bit_teacher': _cfg(
        url='https://storage.googleapis.com/bit_models/distill/R152x2_T_224.npz',
        interpolation='bicubic'),
    'resnetv2_152x2_bit_teacher_384': _cfg(
        url='https://storage.googleapis.com/bit_models/distill/R152x2_T_384.npz',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, interpolation='bicubic'),

    'resnetv2_50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetv2_50_a1h-000cdf49.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnetv2_50d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_38d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),    
    'resnetv2_26d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),    
    'convernetv2_26d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'), 
    'convernetv2_38d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),     
    'convernetv2_50d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),    
    'convernetv2_101d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'convernetv2_26d_s': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'), 
    'convernetv2_38d_s': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),       
    'convernetv2_50d_s': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),    
    'convernetv2_101d_s': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_50t': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetv2_101_a1h-5d01f016.pth',
        interpolation='bicubic', crop_pct=0.95),
    'resnetv2_101d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_152': _cfg(
        interpolation='bicubic'),
    'resnetv2_152d': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),

    'resnetv2_50d_gn': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_50d_evob': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
    'resnetv2_50d_evos': _cfg(
        interpolation='bicubic', first_conv='stem.conv1'),
}


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# class Downsample_Fix(nn.Module): 
#     def __init__(self,in_planes,out_planes,stride=1):#stride=2):
#         super(Downsample_Fix,self).__init__()
#         self.downsample_=nn.Sequential(
#                     nn.AvgPool2d(2),
#                         nn.Conv2d(in_planes, out_planes,
#                                 kernel_size=1, stride=stride, bias=False)
#                         )
class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None #nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        x_preact = self.norm1(x)

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        if self.drop_path is not None:        
            x = self.drop_path(x)
        return x + shortcut


class Bottleneck(nn.Module):
    """Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT.
    """
    def __init__(
            self, in_chs, out_chs=None, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, preact=False,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm1 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm2 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm3 = norm_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None # nn.Identity()
        self.act3 = act_layer(inplace=True)

    def zero_init_last(self):
        nn.init.zeros_(self.norm3.weight)

    def forward(self, x):
        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x


class DownsampleConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None, preact=True,
            conv_layer=None, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=stride)
        # self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)
        self.norm = None if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(self.conv(x))
        else:
            x = self.conv(x)
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
        # return self.conv(self.pool(x))
     
class DownsampleAvgStart(nn.Module):
    def __init__(
            self, in_chs, out_chs, stride=1, dilation=1, first_dilation=None,
            preact=True, conv_layer=StdConv2d, norm_layer=None):
        """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
        super(DownsampleAvgStart, self).__init__()
        self.in_chs = in_chs
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(4, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = None #nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1).cuda()
        self.norm = None if preact else norm_layer(out_chs, apply_act=False)
        # self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        if self.pool is None:        
            if self.norm is None:
                x = self.conv(x)
            else:
                x = self.norm(self.conv(x))     
        else:        
            if self.norm is None:
                x = self.pool(x)
                x = self.conv(x)
            else:
                x = self.norm(self.conv(self.pool(x)))
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

class ResNetStage(nn.Module):
    """ResNet Stage."""
    def __init__(self, in_chs, out_chs,  stride, dilation, depth, bottle_ratio=0.25, groups=1,
                 avg_down=False, block_dpr=None, block_fn=PreActBottleneck,
                 act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(ResNetStage, self).__init__()
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = DownsampleAvg if avg_down else DownsampleConv
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            self.blocks.add_module(str(block_idx), block_fn(
                prev_chs, out_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                **layer_kwargs, **block_kwargs))
            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None

    def forward(self):
        x = self.blocks(x)
        return x

class AnyOrderBottleneck(nn.Module):

    def __init__(
            self, in_chs, out_chs=None, all_chs=[], coesA=[], coesB=[], stepsize=1, IsIni=1, bottle_ratio=0.25, stride=1, dilation=1, first_dilation=None, groups=1,
            act_layer=None, conv_layer=None, norm_layer=None, proj_layer=None, drop_path_rate=0.):
        super().__init__()
        self.coesA = coesA
        self.coesB = coesB
        self.stepsize = stepsize
        self.IsIni = IsIni
        self.order = len(self.coesA)
        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)
        out_chs = out_chs or in_chs
        mid_chs = make_div(out_chs * bottle_ratio)
        
        if proj_layer is not None:
            self.Fix_downsample = proj_layer(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, preact=True,
                conv_layer=conv_layer, norm_layer=norm_layer)
        else:
            self.Fix_downsample = None
        # print('coesB', coesB[0])
        self.norm1 = norm_layer(in_chs)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm2 = norm_layer(mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm4 = norm_layer(out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else None #nn.Identity()
        self.out_chs = out_chs
        self.in_chs = in_chs
        self.all_chs = all_chs
        self.stride = stride
        self.identity = nn.Identity()
        
    # def chs(self):
    #     self.all_chs = self.all_chs + [self.out_chs]
    #     return self.all_chs
        
    def zero_init_last(self):
        nn.init.zeros_(self.conv3.weight)

    def forward(self, all): 
        
        # if self.IsIni:
        #     x = all
        # else:
        x = all[0]
        pre_features = all[1]
        pre_acts = all[2]
            
        x_preact = self.norm1(x)# ori
        # x_preact = x

        # shortcut branch
        shortcut = x
        if self.Fix_downsample is not None:
            shortcut = self.Fix_downsample(x_preact)# ori
            # kernel = torch.empty(self.out_chs, self.in_chs, 1, 1).cuda()
            # nn.init.dirac_(kernel, 2) #self.out_chs/self.in_chs
            # shortcut = F.avg_pool2d(x_preact, kernel_size=self.stride )
            # # print('shortcut', shortcut.shape)
            # # print('kernel', kernel)
            # shortcut = F.conv2d(shortcut, kernel, stride=1, padding='same')

            if not self.IsIni:
                for i in range(self.order-1):
                    # pre_features[i] = self.norm1(pre_features[i])  
                    # pre_acts[i] = self.norm1(pre_acts[i])
                    pre_features[i] = self.Fix_downsample(pre_features[i])# ori
                    pre_acts[i] = self.Fix_downsample(pre_acts[i])# ori
                            
        # # residual branch
        # F_x_n = self.conv1(x_preact)# ori
        # F_x_n = self.conv2(self.norm2(F_x_n))# ori
        # F_x_n = self.conv3(self.norm3(F_x_n))# ori
        # F_x_n = self.drop_path(F_x_n)# ori
        
        # end norm
        F_x_n = self.conv1(x_preact)# ori
        F_x_n = self.conv2(self.norm2(F_x_n))# ori
        F_x_n = self.conv3(self.norm3(F_x_n))# ori
        F_x_n = self.norm4(F_x_n)
        if self.drop_path is not None:
            F_x_n = self.drop_path(F_x_n)# ori        
        
        # if x_preact.shape[1] != F_x_n.shape[1]:
        #     kernel = torch.empty(self.out_chs, self.in_chs, 1, 1).cuda()
        #     nn.init.dirac_(kernel, int(self.out_chs/self.in_chs)) #self.out_chs/self.in_chs
        #     shortcut = F.avg_pool2d(x_preact, kernel_size=self.stride )
        #     shortcut = F.conv2d(shortcut, kernel, stride=1, padding='same')
            
        if self.IsIni:
            x = F_x_n + shortcut
            # print('x', x.shape)
        else:
            # for i in range(self.order-1):
            #     if pre_features[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_features[i].shape[1], 1, 1).cuda()
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_features[i].shape[1])) #F_x_n.shape[1]/pre_features[i].shape[1]
            #         pre_features[i] = F.avg_pool2d( pre_features[i], kernel_size=self.stride )
            #         pre_features[i] = F.conv2d( pre_features[i], kernel, stride=1, padding='same')
            #     if pre_acts[i].shape[1] != F_x_n.shape[1]:
            #         kernel = torch.empty(F_x_n.shape[1], pre_acts[i].shape[1], 1, 1).cuda()
            #         nn.init.dirac_(kernel, int(F_x_n.shape[1]/pre_acts[i].shape[1])) 
            #         pre_acts[i] = F.avg_pool2d(pre_acts[i], kernel_size=self.stride )
            #         # print('shortcut', shortcut.shape)
            #         # print('kernel', kernel)
            #         pre_acts[i] = F.conv2d(pre_acts[i], kernel, stride=1, padding='same')

            sum_features = self.coesA[0].expand_as(shortcut)*shortcut
            
            sum_acts = self.coesB[0].expand_as(F_x_n)*F_x_n
            
            for i in range(self.order-1):
                sum_features = torch.add( sum_features, self.coesA[i+1].expand_as(pre_features[i])*pre_features[i] )
                sum_acts = torch.add( sum_acts, self.coesB[i+1].expand_as(pre_acts[i])*pre_acts[i] )
                    
            x =  torch.add( sum_features, torch.mul(self.stepsize, -sum_acts ) )    
        for i in range(self.order-2, 0, -1): #order-2, order-1, ..., 0 #1, 0 
            pre_features[i] = pre_features[i-1] #pre_features[i-1] = pre_features[i-2]?
            pre_acts[i] = pre_acts[i-1] 
            
        pre_features[0] = shortcut
        pre_acts[0] = F_x_n            
        x = self.identity(x)
        # all = torch.stack( (x, pre_features, pre_acts), dim=0 )
        all = [x] + [pre_features] + [pre_acts]
        # print('all', all.shape)
        # for i in range(self.order-1):
        #     if pre_features[i].shape[1] != F_x_n.shape[1]:
        #         pre_features[i] = self.FixDS[-1](pre_features[i])
            
        #     if pre_acts[i].shape[1] != F_x_n.shape[1]:
        #         pre_acts[i] = self.FixDS[-1](pre_acts[i])
        # sum_features = coesA[0].expand_as(residual)*residual
        # print('x, shortcut', x.shape, shortcut.shape)
        return all
class AnyOrderStage(nn.Module):
    def __init__(self, in_chs, out_chs, all_chs, coesA, coesB, stepsize, IniBlockNum, stride, dilation, depth, bottle_ratio=0.25, groups=1,
                 fix_down=True, block_dpr=None, block_fn=AnyOrderBottleneck,
                 act_layer=None, conv_layer=None, norm_layer=None, **block_kwargs):
        super(AnyOrderStage, self).__init__()
        self.coesA = coesA
        self.coesB = coesB
        self.stepsize = stepsize
        self.order = len(coesA)
        
        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        proj_layer = Downsample_Fix if fix_down else  DownsampleAvg
        prev_chs = in_chs
        self.blocks = nn.Sequential()
        for block_idx in range(depth):
            drop_path_rate = block_dpr[block_idx] if block_dpr else 0.
            stride = stride if block_idx == 0 else 1
            # print("prev_chs, out_chs", prev_chs, out_chs)
            # all_chs = all_chs + [out_chs]
            # print('all_chs', all_chs)
            if block_idx < IniBlockNum-1: # residual
                self.blocks.add_module(str(block_idx), block_fn(
                    prev_chs, out_chs, all_chs, coesA, coesB, stepsize, IsIni=1, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                    first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                    **layer_kwargs, **block_kwargs))
            else: # high order 
                self.blocks.add_module(str(block_idx), block_fn(
                    prev_chs, out_chs, all_chs, coesA, coesB, stepsize, IsIni=0, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
                    first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
                    **layer_kwargs, **block_kwargs))
            # all_chs =block_fn(
            #     prev_chs, out_chs, all_chs, stride=stride, dilation=dilation, bottle_ratio=bottle_ratio, groups=groups,
            #     first_dilation=first_dilation, proj_layer=proj_layer, drop_path_rate=drop_path_rate,
            #     **layer_kwargs, **block_kwargs).chs()

            prev_chs = out_chs
            first_dilation = dilation
            proj_layer = None
    def StepCounter(self):
        return self.stepnum+1
    def forward(self, x):
        x = self.blocks(x)
        return x

def is_stem_deep(stem_type):
    return any([s in stem_type for s in ('deep', 'tiered')])


def create_resnetv2_stem(
        in_chs, out_chs=64, stem_type='', preact=True,
        conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32)):
    stem = OrderedDict()
    assert stem_type in ('', 'fixed', 'same', 'deep', 'deep_fixed', 'deep_same', 'tiered')

    # NOTE conv padding mode can be changed by overriding the conv_layer def
    if is_stem_deep(stem_type):
        # A 3 deep 3x3  conv stack as in ResNet V1D models
        if 'tiered' in stem_type:
            stem_chs = (3 * out_chs // 8, out_chs // 2)  # 'T' resnets in resnet.py
        else:
            stem_chs = (out_chs // 2, out_chs // 2)  # 'D' ResNets
        stem['conv1'] = conv_layer(in_chs, stem_chs[0], kernel_size=3, stride=2)
        stem['norm1'] = norm_layer(stem_chs[0])
        stem['conv2'] = conv_layer(stem_chs[0], stem_chs[1], kernel_size=3, stride=1)
        stem['norm2'] = norm_layer(stem_chs[1])
        stem['conv3'] = conv_layer(stem_chs[1], out_chs, kernel_size=3, stride=1)
        if not preact:
            stem['norm3'] = norm_layer(out_chs)
    else:
        # The usual 7x7 stem conv
        stem['conv'] = conv_layer(in_chs, out_chs, kernel_size=7, stride=2)
        if not preact:
            stem['norm'] = norm_layer(out_chs)

    if 'fixed' in stem_type:
        # 'fixed' SAME padding approximation that is used in BiT models
        stem['pad'] = nn.ConstantPad2d(1, 0.)
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    elif 'same' in stem_type:
        # full, input size based 'SAME' padding, used in ViT Hybrid model
        stem['pool'] = create_pool2d('max', kernel_size=3, stride=2, padding='same')
    else:
        # the usual PyTorch symmetric padding
        stem['pool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    return nn.Sequential(stem)


class ConverNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """
    # def __init__(
    #         self, layers, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], channels=(32, 128, 512, 2048), ini_stepsize=1, 
    #         num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
    #         width_factor=1, stem_chs=64, stem_type='', fix_down=False, preact=True,
    #         act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
    #         drop_rate=0., drop_path_rate=0., zero_init_last=False):
    def __init__(
            self, layers, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], channels=(256, 512, 1024, 2048), ini_stepsize=1, 
            num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            width_factor=1, stem_chs=64, stem_type='', fix_down=False, preact=True,
            act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
            drop_rate=0., drop_path_rate=0., zero_init_last=False):
        super().__init__()
        self.order = len(givenA)
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*givenA[i], requires_grad=False) for i in range(self.order)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*givenB[i], requires_grad=False) for i in range(self.order)]) 
        self.stepsize = ini_stepsize

        
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        all_chs = [prev_chs]
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = AnyOrderBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        # stepnum = 0
        sum_depth = 0
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            all_chs = all_chs + [out_chs]
            # if sum_depth < self.order: #res
            IniBlockNum = self.order - sum_depth
            stage = AnyOrderStage(
                prev_chs, out_chs, all_chs, coesA=self.coesA, coesB=self.coesB, stepsize=self.stepsize, IniBlockNum=IniBlockNum, stride=stride, dilation=dilation, depth=d, fix_down=fix_down,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            sum_depth += d

            # else: 
                
            # stepnum = stage.StepCounter()

            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            # print('self.feature_info', self.feature_info)
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else None # nn.Identity()
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

        self.identityStart0 = nn.Identity()
        self.identityStart1 = nn.Identity()
        self.identityEndM1 = nn.Identity()
        self.identityEnd = nn.Identity()
        
        self.init_weights(zero_init_last=zero_init_last)

    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        pre_features = []
        pre_acts = []
        
        # ds = DownsampleAvgStart(3, 64, stride=4)
        # print('x.shape0', x.shape)
        x = self.stem(x)# + ds(x)
        # print('x.shape1', x.shape)
        
        # x = torch.stack( (x, x, x), dim=0 )
        for i in range(self.order-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        x = [x, pre_features, pre_acts]
        x = self.stages(x)
        x = x[0]
        if self.norm is not None:
            x = self.norm(x)
        
        return x

    def forward(self, x):
        # x = torch.stack( (x, x, x), dim=0 )
        x = self.forward_features(x)
        x = self.identityEndM1(x)        
        x = self.head(x)
        x = self.identityEnd(x)        
        return x


class ConverNetV2_s(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """
    def __init__(
            self, layers, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], channels=(32, 128, 512, 2048), ini_stepsize=1, 
            num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            width_factor=1, stem_chs=64, stem_type='', fix_down=False, preact=True,
            act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
            drop_rate=0., drop_path_rate=0., zero_init_last=False):
    # def __init__(
    #         self, layers, givenA=[0.33333333333, 0.55555555556, 0.1111111111], givenB=[0.888888888889, 0.8888888888889, 0], channels=(256, 512, 1024, 2048), ini_stepsize=1, 
    #         num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
    #         width_factor=1, stem_chs=64, stem_type='', fix_down=False, preact=True,
    #         act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
    #         drop_rate=0., drop_path_rate=0., zero_init_last=False):
        super().__init__()
        self.order = len(givenA)
        self.coesA = nn.ParameterList([nn.Parameter(torch.ones(1)*givenA[i], requires_grad=False) for i in range(self.order)]) 
        self.coesB = nn.ParameterList([nn.Parameter(torch.ones(1)*givenB[i], requires_grad=False) for i in range(self.order)]) 
        self.stepsize = ini_stepsize

        
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        all_chs = [prev_chs]
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = AnyOrderBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        # stepnum = 0
        sum_depth = 0
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            all_chs = all_chs + [out_chs]
            # if sum_depth < self.order: #res
            IniBlockNum = self.order - sum_depth
            stage = AnyOrderStage(
                prev_chs, out_chs, all_chs, coesA=self.coesA, coesB=self.coesB, stepsize=self.stepsize, IniBlockNum=IniBlockNum, stride=stride, dilation=dilation, depth=d, fix_down=fix_down,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            sum_depth += d

            # else: 
                
            # stepnum = stage.StepCounter()

            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            # print('self.feature_info', self.feature_info)
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else None #nn.Identity()
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

        self.init_weights(zero_init_last=zero_init_last)

    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        pre_features = []
        pre_acts = []
        x = self.stem(x)
        # x = torch.stack( (x, x, x), dim=0 )
        for i in range(self.order-1):
            pre_features.append(-i)
            pre_acts.append(-i)
        x = [x, pre_features, pre_acts]
        x = self.stages(x)
        x = x[0]
        if self.norm is not None:       
            x = self.norm(x)
        return x

    def forward(self, x):
        # x = torch.stack( (x, x, x), dim=0 )
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode.
    """

    def __init__(
            self, layers, channels=(256, 512, 1024, 2048),
            num_classes=1000, in_chans=3, global_pool='avg', output_stride=32,
            width_factor=1, stem_chs=64, stem_type='', avg_down=False, preact=True,
            act_layer=nn.ReLU, conv_layer=StdConv2d, norm_layer=partial(GroupNormAct, num_groups=32),
            drop_rate=0., drop_path_rate=0., zero_init_last=False):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        wf = width_factor

        self.feature_info = []
        stem_chs = make_div(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans, stem_chs, stem_type, preact, conv_layer=conv_layer, norm_layer=norm_layer)
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv') if preact else 'stem.norm'
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck if preact else Bottleneck
        self.stages = nn.Sequential()
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_div(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            
            stage = ResNetStage(
                prev_chs, out_chs, stride=stride, dilation=dilation, depth=d, avg_down=avg_down,
                act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer, block_dpr=bdpr, block_fn=block_fn)
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            self.stages.add_module(str(stage_idx), stage)

        self.num_features = prev_chs
        self.norm = norm_layer(self.num_features) if preact else None #nn.Identity()
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

        self.init_weights(zero_init_last=zero_init_last)

    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate, use_conv=True)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str = '', zero_init_last=True):
    if isinstance(module, nn.Linear) or ('head.fc' in name and isinstance(module, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        if name.find('Fix') != -1:
            print ('Fix: ', name)
            nn.init.dirac_(module.weight, 4) # 2
        else:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()

        # nn.init.dirac_(module.downsample_[1].weight.data, 2)

@torch.no_grad()
def _load_weights(model: nn.Module, checkpoint_path: str, prefix: str = 'resnet/'):
    import numpy as np

    def t2p(conv_weights):
        """Possibly convert HWIO to OIHW."""
        if conv_weights.ndim == 4:
            conv_weights = conv_weights.transpose([3, 2, 0, 1])
        return torch.from_numpy(conv_weights)

    weights = np.load(checkpoint_path)
    stem_conv_w = adapt_input_conv(
        model.stem.conv.weight.shape[1], t2p(weights[f'{prefix}root_block/standardized_conv2d/kernel']))
    model.stem.conv.weight.copy_(stem_conv_w)
    model.norm.weight.copy_(t2p(weights[f'{prefix}group_norm/gamma']))
    model.norm.bias.copy_(t2p(weights[f'{prefix}group_norm/beta']))
    if isinstance(getattr(model.head, 'fc', None), nn.Conv2d) and \
            model.head.fc.weight.shape[0] == weights[f'{prefix}head/conv2d/kernel'].shape[-1]:
        model.head.fc.weight.copy_(t2p(weights[f'{prefix}head/conv2d/kernel']))
        model.head.fc.bias.copy_(t2p(weights[f'{prefix}head/conv2d/bias']))
    for i, (sname, stage) in enumerate(model.stages.named_children()):
        for j, (bname, block) in enumerate(stage.blocks.named_children()):
            cname = 'standardized_conv2d'
            block_prefix = f'{prefix}block{i + 1}/unit{j + 1:02d}/'
            block.conv1.weight.copy_(t2p(weights[f'{block_prefix}a/{cname}/kernel']))
            block.conv2.weight.copy_(t2p(weights[f'{block_prefix}b/{cname}/kernel']))
            block.conv3.weight.copy_(t2p(weights[f'{block_prefix}c/{cname}/kernel']))
            block.norm1.weight.copy_(t2p(weights[f'{block_prefix}a/group_norm/gamma']))
            block.norm2.weight.copy_(t2p(weights[f'{block_prefix}b/group_norm/gamma']))
            block.norm3.weight.copy_(t2p(weights[f'{block_prefix}c/group_norm/gamma']))
            block.norm1.bias.copy_(t2p(weights[f'{block_prefix}a/group_norm/beta']))
            block.norm2.bias.copy_(t2p(weights[f'{block_prefix}b/group_norm/beta']))
            block.norm3.bias.copy_(t2p(weights[f'{block_prefix}c/group_norm/beta']))
            if block.downsample is not None:
                w = weights[f'{block_prefix}a/proj/{cname}/kernel']
                block.downsample.conv.weight.copy_(t2p(w))

def _create_convernetv2(variant, pretrained=False, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        ConverNetV2, variant, pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=feature_cfg,
        pretrained_custom_load='_bit' in variant,
        **kwargs)

def _create_convernetv2_s(variant, pretrained=False, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        ConverNetV2_s, variant, pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=feature_cfg,
        pretrained_custom_load='_bit' in variant,
        **kwargs)
    
def _create_resnetv2(variant, pretrained=False, **kwargs):
    feature_cfg = dict(flatten_sequential=True)
    return build_model_with_cfg(
        ResNetV2, variant, pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=feature_cfg,
        pretrained_custom_load='_bit' in variant,
        **kwargs)


def _create_resnetv2_bit(variant, pretrained=False, **kwargs):
    return _create_resnetv2(
        variant, pretrained=pretrained, stem_type='fixed',  conv_layer=partial(StdConv2d, eps=1e-8), **kwargs)


@register_model
def resnetv2_50x1_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_50x1_bitm', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_50x3_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_50x3_bitm', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_101x1_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_101x1_bitm', pretrained=pretrained, layers=[3, 4, 23, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_101x3_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_101x3_bitm', pretrained=pretrained, layers=[3, 4, 23, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_152x2_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x2_bitm', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x4_bitm(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x4_bitm', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=4, **kwargs)


@register_model
def resnetv2_50x1_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_50x1_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_50x3_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_50x3_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 4, 6, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_101x1_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_101x1_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 4, 23, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_101x3_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_101x3_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 4, 23, 3], width_factor=3, **kwargs)


@register_model
def resnetv2_152x2_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x2_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x4_bitm_in21k(pretrained=False, **kwargs):
    return _create_resnetv2_bit(
        'resnetv2_152x4_bitm_in21k', pretrained=pretrained, num_classes=kwargs.pop('num_classes', 21843),
        layers=[3, 8, 36, 3], width_factor=4, **kwargs)


@register_model
def resnetv2_50x1_bit_distilled(pretrained=False, **kwargs):
    """ ResNetV2-50x1-BiT Distilled
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_50x1_bit_distilled', pretrained=pretrained, layers=[3, 4, 6, 3], width_factor=1, **kwargs)


@register_model
def resnetv2_152x2_bit_teacher(pretrained=False, **kwargs):
    """ ResNetV2-152x2-BiT Teacher
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_152x2_bit_teacher', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_152x2_bit_teacher_384(pretrained=False, **kwargs):
    """ ResNetV2-152xx-BiT Teacher @ 384x384
    Paper: Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237
    """
    return _create_resnetv2_bit(
        'resnetv2_152x2_bit_teacher_384', pretrained=pretrained, layers=[3, 8, 36, 3], width_factor=2, **kwargs)


@register_model
def resnetv2_50(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d, **kwargs)


@register_model
def resnetv2_26d(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_26d', pretrained=pretrained,
        layers=[2, 2, 2, 2], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True, **kwargs)
@register_model
def resnetv2_38d(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_38d', pretrained=pretrained,
        layers=[3, 3, 3, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True, **kwargs)
@register_model
def resnetv2_50d(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50d', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True, **kwargs)
    
@register_model
def resnetv2_50t(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50t', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='tiered', avg_down=True, **kwargs)


@register_model
def resnetv2_101(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_101', pretrained=pretrained,
        layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d, **kwargs)


@register_model
def resnetv2_101d(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_101d', pretrained=pretrained,
        layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True, **kwargs)


@register_model
def resnetv2_152(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_152', pretrained=pretrained,
        layers=[3, 8, 36, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d, **kwargs)


@register_model
def resnetv2_152d(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_152d', pretrained=pretrained,
        layers=[3, 8, 36, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', avg_down=True, **kwargs)


# Experimental configs (may change / be removed)

@register_model
def resnetv2_50d_gn(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50d_gn', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=GroupNormAct,
        stem_type='deep', avg_down=True, **kwargs)


@register_model
def resnetv2_50d_evob(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50d_evob', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=EvoNormBatch2d,
        stem_type='deep', avg_down=True, **kwargs)


@register_model
def resnetv2_50d_evos(pretrained=False, **kwargs):
    return _create_resnetv2(
        'resnetv2_50d_evos', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=EvoNormSample2d,
        stem_type='deep', avg_down=True, **kwargs)

@register_model
def convernetv2_26d(pretrained=False, **kwargs):
    return _create_convernetv2(
        'convernetv2_26d', pretrained=pretrained,
        layers=[2, 2, 2, 2], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)
@register_model
def convernetv2_38d(pretrained=False, **kwargs):
    return _create_convernetv2(
        'convernetv2_38d', pretrained=pretrained,
        layers=[3, 3, 3, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)
@register_model
def convernetv2_50d(pretrained=False, **kwargs):
    return _create_convernetv2(
        'convernetv2_50d', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)

@register_model
def convernetv2_101d(pretrained=False, **kwargs):
    return _create_convernetv2(
        'convernetv2_101d', pretrained=pretrained,
        layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)

@register_model
def convernetv2_26d_s(pretrained=False, **kwargs):
    return _create_convernetv2_s(
        'convernetv2_26d_s', pretrained=pretrained,
        layers=[2, 2, 2, 2], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)
@register_model
def convernetv2_38d_s(pretrained=False, **kwargs):
    return _create_convernetv2_s(
        'convernetv2_38d_s', pretrained=pretrained,
        layers=[3, 3, 3, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)
@register_model
def convernetv2_50d_s(pretrained=False, **kwargs):
    return _create_convernetv2_s(
        'convernetv2_50d_s', pretrained=pretrained,
        layers=[3, 4, 6, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)

@register_model
def convernetv2_101d_s(pretrained=False, **kwargs):
    return _create_convernetv2_s(
        'convernetv2_101d_s', pretrained=pretrained,
        layers=[3, 4, 23, 3], conv_layer=create_conv2d, norm_layer=BatchNormAct2d,
        stem_type='deep', fix_down=True, **kwargs)
            
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Fix') != -1:
        print("Found Fix", classname)
        nn.init.dirac_(m.downsample_[1].weight.data, 2)
        
if __name__ == '__main__':
    # w = torch.empty(6, 3, 1, 1)
    # nn.init.dirac_(w,2)
    # # w = torch.empty(3, 24, 5, 5)
    # # nn.init.dirac_(w, 3)
    # print('w', w)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = torch.rand(2, 3, 224, 224).to(device)
    # testloader = torch.utils.data.DataLoader(
    # torchvision.datasets.MNIST('/media/bdc/clm/OverThreeOrders/CIFAR/data' +'/mnist',train=False,download=True,
    #                             transform=torchvision.transforms.Compose([
    #                             torchvision.transforms.ToTensor(),
    #                             torchvision.transforms.Normalize(
    #                                 (0.1307,), (0.3081,))
    #                             ])),
    # batch_size=128, shuffle=False, num_workers=4, pin_memory=True) 
    # #将数据集按批量大小加载到数据集中
    # # data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100, shuffle=True)  #600*100*([[28*28],x])
    # # data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=100, shuffle=False)

    # #for epoch in range(5):  #一共五个周期，其中一个周期（len(dataset_train)=60000）/(batch_size=100)=（len(dataset_train)=600）个批量
    # for i, (images, labels) in enumerate(testloader):

    #     #print(i, images[0].shape, labels[0].shape)


    #     #每100个批量绘制绘制最后一个批量的所有图片
    #     if i==0:#(i + 1) % 100 == 0:
    #         print('batch_number [{}/{}]'.format(i + 1, len(testloader)))
    #         for j in range(len(images)):
    #             if j == 1:
    #                 image = images[j].resize(28, 28).to(device) #将(1,28,28)->(28,28)
    #                 lable = torch.tensor(labels[j]).to(device)
                    
    # d = torch.rand(2, 3, 32, 32).to(device)
    
    # net = ZeroSAny20_Tra()
    net = convernetv2_50d()

    # net = resnetv2_50d()

    
    # net.apply(weights_init) 
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
    print('net:', net)
    # for i in net.named_parameters():
    #     if 1: # "norm" in i[0] and "weight" in i[0]:
    #         print(i[0], i[1].shape)
    onnx_path = "onnx_model_name.onnx"
    torch.onnx.export(net, d, onnx_path)
    netron.start(onnx_path)