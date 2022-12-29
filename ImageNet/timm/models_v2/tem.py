# Code borrowed from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)

_ACTIVATION = {
    'relu': nn.ReLU,
    'swish': nn.SiLU,
}

    
class _BlockIni(nn.Module):
    """
    WideResNet_learn Block.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """
    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU,coesB=[0,0,0],input_size=0,learn=False,IniRes=False,Mask=False,IniCh=16):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=False)#True)
        # We manually pad to obtain the same effect as `SAME` (necessary when `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=False)#True)
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut_k = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                      stride=stride, padding=0, bias=False)
            # self.shortcut_km1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
            #                           stride=stride, padding=0, bias=False)                                      
        else:
            self.shortcut_k = None
            # self.shortcut_km1 = None
        self._stride = stride
        self.start_block_vis = torch.nn.Identity()
    def forward(self, x_and_f):
        x_k = x_and_f[0]
        x_k = self.start_block_vis(x_k)
        # x_km1 = x_and_f[1] 
        # f_km1 = x_and_f[2]
        if self.has_shortcut:
            x_k = self.relu_0(self.batchnorm_0(x_k))
        else:
            f_k = self.relu_0(self.batchnorm_0(x_k))
        v = x_k if self.has_shortcut else f_k
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        f_k = self.conv_0(v)
        f_k = self.relu_1(self.batchnorm_1(f_k))
        f_k = self.conv_1(f_k)
        if self.has_shortcut:
            x_k = self.shortcut_k(x_k)
            # f_km1 = self.shortcut_km1(f_km1)
        x_kp1 = x_k + 1*f_k #- 1*f_km1

        # x_and_f = torch.stack( (x_kp1, f_k), dim=0 )
        # print('x_and_f1',len(x_and_f))
        x_and_f[0] = x_kp1
        if len(x_and_f) == 1:
            x_and_f = [x_and_f[0]] + [f_k]             
        else:
            x_and_f = [x_and_f[0]] + [f_k] + x_and_f[1:]
        # for a in x_and_f:
        #     print('type(a)',type(a))
        # print('x_and_f2',len(x_and_f))
        return x_and_f

class _Block(nn.Module):
    """
    WideResNet_learn Block.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """
    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU,coesB=[0,0,0],input_size=0,learn=False,IniRes=False,Mask=False,IniCh=16,settings='',Layer_idx=None, coesDecay=None):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=False)#True)
        # We manually pad to obtain the same effect as `SAME` (necessary when `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=False)#True)
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        self.coesB = coesB

        if self.has_shortcut:
            self.shortcuts = nn.ModuleList([])
            for i in range(0,len(self.coesB)):
                self.shortcuts.append(nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                      stride=stride, padding=0, bias=False))
            # self.shortcut_k = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
            #                           stride=stride, padding=0, bias=False)
            # self.shortcut_km1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
            #                           stride=stride, padding=0, bias=False)                                      
        else:
            # self.shortcut_k = None
            # self.shortcut_km1 = None
            self.shortcuts = None
        self._stride = stride
        if Mask is True:
            size = input_size*stride
        else:
            size = 1
        print('Mask',Mask)
        # self.coesB = nn.ParameterList([nn.Parameter(torch.ones(size, size)*self.coesB[i], requires_grad=learn) for i in range(len(self.coesB))])
        # for i in range(len(self.coesB)):
        #     self.register_parameter('coesB{}'.format(i), self.coesB[i])
        self.coesB0 = nn.Parameter(torch.ones(size, size)*self.coesB[0], requires_grad=learn)
        self.coesB1 = nn.Parameter(torch.ones(size, size)*self.coesB[1], requires_grad=learn)
            # print(self.coesB_{}.format(i))
        self.stepsize = 1.0
        self.start_block_vis = torch.nn.Identity()
        self.settings = settings
        if 'LearnBal' in self.settings:
            # print('LearnBal')
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
            self.coes_bias_inner = nn.Parameter(torch.zeros(1), requires_grad = True)
            self.coes_bias_outer = nn.Parameter(torch.zeros(1), requires_grad = True)
        else:
            self.coes_bias_inner = nn.Parameter(torch.zeros(1), requires_grad = False)
            self.coes_bias_outer = nn.Parameter(torch.zeros(1), requires_grad = False)
    def forward(self, x_and_f):
        x_k = x_and_f[0]        
        x_k = self.start_block_vis(x_k)
        # x_km1 = x_and_f[1]             
        # x_k = x_and_f[0]
        # x_km1 = x_and_f[1] 
        # x_k = x_and_f[0]
        # x_km1 = x_and_f[1] 
        # f_km1 = x_and_f[2]
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
            # print('self.coesDecay',self.coesDecay)
            if self.has_shortcut:
                x_k = self.relu_0(self.batchnorm_0(x_k))
                if 'InvExp' in self.settings:
                    x_k_exp = torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.coes_bias_inner)
                else:
                    x_k_exp = torch.exp(self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.coes_bias_inner)

            else:
                if 'InvExp' in self.settings:
                    x_k_exp = torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.coes_bias_inner)
                else:
                    x_k_exp = torch.exp(self.coesDecay*self.Layer_idx*self.coes_stepsize)*(x_k-self.coes_bias_inner)
                f_k = self.relu_0(self.batchnorm_0(x_k_exp))
                
            v = x_k_exp if self.has_shortcut else f_k
            if self._stride == 1:
                v = F.pad(v, (1, 1, 1, 1))
            elif self._stride == 2:
                v = F.pad(v, (0, 1, 0, 1))
            else:
                raise ValueError('Unsupported `stride`.')
            f_k = self.conv_0(v)
            f_k = self.relu_1(self.batchnorm_1(f_k))
            f_k = self.conv_1(f_k)
            if 'ExpDecay' in self.settings:
                # print('ExpDecay')
                # print('self.coesDecay',self.coesDecay)
                # f_k =  torch.mul((1-torch.sigmoid(self.coesBalance)), torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize) * (f_k-self.coesDecay*x_k) ) +torch.mul(torch.sigmoid(self.coesBalance), f_k)   
                # f_k.mul(torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize))
                if 'InvExp' in self.settings:
                    f_k = torch.exp(self.coesDecay*self.Layer_idx*self.coes_stepsize) * f_k -self.coesDecay*x_k+self.coes_bias_outer
                else:
                    f_k = torch.exp(-self.coesDecay*self.Layer_idx*self.coes_stepsize) * f_k -self.coesDecay*x_k+self.coes_bias_outer
        else:
            if self.has_shortcut:
                x_k = self.relu_0(self.batchnorm_0(x_k))
            else:
                f_k = self.relu_0(self.batchnorm_0(x_k))
            v = x_k if self.has_shortcut else f_k
            if self._stride == 1:
                v = F.pad(v, (1, 1, 1, 1))
            elif self._stride == 2:
                v = F.pad(v, (0, 1, 0, 1))
            else:
                raise ValueError('Unsupported `stride`.')
            f_k = self.conv_0(v)
            f_k = self.relu_1(self.batchnorm_1(f_k))
            f_k = self.conv_1(f_k)          
            
            
        f_km = [f_k]
        for i in range(1,len(self.coesB)):
            # print('i_1',i)
            # print('type(x_and_f[i])',type(x_and_f[i]))
            f_km += [x_and_f[i]]
        if self.has_shortcut:
            # x_k = self.shortcut_k(x_k)
            # f_km1 = self.shortcut_km1(f_km1)
            x_k = self.shortcuts[0](x_k)
            for i in range(0,len(self.coesB)-1):
                f_km[i] = self.shortcuts[i+1](f_km[i])
                

        # x_kp1 = x_k + 1*f_k - 1*f_km1
        x_kp1 = x_k 
        # print('len(self.coesB)',len(self.coesB))
        # for i in range(0,len(self.coesB)):
        #     print('i_2',i)
        # #     print('self.coesB[i]',float(self.coesB[i].data))
        # #     if isinstance(f_km[i],list):
        # #         print('len(f_km[i])',len(f_km[i]))
        # #     else:
        # #         print('f_km[i].size()',f_km[i].size())
            
        # #     x_kp1 = x_kp1 - self.coesB[i]*self.stepsize*f_km[i]
        #     x_kp1 = x_kp1 - torch.mul(self.coesB[i], self.stepsize*f_km[i])
        # print('self.coesB0,self.coesB1',self.coesB0,self.coesB1)
        x_kp1 = x_kp1 - torch.mul(self.coesB0, self.stepsize*f_km[0]) - torch.mul(self.coesB1, self.stepsize*f_km[1])      
        # for i in range(1,len(self.coesB)):
        #     f_km[i] = f_km[i-1]

        # f_km[0] = f_k
        
        # x_kp1 = x_k - self.coesB[0]*self.stepsize*f_k - self.coesB[1]*self.stepsize*f_km1
        # print('self.coesB',self.coesB.data)
        x_and_f = [x_kp1]
        for i in range(0,len(self.coesB)-1):
            # print('i_3',i)
            # x_and_f = torch.stack( (x_and_f, f_km[i]), dim=0 )
            x_and_f += [f_km[i]]
        # x_and_f = torch.stack( (x_kp1, x_k, f_k), dim=0 )
        return x_and_f        
        # x_kp1 = torch.add(self.shortcut_k(x_k) if self.has_shortcut else x_k, f_k)
        # return x_kp1


class _BlockGroup(nn.Module):
    """
    WideResNet_learn block group.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """
    def __init__(self, num_blocks, nb_ini_block, input_size, in_planes, out_planes, stride, activation_fn=nn.ReLU, coesB=[0,0,0],learn=False,IniRes=False,Mask=False,IniCh=16,settings='', coesDecay=-1):
        super().__init__()
        block = []
        self.settings = settings
        for i in range(num_blocks):
            l_ = i
            
            if 'RestaLayerIdx' in self.settings:
                
                if self.settings.split('RestaLayerIdx')[1]:
                    # print(self.settings.split('RestaLayerIdx')[1])
                    Split = int(self.settings.split('RestaLayerIdx')[1])
                    l_ = i%Split
                    print('Split',Split)    
                    
            print('l_',l_)
            if i < nb_ini_block:
                block.append(
                    _BlockIni(i == 0 and in_planes or out_planes, 
                        out_planes,
                        i == 0 and stride or 1,
                        activation_fn=activation_fn,coesB=coesB,input_size=input_size,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh)
                )
            else:
                block.append(
                    _Block(i == 0 and in_planes or out_planes, 
                        out_planes,
                        i == 0 and stride or 1,
                        activation_fn=activation_fn,coesB=coesB,input_size=input_size,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh, settings=settings,Layer_idx=l_, coesDecay=coesDecay)
                )
        self.block = nn.Sequential(*block)

    def forward(self, x_and_f):
        # return self.block(x_and_f)
        return [self.block(x_and_f)[0]]


class WideResNet_learn(nn.Module):
    """
    WideResNet_learn model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
    """
    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3,
                 coesB=[0,0,0],learn=False,IniRes=False,Mask=False,IniCh=16,settings='', IniDecay=None):
        super().__init__()
        
        self.settings = settings
        if 'LearnDecay' in self.settings:
            LearnDecay = True
        else:
            LearnDecay = False
        # if 'ExpDecay' in self.Settings:   
        coesDecay =nn.Parameter(torch.ones(1)*IniDecay, requires_grad=LearnDecay)
                    
        self.ini_nb = int(len(coesB)-1)
        self.set_ini = self.ini_nb
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        num_channels = [IniCh, IniCh * width, IniCh *2 * width, IniCh *4 * width]
        

        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.input_size = [32,16,8]
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=3, stride=1, padding=1, bias=False)
                                   
                            
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, self.set_ini, self.input_size[0], num_channels[0], num_channels[1], 1,
                        activation_fn=activation_fn,coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh, settings=settings, coesDecay=coesDecay),
            _BlockGroup(num_blocks, self.set_ini, self.input_size[1], num_channels[1], num_channels[2], 2,
                        activation_fn=activation_fn,coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh, settings=settings, coesDecay=coesDecay),
            _BlockGroup(num_blocks, self.set_ini, self.input_size[2], num_channels[2], num_channels[3], 2,
                        activation_fn=activation_fn,coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh, settings=settings, coesDecay=coesDecay))
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=False)#True)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]
        self.IniRes = IniRes
        if self.IniRes is True:
            self.init_ds = nn.Conv2d(num_input_channels, num_channels[0],
                                   kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

                
    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std
        #TODO:
        if self.IniRes is True:
            out = self.init_conv(out)+self.init_ds(out)
        else:
            out = self.init_conv(out)
        # out = torch.stack( (out, out, out), dim=0 )
        out = [out]
        out = self.layer(out)
        out = out[0]
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)
    
    
def narrowresnetwithswish_learn(name, dataset='cifar10', num_classes=10, device='cuda', coesB=[0,0,0],learn=False,IniRes=False,Mask=False,IniCh=16,settings='',IniDecay=None):
    """
    Returns suitable WideResNet_learn model with Swish activation function from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        device (str or torch.device): device to work on.
        dataset (str): dataset to use.
    Returns:
        torch.nn.Module.
    """
    # if 'cifar10' not in dataset:
    #     raise ValueError('WideResNet_learns with Swish activation only support CIFAR-10 and CIFAR-100!')

    name_parts = name.split('-')
    depth = int(name_parts[1])
    widen = int(name_parts[2])
    act_fn = name_parts[3]
    
    print (f'WideResNet_learn-{depth}-{widen}-{act_fn} uses normalization.')
    if 'cifar100' in dataset:
        return WideResNet_learn(num_classes=num_classes, depth=depth, width=widen, activation_fn=_ACTIVATION[act_fn], 
                          mean=CIFAR100_MEAN, std=CIFAR100_STD, coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh,settings=settings,IniDecay=IniDecay)
    elif 'svhn' in dataset:
        return WideResNet_learn(num_classes=num_classes, depth=depth, width=widen, activation_fn=_ACTIVATION[act_fn], 
                          mean=SVHN_MEAN, std=SVHN_STD, coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh,settings=settings,IniDecay=IniDecay)
    return WideResNet_learn(num_classes=num_classes, depth=depth, width=widen, activation_fn=_ACTIVATION[act_fn], coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh,settings=settings,IniDecay=IniDecay)

def create_network():
    net = narrowresnetwithswish_learn(name='nrn-70-1-swish-learn',coesB=[-1,0],learn=True,IniRes=False,Mask=False,IniCh=4,settings='ShareExpDecayLearnDecay_AbsExp_Ini0p07_RestaLayerIdx4',IniDecay=0.07)

    return net


if __name__ == '__main__':
    net = create_network().to(device="cuda:0")
    inp = torch.rand(1,3,32,32).to(device="cuda:0")
    out = net(inp)
    print(net)    
    print('out\n\n',out)
    for name, p in net.named_parameters():   
        if p.requires_grad:
            print(name, p.size()) 