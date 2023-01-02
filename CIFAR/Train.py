from comet_ml import Experiment, OfflineExperiment, Optimizer
import nni

from absl import app, flags
from easydict import EasyDict
import numpy as np
import random

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


# import tensorwatch as tw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import torch.onnx
import netron
import onnx
# from tensorboardX import SummaryWriter
# import sys
# sys.path.append(r"directory")

from models import *
from noise import *

from datetime import datetime
import errno
import shutil
import pandas as pd

# import homura
from torchvision import transforms
# from timm.loss import LabelSmoothingCrossEntropy
from asam import ASAM, SAM
# from bypass_bn import enable_running_stats, disable_running_stats
from torchsummaryX import summary
from torchvision import transforms
import matplotlib.pyplot as plt

import matplotlib
import torchextractor as tx

from torch.utils.data import DataLoader
import math
import warnings
 
from robustbench.data import load_cifar10

from robustbench.utils import load_model 
import foolbox as fb
from torch.autograd import Variable
from models import (convert_splitbn_model, create_model, load_checkpoint,
                         model_parameters, resume_checkpoint, safe_model_name)
warnings.filterwarnings('ignore')
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'

print('device',device)
# seed_torch()


# parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')#TODO 0.000125 #0.1
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=5, help='training epoch')
parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
parser.add_argument('--data', default='./data/', type=str)# /media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/data#/media_SSD_1/datasets/
parser.add_argument('--dataset', default='cifar100', type=str)#cifar10mnist
parser.add_argument('--arch', '-a', default='convnext_conver_nano_hnf', type=str)#ZeroSAny20settings#convnext_conver_nano_hnf#convnext_conver_tiny#convnext_tiny
# parser.add_argument('--arch', '-a', default='ZeroSAny20_Tra', type=str)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')#128
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--opt', default='SGD', type=str)#AdamW
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--notes", default='SumBase_ab1', type=str)
parser.add_argument('--PL', type=float, default=1.0)
parser.add_argument('--sche', default='cos', type=str)
# parser.add_argument('--coe_ini', type=float, default=1)
parser.add_argument('--share_coe', type=bool, default=False)
# parser.add_argument('--given_coe', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)
parser.add_argument('--given_coe', default=None, nargs='+', type=float)
parser.add_argument('--steps', type=int, default=3)
parser.add_argument('--ini_stepsize', default=1, type=float)
# parser.add_argument('--givenA', default=[1.49323762497707, -0.574370781405754, 0.0855838379295368, -0.00445068150085398], nargs='+', type=float)
# parser.add_argument('--givenA', default=[1.,1.,1.,1.,1.], nargs='+', type=float)
# parser.add_argument('--givenB', default=[-1., 1., 1., 1.], nargs='+', type=float)#S4O0
# parser.add_argument('--givenB', default=[-2.10313656405320, 5.60787753612393, -7.29272571153819, 3.20453988951669], nargs='+', type=float)#S4O2
# parser.add_argument('--givenB', default=[ -2.10313656405320, 2.80393876806197, -1.68484817541425, 0.400601121454727], nargs='+', type=float)#S4O4

# parser.add_argument('--givenA', default=[1, 0, 0, 0, 0], nargs='+', type=float)#S5
# parser.add_argument('--givenA', default=[ 1, 0, 0, 0, 0,0], nargs='+', type=float)
# parser.add_argument('--givenB', default=[-1, 0, 0, 0, 0,0], nargs='+', type=float)#S5O1
parser.add_argument('--givenA', default=[1, 0], nargs='+', type=float)#S5
parser.add_argument('--givenB', default=[-1, 0], nargs='+', type=float)#S5O1
# parser.add_argument('--givenB', default=[-2.9701388888889, -5.5020833333333,  -6.9319444444444, -5.0680555555556, -1.9979166666667, -0.32986111111111], nargs='+', type=float)#S6O0
# parser.add_argument('--givenB', default=[5.5020833333333,  -6.9319444444444, 5.0680555555556, -1.9979166666667, 0.32986111111111, -2.9701388888889], nargs='+', type=float)#S6O1
# parser.add_argument('--givenB', default=[-2.9701388888889, 5.5020833333333,  -6.9319444444444, 5.0680555555556, -1.9979166666667, 0.32986111111111], nargs='+', type=float)#S6O6
parser.add_argument("--ConverOrd", default=4, type=int, help="")


parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

parser.add_argument("--minimizer", default=None, type=str, help="ASAM or SAM.")
parser.add_argument("--smoothing", default=None, type=float, help="Label smoothing.")#0.1
parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
parser.add_argument("--eta", default=0.01, type=float, help="Eta for ASAM.")
    
parser.add_argument("--adv_train", action='store_true')
parser.add_argument("--adv_test", action='store_true')
parser.add_argument("--eps", default=0.03137255, type=float, help="")
parser.add_argument("--eps_iter", default=0.007843, type=float, help="step size for each attack iteration")#0.01
parser.add_argument("--nb_iter", default=10, type=int, help="Number of attack iterations.")
parser.add_argument("--norm", default=np.inf, type=float, help="Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.")
parser.add_argument("--clip_min", default=0, type=float, help="Minimum float value for adversarial example components.")
parser.add_argument("--clip_max", default=1, type=float, help="Maximum float value for adversarial example components.")
parser.add_argument("--save_path", default='./runs/features/', type=str, help="save path")
parser.add_argument("--settings", default='BnReluConv_ConvStride2ResLike_ShareExpDecayLearnDecay_AbsExp_Adam_RestaLayerIdx3', type=str, help="settings")#ShareExpDecayLearnDecay_AbsExp#BnReluConvConvStride2ResLikeShareExpDecayLearnDecay_GLExp#mnistLearnCoeExpDecayLearnDecay#_BnReluConvConvAllEle#ShareExpDecayLearnDecay_AbsExp_Adam
parser.add_argument("--req_ord", default=5, type=int, help="")
parser.add_argument("--mode", default='Normal', type=str, help="")#mix#Normal
parser.add_argument("--ini_block_shift", default=None, type=int, help="")
parser.add_argument("--IniDecay", default=0.07, type=float, help="")
parser.add_argument("--CoesLR", default=0.04, type=float, help="")
# TODO: print inputs
    
args =parser.parse_known_args()[0]

# opt = Optimizer(sys.argv[1])

# params = {
#     'IniDecay': -2.0,
# }
# optimized_params = nni.get_next_parameter()
# params.update(optimized_params)
# print(params)

# Next, create an optimizer, passing in the configuration:
# opt = Optimizer(sys.argv[1])

# args = parser.parse_args()
# args, unknown = parser.parse_known_args()
# parser.parse_args(args=[])
# args = parser.parse_args(argv[1:])
# args = EasyDict( {
#     'epoch': 5,
#     "lr": 0.1,
#     'warm': 0,
#     'data': '/media/bdc/clm/OverThreeOrders/CIFAR/data',
#     'dataset': 'mnist',
#     'arch': ZeroSAny32Ablation,
#     "bs": 128, 
#     'momentum': 0.9,
#     'weight_decay': 4e-5,
#     'opt': 'SGD',
#     'PL': 1,
#     'sche': 'cos',
#     'coe_ini': 1,
#     'share_coe': 0,
#     'ini_stepsize': 1,
#     'givenA':[1, 0, 0, 0, 0, 0],
#     'givenB':[-2.9701388888889, 5.5020833333333,  -6.9319444444444, 5.0680555555556, -1.99791Accuracy_pgd:66666667, 0.32986111111111],
#     # 'steps': args.steps,
#     'notes': '',
#     'minimizer': None,
#     'smoothing': None,
#     'rho': 0.5,
#     'eta': 0.01,
#     'adv_train': None,
#     'settings': 'mnistAllEle',
#     'save_path': '/media/bdc/clm/OverThreeOrders/CIFAR/runs/features/',
#     'checkpoint': None,
#     })

def prepare(args):
    givenA_text = ''
    givenB_text = ''
    if args.givenA is not None:
        for i in range(len(args.givenA)): 
            givenA_text += "a"+str(i)+"_"+str(args.givenA[i])[:6]
            givenB_text += "_b"+str(i)+"_"+str(args.givenB[i])[:6]
    else:
        givenA_text = ''
        givenB_text = ''
    if args.share_coe:
        share_coe_text = 'share_coe_True'
    else:
        share_coe_text = 'share_coe_False'
    if args.dataset == "cifar10" or args.dataset == "stl10":
        args.num_classes = 10
        if args.sche == 'step' and args.epoch is None:
            args.epoch = 160
    if args.dataset == "cifar100":
        args.num_classes = 100
        if args.sche == 'step'and args.epoch is None:
            args.epoch = 300
    if args.dataset == "mnist":
        args.num_classes = 10
        if args.sche == 'step' and args.epoch is None:
            args.epoch = 10
        elif args.sche == 'cos' and args.epoch is None:
            args.epoch = 5        
    if args.dataset == "svhn":
        args.num_classes = 10
        if args.sche == 'step' and args.epoch is None:
            args.epoch = 40        
    path_base = './runs/' + args.dataset +str(args.adv_train)+'/ZeroSNet/WithRobSGD_noLS_adv/'
    if args.adv_train:
        print('adv_train=True')
    #    path_base = './adv_train_runs' + path_base.replace('./', '/')+'eps'+str(args.eps)+'/'
        if not args.save_path == None:
            args.save_path = './runs/adv_train_runs'+'eps'+str(args.eps)+'/'+args.save_path.replace('./', '/').replace('runs', '')
    try:
        path_base = "./runs/"+args.dataset+"/CometData"
        os.makedirs(path_base)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path_base):
            pass
        else:
            raise
    try:
        os.makedirs(args.save_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(args.save_path):
            pass
        else:
            raise      

        
    if args.save_path == None:
        args.save_path = path_base + args.arch + '/ini_st' + str(
            args.ini_stepsize)  + givenA_text + givenB_text +'_sche_' + args.sche + str(args.opt) + \
                        '_mini'+args.minimizer+'_BS' + str(args.bs) + '_LR' + \
                        str(args.lr) + 'epoch' + \
                        str(args.epoch) + 'warm' + str(args.warm) + \
                        args.notes +'eps'+str(args.eps)+ 'eps_iter'+str(args.eps_iter)+'nb_iter'+str(args.nb_iter)+\
                            '_G4_'+\
                        "{0:%Y-%m-%dT:%H:%M:%S/}".format(datetime.now())


    # checkpoint
    if args.checkpoint is None:
        args.checkpoint = args.save_path+'checkpoint.pth.tar'
        print('args.checkpoint', args.checkpoint)
    # print('givenB', args.givenB, 'givenA', args.givenA)
    # hyper_params = {
    #     'settings':args.settings,
    #     'req_ord':args.req_ord,
    #     'mode':args.mode,
    #     'epoch': args.epoch,
    #     "learning_rate": args.lr,
    #     'warmup': args.warm,
    #     'dataset': args.dataset,
    #     'arch': args.arch,
    #     "batch_size": args.bs, 
    #     'momentum': args.momentum,
    #     'wd': args.weight_decay,
    #     'opt': args.opt,
    #     # 'PL': args.PL,
    #     'sche': args.sche,
    #     # 'coe_ini': args.coe_ini,
    #     # 'share_coe': args.share_coe,
    #     'ini_stepsize': args.ini_stepsize,
    #     'givenA': args.givenA,
    #     'givenB': args.givenB,
    #     # 'steps': args.steps,
    #     'notes': args.notes,
    #     'minimizer': args.minimizer,
    #     'smoothing': args.smoothing,
    #     'rho': args.rho,
    #     'eta': args.eta,
    #     'adv_train': args.adv_train,
    #     'IniDecay': args.IniDecay,
    #     }
    return args

def coe_constrains(coe_As, coe_Bs,req_ord=1,prin=False):
    # coe_A0 = [torch.tensor(-1.).to(device).reshape(1)]
    # coe_B0 = [torch.tensor(0.).to(device).reshape(1)]
    # coe_As = coe_A0 + coe_As
    # coe_Bs = coe_B0 + coe_Bs
    # print('coe_Bs',coe_Bs)
    # TODO: a0 should not be trained
    coe_A0 = torch.tensor(-1.).to(device).reshape(1)
    coe_B0 = torch.tensor(0.).to(device).reshape(1)
    coe_As = torch.cat((coe_A0,coe_As),0)
    coe_Bs = torch.cat((coe_B0,coe_Bs),0)
    err = torch.tensor([]).to(device)

    # coe_As = coe_A0 + coe_As
    # coe_Bs = coe_B0 + coe_Bs    
    # err = []
    for i in range(0, req_ord+1):
        if i == 0:
            # err += [sum(coe_As)]
            # err = torch.cat((err, torch.sum(coe_As).reshape(1) ),0)
            err = torch.sum(coe_As).reshape(1)
        else:
            As = 0
            Bs = 0
            for j in range(1,len(coe_As)):
                As += j**i*coe_As[j]
            As *= 1/math.factorial(i)
            for j in range(0,len(coe_Bs)):
                Bs += j**(i-1)*coe_Bs[j]
            Bs *= 1./math.factorial(i-1)
            # err += [(-1)**i*(As+Bs)]
            err = torch.cat((err, ((-1)**i*(As+Bs)).reshape(1) ),0)

    if prin:
        print('ord'+str(i),err)
        print('coe_As,coe_Bs',coe_As,coe_Bs)
    return err

def plot_feature(net, epoch,save_path):
    # plt.style.use('ieee')
    plt.style.use(['science','ieee'])  
    # print(matplotlib.rcParams)
    matplotlib.rcParams.update(
        {
            # 'text.usetex': False,
            # 'font.family': 'stixgeneral',
            
            # 'font.size': 24.0,        
            # 'legend.fontsize': 'medium',
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',         
        'axes.labelsize': 'x-large',
        'legend.frameon' : True,
        'legend.fontsize' : 'large',
        'legend.fancybox' : False, 
        "legend.facecolor" : 'white',   
        'axes.grid': True,  
        'axes.grid.axis': 'x', 
            # 'axes.titlesize': 'large','mathtext.fontset': 'stix',
            }
    )      
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)  
    # save_path = args.save_path+args.settings+args.arch+str(args.epoch)+givenB_text+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net.eval()
    # with torch.no_grad():         
    for i, (images, labels) in enumerate(testloader):

        if i==0:#(i + 1) % 100 == 0:
            print('batch_number [{}/{}]'.format(i + 1, len(testloader)))
            for j in range(len(images)):
                if j == 1:
                    if args.dataset == 'mnist':                    
                        image = images[j].resize(28, 28).to(device) 
                    elif 'cifar' in args.dataset or 'svhn' in args.dataset:                   
                        image = images[j].to(device)                         
                    print('labels[j]',labels[j])                    
                    break
            break


    # plt.rcParams.update({'font.size':14})
    if args.dataset == 'mnist':
        d = image.reshape(1,1,28,28)#.reshape(28,28)
        loc = 2
    elif 'cifar' in args.dataset or 'svhn' in args.dataset:
        d = image.reshape(1,3,32,32)#.reshape(28,28)
        loc = 8
    elif 'tiny-imagenet' in args.dataset:
        d = image.reshape(1,3,64,64)#.reshape(28,28)
        loc = 8
    # plt.imshow(d)
    d= d.to(device)

    df = pd.DataFrame()
    plt.xlabel('$n$')#, fontsize=24)
    plt.ylabel('$x_n$')#, fontsize=24)
    out_0 = net(d) 
    MaxDif = []
    MaxDif2 = []
    if 'convnext' in args.arch:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity) and 'end_block_vis' in name #torch.nn.Identity)
    else:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity) #and 'start_block_vis' in name
    lb = 0
    ub = 50
    for mag in range(lb,ub,1):

        # d = d + mag/1000
        if args.dataset == 'mnist':
            x = d + mag/100*torch.rand(1,1,28,28).to(device)
        elif 'cifar' in args.dataset or 'svhn' in args.dataset:
            x = d + mag/100*torch.rand(1,3,32,32).to(device)       

        model = tx.Extractor(net, module_filter_fn=module_filter_fn)
        out, features = model(x) 
        
        fmap = []
        fmap2 = []

        feature_shapes = {name: f.shape for name, f in features.items()}
        for name, f in features.items():
            if len(f.data.shape) == 2:
                # fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2])]
                # print('f.data',f.data.shape)                
                fmap += [float(f.data[f.data.shape[0]//2][loc])]                
                fmap2 += [float(f.data[f.data.shape[0]//2][0])] 

            elif len(f.data.shape) == 4:
                fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][f.data.shape[2]//2][f.data.shape[3]//2])]
                fmap2 += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][0][0])]
                #[float(f.data.mean())]
        fmap = np.array(fmap)
        fmap2 = np.array(fmap2)                
        if mag == lb: 
            fmap_ori = fmap
            fmap_ori2 = fmap2
        fmap_gap = fmap-fmap_ori
        fmap2_gap = fmap2-fmap_ori2
        
        MaxDif += [fmap]
        MaxDif2 += [fmap2]
         
        

        # # df_row = pd.DataFrame([fmap])
        # ts = np.arange(0, len(fmap), 1)
        # pic1, = plt.plot(ts[0:-3], fmap[0:-3], linewidth = 1,color='indianred',alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white') 
        # # df = pd.concat([df,df_row])
        # pic2, = plt.plot(ts[0:-3], fmap2[0:-3], color='steelblue',linewidth = 1,alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white')
        # if mag == lb:
        #     plt.legend(handles=[pic1,pic2], labels=['Center','Corner'])
        # plt.xlabel('$n$')#, fontsize=24)
        # plt.ylabel('$x_n$')#, 
  
        
        # df_row = pd.DataFrame([fmap])
        ts = np.arange(0, len(fmap), 1)
        pic1, = plt.plot(ts[0:-3], fmap_gap[0:-3], linewidth = 1,color='indianred',alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white') 
        # df = pd.concat([df,df_row])
        pic2, = plt.plot(ts[0:-3], fmap2_gap[0:-3], color='steelblue',linewidth = 1,alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white')
        if mag == lb:
            plt.legend(handles=[pic1,pic2], labels=['Center','Corner'])
        plt.xlabel('$n$')#, fontsize=24)
        plt.ylabel('$\Delta x_n$')#,         
        
              
    # plt.show()      

        
    file_name = '/features_normal_ep'+str(epoch)
    # df.to_csv(save_path+file_name+'.csv',index = False)
    ax.figure.savefig(save_path+file_name+'.pdf')
    plt.close()
    # MaxDif_adv = []
    # MaxDif2_adv = []  
    # ax_adv = plt.gca()
    # ax_adv.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)   
    # plt.xlabel('$n$')#, fontsize=24)
    # plt.ylabel('$\Delta x_n$')#, fontsize=24)           
    # for mag in range(1,10):#10,30,1):
 
    #     x_adv = projected_gradient_descent(net, d, eps=4.0/255*mag, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm,clip_min=args.clip_min, clip_max=args.clip_max) 
                   
    #     model = tx.Extractor(net, module_filter_fn=module_filter_fn)
    #     out, features = model(x_adv) 
    #     fmap = []
    #     fmap2 = []
    #     feature_shapes = {name: f.shape for name, f in features.items()}
    #     for name, f in features.items():

    #         if len(f.data.shape) == 2:
    #             # fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2])]
    #             # print('f.data',f.data.shape)                
    #             fmap += [float(f.data[f.data.shape[0]//2][loc])]                
    #             fmap2 += [float(f.data[f.data.shape[0]//2][0])] 

    #         elif len(f.data.shape) == 4:
    #             fmap += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][f.data.shape[2]//2][f.data.shape[3]//2])]
    #             fmap2 += [float(f.data[f.data.shape[0]//2][f.data.shape[1]//2][0][0])]
    #             #[float(f.data.mean())]       
    #     fmap = fmap-fmap_ori
    #     fmap2 = fmap2-fmap_ori2                         
    #     MaxDif_adv += [fmap]
    #     MaxDif2_adv += [fmap2]
    #     df_row = pd.DataFrame([fmap])
    #     ts = np.arange(0, len(fmap), 1)

    #     pic1, = plt.plot(ts[0:-3], fmap[0:-3], linewidth = 1,color='darkorange',alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white') 
    #     df = pd.concat([df,df_row])
    #     pic2, = plt.plot(ts[0:-3], fmap2[0:-3], color='darkcyan',linewidth = 1,alpha=0.4,linestyle='-', marker = 'o', markersize=1, markeredgecolor='black', markerfacecolor='white')
    #     if mag == 1:
    #         plt.legend(handles=[pic1,pic2], labels=['Center$+$AT','Corner$+$AT'])              

        
    # out = net(x) 
    # out_adv = net(x_adv) 
    
    # print('out_diff,out_diff_adv', torch.abs(out-out_0).mean(),torch.abs(out_adv-out_0).mean() )
  
    # # plt.tick_params(labelsize=12)
    # # print("==>> type(plt.ylim()): ", plt.ylim()

    # file_name = '/features_adv_ep'+str(epoch)
    # df.to_csv(save_path+file_name+'.csv',index = False)
    # ax_adv.figure.savefig(save_path+file_name+'.pdf')
    # # plt.show()
    # plt.close()
    ax_dif = plt.gca()
    ax_dif.ticklabel_format(style='sci', scilimits=(-1,2),useLocale=True)  
    plt.xlabel('$n$')#, fontsize=24)
    plt.ylabel('$\max{|\Delta x_n|}$')#, fontsize=24)         
    MaxDif = np.array(MaxDif)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3], color='indianred',label='Center') 
    print('MaxOutDif',MaxDif[-1])
    print('MaxOutFeaDif2',MaxDif[-3])


    MaxDif = np.array(MaxDif2)
    MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    MaxDif = MaxDif.reshape(np.shape(ts))      
    plt.plot(ts[0:-3], MaxDif[0:-3],color='steelblue', label='Corner')
    # plt.legend() 
    print('MaxOutDif2',MaxDif[-1])
    print('MaxOutFeaDif2',MaxDif[-3])

    # MaxDif = np.array(MaxDif_adv)
    # MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    # MaxDif = MaxDif.reshape(np.shape(ts))      
    # plt.plot(ts[0:-3], MaxDif[0:-3], color='darkorange',label='Center$+$AT') 
    # print('MaxOutDif_adv',MaxDif[-1])
    # print('MaxOutFeaDif_adv',MaxDif[-3])

    # # plt.show()

    # MaxDif = np.array(MaxDif2_adv)
    # MaxDif = np.max(MaxDif, axis=0)-np.min(MaxDif, axis=0)
    # MaxDif = MaxDif.reshape(np.shape(ts))      
    # plt.plot(ts[0:-3], MaxDif[0:-3],color='darkcyan', label='Corner$+$AT')
    # plt.legend(facecolor='w',framealpha=0.5) 
    # print('MaxOutDif2_adv',MaxDif[-1])
    # print('MaxOutFeaDif2_adv',MaxDif[-3])    

    # plt.show()
    

    file_name = '/diff_ep'+str(epoch)
    df.to_csv(save_path+file_name+'.csv',index = False)
    ax_dif.figure.savefig(save_path+file_name+'.pdf')
    plt.close()


def train(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.train()
    
    loss = 0.
    acc = 0.
    cnt = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        '''
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_sgd_sgd.zero_grad()
        if 'zeros' in args.arch or 'ZeroS' in args.arch:
            outputs, coes, stepsize = net(inputs)
        elif 'MResNet' in args.arch:
            outputs, coes = net(inputs)
        else:
            outputs = net(inputs)

        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        # print('acc1', acc1, 'acc5', acc5)
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        loss.mean().backward()#asam
        # loss.backward() 
        minimizer.ascent_step()#asam
        # optimizer.step()
        
        if 'zeros' in args.arch or 'ZeroS' in args.arch:
            outputs, coes, stepsize = net(inputs)
        elif 'MResNet' in args.arch:
            outputs, coes = net(inputs)
        else:
            outputs = net(inputs)
            
        criterion(outputs, targets).mean().backward()#asam
        minimizer.descent_step()#asam

        if epoch > args.warm:
            train_scheduler_sgd.step()#epoch)
        if epoch <= args.warm:
            warmup_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
'''
        
        # if args.dataset == 'svhn':
        #     inputs, targets = inputs.to(device), targets.to(device)#.long().squeeze()
        # else:
        inputs, targets = inputs.to(device), targets.to(device)            
        # print('inputs.max(), inputs.min()', inputs.max(), inputs.min())
        
        if args.adv_train:
            # Replace clean example with adversarial example for adversarial training
            inputs = projected_gradient_descent(net, inputs, eps=args.eps, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm)     
            # print('inputs.max()adv, inputs.min()adv', inputs.max(), inputs.min())

            # TODO: add args.adv_train, args.eps
        if args.minimizer in ['ASAM', 'SAM']:
            minimizer.optimizer.zero_grad()

            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
                
            # enable_running_stats(net)

            batch_loss = criterion(outputs, targets)
            # batch_loss.backward()#asam
            
            batch_loss.mean().backward()#asam
            # loss.backward() 
            minimizer.ascent_step()#asam
            # optimizer.step()
            
            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
                
            # disable_running_stats(net)
        
            # criterion(outputs, targets).back.meanward()#asam
            criterion(outputs, targets).mean().backward()#asam
            
            minimizer.descent_step()#asam
            
            with torch.no_grad():
                loss += batch_loss.sum().item()
                acc += (torch.argmax(outputs, 1) == targets).sum().item()
            cnt += len(targets)
        
            if epoch >= args.warm:
                train_scheduler_sgd.step(epoch)#epoch)
            if epoch < args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()
        else:
            optimizer_sgd.zero_grad()
            optimizer_adam.zero_grad()

            # if 'zero' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            # elif 'MRes' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            if 'LearnCoeConstrain' in args.settings:
                loss_coe = 0
                # loss_B = 0
                # out_A = 0
                # sum_A = 0
                # sum_B = 0
                # coes_As = []
                # coes_Bs = []
            if epoch == 1 and batch_idx == 0:
                coes_As = torch.tensor([]).to(device)
                coes_Bs = torch.tensor([]).to(device)
                for name, p in net.named_parameters():
                    if 1:#p.requires_grad: 
                        if name:
                            if 'coesA' in name: # and 'bn3' not in name:
                                # print(name)
                                # sum_A = sum_A+p
                                # coes_As = coes_As+[p]
                                coes_As = torch.cat((coes_As, p ),0)
                            if 'coesB' in name: # and 'bn3' not in name:
                                # print(name)
                                # sum_B = sum_B+p
                                # coes_Bs = coes_Bs+[p]
                                coes_Bs = torch.cat((coes_Bs, p ),0) 
                
                err = coe_constrains(coes_As, coes_Bs,req_ord=args.req_ord,prin=True)

            # inputs = Variable(inputs,requires_grad=True)                
            outputs = net(inputs)
            
            # features_in = []
            # features_out = []   
            # f_ins = torch.tensor([])
            # f_outs = torch.tensor([])
            # for f_in in features_in:
            #     f_in = f_in[0].mean().unsqueeze(0)
            #     f_ins = torch.cat([f_ins,f_in],0)
            # for f_out in features_out:
            #     f_out = f_out[0].mean().unsqueeze(0)
            #     f_outs = torch.cat([f_outs,f_out],0)    
            # diff_loss = torch.sum(torch.abs(f_ins-f_outs))
            # features_in = []
            # features_out = []   
            diff_loss = 0
            loss_coe = 0
            if 'LearnConti' in args.settings:
                Diffs = torch.zeros(1).to(device)
                coe = 1

                # for hook in hookF:
                for hook in hookF_back:                
                    
                    diff = hook.output.mean()#- hook.input[0].mean()  
                    diff = torch.abs(diff)
                    ceo = coe*1
                    Diffs += diff*ceo          
                    # diff_loss = Diffs
                    diff_loss = 1.*criterion2(Diffs, torch.zeros(Diffs.size()).to(device))


            if 'LearnCoeConstrain' in args.settings:
                out_coe = coe_constrains(coes_As, coes_Bs, req_ord=args.req_ord, prin=False)
            ##

                # print('out_coe',out_coe)
                # out_coe = torch.tensor(out_coe_).to(device)
                # out_coe = err_.to(device)

                # out_A_ = torch.ones(1)#outputs.size())#,requires_grad=True)
                # out_A = out_A_.to(device)
                # # out_A.requires_grad_(True)
                # out_A = torch.abs(sum_A-1)*out_A
                # # print('out_A',out_A)
                # loss_coe = torch.sum(out_A + out_B)

                ##

                crt_ord = epoch*(len(out_coe)-1)//args.epoch+2
                out_coe = out_coe[1:crt_ord]
                
                coe_label_ = torch.zeros(out_coe.size())
                coe_label = coe_label_.to(device)
                loss_coe = 0.1 * criterion2(out_coe, coe_label)
                ##

                # loss_A.requires_grad_(True)
                # print('targets',targets)            
                # loss = 0*criterion(outputs, targets)+0*criterion(outputs2, targets)+loss_A

                loss_diff = diff_loss+loss_coe#0.2*loss_coe#0*100*criterion2(outputs2, outputs) + 
            loss = criterion(outputs, targets) #+0.*loss_diff
            # if 'loss_rob' in args.mode:
            #     loss_rob = 0.1*criterion2(outputs2, outputs)
            #     loss_diff = loss_rob
            # print('outputs.size, loss_A.size',outputs.size(), loss_A.size())
            # loss = criterion2(loss_A, coe_label)#should be MSE

            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            # top5.update(acc5[0], inputs.size(0))
            if 'GFNet' in args.mode:
                training_stage = 'GF feature training' 
                loss = loss+0.*loss_diff
                loss.backward(retain_graph=True)   
                goals = []
                loss_GF = 0
                eta = 100000*10
                ly = 0 
                grad_inputs_ =[]
                loss_GF_ls = []
                for hook_b in hookF_GF:
                     
                    inp = hook_b.input# 
                    # print('inp',inp.size()) 
                    out = hook_b.output[0]#.size()
                    # print('out',out.size())  

                    if hook_b.grad_input is not None:
                        grad_input = hook_b.grad_input#.size() 
                        # print('grad_input',grad_input)            
                        
                    if hook_b.grad_input is not None:           
                        grad_output = hook_b.grad_output#.size()

                    if inp is not None and inp.size() == out.size():
                        # fea_gap = (out-inp)-(-20000.*grad_input)#grad_input)
                        # eta = 0.9*eta
                        fea_gap = (out)-(-eta*grad_input)#grad_input)     
                        # if batch_idx == 0:           
                        #     print('GF')
                        goal_ = torch.zeros(fea_gap.size()) 
                        # print('out-inp.max(),out-inp.mean()',(out-inp).max(),(out-inp).mean())
                        # print('out.max(),out.mean()',out.max(),out.mean()) 
                        # print('grad_input.max(),grad_input.mean()',eta*grad_input.max(),eta*grad_input.mean())               
                        goals = goal_.to(device)
                        loss_ele = F.mse_loss(fea_gap, goals)
                        if batch_idx % 100 == 0:
                            # print('ly,loss_ele,grad_input',ly,float(loss_ele.data),float(grad_input.mean()))
                            grad_inputs_ += [abs(grad_input.mean())]
                        ly = ly +1
                        loss_GF += 0.0000001*loss_ele*1
                        loss_GF_ls += [0.0000001*loss_ele*1] 
                optimizer_sgd.zero_grad()
                # loss.backward()                  
                loss_GF.backward()
                      
                optimizer_sgd.step()#
                if batch_idx % 100 == 0:
                    print('loss_GF',loss_GF.data)
                    # plt.plot(grad_inputs_,label='batch_idx='+str(batch_idx) )
                    plt.plot(loss_GF_ls,label='batch_idx='+str(batch_idx)   )
            elif 'mix' in args.mode:
                training_stage = 'Mixed training' 
                loss = loss+0.*loss_diff
                loss.backward()   
                for hook_b in hookF_back:
                    if hook_b.grad_input is not None:
                        grad_input = hook_b.grad_input.size() 
                        # print('grad_input',grad_input)            
                        
                    if hook_b.grad_input is not None:           
                        grad_output = hook_b.grad_output.size()
                         
                        # print('grad_output',grad_output)
                    
                    output = hook_b.output.size()
                    input = hook_b.input.size()                               
                    # print('output',output) 
                # print('hook.output[0].grad',hook.output[0].grad)       
                optimizer_sgd.step()      
                # loss_diff.backward()
                optimizer_adam.step()                          
            else: 
                if 1:#epoch % 2==0:#epoch>=args.epoch*0.5:
                    loss.backward()          
                    optimizer_sgd.step()
                    if not 'convnext' in args.arch and 'Adam' in args.settings:
                        optimizer_adam.step()
                    training_stage = 'Normal training'
                else:
                    if 'loss_rob' in args.mode:
                        inputs2 = inputs + 0.1*torch.randn(inputs.size()).to(device)
                        outputs2 = net(inputs2)                
                        loss_rob = criterion2(outputs2, outputs)
                        loss_diff = loss_rob                    
                    loss_diff.backward()
                    # loss_rob.backward()
                    optimizer_adam.step()
                    training_stage = 'Coe training'

                # print('loss_rob',loss_rob)
            #retain_graph=True)
            # optimizer_adam.step()

            # optimizer_sgd.zero_grad()
            # optimizer_adam.zero_grad()
            # loss.backward()
            # optimizer_adam.step()

            # print('loss',loss.mean())
            # handle.remove()
            # del features_in, features_out, f_ins, f_outs
            if epoch > args.warm:
                train_scheduler_sgd.step(epoch)
                train_scheduler_adam.step(epoch)
                
            if epoch <= args.warm:
                warmup_scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()   
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # print('acc1', acc1, 'acc5', acc5)
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0)) 
            # if args.settings:    
            #     for name, param in net.named_parameters():
            #         if 'Learn' in name or 'Fix' in name:
            #             print(name, param.data.mean())                    
    # loss /= cnt
    # acc *= 100. / cnt
    # print(f"Epoch: {epoch}, Train accuracy: {acc:6.2f} %, Train loss: {loss:8.5f}")        
    # experiment.log_metric("Train/Average loss", loss, step=epoch)
    # experiment.log_metric("Train/Accuracy-top1", acc, step=epoch)
    # experiment.log_metric("Train/Time", batch_time.sum, step=epoch)
    # plt.legend()
    # plt.show() 
    print("Epoch:", epoch,'Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    print('diff_loss',diff_loss)
    print('training_stage',training_stage)
    # print('crt_ord',crt_ord)
    # print('loss_A',loss_A.mean())
    
    if 'LearnCoe' in args.settings:
        coes_As = torch.tensor([]).to(device)
        coes_Bs = torch.tensor([]).to(device)
        for name, p in net.named_parameters():
            if p.requires_grad: 
                if name:
                    if 'coesA' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_A = sum_A+p
                        # coes_As = coes_As+[p]
                        coes_As = torch.cat((coes_As, p ),0)
                        print(name,p.data)
                if 'coesB' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_B = sum_B+p
                        # coes_Bs = coes_Bs+[p]
                        coes_Bs = torch.cat((coes_Bs, p ),0)  
                        print(name,p.data)
                        
        err = coe_constrains(coes_As, coes_Bs,req_ord=args.req_ord,prin=True)
    # print('err',err )                       
    # return acc, loss, batch_time.sum        
    
        # print('hook removed')
    # print('Epoch: {:.1f}, Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, losses.avg, top1.avg))
    # writer.add_scalar('Train/Average loss', losses.avg, epoch)
    # writer.add_scalar('Train/Accuracy-top1', top1.avg, epoch)
    # writer.add_scalar('Train/Accuracy-top5', top5.avg, epoch)
    # writer.add_scalar('Train/Time', batch_time.sum, epoch)

    experiment.log_metric("Train/Average loss", losses.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top1", top1.avg, step=epoch)
    experiment.log_metric("Train/Accuracy-top5", top5.avg, step=epoch)
    experiment.log_metric("Train/Time", batch_time.sum, step=epoch)

    coes_print = []

    return top1.avg, losses.avg, batch_time.sum

def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # losses_fgm = AverageMeter('Loss_fgm', ':.4e')
    # top1_fgm = AverageMeter('fgmAcc@1', ':6.2f')
    # top5_fgm = AverageMeter('fgmAcc@5', ':6.2f')
    
    losses_pgd = AverageMeter('pgdLoss_pgd', ':.4e')
    top1_pgd = AverageMeter('pgdAcc@1', ':6.2f')
    top5_pgd = AverageMeter('pgdAcc@5', ':6.2f')        
    end = time.time()
    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs_fgm = fast_gradient_method(net, inputs, eps=args.eps, norm=args.norm)
            # inputs_pgd = projected_gradient_descent(net, inputs, eps=args.eps, eps_iter=args.eps_iter, nb_iter=args.nb_iter, norm=args.norm)            
            # if 'zeros' in args.arch or 'ZeroS' in args.arch:
            #     outputs, coes, stepsize = net(inputs)
            #     outputs_fgm, coes, stepsize = net(inputs_fgm)
            #     outputs_pgd, coes, stepsize = net(inputs_pgd)
            # elif 'MResNet' in args.arch:
            #     outputs, coes = net(inputs)
            # else:
            #     outputs = net(inputs)
            outputs = net(inputs)
            # outputs_fgm = net(inputs_fgm)
            # outputs_pgd = net(inputs_pgd)                
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))  
                            
            # loss_fgm = criterion(outputs_fgm, targets)
            # acc1_fgm, acc5_fgm = accuracy(outputs_fgm, targets, topk=(1, 5))
            # losses_fgm.update(loss_fgm.item(), inputs_fgm.size(0))
            # top1_fgm.update(acc1_fgm[0], inputs_fgm.size(0))
            # top5_fgm.update(acc5_fgm[0], inputs_fgm.size(0))
            
            # loss_pgd = criterion(outputs_pgd, targets)
            # acc1_pgd, acc5_pgd = accuracy(outputs_pgd, targets, topk=(1, 5))
            # losses_pgd.update(loss_pgd.item(), inputs_pgd.size(0))
            # top1_pgd.update(acc1_pgd[0], inputs_pgd.size(0))
            # top5_pgd.update(acc5_pgd[0], inputs_pgd.size(0))
                                 
            batch_time.update(time.time() - end)
            end = time.time()
        # loss /= cnt
        # acc *= 100. / cnt
    # print(f"Epoch: {epoch}, Test accuracy:  {acc:6.2f} %, Test loss:  {loss:8.5f}")

    # return acc, loss, batch_time.sum
    #         loss = criterion(outputs, targets)
    #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    #         losses.update(loss.item(), inputs.size(0))
    #         top1.update(acc1[0], inputs.size(0))
    #         top5.update(acc5[0], inputs.size(0))
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    
    
    
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    # print('Test set: Average loss_fgm: {:.4f}, Accuracy_fgm: {:.4f}'.format(losses_fgm.avg, top1_fgm.avg))
    print('Test set: Average loss_pgd: {:.4f}, Accuracy_pgd: {:.4f}'.format(losses_pgd.avg, top1_pgd.avg))

    # # writer.add_scalar('Test/Average loss', losses.avg, epoch)
    # # writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    # # writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    # # writer.add_scalar('Test/Time', batch_time.sum, epoch)

    experiment.log_metric("Test/Average loss", losses.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top1", top1.avg, step=epoch)
    experiment.log_metric("Test/Accuracy-top5", top5.avg, step=epoch)

    # experiment.log_metric("Test/Average loss_fgm", losses_fgm.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top1_fgm", top1_fgm.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top5_fgm", top5_fgm.avg, step=epoch)

    # experiment.log_metric("Test/Average loss_pgd", losses_pgd.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top1_pgd", top1_pgd.avg, step=epoch)
    # experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=epoch)
    
    experiment.log_metric("Test/Time", batch_time.sum, step=epoch)
        
    return top1.avg, losses.avg, batch_time.sum

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         if 'zerosnet' in args.arch or 'ZeroSNet' in args.arch:
    #             outputs, coes, stepsize = net(inputs)
    #         elif 'MResNet' in args.arch:
    #             outputs, coes = net(inputs)
    #         else:
    #             outputs = net(inputs)
    #         loss = criterion(outputs, targets)
    #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
    #         losses.update(loss.item(), inputs.size(0))
    #         top1.update(acc1[0], inputs.size(0))
    #         top5.update(acc5[0], inputs.size(0))
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    # print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))

    # writer.add_scalar('Test/Average loss', losses.avg, epoch)
    # writer.add_scalar('Test/Accuracy-top1', top1.avg, epoch)
    # writer.add_scalar('Test/Accuracy-top5', top5.avg, epoch)
    # writer.add_scalar('Test/Time', batch_time.sum, epoch)

    # return top1.avg, losses.avg, batch_time.sum

def eval_adv(net, loader, eps, eps_iter, nb_iter, norm, clip_min, clip_max,):
    batch_time = AverageMeter('Time', ':6.3f')
    
    losses_fgm = AverageMeter('Loss_fgm', ':.4e')
    top1_fgm = AverageMeter('fgmAcc@1', ':6.2f')
    top5_fgm = AverageMeter('fgmAcc@5', ':6.2f')
    
    losses_pgd = AverageMeter('pgdLoss_pgd', ':.4e')
    top1_pgd = AverageMeter('pgdAcc@1', ':6.2f')
    top5_pgd = AverageMeter('pgdAcc@5', ':6.2f')        
    end = time.time()
    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # if args.dataset == 'cifar10':
            #     inputs = UnNormalize(
            #         inputs, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # elif args.dataset == 'cifar100':
            #     inputs = UnNormalize(
            #         inputs, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            # elif args.dataset == 'mnist':
            #         inputs = UnNormalize(
            #         inputs,  (0.1307,), (0.3081,))
                            
            # inputs_fgm = fast_gradient_method(net, inputs, eps=eps, norm=norm,clip_min=clip_min, clip_max=clip_max)
            inputs_pgd = projected_gradient_descent(net, inputs, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm,clip_min=clip_min, clip_max=clip_max)  
            # inputs_fgm = F.relu(F.relu(inputs_fgm.mul_(-1).add_(1)).mul_(-1).add_(1))      
            # inputs_pgd = F.relu(
            #     F.relu(inputs_pgd.mul_(-1).add_(1)).mul_(-1).add_(1))     
                             
            # if args.dataset == 'cifar10':
            #     inputs_fgm = Normalize(
            #         inputs_fgm, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #     inputs_pgd = Normalize(
            #         inputs_pgd, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))                
            # elif args.dataset == 'cifar100':
            #     inputs_fgm = Normalize(
            #         inputs_fgm, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))         
            #     inputs_pgd = Normalize(
            #         inputs_pgd, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))    
            # elif args.dataset == 'mnist':
            #         inputs_fgm = Normalize(
            #         inputs_fgm,  (0.1307,), (0.3081,))     
            #         inputs_pgd = Normalize(
            #         inputs_pgd,  (0.1307,), (0.3081,))          
            
            # outputs_fgm = net(inputs_fgm)
            outputs_pgd = net(inputs_pgd)        
                    
            # loss = criterion(outputs, targets)
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            # losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1[0], inputs.size(0))
            # top5.update(acc5[0], inputs.size(0))  
                            
            # loss_fgm = criterion(outputs_fgm, targets)
            # acc1_fgm, acc5_fgm = accuracy(outputs_fgm, targets, topk=(1, 5))
            # losses_fgm.update(loss_fgm.item(), inputs_fgm.size(0))
            # top1_fgm.update(acc1_fgm[0], inputs_fgm.size(0))
            # top5_fgm.update(acc5_fgm[0], inputs_fgm.size(0))
            
            loss_pgd = criterion(outputs_pgd, targets)
            acc1_pgd, acc5_pgd = accuracy(outputs_pgd, targets, topk=(1, 5))
            losses_pgd.update(loss_pgd.item(), inputs_pgd.size(0))
            top1_pgd.update(acc1_pgd[0], inputs_pgd.size(0))
            top5_pgd.update(acc5_pgd[0], inputs_pgd.size(0))
                                 
            batch_time.update(time.time() - end)
            end = time.time()

    # print('eps, eps_iter, nb_iter, norm', eps, eps_iter, nb_iter, norm)
    # print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
    # print('Test set: Average loss_fgm: {:.4f}, Accuracy_fgm: {:.4f}'.format(losses_fgm.avg, top1_fgm.avg))
    # print('Test set: Average loss_pgd: {:.4f}, Accuracy_pgd: {:.4f}'.format(losses_pgd.avg, top1_pgd.avg))

    # experiment.log_metric("Test/Average loss_fgm", losses_fgm.avg)
    # experiment.log_metric("Test/Accuracy-top1_fgm", top1_fgm.avg)
    # # experiment.log_metric("Test/Accuracy-top5_fgm", top5_fgm.avg, step=0)

    # experiment.log_metric("Test/Average loss_pgd", losses_pgd.avg)
    # experiment.log_metric("Test/Accuracy-top1_pgd", top1_pgd.avg)
    # experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=0)
    # experiment.log_metric("eps", eps)
    # experiment.log_metric("nb_iter", nb_iter)

            
    return top1_fgm.avg, top1_pgd.avg, losses_fgm.avg, losses_pgd.avg, batch_time.sum

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
#     filepath = os.path.join(checkpoint, filename)
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        
def save_checkpoint(net, state, is_best,  checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if is_best:
        torch.save(net, os.path.join(checkpoint, 'model_best.pth.tar'))
        torch.save(state, os.path.join(checkpoint, 'model_state_best.pth.tar'))
        
        # shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        print('best checkpoint saved.')

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def noise_rob(net, loader, noise_type, noise_coff):

    # print('==> Start Robust Evaluation')
    net.eval()
    with torch.no_grad():
        losses_pertur = AverageMeter('Loss', ':.4e')
        top1_pertur = AverageMeter('Acc@1', ':6.2f')
        top5_pertur = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs_pertur, targets = inputs.to(device), targets.to(device)

            if args.dataset == 'cifar10':
                # print('==> UnNormalize CIFAR10')
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.1307,), (0.3081,))
            elif args.dataset == 'svhn':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                     
            if noise_type == 'randn':
                # print('noise_coff, inputs_pertur.size()',noise_coff, inputs_pertur.size())
                inputs_pertur = inputs_pertur + noise_coff * \
                    torch.autograd.Variable(torch.randn(inputs_pertur.size()).to(device), requires_grad=False)

            # 均匀分布
            elif noise_type == 'rand':
                inputs_pertur = inputs_pertur + noise_coff * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).to(device), requires_grad=False)
            # 常数
            elif noise_type == 'const':
                inputs_pertur = inputs_pertur + noise_coff 

            # 截断
            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.1307,), (0.3081,))  
            elif args.dataset == 'svhn':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                   
            outputs_pertur = net(inputs_pertur)
            loss_pertur = criterion(outputs_pertur, targets)
            acc1_pertur, acc5_pertur = accuracy(
                outputs_pertur, targets, topk=(1, 5))
            losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))
        # print('top1_pertur.avg', top1_pertur.avg)
        print(noise_type+str(noise_coff)+'loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
            losses_pertur.avg, top1_pertur.avg))
        # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
        # experiment.log_metric("Test_pertur/Acc-top1/"+noise_type, top1_pertur.avg, step=1)
        
        return top1_pertur.avg
    

def log_feature(net, loader, noise_type, noise_coff):

    print('==> Start log_feature')
    net.eval()
    # seed_torch()
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx != 0: 
                break
            print('inputs.shape',inputs.shape)

            inputs_pertur, targets = inputs[0].to(device), targets[0].to(device)
            inputs_pertur = torch.reshape(inputs_pertur,(1,3, 32, 32))
            
            print('inputs_pertur.shape', inputs_pertur.shape)

            # inputs_pertur = inputs_pertur[0]
            # targets = targets[0]
            if args.dataset == 'cifar10':
                # print('==> UnNormalize CIFAR10')
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.1307,), (0.3081,))     
            elif args.dataset == 'svhn':
                    inputs_pertur = UnNormalize(
                    inputs_pertur,  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            # if noise_type == 'randn':
            #     # print('noise_coff, inputs_pertur.size()',noise_coff, inputs_pertur.size())
            #     inputs_pertur = inputs_pertur + noise_coff * \
            #         torch.autograd.Variable(torch.randn(inputs_pertur.size()).to(device), requires_grad=False)

            # # 均匀分布
            # elif noise_type == 'rand':
            #     inputs_pertur = inputs_pertur + noise_coff * torch.autograd.Variable(torch.rand(
            #         inputs_pertur.size()).to(device), requires_grad=False)
            # 常数
            # elif noise_type == 'const':
            inputs_pertur = inputs_pertur + noise_coff 

            # 截断
            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            elif args.dataset == 'cifar100':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            elif args.dataset == 'mnist':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.1307,), (0.3081,)) 
            elif args.dataset == 'svhn':
                    inputs_pertur = Normalize(
                    inputs_pertur,  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                    

            for name, m in net.named_modules():
                # if not isinstance(m, torch.nn.ModuleList) and \
                #         not isinstance(m, torch.nn.Sequential) and \
                #         type(m) in torch.nn.__dict__.values():
                # 这里只对卷积层的feature map进行显示
                if isinstance(m, torch.nn.Conv2d):
                    m.register_forward_pre_hook(print_feature)

            outputs_pertur = net(inputs_pertur)

            # loss_pertur = criterion(outputs_pertur, targets)
            # acc1_pertur, acc5_pertur = accuracy(
            #     outputs_pertur, targets, topk=(1, 5))
            # losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            # top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            # top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))
        # print('top1_pertur.avg', top1_pertur.avg)
        # print(noise_type+str(noise_coff)+'loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
        #     losses_pertur.avg, top1_pertur.avg))
        # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
        # experiment.log_metric("Test_pertur/Acc-top1/"+noise_type, top1_pertur.avg, step=1)
        
        return outputs_pertur

def print_feature(module, input):
    x = input[0][0]
    print('x.mean()', x.mean())
    #最多显示4张图
    # min_num = np.minimum(4, x.size()[0])
    # for i in range(min_num):
    #     plt.subplot(1, 4, i+1)
    #     plt.imshow(x[i].cpu())
    # plt.show()
    
def target_transform(target):
    return int(target) - 1

# class Hook():
#     def __init__(self, module, backward=False):
#         if backward==False:
#             self.hook = module.register_forward_hook(self.hook_fn)
#         else:
#             self.hook = module.register_backward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output):
#         self.input = input
#         self.output = output
#     def close(self):
#         self.hook.remove()
        
        
class Hooks():
    def __init__(self, layer):
        self.model  = None
        self.input  = None
        self.output = None
        self.grad_input  = None
        self.grad_output = None
        self.forward_hook  = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)
    def hook_fn_act(self, module, input, output):
        self.model  = module
        self.input  = input[0]
        self.output = output
    def hook_fn_grad(self, module, grad_input, grad_output):
        self.grad_input  = grad_input[0]
        self.grad_output = grad_output[0]
    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
if __name__ == '__main__':

    # if 1:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        
    try:
        path_base = "./runs/"+"Tuning/CometData"
        os.makedirs(path_base)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path_base):
            pass
        else:
            raise

    args =parser.parse_known_args()[0]
    

    args = prepare(args)
    try:
        path_base = "./runs/"+args.dataset+"/CometData"
        os.makedirs(path_base)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path_base):
            pass
        else:
            raise
    # experiment = Experiment(
    experiment = OfflineExperiment(
        api_key="YourAPIKey",
        # project_name="OverThreeOrders",
        project_name='project_name',#"over3orders-ConvNextCIFAR1028",
        # project_name="overthreeorders-4channels",
        workspace="workspace",
        # auto_histogram_weight_logging=True,
        offline_directory=path_base,
    )  
    # experiment["offline_directory"] = path_base
    hyper_params = vars(args)
    experiment.log_parameters(hyper_params)
    best_acc = 0
    start_epoch = 0
    if not os.path.isdir(args.save_path) and args.local_rank == 0:
        mkdir_p(args.save_path)

    if args.dataset == 'cifar10':
        print('==> Preparing cifar10 data..')

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.adv_test or args.adv_train:
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                        transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])
        # trainset = torchvision.datasets.CIFAR10(
        #     root= args.data +  '/cifar10', train=False, download=True, transform=transform_train)
        
        trainset = torchvision.datasets.CIFAR10(
            root= args.data +  '/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root= args.data + '/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #            'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.dataset == 'cifar100':
        print('==> Preparing cifar100 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=args.data +'/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(
            root=args.data +'/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    #        classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #                   'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == 'mnist':
        trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data +'/mnist', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(args.data +'/mnist',train=False,download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True) 
        
    elif args.dataset == 'svhn':
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=args.data +'/svhn', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                # target_transform=target_transform,
            ),
            batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True) 
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=args.data +'/svhn', split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
                # target_transform=target_transform
            ),
            batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True) 

    print('==> Building model..')
    net_name = args.arch
    model_name = args.arch
    # net = eval(args.arch)()
    if 'convnext' in args.arch: 
        net = create_model(
        args.arch,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        givenA = args.givenA,
        givenB = args.givenB,
        settings=args.settings,
        PL=args.PL, 
        ini_stepsize=args.ini_stepsize,
        ini_block_shift=args.ini_block_shift,
        IniDecay = args.IniDecay,
        )
    else:
        net = eval(args.arch)(num_classes=args.num_classes, givenA=args.givenA, givenB=args.givenB, PL=args.PL, ini_stepsize=args.ini_stepsize, settings=args.settings,ini_block_shift=args.ini_block_shift,
        IniDecay = args.IniDecay,)
    # print('net',net)
    # tw.draw_model(net, [1, 3, 32, 32])
    net = net.to(device)
    # net.apply(weights_init) 


    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    if 'mnist' in args.dataset:
        d = torch.rand(1, 1, 28, 28).to(device)
    elif 'cifar' in args.dataset or args.dataset == "svhn":
        d = torch.rand(1, 3, 32, 32).to(device)
    elif args.dataset == "stl10":
        d = torch.rand(1, 3, 96, 96).to(device)
    # elif args.dataset == "imagenet":
    # d = torch.rand(1, 3, 224, 224).to(device)        
    # print('d.device, net.device',d.device,net.device)
    summary(net, d)
    
    # torch.save(net, 'convnext.pth')
    # netron.start( 'convnext.pth')
    # flops, params = profile(net,  inputs=(d, ))
    trainable_params = 0
    for name, p in net.named_parameters():
        if p.requires_grad: 
            # print(name)
            if name:
                if 'Fix' not in name: # and 'bn3' not in name:
                    trainable_params +=p.numel()
                # else:
                #     print('name', name)
            else:
                trainable_params +=p.numel()
    all_params = 0
    all_trainable_params= 0 
    for p in net.parameters():
        all_params +=p.numel()
    for p in net.parameters():
        if p.requires_grad:         
            all_trainable_params +=p.numel()        
    # print('flops, params, trainable_params,all_params,all_trainable_params', flops, params, trainable_params,all_params,all_trainable_params)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    if 'convnext' in args.arch:
        optimizer_sgd = torch.optim.AdamW([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]}, {'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.}
        ], lr=args.lr, weight_decay=0.05,eps=5e-9)#lr=0.001*args.lr
    else:
        if 'Adam' in args.settings:
            optimizer_sgd = torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]},#, 'lr': 0.}, 
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer_sgd =  torch.optim.SGD([{'params':[ param for name, param in net.named_parameters() if ('Fix' not in name and 'coes' not in name)]}, {'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.}
                                        ],  lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    # if 'Adam' in args.settings:                
    optimizer_adam = torch.optim.Adam([
    {'params': (p for name, p in net.named_parameters() if 'coes' in name), 'weight_decay': 0.,},
    ], lr=args.CoesLR, weight_decay=args.weight_decay)

    if args.minimizer in ['ASAM', 'SAM']:
        minimizer = eval(args.minimizer)(optimizer_sgd, net, rho=args.rho, eta=args.eta)
        if args.sche == 'cos':
            train_scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epoch)
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[150, 225], gamma=0.1)
            elif args.dataset == 'cifar10' or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(minimizer.optimizer, milestones=[80, 120], gamma=0.1)
    else:
        if args.sche == 'cos':
            train_scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, args.epoch)
            train_scheduler_adam = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, args.epoch)            
        elif args.sche == 'step':
            if args.dataset == 'cifar100':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[150, 225], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[150, 225], gamma=0.1)                
            elif args.dataset == 'cifar10':# or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[80, 120], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[80, 120], gamma=0.1)                
            elif args.dataset == 'svhn' or args.dataset == 'mnist':
                train_scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[20, 30], gamma=0.1)
                train_scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[20, 30], gamma=0.1)                
                
    iter_per_epoch = len(trainloader)
    if args.minimizer in ['ASAM', 'SAM']:
        warmup_scheduler = WarmUpLR(minimizer.optimizer, iter_per_epoch * args.warm)
    else:
        warmup_scheduler = WarmUpLR(optimizer_sgd, iter_per_epoch * args.warm)

    
    # criterion = nn.CrossEntropyLoss()
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterion2 = torch.nn.MSELoss()
    # optionally resume from a checkpoint
    title = 'CIFAR-' + args.arch
    args.lastepoch = -1
    if args.resume:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer_sgd.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            args.lastepoch = checkpoint['epoch']

    train_time = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_top1_acc_fgm = 0.0
    test_top1_acc_pgd = 0.0
    
    test_min_loss = 100
    best_prec1 = -1
    # lr_list = []

    # writer = SummaryWriter(log_dir=args.save_path)
    plot_feature(net, 0,args.save_path)
    if args.adv_test:
        test_top1_fgm, test_top1_pgd, losses_fgm, losses_pgd, batch_time = eval_adv(net, testloader, args.eps, args.eps_iter, args.nb_iter, args.norm,clip_min=args.clip_min, clip_max=args.clip_max)
        print('test_top1_pgd',float(test_top1_pgd.data))
    # features_in = []
    # features_out = []
    # def hook(module, input, output):
    #     features_in.append(input)
    #     features_out.append(output)
    #     return None
    # # for (name, module) in net.named_modules():
    # for module in net.modules():
    #     # print('name,module',name,module)
    #     # print('name',name)
    #     # if name is not None:
    #     # if isinstance(module, torch.nn.ReLU):#isinstance(module, torch.nn.Identity):        
    #     handle = module.register_forward_hook(hook)
    #     handle.remove()
    if 'LearnConti' in args.settings:
        hookF_back = [Hooks(layer[1]) for layer in list(net.module._modules.items())]     
    if 'GFNet' in args.mode:
        # hookF_GF = [Hooks(layer[1]) for layer in list(net._modules.items())]
        hookF_GF = []
        for layer in list(net.module.blocks._modules.items()):
        # for layer in net.modules():
            # print('layer[0]',layer[0])
            # print('layer[1]',layer[1])
            hookF_GF += [Hooks(layer[1])]    
    
    # for epoch in range(1, args.epoch):
    for epoch in range(1, args.epoch+1):
        coes_As = []
        coes_Bs = []
        coesDecay = []
        coesBalance = []
        bias_outer = []
        bias_inner = []
        for name, p in net.named_parameters():
            if 1:#p.requires_grad: 
                if name:
                    if 'coesA' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_A = sum_A+p
                        # coes_As = coes_As+[p]
                        coes_As = coes_As+[float(p.mean().data)]
                    elif 'coesB' in name and not 'coesBalance' in name: # and 'bn3' not in name:
                        # print(name)
                        # sum_B = sum_B+p
                        coes_Bs = coes_Bs+[float(p.mean().data)]
                        # print('coes_Bs',float(p.mean().data),p.size())
                        # coes_Bs = torch.cat((coes_Bs.mean().reshape(1,1), p.mean().reshape(1,1) ),0) 
                    elif 'coesDecay' in name:
                        print('coesDecay.grad', p.grad)
                        
                        if 'RLExp' in args.settings:
                            D = torch.relu(p)
                        elif 'GLExp' in args.settings:
                            D = torch.nn.functional.gelu(p)             
                        elif 'AbsExp' in args.settings:
                            D = torch.abs(p)
                        elif 'PowExp' in args.settings:
                            D = (p)**2                                
                        elif 'SigExp' in args.settings:
                            D = torch.sigmoid(p)
                        else:
                            D = p     
                        coesDecay = coesDecay+[float(D.mean().data)]
                        # print('coesDecay',p)
                    elif 'coesBalance' in name:
                        coesBalance = coesBalance +[float(p.mean().data)]
        # print('coes_Bs', coes_Bs)#.data)
        print('coesDecay', coesDecay)
        for i in range(0,len(coesDecay)):
            experiment.log_metric('coesDecay'+str(i), coesDecay[i], step=epoch)
        for i in range(0,len(bias_outer)):
            experiment.log_metric('bias_outer'+str(i), bias_outer[i], step=epoch)
        for i in range(0,len(bias_inner)):
            experiment.log_metric('bias_inner'+str(i), bias_inner[i], step=epoch)                    
        # print('coesBalance', coesBalance)
        # experiment.log_metric('coesDecay', coesDecay, step=epoch)
        # experiment.log_histogram_3d(coes_Bs, name='coes_Bs', step=epoch)
        
        # print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_acc_epoch, train_loss_epoch, train_epoch_time = train(epoch)
        

        
        if epoch%2==1:
            
            plot_feature(net, epoch,args.save_path)
            # net.eval()
            # x_test, y_test = load_cifar10(n_examples=50)
            # fmodel = fb.PyTorchModel(net, bounds=(0, 1))
            # _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to(device), y_test.to(device), epsilons=[8/255])
            # print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))            
        # print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
        train_top1_acc = max(train_top1_acc, train_acc_epoch)
        train_min_loss = min(train_min_loss, train_loss_epoch)
        train_time += train_epoch_time
        acc, test_loss_epoch, test_epoch_time = test(epoch)
        top1_pertur_test = noise_rob(net, testloader, 'randn', 0.05)
        nni.report_intermediate_result(top1_pertur_test)
        print('top1_pertur_test',top1_pertur_test)
        test_top1_acc = max(test_top1_acc, acc)
        if args.adv_test:
            test_top1_fgm, test_top1_pgd, losses_fgm, losses_pgd, batch_time = eval_adv(net, testloader, args.eps, args.eps_iter, args.nb_iter, args.norm,clip_min=args.clip_min, clip_max=args.clip_max)
            print(' test_top1_pgd', float(test_top1_pgd.data))
        # test_top1_acc_pgd = max(test_top1_acc_pgd, acc_pgd)
        # test_top1_acc_fgm = max(test_top1_acc_fgm, acc_fgm)
        print('optimizer_sgd_lr',optimizer_sgd.state_dict()['param_groups'][0]['lr'])
        print('optimizer_adam_lr',optimizer_adam.state_dict()['param_groups'][0]['lr'])
        test_min_loss = min(test_min_loss, test_loss_epoch)
        test_time += test_epoch_time

        if args.local_rank == 0:

            is_best = test_top1_acc > best_prec1
            best_prec1 = max(test_top1_acc, best_prec1)
            
            # if epoch > args.epoch*0.8:
            #     save_checkpoint(net, {
            #         'epoch': epoch + 1,
            #         'arch': args.arch,
            #         'state_dict': net.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best, checkpoint=args.save_path)

    if 'GFNet' in args.mode:       
        for hook in hookF_GF:
            hook.remove()
    if 'LearnConti' in args.settings:
        for hook in hookF_back:
            hook.remove()

    # writer.close()
    end_train = train_time // 60
    end_test = test_time // 60
    experiment.log_metric("test_top1_best", test_top1_acc)
    experiment.log_metric("top1_pertur_test", top1_pertur_test)

    # if dataset == 'cifar10':
    #     top1_pertur_test_c10 = top1_pertur_test
    #     test_top1_acc_c10 = test_top1_acc
    # elif dataset == 'cifar100':
    #     top1_pertur_test_c100 = top1_pertur_test
    #     test_top1_acc_c100 = test_top1_acc
    # elif dataset == 'svhn':
    #     top1_pertur_test_svhn = top1_pertur_test
    #     test_top1_acc_svhn = test_top1_acc                
    # experiment.log_metric("SumTest"+dataset, top1_pertur_test+test_top1_acc)
    
    # nni.report_final_result(best_prec1)
    # experiment.los_mkjsdu()
    
    print(model_name)
    print("train time: {}D {}H {}M".format(end_train // 1440, (end_train % 1440) // 60, end_train % 60))
    print("test time: {}D {}H {}M".format(end_test // 1440, (end_test % 1440) // 60, end_test % 60))
    print(
        "train_acc_top1:{}, train_min_loss:{}, train_time:{}, test_top1_acc:{}, test_min_loss:{}, test_time:{}".format(
            train_top1_acc, train_min_loss, train_time, test_top1_acc, test_min_loss, test_time))
    print("args.save_path:", args.save_path)
    
    
    givenA_txt = ''
    givenB_txt = ''
    givenA_list = []
    givenB_list = []
    for i in args.givenA:
        givenA_txt += str(i)+'_'
        givenA_list += str(i)
    print('args.givenA', args.givenA)    
    print('givenA_txt', givenA_txt)
    print('givenA_list', givenA_list)

    for i in args.givenB:
        givenB_txt += str(i)+'_'        
        givenB_list += str(i)
    ConverOrder = args.ConverOrd
    if args.adv_train:
        head_list = ['Model',  "Step", "Order", "Alphas", "Betas", 'Noise Type', 'Noise Value', 'Adv. Train Acc.', 'Adv. Test Acc.']#"FLOPs","# Params", "# Trainable Params", 
    else:
        head_list = ['Model',"Step", "Order", "Alphas", "Betas",'Noise Type', 'Noise Value', 'Train Acc.', 'Test Acc.']#"FLOPs","# Params", "# Trainable Params", 
    df = pd.DataFrame(columns=head_list)
    
    # net = torch.load(os.path.join(args.save_path, 'model_state_best.pth.tar')) 
    # seed_torch()
    noise_dict = {'const': [0,0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.45, -0.01, -0.05,-0.1,-0.2, -0.3, -0.4, -0.45],
            'randn': [0.01,  0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
            'rand': [0.04, 0.06, 0.08, 0.1, 0.12, 0.2, -0.04, -0.06, -0.08, -0.1, -0.12,-0.2],
            }
    # exps = ["0", "1", "2"]
    eps_list = [3.0/255, 5.0/255, 8.0/255] #  16.0/255, 32.0/255]
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    norm = args.norm        
    

    
    # for noise_type in noise_dict.keys():  # 噪音类型

    
    for noise_type in noise_dict.keys():  # 噪音类型
        Noise_Coffs = []
        Top1_pertur_tests = []
        Top1_pertur_trains = []
        for noise_coff in noise_dict.get(noise_type):  # 噪音值
            # if noise_type == 'const':
            #     loss_per = log_feature(net, testloader, noise_type, noise_coff)
            #     print('loss_per',loss_per) 
            # if noise_coff ==0 and noise_type =='const':
            #         for eps in eps_list:  # 噪音值
            #             print('eps, eps_iter, nb_iter, norm', eps, eps_iter, nb_iter, norm)

            #             test_top1_fgm, test_top1_pgd, losses_fgm, losses_pgd, batch_time = eval_adv(net, testloader, eps, eps_iter, nb_iter, norm)
                        
            #             # train_top1_fgm, train_top1_pgd, train_losses_fgm, train_losses_pgd, train_batch_time = eval_adv(net, trainloader, eps, eps_iter, nb_iter, norm)
            #             print('test_top1_fgm, test_top1_pgd',test_top1_fgm.data, test_top1_pgd.data)
            #             # print('train_top1_fgm, train_top1_pgd',train_top1_fgm, train_top1_pgd)    
            #             df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, 0, test_top1_pgd.item() ]], columns=head_list)
            #             df = df.append(df_row)                        
            #             df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, 0, test_top1_fgm.item() ]], columns=head_list)                                             
            #             # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, train_top1_pgd.item(), test_top1_pgd.item() ]], columns=head_list)
            #             # df = df.append(df_row)                        
            #             # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, train_top1_fgm.item(), test_top1_fgm.item() ]], columns=head_list)
            #             df = df.append(df_row)
            top1_pertur_test = noise_rob(net, testloader, noise_type, noise_coff)
            # top1_pertur_train = noise_rob(net, trainloader, noise_type, noise_coff)
            Noise_Coffs += [noise_coff]
            Top1_pertur_tests += [top1_pertur_test.item()]
            # Top1_pertur_trains += [0]
            # Top1_pertur_trains += [top1_pertur_train.item()]
            df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, 0, top1_pertur_test.item() ]], columns=head_list)

            # df_row = pd.DataFrame([[args.arch, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, top1_pertur_train.item(), top1_pertur_test.item() ]], columns=head_list)
            df = df.append(df_row)
        # print(noise_type+'Noise_Coffs, Top1_pertur_trains',Noise_Coffs, Top1_pertur_trains)
        # print(noise_type+'Noise_Coffs, Top1_pertur_tests',Noise_Coffs, Top1_pertur_tests)

        # experiment.log_curve('Train'+noise_type, x=Noise_Coffs, y=Top1_pertur_trains)
        # experiment.log_curve('Test'+noise_type, x=Noise_Coffs, y=Top1_pertur_tests)
        
    print('Table \n',df)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = args.arch+args.notes+'.csv'
    df.to_csv(save_path+file_name)
    experiment.log_table(save_path+file_name)    

experiment.end()

