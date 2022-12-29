#!/usr/bin/env python3
from __future__ import absolute_import

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
from comet_ml import Experiment, OfflineExperiment

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import sys,os
sys.path.append(os.getcwd())

from timm.data import (AugMixDataset, FastCollateMixup, Mixup, create_dataset,
                       create_loader, resolve_data_config)
from timm.loss import *
#from timm.models import (convert_splitbn_model, create_model, load_checkpoint,
#                        model_parameters, resume_checkpoint, safe_model_name)
from timm.models_v2 import (convert_splitbn_model, create_model, load_checkpoint,
                         model_parameters, resume_checkpoint, safe_model_name)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# import sys,os
# sys.path.append(os.getcwd())

# from .timm.data import (AugMixDataset, FastCollateMixup, Mixup, create_dataset,
#                        create_loader, resolve_data_config)
# from .timm.loss import *
# from .timm.models import (convert_splitbn_model, create_model, load_checkpoint,
#                          model_parameters, resume_checkpoint, safe_model_name)
# from .timm.optim import create_optimizer_v2, optimizer_kwargs
# from .timm.scheduler import create_scheduler
# from .timm.utils import *
# from .timm.utils import ApexScaler, NativeScaler
# from .timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False
    


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR', default='/media3/datasets/imagenet/',
                    help='path to dataset')#/home/bdc/datasets/ImageNet
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=int, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

parser.add_argument('--ConverOrd', type=int, default=3)
parser.add_argument('--ini_stepsize', default=1, type=float)
parser.add_argument('--givenA', default=None, nargs='+', type=float)
parser.add_argument('--givenB', default=None, nargs='+', type=float)
parser.add_argument("--notes", default='', type=str)
parser.add_argument("--rob_eval", action='store_true')

parser.add_argument("--adv_train", action='store_true')
parser.add_argument("--eps", default=0.1, type=float, help="")
parser.add_argument("--eps_iter", default=0.01, type=float, help="step size for each attack iteration")
parser.add_argument("--nb_iter", default=40, type=int, help="Number of attack iterations.")
parser.add_argument("--norm", default=np.inf, type=float, help="Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.")

parser.add_argument("--ini_block_shift", default=None, type=int, help="")
parser.add_argument("--settings", default='cifa10_BnReluConvConvAllEle', type=str, help="settings")
parser.add_argument("--IniDecay", default=0.07, type=float, help="settings")
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Fix') != -1:
        print("Found Fix", classname)
        nn.init.dirac_(m.downsample_[1].weight.data, 2)
        
def main():
    setup_default_logging()
    args, args_text = _parse_args()
    
    hyper_params = {
    'dist-bn': args.dist_bn,
    'split-bn': args.split_bn,
    'resplit': args.resplit,
    'aug-splits': args.aug_splits,
    'remode': args.remode,        
    'pin-mem':args.pin_mem,
    'native-amp':args.native_amp,
    'apex-amp':args.apex_amp,
    'amp':args.amp,
    'drop': args.drop,
    'smoothing': args.smoothing,
    'mixup': args.mixup,   
    'reprob': args.reprob,
    'bce-loss': args.bce_loss,
    'jsd-loss': args.jsd_loss,
    'aa': args.aa,
    'cooldown-epochs': args.cooldown_epochs,
    'warmup-epochs': args.warmup_epochs,
    'epochs': args.epochs,
    'min-lr': args.min_lr,
    'warmup-lr': args.warmup_lr,
    'lr-k-decay': args.lr_k_decay,
    'lr-cycle-limit': args.lr_cycle_limit,
    'lr-cycle-decay': args.lr_cycle_decay,
    'lr-cycle-mul': args.lr_cycle_mul,
    'sched': args.sched,
    'weight-decay': args.weight_decay,
    'momentum': args.momentum,
    'opt': args.opt,
    'batch-size': args.batch_size,
    'input-size': args.input_size,
    'img-size': args.img_size,
    'resume': args.resume,
    'drop_path_rate':args.drop_path,
    
    # 'PL': args.PL,
    # 'coe_ini': args.coe_ini,
    # 'share_coe': args.share_coe,
    'givenA': args.givenA,
    'givenB': args.givenB,
    'ConverOrd': args.ConverOrd,
    'notes': args.notes,
    'settings': args.settings,
    'IniDecay': args.IniDecay,
    }
    if not os.path.exists("./output/CometData"):
        os.makedirs("./output/CometData")
    # experiment = Experiment(    
    experiment = OfflineExperiment(
        api_key="Your_API_Key",
        project_name="imagenet",
        workspace="Workspace",
        # auto_histogram_weight_logging=True,
        offline_directory="./output/CometData",
    )
    experiment.log_parameters(hyper_params)
    args.ConverOrd = int(args.notes.split('ConOrd')[1].split('PreAct')[0])
    args.steps = len(args.givenA)
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
             
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    model = create_model(
        args.model,
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
        settings = args.settings,
        IniDecay = args.IniDecay,
        )
    # for i in model.named_parameters():
    #     if 1: # "norm" in i[0] and "weight" in i[0]:
    #         print(i[0], i[1].shape)
    # model.apply(weights_init) 

        
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    # optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb, find_unused_parameters=True)
            # model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    try:
        if args.rob_eval:
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
            eval_metrics = val_rob(model, loader_eval, validate_loss_fn, args, output_dir=output_dir, amp_autocast=amp_autocast)
            print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
            
        else:
            # eval_metrics = val_rob(model, loader_eval, validate_loss_fn, args, output_dir=output_dir, amp_autocast=amp_autocast)
            
            for epoch in range(start_epoch, num_epochs):
                coes_As = []
                coes_Bs = []
                coesDecay = []
                coesBalance = []
                bias_outer = []
                bias_inner = []
                for name, p in model.named_parameters():
                    if 1:#p.requires_grad: 
                        if name:
                            if 'coesA' in name: # and 'bn3' not in name:
                                # print(name)
                                # sum_A = sum_A+p
                                # coes_As = coes_As+[p]
                                coes_As = coes_Bs+[float(p.mean().data)]
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
                                elif 'SigExp' in args.settings:
                                    D = torch.sigmoid(p)
                                else:
                                    D = p     
                                coesDecay = coesDecay+[float(D.mean().data)]
                                # print('coesDecay',p)
                            elif 'coesBalance' in name:
                                coesBalance = coesBalance +[float(p.mean().data)]
                            elif 'bias_outer' in name:
                                bias_outer = bias_outer + [float(p.mean().data)]
                            elif 'bias_inner' in name:
                                bias_inner = bias_inner +[float(p.mean().data)]                   
                # print('coes_Bs', coes_Bs)#.data)
                print('coesDecay', coesDecay)
                
                # print('coesBalance', coesBalance)
                for i in range(0,len(coesDecay)):
                    experiment.log_metric('coesDecay'+str(i), coesDecay[i], step=epoch)
                for i in range(0,len(bias_outer)):
                    experiment.log_metric('bias_outer'+str(i), bias_outer[i], step=epoch)
                for i in range(0,len(bias_inner)):
                    experiment.log_metric('bias_inner'+str(i), bias_inner[i], step=epoch)      
                                
                print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
                if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                    loader_train.sampler.set_epoch(epoch)

                train_metrics = train_one_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
                if epoch==1: 
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            print('no gard:', name)
                
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
                print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))

                eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
                
                eval_metrics_pertur = noise_rob(model, loader_eval, validate_loss_fn, args, 'randn', 0.2, amp_autocast=amp_autocast)
                
                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    ema_eval_metrics = validate(
                        model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                    eval_metrics = ema_eval_metrics

                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

                if output_dir is not None:
                    update_summary(
                        epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                        write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                print('train_metrics', train_metrics)
                if args.rank == 0:
                    print('eval_metrics', eval_metrics)
                    for k, v in train_metrics.items():
                        name = 'train_' + k
                        experiment.log_metric(name, v, step=epoch)
                    for k, v in eval_metrics.items():
                        name = 'val_' + k
                        print('name, v', name, v)
                        experiment.log_metric(name, v, step=epoch)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    print('best_metric, best_epoch', best_metric, best_epoch)    
        # for k, v in best_metric.items():
        #     name = 'best_' + k
        #     experiment.log_metric(name, v, step=epoch)
    eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

    print('start time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
    eval_metrics = val_rob(model, loader_eval, validate_loss_fn, args, output_dir=output_dir, amp_autocast=amp_autocast)
    print('end time: ', "{0:%Y-%m-%dT%H-%M/}".format(datetime.now()))
            
def noise_rob(model, loader, loss_fn, args, noise_type, noise_coff, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()    
    
    print('==> Start Robust Evaluation')
    end = time.time()    
    model.eval()
    last_idx = len(loader) - 1    
    with torch.no_grad():

        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                # if args.rank == 0:
                #     print('Before UnNormalize, input.max(),input.min()', input.max(),input.min())
                input = UnNormalize(
                    input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) 
                # if args.rank == 0:    
                #     print('After UnNormalize, input.max(),input.min()', input.max(),input.min())
                                    
                if noise_type == 'randn':
                    # print('noise_coff, input.size()',noise_coff, input.size())
                    input = input + noise_coff * \
                        torch.autograd.Variable(torch.randn(input.size()).cuda(), requires_grad=False)

                # 均匀分布
                elif noise_type == 'rand':
                    input = input + noise_coff * torch.autograd.Variable(torch.rand(
                        input.size()).cuda(), requires_grad=False)
                # 常数
                elif noise_type == 'const':
                    input = input + noise_coff 

                # # 截断
                # input = F.relu(
                #     F.relu(input.mul_(-1).add_(1)).mul_(-1).add_(1))
                input = torch.clamp(input, 0, 1)
                input = Normalize(
                    input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)   
                # if args.rank == 0:   
                #     print('After Normalize, input.max(),input.min()', input.max(),input.min())
            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]         
                # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]     
                                  
            loss_pertur = loss_fn(output, target)

            # loss_pertur = criterion(output, target)
            
            acc1_pertur, acc5_pertur = accuracy(
                output, target, topk=(1, 5))
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss_pertur.data, args.world_size)
                acc1 = reduce_tensor(acc1_pertur, args.world_size)
                acc5 = reduce_tensor(acc5_pertur, args.world_size)
            else:
                reduced_loss = loss_pertur.data

            torch.cuda.synchronize()
                            
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
                                            
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test_rob'+noise_type + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
            # end for
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics
            
        # print(noise_type+str(noise_coff)+'loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
        #     losses_m.avg, top1_m.avg, top5_m.avg))
        # # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
        # # experiment.log_metric("Test_pertur/Acc-top1/"+noise_type, top1_pertur.avg, step=1)
        
        # return top1_m.avg, top5_m.avg
    
def eval_adv(net, loader, loss_fn, args, eps, eps_iter, nb_iter, norm, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()#'Time', ':6.3f')
    
    losses_m_fgm = AverageMeter()#'Loss_fgm', ':.4e')
    top1_m_fgm = AverageMeter()#'fgmAcc@1', ':6.2f')
    top5_m_fgm = AverageMeter()#'fgmAcc@5', ':6.2f')
    
    losses_m_pgd = AverageMeter()#'pgdLoss_pgd', ':.4e')
    top1_m_pgd = AverageMeter()#'pgdAcc@1', ':6.2f')
    top5_m_pgd = AverageMeter()#'pgdAcc@5', ':6.2f')        
    end = time.time()
    last_idx = len(loader) - 1

    net.eval()
    # loss = 0.
    # acc = 0.
    # cnt = 0.
    # with torch.no_grad():
    if 1:
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()                
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)            

            input = UnNormalize(
                input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        
                            
            input_fgm = fast_gradient_method(net, input, eps=eps, norm=norm)
            input_pgd = projected_gradient_descent(net, input, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm)   
            # input_fgm = F.relu(
            #     F.relu(input_fgm.mul_(-1).add_(1)).mul_(-1).add_(1))      
            # input_pgd = F.relu(
            #     F.relu(input_pgd.mul_(-1).add_(1)).mul_(-1).add_(1))     
            input_fgm = torch.clamp(input_fgm, 0, 1)
            input_pgd = torch.clamp(input_pgd, 0, 1)


            input_fgm = Normalize(
                input_fgm, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            input_pgd = Normalize(
                input_pgd, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)                
                
            # if 'zeros' in args.model or 'ZeroS' in args.model:
            #     outputs, coes, stepsize = net(input)
            #     outputs_fgm, coes, stepsize = net(input_fgm)
            #     outputs_pgd, coes, stepsize = net(input_pgd)
            # elif 'MResNet' in args.model:
            #     outputs, coes = net(input)
            # else:
            #     outputs = net(input)
            
            # outputs = net(input)
            with amp_autocast():
                output_fgm = net(input_fgm)
                output_pgd = net(input_pgd)        
            if isinstance(output_fgm, (tuple, list)):
                output_fgm = output_fgm[0]                    
            if isinstance(output_pgd, (tuple, list)):
                output_pgd = output_pgd[0]    
                # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output_fgm = output_fgm.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                output_pgd = output_pgd.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]                               
            # loss = criterion(outputs, target)
            # acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            # losses.update(loss.item(), input.size(0))
            # top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))  
                            
            loss_fgm = loss_fn(output_fgm, target)
            # loss_fgm = criterion(outputs_fgm, target)
            acc1_fgm, acc5_fgm = accuracy(output_fgm, target, topk=(1, 5))
      
                     
            # losses_fgm.update(loss_fgm.item(), input_fgm.size(0))
            # top1_fgm.update(acc1_fgm[0], input_fgm.size(0))
            # top5_fgm.update(acc5_fgm[0], input_fgm.size(0))
            
            loss_pgd = loss_fn(output_pgd, target)
            
            # # loss_pgd = criterion(outputs_pgd, target)
            acc1_pgd, acc5_pgd = accuracy(output_pgd, target, topk=(1, 5))
            # losses_pgd.update(loss_pgd.item(), input_pgd.size(0))
            # top1_pgd.update(acc1_pgd[0], input_pgd.size(0))
            # top5_pgd.update(acc5_pgd[0], input_pgd.size(0))
                                
            if args.distributed:
                reduced_loss_fgm = reduce_tensor(loss_fgm.data, args.world_size)
                acc1_fgm = reduce_tensor(acc1_fgm, args.world_size)
                acc5_fgm = reduce_tensor(acc5_fgm, args.world_size)
                reduced_loss_pgd = reduce_tensor(loss_pgd.data, args.world_size)
                acc1_pgd = reduce_tensor(acc1_pgd, args.world_size)
                acc5_pgd = reduce_tensor(acc5_pgd, args.world_size)                
            else:
                reduced_loss_fgm = loss_fgm.data 
                reduced_loss_pgd = loss_pgd.data 
                                                
            torch.cuda.synchronize()
                                                
            batch_time_m.update(time.time() - end)
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
    # # experiment.log_metric("Test/Accuracy-top5_pgd", top5_pgd.avg, step=0)
    # experiment.log_metric("eps", eps)
    # experiment.log_metric("nb_iter", nb_iter)

            losses_m_fgm.update(reduced_loss_fgm.item(), input_fgm.size(0))
            top1_m_fgm.update(acc1_fgm.item(), output_fgm.size(0))
            top5_m_fgm.update(acc5_fgm.item(), output_fgm.size(0))
            
            losses_m_pgd.update(reduced_loss_pgd.item(), input_pgd.size(0))
            top1_m_pgd.update(acc1_pgd.item(), output_pgd.size(0))
            top5_m_pgd.update(acc5_pgd.item(), output_pgd.size(0))
            
            batch_time_m.update(time.time() - end)
            end = time.time()
            # if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
            #     log_name = 'Test' + log_suffix
            #     _logger.info(
            #         '{0}: [{1:>4d}/{2}]  '
            #         'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            #         'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
            #         'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            #         'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
            #             log_name, batch_idx, last_idx, batch_time=batch_time_m,
            #             loss=losses_m, top1=top1_m, top5=top5_m))    
            
    return top1_m_fgm.avg, top5_m_fgm.avg, losses_m_fgm.avg, top1_m_pgd.avg, top5_m_pgd.avg, losses_m_pgd.avg

def Normalize(tensor, mean, std, inplace=False):
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError(
            'Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                        '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def UnNormalize(tensor, mean, std, inplace=False):
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError(
            'Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                        '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    # tensor.sub_(mean).div_(std)
    return tensor

def val_rob(model, loader, loss_fn, args, output_dir=None, amp_autocast=suppress, log_suffix=''):
    givenA_txt = ''
    givenB_txt = ''
    givenA_list = []
    givenB_list = []
    for i in args.givenA:
        givenA_txt += str(i)+'_'
        givenA_list += str(i)
    print('args.givenA, args.givenB', args.givenA, args.givenB)


    for i in args.givenB:
        givenB_txt += str(i)+'_'        
        givenB_list += str(i)
        
    ConverOrder = args.ConverOrd
    if args.adv_train:
        head_list = ['Model',  "Step", "Order", "Alphas", "Betas", 'Noise Type', 'Noise Value', 'Adv. Train Acc.', 'Adv. Top-1 Acc.', 'Adv. Top-5 Acc.']#"FLOPs","# Params", "# Trainable Params", 
    else:
        head_list = ['Model',"Step", "Order", "Alphas", "Betas",'Noise Type', 'Noise Value', 'Train Acc.', 'Top-1 Acc.', 'Top-5 Acc.']#"FLOPs","# Params", "# Trainable Params", 
    df = pd.DataFrame(columns=head_list)
    
    # model = torch.load(os.path.join(args.save_path, 'model_state_best.pth.tar')) 
    noise_dict = {'const': [0,0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.45, -0.01, -0.05,-0.1,-0.2, -0.3, -0.4, -0.45],
            'randn': [0.01,  0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
            'rand': [0.04, 0.06, 0.08, 0.1, 0.12, 0.2, -0.04, -0.06, -0.08, -0.1, -0.12,-0.2],
            }
    # exps = ["0", "1", "2"]
    eps_list = [3.0/255, 5.0/255, 8.0/255] #  16.0/255, 32.0/255]
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    norm = args.norm        
    
    for noise_type in noise_dict.keys():  
        Noise_Coffs = []
        Top1_pertur_tests = []
        Top5_pertur_tests = []
        
        Top1_pertur_trains = []
        # eval_metrics_pertur = validate(model, loader, loss_fn, args, amp_autocast=amp_autocast)
        # if args.rank == 0:
        #     print('eval_metrics_ori', eval_metrics_pertur)
        #     for k, v in eval_metrics_pertur.items():
        #         name = 'val_' + k
        #         print('name, v', name, v)        
        for noise_coff in noise_dict.get(noise_type):  
            # if noise_type == 'const':
            #     loss_per = log_feature(model, testloader, noise_type, noise_coff)
            #     print('loss_per',loss_per) 
            # top1_pertur_test, top5_pertur_test = noise_rob(model, loader, loss_fn, args, noise_type, noise_coff, amp_autocast=amp_autocast)
            eval_metrics_pertur = noise_rob(model, loader, loss_fn, args, noise_type, noise_coff, amp_autocast=amp_autocast)

            if args.rank == 0:
                print('eval_metrics_rob', eval_metrics_pertur)
                for k, v in eval_metrics_pertur.items():
                    name = 'val_' + k
                    print('name, v', name, v)
                # top1_pertur_train = noise_rob(model, trainloader, noise_type, noise_coff)
                Noise_Coffs += [noise_coff]
                Top1_pertur_tests += [eval_metrics_pertur['top1']]
                Top5_pertur_tests += [eval_metrics_pertur['top5']]
                
                # Top1_pertur_trains += [0]
                # Top1_pertur_trains += [top1_pertur_train.item()]
                df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, 0, eval_metrics_pertur['top1'], eval_metrics_pertur['top5']]], columns=head_list)

                # df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, noise_type, noise_coff, top1_pertur_train.item(), top1_pertur_test.item() ]], columns=head_list)
                df = df.append(df_row)     
                print(noise_type+'Noise_Coffs, Top1_pertur_tests, Top5_pertur_tests',Noise_Coffs, Top1_pertur_tests, Top5_pertur_tests)                   
            if noise_coff ==0 and noise_type =='const' and False:
                    for eps in eps_list:  
                        print('eps, eps_iter, nb_iter, norm', eps, eps_iter, nb_iter, norm)
                        #top1_m_fgm.avg, top5_m_fgm.avg, losses_m_fgm.avg, top1_m_pgd.avg, top5_m_pgd.avg, losses_m_pgd.avg

                        top1_m_fgm, top5_m_fgm, losses_m_fgm, top1_m_pgd, top5_m_pgd, losses_m_pgd = eval_adv(model, loader, loss_fn, args, eps, eps_iter, nb_iter, norm, amp_autocast=amp_autocast)
                      
                        # train_top1_fgm, train_top1_pgd, train_losses_fgm, train_losses_pgd, train_batch_time = eval_adv(model, trainloader, eps, eps_iter, nb_iter, norm)
                        print('test_top1_fgm, test_top1_pgd, test_top5_fgm, test_top5_pgd', top1_m_fgm.data, top1_m_pgd.data, top5_m_fgm.data, top5_m_pgd.data)
                        # print('train_top1_fgm, train_top1_pgd',train_top1_fgm, train_top1_pgd)    
                        df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, 0, top1_m_pgd, top5_m_pgd]], columns=head_list)
                        df = df.append(df_row)                        
                        df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, 0, top1_m_fgm, top5_m_fgm ]], columns=head_list)                                             
                        df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, 0, top1_m_fgm, top5_m_fgm ]], columns=head_list)                                             
                        # df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, 'PGD', eps, train_top1_pgd.item(), test_top1_pgd.item() ]], columns=head_list)
                        # df = df.append(df_row)                        
                        # df_row = pd.DataFrame([[args.model, args.steps, ConverOrder, givenA_txt, givenB_txt, 'FGM', eps, train_top1_fgm.item(), test_top1_fgm.item() ]], columns=head_list)
                        df = df.append(df_row)
                        print(noise_type+'Noise_Coffs, Top1_pertur_tests, Top5_pertur_tests',Noise_Coffs, Top1_pertur_tests, Top5_pertur_tests)  
      
    if args.rank == 0:
        print('Table \n',df)
        save_path = output_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = args.model+args.notes+'.csv'
        print('save_path+file_name', save_path+file_name)
        df.to_csv(save_path+file_name)
        #experiment.log_table(save_path+file_name)
    return df


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        if 1: #batch_idx==0:
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if not args.prefetcher:
                input, target = input.cuda(), target.cuda()
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

            if not args.distributed:
                losses_m.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                #print('loss_scaler')
                loss_scaler(
                    loss, optimizer,
                    clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()
                
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)
                    
            if model_ema is not None:
                model_ema.update(model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            # if args.distributed and args.local_rank == 0:
            #     reduced_loss = reduce_tensor(loss.data, args.world_size)
            #     losses_m.update(reduced_loss.item(), input.size(0))
            #     print('losses_m', losses_m)
                # experiment.log_metric('iter_loss', losses_m, step=epoch*last_idx + batch_idx)
                
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))

                if args.local_rank == 0:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True)

            if saver is not None and args.recovery_interval and (
                    last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if 1: #batch_idx==0:
                last_batch = batch_idx == last_idx
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    acc1 = reduce_tensor(acc1, args.world_size)
                    acc5 = reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
                # end for
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics




    
if __name__ == '__main__':
    main()
