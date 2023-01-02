'''Train CIFAR10 with PyTorch.'''
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
from torchvision import utils as vutils
import os
import argparse

from tensorboardX import SummaryWriter
from GAF import SGD_atan, SGD_atanMom, Adam_atan, SGD_atanMom_Ada
from Adam import Adam_atan
from RMSprop import RMSprop

from models_cifar10 import *
from datetime import datetime
import errno
import shutil

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

torch.manual_seed(1)  
# Training


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume', type=bool, default=False)

parser.add_argument('--epoch', type=int, default=200, help='training epoch')
parser.add_argument('--warm', type=int, default=5,
                    help='warm up training phase')

parser.add_argument('--data', default='./data', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', default='ResNet18', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--bs', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Optimization options
parser.add_argument('--opt_level', default='O2', type=str,
                    help='O2 is fast mixed FP16/32 training, O0 (FP32 training) and O3 (FP16 training), O1 ("conservative mixed precision"), O2 ("fast mixed precision").--opt_level O1 and O2 both use dynamic loss scaling by default unless manually overridden. --opt-level O0 and O3 (the "pure" training modes) do not use loss scaling by default. See more in https://github.com/NVIDIA/apex/tree/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet')
parser.add_argument('--keep-batchnorm-fp32', default=True, action='store_true',
                    help='keeping cudnn bn leads to fast training')
parser.add_argument('--loss-scale', type=float, default=None)
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')
parser.add_argument('--warmup', '--wp', default=5, type=int,
                    help='number of epochs to warmup')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5 for mobile models)')
parser.add_argument('--wd-all', dest='wdall', action='store_true',
                    help='weight decay on all parameters')
parser.add_argument('--opt', default='SGD_ori', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--ex", default=0, type=int)
parser.add_argument("--notes", default='', type=str)
parser.add_argument('--PL', type=float, default=1.0)
parser.add_argument('--sche', default='cos', type=str)
parser.add_argument('--k_ini', type=float, default=1)
parser.add_argument('--fix_k', type=bool, default=False)
parser.add_argument(
    '--given_ks', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)

parser.add_argument("--noise_coff", default=0.1, type=float)
parser.add_argument("--save_path", default='', type=str)


parser.add_argument('--noise_type', required=True, type=str)

args = parser.parse_args()
save_true_path = args.save_path



if args.checkpoint is None:
    args.checkpoint = save_true_path + '/model_best.pth.tar'
    print('args.checkpoint', args.checkpoint)



def test(epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ks, stepsize = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # test_loss += loss.item()
            # _, preds = outputs.max(1)
            # correct += preds.eq(old_labels).sum()
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            losses.avg, top1.avg))

    return top1.avg, losses.avg, batch_time.sum


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
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))





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


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.save_path) and args.local_rank == 0:
        mkdir_p(args.save_path)

    if args.dataset == 'cifar10':

        print('==> Preparing cifar10 data..')

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.bs, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.bs, shuffle=False, num_workers=args.workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    elif args.dataset == 'cifar100':

        print('==> Preparing cifar100 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, args.bs, shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR100(
            root='./data/cifar100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, args.bs, shuffle=False, num_workers=args.workers)


    print('==> Building model..')
    net_name = args.arch
    model_name = args.arch
    
    if args.arch == "HONet20_given":
        net = HONet20_given(PL=args.PL, k_ini=args.k_ini,
                            given_ks=args.given_ks)
    elif args.arch == "HONet32_given":
        net = HONet32_given(PL=args.PL, k_ini=args.k_ini,
                            given_ks=args.given_ks)
    elif args.arch == "HONet44_given":
        net = HONet44_given(PL=args.PL, k_ini=args.k_ini,
                            given_ks=args.given_ks)
    elif args.arch == "HONet56_given":
        net = HONet56_given(PL=args.PL, k_ini=args.k_ini,
                            given_ks=args.given_ks)
    elif args.arch == "HONet110_given":
        net = HONet110_given(PL=args.PL, k_ini=args.k_ini,
                             given_ks=args.given_ks)
    elif args.arch == "HONet164_given":
        net = HONet164_given(PL=args.PL, k_ini=args.k_ini,
                             given_ks=args.given_ks)
    elif args.arch == "HONet326_given":
        net = HONet326_given(PL=args.PL, k_ini=args.k_ini,
                             given_ks=args.given_ks)
    elif args.arch == "HONet650_given":
        net = HONet650_given(PL=args.PL, k_ini=args.k_ini,
                             given_ks=args.given_ks)
    elif args.arch == "HONet1298_given":
        net = HONet1298_given(PL=args.PL, k_ini=args.k_ini,
                              given_ks=args.given_ks)
                              
    if args.arch == "ZeroSNet20_Tra":
        net = ZeroSNet20_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    elif args.arch == "ZeroSNet32_Tra":
        net = ZeroSNet32_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet44_Tra":
        net = ZeroSNet44_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet56_Tra":
        net = ZeroSNet56_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet110_Tra":
        net = ZeroSNet110_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet164_Tra":
        net = ZeroSNet164_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet326_Tra":
        net = ZeroSNet326_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet650_Tra":
        net = ZeroSNet650_Tra(PL=args.PL, coe_ini=args.coe_ini)
    elif args.arch == "ZeroSNet1298_Tra":
        net = ZeroSNet1298_Tra(PL=args.PL, coe_ini=args.coe_ini)


    elif args.arch == "ZeroSNet20_Opt":
        net = ZeroSNet20_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet32_Opt":
        net = ZeroSNet32_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet44_Opt":
        net = ZeroSNet44_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet56_Opt":
        net = ZeroSNet56_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet110_Opt":
        net = ZeroSNet110_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet164_Opt":
        net = ZeroSNet164_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet326_Opt":
        net = ZeroSNet326_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet650_Opt":
        net = ZeroSNet650_Opt(PL=args.PL)
    elif args.arch == "ZeroSNet1298_Opt":
        net = ZeroSNet1298_Opt(PL=args.PL)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    criterion = nn.CrossEntropyLoss()
    if args.opt == 'SGD_ori':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'Adam_ori':
        optimizer = Adam_atan(net.parameters(), betas=(
            0.9, 0.999), weight_decay=5e-4, alpha=-1, beta=-1)
    if args.sche == 'cos':
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epoch)
    elif args.sche == 'step':
        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[250, 375], gamma=0.1)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    title = 'CIFAR-' + args.arch
    args.lastepoch = -1
    if args.resume:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.lastepoch = checkpoint['epoch']


    train_time = 0.0
    test_time = 0.0
    train_top1_acc = 0.0
    train_min_loss = 100
    test_top1_acc = 0.0
    test_min_loss = 100
    best_prec1 = -1

    end = time.time()
    net.eval()
    with torch.no_grad():
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ks, stepsize = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            losses.avg, top1.avg))


        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, ks, stepsize = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
        print('Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            losses.avg, top1.avg))

    # print('-------->', args.noise_type, args.noise_coff)
    net.eval()
    with torch.no_grad():
        losses_pertur = AverageMeter('Loss', ':.4e')
        top1_pertur = AverageMeter('Acc@1', ':6.2f')
        top5_pertur = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs_pertur, targets = inputs.to(device), targets.to(device)

            if args.dataset == 'cifar10':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            if args.noise_type == 'randn':
                inputs_pertur = inputs_pertur + args.noise_coff * \
                    torch.autograd.Variable(torch.randn(
                        inputs_pertur.size()).cuda(), requires_grad=False)


            elif args.noise_type == 'rand':
                inputs_pertur = inputs_pertur + args.noise_coff * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).cuda(), requires_grad=False)

            elif args.noise_type == 'const':
                inputs_pertur = inputs_pertur + args.noise_coff + 0 * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).cuda(), requires_grad=False)


            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            outputs_pertur, ks, stepsize = net(inputs_pertur)
            loss_pertur = criterion(outputs_pertur, targets)
            acc1_pertur, acc5_pertur = accuracy(
                outputs_pertur, targets, topk=(1, 5))
            losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))

        print('Noise Test set: Average loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
            losses_pertur.avg, top1_pertur.avg))

        losses_pertur = AverageMeter('Loss', ':.4e')
        top1_pertur = AverageMeter('Acc@1', ':6.2f')
        top5_pertur = AverageMeter('Acc@5', ':6.2f')
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs_pertur, targets = inputs.to(device), targets.to(device)

            if args.dataset == 'cifar10':
                inputs_pertur = UnNormalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            if args.noise_type == 'randn':
                inputs_pertur = inputs_pertur + args.noise_coff * \
                    torch.autograd.Variable(torch.randn(
                        inputs_pertur.size()).cuda(), requires_grad=False)
            elif args.noise_type == 'rand':
                inputs_pertur = inputs_pertur + args.noise_coff * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).cuda(), requires_grad=False)
            elif args.noise_type == 'const':
                inputs_pertur = inputs_pertur + args.noise_coff + 0 * torch.autograd.Variable(torch.rand(
                    inputs_pertur.size()).cuda(), requires_grad=False)
            inputs_pertur = F.relu(
                F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

            if args.dataset == 'cifar10':
                inputs_pertur = Normalize(
                    inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            outputs_pertur, ks, stepsize = net(inputs_pertur)
            loss_pertur = criterion(outputs_pertur, targets)
            acc1_pertur, acc5_pertur = accuracy(
                outputs_pertur, targets, topk=(1, 5))
            losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
            top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
            top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))

        print('Noise Train set: Average loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
            losses_pertur.avg, top1_pertur.avg))
