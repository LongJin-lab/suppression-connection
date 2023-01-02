from comet_ml import Experiment, OfflineExperiment
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

# from tensorboardX import SummaryWriter

from models import *
from datetime import datetime
import errno
import shutil

# from train_cifar_ZeroSNet import *

# # experiment = Experiment(
# experiment = OfflineExperiment(
#     api_key="KbJPNIfbsNUoZJGJBSX4BofNZ",
#     # project_name="OverThreeOrders",
#     project_name="4channels_robustness",
#     workspace="logichen",
#     # auto_histogram_weight_logging=True,
#     offline_directory="/media/bdc/clm/OverThreeOrders/CIFAR/runs/CometData/robustness",
# )

# torch.manual_seed(1)  # 设置随机种子
# # Training


# parser = argparse.ArgumentParser(description='ZeorSNet CIFAR')
# parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
# parser.add_argument('--resume', type=bool, default=False)
# parser.add_argument('--epoch', type=int, default=160, help='training epoch')
# parser.add_argument('--warm', type=int, default=0, help='warm up training phase')
# parser.add_argument('--data', default='/media/bdc/clm/OverThreeOrders/CIFAR/data', type=str)
# parser.add_argument('--dataset', default='cifar10', type=str)
# parser.add_argument('--arch', '-a', default='ZeroSNet20_Opt', type=str)
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--bs', default=128, type=int,
#                     metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--test-batch', default=32, type=int, metavar='N',
#                     help='test batchsize (default: 200)')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
#                     help='path to save checkpoint (default: checkpoint)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
#                     metavar='W', help='weight decay (default: 4e-5 for mobile models)')
# parser.add_argument('--opt', default='SGD', type=str)
# parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument("--ex", default=0, type=int)
# parser.add_argument("--notes", default='', type=str)
# parser.add_argument('--PL', type=float, default=1.0)
# parser.add_argument('--sche', default='step', type=str)
# parser.add_argument('--coe_ini', type=float, default=1)
# parser.add_argument('--share_coe', type=bool, default=False)
# # parser.add_argument('--given_coe', default=[1.0 / 3, 5.0 / 9, 1.0 / 9, 16.0 / 9], nargs='+', type=float)
# parser.add_argument('--given_coe', default=None, nargs='+', type=float)
# parser.add_argument('--order', type=int, default=3)
# parser.add_argument('--ini_stepsize', default=1, type=float)
# parser.add_argument('--givenA', default=None, nargs='+', type=float)
# parser.add_argument('--givenB', default=None, nargs='+', type=float)

# parser.add_argument("--noise_coff", default=0.1, type=float)
# parser.add_argument("--save_path", default='PL1.0k_ini_-1.8a_0_0.3333333a_1_0.5555556a_2_0.1111111b_0_1.77777778share_k_False_sche_stepSGD_ori_BS128_LR0.1epoch160warmup02021-08-24T05-37', type=str)

# # dychf add
# parser.add_argument('--noise_type', required=True, type=str)

# args = parser.parse_args()
# givenA_text = ''
# givenB_text = ''
# if args.givenA is not None:
#     for i in range(len(args.givenA)): 
#         givenA_text += "a"+str(i)+"_"+str(args.givenA[i])[:4]
#         givenB_text += "_b"+str(i)+"_"+str(args.givenB[i])[:4]
# else:
#     givenA_text = ''
#     givenB_text = ''
# if args.share_coe:
#     share_coe_text = 'share_coe_True'
# else:
#     share_coe_text = 'share_coe_False'
# if args.dataset == "cifar10":
#     args.num_classes = 10
#     args.epoch = 160
# if args.dataset == "cifar100":
#     args.num_classes = 100
#     args.epoch = 300

# # args.save_path = 'runs/' + args.dataset + '/ZeroSNet/' + args.arch + '/PL' + str(args.PL) + 'coe_ini_' + str(
# #     args.coe_ini)  + givenA_text + givenB_text + share_coe_text+'_sche_' + args.sche + str(args.opt) + \
# #                  '_BS' + str(args.bs) + '_LR' + \
# #                  str(args.lr) + 'epoch' + \
# #                  str(args.epoch) + 'warmup' + str(args.warm) + \
# #                  args.notes + \
# #                      '_Ch4_'+\
# #                  "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())
                 
# save_true_path = args.save_path


# # checkpoint
# if args.checkpoint is None:
#     # args.checkpoint='checkpoints/imagenet/'+args.arch
#     args.checkpoint = save_true_path + '/model_best.pth.tar'
#     print('args.checkpoint', args.checkpoint)
    
# # print('givenB', args.givenB, 'givenA', args.givenA)
# hyper_params = {
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
#     'coe_ini': args.coe_ini,
#     # 'share_coe': args.share_coe,
#     'givenA': args.givenA,
#     'givenB': args.givenB,
#     'order': args.order,
#     'notes': args.notes
#     }
# experiment.log_parameters(hyper_params)


# # 测试模型代码
# def test(epoch):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     end = time.time()
#     net.eval()

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             if 'zeros' in args.arch or 'ZeroS' in args.arch:
#                 outputs, coes, stepsize = net(inputs)
#             elif 'MResNet' in args.arch:
#                 outputs, coes = net(inputs)
#             else:
#                 outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(acc1[0], inputs.size(0))
#             top5.update(acc5[0], inputs.size(0))
#             batch_time.update(time.time() - end)
#             end = time.time()
#             # test_loss += loss.item()
#             # _, preds = outputs.max(1)
#             # correct += preds.eq(old_labels).sum()
#         print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(losses.avg, top1.avg))
#     # experiment.log_metric("Test/Average loss", losses.avg, step=epoch)
#     # experiment.log_metric("Test/Accuracy-top1", top1.avg, step=epoch)
#     # experiment.log_metric("Test/Accuracy-top5", top5.avg, step=epoch)
#     # experiment.log_metric("Test/Time", batch_time.sum, step=epoch)
#     return top1.avg, losses.avg, batch_time.sum

def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # print(input_tensor.shape)
    # assert (len(input_tensor.shape) == 4)
    # input_tensor = input_tensor.clone().detach()
    # input_tensor = input_tensor.to(torch.device('cpu'))
    # # input_tensor = unnormalize(input_tensor)
    # if not os.path.isdir('/robust_img/') and args.local_rank == 0:
    #     mkdir_p('/robust_img/')
    # vutils.save_image(input_tensor, '/robust_img/'+filename)


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


# if __name__ == '__main__':

#     # if 1:
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     best_acc = 0  # best test accuracy
#     start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
#     # Data
    
#     # if not os.path.isdir(args.save_path) and args.local_rank == 0:
#         # mkdir_p(args.save_path)

    # if args.dataset == 'cifar10':
    #     # from models_cifar10 import *
    #     print('==> Preparing cifar10 data..')

    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                              (0.2023, 0.1994, 0.2010)),
    #     ])

    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                              (0.2023, 0.1994, 0.2010)),
    #     ])

    #     trainset = torchvision.datasets.CIFAR10(
    #         root= args.data +  '/cifar10', train=True, download=True, transform=transform_train)
    #     trainloader = torch.utils.data.DataLoader(
    #         trainset, args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

    #     testset = torchvision.datasets.CIFAR10(
    #         root= args.data + '/cifar10', train=False, download=True, transform=transform_test)
    #     testloader = torch.utils.data.DataLoader(
    #         testset, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    #     classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck')

    # elif args.dataset == 'cifar100':
    #     # from models_cifar100 import *
    #     print('==> Preparing cifar100 data..')
    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                              (0.2675, 0.2565, 0.2761)),
    #     ])

    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5071, 0.4867, 0.4408),
    #                              (0.2675, 0.2565, 0.2761)),
    #     ])

    #     trainset = torchvision.datasets.CIFAR100(
    #         root=args.data +'/cifar100', train=True, download=True, transform=transform_train)
    #     trainloader = torch.utils.data.DataLoader(
    #         trainset, args.bs, shuffle=True, num_workers=args.workers)

    #     testset = torchvision.datasets.CIFAR100(
    #         root=args.data +'/cifar100', train=False, download=True, transform=transform_test)
    #     testloader = torch.utils.data.DataLoader(
    #         testset, args.bs, shuffle=False, num_workers=args.workers)

    # #        classes = ('plane', 'car', 'bird', 'cat', 'deer',
    # #                   'dog', 'frog', 'horse', 'ship', 'truck')

    # # Model
    # print('==> Building model..')
    # net_name = args.arch
    # model_name = args.arch

    # if args.arch == "ZeroSAny20_Tra":
    #     net = ZeroSAny20_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)
    # if args.arch == "ZeroSAny32_Tra":
    #     net = ZeroSAny32_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)
    # if args.arch == "ZeroSAny44_Tra":
    #     net = ZeroSAny44_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)                
    # if args.arch == "ZeroSAny56_Tra":
    #     net = ZeroSAny56_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)
    # if args.arch == "ZeroSAny68_Tra":
    #     net = ZeroSAny68_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)    
    # if args.arch == "ZeroSAny80_Tra":
    #     net = ZeroSAny80_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)            
    # if args.arch == "ZeroSAny92_Tra":
    #     net = ZeroSAny92_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)            
    # if args.arch == "ZeroSAny104_Tra":
    #     net = ZeroSAny104_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)            
    # if args.arch == "ZeroSAny110_Tra":
    #     net = ZeroSAny110_Tra(order=args.order, givenA=args.givenA, givenB=args.givenB, PL=args.PL, coe_ini=args.coe_ini, num_classes=args.num_classes, ini_stepsize=args.ini_stepsize)            
        
        
    # if args.arch == "ZeroSNet20_Tra":
    #     net = ZeroSNet20_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet32_Tra":
    #     net = ZeroSNet32_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet44_Tra":
    #     net = ZeroSNet44_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet56_Tra":
    #     net = ZeroSNet56_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet110_Tra":
    #     net = ZeroSNet110_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet164_Tra":
    #     net = ZeroSNet164_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet326_Tra":
    #     net = ZeroSNet326_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet650_Tra":
    #     net = ZeroSNet650_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet1298_Tra":
    #     net = ZeroSNet1298_Tra(PL=args.PL, coe_ini=args.coe_ini, num_classes= args.num_classes)


    # elif args.arch == "ZeroSNet20_Opt":
    #     net = ZeroSNet20_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet32_Opt":
    #     net = ZeroSNet32_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet44_Opt":
    #     net = ZeroSNet44_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet56_Opt":
    #     net = ZeroSNet56_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet110_Opt":
    #     net = ZeroSNet110_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet164_Opt":
    #     net = ZeroSNet164_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet326_Opt":
    #     net = ZeroSNet326_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet650_Opt":
    #     net = ZeroSNet650_Opt(PL=args.PL, num_classes= args.num_classes)
    # elif args.arch == "ZeroSNet1298_Opt":
    #     net = ZeroSNet1298_Opt(PL=args.PL, num_classes= args.num_classes)

    # elif args.arch == "MResNet20":
    #     net = MResNet20(PL=args.PL)
    # elif args.arch == "MResNet32":
    #     net = MResNet32(PL=args.PL)
    # elif args.arch == "MResNet44":
    #     net = MResNet44(PL=args.PL)
    # elif args.arch == "MResNet56":
    #     net = MResNet56(PL=args.PL)
    # elif args.arch == "MResNet110":
    #     net = MResNet110(PL=args.PL)
    # elif args.arch == "MResNet164":
    #     net = MResNet164(PL=args.PL)
    # elif args.arch == "MResNet326":
    #     net = MResNet326(PL=args.PL)
    # elif args.arch == "MResNet650":
    #     net = MResNet650(PL=args.PL)
    # elif args.arch == "MResNet1298":
    #     net = MResNet1298(PL=args.PL)

    # elif args.arch == "MResNetSD20":
    #     net = MResNetSD20()
    # elif args.arch == "MResNetSD110":
    #     net = MResNetSD110()
    # elif args.arch == "MResNetC20":
    #     net = MResNetC20()
    # elif args.arch == "MResNetC32":
    #     net = MResNetC32()
    # elif args.arch == "MResNetC44":
    #     net = MResNetC44()
    # elif args.arch == "MResNetC56":
    #     net = MResNetC56()

    # elif args.arch == "DenseResNet20":
    #     net = DenseResNet20()
    # elif args.arch == "DenseResNet110":
    #     net = DenseResNet110()

    # elif args.arch == "ResNet_20":
    #     net = ResNet_20()
    # elif args.arch == "ResNet_32":
    #     net = ResNet_32()
    # elif args.arch == "ResNet_44":
    #     net = ResNet_44()
    # elif args.arch == "ResNet_56":
    #     net = ResNet_56()
    # elif args.arch == "ResNet_110":
    #     net = ResNet_110()
    # elif args.arch == "ResNet_164":
    #     net = ResNet_164()
    # elif args.arch == "ResNet_326":
    #     net = ResNet_326()
    # elif args.arch == "ResNet_650":
    #     net = ResNet_650()
    # elif args.arch == "ResNet_1298":
    #     net = ResNet_1298()

    # net = net.to(device)
    # net.apply(weights_init) 

    # if device == 'cuda':
    #     print('run on GPU')
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    # # Find total parameters and trainable parameters
    # total_params = sum(p.numel() for p in net.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD([{'params':[ param for name, param in net.named_parameters() if 'FixDS' not in name]}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # if args.sche == 'cos':
    #     train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, args.epoch)
    # elif args.sche == 'step':
    #     if args.dataset == 'cifar100':
    #         train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    #     elif args.dataset == 'cifar10':
    #         train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    # iter_per_epoch = len(trainloader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # # optionally resume from a checkpoint
    # title = 'CIFAR-' + args.arch
    # args.lastepoch = -1
    # # print('net:', net)
    # if args.resume:
    #     if os.path.isfile(args.checkpoint):
    #         print("=> loading checkpoint '{}'".format(args.checkpoint))
    #         # , map_location=lambda storage, loc: storage.cuda(args.gpu))
    #         checkpoint = torch.load(args.checkpoint)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         # net.load_state_dict(checkpoint['state_dict'])
    #         # net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    #         # optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #         args.lastepoch = checkpoint['epoch']
        
    #         # if args.local_rank == 0:
    #         #     logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        # else:
        #     print("=> no checkpoint found at '{}'".format(args.resume))

    # train_time = 0.0
    # test_time = 0.0
    # train_top1_acc = 0.0
    # train_min_loss = 100
    # test_top1_acc = 0.0
    # test_min_loss = 100
    # best_prec1 = -1

    # # batch_time = AverageMeter('Time', ':6.3f')
    # # losses = AverageMeter('Loss', ':.4e')
    # # top1 = AverageMeter('Acc@1', ':6.2f')
    # # top5 = AverageMeter('Acc@5', ':6.2f')
    # end = time.time()
# def noise_test(net):

#     print('==> Start Robust Evaluation')
    
#     net.eval()
#     with torch.no_grad():
#         batch_time = AverageMeter('Time', ':6.3f')
#         losses = AverageMeter('Loss', ':.4e')
#         top1 = AverageMeter('Acc@1', ':6.2f')
#         top5 = AverageMeter('Acc@5', ':6.2f')
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs, ks, stepsize = net(inputs)
#             loss = criterion(outputs, targets)
#             acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(acc1[0], inputs.size(0))
#             top5.update(acc5[0], inputs.size(0))
#             # batch_time.update(time.time() - end)
#             # end = time.time()
#             # test_loss += loss.item()
#             # _, preds = outputs.max(1)
#             # correct += preds.eq(old_labels).sum()
#         print('Test set_ori: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
#             losses.avg, top1.avg))
#         # experiment.log_metric("Test_ori/Average loss", losses.avg, step=1)
#         experiment.log_metric("Test_ori/Accuracy-top1", top1.avg, step=1)

#     #     # dychf add trainloader
#     #     batch_time = AverageMeter('Time', ':6.3f')
#     #     losses = AverageMeter('Loss', ':.4e')
#     #     top1 = AverageMeter('Acc@1', ':6.2f')
#     #     top5 = AverageMeter('Acc@5', ':6.2f')
#     #     for batch_idx, (inputs, targets) in enumerate(trainloader):
#     #         inputs, targets = inputs.to(device), targets.to(device)
#     #         outputs, ks, stepsize = net(inputs)
#     #         loss = criterion(outputs, targets)
#     #         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#     #         losses.update(loss.item(), inputs.size(0))
#     #         top1.update(acc1[0], inputs.size(0))
#     #         top5.update(acc5[0], inputs.size(0))
#     #         # batch_time.update(time.time() - end)
#     #         # end = time.time()
#     #         # test_loss += loss.item()
#     #         # _, preds = outputs.max(1)
#     #         # correct += preds.eq(old_labels).sum()
#     #     print('Train set ori: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
#     #         losses.avg, top1.avg))
#     #     experiment.log_metric("Train_ori/Average loss", losses.avg, step=0)
#     #     experiment.log_metric("Train_ori/Accuracy-top1", top1.avg, step=0)
#     #     # experiment.log_metric("Train/Accuracy-top5", top5.avg, step=epoch)
#     #     # experiment.log_metric("Train/Time", batch_time.sum, step=epoch)
#     # # print('-------->', args.noise_type, args.noise_coff)
    
#     net.eval()
#     with torch.no_grad():
#         losses_pertur = AverageMeter('Loss', ':.4e')
#         top1_pertur = AverageMeter('Acc@1', ':6.2f')
#         top5_pertur = AverageMeter('Acc@5', ':6.2f')
#         for batch_idx, (inputs, targets) in enumerate(testloader):

#             inputs_pertur, targets = inputs.to(device), targets.to(device)

#             if args.dataset == 'cifar10':
#                 inputs_pertur = UnNormalize(
#                     inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#             if args.noise_type == 'randn':
#                 inputs_pertur = inputs_pertur + args.noise_coff * \
#                     torch.autograd.Variable(torch.randn(
#                         inputs_pertur.size()).cuda(), requires_grad=False)

#             # 均匀分布
#             elif args.noise_type == 'rand':
#                 inputs_pertur = inputs_pertur + args.noise_coff * torch.autograd.Variable(torch.rand(
#                     inputs_pertur.size()).cuda(), requires_grad=False)
#             # 常数
#             elif args.noise_type == 'const':
#                 inputs_pertur = inputs_pertur + args.noise_coff + 0 * torch.autograd.Variable(torch.rand(
#                     inputs_pertur.size()).cuda(), requires_grad=False)

#             # 截断
#             inputs_pertur = F.relu(
#                 F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

#             if args.dataset == 'cifar10':
#                 inputs_pertur = Normalize(
#                     inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#             outputs_pertur, ks, stepsize = net(inputs_pertur)
#             loss_pertur = criterion(outputs_pertur, targets)
#             acc1_pertur, acc5_pertur = accuracy(
#                 outputs_pertur, targets, topk=(1, 5))
#             losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
#             top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
#             top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))

#         print('Noise Test set: Average loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
#             losses_pertur.avg, top1_pertur.avg))
#         # experiment.log_metric("Test_pertur/Average loss", losses_pertur.avg, step=1)
#         # experiment.log_metric("Test_pertur/Acc-top1/"+args.noise_type, top1_pertur.avg, step=1)
        
#         return top1_pertur.avg
        
        
#         # dychf add trainloader
#         losses_pertur = AverageMeter('Loss', ':.4e')
#         top1_pertur = AverageMeter('Acc@1', ':6.2f')
#         top5_pertur = AverageMeter('Acc@5', ':6.2f')
#         for batch_idx, (inputs, targets) in enumerate(trainloader):

#             inputs_pertur, targets = inputs.to(device), targets.to(device)

#             if args.dataset == 'cifar10':
#                 inputs_pertur = UnNormalize(
#                     inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#             if args.noise_type == 'randn':
#                 inputs_pertur = inputs_pertur + args.noise_coff * \
#                     torch.autograd.Variable(torch.randn(
#                         inputs_pertur.size()).cuda(), requires_grad=False)

#             # 均匀分布
#             elif args.noise_type == 'rand':
#                 inputs_pertur = inputs_pertur + args.noise_coff * torch.autograd.Variable(torch.rand(
#                     inputs_pertur.size()).cuda(), requires_grad=False)
#             # 常数
#             elif args.noise_type == 'const':
#                 inputs_pertur = inputs_pertur + args.noise_coff + 0 * torch.autograd.Variable(torch.rand(
#                     inputs_pertur.size()).cuda(), requires_grad=False)

#             # 截断
#             inputs_pertur = F.relu(
#                 F.relu(inputs_pertur.mul_(-1).add_(1)).mul_(-1).add_(1))

#             if args.dataset == 'cifar10':
#                 inputs_pertur = Normalize(
#                     inputs_pertur, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#             outputs_pertur, ks, stepsize = net(inputs_pertur)
#             loss_pertur = criterion(outputs_pertur, targets)
#             acc1_pertur, acc5_pertur = accuracy(
#                 outputs_pertur, targets, topk=(1, 5))
#             losses_pertur.update(loss_pertur.item(), inputs_pertur.size(0))
#             top1_pertur.update(acc1_pertur[0], inputs_pertur.size(0))
#             top5_pertur.update(acc5_pertur[0], inputs_pertur.size(0))

#         print('Noise Train set: Average loss_pertur: {:.4f}, Accuracy_pertur: {:.4f}'.format(
#             losses_pertur.avg, top1_pertur.avg))
#         experiment.log_metric("Train_pertur/Average loss", losses_pertur.avg, step=1)
#         experiment.log_metric("Train_pertur/Accuracy-top1", top1_pertur.avg, step=1)
