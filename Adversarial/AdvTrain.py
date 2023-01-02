"""
Adversarial Training.
"""
from comet_ml import Experiment, OfflineExperiment, Optimizer

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torchextractor as tx
import sys

import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from core.utils.train import SCHEDULERS

from core.utils import str2bool, str2float
import gc
# from numba import cuda
# parser = parser_train()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')

parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
parser.add_argument('--batch-size-validation', type=int, default=512, help='Batch size for testing.')
parser.add_argument('--data-dir', type=str, default='/media_SSD_1/datasets/')
parser.add_argument('--log-dir', type=str, default='./log/')

parser.add_argument('-d', '--data', type=str, default='cifar100', choices=DATASETS, help='Data to use.')
parser.add_argument('--desc', type=str, default='tuning',
                    help='Description of experiment. It will be used to name directories.')# required=True, 

parser.add_argument('-m', '--model', choices=MODELS, default='nrn-70-1-swish-learn', help='Model architecture to be used.')
parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

parser.add_argument('-na', '--num-adv-epochs', type=int, default=5, help='Number of adversarial training epochs.')
parser.add_argument('--adv-eval-freq', type=int, default=50, help='Adversarial evaluation frequency (in epochs).')

parser.add_argument('--beta', default=6.0, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')

parser.add_argument('--lr', type=float, default=0.2, help='Learning rate for optimizer (SGD).')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

parser.add_argument('--debug', action='store_true', default=False, 
                    help='Debug code. Run 1 epoch of training and evaluation.')
parser.add_argument('--mart', action='store_true', default=False, help='MART training.')

parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.', 
                    default=None)#/media_SSD_1/datasets/cifar100_ddpm.npz

parser.add_argument('--seed', type=int, default=1, help='Random seed.')

### Our methods
parser.add_argument('--LSE', action='store_true', default=True, help='LSE training.')
parser.add_argument('--ls', type=float, default=0.1, help='label smoothing.')
parser.add_argument('--clip_value', default=0, type=float)
parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')
parser.add_argument('--coesB', default=[-1, 0], nargs='+', type=float)
parser.add_argument('--learn', default=False, type=str2bool)
parser.add_argument('--IniRes', default=False, type=str2bool)
parser.add_argument('--Mask', default=False, type=str2bool)
parser.add_argument('--IniCh', default=8, type=int)
parser.add_argument('--settings', default='ShareExpDecayLearnDecay_AbsExp_RestaLayerIdx3', type=str)#ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3#ShareExpDecayLearnDecay_AbsExp_Ini0p07_RestaLayerIdx4
parser.add_argument('--IniDecay', default=0.14, type=float)#0.001
parser.add_argument("--CoesLR", default=0, type=float, help="")#0.001#0.0001




parser.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')





# To speed up training
torch.backends.cudnn.benchmark = True


                    
def plot_feature(net, epoch):
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
    save_path = LOG_DIR+'/'+args.settings+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    net.eval()
    # with torch.no_grad():         
    for i, (images, labels) in enumerate(test_dataloader):

        if i==0:#(i + 1) % 100 == 0:
            print('batch_number [{}/{}]'.format(i + 1, len(test_dataloader)))
            for j in range(len(images)):
                if j == 1:
                    if args.data == 'mnist':                    
                        image = images[j].resize(28, 28).to(device) 
                    elif 'cifar' in args.data or 'svhn' in args.data:                   
                        image = images[j].to(device)                         
                    print('labels[j]',labels[j])                    
                    break
            break


    # plt.rcParams.update({'font.size':14})
    if args.data == 'mnist':
        d = image.reshape(1,1,28,28)#.reshape(28,28)
        loc = 2
    elif 'cifar' in args.data or 'svhn' in args.data:
        d = image.reshape(1,3,32,32)#.reshape(28,28)
        loc = 8
    elif 'tiny-imagenet' in args.data:
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
    if 'convnext' in args.model:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity) and 'end_block_vis' in name #torch.nn.Identity)
    else:
        module_filter_fn = lambda module, name: isinstance(module, torch.nn.Identity) and 'start_block_vis' in name
    lb = 0
    ub = 50
    for mag in range(lb,ub,1):

        # d = d + mag/1000
        if args.data == 'mnist':
            x = d + mag/100*torch.rand(1,1,28,28).to(device)
        elif 'cifar' in args.data or 'svhn' in args.data:
            x = d + mag/100*torch.rand(1,3,32,32).to(device)       

        model = tx.Extractor(net, module_filter_fn=module_filter_fn)
        out, features = model(x) 
        
        fmap = []
        fmap2 = []

        feature_shapes = {name: f.shape for name, f in features.items()}
        for name, f in features.items():
            # print('f.mean()',f.mean())
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
    # experiment.log_image(save_path+file_name+'.pdf')
    plt.close()

    
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


    

    file_name = '/diff_ep'+str(epoch)
    df.to_csv(save_path+file_name+'.csv',index = False)
    ax_dif.figure.savefig(save_path+file_name+'.pdf')
    plt.close()


args =parser.parse_known_args()[0]


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)+'/'
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if not os.path.exists(LOG_DIR):
    # shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1
if not os.path.exists(args.log_dir+"/CometData"):
    os.makedirs(args.log_dir+"/CometData")
# experiment = Experiment(
experiment = OfflineExperiment(
    api_key="YourAPIKey",
    # project_name="OverThreeOrders",
    project_name="score-cifar",
    # project_name="overthreeorders-4channels",
    workspace="workspace",
    # auto_histogram_weight_logging=True,
    offline_directory=args.log_dir+"/CometData",
)
hyper_params = vars(args)
experiment.log_parameters(hyper_params)
seed(args.seed)
train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True, 
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
)
del train_dataset, test_dataset


# Adversarial Training (AT, TRADES and MART)

seed(args.seed)
trainer = Trainer(info, args)
last_lr_NoCoe = args.lr
last_lr_Coe = args.lr
ceosB_text = ''
if args.coesB is not None:
    for i in range(len(args.coesB)): 
        ceosB_text += "_b"+str(i)+"_"+str(args.coesB[i])[:6]
else:
    ceosB_text = ''

for name, p in trainer.model.named_parameters():
    if p.requires_grad: 
        if name:
            print(name,p.size())

if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))
    
    old_score = [0.0, 0.0]
    logger.log('Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    
    
plot_feature(trainer.model, 0)
for epoch in range(1, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    if args.scheduler:
        last_lr_NoCoe = trainer.scheduler_NoCoe.get_last_lr()[0]
        last_lr_Coe = trainer.scheduler_Coe.get_last_lr()[0]
    # coes_As = torch.tensor([]).to(device)
    # coes_Bs = torch.tensor([]).to(device)     
    coes_As = []
    coes_Bs = []
    coesDecay = []
    coesBalance = []
    coes_bias_outer = []
    coes_bias_inner = []
    for name, p in trainer.model.named_parameters():
        if 1:#p.requires_grad: 
            if name:
                if 'coesA' in name: # and 'bn3' not in name:
                    # print(name)
                    # sum_A = sum_A+p
                    # coes_As = coes_As+[p]
                    coes_As = torch.cat((coes_As, p ),0)
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
                elif 'coes_bias_outer' in name:
                    coes_bias_outer = coes_bias_outer + [float(p.mean().data)]
                elif 'coes_bias_inner' in name:
                    coes_bias_inner = coes_bias_inner +[float(p.mean().data)]                   
    # print('coes_Bs', coes_Bs)#.data)
    print('coesDecay', coesDecay)
    
    # print('coesBalance', coesBalance)
    for i in range(0,len(coesDecay)):
        experiment.log_metric('coesDecay'+str(i), coesDecay[i], step=epoch)
    for i in range(0,len(coes_bias_outer)):
        experiment.log_metric('coes_bias_outer'+str(i), coes_bias_outer[i], step=epoch)
    for i in range(0,len(coes_bias_inner)):
        experiment.log_metric('coes_bias_inner'+str(i), coes_bias_inner[i], step=epoch)  
                
    # experiment.log_histogram_3d(coesDecay, name='coesDecay', step=epoch)
    # experiment.log_histogram_3d(coes_Bs, name='coes_Bs', step=epoch)
    res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_acc = trainer.eval(test_dataloader)
    plot_feature(trainer.model, epoch)
    for item in res.items():
        name, value = item[0], item[1]
        print('name, value',name, value)
        experiment.log_metric(name, value, step=epoch)
    logger.log('Loss: {:.4f}.\tLR_NoCoe: {:.4f}.\tLR_Coe: {:.4f}'.format(res['loss'], last_lr_NoCoe,last_lr_Coe))
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    experiment.log_metric('Std. Acc. Test', test_acc, step=epoch)
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': epoch, 'lr_NoCoe': last_lr_NoCoe, 'lr_Coe': last_lr_Coe, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    
    if epoch % args.adv_eval_freq == 0 or epoch > (NUM_ADV_EPOCHS-5) or (epoch >= (NUM_ADV_EPOCHS-10) and NUM_ADV_EPOCHS > 90):
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        experiment.log_metric('Adv. Acc. Test', test_adv_acc, step=epoch)
        experiment.log_metric('SumTest', test_adv_acc+test_acc, step=epoch)
    else:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    
    if test_adv_acc >= old_score[1]:
        old_score[0], old_score[1] = test_acc, test_adv_acc
        trainer.save_model(WEIGHTS)
    trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

    
    
# Record metrics

train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

experiment.end()
logger.log('Script Completed.')

del trainer.model
# gc.collect()
# torch.cuda.empty_cache()
# with torch.no_grad():
#     torch.cuda.empty_cache()
# cuda.select_device(0)
# cuda.close()
# cmd = 'ps -ef|grep '
# cmd += args.desc.split('exp_')[1]
# cmd += '|grep -v grep|cut -c 9-15|xargs kill -9'
# os.system(cmd)