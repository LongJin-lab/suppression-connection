import os
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime
import random
import errno
from random import randint

# from sympy import sec 

def gpu_info(GpuNum):
    gpu_status = os.popen('nvidia-smi -i '+str(GpuNum)+' | grep %').read().split('|')
    # print('gpu_status', gpu_status)
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split(
        '   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

def get_A(B_ab):
    A_ab = np.array([0]*len(B_ab))
    A_ab[0] = A_ab[0]+1
    A_ab = ' '+str(A_ab).replace('[', ' ').replace(']', ' ')+' '
    return A_ab

b_ = np.array([4277, -7923, 9982,-7298, 2877,-475])
B_ab6 = -b_/1440
b_ = np.array([1901, -2774, 2616, -1274, 251])
B_ab5 = -b_/720
b_ = np.array([55,-59,37,9])
B_ab4 = -b_/24
b_ = np.array([23,-16,5])
B_ab3 = -b_/12
b_ = np.array([3,-1])
B_ab2 = -b_/2
b_ = np.array([1, 0])
B_ab1 = -b_
b_ = np.array([-1, 0])
B_ab0 = -b_
A_ab1=get_A(B_ab1)
A_ab6=get_A(B_ab6)
A_ab5=get_A(B_ab5)
A_ab4=get_A(B_ab4)
A_ab3=get_A(B_ab3)
A_ab2=get_A(B_ab2)
A_ab0=get_A(B_ab0)

def SearchAndExe(Gpus, cmd, interval):
    prefix = 'CUDA_VISIBLE_DEVICES='
    foundGPU = 0
    while foundGPU==0:  # set waiting condition

        for u in Gpus: 
            gpu_power, gpu_memory = gpu_info(u)      
            cnt = 0   
            first = 0
            second = 0   
            empty = 1
            print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
            for i in range(12):
                gpu_power, gpu_memory = gpu_info(u)   
                print('gpu, gpu_power, gpu_memory, cnt', u, gpu_power, gpu_memory, cnt)
                if gpu_memory > 2000 or gpu_power > 150: # running
                    empty = 0
                time.sleep(interval)
            if empty == 1:
                foundGPU = 1
                break
            
    if foundGPU == 1:
        prefix += str(u)
        cmd = prefix + ' '+ cmd
        print('\n' + cmd)
        os.system(cmd)
    
def rand_port():
    r = ' '

    r += str(random.randint(1, 5))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += str(random.randint(1, 9))
    r += " "
    return r

def add_sp(str):
    return ' '+str+' '  

modelS = [
    # ' convnext_conver_nano_hnf ',
    ' nrn-28-1-swish-learn ',    
    ' nrn-34-1-swish-learn ',
    ' nrn-52-1-swish-learn ',    
    ' nrn-70-1-swish-learn ', 
    # ' nrn-34-1-swish-learn ',
    # # # ' nrn-52-2-swish ',
    # # # ' nrn-52-2-swish ',
    # ' nrn-58-1-swish-learn ',    
    # # ' nrn-64-1-swish-learn ', 
    
    # # ' nrn-28-10-swish-learn ',
]

coesB = [
    # ' -0.8766 0.1319 0.1264 0.0618 -0.0242 -0.0065 ',
        # ' '+str(B_ab1).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab0).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab2).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab3).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab4).replace('[', ' ').replace(']', ' '),
    #    ' '+str( B_ab5).replace('[', ' ').replace(']', ' '),
        # ' '+str(B_ab6).replace('[', ' ').replace(']', ' '),
   ' -1 0 ',
    # ' -1 1 ',
           ]

GPUS = [0,1,2,3]
datasetS = ['svhn','cifar10','cifar100']#'svhn',]#'svhn']#,'cifar100']#,'svhn']#, 'cifar100']#['mnist']#['cifar100']#, ['cifar10']'cifar10','cifar100',
auxdataS = ' /media_SSD_1/datasets/cifar10_ddpm.npz '#,' /media_SSD_1/datasets/cifar100_ddpm.npz ']#/media2/datasets/cifar10_ddpm.npz

# Settings = ['mnistConvStride2ResLikeExpDecayLearnDecay', 'mnistOriExpDecayLearnDecay', 'mnistAllEleExpDecayLearnDecay']

# note = 'AdvDDPM_ceosB'#'0508AllEleStep0p1',
settingsS = ['ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3']#'Default'] #,#ShareExpDecayLearnDecay_AbsExp_Ini0p5_RestaLayerIdx3#'ExpDecayLearnDecayLearnBal_Sigmoid']#,'Default'] #,'ExpDecayLearnDecayLearnBal_Sigmoid_n4ini'
# scheduler = ' cosinew '#'cos'#step
# bs = ' 512 ' #128
# lr = ' 0.2 ' #0.05
# epoch = ' 400 ' #110

scheduler = ' step '#'cos'#step
bs = ' 512 ' #128
lr = ' 0.2 ' #0.05
epoch = ' 110 ' #110

warm = ' 0 '
datadir = ' /media_SSD_1/datasets/ '#/media2/datasets/#/media_SSD_1/datasets/
beta = ' 6.0 '
unsup_fraction = ' 0.7 '
ls = ' 0 '
freq = ' 20 '
learnS = [' False ']#,' True ']
IniResS = [' False ']#,' True ']
MaskS = [' False ']#,' True ']
IniChS = [' 8 ']#TODO
attackS = [' linf-pgd ']#l2-pgd' linf-pgd ']#, 

attackepsS = [ add_sp(str(8.0/255))]#, add_sp(str(128.0/255)) ]
attackstepS = [ add_sp(str(2.0/255))] #, add_sp(str(16.0/255))
# attackepsS = [ add_sp(str(128.0/255))]#, add_sp(str(128.0/255)) ]
# attackstepS = [ add_sp(str(16.0/255))] #, add_sp(str(16.0/255))
# datapath = '/media_HDD_1/lab415/clm/OverThreeOrders/OverThreeOrders/CIFAR/data'
IniDecay = ' 0.2 '
CoesLR = ' 0.001 '
 
cnt=0

for exp in range(0, 1):
    for model in modelS:
        for settings in settingsS:
            for IniRes in IniResS:
                for Mask in MaskS:
                    for learn in learnS:
                        for IniCh in IniChS:
                            for attack in attackS:
                                for attackeps in attackepsS:
                                    for attackstep in attackstepS:
                                        for i in range(len(coesB)):
                                            
                                            for data in datasetS:
                                                logdir = './log/'+ settings
                                                if not os.path.exists(logdir):
                                                    os.makedirs(logdir)
                                                    
                                                pre = ''
                                                pre += ' nohup python3 trainTuning.py --data-dir '+datadir
                                                pre += ' --log-dir '+logdir
                                                pre += ' --attack-step '+attackstep
                                                pre += ' --attack-eps ' + attackeps
                                                pre += ' --settings '+ settings
                                                pre += ' --unsup-fraction '+unsup_fraction
                                                pre += ' --LSE --ls '+ls
                                                pre += ' --adv-eval-freq ' + freq                                                
                                                cmd = ''
                                                cmd +=  ' --coesB '+ coesB[i]
                                                cmd += ' --learn '+ learn
                                                cmd += ' --IniRes '+ IniRes
                                                cmd += ' --Mask '+ Mask
                                                cmd += ' --IniCh '+ IniCh
                                                
                                                cmd += ' --data '+ data
                                                cmd += ' --batch-size '+bs
                                                cmd += ' --model '+model
                                                cmd += ' --num-adv-epochs '+epoch
                                                cmd += ' --lr '+lr
                                                cmd += ' --scheduler '+ scheduler
                                                cmd += ' --beta '+beta
                                                cmd += ' --attack ' + attack
                                                cmd += ' --IniDecay '+IniDecay + ' --CoesLR '+ CoesLR
                                                desc =  cmd + 'exp_'+str(exp)+'_'+"{0:T%H%M%S}".format(datetime.now())
                                                desc = desc.replace('\n', '#').replace('\r', '#').replace(' ', '#').replace('##', '#').replace('--', '').replace('#','').replace('num-adv-epochs','ep').replace('scheduler','sche').replace('unsup-fraction','up').replace('attack-eps','eps').replace('attack-step','astep').replace('-batch-size','bs').replace('adv-eval-freq','fq')
                                                desc = desc.replace('#','').replace('-','_').replace('True','T').replace('False','F')
                                                cmd += ' --desc ' + desc
                                                # cmd += ' --aux-data-filename ' + auxdataS
                                                cmd = pre + cmd
                                                # if not os.path.exists(logdir+'/'+desc):
                                                #     os.makedirs(logdir+'/'+desc)
                                                # cmd += '   >   '+logdir+'/'+desc+'/log.txt 2>&1 & '
                                                print(cmd)
                                                # SearchAndExe(GPUS, cmd, interval=2)  #4
                                                cnt += 1

