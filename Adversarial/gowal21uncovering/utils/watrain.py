import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)
        
        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter, 
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data == 'tiny-imagenet':
            self.num_classes = 200
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n or 'coes' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups
        # print('group_weight(self.model)[1]',group_weight(self.model)[1])
        
        
        # self.optimizer_NoCoe = torch.optim.SGD([group_weight(self.model)[0]], lr=self.params.lr, weight_decay=self.params.weight_decay, momentum=0.9, nesterov=self.params.nesterov)
        # if 'SGD' in self.params.settings:
        #     self.optimizer_Coe = torch.optim.SGD([group_weight(self.model)[1]], lr=self.params.CoesLR, weight_decay=self.params.weight_decay, momentum=0.9, nesterov=self.params.nesterov)
        # else:
        #     self.optimizer_Coe = torch.optim.Adam([group_weight(self.model)[1]], lr=self.params.CoesLR, weight_decay=self.params.weight_decay) 
            
        self.optimizer_NoCoe = torch.optim.SGD([{'params':[ param for name, param in self.model.named_parameters() if ('Fix' not in name and 'coes' not in name and 'batchnorm' not in name)]}, 
        ], lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)  
        if 'SGD' in self.params.settings:
            self.optimizer_Coe = torch.optim.SGD([
            {'params': (p for name, p in self.model.named_parameters() if 'coes' in name or 'batchnorm' in name), 'weight_decay': 0.,},
            ], lr=self.params.CoesLR, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)        
        else:      
            self.optimizer_Coe = torch.optim.Adam([
            {'params': (p for name, p in self.model.named_parameters() if 'coes' in name or 'batchnorm' in name), 'weight_decay': 0.,},
            ], lr=self.params.CoesLR, weight_decay=self.params.weight_decay)               
        # self.optimizer_NoCoe = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
        #                                  momentum=0.9, nesterov=self.params.nesterov)         
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            if self.params.CutMix:
                x_all, y_all = torch.tensor([]), torch.tensor([])
                for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                    x_tmp, y_tmp = x.detach(), y.detach()
                    x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                    x_all = torch.cat((x_all, x_tmp), dim=0)
                    y_all = torch.cat((y_all, y_tmp), dim=0)
                x, y = x_all.to(device), y_all.to(device)
            else:
                x, y = x.to(device), y.to(device)
            
            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None and self.params.LSE:
                    loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y)
                
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer_NoCoe.step()
            self.optimizer_Coe.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler_NoCoe.step()
                self.scheduler_Coe.step()
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler_NoCoe.step()
            self.scheduler_Coe.step()
        
        update_bn(self.wa_model, self.model) 
        return dict(metrics.mean())
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer_NoCoe, self.optimizer_Coe, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix)
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """
        loss, batch_metrics = trades_loss_LSE(self.model, x, y, self.optimizer_NoCoe, self.optimizer_Coe, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          clip_value=self.params.clip_value,
                                          use_cutmix=self.params.CutMix,
                                          num_classes=self.num_classes)
        return loss, batch_metrics  

    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.wa_model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
    

def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked