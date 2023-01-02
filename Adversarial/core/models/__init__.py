import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .narrowresnetwithswish import narrowresnetwithswish
from .narrowresnetwithswish_learn import narrowresnetwithswish_learn

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          
          'wrn-28-20-swish', 'wrn-34-20-swish', 'wrn-70-20-swish',
          'wrn-28-16-swish', 'wrn-34-16-swish', 'wrn-70-16-swish',
          'wrn-28-10-swish', 'wrn-34-10-swish', 'wrn-70-10-swish',

          'nrn-28-16-swish', 'nrn-34-16-swish', 'nrn-70-16-swish',
          'nrn-28-10-swish', 'nrn-34-10-swish', 'nrn-70-10-swish',
          'nrn-28-4-swish', 'nrn-34-4-swish', 'nrn-70-4-swish',
          'nrn-28-2-swish', 'nrn-34-2-swish', 'nrn-70-2-swish',
          'nrn-28-1-swish', 'nrn-34-1-swish', 'nrn-70-1-swish','nrn-52-1-swish', 'nrn-58-1-swish', 'nrn-64-1-swish', 'nrn-70-1-swish',

          'nrn-28-1-swish-learn', 'nrn-34-1-swish-learn','nrn-52-1-swish-learn', 'nrn-58-1-swish-learn', 'nrn-64-1-swish-learn', 'nrn-70-1-swish-learn',
          'nrn-28-2-swish-learn', 'nrn-34-2-swish-learn', 'nrn-70-2-swish-learn',
          'nrn-28-4-swish-learn', 'nrn-34-4-swish-learn', 'nrn-70-4-swish-learn', 
          'nrn-28-10-swish-learn', 'nrn-34-10-swish-learn', 'nrn-70-10-swish-learn',    
          'nrn-28-16-swish-learn', 'nrn-34-16-swish-learn', 'nrn-70-16-swish-learn',                       
          ]


def create_model(name, normalize, info, device,coesB=[-1,0],learn=False,IniRes=False,Mask=False,IniCh=16,settings='',IniDecay=-4):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if info['data'] in ['tiny-imagenet']:
        assert 'preact-resnet' in name, 'Only preact-resnets are supported for this dataset!'
        from .ti_preact_resnet import ti_preact_resnet
        backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)
    
    elif info['data'] in DATASETS and info['data'] not in ['tiny-imagenet']:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        elif 'nrn' in name and 'swish' in name:
            if 'learn' in name:
                backbone = narrowresnetwithswish_learn(name, dataset=info['data'], num_classes=info['num_classes'], device=device, coesB=coesB,learn=learn,IniRes=IniRes,Mask=Mask,IniCh=IniCh,settings=settings,IniDecay=IniDecay)    #-1.91666667,  1.33333333, -0.41666667#-1.5,  0.5 #-2.29166667,  2.45833333, -1.54166667, -0.375
            else:                                  
                backbone = narrowresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)            
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
