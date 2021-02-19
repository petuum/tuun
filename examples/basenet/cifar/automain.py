#!/usr/bin/env python

"""
    cifar10.py and SVHN
"""

from __future__ import division, print_function

import sys
import json
import argparse
import numpy as np
import nni
from time import time
from PIL import Image

from basenet import BaseNet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds
from basenet.vision import transforms as btransforms

import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True

from torchvision import transforms, datasets
from augment import Policy
# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--extra', type=int, default=0)
    parser.add_argument('--burnout', type=int, default=0)
    parser.add_argument('--lr-schedule', type=str, default='one_cycle')
    parser.add_argument('--lr-max', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    
    parser.add_argument('--sgdr-period-length', type=int, default=10)
    parser.add_argument('--sgdr-t-mult', type=int, default=2)
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--download', action="store_false")
    parser.add_argument('--dataset', type=str, default='svhn')
    return parser.parse_args()

args = vars(parse_args())
tuner_params = nni.get_next_parameter()
args.update(tuner_params)

set_seeds(args["seed"])


operations = [[args["policy1_0"], args["policy1_1"]], 
              [args["policy2_0"], args["policy2_1"]],
              [args["policy3_0"], args["policy3_1"]],
              [args["policy4_0"], args["policy4_1"]],
              [args["policy5_0"], args["policy5_1"]]]
probs = [[args["prob1_0"], args["prob1_1"]], 
         [args["prob2_0"], args["prob2_1"]],
         [args["prob3_0"], args["prob3_1"]],
         [args["prob4_0"], args["prob4_1"]],
         [args["prob5_0"], args["prob5_1"]]]
magnitudes = [[args["magnitude1_0"], args["magnitude1_1"]], 
              [args["magnitude2_0"], args["magnitude2_1"]],
              [args["magnitude3_0"], args["magnitude3_1"]],
              [args["magnitude4_0"], args["magnitude4_1"]],
              [args["magnitude5_0"], args["magnitude5_1"]]]

assert args["policy1_0"] != args["policy1_1"], "policy1_0 should be unequal to policy1_1"
assert args["policy2_0"] != args["policy2_1"], "policy2_0 should be unequal to policy2_1"
assert args["policy3_0"] != args["policy3_1"], "policy3_0 should be unequal to policy3_1"
assert args["policy4_0"] != args["policy4_1"], "policy4_0 should be unequal to policy4_1"
assert args["policy5_0"] != args["policy5_1"], "policy5_0 should be unequal to policy5_1"

## --
# IO
if args['dataset'] == 'cifar10':
    print('cifar10.py: making dataloaders...', file=sys.stderr)

    transform_train = transforms.Compose([
        # btransforms.ReflectionPadding(margin=(4, 4)),
        # transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        Policy(operations, probs, magnitudes),
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    try:
        trainset = datasets.CIFAR10(root='./data', train=True, download=args["download"], transform=transform_train)
        testset  = datasets.CIFAR10(root='./data', train=False, download=args["download"], transform=transform_test)
    except:
        raise Exception('cifar10.py: error loading data -- try rerunning w/ `--download` flag')
elif args['dataset'] == 'cifar100':
    print('cifar10.py: making dataloaders...', file=sys.stderr)

    transform_train = transforms.Compose([
        # btransforms.ReflectionPadding(margin=(4, 4)),
        # transforms.RandomCrop(32),
        # transforms.RandomHorizontalFlip(),
        Policy(operations, probs, magnitudes),
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    try:
        trainset = datasets.CIFAR100(root='./data', train=True, download=args["download"], transform=transform_train)
        testset  = datasets.CIFAR100(root='./data', train=False, download=args["download"], transform=transform_test)
    except:
        raise Exception('cifar100.py: error loading data -- try rerunning w/ `--download` flag')
else:
    print('svhn.py: making dataloaders...', file=sys.stderr)
    transform_train = transforms.Compose([
        #btransforms.ReflectionPadding(margin=(4, 4)),
        #transforms.RandomCrop(32),
        #transforms.RandomHorizontalFlip(),
        Policy(operations, probs, magnitudes),
        transforms.ToTensor(),
        #btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #btransforms.NormalizeDataset(dataset='cifar10'),
    ])

    try:
        trainset = datasets.SVHN(root='./data', split='train', download=args["download"], transform=transform_train)
        testset  = datasets.SVHN(root='./data', split='test', download=args["download"], transform=transform_test)
    except:
        raise Exception('svhn.py: error loading data -- try rerunning w/ `--download` flag')





trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

dataloaders = {
    "train" : trainloader,
    "test"  : testloader,
}

# --
# Model definition
# Derived from models in `https://github.com/kuangliu/pytorch-cifar`

class PreActBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class ResNet18(BaseNet):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__(loss_fn=F.cross_entropy)
        
        self.in_channels = 64
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.half()
        x = self.prep(x)
        
        x = self.layers(x)
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x)
        
        return x

# --
# Define model

print('cifar10.py: initializing model...', file=sys.stderr)

cuda = torch.device('cuda')
model = ResNet18().to(cuda).half()
model.verbose = True
print(model, file=sys.stderr)

# --
# Initialize optimizer

print('initializing optimizer...', file=sys.stderr)

if args["lr_schedule"] == 'linear_cycle':
    lr_scheduler = HPSchedule.linear_cycle(hp_add=args["lr_max"]-0.005, epochs=args["epochs"], extra=args["extra"])
elif args["lr_schedule"] == 'sgdr':
    lr_scheduler = HPSchedule.sgdr(
        hp_init=args["lr_max"],
        period_length=args["sgdr_period_length"],
        t_mult=args["sgdr_t_mult"],
    )
else:
    lr_scheduler = getattr(HPSchedule, args["lr_schedule"])(hp_max=args["lr_max"], epochs=args["epochs"])

model.init_optimizer(
    opt=torch.optim.SGD,
    params=model.parameters(),
    hp_scheduler={"lr" : lr_scheduler},
    momentum=args["momentum"],
    weight_decay=args["weight_decay"],
    nesterov=True,
)

# --
# Train

print('training...', file=sys.stderr)
t = time()
for epoch in range(args["epochs"] + args["extra"] + args["burnout"]):
    train = model.train_epoch(dataloaders, mode='train', metric_fns=['n_correct'])
    test  = model.eval_epoch(dataloaders, mode='test', metric_fns=['n_correct'])
    if epoch < args["epochs"] + args["extra"] + args["burnout"] - 1:
        nni.report_intermediate_result(test['acc'])
    else:
        nni.report_final_result(test['acc'])
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "train_acc" : float(train['acc']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

model.save('weights')
