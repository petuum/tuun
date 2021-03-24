#!/usr/bin/env python

"""
    cifar10, cifar100 and SVHN
"""

from __future__ import division, print_function

import argparse
import json
import nni
import sys
from time import time

from basenet.hp_schedule import HPSchedule
from basenet.helpers import set_seeds
from basenet.vision import transforms as btransforms

import torch

from torchvision import transforms, datasets
from augment import Policy
from resnet import ResNet18


torch.backends.cudnn.benchmark = True


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
    parser.add_argument('--dataset', type=str, default='svhn',
                        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--num-classes', type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    datasets_mapping = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'svhn': datasets.SVHN,
    }

    args = parse_args()
    set_seeds(args.seed)

    # --
    # IO
    print('Making dataloaders ...', file=sys.stderr)
    transform_train = transforms.Compose([
        Policy(nni.get_next_parameter()),
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset=args.dataset),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        btransforms.NormalizeDataset(dataset=args.dataset),
    ])

    try:
        if args.dataset in ['cifar10', 'cifar100']:
            trainset = datasets_mapping[args.dataset](
                root='./data', train=True, download=args.download, transform=transform_train)
            testset = datasets_mapping[args.dataset](
                root='./data', train=False, download=args.download, transform=transform_eval)
        elif args.dataset in ['svhn']:
            trainset = datasets_mapping[args.dataset](
                root='./data', split='train', download=args.download, transform=transform_train)
            testset = datasets_mapping[args.dataset](
                root='./data', split='test', download=args.download, transform=transform_eval)
    except RuntimeError:
        raise Exception(r'{args.dataset}: error loading data -- try rerunning w/ `--download` flag')

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
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
        "train": trainloader,
        "test": testloader,
    }

    # --
    # Define model

    print(r'{args.dataset}: initializing model...', file=sys.stderr)

    cuda = torch.device('cuda')
    model = ResNet18(num_classes=args.num_classes).to(cuda).half()
    model.verbose = True
    print(model, file=sys.stderr)

    # --
    # Initialize optimizer

    print('initializing optimizer...', file=sys.stderr)

    if args.lr_schedule == 'linear_cycle':
        lr_scheduler = HPSchedule.linear_cycle(hp_max=args.lr_max, epochs=args.epochs,
                                               extra=args.extra)
    elif args.lr_schedule == 'sgdr':
        lr_scheduler = HPSchedule.sgdr(
            hp_init=args.lr_max,
            period_length=args.sgdr_period_length,
            t_mult=args.sgdr_t_mult,
        )
    else:
        lr_scheduler = getattr(HPSchedule, args.lr_schedule)(
            hp_add=args.lr_max - 0.005, epochs=args.epochs)

    model.init_optimizer(
        opt=torch.optim.SGD,
        params=model.parameters(),
        hp_scheduler={"lr": lr_scheduler},
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # --
    # Train

    print('training...', file=sys.stderr)
    best_result = 0.
    t = time()
    for epoch in range(args.epochs + args.extra + args.burnout):
        train = model.train_epoch(dataloaders, mode='train', metric_fns=['n_correct'])
        test = model.eval_epoch(dataloaders, mode='test', metric_fns=['n_correct'])
        nni.report_intermediate_result(test['acc'])
        print(json.dumps({
            "epoch": int(epoch),
            "lr": model.hp['lr'],
            "test_acc": float(test['acc']),
            "train_acc": float(train['acc']),
            "time": time() - t,
        }))
        best_result = max(float(test['acc']), best_result)
        sys.stdout.flush()
    nni.report_final_result(best_result)
