#!/usr/bin/env python

"""
    vision.py
"""

import numpy as np
from PIL import Image
from torchvision import transforms

dataset_stats = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
    },
    'cifar100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'fashion_mnist': {
        'mean': (0.28604060411453247,),
        'std': (0.3530242443084717,),
    },
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,),
    },
    'svhn': {
        'mean': (0.4376821, 0.4437697, 0.47280442),
        'std': (0.19803012, 0.20101562, 0.19703614),
    }
}


def ReflectionPadding(margin=(4, 4)):
    def _reflection_padding(x):
        x = np.asarray(x)
        if len(x.shape) == 2:
            x = np.pad(x, [(margin[0], margin[0]), (margin[1], margin[1])], mode='reflect')
        elif len(x.shape) == 3:
            x = np.pad(x, [(margin[0], margin[0]), (margin[1], margin[1]), (0, 0)], mode='reflect')

        return Image.fromarray(x)

    return transforms.Lambda(_reflection_padding)


def Cutout(cut_h, cut_w):
    assert cut_h % 2 == 0, "cut_h must be even"
    assert cut_w % 2 == 0, "cut_w must be even"

    def _cutout(x):
        c, h, w = x.shape

        h_center = np.random.choice(h)
        w_center = np.random.choice(w)

        h_hi = min(h, h_center + (cut_h // 2))
        h_lo = max(0, h_center - (cut_h // 2))

        w_hi = min(w, w_center + (cut_w // 2))
        w_lo = max(0, w_center - (cut_w // 2))

        mask = np.ones((c, h, w), dtype=np.float32)
        mask[:, h_lo:h_hi, w_lo:w_hi] = 0.0
        return x * mask

    return transforms.Lambda(_cutout)


def NormalizeDataset(dataset):
    assert dataset in set(dataset_stats.keys()), 'unknown dataset %s' % dataset
    return transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std'])


def DatasetPipeline(dataset):
    assert dataset in set(['cifar10', 'fashion_mnist']), 'unknown dataset %s' % dataset
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            ReflectionPadding(margin=(4, 4)),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NormalizeDataset(dataset='cifar10'),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            NormalizeDataset(dataset='cifar10'),
        ])

    elif dataset == 'fashion_mnist':
        transform_train = transforms.Compose([
            ReflectionPadding(margin=(4, 4)),
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NormalizeDataset(dataset='fashion_mnist'),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            NormalizeDataset(dataset='fashion_mnist'),
        ])

    return transform_train, transform_test
