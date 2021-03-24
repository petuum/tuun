#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import print_function, division

import random
import numpy as np
from functools import reduce

import torch
from torch import nn
from torch.autograd import Variable

TORCH_VERSION_3 = '0.3' == torch.__version__[:3]


# --
# Utils

def set_seeds(seed=100):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 123)
    _ = torch.cuda.manual_seed(seed + 456)
    _ = random.seed(seed + 789)


def to_device(x, device):
    assert device is not None, "basenet.helpers.to_device: device is None"
    if not TORCH_VERSION_3:
        if isinstance(x, tuple) or isinstance(x, list):
            return [xx.to(device) for xx in x]
        else:
            return x.to(device)
    else:
        if device == 'cuda':
            return x.cuda()
        elif device == 'cpu':
            return x.cpu()
        else:
            raise Exception


if not TORCH_VERSION_3:
    def to_numpy(x):
        if type(x) in [list, tuple]:
            return [to_numpy(xx) for xx in x]
        elif type(x) in [np.ndarray, float, int]:
            return x
        elif x.requires_grad:
            return to_numpy(x.detach())
        else:
            if x.is_cuda:
                return x.cpu().numpy()
            else:
                return x.numpy()
else:
    def to_numpy(x):
        if type(x) in [np.ndarray, float, int]:
            return x
        elif isinstance(x, Variable):
            return to_numpy(x.data)
        else:
            if x.is_cuda:
                return x.cpu().numpy()
            else:
                return x.numpy()


# --
# From `fastai`

def get_children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_freeze(x, mode):
    x.frozen = mode
    for p in x.parameters():
        p.requires_grad = not mode

    for module in get_children(x):
        set_freeze(module, mode)


def apply_init(m, init_fn):
    def _cond_init(m, init_fn):
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'weight'):
                init_fn(m.weight)

            if hasattr(m, 'bias'):
                m.bias.data.fill_(0.)

    m.apply(lambda x: _cond_init(x, init_fn))


def get_num_features(model):
    children = get_children(model)
    if len(children) == 0:
        return None

    for layer in reversed(children):
        if hasattr(layer, 'num_features'):
            return layer.num_features

        res = get_num_features(layer)
        if res is not None:
            return res


def parameters_from_children(x, only_requires_grad=False):
    parameters = [list(c.parameters()) for c in get_children(x)]
    parameters = sum(parameters, [])
    if only_requires_grad:
        parameters = [p for p in parameters if p.requires_grad]
    return parameters
