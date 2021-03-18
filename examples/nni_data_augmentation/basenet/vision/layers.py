#!/usr/bin/env python

"""
    layers.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveMultiPool2d(nn.Module):
    def __init__(self, output_size=(1, 1), op_fns=[F.adaptive_avg_pool2d, F.adaptive_max_pool2d]):
        super(AdaptiveMultiPool2d, self).__init__()

        self.output_size = output_size
        self.op_fns = op_fns

    def forward(self, x):
        return torch.cat([op_fn(x, output_size=self.output_size) for op_fn in self.op_fns], dim=1)

    def __repr__(self):
        return 'AdaptiveMultiPool2d()'


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

    def __repr__(self):
        return 'Flatten()'
