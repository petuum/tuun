#!/usr/bin/env python

"""
    basenet.py
"""

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .helpers import to_numpy, to_device
from .hp_schedule import HPSchedule

TORCH_VERSION_3 = '0.3' == torch.__version__[:3]


# --
# Helpers

def _set_train(x, mode):
    # !! Do we want to always turn off `training` mode when the layer is frozen?
    x.training = False if getattr(x, 'frozen', False) else mode
    for module in x.children():
        _set_train(module, mode)
    return x


def _clip_grad_norm(params, clip_grad_norm):
    clip_fn = torch.nn.utils.clip_grad_norm_ if not TORCH_VERSION_3 \
        else torch.nn.utils.clip_grad_norm
    for p in params:
        if isinstance(p, dict):
            clip_fn(p['params'], clip_grad_norm)
        else:
            clip_fn(p, clip_grad_norm)


class Metrics:
    @staticmethod
    def n_correct(output, target):
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]
        if isinstance(target, tuple) or isinstance(output, list):
            target = target[0]

        correct = (output.max(dim=-1)[1] == target).long().sum()
        return int(correct), int(target.shape[0])


# --
# Model

class BaseNet(nn.Module):
    def __init__(self, loss_fn=F.cross_entropy, verbose=False):
        super().__init__()

        self.loss_fn = loss_fn

        self.opt = None
        self.hp_scheduler = None
        self.hp = None

        self.progress = 0
        self.epoch = 0

        self.verbose = verbose
        self.device = None

    def to(self, device=None):
        self.device = device
        if not TORCH_VERSION_3:
            super().to(device=device)
        else:
            if device == 'cuda':
                self.cuda()
            elif device == 'cpu':
                self.cpu()
            else:
                raise Exception
        return self

    def deepcopy(self):
        _device = self.device
        del self.device
        new_self = deepcopy(self).to(_device)
        self.device = _device
        return new_self

    # --
    # Optimization

    def _filter_requires_grad(self, params):
        # User shouldn't be passing variables that don't require gradients
        if isinstance(params[0], dict):
            check = np.all([np.all([pp.requires_grad for pp in p['params']]) for p in params])
        else:
            check = np.all([p.requires_grad for p in params])

        if not check:
            warnings.warn((
                'BaseNet.init_optimizer: some variables do not require gradients. '
                'Ignoring them, but better to handle explicitly'
            ), RuntimeWarning)

        return params

    def init_optimizer(self, opt, params, hp_scheduler=None, clip_grad_norm=0, **kwargs):
        params = list(params)

        self.clip_grad_norm = clip_grad_norm
        self.hp_scheduler = hp_scheduler

        if hp_scheduler is not None:
            for hp_name, scheduler in hp_scheduler.items():
                assert hp_name not in kwargs.keys(), '%s in kwargs.keys()' % hp_name
                kwargs[hp_name] = scheduler(0)

        self.params = self._filter_requires_grad(params)
        self.opt = opt(self.params, **kwargs)
        self.set_progress(0)

    def set_progress(self, progress):
        self.progress = progress
        self.epoch = np.floor(progress)

        if self.hp_scheduler is not None:
            self.hp = dict([(hp_name, scheduler(progress))
                            for hp_name, scheduler in self.hp_scheduler.items()])
            HPSchedule.set_hp(self.opt, self.hp)

    # --
    # Training states

    def train(self, mode=True):
        """ have to override this function to allow more finegrained control """
        return _set_train(self, mode=mode)

    # --
    # Batch steps

    def train_batch(self, data, target, metric_fns=None, forward=None):
        assert self.opt is not None, "BaseNet: self.opt is None"
        assert self.loss_fn is not None, 'BaseNet: self.loss_fn is None'
        assert self.training, 'BaseNet: self.training == False'
        if forward is None:
            forward = self.forward

        self.opt.zero_grad()

        if TORCH_VERSION_3:
            data, target = Variable(data), Variable(target)

        data, target = to_device(data, self.device), to_device(target, self.device)

        output = forward(data)
        loss = self.loss_fn(output, target)
        loss.backward()

        if self.clip_grad_norm > 0:
            _clip_grad_norm(self.params, self.clip_grad_norm)

        self.opt.step()

        metrics = [m(output, target) for m in metric_fns] if metric_fns is not None else []
        return float(loss), metrics

    def eval_batch(self, data, target, metric_fns=None, forward=None):
        assert not self.training, 'BaseNet: self.training == True'
        if forward is None:
            forward = self.forward

        def _eval(data, target, metric_fns):
            data, target = to_device(data, self.device), to_device(target, self.device)

            output = forward(data)
            loss = self.loss_fn(output, target)

            metrics = [m(output, target) for m in metric_fns] if metric_fns is not None else []
            return float(loss), metrics

        if not TORCH_VERSION_3:
            with torch.no_grad():
                return _eval(data, target, metric_fns)
        else:
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            return _eval(data, target, metric_fns)

    # --
    # Epoch steps

    def _run_epoch(self, dataloaders, mode, batch_fn, set_progress, desc,
                   num_batches=np.inf, compute_acc=False, metric_fns=None):
        metric_fns = metric_fns if metric_fns is not None else []
        if compute_acc:
            warnings.warn((
                'BaseNet._run_epoch: use `metric_fns=["n_correct"]` instead of `compute_acc=True`'
            ), RuntimeWarning)
            metric_fns.append('n_correct')

        compute_acc = 'n_correct' in metric_fns
        metric_fns = [getattr(Metrics, m) for m in metric_fns]

        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='%s:%s' % (desc, mode))

            if hasattr(self, 'reset'):
                self.reset()

            correct, total, loss_hist = 0, 0, [None] * min(num_batches, len(loader))
            for batch_idx, (data, target) in gen:
                if batch_idx >= num_batches:
                    break

                if set_progress:
                    self.set_progress(self.epoch + batch_idx / len(loader))

                loss, metrics = batch_fn(data, target, metric_fns=metric_fns)

                loss_hist[batch_idx] = loss
                if compute_acc:
                    correct += metrics[0][0]
                    total += metrics[0][1]

                if self.verbose:
                    gen.set_postfix(**{
                        "acc": correct / total if compute_acc else -1.0,
                        "loss": loss,
                    })

            if self.verbose:
                gen.set_postfix(**{
                    "acc": correct / total if compute_acc else -1.0,
                    "last_10_loss": np.mean(loss_hist[-10:]),
                })

            if set_progress:
                self.epoch += 1

            return {
                "acc": float(correct / total) if compute_acc else -1.0,
                "loss": list(map(float, loss_hist)),
            }

    def train_epoch(self, dataloaders, mode='train', **kwargs):
        assert self.opt is not None, "BaseNet: self.opt is None"
        _ = self.train()
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            batch_fn=self.train_batch,
            set_progress=True,
            desc="train_epoch",
            **kwargs,
        )

    def eval_epoch(self, dataloaders, mode='val', **kwargs):
        _ = self.eval()
        return self._run_epoch(
            dataloaders=dataloaders,
            mode=mode,
            batch_fn=self.eval_batch,
            set_progress=False,
            desc="eval_epoch",
            **kwargs,
        )

    def predict(self, dataloaders, mode='val'):
        _ = self.eval()

        all_output, all_target = [], []

        loader = dataloaders[mode]
        if loader is None:
            return None
        else:
            gen = enumerate(loader)
            if self.verbose:
                gen = tqdm(gen, total=len(loader), desc='predict:%s' % mode)

            if hasattr(self, 'reset'):
                self.reset()

            for _, (data, target) in gen:
                if not TORCH_VERSION_3:
                    with torch.no_grad():
                        output = self(to_device(data, self.device)).cpu()
                else:
                    data = Variable(data, volatile=True)
                    output = self(to_device(data, self.device)).cpu()

                all_output.append(output)
                all_target.append(target)

        return torch.cat(all_output), torch.cat(all_target)

    def save(self, outpath):
        torch.save(self.state_dict(), outpath)

    def load(self, inpath):
        self.load_state_dict(torch.load(inpath))

    def save_checkpoint(self, outpath):
        ckpt = {
            "model_state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "epoch": self.epoch,
        }
        torch.save(ckpt, outpath)

    def load_checkpoint(self, inpath):
        ckpt = torch.load(inpath, map_location=self.device)
        self.load_state_dict(ckpt['model_state_dict'])
        self.opt.load_state_dict(ckpt['opt_state_dict'])
        self.set_progress(ckpt['epoch'])


class BaseWrapper(BaseNet):
    def __init__(self, net=None, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def forward(self, x):
        return self.net(x)
