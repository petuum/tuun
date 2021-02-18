#!/usr/bin/env python

"""
    hp_schedule.py
    
    Optimizer hyperparameter scheduler
    
    !! Most of the schedulers could be reimplemented as compound schedules (prod or cat)
"""

from __future__ import print_function, division

import sys
import copy
import warnings
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# --
# Helpers

def power_sum(base, k):
    return (base ** (k + 1) - 1) / (base - 1)


def inv_power_sum(x, base):
    return np.log(x * (base - 1) + 1) / np.log(base) - 1


def linterp(x, start_x, end_x, start_y, end_y):
    return start_y + (x - start_x) / (end_x - start_x) * (end_y - start_y)


def _set_hp(optimizer, hp_name, hp_hp):
    num_param_groups = len(list(optimizer.param_groups))
    
    if isinstance(hp_hp, float):
        hp_hp = [hp_hp] * num_param_groups
    else:
        assert len(hp_hp) == num_param_groups, ("len(%s) != num_param_groups" % hp_name)
    
    for i, param_group in enumerate(optimizer.param_groups):
        param_group[hp_name] = hp_hp[i]

def maybe_warn_kwargs(kwargs):
    if len(kwargs):
        warnings.warn("\n\nHPSchedule: unused arguments:\n %s \n\n" % str(kwargs), RuntimeWarning)

# --

class HPSchedule(object):
    
    @staticmethod
    def set_hp(optimizer, hp):
        for hp_name, hp_hp in hp.items():
            _set_hp(optimizer, hp_name, hp_hp)
    
    @staticmethod
    def constant(hp_max=0.1, **kwargs):
        maybe_warn_kwargs(kwargs)
        def f(progress):
            return hp_max
        
        return f
    
    @staticmethod
    def step(hp_max=0.1, breaks=(150, 250), factors=(0.1, 0.1), epochs=None, repeat=True):
        """ Step function learning rate annealing """
        assert len(breaks) == len(factors)
        breaks = np.array(breaks)
        
        def f(progress):
            if repeat:
                progress = progress % epochs
            
            return hp_max * np.prod(factors[:((progress >= breaks).sum())])
        
        return f
    
    @staticmethod
    def linear(hp_max=0.1, epochs=None, repeat=True):
        assert epochs is not None, "epochs is None"
        
        def f(progress):
            """ Linear learning rate annealing """
            
            if repeat:
                progress = progress % epochs
            
            return hp_max * (epochs - progress) / epochs
        
        return f
    
    @staticmethod
    def cyclical(hp_max=0.1, epochs=None, period_length=1, repeat=True):
        assert epochs is not None, "epochs is None"
        
        return HPSchedule.prod_schedule([
            HPSchedule.stepify(HPSchedule.linear(epochs=epochs, hp_max=hp_max, repeat=repeat)),
            HPSchedule.linear(epochs=period_length, hp_max=1, repeat=True),
        ])
    
    @staticmethod
    def linear_cycle(*args, **kwargs):
        raise Exception('!! Renamed to one_cycle')
    
    @staticmethod
    def one_cycle(hp_add=0.095, epochs=10, hp_init=0.0, hp_final=0.005, extra=5):
    #def one_cycle(hp_max=0.1, epochs=10, hp_init=0.0, hp_final=0.005, extra=5):
        def f(progress):
            if progress < epochs / 2:
                return 2 * (hp_final + hp_add) * (1 - (epochs - progress) / epochs)
            elif progress <= epochs:
                return hp_final + 2 * hp_add * (epochs - progress) / epochs
            elif progress <= epochs + extra:
                return hp_final * (extra - (progress - epochs)) / extra
            else:
                return hp_final / 10
        
        return f
    
    @staticmethod
    def piecewise_linear(breaks, vals):
        assert len(breaks) == len(vals)
        
        def _f(progress):
            if progress < breaks[0]:
                return vals[0]
            
            for i in range(1, len(breaks)):
                if progress < breaks[i]:
                    return linterp(progress, breaks[i - 1], breaks[i], vals[i - 1], vals[i])
            
            return vals[-1]
        
        def f(x):
            if isinstance(x, list) or isinstance(x, np.ndarray):
                return [_f(xx) for xx in x]
            else:
                return _f(x)
        
        return f
    
    @staticmethod
    def sgdr(hp_max=0.1, period_length=50, hp_min=0, t_mult=1):
        def f(progress):
            """ SGDR learning rate annealing """
            if t_mult > 1:
                period_id = np.floor(inv_power_sum(progress / period_length, t_mult)) + 1
                offsets = power_sum(t_mult, period_id - 1) * period_length
                period_progress = (progress - offsets) / (t_mult ** period_id * period_length)
            
            else:
                period_progress = (progress % period_length) / period_length
            
            return hp_min + 0.5 * (hp_max - hp_min) * (1 + np.cos(period_progress * np.pi))
        
        return f
    
    @staticmethod
    def burnin_sgdr(hp_init=0.1, burnin_progress=0.15, burnin_factor=100):
        sgdr = HPSchedule.sgdr(hp_init=hp_init, **kwargs)
        
        def f(progress):
            """ SGDR learning rate annealing, w/ constant burnin period """
            if progress < burnin_progress:
                return hp_init / burnin_factor
            else:
                return sgdr(progress)
        
        return f
    
    @staticmethod
    def exponential_increase(hp_init=0.1, hp_max=10, num_steps=100):
        mult = (hp_max / hp_init) ** (1 / num_steps)
        def f(progress):
            return hp_init * mult ** progress
            
        return f
    
    # --
    # Compound schedules
    
    @staticmethod
    def stepify(fn):
        def f(progress):
            progress = np.floor(progress)
            return fn(progress)
        
        return f
    
    @staticmethod
    def prod_schedule(fns):
        def f(progress):
            return np.prod([fn(progress) for fn in fns], axis=0)
        
        return f
        
    @staticmethod
    def cat_schedule(fns, breaks):
        # !! Won't work w/ np.arrays
        assert len(fns) - 1 == len(breaks)
        
        def f(progress):
            assert (isinstance(progess, float) or isinstance(progress, int))
            
            if progress < breaks[0]:
                return fns[0](progress)
            
            for i in range(1, len(breaks)):
                if progress < breaks[i]:
                    return fns[i-1](progress)
            
            return fns[-1](progress)
        
        return f

# --
# HP Finder

class HPFind(object):
    
    @staticmethod
    def find(model, dataloaders, hp_init=1e-5, hp_max=10, hp_mults=None, params=None, mode='train', smooth_loss=False):
        assert mode in dataloaders, '%s not in loader' % mode
        
        # --
        # Setup HP schedule
        
        if model.verbose:
            print('HPFind.find: copying model', file=sys.stderr)
        
        model = model.deepcopy()
        _ = model.train()
        
        if hp_mults is not None:
            hp_init *= hp_mults
            hp_max *= hp_mults # Correct?
        
        hp_scheduler = HPSchedule.exponential_increase(hp_init=hp_init, hp_max=hp_max, num_steps=len(dataloaders[mode]))
        
        if params is None:
            params = filter(lambda x: x.requires_grad, model.parameters())
        
        model.init_optimizer(
            opt=torch.optim.SGD,
            params=params,
            hp_scheduler={
                "lr" : hp_scheduler
            },
            momentum=0.9,
        )
        
        # --
        # Run epoch of training w/ increasing learning rate
        
        avg_mom  = 0.98 # For smooth_loss
        avg_loss = 0.   # For smooth_loss
        
        hp_hist, loss_hist = [], []
        
        gen = enumerate(dataloaders[mode])
        if model.verbose:
            gen = tqdm(gen, total=len(dataloaders[mode]), desc='HPFind.find:')
        
        for batch_idx, (data, target) in gen:
            
            model.set_progress(batch_idx)
            
            loss, _ = model.train_batch(data, target)
            
            if smooth_loss:
                avg_loss    = avg_loss * avg_mom + loss * (1 - avg_mom)
                debias_loss = avg_loss / (1 - avg_mom ** (batch_idx + 1))
                loss_hist.append(debias_loss)
            else:
                loss_hist.append(loss)
            
            if model.verbose:
                gen.set_postfix(**{
                    "loss" : loss,
                })
            
            hp_hist.append(model.hp['lr'])
            
            if loss > np.min(loss_hist) * 4:
                break
        
        return np.vstack(hp_hist[:-1]), loss_hist[:-1]
    
    @staticmethod
    def get_optimal_hp(hp_hist, loss_hist, c=10, burnin=5):
        """
            For now, gets smallest loss and goes back an order of magnitude
            Maybe it'd be better to use the point w/ max slope?  Or not use smoothed estimate? 
        """
        hp_hist, loss_hist = hp_hist[burnin:], loss_hist[burnin:]
        
        min_loss_idx = np.array(loss_hist).argmin()
        min_loss_hp = hp_hist[min_loss_idx]
        opt_hp = min_loss_hp / c
        
        if len(opt_hp) == 1:
            opt_hp = opt_hp[0]
        
        return opt_hp


if __name__ == "__main__":
    from rsub import *
    from matplotlib import pyplot as plt
    
    # Step
    # hp = HPSchedule.step(hp_max=np.array([1, 2]), factors=(0.5, 0.5), breaks=(10, 20), epochs=30)
    # hps = np.vstack([hp(i) for i in np.arange(0, 30, 0.01)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # Linear
    # hp = HPSchedule.linear(epochs=30, hp_max=0.1)
    # hps = np.vstack([hp(i) for i in np.arange(0, 30, 0.01)])
    # _ = plt.plot(hps)
    # show_plot()
    
    # # Linear cycle
    # hp = HPSchedule.one_cycle(epochs=30, hp_max=0.1, extra=10)
    # hps = np.vstack([hp(i) for i in np.arange(0, 40, 0.01)])
    # _ = plt.plot(hps)
    # show_plot()
    
    # Piecewise linear
    # vals = [    
    #     np.array([0.1, 0.5, 1.0]) * 0.0,
    #     np.array([0.1, 0.5, 1.0]) * 1.0,
    #     np.array([0.1, 0.5, 1.0]) * 0.5,
    # ]
    # hp = HPSchedule.piecewise_linear(breaks=[0, 0.5, 1], vals=vals)
    # hps = np.vstack([hp(i) for i in np.arange(-1, 2, 0.01)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # _ = plt.plot(hps[:,2])
    # show_plot()
    
    # Cyclical
    # hp = HPSchedule.cyclical(epochs=30, hp_max=0.1)
    # hps = np.vstack([hp(i) for i in np.arange(0, 40, 0.01)])
    # _ = plt.plot(hps)
    # show_plot()
    
    # SGDR
    # hp = HPSchedule.sgdr(period_length=10, t_mult=2, hp_max=np.array([1, 2]))
    # hps = np.vstack([hp(i) for i in np.arange(0, 70, 0.01)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # show_plot()
    
    # # Product
    # hp = HPSchedule.prod_schedule([
    #     HPSchedule.stepify(HPSchedule.linear(epochs=30, hp_max=0.1)),
    #     HPSchedule.linear(epochs=1, hp_max=1),
    # ])
    # hps = np.vstack([hp(i) for i in np.arange(0, 30, 0.01)])
    # _ = plt.plot(hps)
    # show_plot()
    
    # exponential increase (for setting learning rates)
    # hp = HPSchedule.exponential_increase(hp_init=np.array([1e-5, 1e-4]), hp_max=10, num_steps=100)
    # hps = np.vstack([hp(i) for i in np.linspace(0, 100, 1000)])
    # _ = plt.plot(hps[:,0])
    # _ = plt.plot(hps[:,1])
    # _ = plt.yscale('log')
    # show_plot()
