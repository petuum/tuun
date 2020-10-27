"""
Classes for acqoptimizers implemented with scipy.optimize.
"""

from argparse import Namespace
import numpy as np
from scipy.optimize import minimize

from .acqopt import AcqOptimizer
from ..util.misc_util import dict_to_namespace


class SpoAcqOptimizer(AcqOptimizer):
    """AcqOptimizer using algorithms from scipy.optimize."""

    def __init__(self, params=None, domain=None, print_delta=False, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters.
        domain : Domain_or_Namespace_or_dict
            Domain instance, or Namespace/dict of parameters that specify one
            of the predefined Domains.
        print_delta : bool
            If True, print acquisition function deltas at each iteration.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        self.set_domain(domain)
        self.print_delta = print_delta
        self.set_verbose(verbose)

    def set_params(self, params):
        """Set self.params."""
        params = dict_to_namespace(params)

        self.params = params
        self.params.init_str = getattr(params, 'init_str', 'bsf')  # init type
        self.params.rand_every = getattr(params, 'rand_every', None)  # for bsf
        self.params.n_rand = getattr(params, 'n_rand', 1)  # for bsf_rand
        self.params.k = getattr(params, 'k', 2)  # for topk
        self.params.max_iter = getattr(params, 'max_iter', 1000)  # for spo.minimize
        self.params.rhobeg = getattr(params, 'rhobeg', 0.5)  # for spo.minimize
        self.params.jitter = getattr(params, 'jitter', False)  # to jitter init point
        self.params.jitter_val = getattr(params, 'jitter_val', 0.1)  # init jitter amt
        self.params.domain_params = getattr(params, 'domain_params', None)  # TODO
        self.params.n_init_rs = getattr(params, 'n_init_rs', 0)

        assert self.params.init_str in ['bsf', 'bsf_rand', 'init_opt', 'topk']

    def set_verbose(self, verbose):
        """Set verbose options."""
        self.verbose = verbose
        if self.verbose:
            self.print_str()

    def setup_optimize(self):
        """Setup for self.optimize method"""
        pass

    def optimize(self, acqmap, data):
        """Optimize acqfunction over x in domain"""

        # If there is no data, return a random sample from domain
        if data is None or data.x == []:
            return self.domain.unif_rand_sample(1)[0]

        # Optionally initialize with random search
        if len(data.x) <= self.params.n_init_rs:
            return self.domain.unif_rand_sample(1)[0]

        # Ensure correct format
        data.x = [np.array(xi).reshape(-1) for xi in data.x]
        data.y = np.array(data.y)

        # TODO: ensure data.y isn't empty
        bsf_point = data.x[data.y.argmin()]

        if self.params.init_str == 'bsf':
            # Initialization with best-so-far strategy

            if self.params.rand_every is None:
                self.params.rand_every = len(data.x) + 1

            if len(data.x) % self.params.rand_every == 0:
                init_point = self.domain.unif_rand_sample()[0]
            else:
                init_point = bsf_point

            init_point = self.possibly_apply_jitter(init_point)
            optima = self.run_spo_minimize(self.domain, acqmap, data, init_point)

        if self.params.init_str == 'bsf_rand':
            # Initialization with best-so-far and random-sampling strategy

            init_point_list = [bsf_point] + self.domain.unif_rand_sample(
                self.params.n_rand
            )
            init_point_list = self.possibly_apply_jitter(init_point_list)

            opt_list = [
                self.run_spo_minimize(self.domain, acqmap, data, ip)
                for ip in init_point_list
            ]

            min_idx = np.argmin([acqmap(opt) for opt in opt_list])
            optima = opt_list[min_idx]
            init_point = init_point_list[min_idx]

        if self.params.init_str == 'init_opt':
            # Initialization with explicit initialization to data.init_opt

            init_point = data.init_opt
            init_point = self.possibly_apply_jitter(init_point)

            optima = self.run_spo_minimize(self.domain, acqmap, data, init_point)

        if self.params.init_str == 'topk':
            # Initialization to top k best-so-far strategy

            idx_list = np.argsort(data.y)[: self.params.k]
            init_point_list = [data.x[idx] for idx in idx_list]
            init_point_list = self.possibly_apply_jitter(init_point_list)

            opt_list = [
                self.run_spo_minimize(self.domain, acqmap, data, ip)
                for ip in init_point_list
            ]

            min_idx = np.argmin([acqmap(opt) for opt in opt_list])
            optima = opt_list[min_idx]
            init_point = init_point_list[min_idx]

        optima = self.project_to_bounds(optima, self.domain)

        if self.print_delta:
            self.print_acq_delta(acqmap, init_point, optima)

        optima = self.post_process_optima(optima)

        return optima

    def possibly_apply_jitter(self, point_or_list):
        """Optionally return a jittered version of point or list."""
        if self.params.jitter is True:
            if type(point_or_list) is not list:
                point_or_list = self.get_jitter_point(point_or_list)
            else:
                point_or_list = [self.get_jitter_point(ip) for ip in point_or_list]
        return point_or_list

    def get_jitter_point(self, point):
        """Return a jittered version of point."""

        # Ensure point is a 1 dim numpy array
        point = np.array(point).reshape(-1)

        widths = [np.abs(mm[1] - mm[0]) for mm in self.domain.params.min_max]
        widths = [(w / 2) * self.params.jitter_val for w in widths]

        upper_bounds = point + np.array(widths)
        lower_bounds = point - np.array(widths)

        point_mod = np.array(
            [
                np.random.uniform(lower_bounds[i], upper_bounds[i], 1)[0]
                for i in range(len(point))
            ]
        )
        return point_mod

    def post_process_optima(self, optima):
        """Run post processing for found optima."""

        # Cast optima to list
        optima = list(optima)

        return optima

    def print_acq_delta(self, acqmap, init_point, optima):
        """Print acquisition function delta for optima minus initial point."""
        init_acq = acqmap(init_point)
        final_acq = acqmap(optima)
        acq_delta = final_acq - init_acq
        print(
            ('  Acq delta: {:.7f} = (final acq - init acq) ' + '[spo]').format(
                acq_delta
            )
        )

    def run_spo_minimize(self, real_dom, acqmap, data, init_point):
        """Use scipy.optimize to minimize acqmap over a RealDomain."""

        # Ensure init point is numpy array
        init_point = np.array(init_point).reshape(-1)

        # Set constraints
        constraints = []
        for i, tup in enumerate(real_dom.params.min_max):
            lo = {'type': 'ineq', 'fun': (lambda x, i: x[i] - tup[0]), 'args': (i,)}
            up = {'type': 'ineq', 'fun': (lambda x, i: tup[1] - x[i]), 'args': (i,)}
            constraints.append(lo)
            constraints.append(up)

        # Optimize with minimize function
        ret = self.call_minimize(acqmap, init_point, constraints)

        return ret.x

    def call_minimize(self, acqmap, init_point, constraints):
        """Call minimize function. Implement in child class."""
        raise ValueError('Implement call_minimize in child class.')

    def project_to_bounds(self, optima, real_dom):
        """Project (constrain) optima to within bounds of real_dom."""
        for i, tup in enumerate(real_dom.params.min_max):
            if optima[i] < tup[0]:
                optima[i] = tup[0]
            elif optima[i] > tup[1]:
                optima[i] = tup[1]

        return optima

    def print_str(self):
        """Print a description string."""
        print('*[INFO] ' + str(self))

    def __str__(self):
        return f'SpoAcqOptimizer with params={self.params}'


class CobylaAcqOptimizer(SpoAcqOptimizer):
    """AcqOptimizer using COBYLA algorithm (scipy implementation)."""

    def call_minimize(self, acqmap, init_point, constraints):
        """Call minimize function."""
        return minimize(
            acqmap,
            x0=init_point,
            constraints=constraints,
            method='COBYLA',
            options={
                'rhobeg': self.params.rhobeg,
                'maxiter': self.params.max_iter,
                'disp': False,
                'catol': 0.0,
            },
        )

    def __str__(self):
        return f'CobylaAcqOptimizer with params={self.params}'


class NelderMeadAcqOptimizer(SpoAcqOptimizer):
    """AcqOptimizer using Nelder-Mead algorithm (scipy implementation)."""

    def call_minimize(self, acqmap, init_point, constraints):
        """Call minimize function."""
        return minimize(
            acqmap,
            x0=init_point,
            method='Nelder-Mead',
            options={'adaptive': True, 'maxiter': self.params.max_iter},
        )

    def __str__(self):
        return f'NelderMeadAcqOptimizer with params={self.params}'
