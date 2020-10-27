"""
Code to define Bayesian neural network (BNN) models in Pyro.
"""

import numpy as np
import torch
import pyro
from pyro.distributions import Normal


class BNN:
    """
    Bayesian neural network in Pyro.
    """

    def __init__(self, likelihood_scale):
        self.likelihood_scale = likelihood_scale
        self.n_iter = 1000
        self.batch_size = 32
        self.learning_rate = 1e-1
        self.print_every = 1000
        self.layer_str = '2'
        self.layer_width = 50
        self.activation = torch.nn.ReLU()
        # self.activation = torch.nn.Tanh()

    def model(self, X, y):
        priors = dict()
        for n, p in self.nn.named_parameters():
            if self.layer_str + ".weight" in n:
                loc = torch.zeros_like(p)
                scale = torch.ones_like(p)
                priors[n] = Normal(loc=loc, scale=scale).to_event(1)
            elif self.layer_str + ".bias" in n:
                loc = torch.zeros_like(p)
                scale = torch.ones_like(p)
                priors[n] = Normal(loc=loc, scale=scale).to_event(1)
        lifted_module = pyro.random_module("module", self.nn, priors)
        lifted_reg_model = lifted_module()
        with pyro.plate(
            "map", len(X), subsample_size=min(X.shape[0], self.batch_size)
        ) as ind:
            pred_mean = lifted_reg_model(X[ind]).squeeze(-1)
            pyro.sample(
                "obs",
                Normal(pred_mean, torch.tensor(self.likelihood_scale)),
                obs=y[ind],
            )

    def guide(self, X, y):
        softplus = torch.nn.Softplus()
        priors = dict()
        for n, p in self.nn.named_parameters():
            if self.layer_str + ".weight" in n:
                loc = pyro.param("mu_" + n, 0.0 * torch.randn_like(p))
                scale = pyro.param(
                    "sigma_" + n,
                    softplus(torch.randn_like(p)),
                    constraint=torch.distributions.constraints.positive,
                )
                priors[n] = Normal(loc=loc, scale=scale).to_event(1)
            elif self.layer_str + ".bias" in n:
                loc = pyro.param("mu_" + n, 0.0 * torch.randn_like(p))
                scale = pyro.param(
                    "sigma_" + n,
                    softplus(torch.randn_like(p)),
                    constraint=torch.distributions.constraints.positive,
                )
                priors[n] = Normal(loc=loc, scale=scale).to_event(1)
        lifted_module = pyro.random_module("module", self.nn, priors)
        return lifted_module()

    def train(self, X, y):
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.layer_width),
            self.activation,
            torch.nn.Linear(self.layer_width, 1),
        )
        self.X = X
        self.y = y.reshape(X.shape[0])

        svi = pyro.infer.SVI(
            self.model,
            self.guide,
            pyro.optim.Adam({"lr": self.learning_rate}),
            loss=pyro.infer.Trace_ELBO(),
        )
        pyro.clear_param_store()

        self.rec = []
        for i in range(self.n_iter):
            loss = svi.step(self.X, self.y)
            self.rec.append(loss / X.shape[0])
            if (i + 1) % self.print_every == 0:
                print("[Iteration %05d] loss: %.4f" % (i + 1, loss / X.shape[0]))

    def sample_post(self):
        nn = self.guide(self.X, self.y)
        return nn

    def sample_pred_mean_np(self, nn, x):
        x = torch.tensor(x).float()
        return nn(x).detach().cpu().numpy()

    def pred_nsamp_np(self, nsamp, xmat, nn):
        y_all = []
        for sidx in range(nsamp):
            y_samp = [self.sample_pred_mean_np(nn, pt) for pt in xmat]
            y_all.append(np.array(y_samp).reshape(-1))

        return np.array(y_all)

    def postpred_nsamp_np(self, nsamp, xmat):
        y_all = []
        for sidx in range(nsamp):
            nn = self.sample_post()
            y_samp = [self.sample_pred_mean_np(nn, pt) for pt in xmat]
            y_all.append(np.array(y_samp).reshape(-1))

        return np.array(y_all)
