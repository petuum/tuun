"""
Functions to define and compile the Stan model: hierarchical GP (prior on rho, alpha),
using a squared exponential kernel, and fixed sigma.
"""

import time
import pickle
import pathlib
import pystan


def get_model(recompile=False, print_status=True):
    """Return stan model. Recompile model if recompile is True."""

    model_str = 'gp_fixedsig'

    base_path =  pathlib.Path(__file__).parent
    relative_path_to_model = 'model_pkls/' + model_str + '.pkl'
    model_path = str((base_path / relative_path_to_model).resolve())

    if recompile:
        starttime = time.time()
        model = pystan.StanModel(model_code=get_model_code())
        buildtime = time.time() - starttime
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        if print_status:
            print('*Time taken to compile = ' + str(buildtime) + ' seconds.\n-----')
            print('*Stan model saved in file ' + model_path + '.\n-----')
    else:
        model = pickle.load(open(model_path, 'rb'))
        if print_status:
            print('*Stan model loaded from file {}'.format(model_path))
    return model


def get_model_code():
    """Parse modelp and return stan model code."""

    return """
    data {
        int<lower=1> D;
        int<lower=1> N;
        vector[D] x[N];
        vector[N] y;
        real<lower=0> ig1;
        real<lower=0> ig2;
        real<lower=0> n1;
        real<lower=0> n2;
        real<lower=0> sigma;
    }

    parameters {
        real<lower=0> rho;
        real<lower=0> alpha;
    }

    model {
        matrix[N, N] cov = cov_exp_quad(x, alpha, rho)
                           + diag_matrix(rep_vector(square(sigma), N));
        matrix[N, N] L_cov = cholesky_decompose(cov);
        rho ~ inv_gamma(ig1, ig2);
        alpha ~ normal(n1, n2);
        y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }
    """


if __name__ == '__main__':
    get_model()
