"""
Functions to define and compile Stan GP models. In this file: hierarchical GP (prior on
rho, alpha) using an kernel and fixed sigma.
"""

import time
import pickle
import pystan


def get_model(recompile=False, print_status=True):
    """Return stan model. Recompile model if recompile is True."""

    model_file_str = 'gpystan/stan/model_pkls/gp_fixedsig_ard.pkl'

    if recompile:
        starttime = time.time()
        model = pystan.StanModel(model_code=get_model_code())
        buildtime = time.time() - starttime
        with open(model_file_str, 'wb') as f:
            pickle.dump(model, f)
        if print_status:
            print('[INFO] Time taken to compile = ' + str(buildtime) + ' seconds.')
            print('[INFO] Stan model saved in file ' + model_file_str)
    else:
        model = pickle.load(open(model_file_str, 'rb'))
        if print_status:
            print('[INFO] Stan model loaded from file {}'.format(model_file_str))
    return model


def get_model_code():
    """Parse modelp and return stan model code."""

    return """
    functions {
        matrix cov_matrix(int N, int D, vector[] x, vector ls, real alpha_sq, int cov_idx) {
          matrix[N,N] S;
          real dist_sum;
          real sqrt3;
          real sqrt5;
          sqrt3=sqrt(3.0);
          sqrt5=sqrt(5.0);

          // Matern as nu->Inf become Gaussian (aka squared exponential cov)
          if (cov_idx==1) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_sum = 0;
                for(d in 1:D) {
                  dist_sum = dist_sum + square(x[i][d] - x[j][d]) / square(ls[d]);
                }
                S[i,j] = alpha_sq * exp( -0.5 * dist_sum);
              }
            }
          }

          // fill upper triangle
          for(i in 1:(N-1)) {
            for(j in (i+1):N) {
              S[j,i] = S[i,j];
            }
          }

          // create diagonal: nugget(nonspatial) + spatial variance +  eps ensures positive definiteness
          for(i in 1:N) {
            S[i,i] = alpha_sq;
          }

          return S;
        }
    }

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
        int covid;
    }

    parameters {
        vector<lower=0>[D] rho;
        real<lower=0> alpha;
    }

    model {
        matrix[N, N] cov = cov_matrix(N, D, x, rho, square(alpha), covid)
                           + diag_matrix(rep_vector(square(sigma), N));
        matrix[N, N] L_cov = cholesky_decompose(cov);
        for(d in 1:D) {
          rho[d] ~ inv_gamma(ig1, ig2);
        }
        alpha ~ normal(n1, n2);
        y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }
    """


if __name__ == '__main__':
    get_model()
