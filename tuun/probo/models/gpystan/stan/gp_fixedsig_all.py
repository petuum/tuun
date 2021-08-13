"""
Functions to define and compile Stan GP models. In this file: hierarchical GP (prior on
length-scale and output-scale) with fixed sigma.
"""

import time
import pickle
import pathlib
import pystan


def get_model(recompile=False, print_status=True):
    """Return stan model. Recompile model if recompile is True."""

    model_str = 'gp_fixedsig_all'

    base_path = pathlib.Path(__file__).parent
    relative_path_to_model = 'model_pkls/' + model_str + '.pkl'
    model_path = str((base_path / relative_path_to_model).resolve())

    if recompile:
        starttime = time.time()
        model = pystan.StanModel(model_code=get_model_code())
        buildtime = time.time() - starttime
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        if print_status:
            print('[INFO] Time taken to compile = ' + str(buildtime) + ' seconds.')
            print('[INFO] Stan model saved in file ' + model_path)
    else:
        model = pickle.load(open(model_path, 'rb'))
        if print_status:
            print('[INFO] Stan model loaded from file {}'.format(model_path))
    return model


def get_model_code():
    """Return stan model code."""

    return """
    functions {
        matrix cov_matrix_ard(int N, int D, vector[] x, vector ls, real alpha_sq, int cov_id) {
          matrix[N,N] S;
          real dist_sum;
          real sqrt3;
          real sqrt5;
          sqrt3=sqrt(3.0);
          sqrt5=sqrt(5.0);

          // For RBF ARD kernel
          if (cov_id == 1) {
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

          // Fill upper triangle
          for(i in 1:(N-1)) {
            for(j in (i+1):N) {
              S[j,i] = S[i,j];
            }
          }

          // Create diagonal
          for(i in 1:N) {
            S[i,i] = alpha_sq;
          }

          return S;
        }

        matrix distance_matrix_on_vectors(int N, vector[] x) {
          matrix[N, N] distmat;
          for(i in 1:(N-1)) {
            for(j in (i+1):N) {
              distmat[i, j] = square(distance(x[i], x[j]));
            }
          }
          return distmat;
        }

        matrix cov_matrix_matern(int N, matrix dist, real ls, real alpha_sq, int cov_id) {
          matrix[N,N] S;
          real dist_ls;
          real sqrt3;
          real sqrt5;
          sqrt3=sqrt(3.0);
          sqrt5=sqrt(5.0);

          // For Matern kernel with parameter nu=1/2 (i.e. absolute exponential kernel)
          if (cov_id == 2) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/square(ls);
                S[i,j] = alpha_sq * exp(-1 * dist_ls);
              }
            }
          }

          // For Matern kernel with parameter nu=3/2
          else if (cov_id == 3) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
               dist_ls = fabs(dist[i,j])/ls;
               S[i,j] = alpha_sq * (1 + sqrt3 * dist_ls) * exp(-sqrt3 * dist_ls);
              }
            }
          }

          // For Matern kernel with parameter nu=5/2
          else if (cov_id == 4) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/ls;
                S[i,j] = alpha_sq * (1 + sqrt5 * dist_ls + 5 * pow(dist_ls,2)/3) * exp(-sqrt5 * dist_ls);
              }
            }
          }

          // For Matern kernel with parameter nu tending to infinity (i.e. RBF kernel)
          else if (cov_id == 1) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/ls;
                S[i,j] = alpha_sq * exp( -0.5 * pow(dist_ls, 2) );
              }
            }
          }

          // Fill upper triangle
          for(i in 1:(N-1)) {
            for(j in (i+1):N) {
              S[j,i] = S[i,j];
            }
          }

          // Create diagonal
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
        int kernel_id;
    }

    parameters {
        real<lower=0> rho;
        vector<lower=0>[D] rhovec;
        real<lower=0> alpha;
    }

    model {
        int cov_id;
        matrix[N, N] cov;
        matrix[N, N] L_cov;
        matrix[N, N] distmat;

        // RBF kernel single lengthscale
        if (kernel_id == 1) {
            cov = cov_exp_quad(x, alpha, rho) + diag_matrix(rep_vector(square(sigma), N));
            L_cov = cholesky_decompose(cov);
            rho ~ inv_gamma(ig1, ig2);
            alpha ~ normal(n1, n2);
            y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
        }

        // Matern kernel single lengthscale
        else if (kernel_id >= 2 && kernel_id <= 4) {
            if (kernel_id == 2) { cov_id = 2; }
            if (kernel_id == 3) { cov_id = 3; }
            if (kernel_id == 4) { cov_id = 4; }

            distmat = distance_matrix_on_vectors(N, x);
            cov = cov_matrix_matern(N, distmat, rho, square(alpha), cov_id) + diag_matrix(rep_vector(square(sigma), N));
            L_cov = cholesky_decompose(cov);
            rho ~ inv_gamma(ig1, ig2);
            alpha ~ normal(n1, n2);
            y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
        }

        // RBF kernel with ARD (D-dimensional) lengthscale
        else if (kernel_id == 5) {
            cov_id = 1;
            cov = cov_matrix_ard(N, D, x, rhovec, square(alpha), cov_id) + diag_matrix(rep_vector(square(sigma), N));
            L_cov = cholesky_decompose(cov);
            for(d in 1:D) {
                rhovec[d] ~ inv_gamma(ig1, ig2);
            }
            alpha ~ normal(n1, n2);
            y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
        }

    }
    """


if __name__ == '__main__':
    get_model()
