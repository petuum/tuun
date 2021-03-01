"""
Functions to define and compile Stan GP models. In this file: hierarchical GP (prior on
rho, alpha) using a matern kernel and fixed sigma.
"""

import time
import pickle
import pystan


def get_model(recompile=False, print_status=True):
    """Return stan model. Recompile model if recompile is True."""

    model_file_str = 'gpystan/stan/model_pkls/gp_fixedsig_matern.pkl'

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
        matrix distance_matrix_single(int N, vector[] x) {
          matrix[N, N] distmat;
          for(i in 1:(N-1)) {
            for(j in (i+1):N) {
              distmat[i, j] = square(distance(x[i], x[j]));
            }
          }
          return distmat;
        }

        matrix matern_covariance(int N, matrix dist, real ls, real alpha_sq, int COVFN) {
          matrix[N,N] S;
          real dist_ls; 
          real sqrt3;
          real sqrt5;
          sqrt3=sqrt(3.0);
          sqrt5=sqrt(5.0);
          
          // exponential == Matern nu=1/2 , (p=0; nu=p+1/2)
          if (COVFN==1) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/square(ls);
                S[i,j] = alpha_sq * exp(-0.5 * dist_ls);
              }
            }
          }

          // Matern nu= 3/2 covariance
          else if (COVFN==2) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
               dist_ls = fabs(dist[i,j])/ls; //UPDATE? square(ls)
               S[i,j] = alpha_sq * (1 + sqrt3 * dist_ls) * exp(-sqrt3 * dist_ls); //UPDATE? with 0.5
              }
            }
          }
          
          // Matern nu=5/2 covariance
          else if (COVFN==3) { 
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/ls; //UPDATE? square(ls)
                S[i,j] = alpha_sq * (1 + sqrt5 *dist_ls + 5* pow(dist_ls,2)/3) * exp(-sqrt5 *dist_ls); //UPDATE? with 0.5
              }
            }
          }

          // Matern as nu->Inf become Gaussian (aka squared exponential cov)
          else if (COVFN==4) {
            for(i in 1:(N-1)) {
              for(j in (i+1):N) {
                dist_ls = fabs(dist[i,j])/ls; //UPDATE? square(ls)
                S[i,j] = alpha_sq * exp( -pow(dist_ls,2)/2 ); //UPDATE? with 0.5
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
        real<lower=0> rho;
        real<lower=0> alpha;
    }

    model {
        matrix[N, N] distmat = distance_matrix_single(N, x);
        matrix[N, N] cov = matern_covariance(N, distmat, rho, square(alpha), covid);
        matrix[N, N] L_cov = cholesky_decompose(cov);
        rho ~ inv_gamma(ig1, ig2);
        alpha ~ normal(n1, n2);
        y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
    }
    """


if __name__ == '__main__':
    get_model()
