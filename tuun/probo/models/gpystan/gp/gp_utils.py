"""
Utilities for Gaussian process (GP) inference.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, Matern


def kern_exp_quad(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel).
    """
    return alpha ** 2 * kern_exp_quad_noscale(xmat1, xmat2, ls)


def kern_exp_quad_noscale(xmat1, xmat2, ls):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel), without scale parameter.
    """
    use_sklearn = True

    if use_sklearn:
        kernel = RBF(length_scale=ls)
        return kernel(xmat1, xmat2)
    else:
        sq_norm = (-1 / (2 * ls ** 2)) * cdist(xmat1, xmat2, 'sqeuclidean')
        return np.exp(sq_norm)


def squared_euc_distmat(xmat1, xmat2, coef=1.0):
    """
    Distance matrix of squared euclidean distance (multiplied by coef) between
    points in xmat1 and xmat2.
    """
    return coef * cdist(xmat1, xmat2, 'sqeuclidean')


def kern_distmat(xmat1, xmat2, ls, alpha, distfn):
    """
    Kernel for a given distmat, via passed in distfn (which is assumed to be fn
    of xmat1 and xmat2 only).
    """
    distmat = distfn(xmat1, xmat2)
    sq_norm = -distmat / ls ** 2
    return alpha ** 2 * np.exp(sq_norm)


def kern_matern(xmat1, xmat2, ls, alpha, nu=0.5):
    """Matern kernel."""
    return alpha ** 2 * kern_matern_noscale(xmat1, xmat2, ls, nu)


def kern_matern_noscale(xmat1, xmat2, ls, nu=0.5):
    """Matern kernel without scale parameter."""
    kernel = Matern(length_scale=ls, nu=nu)
    return kernel(xmat1, xmat2)


def get_cholesky_decomp(k11_nonoise, sigma, psd_str):
    """Return cholesky decomposition."""
    if psd_str == 'try_first':
        k11 = k11_nonoise + sigma ** 2 * np.eye(k11_nonoise.shape[0])
        try:
            return stable_cholesky(k11, False)
        except np.linalg.linalg.LinAlgError:
            return get_cholesky_decomp(k11_nonoise, sigma, 'project_first')
    elif psd_str == 'project_first':
        k11_nonoise = project_symmetric_to_psd_cone(k11_nonoise)
        return get_cholesky_decomp(k11_nonoise, sigma, 'is_psd')
    elif psd_str == 'is_psd':
        k11 = k11_nonoise + sigma ** 2 * np.eye(k11_nonoise.shape[0])
        return stable_cholesky(k11)


def stable_cholesky(mmat, make_psd=True):
    """Return a 'stable' cholesky decomposition of mmat."""
    if mmat.size == 0:
        return mmat
    try:
        lmat = np.linalg.cholesky(mmat)
    except np.linalg.linalg.LinAlgError as e:
        if not make_psd:
            raise e
        diag_noise_power = -11
        max_mmat = np.diag(mmat).max()
        diag_noise = np.diag(mmat).max() * 1e-11
        break_loop = False
        while not break_loop:
            try:
                lmat = np.linalg.cholesky(
                    mmat + ((10 ** diag_noise_power) * max_mmat) * np.eye(mmat.shape[0])
                )
                break_loop = True
            except np.linalg.linalg.LinAlgError:
                if diag_noise_power > -9:
                    print(
                        '\tstable_cholesky failed with '
                        'diag_noise_power=%d.' % (diag_noise_power)
                    )
                diag_noise_power += 1
            if diag_noise_power >= 5:
                print(
                    '\t***** stable_cholesky failed: added diag noise '
                    '= %e' % (diag_noise)
                )
    return lmat


def project_symmetric_to_psd_cone(mmat, is_symmetric=True, epsilon=0):
    """Project symmetric matrix mmat to the PSD cone."""
    if is_symmetric:
        try:
            eigvals, eigvecs = np.linalg.eigh(mmat)
        except np.linalg.LinAlgError:
            print('\tLinAlgError encountered with np.eigh. Defaulting to eig.')
            eigvals, eigvecs = np.linalg.eig(mmat)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
    else:
        eigvals, eigvecs = np.linalg.eig(mmat)
    clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
    return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def solve_lower_triangular(amat, b):
    """Solves amat*x=b when amat is lower triangular."""
    return solve_triangular_base(amat, b, lower=True)


def solve_upper_triangular(amat, b):
    """Solves amat*x=b when amat is upper triangular."""
    return solve_triangular_base(amat, b, lower=False)


def solve_triangular_base(amat, b, lower):
    """Solves amat*x=b when amat is a triangular matrix."""
    if amat.size == 0 and b.shape[0] == 0:
        return np.zeros((b.shape))
    else:
        return solve_triangular(amat, b, lower=lower)


def sample_mvn(mu, covmat, nsamp):
    """
    Sample from multivariate normal distribution with mean mu and covariance
    matrix covmat.
    """
    mu = mu.reshape(-1,)
    ndim = len(mu)
    lmat = stable_cholesky(covmat)
    umat = np.random.normal(size=(ndim, nsamp))
    return lmat.dot(umat).T + mu


def gp_post(x_train, y_train, x_pred, ls, alpha, sigma, kernel, full_cov=True):
    """Compute parameters of GP posterior"""
    k11_nonoise = kernel(x_train, x_train, ls, alpha)
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y_train))
    k21 = kernel(x_pred, x_train, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(x_pred, x_pred, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
    if full_cov is False:
        k2 = np.sqrt(np.diag(k2))
    return mu2, k2
