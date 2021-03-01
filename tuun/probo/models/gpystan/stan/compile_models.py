"""
Code to compile stan models
"""

import argparse

from gp import get_model as gp_get_model
from gp_fixedsig import get_model as gp_fixedsig_get_model
from gp_fixedsig_matern import get_model as gp_fixedsig_matern_get_model
from gp_fixedsig_ard import get_model as gp_fixedsig_ard_get_model
from gp_fixedsig_all import get_model as gp_fixedsig_all_get_model
# from gp_distmat import get_model as gp_distmat_get_model
# from gp_distmat_fixedsig import get_model as gp_distmat_fixedsig_get_model


def main(model_str):
    """
    Re-compile model specified by model_str.
    """
    if model_str == 'gp':
        model = gp_get_model(recompile=True)
    elif model_str == 'gp_fixedsig':
        model = gp_fixedsig_get_model(recompile=True)
    elif model_str == 'gp_distmat':
        model = gp_distmat_get_model(recompile=True)
    elif model_str == 'gp_distmat_fixedsig':
        model = gp_distmat_fixedsig_get_model(recompile=True)
    elif model_str == 'gp_fixedsig_matern':
        model = gp_fixedsig_matern_get_model(recompile=True)
    elif model_str == 'gp_fixedsig_ard':
        model = gp_fixedsig_ard_get_model(recompile=True)
    elif model_str == 'gp_fixedsig_all':
        model = gp_fixedsig_all_get_model(recompile=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stan model compiler')
    parser.add_argument(
        '-m', '--model_str', help='Model string', default='gp_fixedsig_all'
    )

    args = parser.parse_args()
    assert args.model_str in [
        'gp',
        'gp_fixedsig',
        'gp_distmat',
        'gp_distmat_fixedsig',
        'gp_fixedsig_matern',
        'gp_fixedsig_ard',
        'gp_fixedsig_all',
    ]
    print('Compiling Stan model: {}'.format(args.model_str))

    main(args.model_str)
