"""
Code to compile stan models
"""

import argparse

import tuumbo.model.stan.gp as gp
import tuumbo.model.stan.gp_fixedsig as gp_fixedsig
#import tuumbo.model.stan.gp_distmat as gp_distmat
#import tuumbo.model.stan.gp_distmat_fixedsig as gp_distmat_fixedsig


def main(model_str):
    """
    Re-compile model specified by model_str.
    """
    if model_str=='gp':
        model = gp.get_model(recompile=True)
    elif model_str=='gp_fixedsig':
        model = gp_fixedsig.get_model(recompile=True)
    elif model_str=='gp_distmat':
        model = gp_distmat.get_model(recompile=True)
    elif model_str=='gp_distmat_fixedsig':
        model = gp_distmat_fixedsig.get_model(recompile=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stan model compiler')
    parser.add_argument('-m', '--model_str', help='Model string',
                        default='gp_fixedsig')

    args = parser.parse_args()
    assert args.model_str in ['gp', 'gp_fixedsig', 'gp_distmat',
                              'gp_distmat_fixedsig']
    print('Compiling Stan model: {}'.format(args.model_str))

    main(args.model_str)
