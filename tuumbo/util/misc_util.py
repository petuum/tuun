"""
Miscellaneous utilities.
"""

from argparse import Namespace


def dict_to_namespace(params)
    """
    If params is a dict, convert it to a Namespace, and return it.
    
    Parameters
    ----------
    params : Namespace_or_dict
        Namespace or dict.

    Returns
    -------
    params : Namespace
        Namespace of params
    """
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)
    
    return params
