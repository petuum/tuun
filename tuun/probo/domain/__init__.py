"""
Code for domain classes.
"""

from .real import RealDomain
from .list import ListDomain
from .integral import IntegralDomain
from .product import ProductDomain
from ..util.misc_util import dict_to_namespace


def get_domain_from_params(dom_params, verbose=False):
    """Return Domain instance."""
    if dom_params.dom_str == 'real':
        return RealDomain(dom_params, verbose)
    elif dom_params.dom_str == 'list':
        return ListDomain(dom_params, verbose)
    elif dom_params.dom_str == 'int':
        return IntegralDomain(dom_params, verbose)


def get_domain_from_config(dom_params, verbose=False):
    """Return Domain instance."""
    dom_params = dict_to_namespace(dom_params)
    return get_domain_from_params(dom_params, verbose=verbose)
