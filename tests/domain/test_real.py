"""
Tests for real (Euclidean) domains.
"""

from argparse import Namespace
import numpy as np

from tuun.domain.real import RealDomain


def test_realdomain_is_in_domain():
    """Test if domain points are in domain."""
    dom_params = Namespace(min_max=[(0, 2), (0, 2), (0, 2)])
    pts = np.array([[0., 0., 0.], [.1, .2, .3], [2., 2., 2.], [-1., 2., 2.],
                    [2., 3., 2.], [-1., 3., 4.]])
    bool_list = [True, True, True, False, False, False]
    realdom = RealDomain(params=dom_params)
    in_dom_list = [realdom.is_in_domain(pt) for pt in pts]
    assert in_dom_list == bool_list

def test_realdomain_unif_rand_samp():
    """Test uniform random sampling of domain points."""
    dom_params = Namespace(min_max=[(0, 2), (0, 2), (0, 2)])
    realdom = RealDomain(params=dom_params)
    nsamp = 10
    samp_list = realdom.unif_rand_sample(n=nsamp)
    assert isinstance(samp_list, list)
    assert len(samp_list) == nsamp
    in_dom_list = [realdom.is_in_domain(pt) for pt in samp_list]
    assert in_dom_list == [True for pt in in_dom_list]
