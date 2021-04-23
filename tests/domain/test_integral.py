"""
Tests for integral domains.
"""

from argparse import Namespace
import numpy as np

from tuun.domain.integral import IntegralDomain


def test_integraldomain_defaults_is_in_domain():
    """Test IntegralDomain with default params."""
    pts_01 = [2, 2.0, 4, 4.0, np.array(2), np.array(2.0), np.array(4.0)]
    pts_02 = [2.1, 4.1, np.array(2.1)]
    domain = IntegralDomain()
    in_dom_list_01 = [domain.is_in_domain(pt) for pt in pts_01]
    in_dom_list_02 = [domain.is_in_domain(pt) for pt in pts_02]
    assert all(b is True for b in in_dom_list_01)
    assert all(b is False for b in in_dom_list_02)


def test_integraldomain_is_in_domain():
    """Test if domain points are in specified domain."""
    dom_params = Namespace(min_max=[(0, 5), (-3.2, 1)])
    pts_01 = np.array([[0, 1], [0.0, 1.0], [3, -3], [4, -2]])
    pts_02 = np.array(
        [[-1, 0], [6, 0], [2, -4], [2, 2], [2.2, 0], [2, 0.3], [2.2, 0.3]]
    )
    domain = IntegralDomain(params=dom_params)
    in_dom_list_01 = [domain.is_in_domain(pt) for pt in pts_01]
    in_dom_list_02 = [domain.is_in_domain(pt) for pt in pts_02]
    assert all(b is True for b in in_dom_list_01)
    assert all(b is False for b in in_dom_list_02)


def test_integraldomain_unif_rand_samp():
    """Test uniform random sampling of domain points."""
    dom_params = Namespace(min_max=[(0, 5), (-3.2, 1), (0, 2)])
    domain = IntegralDomain(params=dom_params)
    nsamp = 10
    samp_list = domain.unif_rand_sample(n=nsamp)
    assert isinstance(samp_list, list)
    assert len(samp_list) == nsamp
    in_dom_list = [domain.is_in_domain(pt) for pt in samp_list]
    assert in_dom_list == [True for pt in in_dom_list]
