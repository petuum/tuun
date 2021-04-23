"""
Tests for list (discrete set) domains.
"""

from argparse import Namespace
import numpy as np

from tuun.domain.list import ListDomain


def test_listdomain_is_in_domain():
    """Test if domain points are in domain (using set_domain_list)."""
    dom_params = Namespace(set_domain_list=True, domain_list=[1.0, 'cat', (5, 5)])
    pts = ['cat', (5.0, 5.0), 1, '1', (5, 6), 'dog']
    bool_list = [True, True, True, False, False, False]
    listdom = ListDomain(params=dom_params)
    in_dom_list = [listdom.is_in_domain(pt) for pt in pts]
    assert in_dom_list == bool_list


def test_listdomain_is_in_domain_auto():
    """Test if domain points are in domain (using set_domain_list_auto)."""
    dom_params = Namespace(
        set_domain_list_auto=True,
        domain_list_exec_str=('self.domain_list = ' + '[1., "cat", (5, 5)]'),
    )
    pts = ['cat', (5.0, 5.0), 1, '1', (5, 6), 'dog']
    bool_list = [True, True, True, False, False, False]
    listdom = ListDomain(params=dom_params)
    in_dom_list = [listdom.is_in_domain(pt) for pt in pts]
    assert in_dom_list == bool_list


def test_listdomain_is_in_domain_manual_set():
    """Test if domain points are in domain (manually set domain_list)."""
    dom_params = Namespace()
    domain_list = [1.0, 'cat', (5, 5)]
    pts = ['cat', (5.0, 5.0), 1, '1', (5, 6), 'dog']
    bool_list = [True, True, True, False, False, False]
    listdom = ListDomain(params=dom_params)
    listdom.set_domain_list(domain_list)
    in_dom_list = [listdom.is_in_domain(pt) for pt in pts]
    assert in_dom_list == bool_list


def test_listdomain_unif_rand_samp():
    """Test uniform random sampling of domain points."""
    dom_params = Namespace(set_domain_list=True, domain_list=[1.0, 'cat', (5, 5)])
    listdom = ListDomain(params=dom_params)
    nsamp = 10
    samp_list = listdom.unif_rand_sample(n=nsamp)
    assert isinstance(samp_list, list)
    assert len(samp_list) == nsamp
    in_dom_list = [listdom.is_in_domain(pt) for pt in samp_list]
    assert in_dom_list == [True for pt in in_dom_list]
