"""
Tests for Tuun class.
"""

from tuun.main import Tuun 


def test_initialize():
    """Test initialize Tuun."""
    config = {
        'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    }
    tu = Tuun(config)
    assert getattr(tu, 'config', None)
    assert getattr(tu.config, 'seed', None)


def test_initialize_with_search_space_list():
    """Test initialize Tuun."""
    config = {
        'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    }
    tu = Tuun(config)
    assert tu.config.domain_config.get('min_max') != [[0.0, 2.0]]

    search_space_list = [('real', [[0.0, 2.0]])]
    tu.set_config_from_list(search_space_list)

    assert tu.config.domain_config.get('min_max') is None
    assert tu.config.domain_config['dom_config_list'][0]['min_max'] == [[0.0, 2.0]]


def test_suggest_to_minimize():
    """Test Tuun suggest_to_minimize on a dataset."""
    config = {
        'acqfunction_config': {'name': 'default', 'acq_str': 'ucb', 'n_gen': 500},
    }
    tu = Tuun(config)
    search_space_list = [('real', [[0.0, 2.0]])]
    tu.set_config_from_list(search_space_list)

    data = {
        'x': [[[0.25]], [[0.5]], [[0.75]], [[1.0]], [[1.25]], [[1.5]], [[1.75]]],
        'y': [8.0, 7.0, 6.0, 1.0, 4.0, 5.0, 6.0],
    }

    suggestion = tu.suggest_to_minimize(data)
    assert 0.75 < suggestion[0][0] < 1.25
