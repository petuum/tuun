"""
Tests for DragonflyBackend class.
"""

from tuun.backend import DragonflyBackend


def test_initialize():
    """Test initialize DragonflyBackend."""
    domain_config = {'name': 'real', 'min_max': [[-5, 5]]}
    opt_config = {'name': 'real'}
    dragonfly_config = {'acq_str': 'ucb-ei', 'n_init_rs': 0}
    db = DragonflyBackend(domain_config, opt_config, dragonfly_config)
    assert getattr(db, 'domain_config', None)

def test_suggest_to_minimize():
    """Test DragonflyBackend suggest_to_minimize on a dataset."""
    domain_config = {'name': 'real', 'min_max': [[0.0, 2.0]]}
    opt_config = {'name': 'real'}
    dragonfly_config = {'acq_str': 'ucb-ei', 'n_init_rs': 0}
    db = DragonflyBackend(domain_config, opt_config, dragonfly_config)

    data = {
        'x': [[0.5], [1.0], [1.5]],
        'y': [6.0, 1.0, 4.0],
    }

    suggestion = db.suggest_to_minimize(data)
    assert 0.75 < suggestion[0] < 1.25
