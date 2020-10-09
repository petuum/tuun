"""
Tests for ProboBackend class.
"""

from tuun.backend import ProboBackend


def test_initialize():
    """Test initialize ProboBackend."""
    model_config = {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5}
    acqfunction_config = {'name': 'default', 'acq_str': 'ei'}
    acqoptimizer_config = {'name': 'default', 'max_iter': 200}
    domain_config = {'name': 'real', 'min_max': [[-5, 5]]}
    pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)
    assert getattr(pb, 'model_config', None)

def test_suggest_to_minimize():
    """Test ProboBackend suggest_to_minimize on a dataset."""
    model_config = {'name': 'simplegp', 'ls': 3.0, 'alpha': 1.5, 'sigma': 1e-5}
    acqfunction_config = {'name': 'default', 'acq_str': 'ei'}
    acqoptimizer_config = {'name': 'default', 'max_iter': 200}
    domain_config = {'name': 'real', 'min_max': [[0.0, 2.0]]}
    pb = ProboBackend(model_config, acqfunction_config, acqoptimizer_config, domain_config)

    data = {
        'x': [[0.5], [1.0], [1.5]],
        'y': [6.0, 1.0, 4.0],
    }

    suggestion = pb.suggest_to_minimize(data)
    assert isinstance(suggestion, list)
    assert 0.75 < suggestion[0] < 1.25
