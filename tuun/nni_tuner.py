"""
Tuun wrapped as a custom Tuner for NNI.
"""
from argparse import Namespace
import numbers
import copy
from nni.tuner import Tuner
from nni.utils import NodeType, OptimizeMode, extract_scalar_reward
from tuun.main import Tuun


def json2space(in_x, name=NodeType.ROOT):
    """
    Change json to search space in tuun

    Parameters
    ----------
    in_x : dict/list/str/int/float
        The part of json.
    name : str
        name could be NodeType.ROOT, NodeType.TYPE, NodeType.VALUE or NodeType.INDEX, NodeType.NAME.
    """
    out_y = copy.deepcopy(in_x)
    if isinstance(in_x, dict):
        if NodeType.TYPE in in_x.keys():
            _type = in_x[NodeType.TYPE]
            name = name + '-' + _type
            _value = json2space(in_x[NodeType.VALUE], name=name)
            if any(isinstance(x, int) for x in _value) and \
                any(isinstance(x, float) for x in _value):
                _value = [float(x) for x in _value]
            print('-----', _value)
            if all(isinstance(x, type(_value[0])) for x in _value):
                if _type == 'choice':
                    if isinstance(_value[0], (int, float)):
                        out_y = {'name': 'list', 'domain_list': [[x] for x in _value]}
                    else:
                        out_y = {'name': 'list', 'domain_list': _value}
                elif _type == 'uniform':
                    out_y = {'name': 'real', 'domain_list': [_value]}
                #elif _type == 'randint': # to do
                else:
                    raise RuntimeError(
                        'the search space type is not supported by tuun'
                    )
            else:
                raise RuntimeError(
                    '\'_value\' should have the same type of elements'
                )
        else:
            out_y = list()
            for key in in_x.keys():
                out_y.append(json2space(in_x[key], name + '[%s]' % str(key)))
    elif isinstance(in_x, list):
        out_y = list()
        for i, x_i in enumerate(in_x):
            if isinstance(x_i, dict):
                # if NodeType.NAME not in x_i.keys():
                #     raise RuntimeError(
                #         '\'_name\' key is not found in this nested search space.'
                #     )
                raise RuntimeError(
                    'nested search space is not supported by tuun'
                )
            out_y.append(json2space(x_i, name + '[%d]' % i))
    return out_y

class TuunTuner(Tuner):
    """
    Tuun as a custom Tuner for NNI.
    """

    def __init__(self, tuun_config, initial_data=None):
        """
        Parameters
        ----------
        tuun_config : dict
            Config to specify Tuun options.
        initial_data : dict
            Dictionary with keys x (list) and y (1D numpy ndarray).
        """
        self._set_tuun(tuun_config)
        self._set_data(initial_data)
    def _set_tuun(self, tuun_config):
        """Configure and instantiate self.tuun."""
        self.tuun = Tuun(tuun_config)
        print('set tuun')

    def _set_data(self, initial_data):
        """Set self.data."""
        if initial_data is None:
            self.data = Namespace(x=[], y=[])
        else:
            initial_data = copy.deepcopy(initial_data)
            if isinstance(initial_data, dict):
                initial_data = Namespace(**initial_data)
                self.data = initial_data
            elif isinstance(initial_data, Namespace):
                self.data = initial_data
            else:
                raise TypeError(
                    'initial_data must be either a dict, Namespace, or None'
                )
        print('set data')

    def update_search_space(self, search_space):
        """
        Update search space. Input search_space contains information that the user
        pre-defines.

        Parameters
        ----------
        search_space : dict
            Information to define a search space.
        """
        print('==',self.tuun.config.domain_config)
        dom_config_list = json2space(search_space)
        print(dom_config_list)
        self.tuun.config.domain_config.update({'dom_config_list': dom_config_list})
        self.tuun._set_backend()

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Return a dict of trial (hyper-)parameters, as a serializable object.

        Parameters
        ----------
        parameter_id : int
            ID number for given (hyper-)parameters.

        Returns
        -------
        dict
            A set of (hyper-)parameters suggested by Tuun.
        """
        suggestion = self.tuun.suggest_to_minimize(self.data)
        parsed_dict = self._parse_suggestion_into_dict(suggestion)
        return parsed_dict

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Record the result from a trial.

        Parameters
        ----------
        parameter_id : int
            ID number for given (hyper-)parameters.
        parameters : dict
            A set of (hyper-)parameters, same format returned by generate_parameters().
        value : dict_or_float
            Final  evaluation/objective metric of the trial. If value is dict, it should
            have "default" key.
        """

        # Define y
        if isinstance(value, dict):
            y = float(value['default'])
        elif isinstance(value, numbers.Number):
            y = float(value)
        else:
            raise TypeError('value must be a Number or dict with "default" key')

        # Define x
        x = parameters['suggestion']

        # Update self.data
        self.data.x.append(x)
        self.data.y.append(y)

    def _parse_suggestion_into_dict(self, suggestion):
        """Parse suggestion from Tuun into dict for NNI."""

        # Keep things simple for now
        parsed_dict = {'suggestion': suggestion}

        return parsed_dict
