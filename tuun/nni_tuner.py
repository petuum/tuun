"""
Tuun wrapped as a custom Tuner for NNI.
"""
from argparse import Namespace
import numbers
import copy
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from tuun.main import Tuun


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
        self._set_optimize_mode(tuun_config)
        self._set_data(initial_data)

    def _set_optimize_mode(self, tuun_config):
        """Configure the mode to choose to minimize or maximize."""
        assert isinstance(tuun_config, dict)
        self._optimize_mode = tuun_config.get('optimize_mode', 'min')
        print('optimize_mode:', self._optimize_mode)
        assert self._optimize_mode in ['min', 'max']

    def _set_tuun(self, tuun_config):
        """Configure and instantiate self.tuun."""
        self.tuun = Tuun(tuun_config)

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

    def update_search_space(self, search_space):
        """
        Update search space. Input search_space contains information that the user
        pre-defines.

        Parameters
        ----------
        search_space : dict
            Information to define a search space.
        """
        pass

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
        if self._optimize_mode == 'min':
            suggestion = self.tuun.suggest_to_minimize(self.data)
        else:  # self._optimize_mode is guaranteed as 'min' or 'max'.
            suggestion = self.tuun.suggest_to_maximize(self.data)
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
