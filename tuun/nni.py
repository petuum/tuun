"""
Tuun wrapped as a custom Tuner for NNI.
"""
from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward

from .main import Tuun


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

    def _set_data(initial_data):
        """Set self.data."""
        if initial_data is None:
            self.data = {x:[], y:np.array([])}  #  TODO
        else:
            self.data = initial_data

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
        suggestion = self.tuun.suggest_to_minimize(self.data)
        #  TODO : parse suggestion into dict, with correct format for NNI
        #  TODO : return the parsed dict

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
        #  TODO : define y: if value is dict, convert to float.
        #  TODO : define x parse parameters, and convert to correct format for x.
        #  TODO : update self.data.x and self.data.y
