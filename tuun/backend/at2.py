from .core import Backend
from .regression_model import RegressionModel


class AT2Backend(Backend):
    def __init__(self,
                 model_config=None,
                 acqfunction_config=None,
                 acqoptimizer_config=None,
                 domain_config=None,
                 at2_config=None):
        self.model_config = model_config
        self.acqfunction_config = acqfunction_config
        self.acqoptimizer_config = acqoptimizer_config
        self.domain_config = domain_config
        self.at2_config = at2_config

    def minimize_function(
        self,
        f,
        n_iter,
        data,
        data_update_fun=None,
        verbose=True,
        seed=None
    ):
        data = self._convert_tuun_data_to_at2(data)

    def suggest_to_minimize(self):
        raise NotImplementedError

    def _convert_tuun_data_to_at2(self, data):
        pass

    def _get_model(self, data):
        self.regressor = RegressionModel(self.model_config['transfer_model'])
        val_accurracies = []
        for i, reg in enumerate(self.regressor.model_fnames):
            val_acc = self.regressor.evaluate_model(reg, data['X'])
            val_accurracies.append((val_acc, i))
        val_accurracies.sort(reverse=True)
        self.regressor.set_best_model(val_accurracies[0][1])

    def _get_acqfunction(self):
        raise NotImplementedError

    def _get_acqoptimizer(self):
        raise NotImplementedError
