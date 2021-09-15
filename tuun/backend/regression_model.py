import os
import joblib
import tarfile
from urllib.request import urlretrieve
import uuid


class RegressionModel:
    """
    Generic regression model class for predicting the prior mean for a GP model.
    """
    def __init__(self, transfer_config):
        self.model_types = {'sklearn.linear_model.Ridge'}
        self.best_model = None
        self.model_type = transfer_config['model_type']
        self.uid = uuid.uuid4()
        self.tmp_dir = f"/tmp/{str(self.uid)}"
        self.tmp_tgz_dir = os.path.join(self.tmp_dir, 'tgz_files/')
        self.tmp_model_dir = os.path.join(self.tmp_dir, 'models/')
        os.makedirs(self.tmp_tgz_dir)
        os.mkdir(self.tmp_model_dir)

        if 'remote_url' in transfer_config.keys():
            tar_path = os.path.join(self.tmp_tgz_dir, 'reg_tar.tgz')
            filename, headers = urlretrieve(transfer_config['remote_url'], tar_path)
        else:
            filename = transfer_config['local_path']
        tarball = tarfile.open(filename)
        tarball.extractall(self.tmp_model_dir)

        self.model_fnames = os.listdir(self.tmp_model_dir)
        if len(self.model_fnames) == 1:
            if not os.path.isfile(os.path.join(self.tmp_model_dir, self.model_fnames[0])):
                self.tmp_model_dir = os.path.join(self.tmp_model_dir, self.model_fnames[0])
                self.model_fnames = os.listdir(self.tmp_model_dir)

    def evaluate_model(self, model_name, X):
        model_path = os.path.join(self.tmp_model_dir, model_name)
        model = self.load_model(model_path)
        # TODO: Support custom regression models
        if 'sklearn' in self.model_type:
            return model.predict(X)

    def set_best_model(self, file_ind):
        model_name = self.model_fnames[file_ind]
        model_path = os.path.join(self.tmp_model_dir, model_name)
        self.best_model = self.load_model(model_path)

    def load_model(self, model_path):
        # TODO: Support custom regression models
        if 'sklearn' in self.model_type:
            model = joblib.load(model_path)
            return model
        else:
            raise NotImplementedError

    def __call__(self, X):
        if self.best_model is None:
            raise ValueError("Regression model not assigned")

        if 'sklearn' in self.model_type:
            return self.best_model.predict(X)
        else:
            return self.best_model(X)
