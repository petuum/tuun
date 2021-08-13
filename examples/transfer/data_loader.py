import numpy as np
import math


class DataLoader:
    def __init__(self,):
        pass

    def process_dataset(self, data):
        # sample_x = self.convert_config(data[0]['config'])
        X = np.zeros((len(data), 16))
        Y = np.zeros((len(data),))
        for i, da in enumerate(data):
            X[i, :] = self.convert_config(da['config'])
            Y[i] = da['val_statistics']['val_top5_accuracy']

        return X, Y

    def convert_config(self, config):
        x = np.zeros((16,))
        if config['optimizer'] == 'Adam':
            start_ind = 0
        elif config['optimizer'] == 'SGD':
            start_ind = 4
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")
        for k, v in config['optimizer_param'].items():
            if k == 'lr':
                x[start_ind] = self.normalize_log10(v, 1e-4, 1e-1)
            elif k == 'weight_decay':
                x[start_ind + 1] = self.normalize_log10(v, 1e-5, 1e-3)
            elif k == 'betas':
                x[start_ind + 2] = self.normalize_log10(v[0], 0.5, 0.999)
                x[start_ind + 3] = self.normalize_log10(v[1], 0.8, 0.999)
            elif k == 'momentum':
                x[start_ind + 2] = self.normalize_log10(v, 1e-3, 1)

        if config['scheduler'] == 'StepLR':
            start_ind = 7
        elif config['scheduler'] == 'ExponentialLR':
            start_ind = 9
        elif config['scheduler'] == 'CyclicLR':
            start_ind = 10
        elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
            start_ind = 13
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")
        for k, v in config['scheduler_param'].items():
            if k == 'step_size':
                x[start_ind] = self.normalize_decimal(v, 2, 20)
            elif k == 'gamma':
                if config['scheduler'] == 'StepLR':
                    x[start_ind + 1] = self.normalize_log10(v, 0.1, 0.5)
                elif config['scheduler'] == 'ExponentialLR':
                    x[start_ind] = self.normalize_log10(v, 0.85, 0.999)
                elif config['scheduler'] == 'CyclicLR':
                    x[start_ind] = self.normalize_log10(v, 0.1, 0.5)
            elif k == 'max_lr':
                x[start_ind + 1] = self.normalize_decimal(v, 1e-4*1.1, 1e-1*1.5)
            elif k == 'step_size_up':
                x[start_ind + 2] = self.normalize_decimal(v, 1, 10)
            elif k == 'T_0':
                x[start_ind] = self.normalize_decimal(v, 2, 20)
            elif k == 'T_mult':
                x[start_ind + 1] = self.normalize_decimal(v, 1, 4)
            elif k == 'eta_min':
                x[start_ind + 2] = self.normalize_decimal(v, 1e-4*0.5, 1e-1*0.9)

        return x

    def normalize_decimal(self, x, low, up):
        return (x-low)/(up-low)

    def normalize_log2(self, x, low, up):
        return self.normalize_decimal(math.log2(x), math.log2(low), math.log2(up))

    def normalize_log10(self, x, low, up):
        return self.normalize_decimal(math.log10(x), math.log10(low), math.log10(up))
