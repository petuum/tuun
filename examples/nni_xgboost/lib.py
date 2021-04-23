from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from functools import partial