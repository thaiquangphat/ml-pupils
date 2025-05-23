import os
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.logger import get_logger

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

from xgboost.callback import EarlyStopping
import xgboost as xgb
from .gradient_boosting import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
import json
warnings.filterwarnings("ignore")


# Using KFold cross validation
from sklearn.model_selection import KFold

# Encoding categorical features
from sklearn.preprocessing import LabelEncoder

# Memory management
import gc


# DEFAULT_ARGS = {
#     'n_folds': 5,
#     'n_estimators': 10000,
#     'objective': 'multiclass',
#     'num_class': 4,
#     'learning_rate': 0.01,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.1,
#     'subsample': 0.9,
#     'early_stopping_rounds': 100,
#     'eval_metric': 'multi_logloss',
#     'n_jobs': -1,
#     'verbose': 10
# }
DEFAULT_ARGS = {
    'n_folds': 5,
    'n_estimators': 10,
    'objective': 'multi:softmax',
    'num_class': 4,
    'early_stopping_rounds': 5,
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'tree_method': 'hist'
}