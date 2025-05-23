import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.logger import get_logger

DEFAULT_ARGS = {
    'n_splits': 5,
}
