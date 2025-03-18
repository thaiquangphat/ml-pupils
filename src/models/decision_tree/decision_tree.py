import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, GridSearchCV
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.testutils import print_tree_details
from utils.logger import get_logger

DEFAULT_ARGS = {
    'n_splits': 5,
    'criterion': ['gini', 'entropy'],
    'max_features': [50,100,1000],
    'min_samples_leaf': [100,500],
}
