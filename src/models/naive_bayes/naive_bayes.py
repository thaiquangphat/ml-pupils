import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.testutils import print_tree_details
from utils.logger import get_logger

DEFAULT_ARGS = {
    'n_splits': 5
}