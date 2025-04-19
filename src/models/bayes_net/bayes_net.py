import os
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from utils.utils import save_pkl, load_pkl, get_save_name, discretize_data
from models.bayes_net.build_features import segment_and_extract_features
from tqdm import tqdm
from pathlib import Path
from visualize.visualization import visualize_bayesian_network
from utils.logger import get_logger
import time

logger = get_logger("bayes_net")

COLS = ['area', 'perimeter', 'eccentricity', 'solidity', 'contrast', 'homogeneity', 'energy', 'correlation']

DEFAULT_ARGS = {
    "discretize_features": True,
    "bins": 3,
    "estimator": "mle",
    "naive": False,
    "chunk_size": 1000,
    "output_dir": "features/output",
    "visualization": {
        "enabled": True,
        "type": "network",
        "output_file": "results/figures/bayes_net.png",
        "show": False
    }
}