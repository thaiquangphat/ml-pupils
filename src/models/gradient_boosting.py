import os
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.logger import get_logger

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

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

def train(dataloader, save_dir, args):
    """Train and tune an XGBoost model using Hyperopt for parameter tuning
    and early stopping to determine the optimal number of iterations.
    """
    args = {**DEFAULT_ARGS, **args}
    logger = get_logger('gradient_boosting')
    logger.info(f"Training arguments: {args}")

    os.makedirs(save_dir, exist_ok=True)

    X, y = dataset.images.reshape(dataset.images.shape[0], -1), dataset.labels

    logger.info("Using Hyperopt to search for best hyperparameters...")

    def objective(params):
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])

        model = xgb.XGBClassifier(
            n_estimators=args['n_estimators'],
            eval_metric=args['eval_metric'],
            objective=args['objective'],
            num_class=args['num_class'],
            tree_method=args['tree_method'],
            early_stopping_rounds=args['early_stopping_rounds'],
            **params
        )

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return {'loss': -accuracy, 'status': STATUS_OK}

    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'gamma': hp.uniform('gamma', 0, 5),
        'reg_alpha': hp.uniform('reg_alpha', 0, 100),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1)
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials
    )

    logger.info(f"Best parameters from Hyperopt: {best}")

    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])

    # Save best hyperparameters to JSON file
    hyperparam_path = save_dir / get_save_name("best_hyperparams", "json")
    with open(hyperparam_path, 'w') as f:
        json.dump(best, f, indent=4)
    logger.info(f"Best hyperparameters saved at {hyperparam_path}")

    final_model = xgb.XGBClassifier(
        n_estimators=args['n_estimators'],
        eval_metric=args['eval_metric'],
        objective=args['objective'],
        num_class=args['num_class'],
        tree_method=args['tree_method'],
        early_stopping_rounds=args['early_stopping_rounds'],
        **best
    )

    logger.info("Retraining best model on full dataset with early stopping to find best n_estimators...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info(f"Best iteration determined via early stopping: {final_model.best_iteration}")
    best_n_estimators = final_model.best_iteration

    retrained_model = xgb.XGBClassifier(
        n_estimators=best_n_estimators,
        eval_metric=args['eval_metric'],
        objective=args['objective'],
        num_class=args['num_class'],
        tree_method=args['tree_method'],
        **best
    )

    logger.info("Training final model on full data with optimal hyperparameters and n_estimators...")
    retrained_model.fit(X, y, verbose=False)

    save_path = save_dir / get_save_name("xgboost", "pkl")
    save_pkl(retrained_model, save_path)

    print(f"Model saved at {save_path}")
    logger.info(f"Model saved at {save_path}")

def evaluate(dataloader, saved_path, args):
    if not saved_path or not os.path.exists(saved_path):
        raise FileNotFoundError("Model not found. Please train first.")

    X, y = dataloader
    X = X.reshape(X.shape[0], -1)
    
    model = load_pkl(saved_path)
    print(f"Model loaded from {saved_path}")
    
    y_preds = model.predict(X)
    y_scores = model.predict_proba(X)
    
    return y, y_preds, y_scores


