# pandas and numpy for data manipulation
from .gradient_boosting import *

def train(dataset, save_dir, args):
    """Train and test a light gradient boosting model using
    cross validation to select the optimal number of training iterations.
    """
    args = {**DEFAULT_ARGS, **args}
    logger = get_logger('gradient_boosting')
    logger.info(f"Training arguments: {args}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    train, targets = dataset.images, dataset.labels
    train = train.reshape(train.shape[0], -1)
    
    print("Gradient Boosting start training...")
    logger.info("Gradient Boosting start training...")

    kfold = KFold(n_splits=args['n_folds'])
    
    best_iterations = 0
    
    logger.info(f"Performing {args['n_folds']}-fold cross-validation to determine optimal iterations.")
    
    model = lgb.LGBMClassifier(
        n_estimators=args['n_estimators'],
        objective=args['objective'],
        num_class=args['num_class'],
        learning_rate=args['learning_rate'],
        reg_alpha=args['reg_alpha'],
        reg_lambda=args['reg_lambda'],
        subsample=args['subsample'],
        n_jobs=args['n_jobs']
    )
    
    for fold, (train_indices, valid_indices) in enumerate(kfold.split(train), start=1):
        train_features, train_targets = train[train_indices], targets[train_indices]
        valid_features, valid_targets = train[valid_indices], targets[valid_indices]

        logger.info(f"Fold {fold}: Training on {len(train_indices)} samples, validating on {len(valid_indices)} samples.")

        model.fit(
            X=train_features, y=train_targets,
            eval_metric=args['eval_metric'],
            eval_set=[(valid_features, valid_targets)],
            callbacks=[
                early_stopping(stopping_rounds=args['early_stopping_rounds']),
                log_to_logger(logger, period=args['verbose'])
            ]
        )


        logger.info(f"Fold {fold}: Best iteration = {model.best_iteration_}")
        best_iterations += model.best_iteration_

    best_iterations = int(best_iterations / kfold.n_splits)
    logger.info(f"Averaged best iteration over folds: {best_iterations}")
    
    model = lgb.LGBMClassifier(
        n_estimators=best_iterations,
        objective=args['objective'],
        num_class=args['num_class'],
        learning_rate=args['learning_rate'],
        reg_alpha=args['reg_alpha'],
        reg_lambda=args['reg_lambda'],
        subsample=args['subsample'],
        n_jobs=args['n_jobs']
    )
    
    logger.info("Retraining model on full dataset with optimal iterations...")
    model.fit(train, targets, verbose=False)
    
    save_path = save_dir / get_save_name("gradient_boosting", "pkl")
    save_pkl(model, save_path)

    print(f"Model saved at {save_path}")
    logger.info(f"Model saved at {save_path}")


def log_to_logger(logger, period=10):
    def _callback(env: CallbackEnv):
        if period > 0 and env.iteration % period == 0:
            if env.evaluation_result_list:
                eval_result = '\t'.join([
                    f"{name}'s {metric}: {value:.5f}"
                    for name, metric, value, _ in env.evaluation_result_list
                ])
                logger.info(f"[{env.iteration}] {eval_result}")
            else:
                logger.info(f"[{env.iteration}] Still training...")  # No eval result yet
    _callback.order = 10
    return _callback

