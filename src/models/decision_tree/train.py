from .decision_tree import *

def train(dataset, save_dir, args):
    args = {**DEFAULT_ARGS,**args}
    logger = get_logger('decision_tree')
    logger.info(f"{args}")
    
    """Train a Decision Tree and save the model."""
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    print("Decision Tree start training...")
    logger.info(f"Decision Tree train with {args['n_splits']}-fold cross-validation")
    
    param_grid = {
        'criterion': args['criterion'],
        'max_features': args['max_features'],
        'min_samples_leaf': args['min_samples_leaf'],
    }
    
    logger.info(f"Grid search parameters: ")
    logger.info(param_grid)

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=15), 
        param_grid, 
        cv=args['n_splits'], 
        scoring='recall_macro',
        n_jobs=4
    )
    
    grid_search.fit(X,y)
    
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_}")
    
    save_path = save_dir / get_save_name("decision_tree", "pkl")
    save_pkl(best_model, save_path)

    print(f"Model saved at {save_path}")
    logger.info(f"Model saved at {save_path}")