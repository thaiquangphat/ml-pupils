from .naive_bayes import *

def train(dataset, save_dir, args):
    logger = get_logger('naive_bayes')
    args = {**DEFAULT_ARGS,**args}
    
    """Train a Naive Bayes Classifier and save the model."""
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    print("Gaussian Naive Bayes start training...")
    logger.info(f"GuassianNB train with {args['n_splits']}-fold cross-validation")
    
    kf = KFold(n_splits=args["n_splits"], shuffle=True)

    best_model = None
    best_score = -1e10 
    total_val_score = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GaussianNB(var_smoothing=1e-8)
        model.fit(X_train, y_train)

        val_score = model.score(X_val, y_val)
        total_val_score += val_score
        logger.info(f"Fold {fold+1} | val accuracy: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model

    logger.info(f"Average val accuracy: {total_val_score / args['n_splits']}")
    
    save_path = save_dir / get_save_name("naive_bayes", "pkl")
    save_pkl(best_model, save_path)

    print(f"Model saved at {save_path}")
    logger.info(f"Model saved at {save_path}")