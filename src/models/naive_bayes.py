import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.testutils import print_tree_details
from utils.logger import get_logger

DEFAULT_ARGS = {
    'n_splits': 5
}

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

def evaluate(dataset, saved_path, args):
    """Load the latest or specified Naive bayes model and evaluate it."""
    if not saved_path or not os.path.exists(saved_path):
        raise FileNotFoundError("Model not found. Please train first.")

    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    model = load_pkl(saved_path)
    print(f"Model loaded from {saved_path}")
    
    y_preds = model.predict(X)
    y_scores = model.predict_proba(X)
    
    return y, y_preds, y_scores
