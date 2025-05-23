from .rbf import *
from sklearn.decomposition import PCA

def train(dataset, save_dir, args):
    logger = get_logger('svm_rbf')
    args = {**DEFAULT_ARGS,**args}
    
    """Train a SVM-RBF Classifier and save the model."""
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    # reduce dimensions
    print("Dimensionality reduction...")
    if os.path.exists(save_dir / 'pca.pkl'):
        pca = load_pkl(save_dir / 'pca.pkl')
        X = pca.transform(X)
        print("PCA loaded.")
    else:
        pca = PCA(n_components=1024)
        X = pca.fit_transform(X)
        save_pkl(pca, save_dir / "pca.pkl")
        print("New PCA saved.")
    print("Done dimensionality reduction.")
    
    print("SVM-RBF start training...")
    logger.info(f"SVM-RBF train with {args['n_splits']}-fold cross-validation")
    
    kf = KFold(n_splits=args["n_splits"], shuffle=True)

    best_model = None
    best_score = -1e10 
    total_val_score = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold} training.")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = SVC(kernel="rbf", probability=True)
        model.fit(X_train, y_train)

        val_score = model.score(X_val, y_val)
        total_val_score += val_score
        logger.info(f"Fold {fold+1} | val accuracy: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model

    logger.info(f"Average val accuracy: {total_val_score / args['n_splits']}")
    
    save_path = save_dir / get_save_name("svm_rbf", "pkl")
    save_pkl(best_model, save_path)

    print(f"Model saved at {save_path}")
    logger.info(f"Model saved at {save_path}")