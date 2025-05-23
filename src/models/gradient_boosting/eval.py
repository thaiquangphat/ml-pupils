from .gradient_boosting import *

def evaluate(dataset, saved_path, args):
    if not saved_path or not os.path.exists(saved_path):
        raise FileNotFoundError("Model not found. Please train first.")

    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    model = load_pkl(saved_path)
    print(f"Model loaded from {saved_path}")
    
    y_preds = model.predict(X)
    y_scores = model.predict_proba(X)
    
    return y, y_preds, y_scores