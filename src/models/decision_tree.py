import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from utils.utils import save_pkl, load_pkl, get_save_name
from utils.testutils import print_tree_details

def train(dataset, save_dir, args):
    """Train a Decision Tree and save the model."""
    print("Decision tree start training...")
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    model = DecisionTreeClassifier(criterion='gini', random_state=42)
    model.fit(X, y)

    save_path = save_dir / get_save_name("decision_tree", "pkl")
    save_pkl(model, save_path)
    
    print(f"Model saved at {save_path}")
    

def evaluate(dataset, saved_path, args):
    """Load the latest or specified Decision Tree model and evaluate it."""
    if not saved_path or not os.path.exists(saved_path):
        raise FileNotFoundError("Model not found. Please train first.")

    X, y = dataset.images, dataset.labels
    X = X.reshape(X.shape[0], -1)
    
    model = load_pkl(saved_path)
    print(f"Model loaded from {saved_path}")
    print_tree_details(model)
    
    y_preds = model.predict(X)
    y_scores = model.predict_proba(X)
    
    return y, y_preds, y_scores
